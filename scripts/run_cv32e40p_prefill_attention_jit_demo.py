from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))

from tinynpu_jit import (  # noqa: E402
    DType,
    ExecutionPlan,
    HostOp,
    MatMulOp,
    NpuSegment,
    TensorKind,
    TensorSpec,
    compile_plan,
    emit_cv32e40p_program_v2,
    make_native_int16_kv_cache_specs,
)
from tinynpu_jit.golden import GoldenModel  # noqa: E402

from run_cv32e40p_b_append_demo import (  # noqa: E402
    CORE_DIR,
    CUSTOM_DIR,
    GENERATED_DIR,
    RUNTIME_DIR,
    TNPU_RISCV_MABI,
    TNPU_RISCV_MARCH,
    _run,
    _runner_source,
    _sanitize,
    _toolchain_prefix,
    _toolchain_root,
)


def _rand_i16(rng: np.random.Generator, shape: tuple[int, ...], low: int = -2, high: int = 3) -> np.ndarray:
    return rng.integers(low, high, size=shape, endpoint=False, dtype=np.int16)


def build_plan(
    *,
    d_model: int = 32,
    d_head: int = 16,
    token_count: int = 8,
    seed: int = 0,
    act_scale: float = 1.0 / 32.0,
    attn_scale: float = 1.0 / 256.0,
):
    if d_model <= 0 or d_model % 8 != 0:
        raise ValueError("d_model must be a positive multiple of 8")
    if d_head <= 0 or d_head % 8 != 0:
        raise ValueError("d_head must be a positive multiple of 8")
    if token_count <= 0 or token_count % 8 != 0:
        raise ValueError("token_count must be a positive multiple of 8")

    rng = np.random.default_rng(seed)

    x = _rand_i16(rng, (token_count, d_model))
    w_q = _rand_i16(rng, (d_model, d_head))
    w_k = _rand_i16(rng, (d_model, d_head))
    w_v = _rand_i16(rng, (d_model, d_head))
    w_o = _rand_i16(rng, (d_head, d_model))

    token_indices = list(range(token_count))
    token_names = [f"t{idx}" for idx in token_indices]

    tensors = {
        "x": TensorSpec("x", x.shape, DType.INT16, TensorKind.CONSTANT, data=x),
        "w_q": TensorSpec("w_q", w_q.shape, DType.INT16, TensorKind.CONSTANT, data=w_q),
        "w_k": TensorSpec("w_k", w_k.shape, DType.INT16, TensorKind.CONSTANT, data=w_k),
        "w_v": TensorSpec("w_v", w_v.shape, DType.INT16, TensorKind.CONSTANT, data=w_v),
        "w_o": TensorSpec("w_o", w_o.shape, DType.INT16, TensorKind.CONSTANT, data=w_o),
        "q_int": TensorSpec("q_int", (token_count, d_head), DType.INT16, TensorKind.INTERMEDIATE),
        "q_a": TensorSpec(
            "q_a",
            (token_count, d_head),
            DType.INT16,
            TensorKind.INTERMEDIATE,
            metadata={"storage_role": "A"},
        ),
        "scores": TensorSpec("scores", (token_count, token_count), DType.INT16, TensorKind.INTERMEDIATE),
        "probs_f16": TensorSpec("probs_f16", (token_count, token_count), DType.INT16, TensorKind.INTERMEDIATE),
        "probs_q": TensorSpec(
            "probs_q",
            (token_count, token_count),
            DType.INT16,
            TensorKind.INTERMEDIATE,
            metadata={"storage_role": "A"},
        ),
        "attn_int": TensorSpec("attn_int", (token_count, d_head), DType.INT16, TensorKind.INTERMEDIATE),
        "out_int": TensorSpec("out_int", (token_count, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        "out": TensorSpec("out", (token_count, d_model), DType.FLOAT32, TensorKind.OUTPUT, is_final_output=True),
    }
    tensors.update(
        make_native_int16_kv_cache_specs(
            k_base_name="k_cache",
            v_base_name="v_cache",
            d_head=d_head,
            token_capacity=token_count,
            token_names=token_names,
            token_indices=token_indices,
            kind=TensorKind.INTERMEDIATE,
        )
    )

    seg_cache_ops: list[MatMulOp] = []
    for token_idx, token_name in zip(token_indices, token_names):
        x_tok = np.array(x[token_idx : token_idx + 1, :], copy=True)
        tensors[f"x_{token_name}"] = TensorSpec(
            f"x_{token_name}",
            x_tok.shape,
            DType.INT16,
            TensorKind.CONSTANT,
            data=x_tok,
        )
        seg_cache_ops.append(
            MatMulOp(f"op_k_{token_name}", f"x_{token_name}", "w_k", f"k_cache_{token_name}", in_dtype=DType.INT16, out_dtype=DType.INT16)
        )
        seg_cache_ops.append(
            MatMulOp(f"op_v_{token_name}", f"x_{token_name}", "w_v", f"v_cache_{token_name}", in_dtype=DType.INT16, out_dtype=DType.INT16)
        )

    steps = [
        NpuSegment("seg_cache", seg_cache_ops, inputs=[], outputs=[]),
        NpuSegment(
            "seg_q",
            [MatMulOp("op_q", "x", "w_q", "q_int", in_dtype=DType.INT16, out_dtype=DType.INT16)],
            inputs=[],
            outputs=["q_int"],
        ),
        HostOp("alias_q_a", "alias", inputs=["q_int"], outputs=["q_a"]),
        NpuSegment(
            "seg_score",
            [MatMulOp("op_qk", "q_a", "k_cache", "scores", in_dtype=DType.INT16, out_dtype=DType.INT16)],
            inputs=["q_a"],
            outputs=["scores"],
        ),
        HostOp("softmax_scores_f16", "softmax_f16", inputs=["scores"], outputs=["probs_f16"], attrs={"axis": -1}),
        HostOp(
            "quantize_probs",
            "quantize",
            inputs=["probs_f16"],
            outputs=["probs_q"],
            attrs={"scale": attn_scale, "zero_point": 0, "dtype": DType.INT16},
        ),
        NpuSegment(
            "seg_value",
            [
                MatMulOp("op_av", "probs_q", "v_cache", "attn_int", shift=8, in_dtype=DType.INT16, out_dtype=DType.INT16),
                MatMulOp("op_o", "attn_int", "w_o", "out_int", in_dtype=DType.INT16, out_dtype=DType.INT16),
            ],
            inputs=["probs_q"],
            outputs=["out_int"],
        ),
        HostOp("dequant_out", "dequantize", inputs=["out_int"], outputs=["out"], attrs={"scale": 1.0, "zero_point": 0}),
    ]

    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=[], outputs=["out"])
    plan.add_verification_step("scores", "prefill_scores")
    plan.add_verification_step("out", "prefill_out")
    return plan


def build_artifact(**kwargs):
    plan = build_plan(**kwargs)
    golden = GoldenModel()
    tensors = plan.tensors
    attn_scale = float(kwargs.get("attn_scale", 1.0 / 256.0))

    x = np.array(tensors["x"].data, dtype=np.int16, copy=True)
    w_q = np.array(tensors["w_q"].data, dtype=np.int16, copy=True)
    w_k = np.array(tensors["w_k"].data, dtype=np.int16, copy=True)
    w_v = np.array(tensors["w_v"].data, dtype=np.int16, copy=True)
    w_o = np.array(tensors["w_o"].data, dtype=np.int16, copy=True)

    q_int = golden.matmul(x, w_q, out_dtype=DType.INT16)
    k_int = golden.matmul(x, w_k, out_dtype=DType.INT16)
    v_int = golden.matmul(x, w_v, out_dtype=DType.INT16)
    k_cache = np.array(k_int.T, dtype=np.int16, copy=True)
    v_cache = np.array(v_int, dtype=np.int16, copy=True)
    scores = golden.matmul(q_int, k_cache, out_dtype=DType.INT16)
    probs = golden.softmax(scores, axis=-1).astype(np.float32)
    probs_q = golden.quantize(probs, scale=attn_scale, zero_point=0, out_dtype=DType.INT16)
    attn_int = golden.matmul(probs_q, v_cache, shift=8, out_dtype=DType.INT16)
    out_int = golden.matmul(attn_int, w_o, out_dtype=DType.INT16)
    out = golden.dequantize(out_int, scale=1.0, zero_point=0).astype(np.float32)

    wanted = {
        "scores": np.array(scores, copy=True),
        "out": np.array(out, copy=True),
    }
    artifact = compile_plan(plan, wanted)
    return artifact, wanted["out"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-head", type=int, default=16)
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    artifact, expected = build_artifact(
        d_model=args.d_model,
        d_head=args.d_head,
        token_count=args.token_count,
        seed=args.seed,
    )

    program_name = (
        f"cv32e40p_prefill_attention_d{args.d_model}_h{args.d_head}"
        f"_t{args.token_count}_s{args.seed}_v2"
    )
    program_symbol = _sanitize(program_name)
    program_path = GENERATED_DIR / f"{program_name}_program.c"
    runner_path = GENERATED_DIR / f"{program_name}_runner.c"

    source = emit_cv32e40p_program_v2(artifact, {}, program_name=program_name)
    GENERATED_DIR.mkdir(exist_ok=True)
    program_path.write_text(source)
    runner_path.write_text(_runner_source(program_symbol))

    prefix = _toolchain_prefix()
    gcc = f"{prefix}gcc"
    objcopy = f"{prefix}objcopy"
    toolchain_root = _toolchain_root(prefix)
    include_dir = toolchain_root / "riscv32-unknown-elf" / "include"
    lib_dir = toolchain_root / "riscv32-unknown-elf" / "lib"
    elf_path = CUSTOM_DIR / f"{program_name}.elf"
    hex_path = CUSTOM_DIR / f"{program_name}.hex"

    build_env = dict(os.environ)
    build_env["CCACHE_DISABLE"] = "1"
    build_env["TMPDIR"] = "/tmp"
    _run(["make", "verilator-build-npu"], cwd=CORE_DIR, env=build_env)
    _run(
        [
            gcc,
            f"-march={TNPU_RISCV_MARCH}",
            f"-mabi={TNPU_RISCV_MABI}",
            "-o",
            str(elf_path),
            "-w",
            "-O3",
            "-g",
            "-nostdlib",
            "-DTINYNPU_USE_SHARED_SRAM=1",
            "-DTNPU_RUNTIME_V2_DUMP_FINAL_OUTPUTS=1",
            "-T",
            "custom/link.ld",
            "-static",
            "custom/crt0.S",
            str(runner_path),
            str(program_path),
            str(RUNTIME_DIR / "tinynpu_runtime_v2.c"),
            "mem_stall/mem_stall.c",
            "custom/syscalls.c",
            "custom/vectors.S",
            "-I",
            str(include_dir),
            "-I",
            "mem_stall",
            "-I",
            str(RUNTIME_DIR),
            "-L",
            str(lib_dir),
            "-lc",
            "-lm",
            "-lgcc",
        ],
        cwd=CORE_DIR,
        env=build_env,
    )
    _run([objcopy, "-O", "verilog", str(elf_path), str(hex_path)], cwd=CORE_DIR, env=build_env)

    env = dict(os.environ)
    env["VERILATOR_MAX_TICKS"] = "3000000000"
    proc = _run(
        [
            str(CORE_DIR / "obj_dir" / "cv32e40p_tb_vlt_npu"),
            "+verilator+noassert",
            f"+firmware={hex_path}",
            "+maxcycles=2000000",
        ],
        cwd=CORE_DIR,
        env=env,
        capture=True,
    )
    print(f"program={program_name}")
    print(
        f"d_model={args.d_model} d_head={args.d_head} "
        f"token_count={args.token_count} seed={args.seed}"
    )
    print(f"expected_checksum={float(np.array(expected, dtype=np.float32).sum()):.6f}")
    print(proc.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
