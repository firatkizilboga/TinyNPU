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
    _run,
    _runner_source,
    _sanitize,
    _toolchain_prefix,
    _toolchain_root,
)


def _parse_token_indices(raw: str | None, token_capacity: int) -> list[int]:
    if raw is None or raw.strip() == "":
        return [idx for idx in (1, 9) if idx < token_capacity]
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("token index list must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("token indices must be unique")
    for idx in values:
        if idx < 0 or idx >= token_capacity:
            raise ValueError(f"token index {idx} is outside token capacity {token_capacity}")
    return values


def _rand_i16(rng: np.random.Generator, shape: tuple[int, ...], low: int = -2, high: int = 3) -> np.ndarray:
    return rng.integers(low, high, size=shape, endpoint=False, dtype=np.int16)


def build_artifact(
    *,
    d_head: int = 8,
    token_capacity: int = 16,
    token_indices: list[int] | None = None,
    seed: int = 0,
    attn_scale: float = 1.0 / 256.0,
):
    if d_head <= 0 or d_head % 8 != 0:
        raise ValueError("d_head must be a positive multiple of 8")
    if token_capacity <= 0 or token_capacity % 8 != 0:
        raise ValueError("token_capacity must be a positive multiple of 8")

    indices = token_indices or [idx for idx in (1, 9) if idx < token_capacity]
    if not indices:
        raise ValueError("at least one token index is required")

    rng = np.random.default_rng(seed)
    golden = GoldenModel()

    tensors: dict[str, TensorSpec] = {}
    seg_score_ops: list[MatMulOp] = []
    token_names = [f"t{idx}" for idx in indices]

    k_cache = np.zeros((d_head, token_capacity), dtype=np.int16)
    v_cache = np.zeros((token_capacity, d_head), dtype=np.int16)

    for tok_idx, token_name in zip(indices, token_names):
        lhs_k = _rand_i16(rng, (1, d_head))
        rhs_k = _rand_i16(rng, (d_head, d_head))
        lhs_v = _rand_i16(rng, (1, d_head))
        rhs_v = _rand_i16(rng, (d_head, d_head))

        k_val = np.clip(lhs_k.astype(np.int32) @ rhs_k.astype(np.int32), -32768, 32767).astype(np.int16)
        v_val = np.clip(lhs_v.astype(np.int32) @ rhs_v.astype(np.int32), -32768, 32767).astype(np.int16)
        k_cache[:, tok_idx] = k_val[0]
        v_cache[tok_idx, :] = v_val[0]

        tensors[f"lhs_k_{token_name}"] = TensorSpec(
            f"lhs_k_{token_name}", lhs_k.shape, DType.INT16, TensorKind.CONSTANT, data=lhs_k
        )
        tensors[f"rhs_k_{token_name}"] = TensorSpec(
            f"rhs_k_{token_name}", rhs_k.shape, DType.INT16, TensorKind.CONSTANT, data=rhs_k
        )
        tensors[f"lhs_v_{token_name}"] = TensorSpec(
            f"lhs_v_{token_name}", lhs_v.shape, DType.INT16, TensorKind.CONSTANT, data=lhs_v
        )
        tensors[f"rhs_v_{token_name}"] = TensorSpec(
            f"rhs_v_{token_name}", rhs_v.shape, DType.INT16, TensorKind.CONSTANT, data=rhs_v
        )

        seg_score_ops.append(
            MatMulOp(
                f"op_k_{token_name}",
                f"lhs_k_{token_name}",
                f"rhs_k_{token_name}",
                f"k_cache_{token_name}",
            )
        )
        seg_score_ops.append(
            MatMulOp(
                f"op_v_{token_name}",
                f"lhs_v_{token_name}",
                f"rhs_v_{token_name}",
                f"v_cache_{token_name}",
            )
        )

    query = _rand_i16(rng, (1, d_head))
    scores = np.clip(query.astype(np.int32) @ k_cache.astype(np.int32), -32768, 32767).astype(np.int16)
    probs = golden.softmax(scores, axis=-1).astype(np.float32)
    attn_q = golden.quantize(probs, scale=attn_scale, out_dtype=DType.INT16)
    expected = golden.matmul(attn_q, v_cache, shift=8, out_dtype=DType.INT16)

    tensors["query"] = TensorSpec("query", query.shape, DType.INT16, TensorKind.CONSTANT, data=query)
    tensors["scores"] = TensorSpec("scores", scores.shape, DType.INT16, TensorKind.INTERMEDIATE)
    tensors["probs"] = TensorSpec("probs", probs.shape, DType.FLOAT32, TensorKind.INTERMEDIATE)
    tensors["attn_q"] = TensorSpec(
        "attn_q",
        attn_q.shape,
        DType.INT16,
        TensorKind.INTERMEDIATE,
        metadata={"storage_role": "A"},
    )
    tensors["out"] = TensorSpec("out", expected.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True)
    tensors.update(
        make_native_int16_kv_cache_specs(
            k_base_name="k_cache",
            v_base_name="v_cache",
            d_head=d_head,
            token_capacity=token_capacity,
            token_names=token_names,
            token_indices=indices,
        )
    )

    seg_score_ops.append(MatMulOp("op_qk", "query", "k_cache", "scores"))

    steps = [
        NpuSegment("seg_score", seg_score_ops, inputs=[], outputs=["scores"]),
        HostOp("softmax_scores", "softmax", inputs=["scores"], outputs=["probs"], attrs={"axis": -1}),
        HostOp(
            "quantize_probs",
            "quantize",
            inputs=["probs"],
            outputs=["attn_q"],
            attrs={"scale": attn_scale, "zero_point": 0, "dtype": DType.INT16},
        ),
        NpuSegment(
            "seg_value",
            [MatMulOp("op_av", "attn_q", "v_cache", "out", shift=8)],
            inputs=["attn_q"],
            outputs=["out"],
        ),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=[], outputs=["out"])
    plan.add_verification_step("scores", "scores")
    plan.add_verification_step("attn_q", "attn_q")
    plan.add_verification_step("out", "decode_attention")
    artifact = compile_plan(
        plan,
        {
            "scores": scores,
            "attn_q": attn_q,
            "out": expected,
        },
    )
    return artifact, expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-head", type=int, default=8)
    parser.add_argument("--token-capacity", type=int, default=16)
    parser.add_argument("--token-indices", type=str, default="1,9")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    token_indices = _parse_token_indices(args.token_indices, args.token_capacity)
    artifact, expected = build_artifact(
        d_head=args.d_head,
        token_capacity=args.token_capacity,
        token_indices=token_indices,
        seed=args.seed,
    )

    program_name = (
        f"cv32e40p_decode_attention_d{args.d_head}_t{args.token_capacity}"
        f"_n{len(token_indices)}_s{args.seed}_v2"
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
            "-march=rv32imfc",
            "-mabi=ilp32",
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
    )
    _run([objcopy, "-O", "verilog", str(elf_path), str(hex_path)], cwd=CORE_DIR)

    env = dict(os.environ)
    env["VERILATOR_MAX_TICKS"] = "3000000000"
    proc = _run(
        [
            str(CORE_DIR / "obj_dir" / "cv32e40p_tb_vlt_npu"),
            "+verilator+noassert",
            f"+firmware={hex_path}",
            "+maxcycles=1000000",
        ],
        cwd=CORE_DIR,
        env=env,
        capture=True,
    )
    print(f"program={program_name}")
    print(f"d_head={args.d_head} token_capacity={args.token_capacity} token_indices={token_indices} seed={args.seed}")
    print(f"expected_checksum={int(expected.astype(np.int64).sum())}")
    print(proc.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
