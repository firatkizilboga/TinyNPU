from __future__ import annotations

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


def build_artifact():
    lhs_k0 = np.array([[1, 2, -1, 0, 3, -2, 1, 4]], dtype=np.int16)
    rhs_k0 = np.array(
        [
            [1, 0, 1, 0, -1, 2, 0, 1],
            [0, 1, 0, 1, 2, -1, 1, 0],
            [1, -1, 1, 0, 0, 1, 2, 1],
            [2, 0, -1, 1, 1, 0, -1, 2],
            [0, 2, 1, -1, 1, 1, 0, 0],
            [1, 1, 0, 2, -1, 0, 1, -1],
            [0, -1, 2, 1, 0, 1, 1, 2],
            [1, 0, 1, 1, 2, 0, -1, 1],
        ],
        dtype=np.int16,
    )
    lhs_k1 = np.array([[2, 1, 0, -1, 1, 2, 0, 1]], dtype=np.int16)
    rhs_k1 = np.array(
        [
            [0, 1, 1, 0, 2, 1, 0, -1],
            [1, 0, 2, 1, -1, 0, 1, 2],
            [2, 1, 0, -1, 1, 2, 1, 0],
            [1, -1, 1, 2, 0, 1, 2, 1],
            [0, 2, 1, 0, 1, -1, 1, 2],
            [1, 1, -1, 1, 2, 0, 0, 1],
            [2, 0, 1, 1, 0, 2, -1, 1],
            [1, 2, 0, 1, -1, 1, 2, 0],
        ],
        dtype=np.int16,
    )
    lhs_v0 = np.array([[1, 0, 2, -1, 1, 0, 1, 2]], dtype=np.int16)
    rhs_v0 = np.array(
        [
            [1, 2, 0, 1, -1, 0, 2, 1],
            [0, 1, 2, 0, 1, 2, -1, 1],
            [2, 0, 1, 2, 0, 1, 1, -1],
            [1, -1, 0, 1, 2, 1, 0, 2],
            [0, 2, 1, 0, 1, -1, 2, 1],
            [1, 0, -1, 2, 1, 0, 1, 2],
            [2, 1, 0, 1, -1, 2, 0, 1],
            [1, 2, 1, 0, 2, 1, -1, 0],
        ],
        dtype=np.int16,
    )
    lhs_v1 = np.array([[0, 1, 1, 2, -1, 1, 0, 1]], dtype=np.int16)
    rhs_v1 = np.array(
        [
            [2, 0, 1, -1, 2, 1, 0, 1],
            [1, 2, 0, 1, 0, -1, 2, 1],
            [0, 1, 2, 1, -1, 0, 1, 2],
            [1, -1, 1, 0, 2, 1, 2, 0],
            [2, 1, 0, 2, 1, 0, -1, 1],
            [0, 2, 1, 1, 0, 2, 1, -1],
            [1, 0, 2, 1, 1, -1, 0, 2],
            [2, 1, -1, 0, 1, 2, 1, 0],
        ],
        dtype=np.int16,
    )
    query = np.array([[1, -1, 2, 0, 1, 3, -2, 1]], dtype=np.int16)
    attn_scale = 1.0 / 256.0
    golden = GoldenModel()

    k0 = np.clip(lhs_k0.astype(np.int32) @ rhs_k0.astype(np.int32), -32768, 32767).astype(np.int16)
    k1 = np.clip(lhs_k1.astype(np.int32) @ rhs_k1.astype(np.int32), -32768, 32767).astype(np.int16)
    v0 = np.clip(lhs_v0.astype(np.int32) @ rhs_v0.astype(np.int32), -32768, 32767).astype(np.int16)
    v1 = np.clip(lhs_v1.astype(np.int32) @ rhs_v1.astype(np.int32), -32768, 32767).astype(np.int16)

    k_cache = np.zeros((8, 16), dtype=np.int16)
    k_cache[:, 1] = k0[0]
    k_cache[:, 9] = k1[0]
    scores = np.clip(query.astype(np.int32) @ k_cache.astype(np.int32), -32768, 32767).astype(np.int16)
    probs = golden.softmax(scores, axis=-1).astype(np.float32)
    attn_q = golden.quantize(probs, scale=attn_scale, out_dtype=DType.INT16)
    v_cache = np.zeros((16, 8), dtype=np.int16)
    v_cache[1, :] = v0[0]
    v_cache[9, :] = v1[0]
    expected = golden.matmul(attn_q, v_cache, shift=8, out_dtype=DType.INT16)

    tensors = {
        "lhs_k0": TensorSpec("lhs_k0", lhs_k0.shape, DType.INT16, TensorKind.CONSTANT, data=lhs_k0),
        "rhs_k0": TensorSpec("rhs_k0", rhs_k0.shape, DType.INT16, TensorKind.CONSTANT, data=rhs_k0),
        "lhs_k1": TensorSpec("lhs_k1", lhs_k1.shape, DType.INT16, TensorKind.CONSTANT, data=lhs_k1),
        "rhs_k1": TensorSpec("rhs_k1", rhs_k1.shape, DType.INT16, TensorKind.CONSTANT, data=rhs_k1),
        "lhs_v0": TensorSpec("lhs_v0", lhs_v0.shape, DType.INT16, TensorKind.CONSTANT, data=lhs_v0),
        "rhs_v0": TensorSpec("rhs_v0", rhs_v0.shape, DType.INT16, TensorKind.CONSTANT, data=rhs_v0),
        "lhs_v1": TensorSpec("lhs_v1", lhs_v1.shape, DType.INT16, TensorKind.CONSTANT, data=lhs_v1),
        "rhs_v1": TensorSpec("rhs_v1", rhs_v1.shape, DType.INT16, TensorKind.CONSTANT, data=rhs_v1),
        "query": TensorSpec("query", query.shape, DType.INT16, TensorKind.CONSTANT, data=query),
        "scores": TensorSpec("scores", scores.shape, DType.INT16, TensorKind.INTERMEDIATE),
        "probs": TensorSpec("probs", probs.shape, DType.FLOAT32, TensorKind.INTERMEDIATE),
        "attn_q": TensorSpec(
            "attn_q",
            attn_q.shape,
            DType.INT16,
            TensorKind.INTERMEDIATE,
            metadata={"storage_role": "A"},
        ),
        "out": TensorSpec("out", expected.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    tensors.update(
        make_native_int16_kv_cache_specs(
            k_base_name="k_cache",
            v_base_name="v_cache",
            d_head=8,
            token_capacity=16,
            token_names=["t1", "t9"],
            token_indices=[1, 9],
        )
    )
    steps = [
        NpuSegment(
            "seg_score",
            [
                MatMulOp("op_k0", "lhs_k0", "rhs_k0", "k_cache_t1"),
                MatMulOp("op_k1", "lhs_k1", "rhs_k1", "k_cache_t9"),
                MatMulOp("op_v0", "lhs_v0", "rhs_v0", "v_cache_t1"),
                MatMulOp("op_v1", "lhs_v1", "rhs_v1", "v_cache_t9"),
                MatMulOp("op_qk", "query", "k_cache", "scores"),
            ],
            inputs=[],
            outputs=["scores"],
        ),
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
            [
                MatMulOp("op_av", "attn_q", "v_cache", "out", shift=8),
            ],
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
    program_name = "cv32e40p_decode_attention_jit_demo_v2"
    program_symbol = _sanitize(program_name)
    program_path = GENERATED_DIR / f"{program_name}_program.c"
    runner_path = GENERATED_DIR / f"{program_name}_runner.c"

    artifact, expected = build_artifact()
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
            "+maxcycles=500000",
        ],
        cwd=CORE_DIR,
        env=env,
        capture=True,
    )
    print(f"program={program_name}")
    print(f"expected_checksum={int(expected.astype(np.int64).sum())}")
    print(proc.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
