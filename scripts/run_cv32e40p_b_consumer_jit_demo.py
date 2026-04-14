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
    MatMulOp,
    NpuSegment,
    TensorKind,
    TensorSpec,
    compile_plan,
    emit_cv32e40p_program_v2,
)

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
    lhs0 = np.array(
        [
            [1, 2, -1, 0, 3, -2, 1, 4],
            [0, -1, 2, 3, -2, 1, 0, 2],
            [3, 1, 0, -1, 2, 2, -3, 1],
            [2, 0, 1, 1, -1, 3, 2, -2],
            [-1, 2, 3, 0, 1, -1, 2, 1],
            [4, -2, 1, 2, 0, 1, -1, 3],
            [1, 1, -2, 4, 2, 0, 3, -1],
            [0, 3, 2, -1, 1, 2, 1, 0],
        ],
        dtype=np.int16,
    )
    rhs0 = np.array(
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
    lhs1 = np.array(
        [
            [2, 1, 0, -1, 1, 2, 0, 1],
            [1, 0, 2, 1, -1, 0, 1, 2],
            [0, 1, 1, 2, 0, -1, 2, 1],
            [1, 2, -1, 0, 2, 1, 1, 0],
            [2, -1, 1, 1, 0, 2, -1, 1],
            [0, 2, 1, -1, 1, 0, 2, 1],
            [1, 1, 0, 2, 1, -1, 0, 2],
            [2, 0, 1, 1, -1, 1, 2, 0],
        ],
        dtype=np.int16,
    )
    rhs1 = np.array(
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
    query = np.array(
        [
            [1, 0, 2, -1, 1, 0, 1, 2],
            [0, 1, 1, 2, -1, 1, 0, 1],
            [2, 1, 0, 1, 1, -1, 2, 0],
            [1, 2, 1, 0, 0, 1, -1, 2],
            [0, 1, 2, 1, 2, 0, 1, -1],
            [1, 0, 1, 2, 1, 2, 0, 1],
            [2, 1, -1, 1, 0, 1, 2, 1],
            [1, 2, 0, 1, 1, 0, 1, 2],
        ],
        dtype=np.int16,
    )

    token1 = np.clip(lhs1.astype(np.int32) @ rhs1.astype(np.int32), -32768, 32767).astype(np.int16)
    expected = np.clip(query.astype(np.int32) @ token1.astype(np.int32), -32768, 32767).astype(np.int16)

    tensors = {
        "lhs0": TensorSpec("lhs0", lhs0.shape, DType.INT16, TensorKind.CONSTANT, data=lhs0),
        "rhs0": TensorSpec("rhs0", rhs0.shape, DType.INT16, TensorKind.CONSTANT, data=rhs0),
        "lhs1": TensorSpec("lhs1", lhs1.shape, DType.INT16, TensorKind.CONSTANT, data=lhs1),
        "rhs1": TensorSpec("rhs1", rhs1.shape, DType.INT16, TensorKind.CONSTANT, data=rhs1),
        "query": TensorSpec("query", query.shape, DType.INT16, TensorKind.CONSTANT, data=query),
        "cache": TensorSpec("cache", (16, 8), DType.INT16, TensorKind.INTERMEDIATE),
        "cache_t0": TensorSpec(
            "cache_t0",
            (8, 8),
            DType.INT16,
            TensorKind.INTERMEDIATE,
            metadata={"storage_view_of": "cache", "storage_role": "B", "storage_word_offset": 0},
        ),
        "cache_t1": TensorSpec(
            "cache_t1",
            (8, 8),
            DType.INT16,
            TensorKind.INTERMEDIATE,
            metadata={"storage_view_of": "cache", "storage_role": "B", "storage_word_offset": 8},
        ),
        "out": TensorSpec("out", expected.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        NpuSegment(
            "seg_cache_views",
            [
                MatMulOp("op0", "lhs0", "rhs0", "cache_t0"),
                MatMulOp("op1", "lhs1", "rhs1", "cache_t1"),
                MatMulOp("op2", "query", "cache_t1", "out"),
            ],
            inputs=["cache"],
            outputs=["out"],
        ),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=[], outputs=["out"])
    plan.add_verification_step("out", "jit_cache_consume")
    artifact = compile_plan(plan, {"out": expected})
    return artifact, expected


def main() -> int:
    program_name = "cv32e40p_b_consumer_jit_demo_v2"
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
            "+maxcycles=250000",
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
