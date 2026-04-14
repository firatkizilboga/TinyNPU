from __future__ import annotations

import os
import re
import shutil
import subprocess
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
    write_cv32e40p_program_v2,
)


CORE_DIR = REPO_ROOT / "external" / "cv32e40p" / "example_tb" / "core"
CUSTOM_DIR = CORE_DIR / "custom"
GENERATED_DIR = REPO_ROOT / "generated"
RUNTIME_DIR = REPO_ROOT / "software" / "compiler" / "tinynpu_jit"


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _runner_source(program_symbol: str) -> str:
    return f"""#include <stdlib.h>
#include <stdint.h>
#include "tinynpu_runtime_v2.h"

extern const TnpuProgram {program_symbol};

int main(void)
{{
    const TnpuProgram *program = &{program_symbol};
    TnpuTensor ins[8];
    const TnpuTensor *ip[8];
    TnpuTensor outs[8];
    const TnpuTensor *op[8];
    if (program->input_count > 8u || program->output_count > 8u) return EXIT_FAILURE;
    for (uint32_t i = 0; i < program->input_count; ++i) {{
        uint16_t t = program->input_tensor_indices[i];
        ins[i].data = program->tensors[t].data;
        ins[i].desc = &program->tensors[t];
        ins[i].elem_count = program->tensors[t].elem_count;
        ip[i] = &ins[i];
    }}
    for (uint32_t i = 0; i < program->output_count; ++i) {{
        uint16_t t = program->output_tensor_indices[i];
        outs[i].data = program->tensors[t].data;
        outs[i].desc = &program->tensors[t];
        outs[i].elem_count = program->tensors[t].elem_count;
        op[i] = &outs[i];
    }}
    return tinynpu_run(program, ip, op, NULL, 0u);
}}
"""


def _toolchain_prefix() -> Path:
    gcc = shutil.which("riscv32-unknown-elf-gcc")
    if gcc is None:
        raise FileNotFoundError("riscv32-unknown-elf-gcc not found in PATH")
    return Path(gcc).resolve().parent / "riscv32-unknown-elf-"


def _toolchain_root(prefix: Path) -> Path:
    return prefix.parent.parent


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )


def build_demo_artifact():
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

    mid = np.clip(lhs0.astype(np.int32) @ rhs0.astype(np.int32), -32768, 32767).astype(np.int16)
    y = np.clip(lhs1.astype(np.int32) @ mid.astype(np.int32), -32768, 32767).astype(np.int16)

    tensors = {
        "lhs0": TensorSpec("lhs0", lhs0.shape, DType.INT16, TensorKind.INPUT),
        "rhs0": TensorSpec("rhs0", rhs0.shape, DType.INT16, TensorKind.CONSTANT, data=rhs0),
        "lhs1": TensorSpec("lhs1", lhs1.shape, DType.INT16, TensorKind.CONSTANT, data=lhs1),
        "mid": TensorSpec("mid", mid.shape, DType.INT16, TensorKind.INTERMEDIATE),
        "y": TensorSpec("y", y.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }

    steps = [
        NpuSegment(
            "segment_000",
            [
                MatMulOp("op0", "lhs0", "rhs0", "mid"),
                MatMulOp("op1", "lhs1", "mid", "y"),
            ],
            inputs=["lhs0", "rhs0", "lhs1"],
            outputs=["y"],
        )
    ]

    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["lhs0"], outputs=["y"])
    plan.add_verification_step("y", "final_y")
    artifact = compile_plan(plan, {"y": y})

    seg = artifact.plan.steps[0]
    assert seg.ops[0].output_layout == "b"
    assert artifact.segment_artifacts["segment_000"].symbol_table["mid"]["role"] == "B"
    return artifact, {"lhs0": lhs0}, y


def main() -> int:
    program_name = "cv32e40p_b_layout_demo_v2"
    program_symbol = _sanitize(program_name)
    program_path = GENERATED_DIR / f"{program_name}_program.c"
    runner_path = GENERATED_DIR / f"{program_name}_runner.c"

    artifact, inputs, expected = build_demo_artifact()
    GENERATED_DIR.mkdir(exist_ok=True)
    write_cv32e40p_program_v2(artifact, inputs, program_path, program_name=program_name)
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
