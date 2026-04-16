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
    HostOp,
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
TNPU_RISCV_MARCH = os.environ.get("TINYNPU_RISCV_MARCH", "rv32imfc")
TNPU_RISCV_MABI = os.environ.get("TINYNPU_RISCV_MABI", "ilp32f")


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
    prefix_override = os.environ.get("TINYNPU_RISCV_PREFIX")
    if prefix_override:
        prefix = Path(prefix_override)
        gcc = prefix.parent / (prefix.name + "gcc")
        if not gcc.exists():
            raise FileNotFoundError(f"{gcc} not found for TINYNPU_RISCV_PREFIX")
        return prefix
    preferred = Path("/opt/riscv-ilp32f/bin/riscv32-unknown-elf-gcc")
    if preferred.exists():
        return preferred.parent / "riscv32-unknown-elf-"
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
    x_f = np.array(
        [
            [1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 2.5, -1.5],
            [0.0, 1.0, -1.0, 2.0, 3.0, -3.0, 4.0, -4.0],
            [2.0, 2.0, 2.0, 2.0, -2.0, -2.0, -2.0, -2.0],
            [1.5, -1.5, 0.0, 0.0, 1.0, -1.0, 2.0, -2.0],
            [3.0, 0.0, -3.0, 1.0, -1.0, 2.0, -2.0, 4.0],
            [0.25, -0.25, 0.75, -0.75, 1.25, -1.25, 1.75, -1.75],
            [4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0],
            [1.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    scale = 0.5
    x_q = np.clip(np.rint(x_f / scale), -32768, 32767).astype(np.int16)
    w = np.eye(8, dtype=np.int16)
    y = x_q.copy()

    tensors = {
        "x_f": TensorSpec("x_f", x_f.shape, DType.FLOAT32, TensorKind.INPUT),
        "x_q": TensorSpec("x_q", x_q.shape, DType.INT16, TensorKind.INTERMEDIATE),
        "w": TensorSpec("w", w.shape, DType.INT16, TensorKind.CONSTANT, data=w),
        "y": TensorSpec("y", y.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        HostOp("quant_x", "quantize", inputs=["x_f"], outputs=["x_q"], attrs={"scale": scale, "zero_point": 0}),
        NpuSegment("seg_quant", [MatMulOp("op0", "x_q", "w", "y")], inputs=["x_q", "w"], outputs=["y"]),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["x_f"], outputs=["y"])
    plan.add_verification_step("y", "final_y")
    artifact = compile_plan(plan, {"y": y})
    return artifact, {"x_f": x_f}


def main() -> int:
    program_name = "cv32e40p_quantize_stage_demo_v2"
    program_symbol = _sanitize(program_name)
    program_path = GENERATED_DIR / f"{program_name}_program.c"
    runner_path = GENERATED_DIR / f"{program_name}_runner.c"

    artifact, inputs = build_demo_artifact()
    GENERATED_DIR.mkdir(exist_ok=True)
    write_cv32e40p_program_v2(artifact, inputs, program_path, program_name=program_name)
    runner_path.write_text(_runner_source(program_symbol))

    source_text = program_path.read_text()
    if "TNPU_WRITE_QUANTIZE_F32_TO_INT16" not in source_text:
        raise RuntimeError("quantize write transform was not emitted")
    if "TNPU_HOST_QUANTIZE" in source_text:
        raise RuntimeError("quantize host op was not absorbed")

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
            "-ffast-math",
            "-fno-builtin-printf",
            "-I.",
            f"-I{include_dir}",
            f"-I{REPO_ROOT / 'software' / 'compiler' / 'tinynpu_jit'}",
            str(program_path),
            str(runner_path),
            str(CORE_DIR / "custom" / "crt0.S"),
            str(CORE_DIR / "custom" / "syscalls.c"),
            str(REPO_ROOT / "software" / "compiler" / "tinynpu_jit" / "tinynpu_runtime_v2.c"),
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

    sim_env = dict(build_env)
    sim_env.setdefault("VERILATOR_MAX_TICKS", "3000000000")
    sim = _run(
        [
            str(CORE_DIR / "obj_dir" / "cv32e40p_tb_vlt_npu"),
            "+verilator+noassert",
            f"+firmware={hex_path}",
            "+maxcycles=2000000",
        ],
        cwd=CORE_DIR,
        env=sim_env,
        capture=True,
    )
    print(sim.stdout, end="")
    if sim.stderr:
        print(sim.stderr, end="", file=sys.stderr)
    if "EXIT SUCCESS" not in sim.stdout:
        raise RuntimeError("simulation did not report EXIT SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
