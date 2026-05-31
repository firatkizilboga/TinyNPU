from __future__ import annotations

from dataclasses import dataclass
import os
import re
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CORE_DIR = REPO_ROOT / "external" / "cv32e40p" / "example_tb" / "core"
CUSTOM_DIR = CORE_DIR / "custom"
GENERATED_DIR = REPO_ROOT / "generated"
RUNTIME_DIR = REPO_ROOT / "software" / "compiler" / "tinynpu_jit"
TNPU_RISCV_MARCH = os.environ.get("TINYNPU_RISCV_MARCH", "rv32imfc")
TNPU_RISCV_MABI = os.environ.get("TINYNPU_RISCV_MABI", "ilp32f")


@dataclass(frozen=True)
class RunnerConfig:
    repeat_count: int = 1
    dump_final_outputs: bool = True
    verbose_steps: bool = True
    force_mmio: bool = False
    timed: bool = False
    banner: str | None = None


def sanitize_program_symbol(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def render_runner_source(program_symbol: str, config: RunnerConfig | None = None) -> str:
    cfg = config or RunnerConfig()
    run_call = (
        f"    return tinynpu_run_repeat(program, ip, op, NULL, 0u, {cfg.repeat_count}u);\n"
        if cfg.timed or cfg.repeat_count > 1
        else "    return tinynpu_run(program, ip, op, NULL, 0u);\n"
    )
    banner = f'    puts("{cfg.banner}");\n' if cfg.banner else ""
    force_mmio = "    tinynpu_set_force_mmio(1);\n" if cfg.force_mmio else ""
    return f"""#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include "tinynpu_runtime_v2.h"

extern const TnpuProgram {program_symbol};

int main(void)
{{
    const TnpuProgram *program = &{program_symbol};
    TnpuTensor ins[8];
    const TnpuTensor *ip[8];
    TnpuTensor outs[8];
    const TnpuTensor *op[8];
    if (program->input_count > 8u) return EXIT_FAILURE;
    if (program->output_count > 8u) return EXIT_FAILURE;
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
{banner}{force_mmio}{run_call}}}
"""


def runtime_cflags(config: RunnerConfig | None = None, *, extra_cflags: list[str] | None = None) -> list[str]:
    cfg = config or RunnerConfig()
    from tinynpu.program import TinyNPUProgram

    im_base_addr = int(TinyNPUProgram().hw.params.get("IM_BASE_ADDR", 0xF000))
    cflags = [
        "-w",
        "-O3",
        "-g",
        "-nostdlib",
        "-DTINYNPU_USE_SHARED_SRAM=1",
        f"-DTINY_IM_BASE_ADDR=0x{im_base_addr:X}u",
        f"-DTNPU_RUNTIME_V2_DUMP_FINAL_OUTPUTS={1 if cfg.dump_final_outputs else 0}",
        f"-DTNPU_RUNTIME_V2_VERBOSE_STEPS={1 if cfg.verbose_steps else 0}",
    ]
    if extra_cflags:
        cflags.extend(extra_cflags)
    return cflags


def toolchain_prefix() -> Path:
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
    if gcc is not None:
        return Path(gcc).resolve().parent / "riscv32-unknown-elf-"
    gcc = shutil.which("riscv64-unknown-elf-gcc")
    if gcc is not None:
        return Path(gcc).resolve().parent / "riscv64-unknown-elf-"
    raise FileNotFoundError("riscv32-unknown-elf-gcc or riscv64-unknown-elf-gcc not found in PATH")


def toolchain_root(prefix: Path) -> Path:
    return prefix.parent.parent


def toolchain_include_lib_dirs(prefix: Path) -> tuple[Path, Path]:
    root = toolchain_root(prefix)
    target = prefix.name[:-1]
    conventional_include = root / target / "include"
    conventional_lib = root / target / "lib"
    if conventional_include.exists() and conventional_lib.exists():
        return conventional_include, conventional_lib

    picolibc_root = Path("/usr/lib/picolibc") / target
    picolibc_include = picolibc_root / "include"
    picolibc_lib = picolibc_root / "lib" / TNPU_RISCV_MARCH / TNPU_RISCV_MABI
    if picolibc_include.exists() and picolibc_lib.exists():
        return picolibc_include, picolibc_lib
    if TNPU_RISCV_MARCH == "rv32imfc" and TNPU_RISCV_MABI == "ilp32f":
        ubuntu_lib = picolibc_root / "lib" / "rv32imafc" / "ilp32f"
        if picolibc_include.exists() and ubuntu_lib.exists():
            return picolibc_include, ubuntu_lib

    raise FileNotFoundError(f"No C library include/lib directories found for {target}")


def run_checked(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )


def build_v2_elf_and_hex(
    program_name: str,
    program_source: str,
    *,
    runner_config: RunnerConfig | None = None,
    extra_cflags: list[str] | None = None,
) -> tuple[Path, Path, Path, Path]:
    cfg = runner_config or RunnerConfig()
    program_symbol = sanitize_program_symbol(program_name)
    program_path = GENERATED_DIR / f"{program_name}_program.c"
    runner_path = GENERATED_DIR / f"{program_name}_runner.c"
    GENERATED_DIR.mkdir(exist_ok=True)
    program_path.write_text(program_source)
    runner_path.write_text(render_runner_source(program_symbol, cfg))

    prefix = toolchain_prefix()
    gcc = f"{prefix}gcc"
    objcopy = f"{prefix}objcopy"
    include_dir, lib_dir = toolchain_include_lib_dirs(prefix)
    elf_path = CUSTOM_DIR / f"{program_name}.elf"
    hex_path = CUSTOM_DIR / f"{program_name}.hex"

    build_env = dict(os.environ)
    build_env["CCACHE_DISABLE"] = "1"
    build_env["TMPDIR"] = "/tmp"
    run_checked(["make", "verilator-build-npu"], cwd=CORE_DIR, env=build_env)

    run_checked(
        [
            gcc,
            f"-march={TNPU_RISCV_MARCH}",
            f"-mabi={TNPU_RISCV_MABI}",
            "-o",
            str(elf_path),
            *runtime_cflags(cfg, extra_cflags=extra_cflags),
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
    run_checked([objcopy, "-O", "verilog", str(elf_path), str(hex_path)], cwd=CORE_DIR, env=build_env)
    return program_path, runner_path, elf_path, hex_path


def run_vlt_npu(
    hex_path: Path,
    *,
    maxcycles: int,
    verilator_max_ticks: int = 3_000_000_000,
    timeout_s: int | None = None,
    noassert: bool = False,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["VERILATOR_MAX_TICKS"] = str(verilator_max_ticks)
    cmd = [str(CORE_DIR / "obj_dir" / "cv32e40p_tb_vlt_npu")]
    if noassert:
        cmd.append("+verilator+noassert")
    cmd.extend([f"+firmware={hex_path}", f"+maxcycles={maxcycles}"])
    return subprocess.run(
        cmd,
        cwd=str(CORE_DIR),
        env=env,
        check=True,
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )


def _runner_source(program_symbol: str, config: RunnerConfig | None = None) -> str:
    return render_runner_source(program_symbol, config)
