from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
COMPILER_ROOT = REPO_ROOT / "software" / "compiler"
if str(COMPILER_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPILER_ROOT))

from software.compiler.tinynpu_jit import write_cv32e40p_program_v2  # noqa: E402
from software.workload.jit_multitile_matmul import (  # noqa: E402
    JitMatmulBenchmarkCase,
    build_configured_matmul_artifact,
    default_gemm_benchmark_cases,
)


CORE_DIR = REPO_ROOT / "external" / "cv32e40p" / "example_tb" / "core"
GENERATED_DIR = REPO_ROOT / "generated"
RUNTIME_DIR = REPO_ROOT / "software" / "compiler" / "tinynpu_jit"
TNPU_RISCV_MARCH = os.environ.get("TINYNPU_RISCV_MARCH", "rv32imfc")
TNPU_RISCV_MABI = os.environ.get("TINYNPU_RISCV_MABI", "ilp32f")
CUSTOM_DIR = CORE_DIR / "custom"
RUNS_DIR = REPO_ROOT / "runs"


@dataclass(frozen=True)
class RunMetrics:
    preload_total: int
    cold_npu: int
    cold_e2e_npu: int
    warm_avg_npu: int | None
    extrapolated_10x_e2e_npu: int | None


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _runner_source(program_symbol: str, repeat_count: int) -> str:
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
    return tinynpu_run_repeat(program, ip, op, NULL, 0u, {repeat_count}u);
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


def _ensure_verilator_built() -> None:
    _run(["make", "verilator-build-npu"], cwd=CORE_DIR)


def _emit_case_sources(case: JitMatmulBenchmarkCase, repeat_count: int) -> tuple[str, Path, Path]:
    artifact, inputs, _ = build_configured_matmul_artifact(case)
    lhs_mode = "runtimein" if case.lhs_runtime_input else "staticlhs"
    program_name = f"cv32e40p_{case.name}_{lhs_mode}_v2"
    program_symbol = _sanitize(program_name)
    program_path = GENERATED_DIR / f"{program_name}_program.c"
    runner_path = GENERATED_DIR / f"{program_name}_repeat{repeat_count}_runner.c"
    write_cv32e40p_program_v2(artifact, inputs, program_path, program_name=program_name)
    runner_path.write_text(_runner_source(program_symbol, repeat_count))
    return program_name, program_path, runner_path


def _compile_case(program_name: str, program_path: Path, runner_path: Path, prefix: Path) -> Path:
    gcc = f"{prefix}gcc"
    objcopy = f"{prefix}objcopy"
    toolchain_root = _toolchain_root(prefix)
    include_dir = toolchain_root / "riscv32-unknown-elf" / "include"
    lib_dir = toolchain_root / "riscv32-unknown-elf" / "lib"
    elf_path = CUSTOM_DIR / f"{program_name}.elf"
    hex_path = CUSTOM_DIR / f"{program_name}.hex"
    compile_cmd = [
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
        "-DTNPU_RUNTIME_V2_DUMP_FINAL_OUTPUTS=0",
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
    ]
    _run(compile_cmd, cwd=CORE_DIR)
    _run([objcopy, "-O", "verilog", str(elf_path), str(hex_path)], cwd=CORE_DIR)
    return hex_path


def _run_case(hex_path: Path, maxcycles: int, max_ticks: int) -> str:
    sim_bin = CORE_DIR / "obj_dir" / "cv32e40p_tb_vlt_npu"
    env = dict(os.environ)
    env["VERILATOR_MAX_TICKS"] = str(max_ticks)
    proc = _run(
        [
            str(sim_bin),
            "+verilator+noassert",
            f"+firmware={hex_path}",
            f"+maxcycles={maxcycles}",
        ],
        cwd=CORE_DIR,
        env=env,
        capture=True,
    )
    return proc.stdout


def _parse_metrics(stdout: str) -> RunMetrics:
    fields = {
        "preload_total": r"preload\.total cycles=(\d+)",
        "cold_npu": r"cold\.npu cycles=(\d+)",
        "cold_e2e_npu": r"cold\.e2e\.npu cycles=(\d+)",
    }
    values: dict[str, int] = {}
    for key, pattern in fields.items():
        match = re.search(pattern, stdout)
        if match is None:
            raise ValueError(f"Missing metric '{key}' in simulator output")
        values[key] = int(match.group(1))
    warm_match = re.search(r"warm\.avg\.npu cycles=(\d+)", stdout)
    extra_10x_match = re.search(r"extrapolated\.10x\.e2e\.npu cycles=(\d+)", stdout)
    if "EXIT SUCCESS" not in stdout:
        raise ValueError("Simulator output missing EXIT SUCCESS")
    lowered = stdout.lower()
    if "verification failed" in lowered or "autoverify failed" in lowered:
        raise ValueError("Simulator output contains verification failure")
    return RunMetrics(
        preload_total=values["preload_total"],
        cold_npu=values["cold_npu"],
        cold_e2e_npu=values["cold_e2e_npu"],
        warm_avg_npu=None if warm_match is None else int(warm_match.group(1)),
        extrapolated_10x_e2e_npu=None if extra_10x_match is None else int(extra_10x_match.group(1)),
    )


def _select_cases(names: list[str] | None) -> list[JitMatmulBenchmarkCase]:
    cases = default_gemm_benchmark_cases()
    if not names:
        return cases
    wanted = set(names)
    selected = [case for case in cases if case.name in wanted]
    missing = sorted(wanted - {case.name for case in selected})
    if missing:
        raise SystemExit(f"Unknown case name(s): {', '.join(missing)}")
    return selected


def _benchmark_case(case: JitMatmulBenchmarkCase, compile_lhs: bool) -> JitMatmulBenchmarkCase:
    if not compile_lhs:
        return case
    return replace(case, lhs_runtime_input=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V2 runtime GEMM E2E benchmarks on CV32E40P+TinyNPU.")
    parser.add_argument("--case", action="append", dest="cases", help="Case name, e.g. gemm_64x64x64_int16")
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--maxcycles", type=int, default=5_000_000)
    parser.add_argument("--max-ticks", type=int, default=30_000_000_000)
    parser.add_argument("--results-dir", type=Path, default=RUNS_DIR / "gemm_v2_e2e")
    parser.add_argument(
        "--lhs-compile-input",
        action="store_true",
        help="Treat lhs as a compile-time constant so it is prepacked into the static UB image.",
    )
    parser.add_argument("--skip-build", action="store_true")
    args = parser.parse_args()

    if args.repeat_count < 1:
        raise SystemExit("--repeat-count must be >= 1")

    prefix = _toolchain_prefix()
    cases = _select_cases(args.cases)
    results_dir = args.results_dir
    logs_dir = results_dir / "logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        _ensure_verilator_built()

    rows: list[dict[str, object]] = []
    compile_lhs = args.lhs_compile_input
    for raw_case in cases:
        case = _benchmark_case(raw_case, compile_lhs)
        program_name, program_path, runner_path = _emit_case_sources(case, args.repeat_count)
        hex_path = _compile_case(program_name, program_path, runner_path, prefix)
        stdout = _run_case(hex_path, args.maxcycles, args.max_ticks)
        (logs_dir / f"{program_name}.log").write_text(stdout)
        metrics = _parse_metrics(stdout)
        logical_macs = case.total_macs
        cold_e2e_macs_per_cycle = logical_macs / metrics.cold_e2e_npu
        warm_e2e_macs_per_cycle = (
            None if metrics.warm_avg_npu is None else logical_macs / metrics.warm_avg_npu
        )
        row = {
            "case": case.name,
            "m": case.m,
            "k": case.k,
            "n": case.n,
            "dtype": case.in_dtype.value,
            "lhs_mode": "runtime" if case.lhs_runtime_input else "static_prepacked",
            "algorithmic_macs": logical_macs,
            "preload_total": metrics.preload_total,
            "cold_npu": metrics.cold_npu,
            "cold_e2e_npu": metrics.cold_e2e_npu,
            "cold_e2e_macs_per_cycle": f"{cold_e2e_macs_per_cycle:.4f}",
            "warm_avg_npu": "" if metrics.warm_avg_npu is None else metrics.warm_avg_npu,
            "warm_e2e_macs_per_cycle": "" if warm_e2e_macs_per_cycle is None else f"{warm_e2e_macs_per_cycle:.4f}",
            "extrapolated_10x_e2e_npu": "" if metrics.extrapolated_10x_e2e_npu is None else metrics.extrapolated_10x_e2e_npu,
        }
        rows.append(row)
        summary = (
            f"{case.name}: macs={logical_macs} cold.e2e={metrics.cold_e2e_npu} "
            f"cold_mac_per_cycle={cold_e2e_macs_per_cycle:.4f} "
            f"lhs_mode={'runtime' if case.lhs_runtime_input else 'static_prepacked'}"
        )
        if metrics.warm_avg_npu is not None and warm_e2e_macs_per_cycle is not None:
            summary += (
                f" warm.avg={metrics.warm_avg_npu} "
                f"warm_mac_per_cycle={warm_e2e_macs_per_cycle:.4f}"
            )
        print(summary)

    csv_path = results_dir / "gemm_v2_e2e_results.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "case",
                "m",
                "k",
                "n",
                "dtype",
                "lhs_mode",
                "algorithmic_macs",
                "preload_total",
                "cold_npu",
                "cold_e2e_npu",
                "cold_e2e_macs_per_cycle",
                "warm_avg_npu",
                "warm_e2e_macs_per_cycle",
                "extrapolated_10x_e2e_npu",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
