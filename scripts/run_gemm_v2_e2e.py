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

from software.compiler.tinynpu_jit import DType, NpuSegment, VerifyTensor, write_cv32e40p_program_v2  # noqa: E402
from software.compiler.tinynpu_jit.rtl_runner import (  # noqa: E402
    RunnerConfig,
    runtime_cflags,
    toolchain_include_lib_dirs as _toolchain_include_lib_dirs,
    toolchain_prefix as _shared_toolchain_prefix,
)
from software.workload.jit_multitile_matmul import (  # noqa: E402
    JitMatmulBenchmarkCase,
    build_configured_matmul_artifact,
    default_gemm_benchmark_cases,
)
from tinynpu import TinyNPUProgram  # noqa: E402


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
    segment_run: int
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
    return _shared_toolchain_prefix()


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


def _strip_readback_and_verify(artifact) -> None:
    """Keep NPU execution intact but avoid large output readback/compare loops."""
    for step in artifact.plan.steps:
        if isinstance(step, NpuSegment):
            step.outputs = []
    artifact.plan.steps = [step for step in artifact.plan.steps if not isinstance(step, VerifyTensor)]
    artifact.plan.outputs = []
    artifact.expected_tensors = {}


def _write_resident_init_files(artifact, program_name: str, results_dir: Path) -> tuple[Path, Path]:
    init_dir = results_dir / "init"
    init_dir.mkdir(parents=True, exist_ok=True)
    ub_path = init_dir / f"{program_name}_ub_init.hex"
    im_path = init_dir / f"{program_name}_im_init.hex"

    with ub_path.open("w") as f:
        for word in artifact.static_ub_image or []:
            f.write(f"{int(word) & ((1 << 128) - 1):032x}\n")

    im_base_addr = int(TinyNPUProgram().hw.params.get("IM_BASE_ADDR", 0xF000))
    im_rows: list[int] = []
    next_im_addr = im_base_addr
    for step in artifact.plan.steps:
        if not isinstance(step, NpuSegment):
            continue
        segment = artifact.segment_artifacts[step.name]
        instructions = [int(inst) for inst in segment.binary["im"]]
        halt_instruction = 0xF << 252
        if instructions and ((instructions[-1] >> 252) & 0xF) == 0xF:
            halt_instruction = instructions[-1]
            instructions = instructions[:-1]
        row = (next_im_addr - im_base_addr) // 2
        while len(im_rows) < row:
            im_rows.append(0)
        im_rows.extend(instructions + [halt_instruction])
        next_im_addr += 2 * (len(instructions) + 1)

    with im_path.open("w") as f:
        for word in im_rows:
            f.write(f"{int(word) & ((1 << 256) - 1):064x}\n")
    return ub_path.resolve(), im_path.resolve()


def _emit_case_sources(
    case: JitMatmulBenchmarkCase,
    repeat_count: int,
    *,
    kernel_only: bool,
    resident_init: bool,
    results_dir: Path,
) -> tuple[str, Path, Path, list[str]]:
    artifact, inputs, _ = build_configured_matmul_artifact(case)
    if kernel_only:
        _strip_readback_and_verify(artifact)
    lhs_mode = "runtimein" if case.lhs_runtime_input else "staticlhs"
    suffix = ("_resident" if resident_init else "") + ("_kernelonly" if kernel_only else "")
    program_name = f"cv32e40p_{case.name}_{lhs_mode}{suffix}_v2"
    program_symbol = _sanitize(program_name)
    program_path = GENERATED_DIR / f"{program_name}_program.c"
    runner_path = GENERATED_DIR / f"{program_name}_repeat{repeat_count}_runner.c"
    sim_args: list[str] = []
    if resident_init:
        ub_init, im_init = _write_resident_init_files(artifact, program_name, results_dir)
        sim_args.extend([f"+npu_ub_init={ub_init}", f"+npu_im_init={im_init}"])
    write_cv32e40p_program_v2(
        artifact,
        inputs,
        program_path,
        program_name=program_name,
        emit_preloads=not resident_init,
    )
    runner_path.write_text(_runner_source(program_symbol, repeat_count))
    return program_name, program_path, runner_path, sim_args


def _compile_case(program_name: str, program_path: Path, runner_path: Path, prefix: Path) -> Path:
    gcc = f"{prefix}gcc"
    objcopy = f"{prefix}objcopy"
    include_dir, lib_dir = _toolchain_include_lib_dirs(prefix)
    elf_path = CUSTOM_DIR / f"{program_name}.elf"
    hex_path = CUSTOM_DIR / f"{program_name}.hex"
    compile_cmd = [
        gcc,
        f"-march={TNPU_RISCV_MARCH}",
        f"-mabi={TNPU_RISCV_MABI}",
        "-o",
        str(elf_path),
        *runtime_cflags(RunnerConfig(dump_final_outputs=False, verbose_steps=True)),
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


def _run_case(hex_path: Path, maxcycles: int, max_ticks: int, extra_args: list[str] | None = None) -> str:
    sim_bin = CORE_DIR / "obj_dir" / "cv32e40p_tb_vlt_npu"
    env = dict(os.environ)
    env["VERILATOR_MAX_TICKS"] = str(max_ticks)
    proc = subprocess.run(
        [
            str(sim_bin),
            "+verilator+noassert",
            f"+firmware={hex_path}",
            f"+maxcycles={maxcycles}",
            *(extra_args or []),
        ],
        cwd=str(CORE_DIR),
        env=env,
        check=False,
        text=True,
        capture_output=True,
    )
    output = proc.stdout + proc.stderr
    return output


def _parse_metrics(stdout: str) -> RunMetrics:
    fields = {
        "preload_total": r"preload\.total cycles=(\d+)",
        "segment_run": r"segment\.[^.]+\.run cycles=(\d+)",
        "cold_npu": r"cold\.(?:body|npu) cycles=(\d+)",
        "cold_e2e_npu": r"cold\.e2e(?:\.npu)? cycles=(\d+)",
    }
    values: dict[str, int] = {}
    for key, pattern in fields.items():
        match = re.search(pattern, stdout)
        if match is None:
            raise ValueError(f"Missing metric '{key}' in simulator output")
        values[key] = int(match.group(1))
    warm_match = re.search(r"warm\.avg\.(?:body|npu) cycles=(\d+)", stdout)
    extra_10x_match = re.search(r"extrapolated\.10x\.e2e(?:\.npu)? cycles=(\d+)", stdout)
    if "EXIT SUCCESS" not in stdout:
        raise ValueError("Simulator output missing EXIT SUCCESS")
    lowered = stdout.lower()
    if "verification failed" in lowered or "autoverify failed" in lowered:
        raise ValueError("Simulator output contains verification failure")
    return RunMetrics(
        preload_total=values["preload_total"],
        segment_run=values["segment_run"],
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


def _dtype_from_name(name: str) -> DType:
    try:
        return DType(name)
    except ValueError as exc:
        valid = ", ".join(dtype.value for dtype in (DType.INT16, DType.INT8, DType.INT4))
        raise SystemExit(f"Unsupported GEMM dtype '{name}'. Expected one of: {valid}") from exc


def _custom_cases(shape: tuple[int, int, int], dtype_names: list[str] | None) -> list[JitMatmulBenchmarkCase]:
    m, k, n = shape
    if m <= 0 or k <= 0 or n <= 0:
        raise SystemExit("--shape dimensions must be positive")
    dtypes = [_dtype_from_name(name) for name in dtype_names] if dtype_names else [DType.INT16, DType.INT8, DType.INT4]
    cases: list[JitMatmulBenchmarkCase] = []
    for index, dtype in enumerate(dtypes, start=1):
        cases.append(
            JitMatmulBenchmarkCase(
                name=f"gemm_{m}x{k}x{n}_{dtype.value}",
                m=m,
                k=k,
                n=n,
                in_dtype=dtype,
                out_dtype=DType.INT16,
                lhs_runtime_input=True,
                seed=index,
            )
        )
    return cases


def _benchmark_case(case: JitMatmulBenchmarkCase, compile_lhs: bool) -> JitMatmulBenchmarkCase:
    if not compile_lhs:
        return case
    return replace(case, lhs_runtime_input=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V2 runtime GEMM E2E benchmarks on CV32E40P+TinyNPU.")
    parser.add_argument("--case", action="append", dest="cases", help="Case name, e.g. gemm_64x64x64_int16")
    parser.add_argument("--shape", nargs=3, type=int, metavar=("M", "K", "N"), help="Run an ad-hoc GEMM shape.")
    parser.add_argument("--dtype", action="append", choices=["int16", "int8", "int4"], help="Precision for --shape; may be repeated. Defaults to all precisions.")
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
    parser.add_argument(
        "--kernel-only",
        action="store_true",
        help="Do not read back or verify outputs. Use only for segment.run kernel-efficiency probes.",
    )
    parser.add_argument(
        "--resident-init",
        action="store_true",
        help="Initialize UB/IM from testbench files and skip firmware preload ops.",
    )
    args = parser.parse_args()

    if args.repeat_count < 1:
        raise SystemExit("--repeat-count must be >= 1")

    prefix = _toolchain_prefix()
    if args.shape and args.cases:
        raise SystemExit("--shape and --case are mutually exclusive")
    cases = _custom_cases(tuple(args.shape), args.dtype) if args.shape else _select_cases(args.cases)
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
        program_name, program_path, runner_path, sim_args = _emit_case_sources(
            case,
            args.repeat_count,
            kernel_only=args.kernel_only,
            resident_init=args.resident_init,
            results_dir=results_dir,
        )
        hex_path = _compile_case(program_name, program_path, runner_path, prefix)
        stdout = _run_case(hex_path, args.maxcycles, args.max_ticks, sim_args)
        (logs_dir / f"{program_name}.log").write_text(stdout)
        metrics = _parse_metrics(stdout)
        logical_macs = case.total_macs
        cold_kernel_macs_per_cycle = logical_macs / metrics.segment_run
        warm_kernel_macs_per_cycle = (
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
            "cold_kernel_npu": metrics.segment_run,
            "cold_body_npu": metrics.cold_npu,
            "cold_e2e_npu": metrics.cold_e2e_npu,
            "cold_kernel_macs_per_cycle": f"{cold_kernel_macs_per_cycle:.4f}",
            "warm_avg_npu": "" if metrics.warm_avg_npu is None else metrics.warm_avg_npu,
            "warm_kernel_macs_per_cycle": "" if warm_kernel_macs_per_cycle is None else f"{warm_kernel_macs_per_cycle:.4f}",
            "extrapolated_10x_e2e_npu": "" if metrics.extrapolated_10x_e2e_npu is None else metrics.extrapolated_10x_e2e_npu,
        }
        rows.append(row)
        summary = (
            f"{case.name}: macs={logical_macs} cold.kernel={metrics.segment_run} "
            f"cold.body={metrics.cold_npu} "
            f"cold.e2e={metrics.cold_e2e_npu} "
            f"cold_kernel_mac_per_cycle={cold_kernel_macs_per_cycle:.4f} "
            f"lhs_mode={'runtime' if case.lhs_runtime_input else 'static_prepacked'}"
        )
        if metrics.warm_avg_npu is not None and warm_kernel_macs_per_cycle is not None:
            summary += (
                f" warm.avg={metrics.warm_avg_npu} "
                f"warm_kernel_mac_per_cycle={warm_kernel_macs_per_cycle:.4f}"
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
                "cold_kernel_npu",
                "cold_body_npu",
                "cold_e2e_npu",
                "cold_kernel_macs_per_cycle",
                "warm_avg_npu",
                "warm_kernel_macs_per_cycle",
                "extrapolated_10x_e2e_npu",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
