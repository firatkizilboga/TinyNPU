from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
COMPILER_ROOT = REPO_ROOT / "software" / "compiler"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(COMPILER_ROOT) not in sys.path:
    sys.path.insert(0, str(COMPILER_ROOT))

from tinynpu_jit import (  # noqa: E402
    DType,
    ExecutionPlan,
    MatMulOp,
    NpuSegment,
    RunnerConfig,
    TensorKind,
    TensorSpec,
    VerificationMode,
    build_v2_elf_and_hex,
    compile_plan,
    emit_cv32e40p_program_v2,
    run_vlt_npu,
)


RUNS_DIR = REPO_ROOT / "runs"


def _small_pattern(shape: tuple[int, ...], *, offset: int) -> np.ndarray:
    values = (np.arange(np.prod(shape), dtype=np.int32) + offset) % 3
    return (values - 1).reshape(shape).astype(np.int16)


def _matmul_i16(lhs: np.ndarray, rhs: np.ndarray, *, activation: str) -> np.ndarray:
    out = lhs.astype(np.int32) @ rhs.astype(np.int32)
    if activation == "relu":
        out = np.maximum(out, 0)
    elif activation != "none":
        raise ValueError(f"unsupported activation: {activation}")
    return np.clip(out, -32768, 32767).astype(np.int16)


def build_plan(*, hidden_dim: int) -> tuple[ExecutionPlan, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    x = _small_pattern((1, hidden_dim), offset=0)
    layer_shapes = (
        ("fc1", (hidden_dim, hidden_dim), "relu"),
        ("fc2", (hidden_dim, hidden_dim), "relu"),
        ("fc3", (hidden_dim, hidden_dim), "relu"),
        ("fc4", (hidden_dim, 1), "none"),
    )

    tensors: dict[str, TensorSpec] = {
        "x": TensorSpec("x", x.shape, DType.INT16, TensorKind.INPUT),
    }
    steps: list[NpuSegment] = []
    macs_by_layer: dict[str, int] = {}

    prev_name = "x"
    prev_value = x
    final_name = ""
    for idx, (name, weight_shape, activation) in enumerate(layer_shapes):
        w_name = f"{name}_w"
        out_name = f"{name}_out"
        weights = _small_pattern(weight_shape, offset=idx + 1)
        out = _matmul_i16(prev_value, weights, activation=activation)
        macs_by_layer[name] = int(prev_value.shape[0] * prev_value.shape[1] * weights.shape[1])

        tensors[w_name] = TensorSpec(w_name, weights.shape, DType.INT16, TensorKind.CONSTANT, data=weights)
        tensors[out_name] = TensorSpec(
            out_name,
            out.shape,
            DType.INT16,
            TensorKind.OUTPUT if idx == len(layer_shapes) - 1 else TensorKind.INTERMEDIATE,
            is_final_output=idx == len(layer_shapes) - 1,
        )
        steps.append(
            NpuSegment(
                f"seg_{name}",
                [MatMulOp(f"op_{name}", prev_name, w_name, out_name, activation=activation)],
                inputs=[prev_name, w_name],
                outputs=[out_name],
            )
        )
        prev_name = out_name
        prev_value = out
        final_name = out_name

    plan = ExecutionPlan(
        tensors=tensors,
        steps=steps,
        inputs=["x"],
        outputs=[final_name],
        metadata={
            "description": "Scaled 4-layer MLP benchmark.",
            "hidden_dim": hidden_dim,
            "logical_macs": sum(macs_by_layer.values()),
        },
    )
    plan.add_verification_step(final_name, "final_mlp_scalar")
    return plan, {"x": x}, {final_name: prev_value}, macs_by_layer


def parse_cycles(stdout: str) -> dict[str, int]:
    cycles: dict[str, int] = {}
    for label, value in re.findall(r"^([A-Za-z0-9_.-]+) cycles=(\d+)$", stdout, flags=re.MULTILINE):
        cycles[label] = int(value)
    return cycles


def write_report(
    *,
    output_path: Path,
    program_name: str,
    hidden_dim: int,
    macs_by_layer: dict[str, int],
    cycles: dict[str, int],
    stdout: str,
    repeat_count: int,
) -> None:
    total_macs = sum(macs_by_layer.values())
    layer_rows = "\n".join(f"| {name} | {macs:,} |" for name, macs in macs_by_layer.items())
    cycle_rows = "\n".join(f"| `{name}` | {value:,} |" for name, value in sorted(cycles.items()))
    if not cycle_rows:
        cycle_rows = "| n/a | n/a |"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"""# Scaled MLP RTL Benchmark

Program: `{program_name}`

Shape: `1x{hidden_dim} -> 1x{hidden_dim} -> 1x{hidden_dim} -> 1x{hidden_dim} -> 1 scalar`.

Repeat count: `{repeat_count}`.

| Workload | Logical MACs |
| --- | ---: |
| This MLP pipeline | {total_macs:,} |

| MLP stage | Logical MACs |
| --- | ---: |
{layer_rows}

## RTL Cycles

| Counter | Cycles |
| --- | ---: |
{cycle_rows}

## Status

`EXIT SUCCESS`: `{str("EXIT SUCCESS" in stdout)}`
"""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a scaled four-layer TinyNPU MLP RTL benchmark.")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--program-name", default=None)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--maxcycles", type=int, default=100_000_000)
    parser.add_argument("--verilator-max-ticks", type=int, default=5_000_000_000)
    parser.add_argument("--timeout-s", type=int, default=420)
    parser.add_argument("--no-run-rtl", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=RUNS_DIR / "SCALED_MLP_BENCHMARK_2026_05_20.md",
    )
    args = parser.parse_args()

    program_name = args.program_name or f"cv32e40p_scaled_mlp_h{args.hidden_dim}_v2"
    plan, inputs, expected, macs_by_layer = build_plan(hidden_dim=args.hidden_dim)
    artifact = compile_plan(plan, expected)
    emu = artifact.run_host_emulation(inputs, verification=VerificationMode.DEBUG)
    if "final_mlp_scalar" not in emu.verified:
        raise RuntimeError(f"host emulation did not verify final output: {emu.verified}")

    program_source = emit_cv32e40p_program_v2(artifact, inputs, program_name=program_name)
    if args.no_run_rtl:
        print(f"emitted {program_name}: hidden_dim={args.hidden_dim} logical_macs={sum(macs_by_layer.values())}")
        return 0

    _, _, _, hex_path = build_v2_elf_and_hex(
        program_name,
        program_source,
        runner_config=RunnerConfig(
            repeat_count=args.repeat_count,
            dump_final_outputs=True,
            verbose_steps=True,
            timed=args.repeat_count > 1,
            banner="Scaled MLP benchmark",
        ),
        extra_cflags=["-ffast-math", "-fno-builtin-printf"],
    )
    sim = run_vlt_npu(
        hex_path,
        maxcycles=args.maxcycles,
        verilator_max_ticks=args.verilator_max_ticks,
        timeout_s=args.timeout_s,
        noassert=True,
    )
    print(sim.stdout, end="")
    if sim.stderr:
        print(sim.stderr, file=sys.stderr, end="")
    if "EXIT SUCCESS" not in sim.stdout:
        raise RuntimeError("simulation did not report EXIT SUCCESS")

    cycles = parse_cycles(sim.stdout)
    write_report(
        output_path=args.report,
        program_name=program_name,
        hidden_dim=args.hidden_dim,
        macs_by_layer=macs_by_layer,
        cycles=cycles,
        stdout=sim.stdout,
        repeat_count=args.repeat_count,
    )
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
