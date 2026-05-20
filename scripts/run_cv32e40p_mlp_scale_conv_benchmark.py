from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
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
    HostOp,
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


@dataclass(frozen=True)
class ConvLayerSpec:
    name: str
    in_h: int
    in_w: int
    in_c: int
    kernel: int
    out_c: int
    activation: str

    @property
    def out_h(self) -> int:
        return self.in_h - self.kernel + 1

    @property
    def out_w(self) -> int:
        return self.in_w - self.kernel + 1

    @property
    def cols_shape(self) -> tuple[int, int]:
        return (self.out_h * self.out_w, self.kernel * self.kernel * self.in_c)

    @property
    def weight_shape(self) -> tuple[int, int]:
        return (self.kernel * self.kernel * self.in_c, self.out_c)

    @property
    def out_shape(self) -> tuple[int, int]:
        return (self.out_h * self.out_w, self.out_c)

    @property
    def macs(self) -> int:
        return self.out_h * self.out_w * self.kernel * self.kernel * self.in_c * self.out_c


MLP_SCALE_LAYERS = (
    ConvLayerSpec("conv1", in_h=8, in_w=8, in_c=1, kernel=3, out_c=8, activation="relu"),
    ConvLayerSpec("conv2", in_h=6, in_w=6, in_c=8, kernel=3, out_c=8, activation="relu"),
    ConvLayerSpec("conv3", in_h=4, in_w=4, in_c=8, kernel=3, out_c=1, activation="relu"),
    ConvLayerSpec("conv4", in_h=2, in_w=2, in_c=1, kernel=2, out_c=1, activation="none"),
)
ONNX_SHAPE_LAYERS = (
    ConvLayerSpec("conv1", in_h=8, in_w=8, in_c=1, kernel=3, out_c=16, activation="relu"),
    ConvLayerSpec("conv2", in_h=6, in_w=6, in_c=16, kernel=3, out_c=16, activation="relu"),
    ConvLayerSpec("conv3", in_h=4, in_w=4, in_c=16, kernel=3, out_c=16, activation="relu"),
    ConvLayerSpec("conv4", in_h=2, in_w=2, in_c=16, kernel=2, out_c=1, activation="none"),
)
WIDE32_LAYERS = (
    ConvLayerSpec("conv1", in_h=8, in_w=8, in_c=1, kernel=3, out_c=32, activation="relu"),
    ConvLayerSpec("conv2", in_h=6, in_w=6, in_c=32, kernel=3, out_c=32, activation="relu"),
    ConvLayerSpec("conv3", in_h=4, in_w=4, in_c=32, kernel=3, out_c=32, activation="relu"),
    ConvLayerSpec("conv4", in_h=2, in_w=2, in_c=32, kernel=2, out_c=1, activation="none"),
)
VARIANTS = {
    "mlp_scale": MLP_SCALE_LAYERS,
    "onnx_shape": ONNX_SHAPE_LAYERS,
    "wide32": WIDE32_LAYERS,
}
MLP_REFERENCE_MACS = (64 * 64 * 3) + 64


def _small_pattern(shape: tuple[int, ...], *, offset: int) -> np.ndarray:
    values = (np.arange(np.prod(shape), dtype=np.int32) + offset) % 3
    return (values - 1).reshape(shape).astype(np.int16)


def _im2col_hwc(image: np.ndarray, kernel: int) -> np.ndarray:
    h, w, c = image.shape
    patches: list[np.ndarray] = []
    for y in range(0, h - kernel + 1):
        for x in range(0, w - kernel + 1):
            patch = image[y : y + kernel, x : x + kernel, :]
            patches.append(patch.transpose(2, 0, 1).reshape(-1))
    return np.array(patches, dtype=image.dtype)


def _matmul_i16(lhs: np.ndarray, rhs: np.ndarray, *, activation: str) -> np.ndarray:
    out = lhs.astype(np.int32) @ rhs.astype(np.int32)
    if activation == "relu":
        out = np.maximum(out, 0)
    elif activation != "none":
        raise ValueError(f"unsupported activation: {activation}")
    return np.clip(out, -32768, 32767).astype(np.int16)


def _shape_chain(layers: tuple[ConvLayerSpec, ...]) -> str:
    parts = [f"{layers[0].in_h}x{layers[0].in_w}x{layers[0].in_c}"]
    parts.extend(f"{layer.out_h}x{layer.out_w}x{layer.out_c}" for layer in layers)
    if layers[-1].out_shape == (1, 1):
        parts[-1] = "1 scalar"
    return " -> ".join(parts)


def build_plan(
    *,
    variant: str,
    layers: tuple[ConvLayerSpec, ...],
) -> tuple[ExecutionPlan, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, int]]:
    x = _small_pattern((layers[0].in_h, layers[0].in_w, layers[0].in_c), offset=0)
    tensors: dict[str, TensorSpec] = {
        "x": TensorSpec("x", x.shape, DType.INT16, TensorKind.INPUT),
    }
    steps: list[HostOp | NpuSegment] = []
    expected: dict[str, np.ndarray] = {}

    prev_name = "x"
    prev_value = x
    for idx, layer in enumerate(layers):
        cols_name = f"{layer.name}_im2col"
        w_name = f"{layer.name}_w"
        out_name = f"{layer.name}_out"

        weights = _small_pattern(layer.weight_shape, offset=idx + 1)
        if idx == 0:
            attrs = {"kernel_size": layer.kernel, "stride": 1, "padding": 0, "input_layout": "hwc"}
            cols = _im2col_hwc(prev_value, layer.kernel)
        else:
            attrs = {
                "kernel_size": layer.kernel,
                "stride": 1,
                "padding": 0,
                "input_layout": "matrix_hwc",
                "matrix_h": layer.in_h,
                "matrix_w": layer.in_w,
                "matrix_c": layer.in_c,
            }
            cols = _im2col_hwc(prev_value.reshape(layer.in_h, layer.in_w, layer.in_c), layer.kernel)

        out = _matmul_i16(cols, weights, activation=layer.activation)
        expected[cols_name] = cols
        expected[out_name] = out

        tensors[cols_name] = TensorSpec(cols_name, cols.shape, DType.INT16, TensorKind.INTERMEDIATE)
        tensors[w_name] = TensorSpec(w_name, weights.shape, DType.INT16, TensorKind.CONSTANT, data=weights)
        tensors[out_name] = TensorSpec(
            out_name,
            out.shape,
            DType.INT16,
            TensorKind.OUTPUT if idx == len(layers) - 1 else TensorKind.INTERMEDIATE,
            is_final_output=idx == len(layers) - 1,
        )
        steps.append(HostOp(f"{layer.name}_im2col", "im2col", inputs=[prev_name], outputs=[cols_name], attrs=attrs))
        steps.append(
            NpuSegment(
                f"seg_{layer.name}",
                [
                    MatMulOp(
                        f"op_{layer.name}",
                        cols_name,
                        w_name,
                        out_name,
                        activation=layer.activation,
                    )
                ],
                inputs=[cols_name, w_name],
                outputs=[out_name],
            )
        )

        prev_name = out_name
        prev_value = out

    plan = ExecutionPlan(
        tensors=tensors,
        steps=steps,
        inputs=["x"],
        outputs=["conv4_out"],
        metadata={
            "description": "Valid-conv benchmark with selectable MLP-scale and ONNX-shape variants.",
            "variant": variant,
            "shape_chain": _shape_chain(layers),
            "logical_macs": sum(layer.macs for layer in layers),
            "mlp_reference_macs": MLP_REFERENCE_MACS,
        },
    )
    plan.add_verification_step("conv4_out", "final_conv_scalar")
    return plan, {"x": x}, {"conv4_out": expected["conv4_out"]}, {layer.name: layer.macs for layer in layers}


def parse_cycles(stdout: str) -> dict[str, int]:
    cycles: dict[str, int] = {}
    for label, value in re.findall(r"^([A-Za-z0-9_.-]+) cycles=(\d+)$", stdout, flags=re.MULTILINE):
        cycles[label] = int(value)
    return cycles


def write_report(
    *,
    output_path: Path,
    program_name: str,
    variant: str,
    shape_chain: str,
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
        f"""# Scaled Conv RTL Benchmark

Program: `{program_name}`

Variant: `{variant}`

This benchmark replaces the accidental 28x28 MNIST-style Conv run with a controlled valid-conv workload.

The bad Conv run that motivated this benchmark used a 28x28 input and produced huge `im2col` tensors, including `(784, 144)` for later layers. That makes host-side lowering dominate before the NPU does useful work, so it is not comparable to the small is-zero MLP benchmark.

| Workload | Logical MACs |
| --- | ---: |
| This Conv pipeline | {total_macs:,} |
| Is-zero MLP reference | {MLP_REFERENCE_MACS:,} |

| Conv stage | Logical MACs |
| --- | ---: |
{layer_rows}

Shape chain: `{shape_chain}`.

Repeat count: `{repeat_count}`.

## RTL Cycles

| Counter | Cycles |
| --- | ---: |
{cycle_rows}

## Status

`EXIT SUCCESS`: `{str("EXIT SUCCESS" in stdout)}`
"""
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an MLP-scale valid-conv TinyNPU RTL benchmark.")
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="mlp_scale")
    parser.add_argument("--program-name", default=None)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--maxcycles", type=int, default=100_000_000)
    parser.add_argument("--verilator-max-ticks", type=int, default=5_000_000_000)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--no-run-rtl", action="store_true")
    parser.add_argument(
        "--report",
        type=Path,
        default=RUNS_DIR / "MLP_SCALE_CONV_BENCHMARK_2026_05_20.md",
    )
    args = parser.parse_args()

    layers = VARIANTS[args.variant]
    program_name = args.program_name or f"cv32e40p_{args.variant}_conv_v2"

    plan, inputs, expected, macs_by_layer = build_plan(variant=args.variant, layers=layers)
    artifact = compile_plan(plan, expected)
    emu = artifact.run_host_emulation(inputs, verification=VerificationMode.DEBUG)
    if "final_conv_scalar" not in emu.verified:
        raise RuntimeError(f"host emulation did not verify final output: {emu.verified}")

    program_source = emit_cv32e40p_program_v2(artifact, inputs, program_name=program_name)
    if args.no_run_rtl:
        print(f"emitted {program_name}: variant={args.variant} logical_macs={sum(macs_by_layer.values())}")
        return 0

    _, _, _, hex_path = build_v2_elf_and_hex(
        program_name,
        program_source,
        runner_config=RunnerConfig(
            repeat_count=args.repeat_count,
            dump_final_outputs=True,
            verbose_steps=True,
            timed=args.repeat_count > 1,
            banner="Scaled Conv benchmark",
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
        variant=args.variant,
        shape_chain=_shape_chain(layers),
        macs_by_layer=macs_by_layer,
        cycles=cycles,
        stdout=sim.stdout,
        repeat_count=args.repeat_count,
    )
    print(f"report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
