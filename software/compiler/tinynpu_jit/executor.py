from __future__ import annotations

import math

import numpy as np

from .artifact import CompiledArtifact, ExecutionResult
from .benchmark import BenchmarkReport, CostModel, estimate_host_op_counts, estimate_npu_segment_cpu_counts
from .golden import GoldenModel
from .ir import HostOp, NpuSegment, TensorKind, VerificationMode, VerifyTensor


class HostEmulationExecutor:
    def __init__(self):
        self.golden = GoldenModel()

    def run(
        self,
        artifact: CompiledArtifact,
        inputs: dict[str, np.ndarray],
        verification: VerificationMode = VerificationMode.OFF,
        debug: bool = False,
        benchmark: bool = False,
        cost_model: CostModel | None = None,
    ) -> ExecutionResult:
        values: dict[str, np.ndarray] = {}
        for name, spec in artifact.plan.tensors.items():
            if spec.kind == TensorKind.CONSTANT and spec.data is not None:
                values[name] = np.array(spec.data, copy=True)

        for name in artifact.plan.inputs:
            if name not in inputs:
                raise KeyError(f"Missing runtime input '{name}'.")
            values[name] = np.array(inputs[name], copy=True)

        verified: list[str] = []
        debug_trace: list[dict[str, np.ndarray | str | dict[str, object] | list[str] | bool]] = []
        benchmark_report = BenchmarkReport(cost_model=cost_model or CostModel()) if benchmark else None
        for step in artifact.plan.steps:
            if isinstance(step, NpuSegment):
                self._run_npu_segment(step, values)
                if benchmark_report is not None:
                    benchmark_report.add_entry(
                        step=step.name,
                        bucket="cpu_replaced",
                        counts=estimate_npu_segment_cpu_counts(step, artifact.plan.tensors),
                        attrs={"kind": "npu_segment", "op_count": len(step.ops)},
                    )
                if debug:
                    debug_trace.append(
                        self._debug_event(
                            step=step.name,
                            kind="npu_segment",
                            inputs={name: values[name] for name in step.inputs if name in values},
                            outputs={name: values[name] for name in step.outputs if name in values},
                            attrs={
                                "ops": [
                                    {
                                        "name": op.name,
                                        "lhs": op.lhs,
                                        "rhs": op.rhs,
                                        "out": op.out,
                                        "bias": op.bias,
                                        "multiplier": op.multiplier,
                                        "shift": op.shift,
                                        "activation": op.activation,
                                        "in_dtype": op.in_dtype.value,
                                        "out_dtype": op.out_dtype.value,
                                    }
                                    for op in step.ops
                                ]
                            },
                        )
                    )
            elif isinstance(step, HostOp):
                self._run_host_op(step, values)
                if benchmark_report is not None:
                    bucket, counts = estimate_host_op_counts(step, values)
                    benchmark_report.add_entry(
                        step=step.name,
                        bucket=bucket,
                        counts=counts,
                        attrs={"kind": step.kind},
                    )
                if debug:
                    debug_trace.append(
                        self._debug_event(
                            step=step.name,
                            kind=f"host_{step.kind}",
                            inputs={name: values[name] for name in step.inputs if name in values},
                            outputs={name: values[name] for name in step.outputs if name in values},
                            attrs=step.attrs,
                        )
                    )
            elif isinstance(step, VerifyTensor):
                if self._should_verify(step, verification):
                    expected = artifact.expected_tensors[step.tensor_name]
                    actual = values[step.tensor_name]
                    if np.issubdtype(actual.dtype, np.floating) or np.issubdtype(expected.dtype, np.floating):
                        matches = np.allclose(actual, expected, rtol=1e-5, atol=1e-6)
                    else:
                        matches = np.array_equal(actual, expected)
                    if not matches:
                        raise AssertionError(
                            f"Verification failed for '{step.label}' ({step.tensor_name})."
                        )
                    verified.append(step.label)
                    if debug:
                        debug_trace.append(
                            self._debug_event(
                                step=step.label,
                                kind="verify",
                                inputs={step.tensor_name: actual},
                                outputs={},
                                attrs={"tensor_name": step.tensor_name, "is_final_output": step.is_final_output},
                            )
                        )

        outputs = {name: np.array(values[name], copy=True) for name in artifact.plan.outputs}
        trace_tensors = {name: np.array(value, copy=True) for name, value in values.items()}
        return ExecutionResult(
            tensors=outputs,
            verified=verified,
            trace_tensors=trace_tensors,
            debug_trace=debug_trace,
            benchmark=benchmark_report,
        )

    def _should_verify(self, step: VerifyTensor, verification: VerificationMode) -> bool:
        if verification == VerificationMode.OFF:
            return False
        if verification == VerificationMode.FINAL:
            return step.is_final_output
        return True

    def _run_npu_segment(self, step: NpuSegment, values: dict[str, np.ndarray]) -> None:
        for op in step.ops:
            bias = values.get(op.bias) if op.bias else None
            activation = "relu" if op.activation == "relu" else "none"
            values[op.out] = self.golden.matmul(
                values[op.lhs],
                values[op.rhs],
                bias=bias,
                multiplier=op.multiplier,
                shift=op.shift,
                activation=activation,
                out_dtype=op.out_dtype,
            )

    def _run_host_op(self, step: HostOp, values: dict[str, np.ndarray]) -> None:
        if step.kind == "softmax":
            axis = int(step.attrs.get("axis", -1))
            values[step.outputs[0]] = self.golden.softmax(values[step.inputs[0]], axis=axis)
            return
        if step.kind == "quantize":
            scale = float(step.attrs["scale"])
            zero_point = int(step.attrs.get("zero_point", 0))
            out_dtype = step.attrs.get("dtype", "int8")
            values[step.outputs[0]] = self.golden.quantize(
                values[step.inputs[0]],
                scale=scale,
                zero_point=zero_point,
                out_dtype=out_dtype,
            )
            return
        if step.kind == "dequantize":
            scale = float(step.attrs["scale"])
            zero_point = int(step.attrs.get("zero_point", 0))
            values[step.outputs[0]] = self.golden.dequantize(
                values[step.inputs[0]],
                scale=scale,
                zero_point=zero_point,
            )
            return
        if step.kind == "sigmoid":
            source = np.array(values[step.inputs[0]], dtype=np.float32)
            values[step.outputs[0]] = 1.0 / (1.0 + np.exp(-source))
            return
        if step.kind == "relu":
            values[step.outputs[0]] = np.maximum(values[step.inputs[0]], 0)
            return
        if step.kind == "mean":
            source = np.array(values[step.inputs[0]], copy=False)
            dims = step.attrs.get("dim")
            axis = None if dims is None else tuple(int(dim) for dim in dims)
            keepdim = bool(step.attrs.get("keepdim", False))
            input_quant = step.attrs.get("input_quantization")
            if input_quant is not None:
                source = self.golden.dequantize(
                    source,
                    scale=float(input_quant["scale"]),
                    zero_point=int(input_quant.get("zero_point", 0)),
                )
            mean_value = np.mean(source.astype(np.float32), axis=axis, keepdims=keepdim)
            values[step.outputs[0]] = mean_value.astype(np.float32)
            return
        if step.kind == "alias":
            values[step.outputs[0]] = np.array(values[step.inputs[0]], copy=True)
            return
        if step.kind == "im2col":
            image = np.array(values[step.inputs[0]], copy=False)
            layout = step.attrs.get("input_layout", "hwc")
            if layout == "chw":
                if image.ndim == 4:
                    if image.shape[0] != 1:
                        raise NotImplementedError(
                            f"im2col host op only supports batch size 1 for CHW input, got shape {image.shape}."
                        )
                    image = image[0]
                image = np.transpose(image, (1, 2, 0))
            values[step.outputs[0]] = self.golden.im2col(
                image,
                kernel_size=int(step.attrs["kernel_size"]),
                stride=int(step.attrs.get("stride", 1)),
                padding=int(step.attrs.get("padding", 0)),
            )
            return
        if step.kind == "reshape":
            values[step.outputs[0]] = np.reshape(values[step.inputs[0]], tuple(step.attrs["shape"]))
            return
        if step.kind == "transpose":
            values[step.outputs[0]] = np.transpose(values[step.inputs[0]], axes=tuple(step.attrs.get("axes", [])) or None)
            return
        if step.kind == "layout_restore":
            source = np.array(values[step.inputs[0]], copy=False)
            hwc = source.reshape(
                int(step.attrs["out_h"]),
                int(step.attrs["out_w"]),
                int(step.attrs["out_channels"]),
            )
            if step.attrs["layout"] == "chw":
                restored = np.transpose(hwc, (2, 0, 1))
                original_shape = tuple(step.attrs["original_shape"])
                if len(original_shape) == 4:
                    restored = np.expand_dims(restored, axis=0)
                values[step.outputs[0]] = restored
                return
            if step.attrs["layout"] == "hwc":
                values[step.outputs[0]] = hwc
                return
            raise ValueError(f"Unsupported layout_restore layout {step.attrs['layout']!r}.")
        if step.kind == "requantize":
            scale = float(step.attrs["scale"])
            zero_point = int(step.attrs.get("zero_point", 0))
            out_dtype = step.attrs.get("dtype", "int16")
            values[step.outputs[0]] = self.golden.requantize(
                values[step.inputs[0]],
                scale=scale,
                zero_point=zero_point,
                out_dtype=out_dtype,
            )
            return
        raise NotImplementedError(f"Unsupported host op '{step.kind}'.")

    def _debug_event(
        self,
        *,
        step: str,
        kind: str,
        inputs: dict[str, np.ndarray],
        outputs: dict[str, np.ndarray],
        attrs: dict[str, object],
    ) -> dict[str, object]:
        return {
            "step": step,
            "kind": kind,
            "inputs": {name: self._tensor_summary(value) for name, value in inputs.items()},
            "outputs": {name: self._tensor_summary(value) for name, value in outputs.items()},
            "attrs": dict(attrs),
        }

    def _tensor_summary(self, value: np.ndarray) -> dict[str, object]:
        arr = np.array(value)
        flat = arr.reshape(-1)
        preview = flat[: min(8, flat.size)].tolist()
        summary: dict[str, object] = {
            "shape": tuple(int(dim) for dim in arr.shape),
            "dtype": str(arr.dtype),
            "preview": preview,
        }
        if flat.size:
            summary["min"] = float(arr.min()) if np.issubdtype(arr.dtype, np.floating) else int(arr.min())
            summary["max"] = float(arr.max()) if np.issubdtype(arr.dtype, np.floating) else int(arr.max())
        return summary
