from __future__ import annotations

import math

import numpy as np

from .artifact import CompiledArtifact, ExecutionResult
from .benchmark import BenchmarkReport, CostModel, estimate_host_op_counts, estimate_npu_segment_cpu_counts
from .golden import GoldenModel
from .host_ops import execute_host_op
from .ir import DType, HostOp, NpuSegment, TensorKind, VerificationMode, VerifyTensor


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
            elif (
                spec.kind != TensorKind.CONSTANT
                and spec.metadata.get("cache_kind") in {"K", "V"}
                and not spec.metadata.get("storage_view_of")
            ):
                dtype = np.float32 if spec.dtype == DType.FLOAT32 else np.int16
                values[name] = np.zeros(spec.shape, dtype=dtype)

        for name in artifact.plan.inputs:
            if name not in inputs:
                raise KeyError(f"Missing runtime input '{name}'.")
            values[name] = np.array(inputs[name], copy=True)

        verified: list[str] = []
        debug_trace: list[dict[str, np.ndarray | str | dict[str, object] | list[str] | bool]] = []
        benchmark_report = BenchmarkReport(cost_model=cost_model or CostModel()) if benchmark else None
        for step in artifact.plan.steps:
            if isinstance(step, NpuSegment):
                self._run_npu_segment(step, values, artifact.plan.tensors)
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

    def _run_npu_segment(self, step: NpuSegment, values: dict[str, np.ndarray], tensors: dict[str, object]) -> None:
        for op in step.ops:
            bias = values.get(op.bias) if op.bias else None
            values[op.out] = self.golden.matmul(
                values[op.lhs],
                values[op.rhs],
                bias=bias,
                multiplier=op.multiplier,
                shift=op.shift,
                activation=op.activation,
                h_gelu_x_scale_shift=op.h_gelu_x_scale_shift,
                out_dtype=op.out_dtype,
            )
            if op.rope_cs_name and op.rope_cs_name in values:
                # Apply RoPE in-place on K using INT16 Q14 cos/sin table.
                # cs layout: flat [1, d_head] where [:half]=cos, [half:]=sin (Q14)
                k = np.asarray(values[op.out], dtype=np.int32).reshape(-1)
                cs = np.asarray(values[op.rope_cs_name], dtype=np.int32).reshape(-1)
                d = len(k)
                half = d // 2
                cos_q14 = cs[:half]
                sin_q14 = cs[half:]
                k_lo = k[:half].copy()
                k_hi = k[half:].copy()
                k_lo_rot = np.clip((k_lo * cos_q14 - k_hi * sin_q14 + (1 << 13)) >> 14, -32768, 32767)
                k_hi_rot = np.clip((k_hi * cos_q14 + k_lo * sin_q14 + (1 << 13)) >> 14, -32768, 32767)
                rotated = np.concatenate([k_lo_rot, k_hi_rot]).astype(np.int16)
                values[op.out] = rotated.reshape(values[op.out].shape)
            out_spec = tensors.get(op.out)
            if out_spec is not None:
                base_name = out_spec.metadata.get("storage_view_of")
                cache_kind = out_spec.metadata.get("cache_kind")
                token_index = out_spec.metadata.get("cache_token_index")
                if base_name and cache_kind in {"K", "V"} and token_index is not None:
                    base_spec = tensors[str(base_name)]
                    if str(base_name) not in values:
                        dtype = np.float32 if base_spec.dtype == DType.FLOAT32 else np.int16
                        values[str(base_name)] = np.zeros(base_spec.shape, dtype=dtype)
                    slot = np.asarray(values[op.out]).reshape(-1)
                    if cache_kind == "K":
                        values[str(base_name)][:, int(token_index)] = slot
                    else:
                        values[str(base_name)][int(token_index), :] = slot

    def _run_host_op(self, step: HostOp, values: dict[str, np.ndarray]) -> None:
        execute_host_op(step, values, golden=self.golden)

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
