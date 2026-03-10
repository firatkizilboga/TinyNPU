from __future__ import annotations

import math

import numpy as np

from .artifact import CompiledArtifact, ExecutionResult
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
        for step in artifact.plan.steps:
            if isinstance(step, NpuSegment):
                self._run_npu_segment(step, values)
            elif isinstance(step, HostOp):
                self._run_host_op(step, values)
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

        outputs = {name: np.array(values[name], copy=True) for name in artifact.plan.outputs}
        trace_tensors = {name: np.array(value, copy=True) for name, value in values.items()}
        return ExecutionResult(tensors=outputs, verified=verified, trace_tensors=trace_tensors)

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
        if step.kind == "sigmoid":
            source = np.array(values[step.inputs[0]], dtype=np.float32)
            values[step.outputs[0]] = 1.0 / (1.0 + np.exp(-source))
            return
        if step.kind == "relu":
            values[step.outputs[0]] = np.maximum(values[step.inputs[0]], 0)
            return
        if step.kind == "alias":
            values[step.outputs[0]] = np.array(values[step.inputs[0]], copy=True)
            return
        if step.kind == "im2col":
            values[step.outputs[0]] = self.golden.im2col(
                values[step.inputs[0]],
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
