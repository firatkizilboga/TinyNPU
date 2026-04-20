from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .ir import DType, ExecutionPlan, HostOp, MatMulOp, NpuSegment, TensorKind, TensorSpec, VerifyTensor


@dataclass
class IRBuilder:
    """Small hand-authored plan builder for the existing low-level ExecutionPlan.

    This is intentionally narrow: it does not replace the current IR, it just
    gives scripts and future typed-IR lowering a stable surface for constructing
    plans without manually juggling dict/list assembly.
    """

    tensors: dict[str, TensorSpec] = field(default_factory=dict)
    steps: list[Any] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_tensor(self, spec: TensorSpec) -> TensorSpec:
        self.tensors[spec.name] = spec
        return spec

    def tensor(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: DType,
        kind: TensorKind,
        *,
        data: np.ndarray | None = None,
        is_final_output: bool = False,
        verify_label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TensorSpec:
        return self.add_tensor(
            TensorSpec(
                name=name,
                shape=shape,
                dtype=dtype,
                kind=kind,
                data=data,
                is_final_output=is_final_output,
                verify_label=verify_label,
                metadata=dict(metadata or {}),
            )
        )

    def constant(
        self,
        name: str,
        data: np.ndarray,
        dtype: DType,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TensorSpec:
        return self.tensor(name, tuple(int(dim) for dim in data.shape), dtype, TensorKind.CONSTANT, data=data, metadata=metadata)

    def intermediate(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: DType,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> TensorSpec:
        return self.tensor(name, shape, dtype, TensorKind.INTERMEDIATE, metadata=metadata)

    def output(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: DType,
        *,
        is_final_output: bool = False,
        verify_label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> TensorSpec:
        return self.tensor(
            name,
            shape,
            dtype,
            TensorKind.OUTPUT if is_final_output else TensorKind.INTERMEDIATE,
            is_final_output=is_final_output,
            verify_label=verify_label,
            metadata=metadata,
        )

    def add_step(self, step: Any) -> Any:
        self.steps.append(step)
        return step

    def host(
        self,
        name: str,
        kind: str,
        *,
        inputs: list[str],
        outputs: list[str],
        attrs: dict[str, Any] | None = None,
    ) -> HostOp:
        return self.add_step(HostOp(name=name, kind=kind, inputs=inputs, outputs=outputs, attrs=dict(attrs or {})))

    def matmul(
        self,
        name: str,
        lhs: str,
        rhs: str,
        out: str,
        **kwargs: Any,
    ) -> MatMulOp:
        return MatMulOp(name=name, lhs=lhs, rhs=rhs, out=out, **kwargs)

    def segment(
        self,
        name: str,
        *,
        ops: list[MatMulOp],
        inputs: list[str],
        outputs: list[str],
    ) -> NpuSegment:
        return self.add_step(NpuSegment(name=name, ops=ops, inputs=inputs, outputs=outputs))

    def add_input(self, name: str) -> None:
        if name not in self.inputs:
            self.inputs.append(name)

    def add_output(self, name: str) -> None:
        if name not in self.outputs:
            self.outputs.append(name)

    def add_verification(self, tensor_name: str, label: str | None = None, *, is_final_output: bool | None = None) -> None:
        tensor = self.tensors[tensor_name]
        self.steps.append(
            VerifyTensor(
                tensor_name=tensor_name,
                label=label or tensor.verify_label or tensor_name,
                is_final_output=tensor.is_final_output if is_final_output is None else bool(is_final_output),
                float_atol=float(tensor.metadata.get("verify_atol", 1.0e-3)),
            )
        )

    def verify(self, tensor_name: str, label: str | None = None, *, is_final_output: bool | None = None) -> None:
        self.add_verification(tensor_name, label, is_final_output=is_final_output)

    def finalize(
        self,
        *,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ExecutionPlan:
        if inputs is not None:
            self.inputs = list(inputs)
        if outputs is not None:
            self.outputs = list(outputs)
        if metadata:
            merged = dict(self.metadata)
            merged.update(metadata)
            self.metadata = merged
        return ExecutionPlan(
            tensors=dict(self.tensors),
            steps=list(self.steps),
            inputs=list(self.inputs),
            outputs=list(self.outputs),
            metadata=dict(self.metadata),
        )
