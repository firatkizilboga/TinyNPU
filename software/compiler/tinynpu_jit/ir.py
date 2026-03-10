from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from tinynpu.isa import PrecisionMode


class DType(str, Enum):
    INT4 = "int4"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    FLOAT32 = "float32"


class TensorKind(str, Enum):
    INPUT = "input"
    CONSTANT = "constant"
    INTERMEDIATE = "intermediate"
    OUTPUT = "output"


class VerificationMode(str, Enum):
    OFF = "off"
    FINAL = "final"
    DEBUG = "debug"


@dataclass
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: DType
    kind: TensorKind
    data: np.ndarray | None = None
    is_final_output: bool = False
    verify_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone_without_data(self, name: str | None = None) -> "TensorSpec":
        return TensorSpec(
            name=name or self.name,
            shape=self.shape,
            dtype=self.dtype,
            kind=self.kind,
            data=None,
            is_final_output=self.is_final_output,
            verify_label=self.verify_label,
            metadata=dict(self.metadata),
        )


@dataclass
class MatMulOp:
    name: str
    lhs: str
    rhs: str
    out: str
    bias: str | None = None
    multiplier: int = 1
    shift: int = 0
    activation: str = "none"
    in_dtype: DType = DType.INT16
    out_dtype: DType = DType.INT16


@dataclass
class NpuSegment:
    name: str
    ops: list[MatMulOp]
    inputs: list[str]
    outputs: list[str]


@dataclass
class HostOp:
    name: str
    kind: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifyTensor:
    tensor_name: str
    label: str
    is_final_output: bool = False


PlanStep = NpuSegment | HostOp | VerifyTensor


@dataclass
class ExecutionPlan:
    tensors: dict[str, TensorSpec]
    steps: list[PlanStep]
    inputs: list[str]
    outputs: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_verification_step(self, tensor_name: str, label: str | None = None) -> None:
        tensor = self.tensors[tensor_name]
        self.steps.append(
            VerifyTensor(
                tensor_name=tensor_name,
                label=label or tensor.verify_label or tensor_name,
                is_final_output=tensor.is_final_output,
            )
        )


def normalize_shape(shape: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in shape)


_DTYPE_TO_PRECISION = {
    DType.INT4: PrecisionMode.INT4,
    DType.INT8: PrecisionMode.INT8,
    DType.INT16: PrecisionMode.INT16,
}


def to_precision_mode(dtype: DType) -> PrecisionMode:
    if dtype not in _DTYPE_TO_PRECISION:
        raise ValueError(f"TinyNPU precision does not support dtype {dtype}.")
    return _DTYPE_TO_PRECISION[dtype]


def numpy_dtype_for(dtype: DType) -> np.dtype:
    if dtype == DType.FLOAT32:
        return np.float32
    if dtype == DType.INT32:
        return np.int32
    return np.int16
