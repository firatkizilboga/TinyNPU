from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from .ir import DType


SemanticValueKind = Literal["input", "constant", "intermediate", "output"]


@dataclass(frozen=True)
class QuantizationSpec:
    scale: float
    zero_point: int
    dtype: DType


@dataclass
class SemanticValue:
    name: str
    shape: tuple[int, ...]
    dtype: DType
    kind: SemanticValueKind
    data: np.ndarray | None = None
    quantization: QuantizationSpec | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticOp:
    name: str
    inputs: list[str]
    outputs: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantizeOp(SemanticOp):
    scale: float = 1.0
    zero_point: int = 0
    dtype: DType = DType.INT16


@dataclass
class DequantizeOp(SemanticOp):
    scale: float = 1.0
    zero_point: int = 0
    output_encoding: str = "float32"


@dataclass
class ActivationOp(SemanticOp):
    kind: str = "relu"


@dataclass
class BinaryOp(SemanticOp):
    kind: str = "add"


@dataclass
class MeanOp(SemanticOp):
    dim: tuple[int, ...] = ()
    keepdim: bool = False


@dataclass
class MaxPool2dOp(SemanticOp):
    kernel_size: tuple[int, int] = (1, 1)
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)


@dataclass
class AvgPool2dOp(SemanticOp):
    kernel_size: tuple[int, int] = (1, 1)
    stride: tuple[int, int] = (1, 1)
    padding: tuple[int, int] = (0, 0)
    count_include_pad: bool = True


@dataclass
class AdaptiveAvgPool2dOp(SemanticOp):
    output_size: tuple[int, int] = (1, 1)


@dataclass
class ReshapeOp(SemanticOp):
    shape: tuple[int, ...] = ()


@dataclass
class LinearOp(SemanticOp):
    weight: np.ndarray | None = None
    bias: np.ndarray | None = None
    in_features: int = 1
    out_features: int = 1
    module_name: str = ""


@dataclass
class Conv2dOp(SemanticOp):
    weight: np.ndarray | None = None
    bias: np.ndarray | None = None
    stride: int = 1
    padding: int = 0
    kernel_size: int = 1
    in_channels: int = 1
    out_channels: int = 1
    module_name: str = ""


@dataclass
class CompilerReadyLinearOp(SemanticOp):
    weight_int: np.ndarray | None = None
    bias_int32: np.ndarray | None = None
    input_scale: float = 1.0
    output_scale: float = 1.0
    in_dtype: DType = DType.INT16
    out_dtype: DType = DType.INT16
    multiplier: int = 1
    shift: int = 0
    activation: str = "none"
    h_gelu_x_scale_shift: int = 7
    module_name: str = ""


@dataclass
class CompilerReadyConv2dOp(SemanticOp):
    weight_int: np.ndarray | None = None
    bias_int32: np.ndarray | None = None
    input_scale: float = 1.0
    output_scale: float = 1.0
    in_dtype: DType = DType.INT16
    out_dtype: DType = DType.INT16
    multiplier: int = 1
    shift: int = 0
    stride: int = 1
    padding: int = 0
    kernel_size: int = 1
    in_channels: int = 1
    out_channels: int = 1
    activation: str = "none"
    h_gelu_x_scale_shift: int = 7
    module_name: str = ""


@dataclass
class VerifyOp(SemanticOp):
    label: str = ""
    is_final_output: bool = False


SemanticOpType = (
    QuantizeOp
    | DequantizeOp
    | ActivationOp
    | BinaryOp
    | MeanOp
    | MaxPool2dOp
    | AvgPool2dOp
    | AdaptiveAvgPool2dOp
    | ReshapeOp
    | LinearOp
    | Conv2dOp
    | CompilerReadyLinearOp
    | CompilerReadyConv2dOp
    | VerifyOp
)


@dataclass
class SemanticGraph:
    values: dict[str, SemanticValue]
    ops: list[SemanticOpType]
    inputs: list[str]
    outputs: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)
