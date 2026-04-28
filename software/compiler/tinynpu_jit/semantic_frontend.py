from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
import operator

import numpy as np

from .ir import DType
from .semantic_capabilities import analyze_semantic_capabilities
from .semantic_ir import (
    AdaptiveAvgPool2dOp,
    ActivationOp,
    AvgPool2dOp,
    BinaryOp,
    Conv2dOp,
    CompilerReadyConv2dOp,
    CompilerReadyLinearOp,
    DequantizeOp,
    LinearOp,
    MaxPool2dOp,
    MeanOp,
    QuantizationSpec,
    QuantizeOp,
    ReshapeOp,
    SemanticGraph,
    SemanticValue,
    VerifyOp,
)


@dataclass
class SemanticBuilderContext:
    values: dict[str, SemanticValue] = field(default_factory=dict)
    ops: list[object] = field(default_factory=list)
    inputs: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)

    def add_value(self, value: SemanticValue) -> None:
        self.values[value.name] = value

    def add_op(self, op: object) -> None:
        self.ops.append(op)


class SemanticFrontendRegistry:
    def __init__(self) -> None:
        self.module_handlers: list[tuple[type[Any], Callable[[SemanticBuilderContext, Any, Any, dict[str, Any]], None]]] = []
        self.function_handlers: dict[str, Callable[[SemanticBuilderContext, Any, dict[str, Any]], None]] = {}
        self.method_handlers: dict[str, Callable[[SemanticBuilderContext, Any, dict[str, Any]], None]] = {}

    def register_module(self, module_type: type[Any]):
        def decorator(fn: Callable[[SemanticBuilderContext, Any, Any, dict[str, Any]], None]):
            self.module_handlers.append((module_type, fn))
            return fn

        return decorator

    def register_function(self, name: str):
        def decorator(fn: Callable[[SemanticBuilderContext, Any, dict[str, Any]], None]):
            self.function_handlers[name] = fn
            return fn

        return decorator

    def register_method(self, name: str):
        def decorator(fn: Callable[[SemanticBuilderContext, Any, dict[str, Any]], None]):
            self.method_handlers[name] = fn
            return fn

        return decorator

    def find_module_handler(self, module: Any):
        for module_type, handler in self.module_handlers:
            if isinstance(module, module_type) or type(module).__name__ == module_type.__name__:
                return handler
        return None


def _normalize_shape(shape: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in tuple(shape))


def _parse_dtype(name: str) -> DType:
    mapping = {
        "int4": DType.INT4,
        "int8": DType.INT8,
        "int16": DType.INT16,
        "int32": DType.INT32,
        "float32": DType.FLOAT32,
    }
    key = str(name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported semantic dtype {name!r}.")
    return mapping[key]


def _infer_numpy_dtype(value: np.ndarray) -> DType:
    if np.issubdtype(value.dtype, np.floating):
        return DType.FLOAT32
    if value.dtype == np.int8:
        return DType.INT8
    if value.dtype == np.int16:
        return DType.INT16
    if value.dtype == np.int32:
        return DType.INT32
    if np.issubdtype(value.dtype, np.integer):
        min_val = int(value.min(initial=0))
        max_val = int(value.max(initial=0))
        if -128 <= min_val and max_val <= 127:
            return DType.INT8
        if -32768 <= min_val and max_val <= 32767:
            return DType.INT16
        return DType.INT32
    raise ValueError(f"Unsupported numpy dtype {value.dtype}.")


def _infer_linear_output_shape(input_shape: tuple[int, ...], out_features: int) -> tuple[int, ...]:
    if not input_shape:
        raise NotImplementedError("Semantic frontend linear expects rank >= 1.")
    return (*input_shape[:-1], out_features)


def _infer_conv_output_shape(
    input_shape: tuple[int, ...],
    *,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    in_channels: int,
    allow_batch: bool = False,
) -> tuple[int, ...]:
    original_rank = len(input_shape)
    if original_rank == 4:
        batch = int(input_shape[0])
        if batch != 1 and not allow_batch:
            raise NotImplementedError("Semantic frontend currently supports batch size 1 only for NPU compiler-ready conv.")
        input_shape = input_shape[1:]
    else:
        batch = None
    if len(input_shape) != 3:
        raise NotImplementedError(f"Semantic conv expects 3D or 4D inputs, got {input_shape}.")
    if input_shape[0] == in_channels:
        h, w = input_shape[1], input_shape[2]
        out_h = ((h + 2 * padding - kernel_size) // stride) + 1
        out_w = ((w + 2 * padding - kernel_size) // stride) + 1
        chw = (out_channels, out_h, out_w)
        return (batch, *chw) if batch is not None else chw
    if input_shape[-1] == in_channels:
        h, w = input_shape[0], input_shape[1]
        out_h = ((h + 2 * padding - kernel_size) // stride) + 1
        out_w = ((w + 2 * padding - kernel_size) // stride) + 1
        hwc = (out_h, out_w, out_channels)
        return (batch, *hwc) if batch is not None else hwc
    raise NotImplementedError(
        f"Could not infer semantic conv layout from shape {input_shape} with in_channels={in_channels}."
    )


def _activation_output_dtype(input_dtype: DType) -> DType:
    return input_dtype


def _resolve_dims(raw_dim: Any, rank: int) -> tuple[int, ...]:
    if raw_dim is None:
        return tuple(range(rank))
    if isinstance(raw_dim, int):
        dims = (raw_dim,)
    else:
        dims = tuple(int(dim) for dim in raw_dim)
    normalized: list[int] = []
    for dim in dims:
        normalized_dim = dim if dim >= 0 else rank + dim
        if normalized_dim < 0 or normalized_dim >= rank:
            raise NotImplementedError(f"Semantic frontend mean dim {dim} out of range for rank {rank}.")
        normalized.append(normalized_dim)
    if len(set(normalized)) != len(normalized):
        raise NotImplementedError(f"Semantic frontend mean dims must be unique, got {dims}.")
    return tuple(normalized)


def _infer_mean_output_shape(input_shape: tuple[int, ...], dims: tuple[int, ...], *, keepdim: bool) -> tuple[int, ...]:
    if keepdim:
        return tuple(1 if idx in dims else dim for idx, dim in enumerate(input_shape))
    return tuple(dim for idx, dim in enumerate(input_shape) if idx not in dims)


def _infer_broadcast_shape(lhs_shape: tuple[int, ...], rhs_shape: tuple[int, ...]) -> tuple[int, ...]:
    max_rank = max(len(lhs_shape), len(rhs_shape))
    lhs_padded = (1,) * (max_rank - len(lhs_shape)) + lhs_shape
    rhs_padded = (1,) * (max_rank - len(rhs_shape)) + rhs_shape
    result: list[int] = []
    for lhs_dim, rhs_dim in zip(lhs_padded, rhs_padded):
        if lhs_dim == rhs_dim:
            result.append(lhs_dim)
        elif lhs_dim == 1:
            result.append(rhs_dim)
        elif rhs_dim == 1:
            result.append(lhs_dim)
        else:
            raise NotImplementedError(f"Semantic frontend cannot broadcast shapes {lhs_shape} and {rhs_shape}.")
    return tuple(result)


def _normalize_pair(value: Any, *, default: tuple[int, int] | None = None) -> tuple[int, int]:
    if value is None:
        if default is None:
            raise NotImplementedError("Semantic frontend expected an explicit 2D pair value.")
        return default
    if isinstance(value, int):
        return (int(value), int(value))
    pair = tuple(int(v) for v in value)
    if len(pair) != 2:
        raise NotImplementedError(f"Semantic frontend expected a 2D pair, got {value!r}.")
    return pair


def _infer_pool2d_output_shape(
    input_shape: tuple[int, ...],
    *,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> tuple[int, ...]:
    if len(input_shape) == 3:
        channels, in_h, in_w = input_shape
        batch = None
    elif len(input_shape) == 4:
        batch, channels, in_h, in_w = input_shape
    else:
        raise NotImplementedError(f"Semantic pooling expects rank-3 or rank-4 NCHW input, got {input_shape}.")
    out_h = ((in_h + (2 * padding[0]) - kernel_size[0]) // stride[0]) + 1
    out_w = ((in_w + (2 * padding[1]) - kernel_size[1]) // stride[1]) + 1
    if out_h <= 0 or out_w <= 0:
        raise NotImplementedError(
            f"Semantic pooling produced non-positive output shape for input={input_shape}, kernel={kernel_size}, stride={stride}, padding={padding}."
        )
    base = (channels, out_h, out_w)
    return (batch, *base) if len(input_shape) == 4 else base


def _infer_adaptive_avg_pool2d_output_shape(input_shape: tuple[int, ...], *, output_size: tuple[int, int]) -> tuple[int, ...]:
    if len(input_shape) == 3:
        channels = input_shape[0]
        base = (channels, output_size[0], output_size[1])
        return base
    if len(input_shape) == 4:
        batch, channels = input_shape[0], input_shape[1]
        return (batch, channels, output_size[0], output_size[1])
    raise NotImplementedError(f"Semantic adaptive avg pool expects rank-3 or rank-4 NCHW input, got {input_shape}.")


def _build_registry() -> SemanticFrontendRegistry:
    registry = SemanticFrontendRegistry()
    try:
        import torch.nn as nn
    except Exception:  # pragma: no cover - exercised in compile_module error path
        nn = None

    try:
        from software.compiler.tinynpu_quant import CompilerDequantize, CompilerQuantize, CompilerReadyConv2d, CompilerReadyLinear
    except Exception:  # pragma: no cover - exercised in compile_module error path
        CompilerQuantize = CompilerDequantize = CompilerReadyLinear = CompilerReadyConv2d = None

    if nn is not None:
        @registry.register_module(nn.Linear)
        def _handle_linear(ctx: SemanticBuilderContext, node: Any, module: Any, env: dict[str, Any]) -> None:
            source = node.args[0].name
            input_value = ctx.values[source]
            if input_value.shape[-1] != int(module.in_features):
                raise NotImplementedError(
                    f"Semantic frontend linear expects input last dimension {module.in_features}, got {input_value.shape}."
                )
            ctx.add_op(
                LinearOp(
                    name=node.name,
                    inputs=[source],
                    outputs=[node.name],
                    weight=np.array(module.weight.detach().cpu().numpy(), dtype=np.float32, copy=True),
                    bias=None if module.bias is None else np.array(module.bias.detach().cpu().numpy(), dtype=np.float32, copy=True),
                    in_features=int(module.in_features),
                    out_features=int(module.out_features),
                    module_name=str(node.target),
                )
            )
            ctx.add_value(
                SemanticValue(
                    name=node.name,
                    shape=_infer_linear_output_shape(input_value.shape, int(module.out_features)),
                    dtype=DType.FLOAT32,
                    kind="intermediate",
                )
            )

        @registry.register_module(nn.Conv2d)
        def _handle_conv2d(ctx: SemanticBuilderContext, node: Any, module: Any, env: dict[str, Any]) -> None:
            source = node.args[0].name
            input_value = ctx.values[source]
            if module.groups != 1:
                raise NotImplementedError("Semantic frontend currently supports Conv2d with groups=1 only.")
            if isinstance(module.dilation, tuple):
                if len(module.dilation) != 2 or module.dilation[0] != 1 or module.dilation[1] != 1:
                    raise NotImplementedError("Semantic frontend currently supports Conv2d with dilation=1 only.")
            elif int(module.dilation) != 1:
                raise NotImplementedError("Semantic frontend currently supports Conv2d with dilation=1 only.")
            if str(module.padding_mode) != "zeros":
                raise NotImplementedError("Semantic frontend currently supports Conv2d with zero padding mode only.")
            if isinstance(module.kernel_size, tuple):
                if len(module.kernel_size) != 2 or module.kernel_size[0] != module.kernel_size[1]:
                    raise NotImplementedError("Semantic frontend currently supports square Conv2d kernels only.")
                kernel_size = int(module.kernel_size[0])
            else:
                kernel_size = int(module.kernel_size)
            if isinstance(module.stride, tuple):
                if len(module.stride) != 2 or module.stride[0] != module.stride[1]:
                    raise NotImplementedError("Semantic frontend currently supports symmetric Conv2d stride only.")
                stride = int(module.stride[0])
            else:
                stride = int(module.stride)
            if isinstance(module.padding, tuple):
                if len(module.padding) != 2 or module.padding[0] != module.padding[1]:
                    raise NotImplementedError("Semantic frontend currently supports symmetric Conv2d padding only.")
                padding = int(module.padding[0])
            else:
                padding = int(module.padding)
            ctx.add_op(
                Conv2dOp(
                    name=node.name,
                    inputs=[source],
                    outputs=[node.name],
                    weight=np.array(module.weight.detach().cpu().numpy(), dtype=np.float32, copy=True),
                    bias=None if module.bias is None else np.array(module.bias.detach().cpu().numpy(), dtype=np.float32, copy=True),
                    stride=stride,
                    padding=padding,
                    kernel_size=kernel_size,
                    in_channels=int(module.in_channels),
                    out_channels=int(module.out_channels),
                    module_name=str(node.target),
                )
            )
            ctx.add_value(
                SemanticValue(
                    name=node.name,
                    shape=_infer_conv_output_shape(
                        input_value.shape,
                        out_channels=int(module.out_channels),
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        in_channels=int(module.in_channels),
                        allow_batch=True,
                    ),
                    dtype=DType.FLOAT32,
                    kind="intermediate",
                )
            )

        @registry.register_module(nn.MaxPool2d)
        def _handle_maxpool2d(ctx: SemanticBuilderContext, node: Any, module: Any, env: dict[str, Any]) -> None:
            source = node.args[0].name
            input_value = ctx.values[source]
            if isinstance(module.dilation, tuple):
                if tuple(int(v) for v in module.dilation) != (1, 1):
                    raise NotImplementedError("Semantic frontend currently supports MaxPool2d with dilation=1 only.")
            elif int(module.dilation) != 1:
                raise NotImplementedError("Semantic frontend currently supports MaxPool2d with dilation=1 only.")
            if bool(module.ceil_mode):
                raise NotImplementedError("Semantic frontend currently supports MaxPool2d with ceil_mode=False only.")
            kernel_size = _normalize_pair(module.kernel_size)
            stride = _normalize_pair(module.stride, default=kernel_size)
            padding = _normalize_pair(module.padding, default=(0, 0))
            ctx.add_op(
                MaxPool2dOp(
                    name=node.name,
                    inputs=[source],
                    outputs=[node.name],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            ctx.add_value(
                SemanticValue(
                    name=node.name,
                    shape=_infer_pool2d_output_shape(input_value.shape, kernel_size=kernel_size, stride=stride, padding=padding),
                    dtype=input_value.dtype,
                    kind="intermediate",
                )
            )

        @registry.register_module(nn.AvgPool2d)
        def _handle_avgpool2d(ctx: SemanticBuilderContext, node: Any, module: Any, env: dict[str, Any]) -> None:
            source = node.args[0].name
            input_value = ctx.values[source]
            if bool(module.ceil_mode):
                raise NotImplementedError("Semantic frontend currently supports AvgPool2d with ceil_mode=False only.")
            if module.divisor_override is not None:
                raise NotImplementedError("Semantic frontend currently supports AvgPool2d with divisor_override=None only.")
            kernel_size = _normalize_pair(module.kernel_size)
            stride = _normalize_pair(module.stride, default=kernel_size)
            padding = _normalize_pair(module.padding, default=(0, 0))
            ctx.add_op(
                AvgPool2dOp(
                    name=node.name,
                    inputs=[source],
                    outputs=[node.name],
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    count_include_pad=bool(module.count_include_pad),
                )
            )
            ctx.add_value(
                SemanticValue(
                    name=node.name,
                    shape=_infer_pool2d_output_shape(input_value.shape, kernel_size=kernel_size, stride=stride, padding=padding),
                    dtype=input_value.dtype,
                    kind="intermediate",
                )
            )

        @registry.register_module(nn.AdaptiveAvgPool2d)
        def _handle_adaptive_avgpool2d(ctx: SemanticBuilderContext, node: Any, module: Any, env: dict[str, Any]) -> None:
            source = node.args[0].name
            input_value = ctx.values[source]
            output_size = _normalize_pair(module.output_size)
            ctx.add_op(
                AdaptiveAvgPool2dOp(
                    name=node.name,
                    inputs=[source],
                    outputs=[node.name],
                    output_size=output_size,
                )
            )
            ctx.add_value(
                SemanticValue(
                    name=node.name,
                    shape=_infer_adaptive_avg_pool2d_output_shape(input_value.shape, output_size=output_size),
                    dtype=input_value.dtype,
                    kind="intermediate",
                )
            )

    if CompilerQuantize is not None:
        @registry.register_module(CompilerQuantize)
        def _handle_compiler_quantize(ctx: SemanticBuilderContext, node: Any, module: Any, env: dict[str, Any]) -> None:
            source = node.args[0].name
            input_value = ctx.values[source]
            dtype = _parse_dtype(module.dtype)
            quant = QuantizationSpec(scale=float(module.scale), zero_point=int(module.zero_point), dtype=dtype)
            ctx.add_op(
                QuantizeOp(
                    name=node.name,
                    inputs=[source],
                    outputs=[node.name],
                    scale=quant.scale,
                    zero_point=quant.zero_point,
                    dtype=quant.dtype,
                )
            )
            ctx.add_value(
                SemanticValue(
                    name=node.name,
                    shape=input_value.shape,
                    dtype=dtype,
                    kind="intermediate",
                    quantization=quant,
                )
            )

    if CompilerDequantize is not None:
        @registry.register_module(CompilerDequantize)
        def _handle_compiler_dequantize(ctx: SemanticBuilderContext, node: Any, module: Any, env: dict[str, Any]) -> None:
            source = node.args[0].name
            input_value = ctx.values[source]
            output_encoding = str(getattr(module, "output_encoding", "float32"))
            ctx.add_op(
                DequantizeOp(
                    name=node.name,
                    inputs=[source],
                    outputs=[node.name],
                    scale=float(module.scale),
                    zero_point=int(module.zero_point),
                    output_encoding=output_encoding,
                )
            )
            metadata = {"value_encoding": "fp16_bits"} if output_encoding == "fp16_bits" else {}
            ctx.add_value(
                SemanticValue(
                    name=node.name,
                    shape=input_value.shape,
                    dtype=DType.INT16 if output_encoding == "fp16_bits" else DType.FLOAT32,
                    kind="intermediate",
                    metadata=metadata,
                )
            )

    if CompilerReadyLinear is not None:
        @registry.register_module(CompilerReadyLinear)
        def _handle_compiler_ready_linear(ctx: SemanticBuilderContext, node: Any, module: Any, env: dict[str, Any]) -> None:
            source = node.args[0].name
            input_value = ctx.values[source]
            out_dtype = _parse_dtype(module.out_dtype)
            ctx.add_op(
                CompilerReadyLinearOp(
                    name=node.name,
                    inputs=[source],
                    outputs=[node.name],
                    weight_int=np.array(module.weight_int.detach().cpu().numpy(), dtype=np.int16, copy=True),
                    bias_int32=None if module.bias_int32 is None else np.array(module.bias_int32.detach().cpu().numpy(), dtype=np.int32, copy=True).reshape(1, -1),
                    input_scale=float(module.input_scale),
                    output_scale=float(module.output_scale),
                    in_dtype=_parse_dtype(module.in_dtype),
                    out_dtype=out_dtype,
                    multiplier=int(module.multiplier),
                    shift=int(module.shift),
                    h_gelu_x_scale_shift=int(getattr(module, "h_gelu_x_scale_shift", 7)),
                    module_name=str(node.target),
                )
            )
            ctx.add_value(
                SemanticValue(
                    name=node.name,
                    shape=_infer_linear_output_shape(input_value.shape, int(module.weight_int.shape[0])),
                    dtype=out_dtype,
                    kind="intermediate",
                    quantization=QuantizationSpec(scale=float(module.output_scale), zero_point=0, dtype=out_dtype),
                )
            )

    if CompilerReadyConv2d is not None:
        @registry.register_module(CompilerReadyConv2d)
        def _handle_compiler_ready_conv2d(ctx: SemanticBuilderContext, node: Any, module: Any, env: dict[str, Any]) -> None:
            source = node.args[0].name
            input_value = ctx.values[source]
            out_dtype = _parse_dtype(module.out_dtype)
            ctx.add_op(
                CompilerReadyConv2dOp(
                    name=node.name,
                    inputs=[source],
                    outputs=[node.name],
                    weight_int=np.array(module.weight_int.detach().cpu().numpy(), dtype=np.int16, copy=True),
                    bias_int32=None if module.bias_int32 is None else np.array(module.bias_int32.detach().cpu().numpy(), dtype=np.int32, copy=True).reshape(1, -1),
                    input_scale=float(module.input_scale),
                    output_scale=float(module.output_scale),
                    in_dtype=_parse_dtype(module.in_dtype),
                    out_dtype=out_dtype,
                    multiplier=int(module.multiplier),
                    shift=int(module.shift),
                    stride=int(module.stride),
                    padding=int(module.padding),
                    kernel_size=int(module.kernel_size),
                    in_channels=int(module.in_channels),
                    out_channels=int(module.out_channels),
                    h_gelu_x_scale_shift=int(getattr(module, "h_gelu_x_scale_shift", 7)),
                    module_name=str(node.target),
                )
            )
            ctx.add_value(
                SemanticValue(
                    name=node.name,
                    shape=_infer_conv_output_shape(
                        input_value.shape,
                        out_channels=int(module.out_channels),
                        kernel_size=int(module.kernel_size),
                        stride=int(module.stride),
                        padding=int(module.padding),
                        in_channels=int(module.in_channels),
                    ),
                    dtype=out_dtype,
                    kind="intermediate",
                    quantization=QuantizationSpec(scale=float(module.output_scale), zero_point=0, dtype=out_dtype),
                )
            )

    def _handle_activation(ctx: SemanticBuilderContext, node: Any, kind: str) -> None:
        source = node.args[0].name
        input_value = ctx.values[source]
        ctx.add_op(ActivationOp(name=node.name, inputs=[source], outputs=[node.name], kind=kind))
        ctx.add_value(
            SemanticValue(
                name=node.name,
                shape=input_value.shape,
                dtype=_activation_output_dtype(input_value.dtype),
                kind="intermediate",
                quantization=input_value.quantization,
            )
        )

    def _handle_binary(ctx: SemanticBuilderContext, node: Any, kind: str) -> None:
        lhs = node.args[0].name
        rhs = node.args[1].name
        lhs_value = ctx.values[lhs]
        rhs_value = ctx.values[rhs]
        ctx.add_op(BinaryOp(name=node.name, inputs=[lhs, rhs], outputs=[node.name], kind=kind))
        ctx.add_value(
            SemanticValue(
                name=node.name,
                shape=_infer_broadcast_shape(lhs_value.shape, rhs_value.shape),
                dtype=lhs_value.dtype,
                kind="intermediate",
            )
        )

    def _handle_mean(ctx: SemanticBuilderContext, node: Any, dim: Any, keepdim: bool) -> None:
        source = node.args[0].name
        input_value = ctx.values[source]
        dims = _resolve_dims(dim, len(input_value.shape))
        ctx.add_op(MeanOp(name=node.name, inputs=[source], outputs=[node.name], dim=dims, keepdim=bool(keepdim)))
        ctx.add_value(
            SemanticValue(
                name=node.name,
                shape=_infer_mean_output_shape(input_value.shape, dims, keepdim=bool(keepdim)),
                dtype=input_value.dtype,
                kind="intermediate",
            )
        )

    @registry.register_function("relu")
    def _handle_relu_function(ctx: SemanticBuilderContext, node: Any, env: dict[str, Any]) -> None:
        _handle_activation(ctx, node, "relu")

    @registry.register_function("sigmoid")
    def _handle_sigmoid_function(ctx: SemanticBuilderContext, node: Any, env: dict[str, Any]) -> None:
        _handle_activation(ctx, node, "sigmoid")

    @registry.register_function("gelu")
    def _handle_gelu_function(ctx: SemanticBuilderContext, node: Any, env: dict[str, Any]) -> None:
        _handle_activation(ctx, node, "gelu")

    @registry.register_function("add")
    def _handle_add_function(ctx: SemanticBuilderContext, node: Any, env: dict[str, Any]) -> None:
        _handle_binary(ctx, node, "add")

    @registry.register_function("mul")
    def _handle_mul_function(ctx: SemanticBuilderContext, node: Any, env: dict[str, Any]) -> None:
        _handle_binary(ctx, node, "mul")

    @registry.register_function("max_pool2d")
    def _handle_max_pool2d_function(ctx: SemanticBuilderContext, node: Any, env: dict[str, Any]) -> None:
        source = node.args[0].name
        input_value = ctx.values[source]
        kernel_size = _normalize_pair(node.args[1] if len(node.args) > 1 else node.kwargs.get("kernel_size"))
        stride = _normalize_pair(node.args[2] if len(node.args) > 2 else node.kwargs.get("stride"), default=kernel_size)
        padding = _normalize_pair(node.args[3] if len(node.args) > 3 else node.kwargs.get("padding"), default=(0, 0))
        dilation = node.args[4] if len(node.args) > 4 else node.kwargs.get("dilation", 1)
        ceil_mode = bool(node.args[5] if len(node.args) > 5 else node.kwargs.get("ceil_mode", False))
        if _normalize_pair(dilation) != (1, 1):
            raise NotImplementedError("Semantic frontend currently supports max_pool2d with dilation=1 only.")
        if ceil_mode:
            raise NotImplementedError("Semantic frontend currently supports max_pool2d with ceil_mode=False only.")
        ctx.add_op(MaxPool2dOp(name=node.name, inputs=[source], outputs=[node.name], kernel_size=kernel_size, stride=stride, padding=padding))
        ctx.add_value(
            SemanticValue(
                name=node.name,
                shape=_infer_pool2d_output_shape(input_value.shape, kernel_size=kernel_size, stride=stride, padding=padding),
                dtype=input_value.dtype,
                kind="intermediate",
            )
        )

    @registry.register_function("avg_pool2d")
    def _handle_avg_pool2d_function(ctx: SemanticBuilderContext, node: Any, env: dict[str, Any]) -> None:
        source = node.args[0].name
        input_value = ctx.values[source]
        kernel_size = _normalize_pair(node.args[1] if len(node.args) > 1 else node.kwargs.get("kernel_size"))
        stride = _normalize_pair(node.args[2] if len(node.args) > 2 else node.kwargs.get("stride"), default=kernel_size)
        padding = _normalize_pair(node.args[3] if len(node.args) > 3 else node.kwargs.get("padding"), default=(0, 0))
        ceil_mode = bool(node.args[4] if len(node.args) > 4 else node.kwargs.get("ceil_mode", False))
        count_include_pad = bool(node.args[5] if len(node.args) > 5 else node.kwargs.get("count_include_pad", True))
        divisor_override = node.args[6] if len(node.args) > 6 else node.kwargs.get("divisor_override", None)
        if ceil_mode:
            raise NotImplementedError("Semantic frontend currently supports avg_pool2d with ceil_mode=False only.")
        if divisor_override is not None:
            raise NotImplementedError("Semantic frontend currently supports avg_pool2d with divisor_override=None only.")
        ctx.add_op(
            AvgPool2dOp(
                name=node.name,
                inputs=[source],
                outputs=[node.name],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                count_include_pad=count_include_pad,
            )
        )
        ctx.add_value(
            SemanticValue(
                name=node.name,
                shape=_infer_pool2d_output_shape(input_value.shape, kernel_size=kernel_size, stride=stride, padding=padding),
                dtype=input_value.dtype,
                kind="intermediate",
            )
        )

    @registry.register_function("adaptive_avg_pool2d")
    def _handle_adaptive_avg_pool2d_function(ctx: SemanticBuilderContext, node: Any, env: dict[str, Any]) -> None:
        source = node.args[0].name
        input_value = ctx.values[source]
        output_size = _normalize_pair(node.args[1] if len(node.args) > 1 else node.kwargs.get("output_size"))
        ctx.add_op(AdaptiveAvgPool2dOp(name=node.name, inputs=[source], outputs=[node.name], output_size=output_size))
        ctx.add_value(
            SemanticValue(
                name=node.name,
                shape=_infer_adaptive_avg_pool2d_output_shape(input_value.shape, output_size=output_size),
                dtype=input_value.dtype,
                kind="intermediate",
            )
        )

    @registry.register_method("relu")
    def _handle_relu_method(ctx: SemanticBuilderContext, node: Any, env: dict[str, Any]) -> None:
        _handle_activation(ctx, node, "relu")

    @registry.register_method("sigmoid")
    def _handle_sigmoid_method(ctx: SemanticBuilderContext, node: Any, env: dict[str, Any]) -> None:
        _handle_activation(ctx, node, "sigmoid")

    @registry.register_method("gelu")
    def _handle_gelu_method(ctx: SemanticBuilderContext, node: Any, env: dict[str, Any]) -> None:
        _handle_activation(ctx, node, "gelu")

    @registry.register_method("reshape")
    def _handle_reshape_method(ctx: SemanticBuilderContext, node: Any, env: dict[str, Any]) -> None:
        source = node.args[0].name
        shape = tuple(int(dim) for dim in node.args[1:])
        input_value = ctx.values[source]
        ctx.add_op(ReshapeOp(name=node.name, inputs=[source], outputs=[node.name], shape=shape))
        ctx.add_value(
            SemanticValue(
                name=node.name,
                shape=shape,
                dtype=input_value.dtype,
                kind="intermediate",
                quantization=input_value.quantization,
            )
        )

    @registry.register_method("flatten")
    def _handle_flatten_method(ctx: SemanticBuilderContext, node: Any, env: dict[str, Any]) -> None:
        source = node.args[0].name
        input_value = ctx.values[source]
        start_dim = int(node.args[1]) if len(node.args) > 1 else 0
        end_dim = int(node.args[2]) if len(node.args) > 2 else -1
        rank = len(input_value.shape)
        start_dim = start_dim if start_dim >= 0 else rank + start_dim
        end_dim = end_dim if end_dim >= 0 else rank + end_dim
        if start_dim < 0 or end_dim >= rank or start_dim > end_dim:
            raise NotImplementedError(f"Semantic frontend flatten dims ({start_dim}, {end_dim}) invalid for shape {input_value.shape}.")
        flat_dim = int(np.prod(input_value.shape[start_dim : end_dim + 1], dtype=np.int64))
        out_shape = (*input_value.shape[:start_dim], flat_dim, *input_value.shape[end_dim + 1 :])
        ctx.add_op(ReshapeOp(name=node.name, inputs=[source], outputs=[node.name], shape=out_shape))
        ctx.add_value(
            SemanticValue(
                name=node.name,
                shape=out_shape,
                dtype=input_value.dtype,
                kind="intermediate",
                quantization=input_value.quantization,
            )
        )

    @registry.register_function("flatten")
    def _handle_flatten_function(ctx: SemanticBuilderContext, node: Any, env: dict[str, Any]) -> None:
        source = node.args[0].name
        input_value = ctx.values[source]
        start_dim = int(node.args[1]) if len(node.args) > 1 else 0
        end_dim = int(node.args[2]) if len(node.args) > 2 else -1
        rank = len(input_value.shape)
        start_dim = start_dim if start_dim >= 0 else rank + start_dim
        end_dim = end_dim if end_dim >= 0 else rank + end_dim
        if start_dim < 0 or end_dim >= rank or start_dim > end_dim:
            raise NotImplementedError(f"Semantic frontend flatten dims ({start_dim}, {end_dim}) invalid for shape {input_value.shape}.")
        flat_dim = int(np.prod(input_value.shape[start_dim : end_dim + 1], dtype=np.int64))
        out_shape = (*input_value.shape[:start_dim], flat_dim, *input_value.shape[end_dim + 1 :])
        ctx.add_op(ReshapeOp(name=node.name, inputs=[source], outputs=[node.name], shape=out_shape))
        ctx.add_value(
            SemanticValue(
                name=node.name,
                shape=out_shape,
                dtype=input_value.dtype,
                kind="intermediate",
                quantization=input_value.quantization,
            )
        )

    @registry.register_method("mean")
    def _handle_mean_method(ctx: SemanticBuilderContext, node: Any, env: dict[str, Any]) -> None:
        dim = node.kwargs.get("dim") if "dim" in node.kwargs else (node.args[1] if len(node.args) > 1 else None)
        keepdim = bool(node.kwargs.get("keepdim", node.args[2] if len(node.args) > 2 else False))
        _handle_mean(ctx, node, dim, keepdim)

    if nn is not None:
        @registry.register_module(nn.Flatten)
        def _handle_flatten_module(ctx: SemanticBuilderContext, node: Any, module: Any, env: dict[str, Any]) -> None:
            source = node.args[0].name
            input_value = ctx.values[source]
            rank = len(input_value.shape)
            start_dim = int(module.start_dim)
            end_dim = int(module.end_dim)
            start_dim = start_dim if start_dim >= 0 else rank + start_dim
            end_dim = end_dim if end_dim >= 0 else rank + end_dim
            if start_dim < 0 or end_dim >= rank or start_dim > end_dim:
                raise NotImplementedError(f"Semantic frontend Flatten dims ({start_dim}, {end_dim}) invalid for shape {input_value.shape}.")
            flat_dim = int(np.prod(input_value.shape[start_dim : end_dim + 1], dtype=np.int64))
            out_shape = (*input_value.shape[:start_dim], flat_dim, *input_value.shape[end_dim + 1 :])
            ctx.add_op(ReshapeOp(name=node.name, inputs=[source], outputs=[node.name], shape=out_shape))
            ctx.add_value(
                SemanticValue(
                    name=node.name,
                    shape=out_shape,
                    dtype=input_value.dtype,
                    kind="intermediate",
                    quantization=input_value.quantization,
                )
            )

    return registry


_REGISTRY = _build_registry()


def build_semantic_graph(graph_module: Any, example_inputs: tuple[Any, ...]) -> SemanticGraph:
    ctx = SemanticBuilderContext()
    modules = dict(graph_module.named_modules())

    example_iter = iter(example_inputs)
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            raw = next(example_iter)
            if hasattr(raw, "detach"):
                raw = raw.detach()
            if hasattr(raw, "cpu"):
                raw = raw.cpu()
            if hasattr(raw, "numpy"):
                raw = raw.numpy()
            value = np.array(raw)
            ctx.add_value(
                SemanticValue(
                    name=node.name,
                    shape=_normalize_shape(value.shape),
                    dtype=_infer_numpy_dtype(value),
                    kind="input",
                )
            )
            ctx.inputs.append(node.name)
            continue

        if node.op == "get_attr":
            target = graph_module
            for part in str(node.target).split("."):
                target = getattr(target, part)
            raw = target
            if hasattr(raw, "detach"):
                raw = raw.detach()
            if hasattr(raw, "cpu"):
                raw = raw.cpu()
            if hasattr(raw, "numpy"):
                raw = raw.numpy()
            value = np.array(raw)
            ctx.add_value(
                SemanticValue(
                    name=node.name,
                    shape=_normalize_shape(value.shape),
                    dtype=_infer_numpy_dtype(value),
                    kind="constant",
                    data=np.array(value, copy=True),
                )
            )
            continue

        if node.op == "call_module":
            module = modules[node.target]
            handler = _REGISTRY.find_module_handler(module)
            if handler is None:
                raise NotImplementedError(
                    f"Semantic frontend does not support call_module node '{node.name}' targeting {type(module).__name__}."
                )
            handler(ctx, node, module, modules)
            continue

        if node.op == "call_function":
            name = getattr(node.target, "__name__", None)
            handler = _REGISTRY.function_handlers.get(str(name))
            if handler is None and node.target is operator.add:
                handler = _REGISTRY.function_handlers.get("add")
            if handler is None and node.target is operator.mul:
                handler = _REGISTRY.function_handlers.get("mul")
            if handler is None:
                raise NotImplementedError(
                    f"Semantic frontend does not support call_function node '{node.name}' targeting {node.target!r}."
                )
            handler(ctx, node, modules)
            continue

        if node.op == "call_method":
            handler = _REGISTRY.method_handlers.get(str(node.target))
            if handler is None:
                raise NotImplementedError(
                    f"Semantic frontend does not support call_method node '{node.name}' targeting {node.target!r}."
                )
            handler(ctx, node, modules)
            continue

        if node.op == "output":
            raw_outputs = node.args[0]
            output_nodes = raw_outputs if isinstance(raw_outputs, (tuple, list)) else (raw_outputs,)
            for output_node in output_nodes:
                output_value = ctx.values[output_node.name]
                output_value.kind = "output"
                ctx.outputs.append(output_node.name)
                ctx.add_op(
                    VerifyOp(
                        name=f"{output_node.name}_verify",
                        inputs=[output_node.name],
                        outputs=[output_node.name],
                        label=output_node.name,
                        is_final_output=True,
                    )
                )
            continue

        raise NotImplementedError(
            f"Semantic frontend does not support FX node op={node.op!r}, target={node.target!r}."
        )

    graph = SemanticGraph(
        values=dict(ctx.values),
        ops=list(ctx.ops),
        inputs=list(ctx.inputs),
        outputs=list(ctx.outputs),
        metadata={"frontend": "semantic_fx"},
    )
    report = analyze_semantic_capabilities(graph)
    if not report.is_supported:
        details = "; ".join(f"{issue.op_name}:{issue.reason}" for issue in report.issues)
        raise NotImplementedError(f"Semantic frontend capability check failed: {details}")
    return graph
