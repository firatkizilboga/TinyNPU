from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable

import numpy as np

from .golden import GoldenModel
from .ir import DType, HostOp
from .runtime_approx import (
    quantize_fp16_bits_to_i16_xform as _quantize_fp16_bits_to_i16_xform,
    rmsnorm_approx as _rmsnorm_runtime_approx,
    sigmoid_approx as _sigmoid_runtime_approx,
    silu_approx as _silu_runtime_approx,
    softmax_f16_approx as _softmax_f16_runtime_approx,
)


HostOpEvaluatorFn = Callable[[HostOp, dict[str, np.ndarray], GoldenModel], None]
HostOpBenchmarkFn = Callable[[HostOp, dict[str, np.ndarray]], tuple[str, Any]]


@dataclass(frozen=True)
class HostOpSpec:
    kind: str
    evaluator: HostOpEvaluatorFn
    benchmark: HostOpBenchmarkFn
    input_arity: int | None = 1
    output_arity: int | None = 1
    required_attrs: tuple[str, ...] = field(default_factory=tuple)
    quant_boundary_policy: str = "passthrough"
    semantic_validator: Callable[[HostOp], None] | None = None

    def validate(self, step: HostOp) -> None:
        if self.input_arity is not None and len(step.inputs) != self.input_arity:
            raise ValueError(
                f"Host op {self.kind!r} expects {self.input_arity} inputs, got {len(step.inputs)}."
            )
        if self.output_arity is not None and len(step.outputs) != self.output_arity:
            raise ValueError(
                f"Host op {self.kind!r} expects {self.output_arity} outputs, got {len(step.outputs)}."
            )
        missing = tuple(attr for attr in self.required_attrs if attr not in step.attrs)
        if missing:
            raise ValueError(f"Host op {self.kind!r} is missing required attrs: {', '.join(missing)}.")
        if self.semantic_validator is not None:
            self.semantic_validator(step)


_HOST_OP_REGISTRY: dict[str, HostOpSpec] = {}


def _counts(**kwargs: int):
    from .benchmark import PrimitiveCounts

    return PrimitiveCounts(**kwargs)


def _dtype_attr(value: Any) -> DType:
    if isinstance(value, DType):
        return value
    enum_value = getattr(value, "value", None)
    if isinstance(enum_value, str):
        try:
            return DType(enum_value)
        except ValueError:
            pass
    enum_name = getattr(value, "name", None)
    if isinstance(enum_name, str) and enum_name in DType.__members__:
        return DType[enum_name]
    return DType(str(value))


def _require_positive_scale(step: HostOp) -> None:
    scale = float(step.attrs["scale"])
    if scale <= 0.0:
        raise ValueError(f"Host op {step.kind!r} requires scale > 0, got {scale}.")


def _validate_mean(step: HostOp) -> None:
    dims = step.attrs.get("dim")
    if dims is not None:
        if not isinstance(dims, (list, tuple)):
            raise ValueError(f"Host op 'mean' expects 'dim' to be a list/tuple, got {type(dims).__name__}.")
        for dim in dims:
            if not isinstance(dim, int):
                raise ValueError(f"Host op 'mean' expects integer dims, got {dim!r}.")
    input_quant = step.attrs.get("input_quantization")
    if input_quant is not None:
        if not isinstance(input_quant, dict):
            raise ValueError("Host op 'mean' expects input_quantization to be a dict when provided.")
        if "scale" not in input_quant:
            raise ValueError("Host op 'mean' input_quantization is missing required attr 'scale'.")
        scale = float(input_quant["scale"])
        if scale <= 0.0:
            raise ValueError(f"Host op 'mean' requires input_quantization.scale > 0, got {scale}.")


def _validate_im2col(step: HostOp) -> None:
    kernel_size = int(step.attrs["kernel_size"])
    stride = int(step.attrs.get("stride", 1))
    padding = int(step.attrs.get("padding", 0))
    layout = str(step.attrs.get("input_layout", "hwc"))
    if kernel_size <= 0:
        raise ValueError(f"Host op 'im2col' requires kernel_size > 0, got {kernel_size}.")
    if stride <= 0:
        raise ValueError(f"Host op 'im2col' requires stride > 0, got {stride}.")
    if padding < 0:
        raise ValueError(f"Host op 'im2col' requires padding >= 0, got {padding}.")
    if layout not in {"hwc", "chw", "matrix_hwc"}:
        raise ValueError(f"Host op 'im2col' does not support input_layout={layout!r}.")
    if layout == "matrix_hwc":
        matrix_h = int(step.attrs.get("matrix_h", 0))
        matrix_w = int(step.attrs.get("matrix_w", 0))
        matrix_c = int(step.attrs.get("matrix_c", 0))
        if matrix_h <= 0 or matrix_w <= 0 or matrix_c <= 0:
            raise ValueError("Host op 'im2col' matrix_hwc layout requires positive matrix_h/matrix_w/matrix_c attrs.")


def _validate_layout_restore(step: HostOp) -> None:
    layout = str(step.attrs["layout"])
    original_shape = tuple(step.attrs["original_shape"])
    out_h = int(step.attrs["out_h"])
    out_w = int(step.attrs["out_w"])
    out_channels = int(step.attrs["out_channels"])
    if layout not in {"chw", "hwc"}:
        raise ValueError(f"Host op 'layout_restore' does not support layout={layout!r}.")
    if len(original_shape) not in {3, 4}:
        raise ValueError(
            f"Host op 'layout_restore' expects original_shape rank 3 or 4, got rank {len(original_shape)}."
        )
    if out_h <= 0 or out_w <= 0 or out_channels <= 0:
        raise ValueError("Host op 'layout_restore' requires positive out_h/out_w/out_channels.")


def _validate_reshape(step: HostOp) -> None:
    shape = tuple(step.attrs["shape"])
    if not shape:
        raise ValueError("Host op 'reshape' requires a non-empty shape.")
    if any(int(dim) <= 0 for dim in shape):
        raise ValueError(f"Host op 'reshape' requires all dimensions > 0, got {shape}.")


def _validate_transpose(step: HostOp) -> None:
    axes = step.attrs.get("axes")
    if axes is None:
        return
    if not isinstance(axes, (list, tuple)):
        raise ValueError(f"Host op 'transpose' expects 'axes' to be a list/tuple, got {type(axes).__name__}.")
    normalized = tuple(int(axis) for axis in axes)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"Host op 'transpose' axes must be unique, got {normalized}.")


def _validate_rmsnorm(step: HostOp) -> None:
    eps = float(step.attrs.get("eps", 1.0e-6))
    if eps <= 0.0:
        raise ValueError(f"Host op 'rmsnorm' requires eps > 0, got {eps}.")


def _validate_layernorm(step: HostOp) -> None:
    eps = float(step.attrs.get("eps", 1.0e-6))
    if eps <= 0.0:
        raise ValueError(f"Host op 'layernorm' requires eps > 0, got {eps}.")


def _validate_rope(step: HostOp) -> None:
    head_dim = int(step.attrs["head_dim"])
    theta = float(step.attrs.get("theta", 10000.0))
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(f"Host op 'rope' requires positive even head_dim, got {head_dim}.")
    if theta <= 0.0:
        raise ValueError(f"Host op 'rope' requires theta > 0, got {theta}.")


def _validate_binary_same_shape(step: HostOp) -> None:
    return


def _validate_linear(step: HostOp) -> None:
    if len(step.inputs) not in {2, 3}:
        raise ValueError(f"Host op 'linear' expects 2 or 3 inputs, got {len(step.inputs)}.")


def _validate_conv2d(step: HostOp) -> None:
    if len(step.inputs) not in {2, 3}:
        raise ValueError(f"Host op 'conv2d' expects 2 or 3 inputs, got {len(step.inputs)}.")
    stride = int(step.attrs.get("stride", 1))
    padding = int(step.attrs.get("padding", 0))
    kernel_size = int(step.attrs.get("kernel_size", 0))
    in_channels = int(step.attrs.get("in_channels", 0))
    out_channels = int(step.attrs.get("out_channels", 0))
    if stride <= 0:
        raise ValueError(f"Host op 'conv2d' requires stride > 0, got {stride}.")
    if padding < 0:
        raise ValueError(f"Host op 'conv2d' requires padding >= 0, got {padding}.")
    if kernel_size <= 0:
        raise ValueError(f"Host op 'conv2d' requires kernel_size > 0, got {kernel_size}.")
    if in_channels <= 0 or out_channels <= 0:
        raise ValueError("Host op 'conv2d' requires positive in_channels and out_channels.")


def _validate_pool2d(step: HostOp) -> None:
    kernel_size = tuple(int(v) for v in step.attrs.get("kernel_size", ()))
    stride = tuple(int(v) for v in step.attrs.get("stride", ()))
    padding = tuple(int(v) for v in step.attrs.get("padding", ()))
    if len(kernel_size) != 2 or len(stride) != 2 or len(padding) != 2:
        raise ValueError(f"Host op {step.kind!r} expects 2D kernel/stride/padding tuples.")
    if any(v <= 0 for v in kernel_size):
        raise ValueError(f"Host op {step.kind!r} requires kernel_size > 0, got {kernel_size}.")
    if any(v <= 0 for v in stride):
        raise ValueError(f"Host op {step.kind!r} requires stride > 0, got {stride}.")
    if any(v < 0 for v in padding):
        raise ValueError(f"Host op {step.kind!r} requires padding >= 0, got {padding}.")


def _validate_adaptive_avg_pool2d(step: HostOp) -> None:
    output_size = tuple(int(v) for v in step.attrs.get("output_size", ()))
    if len(output_size) != 2:
        raise ValueError("Host op 'adaptive_avg_pool2d' expects output_size to have length 2.")
    if any(v <= 0 for v in output_size):
        raise ValueError(f"Host op 'adaptive_avg_pool2d' requires positive output_size, got {output_size}.")


def _validate_causal_mask(step: HostOp) -> None:
    past_kv_len = int(step.attrs.get("past_kv_len", 0))
    if past_kv_len < 0:
        raise ValueError(f"Host op 'causal_mask' requires past_kv_len >= 0, got {past_kv_len}.")


def _validate_concat_lastdim2(step: HostOp) -> None:
    return


def _validate_slice_row(step: HostOp) -> None:
    row_index = int(step.attrs.get("row_index", 0))
    if row_index < 0:
        raise ValueError(f"Host op 'slice_row' requires row_index >= 0, got {row_index}.")


def _softmax_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    axis = int(step.attrs.get("axis", -1))
    values[step.outputs[0]] = golden.softmax(values[step.inputs[0]], axis=axis)


def _softmax_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source_elements = int(np.asarray(values[step.inputs[0]]).size)
    elems = int(np.asarray(values[step.outputs[0]]).size)
    return "host_intrinsic", _counts(
        reads=source_elements * 2,
        adds=elems * 3,
        divs=elems,
        nonlinear=elems,
        writes=elems,
        branches=elems * 2,
    )


def _softmax_f16_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    axis = int(step.attrs.get("axis", -1))
    probs = _softmax_f16_runtime_approx(values[step.inputs[0]], axis=axis)
    probs_f16_bits = probs.astype(np.float16).view(np.uint16).astype(np.int16)
    values[step.outputs[0]] = probs_f16_bits


def _linear_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    x = np.asarray(values[step.inputs[0]], dtype=np.float32)
    weight = np.asarray(values[step.inputs[1]], dtype=np.float32)
    bias = None if len(step.inputs) < 3 else np.asarray(values[step.inputs[2]], dtype=np.float32)
    if weight.ndim != 2:
        raise ValueError(f"linear expects rank-2 weight, got shape {weight.shape}.")
    if x.shape[-1] != weight.shape[1]:
        raise ValueError(f"linear input last dim {x.shape[-1]} does not match weight in_features {weight.shape[1]}.")
    x_rows = x.reshape(-1, x.shape[-1])
    out = x_rows @ weight.T
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != weight.shape[0]:
            raise ValueError(f"linear bias shape {bias.shape} does not match out_features {weight.shape[0]}.")
        out = out + bias.reshape(1, -1)
    values[step.outputs[0]] = out.reshape(*x.shape[:-1], weight.shape[0]).astype(np.float32)


def _linear_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    x = np.asarray(values[step.inputs[0]])
    weight = np.asarray(values[step.inputs[1]])
    rows = int(np.prod(x.shape[:-1], dtype=np.int64)) if x.ndim > 1 else 1
    in_features = int(weight.shape[1])
    out_features = int(weight.shape[0])
    out_elems = rows * out_features
    bias_reads = out_elems if len(step.inputs) >= 3 else 0
    return "host_intrinsic", _counts(
        reads=rows * in_features + rows * out_features * in_features + bias_reads,
        muls=rows * out_features * in_features,
        adds=rows * out_features * max(in_features - 1, 0) + bias_reads,
        writes=out_elems,
        branches=rows,
    )


def _conv2d_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    x = np.asarray(values[step.inputs[0]], dtype=np.float32)
    weight = np.asarray(values[step.inputs[1]], dtype=np.float32)
    bias = None if len(step.inputs) < 3 else np.asarray(values[step.inputs[2]], dtype=np.float32)
    stride = int(step.attrs["stride"])
    padding = int(step.attrs["padding"])
    if weight.ndim != 4:
        raise ValueError(f"conv2d expects rank-4 weight, got shape {weight.shape}.")
    if x.ndim == 3:
        batch = 1
        in_channels, in_h, in_w = x.shape
        x_nchw = x.reshape(1, in_channels, in_h, in_w)
        restore_rank3 = True
    elif x.ndim == 4:
        batch, in_channels, in_h, in_w = x.shape
        x_nchw = x
        restore_rank3 = False
    else:
        raise ValueError(f"conv2d expects rank-3 or rank-4 input, got shape {x.shape}.")
    out_channels, weight_in_channels, kernel_h, kernel_w = weight.shape
    if kernel_h != kernel_w:
        raise ValueError(f"conv2d expects square kernels, got shape {weight.shape}.")
    if in_channels != weight_in_channels:
        raise ValueError(f"conv2d input channels {in_channels} do not match weight channels {weight_in_channels}.")
    out_h = ((in_h + (2 * padding) - kernel_h) // stride) + 1
    out_w = ((in_w + (2 * padding) - kernel_w) // stride) + 1
    padded = np.pad(x_nchw, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode="constant")
    out = np.zeros((batch, out_channels, out_h, out_w), dtype=np.float32)
    for n in range(batch):
        for oc in range(out_channels):
            for oh in range(out_h):
                h_start = oh * stride
                for ow in range(out_w):
                    w_start = ow * stride
                    acc = 0.0
                    for ic in range(in_channels):
                        window = padded[n, ic, h_start : h_start + kernel_h, w_start : w_start + kernel_w]
                        acc += float(np.sum(window * weight[oc, ic], dtype=np.float32))
                    if bias is not None:
                        acc += float(bias[oc])
                    out[n, oc, oh, ow] = np.float32(acc)
    values[step.outputs[0]] = out[0] if restore_rank3 else out


def _conv2d_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    x = np.asarray(values[step.inputs[0]])
    weight = np.asarray(values[step.inputs[1]])
    batch = 1 if x.ndim == 3 else int(x.shape[0])
    in_h = int(x.shape[-2])
    in_w = int(x.shape[-1])
    out_channels = int(weight.shape[0])
    in_channels = int(weight.shape[1])
    kernel_h = int(weight.shape[2])
    kernel_w = int(weight.shape[3])
    stride = int(step.attrs["stride"])
    padding = int(step.attrs["padding"])
    out_h = ((in_h + (2 * padding) - kernel_h) // stride) + 1
    out_w = ((in_w + (2 * padding) - kernel_w) // stride) + 1
    macs = batch * out_channels * out_h * out_w * in_channels * kernel_h * kernel_w
    out_elems = batch * out_channels * out_h * out_w
    bias_reads = out_elems if len(step.inputs) >= 3 else 0
    return "host_intrinsic", _counts(
        reads=macs * 2 + bias_reads,
        muls=macs,
        adds=macs + bias_reads,
        writes=out_elems,
        branches=batch * out_channels * out_h,
    )


def _pool2d_shapes(source: np.ndarray) -> tuple[np.ndarray, int, int, int, int, bool]:
    if source.ndim == 3:
        channels, in_h, in_w = source.shape
        return source.reshape(1, channels, in_h, in_w), 1, channels, in_h, in_w, True
    if source.ndim == 4:
        batch, channels, in_h, in_w = source.shape
        return source, batch, channels, in_h, in_w, False
    raise ValueError(f"pool2d expects rank-3 or rank-4 input, got shape {source.shape}.")


def _maxpool2d_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    source = np.asarray(values[step.inputs[0]], dtype=np.float32)
    x, batch, channels, in_h, in_w, restore_rank3 = _pool2d_shapes(source)
    kernel_h, kernel_w = (int(v) for v in step.attrs["kernel_size"])
    stride_h, stride_w = (int(v) for v in step.attrs["stride"])
    pad_h, pad_w = (int(v) for v in step.attrs["padding"])
    out_h = ((in_h + (2 * pad_h) - kernel_h) // stride_h) + 1
    out_w = ((in_w + (2 * pad_w) - kernel_w) // stride_w) + 1
    padded = np.pad(
        x,
        ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=-np.inf,
    )
    out = np.empty((batch, channels, out_h, out_w), dtype=np.float32)
    for n in range(batch):
        for c in range(channels):
            for oh in range(out_h):
                h_start = oh * stride_h
                for ow in range(out_w):
                    w_start = ow * stride_w
                    window = padded[n, c, h_start : h_start + kernel_h, w_start : w_start + kernel_w]
                    out[n, c, oh, ow] = np.float32(np.max(window))
    values[step.outputs[0]] = out[0] if restore_rank3 else out


def _maxpool2d_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source = np.asarray(values[step.inputs[0]])
    _, batch, channels, in_h, in_w, _ = _pool2d_shapes(source)
    kernel_h, kernel_w = (int(v) for v in step.attrs["kernel_size"])
    stride_h, stride_w = (int(v) for v in step.attrs["stride"])
    pad_h, pad_w = (int(v) for v in step.attrs["padding"])
    out_h = ((in_h + (2 * pad_h) - kernel_h) // stride_h) + 1
    out_w = ((in_w + (2 * pad_w) - kernel_w) // stride_w) + 1
    windows = batch * channels * out_h * out_w
    window_elems = kernel_h * kernel_w
    out_elems = windows
    return "host_intrinsic", _counts(
        reads=windows * window_elems,
        clamps=windows * max(window_elems - 1, 0),
        writes=out_elems,
        branches=windows,
    )


def _avgpool2d_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    source = np.asarray(values[step.inputs[0]], dtype=np.float32)
    x, batch, channels, in_h, in_w, restore_rank3 = _pool2d_shapes(source)
    kernel_h, kernel_w = (int(v) for v in step.attrs["kernel_size"])
    stride_h, stride_w = (int(v) for v in step.attrs["stride"])
    pad_h, pad_w = (int(v) for v in step.attrs["padding"])
    count_include_pad = bool(step.attrs.get("count_include_pad", True))
    out_h = ((in_h + (2 * pad_h) - kernel_h) // stride_h) + 1
    out_w = ((in_w + (2 * pad_w) - kernel_w) // stride_w) + 1
    padded = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode="constant")
    out = np.empty((batch, channels, out_h, out_w), dtype=np.float32)
    for n in range(batch):
        for c in range(channels):
            for oh in range(out_h):
                h_start = oh * stride_h
                h_end = h_start + kernel_h
                valid_h_start = max(h_start - pad_h, 0)
                valid_h_end = min(h_end - pad_h, in_h)
                for ow in range(out_w):
                    w_start = ow * stride_w
                    w_end = w_start + kernel_w
                    valid_w_start = max(w_start - pad_w, 0)
                    valid_w_end = min(w_end - pad_w, in_w)
                    window = padded[n, c, h_start : h_start + kernel_h, w_start : w_start + kernel_w]
                    if count_include_pad:
                        denom = kernel_h * kernel_w
                    else:
                        denom = max(valid_h_end - valid_h_start, 0) * max(valid_w_end - valid_w_start, 0)
                    if denom <= 0:
                        out[n, c, oh, ow] = 0.0
                    else:
                        out[n, c, oh, ow] = np.float32(np.sum(window, dtype=np.float32) / np.float32(denom))
    values[step.outputs[0]] = out[0] if restore_rank3 else out


def _avgpool2d_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source = np.asarray(values[step.inputs[0]])
    _, batch, channels, in_h, in_w, _ = _pool2d_shapes(source)
    kernel_h, kernel_w = (int(v) for v in step.attrs["kernel_size"])
    stride_h, stride_w = (int(v) for v in step.attrs["stride"])
    pad_h, pad_w = (int(v) for v in step.attrs["padding"])
    out_h = ((in_h + (2 * pad_h) - kernel_h) // stride_h) + 1
    out_w = ((in_w + (2 * pad_w) - kernel_w) // stride_w) + 1
    windows = batch * channels * out_h * out_w
    window_elems = kernel_h * kernel_w
    out_elems = windows
    return "host_intrinsic", _counts(
        reads=windows * window_elems,
        adds=windows * max(window_elems - 1, 0),
        divs=windows,
        writes=out_elems,
        branches=windows,
    )


def _adaptive_avg_pool2d_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    source = np.asarray(values[step.inputs[0]], dtype=np.float32)
    x, batch, channels, in_h, in_w, restore_rank3 = _pool2d_shapes(source)
    out_h, out_w = (int(v) for v in step.attrs["output_size"])
    out = np.empty((batch, channels, out_h, out_w), dtype=np.float32)
    for n in range(batch):
        for c in range(channels):
            for oh in range(out_h):
                h_start = int(np.floor((oh * in_h) / out_h))
                h_end = int(np.ceil(((oh + 1) * in_h) / out_h))
                for ow in range(out_w):
                    w_start = int(np.floor((ow * in_w) / out_w))
                    w_end = int(np.ceil(((ow + 1) * in_w) / out_w))
                    window = x[n, c, h_start:h_end, w_start:w_end]
                    out[n, c, oh, ow] = np.float32(np.mean(window, dtype=np.float32))
    values[step.outputs[0]] = out[0] if restore_rank3 else out


def _adaptive_avg_pool2d_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source = np.asarray(values[step.inputs[0]])
    _, batch, channels, in_h, in_w, _ = _pool2d_shapes(source)
    out_h, out_w = (int(v) for v in step.attrs["output_size"])
    total_reads = 0
    for oh in range(out_h):
        h_start = int(np.floor((oh * in_h) / out_h))
        h_end = int(np.ceil(((oh + 1) * in_h) / out_h))
        for ow in range(out_w):
            w_start = int(np.floor((ow * in_w) / out_w))
            w_end = int(np.ceil(((ow + 1) * in_w) / out_w))
            total_reads += max(h_end - h_start, 0) * max(w_end - w_start, 0)
    total_reads *= batch * channels
    out_elems = batch * channels * out_h * out_w
    return "host_intrinsic", _counts(
        reads=total_reads,
        adds=max(total_reads - out_elems, 0),
        divs=out_elems,
        writes=out_elems,
        branches=out_elems,
    )


def _fp16_bits_to_float32_array(value: np.ndarray) -> np.ndarray:
    bits = np.asarray(value, dtype=np.int16).view(np.uint16)
    return bits.view(np.float16).astype(np.float32)


def _host_exp_approx_scalar(x: float) -> float:
    exp_neg_int = (
        1.0,
        0.36787945,
        0.13533528,
        0.049787067,
        0.01831564,
        0.006737947,
        0.0024787523,
        0.00091188197,
        0.00033546263,
        0.00012340980,
        0.00004539993,
        0.0000167017,
        0.0000061442124,
        0.0000022603294,
        0.00000083152872,
        0.00000030590232,
        0.00000011253518,
    )
    if x == 0.0:
        return 1.0
    if x > 0.0:
        return _host_recip_approx_scalar(_host_exp_approx_scalar(-x))
    if x <= -16.0:
        return 0.0
    k = int(-x)
    r = x + float(k)
    poly = 1.0 + r * (1.0 + r * (0.5 + r * (0.16666667 + r * (0.04166667 + r * 0.0083333333))))
    return exp_neg_int[k] * max(poly, 0.0)


def _host_recip_approx_scalar(x: float) -> float:
    if x == 0.0:
        return math.inf if math.copysign(1.0, x) > 0 else -math.inf
    ax = abs(x)
    if ax < 0.5:
        y = 1.0 / x
        return y * (2.0 - x * y)
    y = 48.0 / (17.0 * x + 31.0 * math.copysign(1.0, x))
    y = y * (2.0 - x * y)
    y = y * (2.0 - x * y)
    return y


def _host_erf_approx_scalar(x: float) -> float:
    p = 0.3275911
    a1 = 0.25482959
    a2 = -0.28449672
    a3 = 1.4214138
    a4 = -1.4531521
    a5 = 1.0614054
    sign = 1.0
    if x < 0.0:
        sign = -1.0
        x = -x
    t = _host_recip_approx_scalar(1.0 + p * x)
    poly = (((((a5 * t) + a4) * t + a3) * t + a2) * t + a1) * t
    return sign * (1.0 - poly * _host_exp_approx_scalar(-(x * x)))


def _gelu_runtime_approx(source: np.ndarray) -> np.ndarray:
    source_f32 = np.asarray(source, dtype=np.float32)
    erf = np.vectorize(lambda v: _host_erf_approx_scalar(float(v) * 0.70710678), otypes=[np.float32])(source_f32)
    return (np.float32(0.5) * source_f32 * (np.float32(1.0) + erf)).astype(np.float32)


def _softmax_f16_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source_elements = int(np.asarray(values[step.inputs[0]]).size)
    elems = int(np.asarray(values[step.outputs[0]]).size)
    return "host_intrinsic", _counts(
        reads=source_elements * 2,
        adds=elems * 3,
        divs=elems,
        nonlinear=elems,
        writes=elems,
        branches=elems * 2,
    )


def _quantize_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    source = values[step.inputs[0]]
    if str(step.attrs.get("input_encoding", "")) == "fp16_bits":
        if str(step.attrs.get("_npu_write_transform", "")) == "xform_q_f16_i16":
            if int(step.attrs.get("zero_point", 0)) != 0:
                raise ValueError("xform_q_f16_i16 quantize does not support non-zero zero_point.")
            if _dtype_attr(step.attrs.get("dtype", DType.INT8)) != DType.INT16:
                raise ValueError("xform_q_f16_i16 quantize only supports INT16 output.")
            values[step.outputs[0]] = _quantize_fp16_bits_to_i16_xform(
                np.asarray(source),
                scale=float(step.attrs["scale"]),
            )
            return
        source = _fp16_bits_to_float32_array(np.asarray(source))
    values[step.outputs[0]] = golden.quantize(
        source,
        scale=float(step.attrs["scale"]),
        zero_point=int(step.attrs.get("zero_point", 0)),
        out_dtype=_dtype_attr(step.attrs.get("dtype", DType.INT8)),
    )


def _quantize_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source_elements = int(np.asarray(values[step.inputs[0]]).size)
    elems = int(np.asarray(values[step.outputs[0]]).size)
    return "host_intrinsic", _counts(
        reads=source_elements,
        divs=elems,
        adds=elems * 2,
        clamps=elems,
        writes=elems,
        branches=elems,
    )


def _dequantize_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    values[step.outputs[0]] = golden.dequantize(
        values[step.inputs[0]],
        scale=float(step.attrs["scale"]),
        zero_point=int(step.attrs.get("zero_point", 0)),
    )


def _dequantize_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source_elements = int(np.asarray(values[step.inputs[0]]).size)
    elems = int(np.asarray(values[step.outputs[0]]).size)
    adds = elems * 2 if int(step.attrs.get("zero_point", 0)) != 0 else elems
    return "host_intrinsic", _counts(
        reads=source_elements,
        adds=adds,
        muls=elems,
        writes=elems,
        branches=elems,
    )


def _sigmoid_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    source = np.array(values[step.inputs[0]], dtype=np.float32)
    values[step.outputs[0]] = _sigmoid_runtime_approx(source)


def _sigmoid_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source_elements = int(np.asarray(values[step.inputs[0]]).size)
    elems = int(np.asarray(values[step.outputs[0]]).size)
    return "host_intrinsic", _counts(
        reads=source_elements,
        adds=elems * 2,
        divs=elems,
        nonlinear=elems,
        writes=elems,
        branches=elems,
    )


def _gelu_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    source = np.array(values[step.inputs[0]], dtype=np.float32)
    out = _gelu_runtime_approx(source)
    if str(step.attrs.get("output_encoding", "")) == "fp16_bits":
        values[step.outputs[0]] = out.astype(np.float16).view(np.uint16).astype(np.int16)
    else:
        values[step.outputs[0]] = out


def _gelu_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source_elements = int(np.asarray(values[step.inputs[0]]).size)
    elems = int(np.asarray(values[step.outputs[0]]).size)
    return "host_intrinsic", _counts(
        reads=source_elements,
        adds=elems * 3,
        muls=elems * 2,
        nonlinear=elems,
        writes=elems,
        branches=elems,
    )


def _k_cache_scatter_write_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    """Python-side simulation of the scatter-write into the K-cache.

    On hardware this writes each INT16 key element to its per-token lane in the UB via
    shared SRAM.  In the Python evaluator we replicate the logical result: write the
    [1, d_head] key into the slot tensor AND update the corresponding column of the base
    k_cache [d_head, token_count] matrix so that downstream ops see the correct values.
    """
    key = np.asarray(values[step.inputs[0]], dtype=np.int16)  # [1, d_head]
    token_index = int(step.attrs["token_index"])
    k_cache_base = step.inputs[1] if len(step.inputs) > 1 else str(step.attrs["k_cache_base"])
    values[step.outputs[0]] = key
    if k_cache_base in values:
        values[k_cache_base][:, token_index] = key.flatten()


def _k_cache_scatter_write_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    d_head = int(np.asarray(values[step.inputs[0]]).size)
    return "host_intrinsic", _counts(reads=d_head, writes=d_head)


def _v_cache_scatter_write_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    value = np.asarray(values[step.inputs[0]], dtype=np.int16)
    token_index = int(step.attrs["token_index"])
    v_cache_base = step.inputs[1] if len(step.inputs) > 1 else str(step.attrs["v_cache_base"])
    values[step.outputs[0]] = value
    if v_cache_base in values:
        values[v_cache_base][token_index, :] = value.reshape(-1)


def _v_cache_scatter_write_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    d_head = int(np.asarray(values[step.inputs[0]]).size)
    return "host_intrinsic", _counts(reads=d_head, writes=d_head)


def _k_cache_scatter_matrix_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    source = np.asarray(values[step.inputs[0]], dtype=np.int16)
    if source.ndim != 2:
        raise ValueError(f"k_cache_scatter_matrix expects rank-2 source, got {source.shape}.")
    values[step.outputs[0]] = np.asarray(source.T, dtype=np.int16, copy=True)


def _k_cache_scatter_matrix_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    elems = int(np.asarray(values[step.inputs[0]]).size)
    return "host_intrinsic", _counts(reads=elems, writes=elems)


def _v_cache_scatter_matrix_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    source = np.asarray(values[step.inputs[0]], dtype=np.int16)
    if source.ndim != 2:
        raise ValueError(f"v_cache_scatter_matrix expects rank-2 source, got {source.shape}.")
    values[step.outputs[0]] = np.asarray(source, dtype=np.int16, copy=True)


def _v_cache_scatter_matrix_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    elems = int(np.asarray(values[step.inputs[0]]).size)
    return "host_intrinsic", _counts(reads=elems, writes=elems)


def _silu_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    source = np.asarray(values[step.inputs[0]], dtype=np.float32)
    values[step.outputs[0]] = _silu_runtime_approx(source)


def _silu_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source_elements = int(np.asarray(values[step.inputs[0]]).size)
    elems = int(np.asarray(values[step.outputs[0]]).size)
    return "host_intrinsic", _counts(
        reads=source_elements,
        adds=elems * 2,
        muls=elems,
        divs=elems,
        nonlinear=elems,
        writes=elems,
        branches=elems,
    )


def _relu_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    values[step.outputs[0]] = np.maximum(values[step.inputs[0]], 0)


def _relu_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source_elements = int(np.asarray(values[step.inputs[0]]).size)
    elems = int(np.asarray(values[step.outputs[0]]).size)
    return "host_intrinsic", _counts(
        reads=source_elements,
        clamps=elems,
        writes=elems,
        branches=elems,
    )


def _mean_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    source = np.asarray(values[step.inputs[0]])
    dims = step.attrs.get("dim")
    axis = None if dims is None else tuple(int(dim) for dim in dims)
    keepdim = bool(step.attrs.get("keepdim", False))
    input_quant = step.attrs.get("input_quantization")
    if input_quant is not None:
        source = golden.dequantize(
            source,
            scale=float(input_quant["scale"]),
            zero_point=int(input_quant.get("zero_point", 0)),
        )
    mean_value = np.mean(source.astype(np.float32), axis=axis, keepdims=keepdim)
    values[step.outputs[0]] = mean_value.astype(np.float32)


def _mean_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source_elements = int(np.asarray(values[step.inputs[0]]).size)
    out_elems = int(np.asarray(values[step.outputs[0]]).size)
    counts = _counts(
        reads=source_elements,
        adds=source_elements,
        divs=out_elems,
        writes=out_elems,
        branches=source_elements + out_elems,
    )
    input_quant = step.attrs.get("input_quantization")
    if input_quant is not None:
        counts.reads += source_elements
        counts.adds += source_elements * (2 if int(input_quant.get("zero_point", 0)) != 0 else 1)
        counts.muls += source_elements
        counts.branches += source_elements
    return "host_intrinsic", counts


def _alias_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    values[step.outputs[0]] = np.array(values[step.inputs[0]], copy=True)


def _movement_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    elems = int(np.asarray(values[step.outputs[0]]).size)
    return "cpu_replaced", _counts(
        reads=elems,
        writes=elems,
        adds=elems,
        branches=elems,
    )


def _im2col_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    image = np.asarray(values[step.inputs[0]])
    layout = step.attrs.get("input_layout", "hwc")
    if layout == "matrix_hwc":
        matrix_h = int(step.attrs["matrix_h"])
        matrix_w = int(step.attrs["matrix_w"])
        matrix_c = int(step.attrs["matrix_c"])
        expected_shape = (matrix_h * matrix_w, matrix_c)
        if image.shape != expected_shape:
            raise ValueError(
                f"im2col matrix_hwc input shape mismatch: expected {expected_shape}, got {tuple(image.shape)}."
            )
        image = image.reshape(matrix_h, matrix_w, matrix_c)
    elif layout == "chw":
        if image.ndim == 4:
            if image.shape[0] != 1:
                raise NotImplementedError(
                    f"im2col host op only supports batch size 1 for CHW input, got shape {image.shape}."
                )
            image = image[0]
        image = np.transpose(image, (1, 2, 0))
    values[step.outputs[0]] = golden.im2col(
        image,
        kernel_size=int(step.attrs["kernel_size"]),
        stride=int(step.attrs.get("stride", 1)),
        padding=int(step.attrs.get("padding", 0)),
    )


def _im2col_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    elems = int(np.asarray(values[step.outputs[0]]).size)
    return "cpu_replaced", _counts(
        reads=elems,
        writes=elems,
        adds=elems * 3,
        branches=elems,
    )


def _reshape_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    values[step.outputs[0]] = np.reshape(values[step.inputs[0]], tuple(step.attrs["shape"]))


def _transpose_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    values[step.outputs[0]] = np.transpose(values[step.inputs[0]], axes=tuple(step.attrs.get("axes", [])) or None)


def _layout_restore_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    source = np.asarray(values[step.inputs[0]])
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


def _requantize_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    values[step.outputs[0]] = golden.requantize(
        values[step.inputs[0]],
        scale=float(step.attrs["scale"]),
        zero_point=int(step.attrs.get("zero_point", 0)),
        out_dtype=_dtype_attr(step.attrs.get("dtype", DType.INT16)),
    )


def _requantize_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source_elements = int(np.asarray(values[step.inputs[0]]).size)
    elems = int(np.asarray(values[step.outputs[0]]).size)
    return "host_intrinsic", _counts(
        reads=source_elements,
        muls=elems,
        adds=elems * 2,
        clamps=elems,
        writes=elems,
        branches=elems,
    )


def _rmsnorm_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    x = np.asarray(values[step.inputs[0]], dtype=np.float32)
    weight = np.asarray(values[step.inputs[1]], dtype=np.float32).reshape(-1)
    eps = np.float32(step.attrs.get("eps", 1.0e-6))
    values[step.outputs[0]] = _rmsnorm_runtime_approx(x, weight, float(eps))


def _rmsnorm_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    x = np.asarray(values[step.inputs[0]])
    elems = int(x.size)
    rows = int(elems // x.shape[-1])
    return "host_intrinsic", _counts(
        reads=elems * 2,
        muls=elems * 3,
        adds=elems + rows,
        divs=elems,
        nonlinear=rows,
        writes=elems,
        branches=elems,
    )


def _layernorm_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    x = np.asarray(values[step.inputs[0]], dtype=np.float32)
    weight_bias = np.asarray(values[step.inputs[1]], dtype=np.float32).reshape(-1)
    hidden = x.shape[-1]
    if weight_bias.size != hidden * 2:
        raise ValueError(f"layernorm weight/bias size mismatch: hidden={hidden}, weight_bias={weight_bias.size}.")
    weight = weight_bias[:hidden]
    bias = weight_bias[hidden:]
    eps = np.float32(step.attrs.get("eps", 1.0e-6))
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    centered = x - mean
    var = np.mean(np.square(centered, dtype=np.float32), axis=-1, keepdims=True, dtype=np.float32)
    y = centered / np.sqrt(var + eps).astype(np.float32)
    out = (y * weight.reshape((1,) * (x.ndim - 1) + (hidden,)) + bias.reshape((1,) * (x.ndim - 1) + (hidden,))).astype(np.float32)
    if str(step.attrs.get("output_encoding", "")) == "fp16_bits":
        values[step.outputs[0]] = out.astype(np.float16).view(np.uint16).astype(np.int16)
    else:
        values[step.outputs[0]] = out


def _layernorm_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    x = np.asarray(values[step.inputs[0]])
    elems = int(x.size)
    rows = int(elems // x.shape[-1])
    return "host_intrinsic", _counts(
        reads=elems * 3,
        muls=elems * 3,
        adds=elems * 3 + rows,
        divs=elems,
        nonlinear=rows,
        writes=elems,
        branches=elems,
    )


def _mul_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    lhs = np.asarray(values[step.inputs[0]], dtype=np.float32)
    rhs = np.asarray(values[step.inputs[1]], dtype=np.float32)
    values[step.outputs[0]] = (lhs * rhs).astype(np.float32)


def _mul_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    elems = int(np.asarray(values[step.outputs[0]]).size)
    return "host_intrinsic", _counts(
        reads=elems * 2,
        muls=elems,
        writes=elems,
        branches=elems,
    )


def _add_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    lhs = np.asarray(values[step.inputs[0]], dtype=np.float32)
    rhs = np.asarray(values[step.inputs[1]], dtype=np.float32)
    values[step.outputs[0]] = (lhs + rhs).astype(np.float32)


def _add_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    elems = int(np.asarray(values[step.outputs[0]]).size)
    return "host_intrinsic", _counts(
        reads=elems * 2,
        adds=elems,
        writes=elems,
        branches=elems,
    )


def _causal_mask_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    source = np.asarray(values[step.inputs[0]])
    if source.ndim < 2:
        raise ValueError(f"causal_mask expects rank >= 2, got rank {source.ndim}.")
    out = np.array(source, copy=True)
    q_len = source.shape[-2]
    k_len = source.shape[-1]
    past_kv_len = int(step.attrs.get("past_kv_len", 0))
    if source.dtype.kind == "f":
        fill_value = np.float32(step.attrs.get("fill_value", -1.0e9))
    else:
        fill_value = np.int32(step.attrs.get("fill_value", np.iinfo(np.int16).min))
    for row in range(q_len):
        max_col = past_kv_len + row
        if max_col + 1 < k_len:
            out[..., row, max_col + 1 :] = fill_value
    values[step.outputs[0]] = out


def _causal_mask_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    elems = int(np.asarray(values[step.outputs[0]]).size)
    return "host_intrinsic", _counts(
        reads=elems,
        writes=elems,
        branches=elems,
    )


def _concat_lastdim2_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    lhs = np.asarray(values[step.inputs[0]])
    rhs = np.asarray(values[step.inputs[1]])
    if lhs.ndim == 0 or rhs.ndim == 0:
        raise ValueError("concat_lastdim2 expects rank >= 1 inputs.")
    if lhs.shape[:-1] != rhs.shape[:-1]:
        raise ValueError(f"concat_lastdim2 prefix-shape mismatch: {lhs.shape} vs {rhs.shape}.")
    values[step.outputs[0]] = np.concatenate([lhs, rhs], axis=-1)


def _concat_lastdim2_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    lhs_elems = int(np.asarray(values[step.inputs[0]]).size)
    rhs_elems = int(np.asarray(values[step.inputs[1]]).size)
    out_elems = int(np.asarray(values[step.outputs[0]]).size)
    return "host_intrinsic", _counts(
        reads=lhs_elems + rhs_elems,
        writes=out_elems,
        branches=out_elems,
    )


def _slice_row_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    source = np.asarray(values[step.inputs[0]])
    if source.ndim < 2:
        raise ValueError(f"slice_row expects rank >= 2 input, got rank {source.ndim}.")
    row_index = int(step.attrs.get("row_index", 0))
    if row_index >= source.shape[0]:
        raise ValueError(f"slice_row row_index {row_index} out of range for shape {source.shape}.")
    values[step.outputs[0]] = np.asarray(source[row_index : row_index + 1, ...], copy=True)


def _slice_row_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    out_elems = int(np.asarray(values[step.outputs[0]]).size)
    return "host_intrinsic", _counts(
        reads=out_elems,
        writes=out_elems,
        branches=1,
    )


def _rope_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    x = np.asarray(values[step.inputs[0]], dtype=np.float32)
    out = np.array(x, copy=True)
    head_dim = int(step.attrs["head_dim"])
    position = int(step.attrs.get("position", 0))
    theta = float(step.attrs.get("theta", 10000.0))
    if x.shape[-1] != head_dim:
        raise ValueError(f"rope expects last dimension {head_dim}, got {x.shape[-1]}.")
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / np.float32(half)))

    if x.ndim == 2:
        positions = np.arange(position, position + x.shape[0], dtype=np.float32)
        reshape = (x.shape[0], 1)
    elif x.ndim >= 3:
        positions = np.arange(position, position + x.shape[-2], dtype=np.float32)
        reshape = (1,) * (x.ndim - 2) + (x.shape[-2], 1)
    else:
        raise ValueError(f"rope expects rank >= 2, got rank {x.ndim}.")

    angles = positions.reshape(reshape) * inv_freq.reshape((1,) * len(reshape[:-1]) + (half,))
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    first = x[..., :half]
    second = x[..., half:head_dim]
    out[..., :half] = first * cos - second * sin
    out[..., half:head_dim] = second * cos + first * sin
    values[step.outputs[0]] = out.astype(np.float32)


def _rope_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    x = np.asarray(values[step.inputs[0]])
    elems = int(x.size)
    return "host_intrinsic", _counts(
        reads=elems,
        muls=elems * 2,
        adds=elems,
        nonlinear=elems,
        writes=elems,
        branches=elems,
    )


def register_host_op(spec: HostOpSpec, *, replace: bool = False) -> HostOpSpec:
    if spec.kind in _HOST_OP_REGISTRY and not replace:
        raise ValueError(f"Host op {spec.kind!r} is already registered.")
    _HOST_OP_REGISTRY[spec.kind] = spec
    return spec


def get_host_op_spec(kind: str) -> HostOpSpec:
    try:
        return _HOST_OP_REGISTRY[kind]
    except KeyError as exc:
        raise NotImplementedError(f"Unsupported host op '{kind}'.") from exc


def registered_host_op_kinds() -> tuple[str, ...]:
    return tuple(sorted(_HOST_OP_REGISTRY))


def execute_host_op(step: HostOp, values: dict[str, np.ndarray], *, golden: GoldenModel | None = None) -> None:
    spec = get_host_op_spec(step.kind)
    spec.validate(step)
    spec.evaluator(step, values, golden or GoldenModel())


def benchmark_host_op(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    spec = get_host_op_spec(step.kind)
    spec.validate(step)
    return spec.benchmark(step, values)


for _spec in (
    HostOpSpec("add", _add_eval, _add_benchmark, input_arity=2, semantic_validator=_validate_binary_same_shape),
    HostOpSpec("alias", _alias_eval, _movement_benchmark),
    HostOpSpec("causal_mask", _causal_mask_eval, _causal_mask_benchmark, semantic_validator=_validate_causal_mask),
    HostOpSpec(
        "concat_lastdim2",
        _concat_lastdim2_eval,
        _concat_lastdim2_benchmark,
        input_arity=2,
        semantic_validator=_validate_concat_lastdim2,
    ),
    HostOpSpec("slice_row", _slice_row_eval, _slice_row_benchmark, semantic_validator=_validate_slice_row),
    HostOpSpec(
        "dequantize",
        _dequantize_eval,
        _dequantize_benchmark,
        required_attrs=("scale",),
        quant_boundary_policy="npu_to_host",
        semantic_validator=_require_positive_scale,
    ),
    HostOpSpec("linear", _linear_eval, _linear_benchmark, input_arity=None, semantic_validator=_validate_linear),
    HostOpSpec(
        "conv2d",
        _conv2d_eval,
        _conv2d_benchmark,
        input_arity=None,
        required_attrs=("stride", "padding", "kernel_size", "in_channels", "out_channels"),
        semantic_validator=_validate_conv2d,
    ),
    HostOpSpec(
        "maxpool2d",
        _maxpool2d_eval,
        _maxpool2d_benchmark,
        required_attrs=("kernel_size", "stride", "padding"),
        semantic_validator=_validate_pool2d,
    ),
    HostOpSpec(
        "avgpool2d",
        _avgpool2d_eval,
        _avgpool2d_benchmark,
        required_attrs=("kernel_size", "stride", "padding"),
        semantic_validator=_validate_pool2d,
    ),
    HostOpSpec(
        "adaptive_avg_pool2d",
        _adaptive_avg_pool2d_eval,
        _adaptive_avg_pool2d_benchmark,
        required_attrs=("output_size",),
        semantic_validator=_validate_adaptive_avg_pool2d,
    ),
    HostOpSpec(
        "im2col",
        _im2col_eval,
        _im2col_benchmark,
        required_attrs=("kernel_size",),
        quant_boundary_policy="layout_transform",
        semantic_validator=_validate_im2col,
    ),
    HostOpSpec("gelu", _gelu_eval, _gelu_benchmark),
    HostOpSpec(
        "k_cache_scatter_write",
        _k_cache_scatter_write_eval,
        _k_cache_scatter_write_benchmark,
        input_arity=None,
        required_attrs=("token_index", "k_cache_base"),
        quant_boundary_policy="host_to_npu",
    ),
    HostOpSpec(
        "v_cache_scatter_write",
        _v_cache_scatter_write_eval,
        _v_cache_scatter_write_benchmark,
        input_arity=None,
        required_attrs=("token_index", "v_cache_base"),
        quant_boundary_policy="host_to_npu",
    ),
    HostOpSpec(
        "k_cache_scatter_matrix",
        _k_cache_scatter_matrix_eval,
        _k_cache_scatter_matrix_benchmark,
        quant_boundary_policy="host_to_npu",
    ),
    HostOpSpec(
        "v_cache_scatter_matrix",
        _v_cache_scatter_matrix_eval,
        _v_cache_scatter_matrix_benchmark,
        quant_boundary_policy="host_to_npu",
    ),
    HostOpSpec(
        "layout_restore",
        _layout_restore_eval,
        _movement_benchmark,
        required_attrs=("layout", "original_shape", "out_h", "out_w", "out_channels"),
        quant_boundary_policy="layout_transform",
        semantic_validator=_validate_layout_restore,
    ),
    HostOpSpec("mean", _mean_eval, _mean_benchmark, semantic_validator=_validate_mean),
    HostOpSpec(
        "layernorm",
        _layernorm_eval,
        _layernorm_benchmark,
        input_arity=2,
        required_attrs=("eps",),
        semantic_validator=_validate_layernorm,
    ),
    HostOpSpec("mul", _mul_eval, _mul_benchmark, input_arity=2, semantic_validator=_validate_binary_same_shape),
    HostOpSpec(
        "quantize",
        _quantize_eval,
        _quantize_benchmark,
        required_attrs=("scale",),
        quant_boundary_policy="host_to_npu",
        semantic_validator=_require_positive_scale,
    ),
    HostOpSpec("relu", _relu_eval, _relu_benchmark),
    HostOpSpec(
        "rmsnorm",
        _rmsnorm_eval,
        _rmsnorm_benchmark,
        input_arity=2,
        required_attrs=("eps",),
        semantic_validator=_validate_rmsnorm,
    ),
    HostOpSpec(
        "requantize",
        _requantize_eval,
        _requantize_benchmark,
        required_attrs=("scale",),
        quant_boundary_policy="host_to_npu",
        semantic_validator=_require_positive_scale,
    ),
    HostOpSpec("reshape", _reshape_eval, _movement_benchmark, required_attrs=("shape",), semantic_validator=_validate_reshape),
    HostOpSpec("rope", _rope_eval, _rope_benchmark, required_attrs=("head_dim",), semantic_validator=_validate_rope),
    HostOpSpec("sigmoid", _sigmoid_eval, _sigmoid_benchmark),
    HostOpSpec("silu", _silu_eval, _silu_benchmark),
    HostOpSpec("softmax", _softmax_eval, _softmax_benchmark),
    HostOpSpec("softmax_f16", _softmax_f16_eval, _softmax_f16_benchmark),
    HostOpSpec("transpose", _transpose_eval, _movement_benchmark, semantic_validator=_validate_transpose),
):
    register_host_op(_spec)
