from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable

import numpy as np

from .golden import GoldenModel
from .ir import DType, HostOp


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


def _validate_rope(step: HostOp) -> None:
    head_dim = int(step.attrs["head_dim"])
    theta = float(step.attrs.get("theta", 10000.0))
    if head_dim <= 0 or head_dim % 2 != 0:
        raise ValueError(f"Host op 'rope' requires positive even head_dim, got {head_dim}.")
    if theta <= 0.0:
        raise ValueError(f"Host op 'rope' requires theta > 0, got {theta}.")


def _validate_binary_same_shape(step: HostOp) -> None:
    return


def _softmax_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    axis = int(step.attrs.get("axis", -1))
    values[step.outputs[0]] = golden.softmax(values[step.inputs[0]], axis=axis)


def _softmax_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source_elements = int(np.array(values[step.inputs[0]], copy=False).size)
    elems = int(np.array(values[step.outputs[0]], copy=False).size)
    return "host_intrinsic", _counts(
        reads=source_elements * 2,
        adds=elems * 3,
        divs=elems,
        nonlinear=elems,
        writes=elems,
        branches=elems * 2,
    )


def _quantize_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    values[step.outputs[0]] = golden.quantize(
        values[step.inputs[0]],
        scale=float(step.attrs["scale"]),
        zero_point=int(step.attrs.get("zero_point", 0)),
        out_dtype=_dtype_attr(step.attrs.get("dtype", DType.INT8)),
    )


def _quantize_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source_elements = int(np.array(values[step.inputs[0]], copy=False).size)
    elems = int(np.array(values[step.outputs[0]], copy=False).size)
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
    source_elements = int(np.array(values[step.inputs[0]], copy=False).size)
    elems = int(np.array(values[step.outputs[0]], copy=False).size)
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
    values[step.outputs[0]] = 1.0 / (1.0 + np.exp(-source))


def _sigmoid_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source_elements = int(np.array(values[step.inputs[0]], copy=False).size)
    elems = int(np.array(values[step.outputs[0]], copy=False).size)
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
    erf = np.vectorize(math.erf, otypes=[np.float32])(source / np.float32(np.sqrt(2.0)))
    values[step.outputs[0]] = np.float32(0.5) * source * (np.float32(1.0) + erf)


def _gelu_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source_elements = int(np.array(values[step.inputs[0]], copy=False).size)
    elems = int(np.array(values[step.outputs[0]], copy=False).size)
    return "host_intrinsic", _counts(
        reads=source_elements,
        adds=elems * 3,
        muls=elems * 2,
        nonlinear=elems,
        writes=elems,
        branches=elems,
    )


def _silu_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    source = np.array(values[step.inputs[0]], dtype=np.float32, copy=False)
    values[step.outputs[0]] = (source / (np.float32(1.0) + np.exp(-source))).astype(np.float32)


def _silu_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source_elements = int(np.array(values[step.inputs[0]], copy=False).size)
    elems = int(np.array(values[step.outputs[0]], copy=False).size)
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
    source_elements = int(np.array(values[step.inputs[0]], copy=False).size)
    elems = int(np.array(values[step.outputs[0]], copy=False).size)
    return "host_intrinsic", _counts(
        reads=source_elements,
        clamps=elems,
        writes=elems,
        branches=elems,
    )


def _mean_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    source = np.array(values[step.inputs[0]], copy=False)
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
    source_elements = int(np.array(values[step.inputs[0]], copy=False).size)
    out_elems = int(np.array(values[step.outputs[0]], copy=False).size)
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
    elems = int(np.array(values[step.outputs[0]], copy=False).size)
    return "cpu_replaced", _counts(
        reads=elems,
        writes=elems,
        adds=elems,
        branches=elems,
    )


def _im2col_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    image = np.array(values[step.inputs[0]], copy=False)
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
    elems = int(np.array(values[step.outputs[0]], copy=False).size)
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


def _requantize_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    values[step.outputs[0]] = golden.requantize(
        values[step.inputs[0]],
        scale=float(step.attrs["scale"]),
        zero_point=int(step.attrs.get("zero_point", 0)),
        out_dtype=_dtype_attr(step.attrs.get("dtype", DType.INT16)),
    )


def _requantize_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    source_elements = int(np.array(values[step.inputs[0]], copy=False).size)
    elems = int(np.array(values[step.outputs[0]], copy=False).size)
    return "host_intrinsic", _counts(
        reads=source_elements,
        muls=elems,
        adds=elems * 2,
        clamps=elems,
        writes=elems,
        branches=elems,
    )


def _rmsnorm_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    x = np.array(values[step.inputs[0]], dtype=np.float32, copy=False)
    weight = np.array(values[step.inputs[1]], dtype=np.float32, copy=False).reshape(-1)
    hidden = x.shape[-1]
    if weight.size != hidden:
        raise ValueError(f"rmsnorm weight size mismatch: hidden={hidden}, weight={weight.size}.")
    eps = np.float32(step.attrs.get("eps", 1.0e-6))
    mean_sq = np.mean(np.square(x, dtype=np.float32), axis=-1, keepdims=True, dtype=np.float32)
    rms = np.sqrt(mean_sq + eps).astype(np.float32)
    values[step.outputs[0]] = ((x / rms) * weight.reshape((1,) * (x.ndim - 1) + (hidden,))).astype(np.float32)


def _rmsnorm_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    x = np.array(values[step.inputs[0]], copy=False)
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


def _mul_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    lhs = np.array(values[step.inputs[0]], dtype=np.float32, copy=False)
    rhs = np.array(values[step.inputs[1]], dtype=np.float32, copy=False)
    if lhs.shape != rhs.shape:
        raise ValueError(f"mul requires same-shape inputs, got {lhs.shape} and {rhs.shape}.")
    values[step.outputs[0]] = (lhs * rhs).astype(np.float32)


def _mul_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    elems = int(np.array(values[step.outputs[0]], copy=False).size)
    return "host_intrinsic", _counts(
        reads=elems * 2,
        muls=elems,
        writes=elems,
        branches=elems,
    )


def _add_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    lhs = np.array(values[step.inputs[0]], dtype=np.float32, copy=False)
    rhs = np.array(values[step.inputs[1]], dtype=np.float32, copy=False)
    if lhs.shape != rhs.shape:
        raise ValueError(f"add requires same-shape inputs, got {lhs.shape} and {rhs.shape}.")
    values[step.outputs[0]] = (lhs + rhs).astype(np.float32)


def _add_benchmark(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, Any]:
    elems = int(np.array(values[step.outputs[0]], copy=False).size)
    return "host_intrinsic", _counts(
        reads=elems * 2,
        adds=elems,
        writes=elems,
        branches=elems,
    )


def _rope_eval(step: HostOp, values: dict[str, np.ndarray], golden: GoldenModel) -> None:
    x = np.array(values[step.inputs[0]], dtype=np.float32, copy=False)
    out = np.array(x, copy=True)
    head_dim = int(step.attrs["head_dim"])
    position = int(step.attrs.get("position", 0))
    theta = float(step.attrs.get("theta", 10000.0))
    if x.shape[-1] != head_dim:
        raise ValueError(f"rope expects last dimension {head_dim}, got {x.shape[-1]}.")
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / np.float32(half)))

    if x.ndim == 2:
        positions = np.full((x.shape[0],), position, dtype=np.float32)
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
    x = np.array(values[step.inputs[0]], copy=False)
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
    HostOpSpec(
        "dequantize",
        _dequantize_eval,
        _dequantize_benchmark,
        required_attrs=("scale",),
        quant_boundary_policy="npu_to_host",
        semantic_validator=_require_positive_scale,
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
        "layout_restore",
        _layout_restore_eval,
        _movement_benchmark,
        required_attrs=("layout", "original_shape", "out_h", "out_w", "out_channels"),
        quant_boundary_policy="layout_transform",
        semantic_validator=_validate_layout_restore,
    ),
    HostOpSpec("mean", _mean_eval, _mean_benchmark, semantic_validator=_validate_mean),
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
    HostOpSpec("transpose", _transpose_eval, _movement_benchmark, semantic_validator=_validate_transpose),
):
    register_host_op(_spec)
