from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .builder import IRBuilder
from .golden import GoldenModel
from .host_ops import execute_host_op
from .ir import DType, HostOp, TensorKind, TensorSpec, normalize_shape, supports_fused_activation
from .semantic_ir import (
    ActivationOp,
    AdaptiveAvgPool2dOp,
    AvgPool2dOp,
    BinaryOp,
    Conv2dOp,
    CompilerReadyConv2dOp,
    CompilerReadyLinearOp,
    DequantizeOp,
    LinearOp,
    MaxPool2dOp,
    MeanOp,
    QuantizeOp,
    ReshapeOp,
    SemanticGraph,
    SemanticValue,
    VerifyOp,
)


def _dtype_storage_array(value: np.ndarray, dtype: DType) -> np.ndarray:
    array = np.array(value, copy=False)
    if dtype == DType.INT4:
        return np.clip(array.astype(np.int32), -8, 7).astype(np.int8)
    if dtype == DType.INT8:
        return np.clip(array.astype(np.int32), -128, 127).astype(np.int8)
    if dtype == DType.INT16:
        return np.clip(array.astype(np.int32), -32768, 32767).astype(np.int16)
    if dtype == DType.INT32:
        return array.astype(np.int32)
    if dtype == DType.FLOAT32:
        return array.astype(np.float32)
    raise ValueError(f"Unsupported semantic storage dtype {dtype}.")


def _normalize_conv_input(value: np.ndarray, *, in_channels: int) -> tuple[np.ndarray, tuple[int, ...], str]:
    arr = np.array(value, copy=False)
    original_shape = normalize_shape(arr.shape)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise NotImplementedError("Semantic conv execution supports batch size 1 only.")
        arr = arr[0]
    if arr.ndim != 3:
        raise NotImplementedError(f"Semantic conv execution expects 3D/4D inputs, got {original_shape}.")
    if arr.shape[0] == in_channels:
        return np.transpose(arr, (1, 2, 0)), original_shape, "chw"
    if arr.shape[-1] == in_channels:
        return arr, original_shape, "hwc"
    raise NotImplementedError(
        f"Could not infer semantic conv input layout from shape {original_shape} with in_channels={in_channels}."
    )


def _restore_conv_output_layout(value: np.ndarray, layout: str, original_shape: tuple[int, ...]) -> np.ndarray:
    if layout == "chw":
        chw = np.transpose(value, (2, 0, 1))
        if len(original_shape) == 4:
            return np.expand_dims(chw, axis=0)
        return chw
    if layout == "hwc":
        return value
    raise ValueError(f"Unsupported conv layout tag {layout!r}.")


def _normalize_linear_input(value: np.ndarray, *, in_features: int) -> tuple[np.ndarray, tuple[int, ...], str]:
    arr = np.array(value, copy=False)
    original_shape = normalize_shape(arr.shape)
    if arr.ndim == 1:
        if arr.shape[0] != in_features:
            raise NotImplementedError(f"Semantic linear execution expects {in_features} features, got {original_shape}.")
        return arr.reshape(1, in_features), original_shape, "vector"
    if arr.ndim == 2:
        if arr.shape == (1, in_features):
            return arr, original_shape, "row_batch1"
        if arr.shape == (in_features, 1):
            return arr.T, original_shape, "column"
    raise NotImplementedError(
        f"Semantic linear execution supports vectors, row batch-1, or column vectors only. Got {original_shape}."
    )


def _restore_linear_output_layout(value: np.ndarray, layout: str, original_shape: tuple[int, ...]) -> np.ndarray:
    if layout == "vector":
        return value.reshape(value.shape[1])
    if layout == "row_batch1":
        return value
    if layout == "column":
        return value.T.reshape(value.shape[1], 1)
    raise ValueError(f"Unsupported linear layout tag {layout!r} for original shape {original_shape}.")


def _maybe_fused_activation(graph: SemanticGraph, index: int) -> tuple[str, str | None, int]:
    op = graph.ops[index]
    if index + 1 >= len(graph.ops):
        return op.outputs[0], None, 1
    next_op = graph.ops[index + 1]
    if not isinstance(next_op, ActivationOp):
        return op.outputs[0], None, 1
    if next_op.inputs != [op.outputs[0]]:
        return op.outputs[0], None, 1
    return next_op.outputs[0], next_op.kind, 2


def execute_semantic_graph(graph: SemanticGraph, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    env: dict[str, np.ndarray] = {}
    golden = GoldenModel()

    for name, value in graph.values.items():
        if value.kind == "constant" and value.data is not None:
            env[name] = np.array(value.data, copy=True)
        elif value.kind == "input":
            env[name] = np.array(inputs[name], copy=True)

    index = 0
    while index < len(graph.ops):
        op = graph.ops[index]
        if isinstance(op, VerifyOp):
            index += 1
            continue
        if isinstance(op, QuantizeOp):
            step = HostOp(
                name=op.name,
                kind="quantize",
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attrs={"scale": op.scale, "zero_point": op.zero_point, "dtype": op.dtype},
            )
            execute_host_op(step, env, golden=golden)
            env[op.outputs[0]] = _dtype_storage_array(env[op.outputs[0]], op.dtype)
            index += 1
            continue
        if isinstance(op, DequantizeOp):
            step = HostOp(
                name=op.name,
                kind="dequantize",
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attrs={"scale": op.scale, "zero_point": op.zero_point, "output_encoding": op.output_encoding},
            )
            execute_host_op(step, env, golden=golden)
            if op.output_encoding == "fp16_bits":
                env[op.outputs[0]] = env[op.outputs[0]].astype(np.int16)
            else:
                env[op.outputs[0]] = env[op.outputs[0]].astype(np.float32)
            index += 1
            continue
        if isinstance(op, ReshapeOp):
            step = HostOp(
                name=op.name,
                kind="reshape",
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attrs={"shape": tuple(op.shape)},
            )
            execute_host_op(step, env, golden=golden)
            index += 1
            continue
        if isinstance(op, MeanOp):
            step = HostOp(
                name=op.name,
                kind="mean",
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attrs={"dim": list(op.dim), "keepdim": op.keepdim},
            )
            execute_host_op(step, env, golden=golden)
            index += 1
            continue
        if isinstance(op, MaxPool2dOp):
            step = HostOp(
                name=op.name,
                kind="maxpool2d",
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attrs={
                    "kernel_size": tuple(op.kernel_size),
                    "stride": tuple(op.stride),
                    "padding": tuple(op.padding),
                },
            )
            execute_host_op(step, env, golden=golden)
            index += 1
            continue
        if isinstance(op, AvgPool2dOp):
            step = HostOp(
                name=op.name,
                kind="avgpool2d",
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attrs={
                    "kernel_size": tuple(op.kernel_size),
                    "stride": tuple(op.stride),
                    "padding": tuple(op.padding),
                    "count_include_pad": op.count_include_pad,
                },
            )
            execute_host_op(step, env, golden=golden)
            index += 1
            continue
        if isinstance(op, AdaptiveAvgPool2dOp):
            step = HostOp(
                name=op.name,
                kind="adaptive_avg_pool2d",
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attrs={"output_size": tuple(op.output_size)},
            )
            execute_host_op(step, env, golden=golden)
            index += 1
            continue
        if isinstance(op, BinaryOp):
            step = HostOp(name=op.name, kind=op.kind, inputs=list(op.inputs), outputs=list(op.outputs))
            execute_host_op(step, env, golden=golden)
            index += 1
            continue
        if isinstance(op, LinearOp):
            step_inputs = [op.inputs[0], f"{op.name}_weight"]
            env[f"{op.name}_weight"] = np.array(op.weight, copy=True)
            if op.bias is not None:
                step_inputs.append(f"{op.name}_bias")
                env[f"{op.name}_bias"] = np.array(op.bias, copy=True)
            step = HostOp(name=op.name, kind="linear", inputs=step_inputs, outputs=list(op.outputs))
            execute_host_op(step, env, golden=golden)
            index += 1
            continue
        if isinstance(op, Conv2dOp):
            step_inputs = [op.inputs[0], f"{op.name}_weight"]
            env[f"{op.name}_weight"] = np.array(op.weight, copy=True)
            if op.bias is not None:
                step_inputs.append(f"{op.name}_bias")
                env[f"{op.name}_bias"] = np.array(op.bias, copy=True)
            step = HostOp(
                name=op.name,
                kind="conv2d",
                inputs=step_inputs,
                outputs=list(op.outputs),
                attrs={
                    "stride": int(op.stride),
                    "padding": int(op.padding),
                    "kernel_size": int(op.kernel_size),
                    "in_channels": int(op.in_channels),
                    "out_channels": int(op.out_channels),
                },
            )
            execute_host_op(step, env, golden=golden)
            index += 1
            continue
        if isinstance(op, CompilerReadyLinearOp):
            output_name, activation, consumed = _maybe_fused_activation(graph, index)
            if not supports_fused_activation(
                activation,
                shift=op.shift,
                h_gelu_x_scale_shift=op.h_gelu_x_scale_shift,
            ):
                output_name, activation, consumed = op.outputs[0], None, 1
            weight_t = np.array(op.weight_int, copy=False).T
            lhs_value, original_shape, layout = _normalize_linear_input(env[op.inputs[0]], in_features=int(op.weight_int.shape[1]))
            matrix = golden.matmul(
                _dtype_storage_array(lhs_value, op.in_dtype),
                _dtype_storage_array(weight_t, op.in_dtype),
                bias=op.bias_int32,
                multiplier=op.multiplier,
                shift=op.shift,
                activation="h_gelu" if activation == "gelu" else (activation or "none"),
                h_gelu_x_scale_shift=op.h_gelu_x_scale_shift,
                out_dtype=op.out_dtype,
            )
            env[output_name] = _restore_linear_output_layout(matrix, layout, original_shape)
            index += consumed
            continue
        if isinstance(op, CompilerReadyConv2dOp):
            output_name, activation, consumed = _maybe_fused_activation(graph, index)
            if not supports_fused_activation(
                activation,
                shift=op.shift,
                h_gelu_x_scale_shift=op.h_gelu_x_scale_shift,
            ):
                output_name, activation, consumed = op.outputs[0], None, 1
            image_hwc, original_shape, layout = _normalize_conv_input(env[op.inputs[0]], in_channels=op.in_channels)
            im2col_env = {op.inputs[0]: env[op.inputs[0]]}
            execute_host_op(
                HostOp(
                    name=f"{op.name}_im2col",
                    kind="im2col",
                    inputs=[op.inputs[0]],
                    outputs=[f"{op.name}_im2col"],
                    attrs={
                        "kernel_size": int(op.kernel_size),
                        "stride": int(op.stride),
                        "padding": int(op.padding),
                        "input_layout": layout,
                        "input_channels": int(op.in_channels),
                    },
                ),
                im2col_env,
                golden=golden,
            )
            cols = _dtype_storage_array(im2col_env[f"{op.name}_im2col"], op.in_dtype)
            kernel_t = np.array(op.weight_int, copy=False).reshape(op.out_channels, -1).T
            matrix = golden.matmul(
                cols,
                _dtype_storage_array(kernel_t, op.in_dtype),
                bias=op.bias_int32,
                multiplier=op.multiplier,
                shift=op.shift,
                activation="h_gelu" if activation == "gelu" else (activation or "none"),
                h_gelu_x_scale_shift=op.h_gelu_x_scale_shift,
                out_dtype=op.out_dtype,
            )
            out_h = ((image_hwc.shape[0] + (2 * op.padding) - op.kernel_size) // op.stride) + 1
            out_w = ((image_hwc.shape[1] + (2 * op.padding) - op.kernel_size) // op.stride) + 1
            env[output_name] = _restore_conv_output_layout(matrix.reshape(out_h, out_w, op.out_channels), layout, original_shape)
            index += consumed
            continue
        if isinstance(op, ActivationOp):
            step = HostOp(name=op.name, kind=op.kind, inputs=list(op.inputs), outputs=list(op.outputs))
            execute_host_op(step, env, golden=golden)
            index += 1
            continue
        raise NotImplementedError(f"Unsupported semantic execution op {type(op).__name__}.")

    return env


def lower_semantic_graph_to_plan(graph: SemanticGraph, materialized_values: dict[str, np.ndarray]):
    builder = IRBuilder(metadata={"frontend": "semantic"})
    expected_tensors: dict[str, np.ndarray] = {}

    for value in graph.values.values():
        if value.kind == "input":
            builder.add_tensor(TensorSpec(value.name, value.shape, value.dtype, TensorKind.INPUT, metadata=dict(value.metadata)))
            builder.add_input(value.name)
        elif value.kind == "constant":
            builder.add_tensor(TensorSpec(value.name, value.shape, value.dtype, TensorKind.CONSTANT, data=np.array(value.data, copy=True), metadata=dict(value.metadata)))

    index = 0
    while index < len(graph.ops):
        op = graph.ops[index]
        if isinstance(op, VerifyOp):
            builder.add_output(op.inputs[0])
            builder.verify(op.inputs[0], op.label, is_final_output=op.is_final_output)
            expected_tensors[op.inputs[0]] = np.array(materialized_values[op.inputs[0]], copy=True)
            index += 1
            continue
        if isinstance(op, QuantizeOp):
            output_value = graph.values[op.outputs[0]]
            builder.add_tensor(
                TensorSpec(
                    op.outputs[0],
                    output_value.shape,
                    output_value.dtype,
                    TensorKind.INTERMEDIATE,
                    metadata={
                        "quantization": {
                            "scale": op.scale,
                            "zero_point": op.zero_point,
                            "dtype": op.dtype.value,
                        }
                    },
                )
            )
            builder.host(
                op.name,
                "quantize",
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attrs={"scale": op.scale, "zero_point": op.zero_point, "dtype": op.dtype},
            )
            expected_tensors[op.outputs[0]] = np.array(materialized_values[op.outputs[0]], copy=True)
            index += 1
            continue
        if isinstance(op, DequantizeOp):
            output_value = graph.values[op.outputs[0]]
            output_encoding = str(op.output_encoding)
            output_dtype = DType.INT16 if output_encoding == "fp16_bits" else DType.FLOAT32
            metadata = {"value_encoding": "fp16_bits"} if output_encoding == "fp16_bits" else {}
            builder.add_tensor(TensorSpec(op.outputs[0], output_value.shape, output_dtype, TensorKind.INTERMEDIATE, metadata=metadata))
            attrs = {"scale": op.scale, "zero_point": op.zero_point}
            if output_encoding == "fp16_bits":
                attrs["output_encoding"] = "fp16_bits"
            builder.host(
                op.name,
                "dequantize",
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attrs=attrs,
            )
            expected_tensors[op.outputs[0]] = np.array(materialized_values[op.outputs[0]], copy=True)
            index += 1
            continue
        if isinstance(op, ReshapeOp):
            output_value = graph.values[op.outputs[0]]
            builder.add_tensor(TensorSpec(op.outputs[0], output_value.shape, output_value.dtype, TensorKind.INTERMEDIATE))
            builder.host(op.name, "reshape", inputs=list(op.inputs), outputs=list(op.outputs), attrs={"shape": tuple(op.shape)})
            expected_tensors[op.outputs[0]] = np.array(materialized_values[op.outputs[0]], copy=True)
            index += 1
            continue
        if isinstance(op, MeanOp):
            output_value = graph.values[op.outputs[0]]
            builder.add_tensor(TensorSpec(op.outputs[0], output_value.shape, output_value.dtype, TensorKind.INTERMEDIATE))
            builder.host(
                op.name,
                "mean",
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attrs={"dim": list(op.dim), "keepdim": op.keepdim},
            )
            expected_tensors[op.outputs[0]] = np.array(materialized_values[op.outputs[0]], copy=True)
            index += 1
            continue
        if isinstance(op, MaxPool2dOp):
            output_value = graph.values[op.outputs[0]]
            builder.add_tensor(TensorSpec(op.outputs[0], output_value.shape, output_value.dtype, TensorKind.INTERMEDIATE))
            builder.host(
                op.name,
                "maxpool2d",
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attrs={
                    "kernel_size": tuple(op.kernel_size),
                    "stride": tuple(op.stride),
                    "padding": tuple(op.padding),
                },
            )
            expected_tensors[op.outputs[0]] = np.array(materialized_values[op.outputs[0]], copy=True)
            index += 1
            continue
        if isinstance(op, AvgPool2dOp):
            output_value = graph.values[op.outputs[0]]
            builder.add_tensor(TensorSpec(op.outputs[0], output_value.shape, output_value.dtype, TensorKind.INTERMEDIATE))
            builder.host(
                op.name,
                "avgpool2d",
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attrs={
                    "kernel_size": tuple(op.kernel_size),
                    "stride": tuple(op.stride),
                    "padding": tuple(op.padding),
                    "count_include_pad": op.count_include_pad,
                },
            )
            expected_tensors[op.outputs[0]] = np.array(materialized_values[op.outputs[0]], copy=True)
            index += 1
            continue
        if isinstance(op, AdaptiveAvgPool2dOp):
            output_value = graph.values[op.outputs[0]]
            builder.add_tensor(TensorSpec(op.outputs[0], output_value.shape, output_value.dtype, TensorKind.INTERMEDIATE))
            builder.host(
                op.name,
                "adaptive_avg_pool2d",
                inputs=list(op.inputs),
                outputs=list(op.outputs),
                attrs={"output_size": tuple(op.output_size)},
            )
            expected_tensors[op.outputs[0]] = np.array(materialized_values[op.outputs[0]], copy=True)
            index += 1
            continue
        if isinstance(op, BinaryOp):
            output_value = graph.values[op.outputs[0]]
            builder.add_tensor(TensorSpec(op.outputs[0], output_value.shape, output_value.dtype, TensorKind.INTERMEDIATE))
            builder.host(op.name, op.kind, inputs=list(op.inputs), outputs=list(op.outputs))
            expected_tensors[op.outputs[0]] = np.array(materialized_values[op.outputs[0]], copy=True)
            index += 1
            continue
        if isinstance(op, LinearOp):
            output_value = graph.values[op.outputs[0]]
            weight_name = f"{op.name}_weight"
            builder.add_tensor(TensorSpec(weight_name, normalize_shape(op.weight.shape), DType.FLOAT32, TensorKind.CONSTANT, data=np.array(op.weight, copy=True)))
            step_inputs = [op.inputs[0], weight_name]
            if op.bias is not None:
                bias_name = f"{op.name}_bias"
                builder.add_tensor(TensorSpec(bias_name, normalize_shape(op.bias.shape), DType.FLOAT32, TensorKind.CONSTANT, data=np.array(op.bias, copy=True)))
                step_inputs.append(bias_name)
            builder.add_tensor(TensorSpec(op.outputs[0], output_value.shape, DType.FLOAT32, TensorKind.INTERMEDIATE))
            builder.host(op.name, "linear", inputs=step_inputs, outputs=list(op.outputs))
            expected_tensors[op.outputs[0]] = np.array(materialized_values[op.outputs[0]], copy=True)
            index += 1
            continue
        if isinstance(op, Conv2dOp):
            output_value = graph.values[op.outputs[0]]
            weight_name = f"{op.name}_weight"
            builder.add_tensor(TensorSpec(weight_name, normalize_shape(op.weight.shape), DType.FLOAT32, TensorKind.CONSTANT, data=np.array(op.weight, copy=True)))
            step_inputs = [op.inputs[0], weight_name]
            if op.bias is not None:
                bias_name = f"{op.name}_bias"
                builder.add_tensor(TensorSpec(bias_name, normalize_shape(op.bias.shape), DType.FLOAT32, TensorKind.CONSTANT, data=np.array(op.bias, copy=True)))
                step_inputs.append(bias_name)
            builder.add_tensor(TensorSpec(op.outputs[0], output_value.shape, DType.FLOAT32, TensorKind.INTERMEDIATE))
            builder.host(
                op.name,
                "conv2d",
                inputs=step_inputs,
                outputs=list(op.outputs),
                attrs={
                    "stride": int(op.stride),
                    "padding": int(op.padding),
                    "kernel_size": int(op.kernel_size),
                    "in_channels": int(op.in_channels),
                    "out_channels": int(op.out_channels),
                },
            )
            expected_tensors[op.outputs[0]] = np.array(materialized_values[op.outputs[0]], copy=True)
            index += 1
            continue
        if isinstance(op, CompilerReadyLinearOp):
            output_name, activation, consumed = _maybe_fused_activation(graph, index)
            if not supports_fused_activation(
                activation,
                shift=op.shift,
                h_gelu_x_scale_shift=op.h_gelu_x_scale_shift,
            ):
                output_name, activation, consumed = op.outputs[0], None, 1
            output_value = graph.values[output_name]
            weight_t_name = f"{op.name}_weight_t"
            weight_t = np.array(op.weight_int, copy=False).T.astype(np.int16)
            builder.add_tensor(TensorSpec(weight_t_name, normalize_shape(weight_t.shape), op.in_dtype, TensorKind.CONSTANT, data=weight_t))
            bias_name = None
            if op.bias_int32 is not None:
                bias_name = f"{op.name}_bias"
                builder.add_tensor(TensorSpec(bias_name, normalize_shape(op.bias_int32.shape), DType.INT32, TensorKind.CONSTANT, data=np.array(op.bias_int32, copy=True)))

            source_value = materialized_values[op.inputs[0]]
            lhs_value, original_shape, layout = _normalize_linear_input(source_value, in_features=int(op.weight_int.shape[1]))
            lhs_name = op.inputs[0]
            if layout != "row_batch1":
                lhs_name = f"{op.name}_input_row"
                transform_kind = "reshape" if layout == "vector" else "transpose"
                transform_attrs = {"shape": tuple(lhs_value.shape)} if layout == "vector" else {"axes": (1, 0)}
                builder.add_tensor(TensorSpec(lhs_name, normalize_shape(lhs_value.shape), graph.values[op.inputs[0]].dtype, TensorKind.INTERMEDIATE, metadata=dict(graph.values[op.inputs[0]].metadata)))
                builder.host(lhs_name, transform_kind, inputs=[op.inputs[0]], outputs=[lhs_name], attrs=transform_attrs)
                expected_tensors[lhs_name] = np.array(lhs_value, copy=True)

            internal_out_name = output_name if layout == "row_batch1" else f"{op.name}_matmul"
            if layout == "row_batch1":
                internal_shape = output_value.shape
                internal_expected = np.array(materialized_values[output_name], copy=True)
            else:
                internal_shape = normalize_shape((lhs_value.shape[0], int(op.weight_int.shape[0])))
                if layout == "vector":
                    internal_expected = np.array(materialized_values[output_name], copy=True).reshape(1, -1)
                else:
                    internal_expected = np.array(materialized_values[output_name], copy=True).T
            builder.add_tensor(
                TensorSpec(
                    internal_out_name,
                    internal_shape,
                    op.out_dtype,
                    TensorKind.INTERMEDIATE,
                    metadata={
                        "quantization": {
                            "scale": op.output_scale,
                            "zero_point": 0,
                            "dtype": op.out_dtype.value,
                        },
                        "h_gelu_x_scale_shift": op.h_gelu_x_scale_shift,
                    },
                )
            )
            builder.segment(
                op.name,
                ops=[
                    builder.matmul(
                        internal_out_name,
                        lhs_name,
                        weight_t_name,
                        internal_out_name,
                        bias=bias_name,
                        multiplier=op.multiplier,
                        shift=op.shift,
                        activation="h_gelu" if activation == "gelu" else (activation or "none"),
                        h_gelu_x_scale_shift=op.h_gelu_x_scale_shift,
                        in_dtype=op.in_dtype,
                        out_dtype=op.out_dtype,
                    )
                ],
                inputs=[lhs_name, weight_t_name] + ([bias_name] if bias_name else []),
                outputs=[internal_out_name],
            )
            expected_tensors[internal_out_name] = internal_expected
            if layout != "row_batch1":
                transform_kind = "transpose" if layout == "column" else "reshape"
                transform_attrs = {"axes": (1, 0)} if layout == "column" else {"shape": output_value.shape}
                builder.add_tensor(
                    TensorSpec(
                        output_name,
                        output_value.shape,
                        output_value.dtype,
                        TensorKind.INTERMEDIATE,
                        metadata={
                            "quantization": {
                                "scale": op.output_scale,
                                "zero_point": 0,
                                "dtype": op.out_dtype.value,
                            }
                        },
                    )
                )
                builder.host(output_name, transform_kind, inputs=[internal_out_name], outputs=[output_name], attrs=transform_attrs)
            else:
                if output_name != internal_out_name:
                    builder.add_tensor(TensorSpec(output_name, output_value.shape, output_value.dtype, TensorKind.INTERMEDIATE))
                    builder.host(output_name, "alias", inputs=[internal_out_name], outputs=[output_name])
            expected_tensors[output_name] = np.array(materialized_values[output_name], copy=True)
            index += consumed
            continue
        if isinstance(op, CompilerReadyConv2dOp):
            output_name, activation, consumed = _maybe_fused_activation(graph, index)
            if not supports_fused_activation(
                activation,
                shift=op.shift,
                h_gelu_x_scale_shift=op.h_gelu_x_scale_shift,
            ):
                output_name, activation, consumed = op.outputs[0], None, 1
            output_value = graph.values[output_name]
            cols_name = f"{op.name}_im2col"
            matmul_name = f"{op.name}_matmul"
            kernel_t_name = f"{op.name}_kernel_t"
            source_value = materialized_values[op.inputs[0]]
            image_hwc, original_shape, layout = _normalize_conv_input(source_value, in_channels=op.in_channels)
            cols_env = {op.inputs[0]: source_value}
            execute_host_op(
                HostOp(
                    name=cols_name,
                    kind="im2col",
                    inputs=[op.inputs[0]],
                    outputs=[cols_name],
                    attrs={
                        "kernel_size": int(op.kernel_size),
                        "stride": int(op.stride),
                        "padding": int(op.padding),
                        "input_layout": layout,
                        "input_channels": int(op.in_channels),
                    },
                ),
                cols_env,
                golden=GoldenModel(),
            )
            kernel_t = np.array(op.weight_int, copy=False).reshape(op.out_channels, -1).T.astype(np.int16)
            builder.add_tensor(TensorSpec(cols_name, normalize_shape(cols_env[cols_name].shape), op.in_dtype, TensorKind.INTERMEDIATE))
            builder.host(
                cols_name,
                "im2col",
                inputs=[op.inputs[0]],
                outputs=[cols_name],
                attrs={
                    "kernel_size": int(op.kernel_size),
                    "stride": int(op.stride),
                    "padding": int(op.padding),
                    "input_layout": layout,
                    "input_channels": int(op.in_channels),
                },
            )
            expected_tensors[cols_name] = np.array(cols_env[cols_name], copy=True)
            builder.add_tensor(TensorSpec(kernel_t_name, normalize_shape(kernel_t.shape), op.in_dtype, TensorKind.CONSTANT, data=kernel_t))
            bias_name = None
            if op.bias_int32 is not None:
                bias_name = f"{op.name}_bias"
                builder.add_tensor(TensorSpec(bias_name, normalize_shape(op.bias_int32.shape), DType.INT32, TensorKind.CONSTANT, data=np.array(op.bias_int32, copy=True)))
            out_h = ((image_hwc.shape[0] + (2 * op.padding) - op.kernel_size) // op.stride) + 1
            out_w = ((image_hwc.shape[1] + (2 * op.padding) - op.kernel_size) // op.stride) + 1
            builder.add_tensor(
                TensorSpec(
                    matmul_name,
                    normalize_shape((out_h * out_w, op.out_channels)),
                    op.out_dtype,
                    TensorKind.INTERMEDIATE,
                    metadata={
                        "quantization": {
                            "scale": op.output_scale,
                            "zero_point": 0,
                            "dtype": op.out_dtype.value,
                        },
                        "h_gelu_x_scale_shift": op.h_gelu_x_scale_shift,
                    },
                )
            )
            builder.segment(
                op.name,
                ops=[
                    builder.matmul(
                        matmul_name,
                        cols_name,
                        kernel_t_name,
                        matmul_name,
                        bias=bias_name,
                        multiplier=op.multiplier,
                        shift=op.shift,
                        activation="h_gelu" if activation == "gelu" else (activation or "none"),
                        h_gelu_x_scale_shift=op.h_gelu_x_scale_shift,
                        in_dtype=op.in_dtype,
                        out_dtype=op.out_dtype,
                    )
                ],
                inputs=[cols_name, kernel_t_name] + ([bias_name] if bias_name else []),
                outputs=[matmul_name],
            )
            matrix_value = materialized_values[output_name]
            if layout == "chw":
                matmul_expected = np.transpose(matrix_value[0] if matrix_value.ndim == 4 else matrix_value, (1, 2, 0)).reshape(out_h * out_w, op.out_channels)
            else:
                matmul_expected = matrix_value.reshape(out_h * out_w, op.out_channels)
            expected_tensors[matmul_name] = np.array(matmul_expected, copy=True)
            builder.add_tensor(
                TensorSpec(
                    output_name,
                    output_value.shape,
                    output_value.dtype,
                    TensorKind.INTERMEDIATE,
                    metadata={
                        "quantization": {
                            "scale": op.output_scale,
                            "zero_point": 0,
                            "dtype": op.out_dtype.value,
                        }
                    },
                )
            )
            builder.host(
                f"{op.name}_layout_restore",
                "layout_restore",
                inputs=[matmul_name],
                outputs=[output_name],
                attrs={
                    "layout": layout,
                    "original_shape": original_shape,
                    "out_h": out_h,
                    "out_w": out_w,
                    "out_channels": int(op.out_channels),
                },
            )
            expected_tensors[output_name] = np.array(materialized_values[output_name], copy=True)
            index += consumed
            continue
        if isinstance(op, ActivationOp):
            output_value = graph.values[op.outputs[0]]
            builder.add_tensor(TensorSpec(op.outputs[0], output_value.shape, output_value.dtype, TensorKind.INTERMEDIATE))
            builder.host(op.name, op.kind, inputs=list(op.inputs), outputs=list(op.outputs))
            expected_tensors[op.outputs[0]] = np.array(materialized_values[op.outputs[0]], copy=True)
            index += 1
            continue
        raise NotImplementedError(f"Unsupported semantic lowering op {type(op).__name__}.")

    plan = builder.finalize(inputs=list(graph.inputs), outputs=list(graph.outputs), metadata={"frontend": "semantic"})
    return plan, expected_tensors


def partition_fx_graph_semantic(graph_module: Any, example_inputs: tuple[Any, ...], **_: Any):
    from .semantic_frontend import build_semantic_graph

    graph = build_semantic_graph(graph_module, example_inputs)
    input_map = {
        name: np.array(example.detach().cpu().numpy() if hasattr(example, "detach") else example, copy=True)
        for name, example in zip(graph.inputs, example_inputs)
    }
    materialized_values = execute_semantic_graph(graph, input_map)
    return lower_semantic_graph_to_plan(graph, materialized_values)
