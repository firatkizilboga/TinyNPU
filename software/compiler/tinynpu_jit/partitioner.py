from __future__ import annotations

from dataclasses import dataclass
import operator
from typing import Any

import numpy as np

from .golden import GoldenModel
from .ir import DType, ExecutionPlan, HostOp, MatMulOp, NpuSegment, TensorKind, TensorSpec, VerifyTensor, normalize_shape
from .quantization import synthesize_rescale


@dataclass
class PartitionState:
    tensors: dict[str, TensorSpec]
    steps: list[Any]
    inputs: list[str]
    outputs: list[str]
    expected_tensors: dict[str, np.ndarray]


def partition_fx_graph(graph_module: Any, example_inputs: tuple[Any, ...], verify_policy: str | None = None, **_: Any):
    try:
        import torch
        import torch.nn as nn
        try:
            from torch.ao.nn.quantized import DeQuantize as QDeQuantize, Quantize as QQuantize
        except Exception:
            QQuantize = QDeQuantize = ()
        try:
            from torch.ao.nn.quantized import Conv2d as QConv2d, Linear as QLinear
        except Exception:
            QLinear = QConv2d = ()
        try:
            from torch.ao.quantization import DeQuantStub, QuantStub
        except Exception:
            try:
                from torch.quantization import DeQuantStub, QuantStub
            except Exception:
                QuantStub = DeQuantStub = ()
        try:
            from software.compiler.tinynpu_quant import (
                CompilerDequantize,
                CompilerQuantize,
                CompilerReadyConv2d,
                CompilerReadyLinear,
            )
        except Exception:
            CompilerQuantize = CompilerDequantize = CompilerReadyLinear = CompilerReadyConv2d = ()
    except Exception as exc:
        raise ImportError("torch is required for FX partitioning.") from exc

    tensors: dict[str, TensorSpec] = {}
    steps: list[Any] = []
    expected_tensors: dict[str, np.ndarray] = {}
    inputs: list[str] = []
    outputs: list[str] = []
    current_ops: list[MatMulOp] = []
    current_inputs: set[str] = set()

    env: dict[str, np.ndarray] = {}
    example_iter = iter(example_inputs)
    golden = GoldenModel()

    def to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.array(value)

    def parse_dtype(name: str) -> DType:
        normalized = str(name).lower()
        mapping = {
            "int4": DType.INT4,
            "int8": DType.INT8,
            "int16": DType.INT16,
            "int32": DType.INT32,
            "float32": DType.FLOAT32,
        }
        if normalized not in mapping:
            raise ValueError(f"Unsupported dtype annotation {name!r}.")
        return mapping[normalized]

    def parse_torch_quant_dtype(dtype: Any) -> DType:
        if dtype in {getattr(torch, "qint8", None), getattr(torch, "int8", None)}:
            return DType.INT8
        if dtype in {getattr(torch, "quint8", None), getattr(torch, "uint8", None)}:
            raise NotImplementedError(
                "TinyNPU PyTorch lowering currently supports signed qint8 activations only. "
                "quint8/uint8 activation tensors need an explicit recentering path that is not implemented yet."
            )
        if dtype in {getattr(torch, "qint32", None), getattr(torch, "int32", None)}:
            return DType.INT32
        raise ValueError(
            f"Unsupported torch quantized dtype {dtype!r}. "
            "Current TinyNPU quant-stub support accepts qint8 and qint32 only."
        )

    def infer_dtype(value: np.ndarray) -> DType:
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

    def coerce_npu_tensor(name: str) -> np.ndarray:
        coerced = golden.coerce_npu_input(env[name], out_dtype=DType.INT16, tensor_name=name)
        env[name] = coerced
        if name in tensors and tensors[name].dtype == DType.FLOAT32:
            tensors[name].dtype = DType.INT16
            if tensors[name].data is not None:
                tensors[name].data = np.array(coerced, copy=True)
        return coerced

    def require_symmetric_quantization(qparams: dict[str, Any], *, tensor_name: str) -> None:
        zero_point = int(qparams.get("zero_point", 0))
        if zero_point != 0:
            raise NotImplementedError(
                f"Tensor {tensor_name!r} uses zero_point={zero_point}. "
                "TinyNPU NPU segments currently require symmetric zero_point=0 quantization."
            )

    def quant_metadata(name: str) -> dict[str, Any]:
        quant = tensors[name].metadata.get("quantization")
        if quant is None:
            raise NotImplementedError(
                f"Tensor {name!r} does not carry quantization metadata. "
                "Quantized Linear/Conv2d lowering requires an explicit Quantize boundary or prior quantized op."
            )
        require_symmetric_quantization(quant, tensor_name=name)
        return quant

    def set_tensor_dtype(name: str, dtype: DType, value: np.ndarray) -> None:
        if name in tensors:
            tensors[name].dtype = dtype
            if tensors[name].data is not None:
                tensors[name].data = np.array(value, copy=True)

    def quantized_weight_params(module_name: str, weight_qtensor: Any) -> tuple[np.ndarray, DType, float]:
        qscheme = weight_qtensor.qscheme()
        if qscheme not in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
            raise NotImplementedError(
                f"Module {module_name!r} uses unsupported weight qscheme {qscheme}. "
                "Current TinyNPU lowering supports per-tensor qint8 weights only."
            )
        w_dtype = parse_torch_quant_dtype(weight_qtensor.dtype)
        if w_dtype != DType.INT8:
            raise NotImplementedError(
                f"Module {module_name!r} uses unsupported weight dtype {weight_qtensor.dtype!r}. "
                "Current TinyNPU lowering supports qint8 weights only."
            )
        if int(weight_qtensor.q_zero_point()) != 0:
            raise NotImplementedError(
                f"Module {module_name!r} uses weight zero_point={int(weight_qtensor.q_zero_point())}. "
                "TinyNPU NPU segments currently require symmetric zero_point=0 weights."
            )
        return (
            weight_qtensor.int_repr().detach().cpu().numpy().astype(np.int16),
            w_dtype,
            float(weight_qtensor.q_scale()),
        )

    def bias_to_int32(module_name: str, bias_tensor: Any, *, input_scale: float, weight_scale: float) -> np.ndarray | None:
        if bias_tensor is None:
            return None
        bias_scale = input_scale * weight_scale
        if bias_scale <= 0:
            raise ValueError(f"Module {module_name!r} has non-positive bias scale {bias_scale}.")
        bias_fp = to_numpy(bias_tensor).astype(np.float32).reshape(1, -1)
        return np.rint(bias_fp / np.float32(bias_scale)).astype(np.int32)

    def synthesize_qparams(
        module_name: str,
        *,
        input_quant: dict[str, Any],
        weight_scale: float,
        output_scale: float,
    ) -> tuple[int, int]:
        input_scale = float(input_quant["scale"])
        effective_scale = (input_scale * float(weight_scale)) / float(output_scale)
        rescale = synthesize_rescale(effective_scale)
        return rescale.multiplier, rescale.shift

    def lower_quantized_linear(module_name: str, module: Any, source_name: str, out_name: str) -> None:
        input_quant = quant_metadata(source_name)
        input_dtype = parse_dtype(input_quant["dtype"])
        weight_int, weight_dtype, weight_scale = quantized_weight_params(module_name, module.weight())
        if input_dtype != weight_dtype:
            raise NotImplementedError(
                f"Module {module_name!r} uses activation dtype {input_dtype.value} and weight dtype {weight_dtype.value}. "
                "Current TinyNPU lowering requires matching signed integer precisions."
            )
        output_scale = float(module.scale)
        output_zero_point = int(module.zero_point)
        if output_zero_point != 0:
            raise NotImplementedError(
                f"Module {module_name!r} uses output zero_point={output_zero_point}. "
                "TinyNPU NPU segments currently require symmetric zero_point=0 outputs."
            )
        multiplier, shift = synthesize_qparams(
            module_name,
            input_quant=input_quant,
            weight_scale=weight_scale,
            output_scale=output_scale,
        )
        bias_value = bias_to_int32(
            module_name,
            module.bias(),
            input_scale=float(input_quant["scale"]),
            weight_scale=weight_scale,
        )
        lhs_name = f"{out_name}_weight"
        bias_name = None
        tensors[lhs_name] = TensorSpec(
            lhs_name,
            normalize_shape(weight_int.shape),
            weight_dtype,
            TensorKind.CONSTANT,
            data=np.array(weight_int, copy=True),
        )
        current_inputs.add(lhs_name)
        if bias_value is not None:
            bias_name = f"{out_name}_bias"
            tensors[bias_name] = TensorSpec(
                bias_name,
                normalize_shape(bias_value.shape),
                DType.INT32,
                TensorKind.CONSTANT,
                data=np.array(bias_value, copy=True),
            )
            current_inputs.add(bias_name)
        rhs_value = golden.coerce_npu_input(env[source_name], out_dtype=input_dtype, tensor_name=source_name)
        env[source_name] = rhs_value
        set_tensor_dtype(source_name, input_dtype, rhs_value)
        expected = golden.matmul(
            weight_int,
            rhs_value,
            bias=bias_value,
            multiplier=multiplier,
            shift=shift,
            activation="none",
            out_dtype=DType.INT8,
        )
        env[out_name] = expected.astype(np.int32)
        tensors[out_name] = TensorSpec(
            out_name,
            normalize_shape(expected.shape),
            DType.INT8,
            TensorKind.INTERMEDIATE,
            metadata={
                "quantization": {
                    "scale": output_scale,
                    "zero_point": 0,
                    "dtype": DType.INT8.value,
                    "source": "quantized_linear",
                    "module_name": module_name,
                }
            },
        )
        current_ops.append(
            MatMulOp(
                name=out_name,
                lhs=lhs_name,
                rhs=source_name,
                out=out_name,
                bias=bias_name,
                multiplier=multiplier,
                shift=shift,
                activation="none",
                in_dtype=input_dtype,
                out_dtype=DType.INT8,
            )
        )

    def normalize_conv_input(module_name: str, value: np.ndarray, *, in_channels: int) -> tuple[np.ndarray, tuple[int, ...], str]:
        arr = np.array(value, copy=False)
        original_shape = normalize_shape(arr.shape)
        if arr.ndim == 4:
            if arr.shape[0] != 1:
                raise NotImplementedError(
                    f"Module {module_name!r} received batch size {arr.shape[0]}. "
                    "Current TinyNPU conv lowering supports batch size 1 only."
                )
            arr = arr[0]
        if arr.ndim != 3:
            raise NotImplementedError(
                f"Module {module_name!r} expects a 3D or 4D input tensor, got shape {original_shape}."
            )
        if arr.shape[0] == in_channels:
            return np.transpose(arr, (1, 2, 0)), original_shape, "chw"
        if arr.shape[-1] == in_channels:
            return arr, original_shape, "hwc"
        raise NotImplementedError(
            f"Module {module_name!r} could not infer conv input layout from shape {original_shape} "
            f"with in_channels={in_channels}."
        )

    def restore_conv_output_layout(value: np.ndarray, layout: str, original_shape: tuple[int, ...]) -> np.ndarray:
        if layout == "chw":
            chw = np.transpose(value, (2, 0, 1))
            if len(original_shape) == 4:
                return np.expand_dims(chw, axis=0)
            return chw
        if layout == "hwc":
            return value
        raise ValueError(f"Unsupported conv layout tag {layout!r}.")

    def normalize_linear_input(
        module_name: str,
        value: np.ndarray,
        *,
        in_features: int,
    ) -> tuple[np.ndarray, tuple[int, ...], str]:
        arr = np.array(value, copy=False)
        original_shape = normalize_shape(arr.shape)
        if arr.ndim == 1:
            if arr.shape[0] != in_features:
                raise NotImplementedError(
                    f"Module {module_name!r} expects {in_features} input features, got shape {original_shape}."
                )
            return arr.reshape(1, in_features), original_shape, "vector"
        if arr.ndim == 2:
            if arr.shape == (1, in_features):
                return arr, original_shape, "row_batch1"
            if arr.shape == (in_features, 1):
                return arr.T, original_shape, "column"
        raise NotImplementedError(
            f"Module {module_name!r} currently supports 1D vectors, (features,1), or batch-1 row vectors only. "
            f"Got shape {original_shape} for in_features={in_features}."
        )

    def restore_linear_output_layout(value: np.ndarray, layout: str, original_shape: tuple[int, ...]) -> np.ndarray:
        if layout == "vector":
            return value.reshape(value.shape[1])
        if layout == "row_batch1":
            return value
        if layout == "column":
            return value.T.reshape(value.shape[1], 1)
        raise ValueError(f"Unsupported linear layout tag {layout!r} for original shape {original_shape}.")

    def lower_quantized_conv2d(module_name: str, module: Any, source_name: str, out_name: str) -> None:
        input_quant = quant_metadata(source_name)
        input_dtype = parse_dtype(input_quant["dtype"])
        weight_qtensor = module.weight()
        weight_int, weight_dtype, weight_scale = quantized_weight_params(module_name, weight_qtensor)
        if input_dtype != weight_dtype:
            raise NotImplementedError(
                f"Module {module_name!r} uses activation dtype {input_dtype.value} and weight dtype {weight_dtype.value}. "
                "Current TinyNPU lowering requires matching signed integer precisions."
            )
        output_scale = float(module.scale)
        output_zero_point = int(module.zero_point)
        if output_zero_point != 0:
            raise NotImplementedError(
                f"Module {module_name!r} uses output zero_point={output_zero_point}. "
                "TinyNPU NPU segments currently require symmetric zero_point=0 outputs."
            )
        kernel_h, kernel_w = module.kernel_size
        if kernel_h != kernel_w:
            raise NotImplementedError(
                f"Module {module_name!r} uses kernel_size={module.kernel_size}. "
                "Current TinyNPU lowering supports square kernels only."
            )
        if module.dilation != (1, 1):
            raise NotImplementedError(
                f"Module {module_name!r} uses dilation={module.dilation}. "
                "Current TinyNPU lowering supports dilation=1 only."
            )
        if module.groups != 1:
            raise NotImplementedError(
                f"Module {module_name!r} uses groups={module.groups}. "
                "Current TinyNPU lowering supports groups=1 only."
            )
        image_hwc, original_shape, layout = normalize_conv_input(
            module_name,
            env[source_name],
            in_channels=int(module.in_channels),
        )
        cols_name = f"{out_name}_im2col"
        flush_segment()
        steps.append(
            HostOp(
                name=cols_name,
                kind="im2col",
                inputs=[source_name],
                outputs=[cols_name],
                attrs={
                    "kernel_size": int(kernel_h),
                    "stride": int(module.stride[0]),
                    "padding": int(module.padding[0]),
                    "input_layout": layout,
                    "input_channels": int(module.in_channels),
                },
            )
        )
        cols_value = golden.im2col(
            image_hwc,
            kernel_size=int(kernel_h),
            stride=int(module.stride[0]),
            padding=int(module.padding[0]),
        )
        env[cols_name] = cols_value
        tensors[cols_name] = TensorSpec(
            cols_name,
            normalize_shape(cols_value.shape),
            input_dtype,
            TensorKind.INTERMEDIATE,
        )
        current_inputs.update({cols_name})

        kernel_t_name = f"{out_name}_kernel_t"
        kernel_t = weight_int.reshape(weight_int.shape[0], -1).T
        tensors[kernel_t_name] = TensorSpec(
            kernel_t_name,
            normalize_shape(kernel_t.shape),
            weight_dtype,
            TensorKind.CONSTANT,
            data=np.array(kernel_t, copy=True),
        )
        current_inputs.add(kernel_t_name)

        bias_value = bias_to_int32(
            module_name,
            module.bias(),
            input_scale=float(input_quant["scale"]),
            weight_scale=weight_scale,
        )
        bias_name = None
        if bias_value is not None:
            bias_name = f"{out_name}_bias"
            tensors[bias_name] = TensorSpec(
                bias_name,
                normalize_shape(bias_value.shape),
                DType.INT32,
                TensorKind.CONSTANT,
                data=np.array(bias_value, copy=True),
            )
            current_inputs.add(bias_name)

        multiplier, shift = synthesize_qparams(
            module_name,
            input_quant=input_quant,
            weight_scale=weight_scale,
            output_scale=output_scale,
        )
        expected_cols = golden.coerce_npu_input(cols_value, out_dtype=input_dtype, tensor_name=cols_name)
        env[cols_name] = expected_cols
        expected_matrix = golden.matmul(
            expected_cols,
            kernel_t,
            bias=bias_value,
            multiplier=multiplier,
            shift=shift,
            activation="none",
            out_dtype=DType.INT8,
        )
        matmul_name = f"{out_name}_matmul"
        tensors[matmul_name] = TensorSpec(
            matmul_name,
            normalize_shape(expected_matrix.shape),
            DType.INT8,
            TensorKind.INTERMEDIATE,
            metadata={
                "quantization": {
                    "scale": output_scale,
                    "zero_point": 0,
                    "dtype": DType.INT8.value,
                    "source": "quantized_conv2d_matrix",
                    "module_name": module_name,
                }
            },
        )
        env[matmul_name] = expected_matrix.astype(np.int32)
        current_ops.append(
            MatMulOp(
                name=matmul_name,
                lhs=cols_name,
                rhs=kernel_t_name,
                out=matmul_name,
                bias=bias_name,
                multiplier=multiplier,
                shift=shift,
                activation="none",
                in_dtype=input_dtype,
                out_dtype=DType.INT8,
            )
        )
        flush_segment()

        out_h = ((image_hwc.shape[0] + (2 * int(module.padding[0])) - int(kernel_h)) // int(module.stride[0])) + 1
        out_w = ((image_hwc.shape[1] + (2 * int(module.padding[1])) - int(kernel_w)) // int(module.stride[1])) + 1
        hwc_out = expected_matrix.reshape(out_h, out_w, int(module.out_channels))
        final_value = restore_conv_output_layout(hwc_out, layout, original_shape)
        steps.append(
            HostOp(
                name=f"{out_name}_layout_restore",
                kind="layout_restore",
                inputs=[matmul_name],
                outputs=[out_name],
                attrs={
                    "layout": layout,
                    "original_shape": original_shape,
                    "out_h": out_h,
                    "out_w": out_w,
                    "out_channels": int(module.out_channels),
                },
            )
        )
        env[out_name] = final_value
        tensors[out_name] = TensorSpec(
            out_name,
            normalize_shape(final_value.shape),
            DType.INT8,
            TensorKind.INTERMEDIATE,
            metadata={
                "quantization": {
                    "scale": output_scale,
                    "zero_point": 0,
                    "dtype": DType.INT8.value,
                    "source": "quantized_conv2d",
                    "module_name": module_name,
                }
            },
        )
        expected_tensors[cols_name] = np.array(env[cols_name], copy=True)
        expected_tensors[matmul_name] = np.array(expected_matrix, copy=True)
        expected_tensors[out_name] = np.array(final_value, copy=True)

    def lower_compiler_ready_linear(module_name: str, module: Any, source_name: str, out_name: str) -> None:
        input_quant = quant_metadata(source_name)
        input_dtype = parse_dtype(input_quant["dtype"])
        module_in_dtype = parse_dtype(module.in_dtype)
        module_out_dtype = parse_dtype(module.out_dtype)
        if input_dtype != module_in_dtype:
            raise NotImplementedError(
                f"Module {module_name!r} expects activation dtype {module_in_dtype.value}, "
                f"but source tensor {source_name!r} is {input_dtype.value}."
            )

        weight_int = to_numpy(module.weight_int).astype(np.int16)
        weight_t = weight_int.T
        weight_dtype = parse_dtype(module.in_dtype)
        if module_in_dtype != weight_dtype:
            raise NotImplementedError(
                f"Module {module_name!r} uses activation dtype {module_in_dtype.value} and weight dtype {weight_dtype.value}. "
                "TinyNPU matmul currently requires matching input/weight precision."
            )
        output_zero_point = 0
        multiplier = int(module.multiplier)
        shift = int(module.shift)
        bias_name = None
        rhs_name = f"{out_name}_weight_t"
        tensors[rhs_name] = TensorSpec(
            rhs_name,
            normalize_shape(weight_t.shape),
            weight_dtype,
            TensorKind.CONSTANT,
            data=np.array(weight_t, copy=True),
        )
        current_inputs.add(rhs_name)

        bias_tensor = getattr(module, "bias_int32", None)
        bias_value = None
        if bias_tensor is not None:
            bias_value = to_numpy(bias_tensor).astype(np.int32).reshape(1, -1)
            bias_name = f"{out_name}_bias"
            tensors[bias_name] = TensorSpec(
                bias_name,
                normalize_shape(bias_value.shape),
                DType.INT32,
                TensorKind.CONSTANT,
                data=np.array(bias_value, copy=True),
            )
            current_inputs.add(bias_name)

        lhs_value, original_shape, layout = normalize_linear_input(
            module_name,
            env[source_name],
            in_features=int(weight_int.shape[1]),
        )
        lhs_name = source_name
        if layout != "row_batch1":
            flush_segment()
            lhs_name = f"{out_name}_input_row"
            transform_kind = "reshape" if layout == "vector" else "transpose"
            transform_attrs = {"shape": tuple(lhs_value.shape)} if layout == "vector" else {"axes": (1, 0)}
            steps.append(
                HostOp(
                    name=lhs_name,
                    kind=transform_kind,
                    inputs=[source_name],
                    outputs=[lhs_name],
                    attrs=transform_attrs,
                )
            )
            tensors[lhs_name] = TensorSpec(
                lhs_name,
                normalize_shape(lhs_value.shape),
                input_dtype,
                TensorKind.INTERMEDIATE,
                metadata=dict(tensors[source_name].metadata),
            )
            expected_tensors[lhs_name] = np.array(lhs_value, copy=True)
            current_inputs.add(lhs_name)
        else:
            current_inputs.add(source_name)
        env[lhs_name] = golden.coerce_npu_input(lhs_value, out_dtype=input_dtype, tensor_name=lhs_name)
        set_tensor_dtype(lhs_name, input_dtype, env[lhs_name])

        expected_matrix = golden.matmul(
            env[lhs_name],
            weight_t,
            bias=bias_value,
            multiplier=multiplier,
            shift=shift,
            activation="none",
            out_dtype=module_out_dtype,
        )
        internal_out_name = out_name if layout == "row_batch1" else f"{out_name}_matmul"
        env[internal_out_name] = expected_matrix.astype(np.int32)
        tensors[internal_out_name] = TensorSpec(
            internal_out_name,
            normalize_shape(expected_matrix.shape),
            module_out_dtype,
            TensorKind.INTERMEDIATE,
            metadata={
                "quantization": {
                    "scale": float(module.output_scale),
                    "zero_point": output_zero_point,
                    "dtype": module_out_dtype.value,
                    "source": "compiler_ready_linear",
                    "module_name": module_name,
                }
            },
        )
        current_ops.append(
            MatMulOp(
                name=internal_out_name,
                lhs=lhs_name,
                rhs=rhs_name,
                out=internal_out_name,
                bias=bias_name,
                multiplier=multiplier,
                shift=shift,
                activation="none",
                in_dtype=module_in_dtype,
                out_dtype=module_out_dtype,
            )
        )
        if layout != "row_batch1":
            flush_segment()
            restored = restore_linear_output_layout(expected_matrix, layout, original_shape)
            if layout == "column":
                steps.append(
                    HostOp(
                        name=out_name,
                        kind="transpose",
                        inputs=[internal_out_name],
                        outputs=[out_name],
                        attrs={"axes": (1, 0)},
                    )
                )
            else:
                steps.append(
                    HostOp(
                        name=out_name,
                        kind="reshape",
                        inputs=[internal_out_name],
                        outputs=[out_name],
                        attrs={"shape": tuple(restored.shape)},
                    )
                )
            env[out_name] = restored
            tensors[out_name] = TensorSpec(
                out_name,
                normalize_shape(restored.shape),
                module_out_dtype,
                TensorKind.INTERMEDIATE,
                metadata=dict(tensors[internal_out_name].metadata),
            )
            expected_tensors[internal_out_name] = np.array(expected_matrix, copy=True)
            expected_tensors[out_name] = np.array(restored, copy=True)

    def lower_compiler_ready_conv2d(module_name: str, module: Any, source_name: str, out_name: str) -> None:
        input_quant = quant_metadata(source_name)
        input_dtype = parse_dtype(input_quant["dtype"])
        module_in_dtype = parse_dtype(module.in_dtype)
        module_out_dtype = parse_dtype(module.out_dtype)
        if input_dtype != module_in_dtype:
            raise NotImplementedError(
                f"Module {module_name!r} expects activation dtype {module_in_dtype.value}, "
                f"but source tensor {source_name!r} is {input_dtype.value}."
            )

        image_hwc, original_shape, layout = normalize_conv_input(
            module_name,
            env[source_name],
            in_channels=int(module.in_channels),
        )
        cols_name = f"{out_name}_im2col"
        flush_segment()
        steps.append(
            HostOp(
                name=cols_name,
                kind="im2col",
                inputs=[source_name],
                outputs=[cols_name],
                attrs={
                    "kernel_size": int(module.kernel_size),
                    "stride": int(module.stride),
                    "padding": int(module.padding),
                    "input_layout": layout,
                    "input_channels": int(module.in_channels),
                },
            )
        )
        cols_value = golden.im2col(
            image_hwc,
            kernel_size=int(module.kernel_size),
            stride=int(module.stride),
            padding=int(module.padding),
        )
        env[cols_name] = cols_value
        tensors[cols_name] = TensorSpec(
            cols_name,
            normalize_shape(cols_value.shape),
            module_in_dtype,
            TensorKind.INTERMEDIATE,
        )
        current_inputs.update({cols_name})

        kernel_t_name = f"{out_name}_kernel_t"
        weight_int = to_numpy(module.weight_int).astype(np.int16)
        kernel_t = weight_int.reshape(weight_int.shape[0], -1).T
        tensors[kernel_t_name] = TensorSpec(
            kernel_t_name,
            normalize_shape(kernel_t.shape),
            module_in_dtype,
            TensorKind.CONSTANT,
            data=np.array(kernel_t, copy=True),
        )
        current_inputs.add(kernel_t_name)

        bias_name = None
        bias_tensor = getattr(module, "bias_int32", None)
        bias_value = None
        if bias_tensor is not None:
            bias_value = to_numpy(bias_tensor).astype(np.int32).reshape(1, -1)
            bias_name = f"{out_name}_bias"
            tensors[bias_name] = TensorSpec(
                bias_name,
                normalize_shape(bias_value.shape),
                DType.INT32,
                TensorKind.CONSTANT,
                data=np.array(bias_value, copy=True),
            )
            current_inputs.add(bias_name)

        multiplier = int(module.multiplier)
        shift = int(module.shift)
        expected_cols = golden.coerce_npu_input(cols_value, out_dtype=module_in_dtype, tensor_name=cols_name)
        env[cols_name] = expected_cols
        expected_matrix = golden.matmul(
            expected_cols,
            kernel_t,
            bias=bias_value,
            multiplier=multiplier,
            shift=shift,
            activation="none",
            out_dtype=module_out_dtype,
        )
        matmul_name = f"{out_name}_matmul"
        tensors[matmul_name] = TensorSpec(
            matmul_name,
            normalize_shape(expected_matrix.shape),
            module_out_dtype,
            TensorKind.INTERMEDIATE,
            metadata={
                "quantization": {
                    "scale": float(module.output_scale),
                    "zero_point": 0,
                    "dtype": module_out_dtype.value,
                    "source": "compiler_ready_conv2d_matrix",
                    "module_name": module_name,
                }
            },
        )
        env[matmul_name] = expected_matrix.astype(np.int32)
        current_ops.append(
            MatMulOp(
                name=matmul_name,
                lhs=cols_name,
                rhs=kernel_t_name,
                out=matmul_name,
                bias=bias_name,
                multiplier=multiplier,
                shift=shift,
                activation="none",
                in_dtype=module_in_dtype,
                out_dtype=module_out_dtype,
            )
        )
        flush_segment()

        out_h = ((image_hwc.shape[0] + (2 * int(module.padding)) - int(module.kernel_size)) // int(module.stride)) + 1
        out_w = ((image_hwc.shape[1] + (2 * int(module.padding)) - int(module.kernel_size)) // int(module.stride)) + 1
        hwc_out = expected_matrix.reshape(out_h, out_w, int(module.out_channels))
        final_value = restore_conv_output_layout(hwc_out, layout, original_shape)
        steps.append(
            HostOp(
                name=f"{out_name}_layout_restore",
                kind="layout_restore",
                inputs=[matmul_name],
                outputs=[out_name],
                attrs={
                    "layout": layout,
                    "original_shape": original_shape,
                    "out_h": out_h,
                    "out_w": out_w,
                    "out_channels": int(module.out_channels),
                },
            )
        )
        env[out_name] = final_value
        tensors[out_name] = TensorSpec(
            out_name,
            normalize_shape(final_value.shape),
            module_out_dtype,
            TensorKind.INTERMEDIATE,
            metadata={
                "quantization": {
                    "scale": float(module.output_scale),
                    "zero_point": 0,
                    "dtype": module_out_dtype.value,
                    "source": "compiler_ready_conv2d",
                    "module_name": module_name,
                }
            },
        )
        expected_tensors[cols_name] = np.array(env[cols_name], copy=True)
        expected_tensors[matmul_name] = np.array(expected_matrix, copy=True)
        expected_tensors[out_name] = np.array(final_value, copy=True)

    modules = dict(graph_module.named_modules())
    quantize_module_types = tuple(t for t in (QQuantize, QuantStub, CompilerQuantize) if t)
    dequantize_module_types = tuple(t for t in (QDeQuantize, DeQuantStub, CompilerDequantize) if t)
    quantized_linear_types = tuple(t for t in (QLinear,) if t)
    quantized_conv2d_types = tuple(t for t in (QConv2d,) if t)
    compiler_ready_linear_types = tuple(t for t in (CompilerReadyLinear,) if t)
    compiler_ready_conv2d_types = tuple(t for t in (CompilerReadyConv2d,) if t)

    def quant_params_from_module(module: Any) -> tuple[float, int, DType]:
        scale = getattr(module, "scale", None)
        zero_point = getattr(module, "zero_point", None)
        dtype = getattr(module, "dtype", None)
        if scale is None or zero_point is None or dtype is None:
            raise NotImplementedError(
                "QuantStub/Quantize nodes require explicit quantization parameters. "
                "Use a converted torch.ao.nn.quantized.Quantize module or attach "
                "`scale`, `zero_point`, and `dtype` attributes to the stub before compile."
            )
        if isinstance(dtype, str):
            return float(scale), int(zero_point), parse_dtype(dtype)
        return float(scale), int(zero_point), parse_torch_quant_dtype(dtype)

    def flush_segment() -> None:
        nonlocal current_ops, current_inputs
        if not current_ops:
            return
        outputs_local = [current_ops[-1].out]
        steps.append(
            NpuSegment(
                name=f"segment_{len([s for s in steps if isinstance(s, NpuSegment)]):03d}",
                ops=list(current_ops),
                inputs=sorted(current_inputs),
                outputs=outputs_local,
            )
        )
        current_ops = []
        current_inputs = set()
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            value = to_numpy(next(example_iter))
            env[node.name] = value
            tensors[node.name] = TensorSpec(
                name=node.name,
                shape=normalize_shape(value.shape),
                dtype=infer_dtype(value),
                kind=TensorKind.INPUT,
            )
            inputs.append(node.name)
            continue

        if node.op == "get_attr":
            target = graph_module
            for part in str(node.target).split("."):
                target = getattr(target, part)
            value = to_numpy(target)
            env[node.name] = value
            tensors[node.name] = TensorSpec(
                name=node.name,
                shape=normalize_shape(value.shape),
                dtype=infer_dtype(value),
                kind=TensorKind.CONSTANT,
                data=np.array(value, copy=True),
            )
            continue

        if node.op == "call_module" and isinstance(modules[node.target], nn.Linear):
            module = modules[node.target]
            inp_node = node.args[0]
            lhs_name = f"{node.name}_weight"
            rhs_name = inp_node.name
            out_name = node.name
            bias_name = None
            weight = module.weight.detach().cpu().numpy().astype(np.int16)
            tensors[lhs_name] = TensorSpec(lhs_name, normalize_shape(weight.shape), DType.INT16, TensorKind.CONSTANT, data=weight)
            current_inputs.update({lhs_name, rhs_name})
            if module.bias is not None:
                bias = module.bias.detach().cpu().numpy().reshape(1, -1).astype(np.int32)
                bias_name = f"{node.name}_bias"
                tensors[bias_name] = TensorSpec(bias_name, normalize_shape(bias.shape), DType.INT32, TensorKind.CONSTANT, data=bias)
                current_inputs.add(bias_name)
            rhs = coerce_npu_tensor(rhs_name)
            expected = golden.matmul(weight, rhs, bias=tensors[bias_name].data if bias_name else None)
            env[out_name] = expected.astype(np.int32)
            tensors[out_name] = TensorSpec(out_name, normalize_shape(expected.shape), DType.INT16, TensorKind.INTERMEDIATE)
            current_ops.append(MatMulOp(name=node.name, lhs=lhs_name, rhs=rhs_name, out=out_name, bias=bias_name))
            continue

        if node.op == "call_module" and quantized_linear_types and isinstance(modules[node.target], quantized_linear_types):
            lower_quantized_linear(node.target, modules[node.target], node.args[0].name, node.name)
            continue

        if node.op == "call_module" and quantized_conv2d_types and isinstance(modules[node.target], quantized_conv2d_types):
            lower_quantized_conv2d(node.target, modules[node.target], node.args[0].name, node.name)
            continue

        if node.op == "call_module" and compiler_ready_linear_types and isinstance(modules[node.target], compiler_ready_linear_types):
            lower_compiler_ready_linear(node.target, modules[node.target], node.args[0].name, node.name)
            continue

        if node.op == "call_module" and compiler_ready_conv2d_types and isinstance(modules[node.target], compiler_ready_conv2d_types):
            lower_compiler_ready_conv2d(node.target, modules[node.target], node.args[0].name, node.name)
            continue

        if node.op == "call_module" and quantize_module_types and isinstance(modules[node.target], quantize_module_types):
            flush_segment()
            source = node.args[0].name
            scale, zero_point, dtype = quant_params_from_module(modules[node.target])
            out_name = node.name
            steps.append(
                HostOp(
                    name=node.name,
                    kind="quantize",
                    inputs=[source],
                    outputs=[out_name],
                    attrs={"scale": scale, "zero_point": zero_point, "dtype": dtype},
                )
            )
            env[out_name] = golden.quantize(env[source], scale=scale, zero_point=zero_point, out_dtype=dtype)
            tensors[out_name] = TensorSpec(
                out_name,
                normalize_shape(env[out_name].shape),
                dtype,
                TensorKind.INTERMEDIATE,
                metadata={"quantization": {"scale": scale, "zero_point": zero_point, "dtype": dtype.value}},
            )
            expected_tensors[out_name] = np.array(env[out_name], copy=True)
            continue

        if node.op == "call_module" and dequantize_module_types and isinstance(modules[node.target], dequantize_module_types):
            flush_segment()
            source = node.args[0].name
            quant = tensors[source].metadata.get("quantization")
            if quant is None:
                raise NotImplementedError(
                    f"DeQuantStub/DeQuantize on tensor {source!r} requires upstream quantization metadata."
                )
            out_name = node.name
            steps.append(
                HostOp(
                    name=node.name,
                    kind="dequantize",
                    inputs=[source],
                    outputs=[out_name],
                    attrs={"scale": float(quant["scale"]), "zero_point": int(quant.get("zero_point", 0))},
                )
            )
            env[out_name] = golden.dequantize(
                env[source],
                scale=float(quant["scale"]),
                zero_point=int(quant.get("zero_point", 0)),
            )
            tensors[out_name] = TensorSpec(
                out_name,
                normalize_shape(env[out_name].shape),
                DType.FLOAT32,
                TensorKind.INTERMEDIATE,
            )
            expected_tensors[out_name] = np.array(env[out_name], copy=True)
            continue

        if (
            node.op == "call_function"
            and node.target in {operator.matmul, getattr(torch, "matmul", None)}
        ) or (node.op == "call_method" and node.target == "matmul"):
            lhs_node = node.args[0]
            rhs_node = node.args[1]
            lhs_name = lhs_node.name
            rhs_name = rhs_node.name
            out_name = node.name
            current_inputs.update({lhs_name, rhs_name})
            lhs_value = coerce_npu_tensor(lhs_name)
            rhs_value = coerce_npu_tensor(rhs_name)
            expected = golden.matmul(lhs_value, rhs_value)
            env[out_name] = expected.astype(np.int32)
            tensors[out_name] = TensorSpec(out_name, normalize_shape(expected.shape), DType.INT16, TensorKind.INTERMEDIATE)
            current_ops.append(MatMulOp(name=node.name, lhs=lhs_name, rhs=rhs_name, out=out_name))
            continue

        if node.op == "call_function" and getattr(node.target, "__name__", None) == "npu_matmul":
            lhs_name = node.args[0].name
            rhs_name = node.args[1].name
            multiplier = int(node.kwargs.get("multiplier", 1))
            shift = int(node.kwargs.get("shift", 0))
            activation = str(node.kwargs.get("activation", "none"))
            in_dtype = parse_dtype(node.kwargs.get("in_dtype", "int16"))
            out_dtype = parse_dtype(node.kwargs.get("out_dtype", "int16"))
            output_scale = node.kwargs.get("output_scale")
            output_zero_point = int(node.kwargs.get("output_zero_point", 0))
            out_name = node.name
            current_inputs.update({lhs_name, rhs_name})
            lhs_value = golden.coerce_npu_input(env[lhs_name], out_dtype=in_dtype, tensor_name=lhs_name)
            rhs_value = golden.coerce_npu_input(env[rhs_name], out_dtype=in_dtype, tensor_name=rhs_name)
            env[lhs_name] = lhs_value
            env[rhs_name] = rhs_value
            if lhs_name in tensors:
                tensors[lhs_name].dtype = in_dtype
                if tensors[lhs_name].data is not None:
                    tensors[lhs_name].data = np.array(lhs_value, copy=True)
            if rhs_name in tensors:
                tensors[rhs_name].dtype = in_dtype
                if tensors[rhs_name].data is not None:
                    tensors[rhs_name].data = np.array(rhs_value, copy=True)
            expected = golden.matmul(
                lhs_value,
                rhs_value,
                multiplier=multiplier,
                shift=shift,
                activation=activation,
                out_dtype=out_dtype,
            )
            env[out_name] = expected.astype(np.int32)
            metadata = {}
            if output_scale is not None:
                metadata["quantization"] = {
                    "scale": float(output_scale),
                    "zero_point": output_zero_point,
                    "dtype": out_dtype.value,
                }
            tensors[out_name] = TensorSpec(out_name, normalize_shape(expected.shape), out_dtype, TensorKind.INTERMEDIATE, metadata=metadata)
            current_ops.append(
                MatMulOp(
                    name=node.name,
                    lhs=lhs_name,
                    rhs=rhs_name,
                    out=out_name,
                    multiplier=multiplier,
                    shift=shift,
                    activation=activation,
                    in_dtype=in_dtype,
                    out_dtype=out_dtype,
                )
            )
            continue

        if node.op == "call_function" and node.target in {
            getattr(torch, "relu", None),
            getattr(nn.functional, "relu", None),
        }:
            source = node.args[0].name
            if current_ops and current_ops[-1].out == source and current_ops[-1].activation == "none":
                matmul_op = current_ops[-1]
                matmul_op.activation = "relu"
                matmul_op.out = node.name
                bias_value = tensors[matmul_op.bias].data if matmul_op.bias else None
                expected = golden.matmul(
                    env[matmul_op.lhs],
                    env[matmul_op.rhs],
                    bias=bias_value,
                    multiplier=matmul_op.multiplier,
                    shift=matmul_op.shift,
                    activation="relu",
                    out_dtype=matmul_op.out_dtype,
                )
                env[node.name] = expected.astype(np.int32)
                tensors[node.name] = TensorSpec(
                    node.name,
                    normalize_shape(expected.shape),
                    matmul_op.out_dtype,
                    TensorKind.INTERMEDIATE,
                    metadata=dict(tensors[source].metadata),
                )
                continue

            flush_segment()
            steps.append(HostOp(name=node.name, kind="relu", inputs=[source], outputs=[node.name]))
            env[node.name] = np.maximum(env[source], 0)
            tensors[node.name] = TensorSpec(
                name=node.name,
                shape=normalize_shape(env[node.name].shape),
                dtype=tensors[source].dtype,
                kind=TensorKind.INTERMEDIATE,
                metadata=dict(tensors[source].metadata),
            )
            expected_tensors[node.name] = np.array(env[node.name], copy=True)
            continue

        if node.op == "call_method" and node.target == "reshape":
            source = node.args[0].name
            shape = tuple(int(dim) for dim in node.args[1:])
            flush_segment()
            steps.append(HostOp(name=node.name, kind="reshape", inputs=[source], outputs=[node.name], attrs={"shape": shape}))
            env[node.name] = np.reshape(env[source], shape)
            tensors[node.name] = TensorSpec(
                name=node.name,
                shape=normalize_shape(env[node.name].shape),
                dtype=tensors[source].dtype,
                kind=tensors[source].kind,
                data=np.array(env[node.name], copy=True) if tensors[source].kind == TensorKind.CONSTANT else None,
            )
            continue

        if node.op == "call_function" and node.target in {operator.add, getattr(torch, "add", None)}:
            lhs_node = node.args[0]
            rhs_node = node.args[1]
            lhs_name = lhs_node.name
            rhs_name = rhs_node.name
            if current_ops and current_ops[-1].out == lhs_name and tensors[rhs_name].kind == TensorKind.CONSTANT:
                bias_value = np.array(env[rhs_name], copy=True).reshape(1, -1).astype(np.int32)
                bias_name = f"{node.name}_bias"
                tensors[bias_name] = TensorSpec(
                    name=bias_name,
                    shape=normalize_shape(bias_value.shape),
                    dtype=DType.INT32,
                    kind=TensorKind.CONSTANT,
                    data=bias_value,
                )
                current_inputs.add(bias_name)
                matmul_op = current_ops[-1]
                matmul_op.bias = bias_name
                matmul_op.out = node.name
                expected = golden.matmul(
                    env[matmul_op.lhs],
                    env[matmul_op.rhs],
                    bias=bias_value,
                    multiplier=matmul_op.multiplier,
                    shift=matmul_op.shift,
                    activation=matmul_op.activation,
                    out_dtype=matmul_op.out_dtype,
                )
                env[node.name] = expected.astype(np.int32)
                tensors[node.name] = TensorSpec(
                    node.name,
                    normalize_shape(expected.shape),
                    matmul_op.out_dtype,
                    TensorKind.INTERMEDIATE,
                )
                continue
            raise NotImplementedError(
                "Only constant bias-add immediately following a matmul is currently supported."
            )

        if node.op == "call_function" and getattr(node.target, "__name__", None) == "mark_for_verify":
            flush_segment()
            source = node.args[0].name
            label = node.args[1] if len(node.args) > 1 else None
            env[node.name] = env[source]
            tensors[node.name] = tensors[source].clone_without_data(name=node.name)
            tensors[node.name].verify_label = label or node.name
            steps.append(HostOp(name=f"{node.name}_alias", kind="alias", inputs=[source], outputs=[node.name]))
            steps.append(
                VerifyTensor(
                    tensor_name=node.name,
                    label=tensors[node.name].verify_label,
                    is_final_output=tensors[node.name].is_final_output,
                )
            )
            expected_tensors[node.name] = np.array(env[node.name], copy=True)
            continue

        if node.op == "call_function" and getattr(node.target, "__name__", None) == "softmax":
            flush_segment()
            source = node.args[0].name
            axis = int(node.kwargs.get("dim", -1))
            out_name = node.name
            steps.append(HostOp(name=node.name, kind="softmax", inputs=[source], outputs=[out_name], attrs={"axis": axis}))
            env[out_name] = golden.softmax(env[source], axis=axis)
            tensors[out_name] = TensorSpec(out_name, normalize_shape(env[out_name].shape), DType.FLOAT32, TensorKind.INTERMEDIATE)
            expected_tensors[out_name] = np.array(env[out_name], copy=True)
            continue

        if node.op == "call_function" and getattr(node.target, "__name__", None) == "im2col_for_npu":
            flush_segment()
            source = node.args[0].name
            kernel_size = int(node.args[1])
            stride = int(node.args[2]) if len(node.args) > 2 else 1
            padding = int(node.args[3]) if len(node.args) > 3 else 0
            out_name = node.name
            steps.append(
                HostOp(
                    name=node.name,
                    kind="im2col",
                    inputs=[source],
                    outputs=[out_name],
                    attrs={"kernel_size": kernel_size, "stride": stride, "padding": padding},
                )
            )
            env[out_name] = golden.im2col(
                env[source],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            tensors[out_name] = TensorSpec(out_name, normalize_shape(env[out_name].shape), infer_dtype(env[out_name]), TensorKind.INTERMEDIATE)
            expected_tensors[out_name] = np.array(env[out_name], copy=True)
            continue

        if (
            node.op == "call_method"
            and node.target == "mean"
        ) or (
            node.op == "call_function"
            and node.target in {getattr(torch, "mean", None), np.mean}
        ):
            flush_segment()
            source = node.args[0].name
            dims = node.kwargs.get("dim")
            if dims is None and len(node.args) > 1:
                dims = node.args[1]
            if isinstance(dims, int):
                dims = [dims]
            elif dims is not None:
                dims = [int(dim) for dim in dims]
            keepdim = bool(node.kwargs.get("keepdim", False))
            quant = tensors[source].metadata.get("quantization")
            mean_value = np.mean(
                np.array(env[source], dtype=np.float32),
                axis=None if dims is None else tuple(dims),
                keepdims=keepdim,
            )
            out_dtype = DType.FLOAT32
            metadata = {}
            attrs = {"dim": dims, "keepdim": keepdim}
            if quant is not None:
                out_dtype = parse_dtype(quant["dtype"])
                attrs["quantization"] = {
                    "scale": float(quant["scale"]),
                    "zero_point": int(quant.get("zero_point", 0)),
                    "dtype": out_dtype,
                }
                mean_value = golden.quantized_mean(
                    env[source],
                    axis=None if dims is None else tuple(dims),
                    keepdims=keepdim,
                    zero_point=int(quant.get("zero_point", 0)),
                    out_dtype=out_dtype,
                )
                metadata = {"quantization": dict(quant)}
            steps.append(HostOp(name=node.name, kind="mean", inputs=[source], outputs=[node.name], attrs=attrs))
            env[node.name] = np.array(mean_value, copy=True)
            tensors[node.name] = TensorSpec(
                name=node.name,
                shape=normalize_shape(env[node.name].shape),
                dtype=out_dtype,
                kind=TensorKind.INTERMEDIATE,
                metadata=metadata,
            )
            expected_tensors[node.name] = np.array(env[node.name], copy=True)
            continue

        if node.op == "call_function" and getattr(node.target, "__name__", None) == "quantize_for_npu":
            flush_segment()
            source = node.args[0].name
            scale = float(node.args[1])
            zero_point = int(node.args[2]) if len(node.args) > 2 else 0
            dtype = parse_dtype(node.args[3] if len(node.args) > 3 else "int16")
            out_name = node.name
            steps.append(
                HostOp(
                    name=node.name,
                    kind="requantize",
                    inputs=[source],
                    outputs=[out_name],
                    attrs={"scale": scale, "zero_point": zero_point, "dtype": dtype},
                )
            )
            env[out_name] = golden.requantize(env[source], scale=scale, zero_point=zero_point, out_dtype=dtype)
            tensors[out_name] = TensorSpec(
                out_name,
                normalize_shape(env[out_name].shape),
                dtype,
                TensorKind.INTERMEDIATE,
                metadata={"quantization": {"scale": scale, "zero_point": zero_point, "dtype": dtype.value}},
            )
            expected_tensors[out_name] = np.array(env[out_name], copy=True)
            continue

        if node.op == "output":
            flush_segment()
            raw_outputs = node.args[0]
            output_nodes = raw_outputs if isinstance(raw_outputs, (tuple, list)) else (raw_outputs,)
            for out in output_nodes:
                tensors[out.name].is_final_output = True
                steps.append(VerifyTensor(tensor_name=out.name, label=tensors[out.name].verify_label or out.name, is_final_output=True))
                outputs.append(out.name)
                expected_tensors[out.name] = np.array(env[out.name], copy=True)
            continue

        raise NotImplementedError(
            f"Unsupported FX node for initial frontend: op={node.op}, target={node.target!r}."
        )

    for step in steps:
        if isinstance(step, NpuSegment):
            for out_name in step.outputs:
                expected_tensors[out_name] = np.array(env[out_name], copy=True)

    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=inputs, outputs=outputs, metadata={"frontend": "torch.fx"})
    return plan, expected_tensors
