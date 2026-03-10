from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .fused_params import compute_fused_params
from .qat_modules import QConv2d, QLinear


def collect_qat_layer_names(model: nn.Module) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, (QConv2d, QLinear))]


def infer_chain_output_scales(model: nn.Module, layer_order: list[str]) -> dict[str, float]:
    if not layer_order:
        raise ValueError("layer_order must contain at least one QAT layer name.")
    modules = dict(model.named_modules())
    scales: dict[str, float] = {}
    for index, name in enumerate(layer_order):
        layer = modules[name]
        if not isinstance(layer, (QConv2d, QLinear)):
            raise TypeError(f"Layer {name!r} is not a supported QAT layer.")
        if index < len(layer_order) - 1:
            next_layer = modules[layer_order[index + 1]]
            scales[name] = float(next_layer.a_scale.item())
        else:
            scales[name] = float(layer.w_scale.item() * layer.a_scale.item())
    return scales


def infer_chain_output_bits(model: nn.Module, layer_order: list[str]) -> dict[str, int]:
    if not layer_order:
        raise ValueError("layer_order must contain at least one QAT layer name.")
    modules = dict(model.named_modules())
    bits: dict[str, int] = {}
    for index, name in enumerate(layer_order):
        layer = modules[name]
        if not isinstance(layer, (QConv2d, QLinear)):
            raise TypeError(f"Layer {name!r} is not a supported QAT layer.")
        if index < len(layer_order) - 1:
            next_layer = modules[layer_order[index + 1]]
            bits[name] = int(next_layer.a_bits)
        else:
            # Final logits do not feed another quantized activation by default.
            # Keep them in accumulator-space precision unless the caller overrides it.
            bits[name] = 16
    return bits


def convert_qat_model_for_compiler(
    model: nn.Module,
    *,
    layer_order: list[str] | None = None,
    input_scale: float | None = None,
    output_scales: dict[str, float] | None = None,
    output_bits: dict[str, int] | None = None,
    dequantize_output: bool = True,
) -> nn.Module:
    model = copy.deepcopy(model).cpu().eval()
    layer_order = layer_order or collect_qat_layer_names(model)
    if not layer_order:
        raise ValueError("Model contains no QConv2d/QLinear layers to convert.")
    modules = dict(model.named_modules())
    output_scales = output_scales or infer_chain_output_scales(model, layer_order)
    output_bits = output_bits or infer_chain_output_bits(model, layer_order)
    first_layer = modules[layer_order[0]]
    if not isinstance(first_layer, (QConv2d, QLinear)):
        raise TypeError(f"First layer {layer_order[0]!r} is not a supported QAT layer.")
    input_scale = float(input_scale if input_scale is not None else first_layer.a_scale.item())
    input_bits = int(first_layer.a_bits)

    previous_output_scale: float | None = None
    for index, name in enumerate(layer_order):
        layer = modules[name]
        if not isinstance(layer, (QConv2d, QLinear)):
            raise TypeError(f"Unsupported layer type for {name!r}: {type(layer)!r}")
        if int(layer.w_bits) != int(layer.a_bits):
            raise NotImplementedError(
                f"Layer {name!r} uses W{layer.w_bits}A{layer.a_bits}. "
                "Current TinyNPU compiler-ready conversion requires matching activation/weight precision "
                "inside each NPU layer because the hardware exposes one input precision per matmul."
            )
        layer_input_scale = float(input_scale) if index == 0 else float(previous_output_scale)
        layer_output_scale = float(output_scales[name])
        layer_output_bits = int(output_bits[name])
        if isinstance(layer, QConv2d):
            replacement = _convert_qconv2d(
                layer,
                input_scale=layer_input_scale,
                output_scale=layer_output_scale,
                output_bits=layer_output_bits,
            )
        else:
            replacement = _convert_qlinear(
                layer,
                input_scale=layer_input_scale,
                output_scale=layer_output_scale,
                output_bits=layer_output_bits,
            )
        _set_submodule(model, name, replacement)
        previous_output_scale = layer_output_scale

    final_output_scale = float(output_scales[layer_order[-1]])
    return CompilerReadyWrapper(
        model,
        input_scale=input_scale,
        input_bits=input_bits,
        output_scale=final_output_scale,
        dequantize_output=dequantize_output,
    ).eval()


class CompilerQuantize(nn.Module):
    def __init__(self, *, scale: float, zero_point: int = 0, dtype: str = "int8"):
        super().__init__()
        self.scale = float(scale)
        self.zero_point = int(zero_point)
        self.dtype = str(dtype)

    def forward(self, x):
        return x


class CompilerDequantize(nn.Module):
    def __init__(self, *, scale: float, zero_point: int = 0):
        super().__init__()
        self.scale = float(scale)
        self.zero_point = int(zero_point)

    def forward(self, x):
        return x


class CompilerReadyLinear(nn.Module):
    def __init__(
        self,
        *,
        weight_int: torch.Tensor,
        bias_int32: torch.Tensor | None,
        input_scale: float,
        weight_scale: float,
        output_scale: float,
        in_bits: int,
        out_bits: int,
    ):
        super().__init__()
        self.register_buffer("weight_int", weight_int.to(torch.int16))
        if bias_int32 is not None:
            self.register_buffer("bias_int32", bias_int32.to(torch.int32))
        else:
            self.bias_int32 = None
        self.input_scale = float(input_scale)
        self.weight_scale = float(weight_scale)
        self.output_scale = float(output_scale)
        self.in_bits = int(in_bits)
        self.out_bits = int(out_bits)
        fused = compute_fused_params(self.input_scale, self.weight_scale, self.output_scale)
        self.multiplier = int(fused[0])
        self.shift = int(fused[1])
        self.in_dtype = bits_to_dtype_name(self.in_bits)
        self.out_dtype = bits_to_dtype_name(self.out_bits)

    def forward(self, x):
        return x


class CompilerReadyConv2d(nn.Module):
    def __init__(
        self,
        *,
        weight_int: torch.Tensor,
        bias_int32: torch.Tensor | None,
        input_scale: float,
        weight_scale: float,
        output_scale: float,
        in_bits: int,
        out_bits: int,
        stride: int,
        padding: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
    ):
        super().__init__()
        self.register_buffer("weight_int", weight_int.to(torch.int16))
        if bias_int32 is not None:
            self.register_buffer("bias_int32", bias_int32.to(torch.int32))
        else:
            self.bias_int32 = None
        self.input_scale = float(input_scale)
        self.weight_scale = float(weight_scale)
        self.output_scale = float(output_scale)
        self.in_bits = int(in_bits)
        self.out_bits = int(out_bits)
        self.stride = int(stride)
        self.padding = int(padding)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        fused = compute_fused_params(self.input_scale, self.weight_scale, self.output_scale)
        self.multiplier = int(fused[0])
        self.shift = int(fused[1])
        self.in_dtype = bits_to_dtype_name(self.in_bits)
        self.out_dtype = bits_to_dtype_name(self.out_bits)

    def forward(self, x):
        return x


class CompilerReadyWrapper(nn.Module):
    def __init__(
        self,
        inner: nn.Module,
        *,
        input_scale: float,
        input_bits: int,
        output_scale: float,
        dequantize_output: bool,
    ):
        super().__init__()
        self.q_in = CompilerQuantize(scale=input_scale, zero_point=0, dtype=bits_to_dtype_name(input_bits))
        self.inner = inner
        self.dq_out = CompilerDequantize(scale=output_scale, zero_point=0) if dequantize_output else None

    def forward(self, x):
        x = self.q_in(x)
        x = self.inner(x)
        if self.dq_out is not None:
            x = self.dq_out(x)
        return x


def bits_to_dtype_name(bits: int) -> str:
    mapping = {4: "int4", 8: "int8", 16: "int16"}
    if int(bits) not in mapping:
        raise NotImplementedError(
            f"TinyNPU compiler-ready conversion supports 4-bit, 8-bit, and 16-bit tensors only, got {bits}."
        )
    return mapping[int(bits)]


def _quantize_symmetric_tensor(value: torch.Tensor, *, scale: float, bits: int) -> torch.Tensor:
    qmax = (1 << (int(bits) - 1)) - 1
    qmin = -qmax
    quantized = torch.round(value.detach().cpu().float() / float(scale))
    return torch.clamp(quantized, qmin, qmax).to(torch.int16)


def _quantize_bias(bias: torch.Tensor | None, *, input_scale: float, weight_scale: float) -> torch.Tensor | None:
    if bias is None:
        return None
    bias_scale = float(input_scale) * float(weight_scale)
    if bias_scale <= 0:
        raise ValueError(f"Bias scale must be positive, got {bias_scale}.")
    return torch.round(bias.detach().cpu().float() / bias_scale).to(torch.int32).reshape(1, -1)


def _convert_qconv2d(
    layer: QConv2d,
    *,
    input_scale: float,
    output_scale: float,
    output_bits: int,
) -> CompilerReadyConv2d:
    weight_scale = float(layer.w_scale.item())
    weight_int = _quantize_symmetric_tensor(layer.conv.weight, scale=weight_scale, bits=int(layer.w_bits))
    bias_int32 = _quantize_bias(layer.conv.bias, input_scale=float(input_scale), weight_scale=weight_scale)
    return CompilerReadyConv2d(
        weight_int=weight_int,
        bias_int32=bias_int32,
        input_scale=float(input_scale),
        weight_scale=weight_scale,
        output_scale=float(output_scale),
        in_bits=int(layer.a_bits),
        out_bits=int(output_bits),
        stride=int(layer.conv.stride[0]),
        padding=int(layer.conv.padding[0]),
        in_channels=int(layer.conv.in_channels),
        out_channels=int(layer.conv.out_channels),
        kernel_size=int(layer.conv.kernel_size[0]),
    )


def _convert_qlinear(
    layer: QLinear,
    *,
    input_scale: float,
    output_scale: float,
    output_bits: int,
) -> CompilerReadyLinear:
    weight_scale = float(layer.w_scale.item())
    weight_int = _quantize_symmetric_tensor(layer.linear.weight, scale=weight_scale, bits=int(layer.w_bits))
    bias_int32 = _quantize_bias(layer.linear.bias, input_scale=float(input_scale), weight_scale=weight_scale)
    return CompilerReadyLinear(
        weight_int=weight_int,
        bias_int32=bias_int32,
        input_scale=float(input_scale),
        weight_scale=weight_scale,
        output_scale=float(output_scale),
        in_bits=int(layer.a_bits),
        out_bits=int(output_bits),
    )


def _set_submodule(module: nn.Module, qualified_name: str, replacement: nn.Module) -> None:
    parent = module
    parts = qualified_name.split(".")
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], replacement)
