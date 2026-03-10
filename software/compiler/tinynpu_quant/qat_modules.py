from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LayerQuantConfig, ensure_layer_quant_config
from .fake_quant import fake_quantize


class QConv2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        *,
        padding: int = 0,
        stride: int = 1,
        w_bits: int = 8,
        a_bits: int = 8,
        config: LayerQuantConfig | None = None,
    ):
        super().__init__()
        cfg = ensure_layer_quant_config(config) if config is not None else LayerQuantConfig(w_bits=w_bits, a_bits=a_bits)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, stride=stride, bias=True)
        self.w_bits = int(cfg.w_bits)
        self.a_bits = int(cfg.a_bits)
        self.signed_activations = bool(cfg.signed_activations)
        self.w_scale = nn.Parameter(torch.tensor(0.05))
        self.a_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        with torch.no_grad():
            w_qmax = 2 ** (self.w_bits - 1) - 1
            self.w_scale.data.copy_((self.conv.weight.abs().max() / w_qmax).clamp(min=1e-8))
        x_q = fake_quantize(
            x,
            self.a_scale,
            self.a_bits,
            is_weight=False,
            signed_activations=self.signed_activations,
        )
        w_q = fake_quantize(self.conv.weight, self.w_scale, self.w_bits, is_weight=True)
        return F.conv2d(x_q, w_q, self.conv.bias, self.conv.stride, self.conv.padding)


class QLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        w_bits: int = 8,
        a_bits: int = 8,
        config: LayerQuantConfig | None = None,
    ):
        super().__init__()
        cfg = ensure_layer_quant_config(config) if config is not None else LayerQuantConfig(w_bits=w_bits, a_bits=a_bits)
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.w_bits = int(cfg.w_bits)
        self.a_bits = int(cfg.a_bits)
        self.signed_activations = bool(cfg.signed_activations)
        self.w_scale = nn.Parameter(torch.tensor(0.05))
        self.a_scale = nn.Parameter(torch.tensor(0.05))

    def forward(self, x):
        with torch.no_grad():
            w_qmax = 2 ** (self.w_bits - 1) - 1
            self.w_scale.data.copy_((self.linear.weight.abs().max() / w_qmax).clamp(min=1e-8))
        x_q = fake_quantize(
            x,
            self.a_scale,
            self.a_bits,
            is_weight=False,
            signed_activations=self.signed_activations,
        )
        w_q = fake_quantize(self.linear.weight, self.w_scale, self.w_bits, is_weight=True)
        return F.linear(x_q, w_q, self.linear.bias)
