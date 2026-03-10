from __future__ import annotations

import torch


class SymmetricQuantizer(torch.autograd.Function):
    """STE-based fake quantizer used by the TinyNPU QAT utilities.

    Current behavior matches the existing MNIST training script:
    - weights use signed symmetric quantization
    - activations use unsigned [0, 2^bits-1] fake quantization
    """

    @staticmethod
    def forward(ctx, x, scale, num_bits, is_weight, signed_activations):
        if is_weight:
            qmin = -(2 ** (num_bits - 1)) + 1
            qmax = 2 ** (num_bits - 1) - 1
        else:
            if signed_activations:
                qmin = -(2 ** (num_bits - 1)) + 1
                qmax = 2 ** (num_bits - 1) - 1
            else:
                qmin = 0
                qmax = 2 ** num_bits - 1
        x_scaled = x / scale
        x_clamped = torch.clamp(x_scaled, qmin, qmax)
        x_quant = torch.round(x_clamped)
        x_dequant = x_quant * scale
        ctx.save_for_backward(
            x_scaled,
            torch.tensor(qmin, device=x.device, dtype=x.dtype),
            torch.tensor(qmax, device=x.device, dtype=x.dtype),
        )
        return x_dequant

    @staticmethod
    def backward(ctx, grad_output):
        x_scaled, qmin, qmax = ctx.saved_tensors
        mask = (x_scaled >= qmin) & (x_scaled <= qmax)
        return grad_output * mask.float(), None, None, None, None


def fake_quantize(x, scale, num_bits, is_weight: bool = False, signed_activations: bool = False):
    return SymmetricQuantizer.apply(x, scale, num_bits, is_weight, signed_activations)
