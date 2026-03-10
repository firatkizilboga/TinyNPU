from __future__ import annotations

import numpy as np

from software.compiler.tinynpu_jit import VerificationMode, compile_module, run_host_emulation


def _make_quantized_linear_module():
    import torch
    import torch.nn as nn
    from torch.ao.nn.quantized import DeQuantize, Linear, Quantize

    weight_fp = torch.tensor(
        [
            [0.75, -0.50, 0.25, 0.00],
            [-0.25, 0.50, -0.75, 0.25],
            [0.125, -0.375, 0.625, -0.50],
        ],
        dtype=torch.float32,
    )
    bias_fp = torch.tensor([0.125, -0.25, 0.375], dtype=torch.float32)
    weight_q = torch.quantize_per_tensor(weight_fp, scale=0.125, zero_point=0, dtype=torch.qint8)

    class QuantizedLinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = Quantize(scale=0.125, zero_point=0, dtype=torch.qint8)
            self.linear = Linear(4, 3)
            self.linear.set_weight_bias(weight_q, bias_fp)
            self.linear.scale = 0.0625
            self.linear.zero_point = 0
            self.dq = DeQuantize()

        def forward(self, x):
            x = self.q(x)
            x = self.linear(x)
            return self.dq(x)

    return QuantizedLinearModel().eval()


def _make_quantized_conv_module():
    import torch
    import torch.nn as nn
    from torch.ao.nn.quantized import Conv2d, DeQuantize, Quantize

    weight_fp = torch.tensor(
        [
            [[[0.5, -0.25, 0.125], [0.25, 0.75, -0.5], [0.0, 0.125, -0.25]]],
            [[[-0.125, 0.375, 0.5], [0.25, -0.625, 0.25], [0.125, 0.0, -0.125]]],
        ],
        dtype=torch.float32,
    )
    bias_fp = torch.tensor([0.125, -0.25], dtype=torch.float32)
    weight_q = torch.quantize_per_tensor(weight_fp, scale=0.125, zero_point=0, dtype=torch.qint8)

    class QuantizedConvModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = Quantize(scale=0.125, zero_point=0, dtype=torch.qint8)
            self.conv = Conv2d(1, 2, 3, stride=1, padding=1)
            self.conv.set_weight_bias(weight_q, bias_fp)
            self.conv.scale = 0.0625
            self.conv.zero_point = 0
            self.dq = DeQuantize()

        def forward(self, x):
            x = self.q(x)
            x = self.conv(x)
            return self.dq(x)

    return QuantizedConvModel().eval()


def build_quantized_linear_artifact():
    import torch

    module = _make_quantized_linear_module()
    example = torch.tensor(
        [[0.25], [-0.5], [0.75], [0.125]],
        dtype=torch.float32,
    )
    artifact = compile_module(module, (example,))
    return artifact, example.numpy()


def build_quantized_conv_artifact():
    import torch

    module = _make_quantized_conv_module()
    example = torch.tensor(
        [[[[0.25, -0.5, 0.75], [0.125, -0.25, 0.5], [0.0, 0.375, -0.125]]]],
        dtype=torch.float32,
    )
    artifact = compile_module(module, (example,))
    return artifact, example.numpy()


def smoke_run_quantized_modules():
    linear_artifact, linear_input = build_quantized_linear_artifact()
    linear_result = run_host_emulation(
        linear_artifact,
        {linear_artifact.plan.inputs[0]: linear_input},
        VerificationMode.DEBUG,
        debug=True,
    )

    conv_artifact, conv_input = build_quantized_conv_artifact()
    conv_result = run_host_emulation(
        conv_artifact,
        {conv_artifact.plan.inputs[0]: conv_input},
        VerificationMode.DEBUG,
        debug=True,
    )

    return {
        "linear_output": linear_result.tensors[linear_artifact.plan.outputs[0]],
        "linear_debug_kinds": [event["kind"] for event in linear_result.debug_trace],
        "conv_output": conv_result.tensors[conv_artifact.plan.outputs[0]],
        "conv_debug_kinds": [event["kind"] for event in conv_result.debug_trace],
    }


if __name__ == "__main__":
    results = smoke_run_quantized_modules()
    print("linear_output", results["linear_output"].reshape(-1).tolist())
    print("linear_debug_kinds", results["linear_debug_kinds"])
    print("conv_output_shape", list(results["conv_output"].shape))
    print("conv_debug_kinds", results["conv_debug_kinds"])
