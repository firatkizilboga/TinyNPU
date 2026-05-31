import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from tinynpu_jit import DType, compile_module, compile_module_legacy, emit_cv32e40p_program_v2, run_host_emulation
from tinynpu_jit.golden import GoldenModel
from tinynpu_quant import CompilerDequantize, CompilerQuantize, CompilerReadyConv2d, CompilerReadyLinear


class TinySemanticLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_in = CompilerQuantize(scale=0.25, zero_point=0, dtype="int16")
        self.fc1 = CompilerReadyLinear(
            weight_int=torch.tensor([[2, -1, 0], [1, 1, -2]], dtype=torch.int16),
            bias_int32=torch.tensor([3, -2], dtype=torch.int32),
            input_scale=0.25,
            weight_scale=0.5,
            output_scale=0.125,
            in_bits=16,
            out_bits=16,
        )
        self.fc2 = CompilerReadyLinear(
            weight_int=torch.tensor([[1, -1], [2, 0]], dtype=torch.int16),
            bias_int32=torch.tensor([1, -3], dtype=torch.int32),
            input_scale=0.125,
            weight_scale=0.5,
            output_scale=0.125,
            in_bits=16,
            out_bits=16,
        )
        self.dq_out = CompilerDequantize(scale=0.125, zero_point=0)

    def forward(self, x):
        x = self.q_in(x)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = self.dq_out(x)
        return x


class TinySemanticFp16DequantModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_in = CompilerQuantize(scale=0.25, zero_point=0, dtype="int16")
        self.fc = CompilerReadyLinear(
            weight_int=torch.eye(8, dtype=torch.int16),
            bias_int32=None,
            input_scale=0.25,
            weight_scale=1.0,
            output_scale=0.25,
            in_bits=16,
            out_bits=16,
        )
        self.dq = CompilerDequantize(scale=0.25, zero_point=0, output_encoding="fp16_bits")

    def forward(self, x):
        x = self.q_in(x)
        x = self.fc(x)
        x = self.dq(x)
        return x


class TinySemanticConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_in = CompilerQuantize(scale=0.25, zero_point=0, dtype="int8")
        self.conv = CompilerReadyConv2d(
            weight_int=torch.tensor([[[[1, 0], [0, -1]]], [[[0, 1], [1, 0]]]], dtype=torch.int16),
            bias_int32=torch.tensor([2, -1], dtype=torch.int32),
            input_scale=0.25,
            weight_scale=0.5,
            output_scale=0.125,
            in_bits=8,
            out_bits=8,
            stride=1,
            padding=0,
            in_channels=1,
            out_channels=2,
            kernel_size=2,
        )
        self.dq_out = CompilerDequantize(scale=0.125, zero_point=0)

    def forward(self, x):
        x = self.q_in(x)
        x = self.conv(x)
        x = torch.relu(x)
        x = self.dq_out(x)
        return x


class InvalidSemanticActivationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_in = CompilerQuantize(scale=0.25, zero_point=0, dtype="int16")
        self.dq_out = CompilerDequantize(scale=0.25, zero_point=0)

    def forward(self, x):
        x = self.q_in(x)
        x = torch.relu(x)
        x = self.dq_out(x)
        return x


class TinyPlainLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(3, 4)
        self.fc2 = nn.Linear(4, 2)
        with torch.no_grad():
            self.fc1.weight.copy_(torch.tensor([[1.0, -0.5, 0.25], [0.0, 1.0, -1.0], [0.5, 0.5, 0.5], [-1.0, 0.25, 0.75]]))
            self.fc1.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.0]))
            self.fc2.weight.copy_(torch.tensor([[1.0, -1.0, 0.5, 0.25], [-0.5, 0.5, 1.0, -1.0]]))
            self.fc2.bias.copy_(torch.tensor([0.25, -0.75]))

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


class TinyPlainConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.tensor(
                    [
                        [[[1.0, 0.0], [0.0, -1.0]]],
                        [[[0.5, 0.5], [-0.5, 1.0]]],
                    ]
                )
            )
            self.conv.bias.copy_(torch.tensor([0.25, -0.5]))

    def forward(self, x):
        return torch.relu(self.conv(x))


class TinyPlainConvBatchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.tensor(
                    [
                        [[[1.0, 0.0], [0.0, -1.0]]],
                        [[[0.5, 0.5], [-0.5, 1.0]]],
                    ]
                )
            )
            self.conv.bias.copy_(torch.tensor([0.25, -0.5]))

    def forward(self, x):
        return torch.relu(self.conv(x))


class TinyFlattenLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8, 3)
        with torch.no_grad():
            self.fc.weight.copy_(
                torch.tensor(
                    [
                        [1.0, -0.5, 0.25, 0.0, 0.5, -1.0, 0.75, -0.25],
                        [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
                        [-1.0, 0.25, 0.75, -0.25, 1.0, 0.0, -0.5, 0.5],
                    ]
                )
            )
            self.fc.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))

    def forward(self, x):
        return self.fc(self.flatten(x))


class TinyMeanResidualModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)
        with torch.no_grad():
            self.fc1.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, -0.5, 0.25],
                        [0.5, -1.0, 0.25, 0.75],
                        [-0.25, 0.5, 0.5, -0.5],
                        [0.0, 0.25, -0.75, 1.0],
                    ]
                )
            )
            self.fc1.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.0]))
            self.fc2.weight.copy_(
                torch.tensor(
                    [
                        [0.5, -0.5, 0.0, 0.25],
                        [0.0, 0.5, -1.0, 0.5],
                        [1.0, 0.0, 0.25, -0.25],
                        [-0.5, 0.75, 0.5, 0.0],
                    ]
                )
            )
            self.fc2.bias.copy_(torch.tensor([0.0, 0.25, -0.1, 0.2]))

    def forward(self, x):
        x = self.fc1(x) + self.fc2(x)
        return x.mean(dim=1, keepdim=True)


class TinyMaxPoolLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8, 3)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.tensor(
                    [
                        [[[1.0, 0.0], [0.0, -1.0]]],
                        [[[0.5, 0.5], [-0.5, 1.0]]],
                    ]
                )
            )
            self.conv.bias.copy_(torch.tensor([0.25, -0.5]))
            self.fc.weight.copy_(
                torch.tensor(
                    [
                        [1.0, -0.5, 0.25, 0.0, 0.5, -1.0, 0.75, -0.25],
                        [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
                        [-1.0, 0.25, 0.75, -0.25, 1.0, 0.0, -0.5, 0.5],
                    ]
                )
            )
            self.fc.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))

    def forward(self, x):
        return self.fc(self.flatten(self.pool(torch.relu(self.conv(x)))))


class TinyAvgPoolLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8, 2)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.tensor(
                    [
                        [[[1.0, 0.0], [0.0, -1.0]]],
                        [[[0.5, 0.5], [-0.5, 1.0]]],
                    ]
                )
            )
            self.conv.bias.copy_(torch.tensor([0.25, -0.5]))
            self.fc.weight.copy_(
                torch.tensor(
                    [
                        [1.0, -1.0, 0.5, 0.25, -0.5, 0.5, 0.25, -0.25],
                        [-0.5, 0.25, 1.0, -1.0, 0.75, -0.25, 0.5, 0.0],
                    ]
                )
            )
            self.fc.bias.copy_(torch.tensor([0.25, -0.75]))

    def forward(self, x):
        return self.fc(self.flatten(self.pool(torch.relu(self.conv(x)))))


class TinyAdaptiveAvgPoolLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2, 2)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.tensor(
                    [
                        [[[1.0, 0.0], [0.0, -1.0]]],
                        [[[0.5, 0.5], [-0.5, 1.0]]],
                    ]
                )
            )
            self.conv.bias.copy_(torch.tensor([0.25, -0.5]))
            self.fc.weight.copy_(torch.tensor([[1.0, -0.5], [-0.25, 0.75]]))
            self.fc.bias.copy_(torch.tensor([0.1, -0.2]))

    def forward(self, x):
        return self.fc(self.flatten(self.pool(torch.relu(self.conv(x)))))


class TinyHybridMaxPoolModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_in = CompilerQuantize(scale=0.25, zero_point=0, dtype="int8")
        self.conv = CompilerReadyConv2d(
            weight_int=torch.tensor([[[[1, 0], [0, -1]]], [[[0, 1], [1, 0]]]], dtype=torch.int16),
            bias_int32=torch.tensor([2, -1], dtype=torch.int32),
            input_scale=0.25,
            weight_scale=0.5,
            output_scale=0.125,
            in_bits=8,
            out_bits=8,
            stride=1,
            padding=0,
            in_channels=1,
            out_channels=2,
            kernel_size=2,
        )
        self.dq_mid = CompilerDequantize(scale=0.125, zero_point=0)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.q_post = CompilerQuantize(scale=0.125, zero_point=0, dtype="int16")
        self.fc = CompilerReadyLinear(
            weight_int=torch.tensor([[2, -1, 0, 1, 1, -2, 1, 0], [1, 1, -2, 0, 2, -1, 1, -1]], dtype=torch.int16),
            bias_int32=torch.tensor([3, -2], dtype=torch.int32),
            input_scale=0.125,
            weight_scale=0.5,
            output_scale=0.125,
            in_bits=16,
            out_bits=16,
        )
        self.dq_out = CompilerDequantize(scale=0.125, zero_point=0)

    def forward(self, x):
        x = self.q_in(x)
        x = self.conv(x)
        x = self.dq_mid(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.q_post(x)
        x = self.fc(x)
        x = self.dq_out(x)
        return x


class TinyHybridAvgPoolModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_in = CompilerQuantize(scale=0.25, zero_point=0, dtype="int8")
        self.conv = CompilerReadyConv2d(
            weight_int=torch.tensor([[[[1, 0], [0, -1]]], [[[0, 1], [1, 0]]]], dtype=torch.int16),
            bias_int32=torch.tensor([2, -1], dtype=torch.int32),
            input_scale=0.25,
            weight_scale=0.5,
            output_scale=0.125,
            in_bits=8,
            out_bits=8,
            stride=1,
            padding=0,
            in_channels=1,
            out_channels=2,
            kernel_size=2,
        )
        self.dq_mid = CompilerDequantize(scale=0.125, zero_point=0)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.q_post = CompilerQuantize(scale=0.125, zero_point=0, dtype="int16")
        self.fc = CompilerReadyLinear(
            weight_int=torch.tensor([[2, -1, 0, 1, 1, -2, 1, 0], [1, 1, -2, 0, 2, -1, 1, -1]], dtype=torch.int16),
            bias_int32=torch.tensor([3, -2], dtype=torch.int32),
            input_scale=0.125,
            weight_scale=0.5,
            output_scale=0.125,
            in_bits=16,
            out_bits=16,
        )
        self.dq_out = CompilerDequantize(scale=0.125, zero_point=0)

    def forward(self, x):
        x = self.q_in(x)
        x = self.conv(x)
        x = self.dq_mid(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.q_post(x)
        x = self.fc(x)
        x = self.dq_out(x)
        return x


class TinyHybridAdaptiveAvgPoolModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_in = CompilerQuantize(scale=0.25, zero_point=0, dtype="int8")
        self.conv = CompilerReadyConv2d(
            weight_int=torch.tensor([[[[1, 0], [0, -1]]], [[[0, 1], [1, 0]]]], dtype=torch.int16),
            bias_int32=torch.tensor([2, -1], dtype=torch.int32),
            input_scale=0.25,
            weight_scale=0.5,
            output_scale=0.125,
            in_bits=8,
            out_bits=8,
            stride=1,
            padding=0,
            in_channels=1,
            out_channels=2,
            kernel_size=2,
        )
        self.dq_mid = CompilerDequantize(scale=0.125, zero_point=0)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.q_post = CompilerQuantize(scale=0.125, zero_point=0, dtype="int16")
        self.fc = CompilerReadyLinear(
            weight_int=torch.tensor([[2, -1], [1, 1]], dtype=torch.int16),
            bias_int32=torch.tensor([3, -2], dtype=torch.int32),
            input_scale=0.125,
            weight_scale=0.5,
            output_scale=0.125,
            in_bits=16,
            out_bits=16,
        )
        self.dq_out = CompilerDequantize(scale=0.125, zero_point=0)

    def forward(self, x):
        x = self.q_in(x)
        x = self.conv(x)
        x = self.dq_mid(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.q_post(x)
        x = self.fc(x)
        x = self.dq_out(x)
        return x


class TinyBroadcastBinaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.register_buffer("bias", torch.tensor([0.25, -0.5, 0.75, -0.25], dtype=torch.float32))
        self.register_buffer("scale", torch.tensor([[1.0, 0.5, -1.0, 2.0]], dtype=torch.float32))
        with torch.no_grad():
            self.fc.weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0, -0.5, 0.25],
                        [0.5, -1.0, 0.25, 0.75],
                        [-0.25, 0.5, 0.5, -0.5],
                        [0.0, 0.25, -0.75, 1.0],
                    ]
                )
            )
            self.fc.bias.copy_(torch.tensor([0.1, -0.2, 0.3, 0.0]))

    def forward(self, x):
        x = self.fc(x)
        x = x + self.bias
        return x * self.scale


class TinyFunctionMaxPoolModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8, 2)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.tensor(
                    [
                        [[[1.0, 0.0], [0.0, -1.0]]],
                        [[[0.5, 0.5], [-0.5, 1.0]]],
                    ]
                )
            )
            self.conv.bias.copy_(torch.tensor([0.25, -0.5]))
            self.fc.weight.copy_(
                torch.tensor(
                    [
                        [1.0, -1.0, 0.5, 0.25, -0.5, 0.5, 0.25, -0.25],
                        [-0.5, 0.25, 1.0, -1.0, 0.75, -0.25, 0.5, 0.0],
                    ]
                )
            )
            self.fc.bias.copy_(torch.tensor([0.25, -0.75]))

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = F.max_pool2d(x, kernel_size=2, stride=1, padding=0)
        return self.fc(self.flatten(x))


class TinyFunctionAdaptiveAvgPoolModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=2, stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8, 3)
        with torch.no_grad():
            self.conv.weight.copy_(
                torch.tensor(
                    [
                        [[[1.0, 0.0], [0.0, -1.0]]],
                        [[[0.5, 0.5], [-0.5, 1.0]]],
                    ]
                )
            )
            self.conv.bias.copy_(torch.tensor([0.25, -0.5]))
            self.fc.weight.copy_(
                torch.tensor(
                    [
                        [1.0, -0.5, 0.25, 0.0, 0.5, -1.0, 0.75, -0.25],
                        [0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, -0.5],
                        [-1.0, 0.25, 0.75, -0.25, 1.0, 0.0, -0.5, 0.5],
                    ]
                )
            )
            self.fc.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))

    def forward(self, x):
        x = torch.relu(self.conv(x))
        x = F.adaptive_avg_pool2d(x, (2, 2))
        return self.fc(self.flatten(x))


class TinyHybridFunctionAvgPoolModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_in = CompilerQuantize(scale=0.25, zero_point=0, dtype="int8")
        self.conv = CompilerReadyConv2d(
            weight_int=torch.tensor([[[[1, 0], [0, -1]]], [[[0, 1], [1, 0]]]], dtype=torch.int16),
            bias_int32=torch.tensor([2, -1], dtype=torch.int32),
            input_scale=0.25,
            weight_scale=0.5,
            output_scale=0.125,
            in_bits=8,
            out_bits=8,
            stride=1,
            padding=0,
            in_channels=1,
            out_channels=2,
            kernel_size=2,
        )
        self.dq_mid = CompilerDequantize(scale=0.125, zero_point=0)
        self.flatten = nn.Flatten()
        self.q_post = CompilerQuantize(scale=0.125, zero_point=0, dtype="int16")
        self.fc = CompilerReadyLinear(
            weight_int=torch.tensor([[2, -1, 0, 1, 1, -2, 1, 0], [1, 1, -2, 0, 2, -1, 1, -1]], dtype=torch.int16),
            bias_int32=torch.tensor([3, -2], dtype=torch.int32),
            input_scale=0.125,
            weight_scale=0.5,
            output_scale=0.125,
            in_bits=16,
            out_bits=16,
        )
        self.dq_out = CompilerDequantize(scale=0.125, zero_point=0)

    def forward(self, x):
        x = self.q_in(x)
        x = self.conv(x)
        x = self.dq_mid(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=1, padding=0)
        x = self.flatten(x)
        x = self.q_post(x)
        x = self.fc(x)
        x = self.dq_out(x)
        return x


def _run_output(artifact, example: torch.Tensor) -> np.ndarray:
    result = run_host_emulation(artifact, {artifact.plan.inputs[0]: example.detach().cpu().numpy()})
    return np.array(result.tensors[artifact.plan.outputs[0]], copy=True)


def _hybrid_pool_reference(model: nn.Module, example: torch.Tensor) -> np.ndarray:
    golden = GoldenModel()
    x = example.detach().cpu().numpy().astype(np.float32)
    q_in = golden.quantize(x, scale=model.q_in.scale, zero_point=model.q_in.zero_point, out_dtype=DType(model.q_in.dtype))
    image_hwc = np.transpose(q_in[0], (1, 2, 0))
    cols = golden.im2col(image_hwc, kernel_size=model.conv.kernel_size, stride=model.conv.stride, padding=model.conv.padding)
    kernel_t = model.conv.weight_int.detach().cpu().numpy().reshape(model.conv.out_channels, -1).T
    conv_matrix = golden.matmul(
        cols,
        kernel_t,
        bias=model.conv.bias_int32.detach().cpu().numpy() if model.conv.bias_int32 is not None else None,
        multiplier=model.conv.multiplier,
        shift=model.conv.shift,
        out_dtype=DType(model.conv.out_dtype),
    )
    out_h = ((image_hwc.shape[0] + (2 * model.conv.padding) - model.conv.kernel_size) // model.conv.stride) + 1
    out_w = ((image_hwc.shape[1] + (2 * model.conv.padding) - model.conv.kernel_size) // model.conv.stride) + 1
    conv_nchw = np.transpose(conv_matrix.reshape(out_h, out_w, model.conv.out_channels), (2, 0, 1))[None, ...]
    conv_float = golden.dequantize(conv_nchw, scale=model.dq_mid.scale, zero_point=model.dq_mid.zero_point).astype(np.float32)
    pooled = model.pool(torch.from_numpy(conv_float)).detach().cpu().numpy().astype(np.float32)
    flat = pooled.reshape(pooled.shape[0], -1)
    q_post = golden.quantize(flat, scale=model.q_post.scale, zero_point=model.q_post.zero_point, out_dtype=DType(model.q_post.dtype))
    linear_out = golden.matmul(
        q_post,
        model.fc.weight_int.detach().cpu().numpy().T,
        bias=model.fc.bias_int32.detach().cpu().numpy() if model.fc.bias_int32 is not None else None,
        multiplier=model.fc.multiplier,
        shift=model.fc.shift,
        out_dtype=DType(model.fc.out_dtype),
    )
    return golden.dequantize(linear_out, scale=model.dq_out.scale, zero_point=model.dq_out.zero_point).astype(np.float32)


def _hybrid_function_pool_reference(model: nn.Module, example: torch.Tensor) -> np.ndarray:
    golden = GoldenModel()
    x = example.detach().cpu().numpy().astype(np.float32)
    q_in = golden.quantize(x, scale=model.q_in.scale, zero_point=model.q_in.zero_point, out_dtype=DType(model.q_in.dtype))
    image_hwc = np.transpose(q_in[0], (1, 2, 0))
    cols = golden.im2col(image_hwc, kernel_size=model.conv.kernel_size, stride=model.conv.stride, padding=model.conv.padding)
    kernel_t = model.conv.weight_int.detach().cpu().numpy().reshape(model.conv.out_channels, -1).T
    conv_matrix = golden.matmul(
        cols,
        kernel_t,
        bias=model.conv.bias_int32.detach().cpu().numpy() if model.conv.bias_int32 is not None else None,
        multiplier=model.conv.multiplier,
        shift=model.conv.shift,
        out_dtype=DType(model.conv.out_dtype),
    )
    out_h = ((image_hwc.shape[0] + (2 * model.conv.padding) - model.conv.kernel_size) // model.conv.stride) + 1
    out_w = ((image_hwc.shape[1] + (2 * model.conv.padding) - model.conv.kernel_size) // model.conv.stride) + 1
    conv_nchw = np.transpose(conv_matrix.reshape(out_h, out_w, model.conv.out_channels), (2, 0, 1))[None, ...]
    conv_float = golden.dequantize(conv_nchw, scale=model.dq_mid.scale, zero_point=model.dq_mid.zero_point).astype(np.float32)
    pooled = F.avg_pool2d(torch.from_numpy(conv_float), kernel_size=2, stride=1, padding=0).detach().cpu().numpy().astype(np.float32)
    flat = pooled.reshape(pooled.shape[0], -1)
    q_post = golden.quantize(flat, scale=model.q_post.scale, zero_point=model.q_post.zero_point, out_dtype=DType(model.q_post.dtype))
    linear_out = golden.matmul(
        q_post,
        model.fc.weight_int.detach().cpu().numpy().T,
        bias=model.fc.bias_int32.detach().cpu().numpy() if model.fc.bias_int32 is not None else None,
        multiplier=model.fc.multiplier,
        shift=model.fc.shift,
        out_dtype=DType(model.fc.out_dtype),
    )
    return golden.dequantize(linear_out, scale=model.dq_out.scale, zero_point=model.dq_out.zero_point).astype(np.float32)


def test_semantic_frontend_matches_legacy_for_compiler_ready_linear_chain():
    model = TinySemanticLinearModel().eval()
    example = torch.tensor([0.5, -0.25, 0.75], dtype=torch.float32)

    legacy = compile_module_legacy(model, (example,))
    semantic = compile_module(model, (example,))

    legacy_out = _run_output(legacy, example)
    semantic_out = _run_output(semantic, example)

    np.testing.assert_array_equal(semantic_out, legacy_out)
    assert sum(1 for step in semantic.plan.steps if step.__class__.__name__ == "NpuSegment") == 2


def test_semantic_frontend_matches_legacy_for_compiler_ready_conv():
    model = TinySemanticConvModel().eval()
    example = torch.tensor([[[[0.25, -0.25, 0.5], [0.0, 0.75, -0.5], [0.25, 0.5, -0.25]]]], dtype=torch.float32)

    legacy = compile_module_legacy(model, (example,))
    semantic = compile_module(model, (example,))

    legacy_out = _run_output(legacy, example)
    semantic_out = _run_output(semantic, example)

    np.testing.assert_array_equal(semantic_out, legacy_out)
    assert sum(1 for step in semantic.plan.steps if step.__class__.__name__ == "NpuSegment") == 1


def test_semantic_frontend_reports_non_fuseable_activation():
    model = InvalidSemanticActivationModel().eval()
    example = torch.tensor([0.25, -0.5], dtype=torch.float32)

    with pytest.raises(NotImplementedError, match="capability check failed"):
        compile_module(model, (example,))


def test_compiler_dequantize_stub_lowers_fp16_bits_to_host_op():
    model = TinySemanticFp16DequantModel().eval()
    example = torch.tensor([[0.25, -0.25, 0.5, 0.0, 0.75, -0.5, 1.0, -1.0]], dtype=torch.float32)

    artifact = compile_module(model, (example,))

    assert artifact.plan.outputs == ["dq"]
    assert artifact.plan.tensors["dq"].dtype == DType.INT16
    assert artifact.plan.tensors["dq"].metadata.get("value_encoding") == "fp16_bits"
    segment = next(step for step in artifact.plan.steps if step.__class__.__name__ == "NpuSegment")
    assert segment.outputs == ["fc"]
    dq_step = next(step for step in artifact.plan.steps if getattr(step, "name", "") == "dq")
    assert dq_step.kind == "dequantize"
    assert dq_step.inputs == ["fc"]
    assert dq_step.outputs == ["dq"]


def test_semantic_frontend_supports_plain_float_linear_chain():
    model = TinyPlainLinearModel().eval()
    example = torch.tensor([[0.5, -0.25, 0.75], [-1.0, 0.5, 0.25]], dtype=torch.float32)

    artifact = compile_module(model, (example,))
    semantic_out = _run_output(artifact, example)
    expected = model(example).detach().cpu().numpy()

    np.testing.assert_allclose(semantic_out, expected, atol=1.0e-6, rtol=1.0e-6)
    assert sum(1 for step in artifact.plan.steps if step.__class__.__name__ == "NpuSegment") == 0
    source = emit_cv32e40p_program_v2(artifact, {artifact.plan.inputs[0]: example.detach().cpu().numpy()}, program_name="unit_test_semantic_linear")
    assert "TNPU_HOST_LINEAR" in source
    assert ".input2_idx = " in source


def test_semantic_frontend_supports_plain_float_conv2d():
    model = TinyPlainConvModel().eval()
    example = torch.tensor([[[[0.25, -0.25, 0.5], [0.0, 0.75, -0.5], [0.25, 0.5, -0.25]]]], dtype=torch.float32)

    artifact = compile_module(model, (example,))
    semantic_out = _run_output(artifact, example)
    expected = model(example).detach().cpu().numpy()

    np.testing.assert_allclose(semantic_out, expected, atol=1.0e-6, rtol=1.0e-6)
    assert sum(1 for step in artifact.plan.steps if step.__class__.__name__ == "NpuSegment") == 0
    source = emit_cv32e40p_program_v2(artifact, {artifact.plan.inputs[0]: example.detach().cpu().numpy()}, program_name="unit_test_semantic_conv")
    assert "TNPU_HOST_CONV2D" in source


def test_semantic_frontend_supports_plain_float_batched_conv2d():
    model = TinyPlainConvBatchModel().eval()
    example = torch.tensor(
        [
            [[[0.25, -0.25, 0.5], [0.0, 0.75, -0.5], [0.25, 0.5, -0.25]]],
            [[[-0.5, 0.1, 0.3], [0.2, -0.1, 0.4], [0.75, -0.25, 0.0]]],
        ],
        dtype=torch.float32,
    )

    artifact = compile_module(model, (example,))
    semantic_out = _run_output(artifact, example)
    expected = model(example).detach().cpu().numpy()

    np.testing.assert_allclose(semantic_out, expected, atol=1.0e-6, rtol=1.0e-6)
    assert sum(1 for step in artifact.plan.steps if step.__class__.__name__ == "NpuSegment") == 0


def test_semantic_frontend_supports_flatten_then_linear():
    model = TinyFlattenLinearModel().eval()
    example = torch.tensor([[[[0.25, -0.25], [0.5, 0.0]], [[-0.5, 0.75], [0.25, -0.25]]]], dtype=torch.float32)

    artifact = compile_module(model, (example,))
    semantic_out = _run_output(artifact, example)
    expected = model(example).detach().cpu().numpy()

    np.testing.assert_allclose(semantic_out, expected, atol=1.0e-6, rtol=1.0e-6)
    source = emit_cv32e40p_program_v2(artifact, {artifact.plan.inputs[0]: example.detach().cpu().numpy()}, program_name="unit_test_semantic_flatten_linear")
    assert "TNPU_HOST_RESHAPE" in source
    assert "TNPU_HOST_LINEAR" in source


def test_semantic_frontend_supports_add_and_mean():
    model = TinyMeanResidualModel().eval()
    example = torch.tensor([[0.5, -0.25, 0.75, 0.1], [-1.0, 0.5, 0.25, -0.5]], dtype=torch.float32)

    artifact = compile_module(model, (example,))
    semantic_out = _run_output(artifact, example)
    expected = model(example).detach().cpu().numpy()

    np.testing.assert_allclose(semantic_out, expected, atol=1.0e-6, rtol=1.0e-6)
    source = emit_cv32e40p_program_v2(artifact, {artifact.plan.inputs[0]: example.detach().cpu().numpy()}, program_name="unit_test_semantic_add_mean")
    assert "TNPU_HOST_ADD" in source
    assert "TNPU_HOST_MEAN" in source


def test_semantic_frontend_supports_broadcast_add_and_mul():
    model = TinyBroadcastBinaryModel().eval()
    example = torch.tensor([[0.5, -0.25, 0.75, 0.1], [-1.0, 0.5, 0.25, -0.5]], dtype=torch.float32)

    artifact = compile_module(model, (example,))
    semantic_out = _run_output(artifact, example)
    expected = model(example).detach().cpu().numpy()

    np.testing.assert_allclose(semantic_out, expected, atol=1.0e-6, rtol=1.0e-6)
    source = emit_cv32e40p_program_v2(artifact, {artifact.plan.inputs[0]: example.detach().cpu().numpy()}, program_name="unit_test_semantic_broadcast_binary")
    assert "TNPU_HOST_ADD" in source
    assert "TNPU_HOST_MUL" in source


@pytest.mark.parametrize(
    ("model_cls", "enum_name"),
    [
        (TinyFunctionMaxPoolModel, "TNPU_HOST_MAXPOOL2D"),
        (TinyFunctionAdaptiveAvgPoolModel, "TNPU_HOST_ADAPTIVE_AVGPOOL2D"),
    ],
)
def test_semantic_frontend_supports_function_form_pooling_on_cpu_only_path(model_cls, enum_name):
    model = model_cls().eval()
    example = torch.tensor([[[[0.25, -0.25, 0.5, 0.75], [0.0, 0.75, -0.5, 0.25], [0.25, 0.5, -0.25, -0.75], [0.5, 0.1, -0.2, 0.3]]]], dtype=torch.float32)

    artifact = compile_module(model, (example,))
    semantic_out = _run_output(artifact, example)
    expected = model(example).detach().cpu().numpy()

    np.testing.assert_allclose(semantic_out, expected, atol=1.0e-6, rtol=1.0e-6)
    assert sum(1 for step in artifact.plan.steps if step.__class__.__name__ == "NpuSegment") == 0
    source = emit_cv32e40p_program_v2(
        artifact,
        {artifact.plan.inputs[0]: example.detach().cpu().numpy()},
        program_name=f"unit_test_{model_cls.__name__.lower()}",
    )
    assert enum_name in source


@pytest.mark.parametrize(
    ("model_cls", "enum_name"),
    [
        (TinyMaxPoolLinearModel, "TNPU_HOST_MAXPOOL2D"),
        (TinyAvgPoolLinearModel, "TNPU_HOST_AVGPOOL2D"),
        (TinyAdaptiveAvgPoolLinearModel, "TNPU_HOST_ADAPTIVE_AVGPOOL2D"),
    ],
)
def test_semantic_frontend_supports_pooling_on_cpu_only_path(model_cls, enum_name):
    model = model_cls().eval()
    example = torch.tensor([[[[0.25, -0.25, 0.5, 0.75], [0.0, 0.75, -0.5, 0.25], [0.25, 0.5, -0.25, -0.75], [0.5, 0.1, -0.2, 0.3]]]], dtype=torch.float32)

    artifact = compile_module(model, (example,))
    semantic_out = _run_output(artifact, example)
    expected = model(example).detach().cpu().numpy()

    np.testing.assert_allclose(semantic_out, expected, atol=1.0e-6, rtol=1.0e-6)
    assert sum(1 for step in artifact.plan.steps if step.__class__.__name__ == "NpuSegment") == 0
    source = emit_cv32e40p_program_v2(
        artifact,
        {artifact.plan.inputs[0]: example.detach().cpu().numpy()},
        program_name=f"unit_test_{model_cls.__name__.lower()}",
    )
    assert enum_name in source


@pytest.mark.parametrize(
    ("model_cls", "enum_name"),
    [
        (TinyHybridMaxPoolModel, "TNPU_HOST_MAXPOOL2D"),
        (TinyHybridAvgPoolModel, "TNPU_HOST_AVGPOOL2D"),
        (TinyHybridAdaptiveAvgPoolModel, "TNPU_HOST_ADAPTIVE_AVGPOOL2D"),
    ],
)
def test_semantic_frontend_supports_pooling_on_hybrid_npu_path(model_cls, enum_name):
    model = model_cls().eval()
    example = torch.tensor([[[[0.25, -0.25, 0.5, 0.75], [0.0, 0.75, -0.5, 0.25], [0.25, 0.5, -0.25, -0.75], [0.5, 0.1, -0.2, 0.3]]]], dtype=torch.float32)

    artifact = compile_module(model, (example,))
    semantic_out = _run_output(artifact, example)
    expected = _hybrid_pool_reference(model, example)

    np.testing.assert_allclose(semantic_out, expected, atol=1.0e-6, rtol=1.0e-6)
    assert sum(1 for step in artifact.plan.steps if step.__class__.__name__ == "NpuSegment") == 2
    source = emit_cv32e40p_program_v2(
        artifact,
        {artifact.plan.inputs[0]: example.detach().cpu().numpy()},
        program_name=f"unit_test_{model_cls.__name__.lower()}",
    )
    assert enum_name in source


def test_semantic_frontend_supports_function_form_pooling_on_hybrid_npu_path():
    model = TinyHybridFunctionAvgPoolModel().eval()
    example = torch.tensor([[[[0.25, -0.25, 0.5, 0.75], [0.0, 0.75, -0.5, 0.25], [0.25, 0.5, -0.25, -0.75], [0.5, 0.1, -0.2, 0.3]]]], dtype=torch.float32)

    artifact = compile_module(model, (example,))
    semantic_out = _run_output(artifact, example)
    expected = _hybrid_function_pool_reference(model, example)

    np.testing.assert_allclose(semantic_out, expected, atol=1.0e-6, rtol=1.0e-6)
    assert sum(1 for step in artifact.plan.steps if step.__class__.__name__ == "NpuSegment") == 2
    source = emit_cv32e40p_program_v2(
        artifact,
        {artifact.plan.inputs[0]: example.detach().cpu().numpy()},
        program_name="unit_test_tinyhybridfunctionavgpoolmodel",
    )
    assert "TNPU_HOST_AVGPOOL2D" in source
