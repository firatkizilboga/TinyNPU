import numpy as np
import os
import json
import sys
import math

# Add compiler to path
sys.path.append(os.path.join(os.getcwd(), "software/compiler"))
from tinynpu import TinyNPUProgram, PrecisionMode


def precision_from_bits(bits):
    bits = int(bits)
    if bits == 4:
        return PrecisionMode.INT4
    if bits == 8:
        return PrecisionMode.INT8
    if bits == 16:
        return PrecisionMode.INT16
    raise ValueError(f"Unsupported precision bit-width: {bits}")


def prepare_activation_for_hw(data, a_bits):
    """
    Map host activations into the signed range represented on NPU input lanes.

    For INT8, this intentionally performs two's-complement wrapping so inputs
    like [128..255] are interpreted as [-128..-1], matching hardware packing.
    """
    bits = int(a_bits)
    arr = np.array(data, dtype=np.int32)
    if bits >= 16:
        return arr.astype(np.int16)

    mod = 1 << bits
    half = 1 << (bits - 1)
    wrapped = ((arr + half) % mod) - half
    return wrapped.astype(np.int16)

def get_im2col_matrix(img_data, kh, kw, stride, padding):
    H, W, C = img_data.shape
    if padding > 0:
        img_data = np.pad(img_data, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
    
    col_matrix = []
    for y in range(0, H + 2*padding - kh + 1, stride):
        for x in range(0, W + 2*padding - kw + 1, stride):
            patch = img_data[y:y+kh, x:x+kw, :]
            # Transpose to (C, H, W) to match PyTorch/Kernel format
            col_matrix.append(patch.transpose(2, 0, 1).flatten())
    # Logical layout used by compiler conv path:
    # Input matrix is (HW, K), Kernel matrix is (K, OC), Output is (HW, OC).
    return np.array(col_matrix, dtype=np.int16)

def compile_mnist_layer(name, layer_info, input_data, export_dir='./mnist_mixed_export'):
    prog = TinyNPUProgram()
    a_bits = layer_info.get('a_bits', 16)
    a_prec = precision_from_bits(a_bits)
    input_hw = prepare_activation_for_hw(input_data, a_bits)
    
    if layer_info['type'] == 'conv2d':
        w_gemm = np.load(os.path.join(export_dir, f"{name}_weights_gemm.npy"))
        b = np.load(os.path.join(export_dir, f"{name}_bias.npy"))

        # Matrix A: Input ImageCol (HW, K)
        prog.declare_data("Input", input_hw, precision=a_prec, role='A')

        # Matrix B: Kernel transposed to (K, OC)
        w_prec = precision_from_bits(layer_info['w_bits'])
        if w_prec != a_prec:
            raise ValueError(
                f"Layer {name} has mismatched input/weight precision "
                f"(a_bits={a_bits}, w_bits={layer_info['w_bits']})."
            )
        prog.declare_data("Kernel", w_gemm.T, precision=w_prec, role='B')

        # Bias
        prog.declare_data("Bias", b.reshape(1, -1), precision=PrecisionMode.INT16, role='BIAS')

        # Emit conv outputs at activation precision to keep layer buffers in UB.
        conv_out_prec = a_prec
        prog.matmul("Input", "Kernel", "Output", bias_name="Bias",
                    shift=layer_info['shift'], multiplier=layer_info['M0'],
                    activation=1,
                    in_precision=a_prec, out_precision=conv_out_prec)
        
    elif layer_info['type'] == 'linear':
        w = np.load(os.path.join(export_dir, f"{name}_weights.npy"))
        b = np.load(os.path.join(export_dir, f"{name}_bias.npy"))
        w_prec = precision_from_bits(layer_info['w_bits'])
        if w_prec != a_prec:
            raise ValueError(
                f"Layer {name} has mismatched input/weight precision "
                f"(a_bits={a_bits}, w_bits={layer_info['w_bits']})."
            )
        prog.declare_data("Input", input_hw, precision=a_prec, role='B')
        prog.declare_data("Weight", w, precision=w_prec, role='A')
        prog.declare_data("Bias", b.reshape(1, -1), precision=PrecisionMode.INT16, role='BIAS')
        prog.matmul("Weight", "Input", "Output", bias_name="Bias",
                    shift=layer_info['shift'], multiplier=layer_info['M0'],
                    activation=0,
                    in_precision=a_prec, out_precision=PrecisionMode.INT16)

    prog.halt()
    prog.compile()
    return prog


def compile_mnist_layer_jit(name, layer_info, input_data, export_dir='./mnist_mixed_export'):
    """
    Compile a supported MNIST layer through the new PyTorch-facing JIT path.

    Current support is intentionally narrow:
    - linear layers only
    - explicit matmul + bias lowering
    """
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise ImportError("torch is required for compile_mnist_layer_jit().") from exc

    from software.compiler.tinynpu_jit import compile_module

    a_bits = layer_info.get('a_bits', 16)
    input_hw = prepare_activation_for_hw(input_data, a_bits)

    if layer_info['type'] != 'linear':
        raise NotImplementedError(
            f"JIT MNIST path currently supports only linear layers, got {layer_info['type']!r}."
        )

    w = np.load(os.path.join(export_dir, f"{name}_weights.npy")).astype(np.int16)
    b = np.load(os.path.join(export_dir, f"{name}_bias.npy")).astype(np.int32)

    class ExportedLinearModule(nn.Module):
        def __init__(self, weight, bias):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(weight.astype(np.float32)), requires_grad=False)
            self.bias = nn.Parameter(torch.tensor(bias.astype(np.float32)), requires_grad=False)

        def forward(self, x):
            y = torch.matmul(self.weight, x)
            return y + self.bias.reshape(-1, 1)

    module = ExportedLinearModule(w, b)
    example = torch.tensor(input_hw.astype(np.float32))
    return compile_module(module, (example,))
