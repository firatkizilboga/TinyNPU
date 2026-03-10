import numpy as np
import os
import json
import sys
import math

# Add compiler to path
sys.path.append(os.path.join(os.getcwd(), "software/compiler"))
from tinynpu import TinyNPUProgram, PrecisionMode
from software.compiler.tinynpu_jit import im2col_for_npu as jit_im2col_for_npu
from software.compiler.tinynpu_jit import npu_matmul as jit_npu_matmul


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


def global_average_pool_for_fc_input(data):
    arr = np.array(data, dtype=np.int32)
    if arr.ndim != 3:
        raise ValueError(f"global_average_pool_for_fc_input expects HWC tensor, got shape {arr.shape}.")
    gap = np.mean(arr, axis=(0, 1))
    return gap.reshape(-1, 1).astype(np.int16)

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
    - exported linear layers
    - exported conv2d layers lowered through host-side im2col
    - explicit quantized matmul + bias lowering
    """
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        raise ImportError("torch is required for compile_mnist_layer_jit().") from exc

    from software.compiler.tinynpu_jit import compile_module

    a_bits = layer_info.get('a_bits', 16)
    input_hw = prepare_activation_for_hw(input_data, a_bits)

    in_dtype_name = f"int{int(layer_info['a_bits'])}"
    if int(layer_info['w_bits']) != int(layer_info['a_bits']):
        raise ValueError(
            f"JIT MNIST path currently requires a_bits == w_bits, got a_bits={layer_info['a_bits']} "
            f"and w_bits={layer_info['w_bits']} for layer {name!r}."
        )
    if int(layer_info['a_bits']) == 8:
        input_hw = input_hw.astype(np.int8)
        weight_dtype = np.int8
    elif int(layer_info['a_bits']) == 16:
        input_hw = input_hw.astype(np.int16)
        weight_dtype = np.int16
    else:
        raise ValueError(f"Unsupported JIT MNIST activation precision a_bits={layer_info['a_bits']}.")

    if layer_info['type'] == 'linear':
        w = np.load(os.path.join(export_dir, f"{name}_weights.npy")).astype(weight_dtype)
        b = np.load(os.path.join(export_dir, f"{name}_bias.npy")).astype(np.int32)

        class ExportedLinearModule(nn.Module):
            def __init__(self, weight, bias):
                super().__init__()
                self.register_buffer("weight", torch.tensor(weight))
                self.register_buffer("bias", torch.tensor(bias))

            def forward(self, x):
                y = jit_npu_matmul(
                    self.weight,
                    x,
                    multiplier=int(layer_info["M0"]),
                    shift=int(layer_info["shift"]),
                    activation="none",
                    in_dtype=in_dtype_name,
                    out_dtype="int16",
                )
                return y + self.bias.reshape(-1, 1)

        module = ExportedLinearModule(w, b)
        example = torch.tensor(input_hw)
        return compile_module(module, (example,))

    if layer_info['type'] == 'conv2d':
        w_gemm = np.load(os.path.join(export_dir, f"{name}_weights_gemm.npy")).astype(weight_dtype)
        kernel_t = w_gemm.T
        b = np.load(os.path.join(export_dir, f"{name}_bias.npy")).astype(np.int32)
        kernel_size = int(layer_info["kernel_size"])
        stride = int(layer_info["stride"])
        padding = int(layer_info["padding"])
        out_h = ((input_hw.shape[0] + 2 * padding - kernel_size) // stride) + 1
        out_w = ((input_hw.shape[1] + 2 * padding - kernel_size) // stride) + 1
        out_channels = int(layer_info["out_channels"])

        class ExportedConvModule(nn.Module):
            def __init__(self, kernel_t, bias):
                super().__init__()
                self.register_buffer("kernel_t", torch.tensor(kernel_t))
                self.register_buffer("bias", torch.tensor(bias))

            def forward(self, x):
                cols = jit_im2col_for_npu(x, kernel_size, stride, padding)
                y = jit_npu_matmul(
                    cols,
                    self.kernel_t,
                    multiplier=int(layer_info["M0"]),
                    shift=int(layer_info["shift"]),
                    activation="relu",
                    in_dtype=in_dtype_name,
                    out_dtype=in_dtype_name,
                )
                y = y + self.bias.reshape(1, -1)
                return y.reshape(out_h, out_w, out_channels)

        module = ExportedConvModule(kernel_t, b)
        example = torch.tensor(input_hw)
        return compile_module(module, (example,))

    raise NotImplementedError(
        f"JIT MNIST path currently supports only linear and conv2d layers, got {layer_info['type']!r}."
    )
