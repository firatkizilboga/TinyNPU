from __future__ import annotations

import numpy as np

from .ir import DType


class GoldenModel:
    def quantize(
        self,
        value,
        *,
        scale: float,
        zero_point: int = 0,
        out_dtype: DType = DType.INT8,
    ) -> np.ndarray:
        if scale <= 0:
            raise ValueError(f"Quantization scale must be positive, got {scale}.")
        source = np.array(value, dtype=np.float32)
        quantized = np.rint(source / np.float32(scale)).astype(np.int64) + np.int64(zero_point)
        if out_dtype == DType.INT4:
            return np.clip(quantized, -8, 7).astype(np.int16)
        if out_dtype == DType.INT8:
            return np.clip(quantized, -128, 127).astype(np.int16)
        if out_dtype == DType.INT16:
            return np.clip(quantized, -32768, 32767).astype(np.int16)
        if out_dtype == DType.INT32:
            return np.clip(quantized, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)
        raise ValueError(f"Unsupported quantize dtype {out_dtype}.")

    def dequantize(
        self,
        value,
        *,
        scale: float,
        zero_point: int = 0,
    ) -> np.ndarray:
        if scale <= 0:
            raise ValueError(f"Quantization scale must be positive, got {scale}.")
        source = np.array(value, dtype=np.float32)
        return (source - np.float32(zero_point)) * np.float32(scale)

    def im2col(
        self,
        image,
        *,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ) -> np.ndarray:
        img = np.array(image, copy=False)
        if img.ndim != 3:
            raise ValueError(f"im2col expects HWC image input, got shape {img.shape}.")
        h, w, c = img.shape
        if padding > 0:
            img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), mode="constant")

        patches = []
        for y in range(0, h + 2 * padding - kernel_size + 1, stride):
            for x in range(0, w + 2 * padding - kernel_size + 1, stride):
                patch = img[y : y + kernel_size, x : x + kernel_size, :]
                patches.append(patch.transpose(2, 0, 1).reshape(-1))
        return np.array(patches, dtype=img.dtype)

    def coerce_npu_input(
        self,
        value,
        *,
        out_dtype: DType = DType.INT16,
        tensor_name: str | None = None,
    ) -> np.ndarray:
        source = np.array(value, copy=False)
        if np.issubdtype(source.dtype, np.integer):
            if out_dtype == DType.INT4:
                return np.clip(source, -8, 7).astype(np.int16, copy=False)
            if out_dtype == DType.INT8:
                return np.clip(source, -128, 127).astype(np.int16, copy=False)
            if out_dtype == DType.INT16:
                return source.astype(np.int16, copy=False)
            if out_dtype == DType.INT32:
                return source.astype(np.int32, copy=False)
            raise ValueError(f"Unsupported NPU input dtype {out_dtype}.")

        rounded = np.rint(source)
        if not np.allclose(source, rounded, rtol=0.0, atol=1e-6):
            name = f" '{tensor_name}'" if tensor_name else ""
            raise NotImplementedError(
                f"Tensor{name} is floating-point at an NPU boundary. "
                "Insert quantize_for_npu(...) before feeding it into a TinyNPU segment."
            )

        if out_dtype == DType.INT4:
            return np.clip(rounded, -8, 7).astype(np.int16)
        if out_dtype == DType.INT8:
            return np.clip(rounded, -128, 127).astype(np.int16)
        if out_dtype == DType.INT16:
            return rounded.astype(np.int16)
        if out_dtype == DType.INT32:
            return rounded.astype(np.int32)
        raise ValueError(f"Unsupported NPU input dtype {out_dtype}.")

    def softmax(self, value, *, axis: int = -1) -> np.ndarray:
        source = np.array(value, dtype=np.float32)
        shifted = source - np.max(source, axis=axis, keepdims=True)
        exp = np.exp(shifted)
        return exp / np.sum(exp, axis=axis, keepdims=True)

    def requantize(
        self,
        value,
        *,
        scale: float,
        zero_point: int = 0,
        out_dtype: DType = DType.INT16,
    ) -> np.ndarray:
        source = np.array(value, dtype=np.float32)
        quantized = np.rint(source * np.float32(scale)).astype(np.int64) + np.int64(zero_point)
        if out_dtype == DType.INT4:
            return np.clip(quantized, -8, 7).astype(np.int16)
        if out_dtype == DType.INT8:
            return np.clip(quantized, -128, 127).astype(np.int16)
        if out_dtype == DType.INT16:
            return np.clip(quantized, -32768, 32767).astype(np.int16)
        raise ValueError(f"Unsupported requantize dtype {out_dtype}.")

    def matmul(
        self,
        lhs,
        rhs,
        bias=None,
        multiplier: int = 1,
        shift: int = 0,
        activation: str = "none",
        out_dtype: DType = DType.INT16,
    ) -> np.ndarray:
        lhs_arr = np.array(lhs, dtype=np.int64)
        rhs_arr = np.array(rhs, dtype=np.int64)

        if bias is None:
            bias_arr = np.zeros(rhs_arr.shape[1], dtype=np.int64)
        else:
            bias_arr = np.array(bias, dtype=np.int64).flatten()

        acc = np.matmul(lhs_arr, rhs_arr)
        out = np.zeros(acc.shape, dtype=np.int32)
        for row in range(acc.shape[0]):
            for col in range(acc.shape[1]):
                out[row, col] = self._ppu(
                    acc[row, col],
                    bias_arr[col],
                    multiplier=multiplier,
                    shift=shift,
                    activation=activation,
                    out_dtype=out_dtype,
                )
        return out

    def _ppu(self, acc: int, bias: int, multiplier: int, shift: int, activation: str, out_dtype: DType) -> int:
        value = np.int64(acc) + np.int64(np.int32(bias))
        value *= np.int64(multiplier & 0xFFFF)
        if shift > 0:
            value = (value + (np.int64(1) << (shift - 1))) >> shift
        if activation == "relu":
            value = max(0, value)
        if out_dtype == DType.INT4:
            return int(np.clip(value, -8, 7))
        if out_dtype == DType.INT8:
            return int(np.clip(value, -128, 127))
        return int(np.clip(value, -32768, 32767))
