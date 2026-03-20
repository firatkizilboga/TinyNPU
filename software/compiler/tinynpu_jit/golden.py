from __future__ import annotations

import numpy as np

from .ir import DType


def di_exp(x_in: int, *, m_i: int, k_i: int) -> int:
    """
    Software reference for Appendix A.2 Algorithm 1 (DI-Exp).

    This follows the paper's integer interface directly: integer input ``x_in``,
    integer multiplier-like parameter ``m_i``, and integer shift factor ``k_i``.
    The implementation keeps the paper's structure but spells out the reciprocal
    term explicitly so the code remains numerically well defined.
    """
    if m_i <= 0:
        raise ValueError(f"m_i must be positive, got {m_i}.")
    if k_i < 0:
        raise ValueError(f"k_i must be non-negative, got {k_i}.")

    # Algorithm 1, line 1: m_f = m_i + (m_i >> 1) - (m_i >> 4).
    # This is the paper's shift-add approximation used before the reciprocal step.
    m_f = int(m_i + (m_i >> 1) - (m_i >> 4))
    if m_f <= 0:
        raise ValueError(f"Derived m_f must be positive, got {m_f}.")

    # Algorithm 1, lines 2-3: the paper writes s_f = m_f >> k_i and
    # t = round(-1 / s_f). To keep the integer arithmetic explicit, compute the
    # reciprocal period directly as round(2^k_i / m_f), then negate it to obtain t.
    s_i = ((1 << k_i) + (m_f // 2)) // m_f
    if s_i <= 0:
        raise ValueError(
            f"Chosen fixed parameters collapse DI-Exp period to zero: m_i={m_i}, k_i={k_i}, m_f={m_f}."
        )
    t = -int(s_i)

    # Algorithm 1, lines 4-5: q_i = floor(x_in / t), r_i = x_in - q_i * t.
    q_i = int(x_in // t)
    r_i = int(x_in - (q_i * t))

    # Algorithm 1, line 6: unshifted_exp = (r_i >> 1) - t.
    unshifted_exp = int((r_i >> 1) - t)

    # Algorithm 1, line 7: result = unshifted_exp >> q_i.
    result = int(unshifted_exp >> q_i)
    return max(0, result)


def _int_div_prob(numer: int, denom: int, *, p_out: int) -> int:
    if p_out <= 0:
        raise ValueError(f"p_out must be positive, got {p_out}.")
    if denom <= 0:
        raise ValueError(f"Probability denominator must be positive, got {denom}.")

    scale = (1 << (p_out - 1)) - 1
    return int((numer * scale + (denom // 2)) // denom)


def di_sigmoid(x_in: int, *, m_i: int, k_i: int, p_out: int = 8, alpha_smooth: int = 1) -> int:
    """
    Scalar DI-Sigmoid built from DI-Exp and the integer division pattern used by
    the paper's DI-Softmax/DI-SwiGLU appendix.

    The input is optionally smoothed, then mapped to a scalar sigmoid using
    sigma(x) = 1 / (1 + exp(-x)) for x >= 0 and
    sigma(x) = exp(x) / (1 + exp(x)) for x < 0.
    """
    if alpha_smooth <= 0:
        raise ValueError(f"alpha_smooth must be positive, got {alpha_smooth}.")

    x_smoothed = int(x_in // alpha_smooth)
    exp_zero = di_exp(0, m_i=m_i, k_i=k_i)

    # Keep DI-Exp on its natural non-positive domain.
    if x_smoothed >= 0:
        exp_term = di_exp(-x_smoothed, m_i=m_i, k_i=k_i)
        numer = exp_zero
    else:
        exp_term = di_exp(x_smoothed, m_i=m_i, k_i=k_i)
        numer = exp_term

    denom = exp_zero + exp_term
    return _int_div_prob(numer, denom, p_out=p_out)


def _round_shift_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return int(value)
    rounder = 1 << (shift - 1)
    if value >= 0:
        return int((value + rounder) >> shift)
    return -int(((-value) + rounder) >> shift)


def _round_div_signed(numer: int, denom: int) -> int:
    if denom <= 0:
        raise ValueError(f"denom must be positive, got {denom}.")
    half = denom // 2
    if numer >= 0:
        return int((numer + half) // denom)
    return -int(((-numer) + half) // denom)


def h_gelu(x_in: int, *, x_scale_shift: int = 7, slope_num: int = 218, slope_shift: int = 7) -> int:
    """
    Integer hard-GELU approximation using

        h_gelu(x) = x * ReLU6(1.702x + 3) / 6

    with ``x_in`` and the output represented in the same fixed-point domain:

        x_real ~= x_in / 2^x_scale_shift

    The default ``slope_num / 2^slope_shift`` approximates 1.702 as 218 / 128.
    """
    if x_scale_shift < 0:
        raise ValueError(f"x_scale_shift must be non-negative, got {x_scale_shift}.")
    if slope_num <= 0:
        raise ValueError(f"slope_num must be positive, got {slope_num}.")
    if slope_shift < 0:
        raise ValueError(f"slope_shift must be non-negative, got {slope_shift}.")

    scale_denom = 1 << x_scale_shift
    three_int = 3 * scale_denom
    six_int = 6 * scale_denom

    slope_term = _round_shift_signed(int(x_in) * int(slope_num), slope_shift)
    gate_int = min(max(slope_term + three_int, 0), six_int)

    return _round_div_signed(int(x_in) * gate_int, six_int)


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

    def di_exp(self, x_in: int, *, m_i: int, k_i: int) -> int:
        return di_exp(x_in, m_i=m_i, k_i=k_i)

    def di_sigmoid(self, x_in: int, *, m_i: int, k_i: int, p_out: int = 8, alpha_smooth: int = 1) -> int:
        return di_sigmoid(x_in, m_i=m_i, k_i=k_i, p_out=p_out, alpha_smooth=alpha_smooth)

    def h_gelu(self, x_in: int, *, x_scale_shift: int = 7, slope_num: int = 218, slope_shift: int = 7) -> int:
        return h_gelu(x_in, x_scale_shift=x_scale_shift, slope_num=slope_num, slope_shift=slope_shift)

    def quantized_mean(
        self,
        value,
        *,
        axis=None,
        keepdims: bool = False,
        zero_point: int = 0,
        out_dtype: DType = DType.INT16,
    ) -> np.ndarray:
        source = np.array(value, dtype=np.float32)
        averaged = np.mean(source - np.float32(zero_point), axis=axis, keepdims=keepdims)
        requantized = np.rint(averaged).astype(np.int64) + np.int64(zero_point)
        if out_dtype == DType.INT4:
            return np.clip(requantized, -8, 7).astype(np.int16)
        if out_dtype == DType.INT8:
            return np.clip(requantized, -128, 127).astype(np.int16)
        if out_dtype == DType.INT16:
            return np.clip(requantized, -32768, 32767).astype(np.int16)
        if out_dtype == DType.INT32:
            return np.clip(requantized, np.iinfo(np.int32).min, np.iinfo(np.int32).max).astype(np.int32)
        raise ValueError(f"Unsupported quantized_mean dtype {out_dtype}.")

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
        h_gelu_x_scale_shift: int = 7,
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
                    h_gelu_x_scale_shift=h_gelu_x_scale_shift,
                    out_dtype=out_dtype,
                )
        return out

    def _ppu(
        self,
        acc: int,
        bias: int,
        multiplier: int,
        shift: int,
        activation: str,
        h_gelu_x_scale_shift: int,
        out_dtype: DType,
    ) -> int:
        value = np.int64(acc) + np.int64(np.int32(bias))
        value *= np.int64(multiplier & 0xFFFF)
        if shift > 0:
            value = (value + (np.int64(1) << (shift - 1))) >> shift
        if activation == "relu":
            value = max(0, value)
        elif activation == "sigmoid":
            if out_dtype == DType.INT4:
                p_out = 4
            elif out_dtype == DType.INT8:
                p_out = 8
            else:
                p_out = 16
            clamped = int(np.clip(value, -32768, 32767))
            value = np.int64(di_sigmoid(clamped, m_i=int(multiplier & 0xFFFF), k_i=int(shift), p_out=p_out))
        elif activation == "h_gelu":
            clamped = int(np.clip(value, -32768, 32767))
            value = np.int64(h_gelu(clamped, x_scale_shift=int(h_gelu_x_scale_shift)))
        if out_dtype == DType.INT4:
            return int(np.clip(value, -8, 7))
        if out_dtype == DType.INT8:
            return int(np.clip(value, -128, 127))
        return int(np.clip(value, -32768, 32767))
