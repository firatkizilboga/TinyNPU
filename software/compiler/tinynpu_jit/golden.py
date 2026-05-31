from __future__ import annotations

import numpy as np

from .ir import DType


def di_exp(x_in: int, *, m_i: int, k_i: int) -> int:
    if m_i <= 0:
        raise ValueError(f"m_i must be positive, got {m_i}.")
    if k_i < 0:
        raise ValueError(f"k_i must be non-negative, got {k_i}.")

    m_f = int(m_i + (m_i >> 1) - (m_i >> 4))
    if m_f <= 0:
        raise ValueError(f"Derived m_f must be positive, got {m_f}.")

    s_i = ((1 << k_i) + (m_f // 2)) // m_f
    if s_i <= 0:
        raise ValueError(
            f"Chosen fixed parameters collapse DI-Exp period to zero: m_i={m_i}, k_i={k_i}, m_f={m_f}."
        )
    t = -int(s_i)

    q_i = int(x_in // t)
    r_i = int(x_in - (q_i * t))
    unshifted_exp = int((r_i >> 1) - t)
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
    if alpha_smooth <= 0:
        raise ValueError(f"alpha_smooth must be positive, got {alpha_smooth}.")

    x_smoothed = int(x_in // alpha_smooth)
    exp_zero = di_exp(0, m_i=m_i, k_i=k_i)

    if x_smoothed >= 0:
        exp_term = di_exp(-x_smoothed, m_i=m_i, k_i=k_i)
        numer = exp_zero
    else:
        exp_term = di_exp(x_smoothed, m_i=m_i, k_i=k_i)
        numer = exp_term

    denom = exp_zero + exp_term
    return _int_div_prob(numer, denom, p_out=p_out)


def ppu_hard_sigmoid(x_in: int, *, shift: int, p_out: int) -> int:
    if p_out <= 0:
        raise ValueError(f"p_out must be positive, got {p_out}.")
    if shift < 0:
        raise ValueError(f"shift must be non-negative, got {shift}.")
    bound = 8 << int(shift) if int(shift) <= 29 else 0
    qmax = (1 << (int(p_out) - 1)) - 1
    if bound <= 0 or qmax <= 0 or x_in <= -bound:
        numer = 0
    elif x_in >= bound:
        numer = qmax << (int(shift) + 4)
    else:
        numer = (int(x_in) + bound) * qmax
    numer += 1 << (int(shift) + 3)
    return int(numer >> (int(shift) + 4))


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


def _clip_signed(value: int, bits: int) -> int:
    max_value = (1 << (bits - 1)) - 1
    min_value = -(1 << (bits - 1))
    return int(min(max(int(value), min_value), max_value))


def _rtl_round_shift_positive(value: int, shift: int) -> int:
    """Match the current PPU rescale path.

    The RTL adds a positive rounding constant before arithmetic right shift.
    This is intentionally not symmetric for negative values.
    """

    if shift <= 0:
        return int(value)
    return int((int(value) + (1 << (shift - 1))) >> shift)


def h_gelu(x_in: int, *, x_scale_shift: int = 7, slope_num: int = 218, slope_shift: int = 7) -> int:
    if x_scale_shift < 0:
        raise ValueError(f"x_scale_shift must be non-negative, got {x_scale_shift}.")
    if slope_num <= 0:
        raise ValueError(f"slope_num must be positive, got {slope_num}.")
    if slope_shift < 0:
        raise ValueError(f"slope_shift must be non-negative, got {slope_shift}.")

    effective_shift = min(int(x_scale_shift), 15)
    x_ext = _clip_signed(int(x_in), 16)
    scale = 1 << effective_shift
    gate_int = _rtl_round_shift_positive(x_ext * int(slope_num), int(slope_shift)) + 3 * scale
    gate_int = min(max(gate_int, 0), 6 * scale)

    # RTL replaces division by 6 with round((x * gate) * 10923 / 2**16), then
    # applies the configured scale shift.
    div6 = _rtl_round_shift_positive(x_ext * gate_int * 10923, 16)
    return _clip_signed(_round_shift_signed(div6, effective_shift), 16)


def h_gelu_ideal(x_in: int, *, x_scale_shift: int = 7, slope_num: int = 218, slope_shift: int = 7) -> int:
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


def _clip_for_dtype(value: int, out_dtype: DType) -> int:
    if out_dtype == DType.INT4:
        return int(np.clip(value, -8, 7))
    if out_dtype == DType.INT8:
        return int(np.clip(value, -128, 127))
    if out_dtype == DType.INT16:
        return int(np.clip(value, -32768, 32767))
    if out_dtype == DType.INT32:
        return int(np.clip(value, np.iinfo(np.int32).min, np.iinfo(np.int32).max))
    raise ValueError(f"Unsupported integer dtype {out_dtype}.")


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

    def h_gelu_ideal(self, x_in: int, *, x_scale_shift: int = 7, slope_num: int = 218, slope_shift: int = 7) -> int:
        return h_gelu_ideal(x_in, x_scale_shift=x_scale_shift, slope_num=slope_num, slope_shift=slope_shift)

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
        value = _clip_signed(int(acc) + int(np.int32(bias)), 48)
        value *= int(multiplier & 0xFFFF)
        if shift > 0:
            value = _rtl_round_shift_positive(value, int(shift))
        value = _clip_signed(value, 16)
        if activation == "relu":
            value = max(0, value)
        elif activation == "sigmoid":
            p_out = 4 if out_dtype == DType.INT4 else 8 if out_dtype == DType.INT8 else 16
            return _clip_for_dtype(
                ppu_hard_sigmoid(value, shift=int(h_gelu_x_scale_shift), p_out=p_out),
                out_dtype,
            )
        elif activation == "h_gelu":
            value = self.h_gelu(value, x_scale_shift=int(h_gelu_x_scale_shift))
        return _clip_for_dtype(value, out_dtype)
