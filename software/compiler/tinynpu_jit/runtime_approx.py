from __future__ import annotations

import math

import numpy as np


def fp16_roundtrip(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).astype(np.float16).astype(np.float32)


def float32_to_fp16_bits(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).astype(np.float16).view(np.uint16)


def choose_xform_q_f16_i16_scale_params(inv_scale: float) -> tuple[int, int]:
    best_err = float("inf")
    best_mult = 1
    best_shift = 0
    inv_scale_f = float(inv_scale)
    if inv_scale_f <= 0.0:
        raise ValueError(f"xform inv_scale must be positive, got {inv_scale}.")
    if inv_scale_f >= 1.0:
        rounded = int(inv_scale_f + 0.5)
        if float(rounded) == inv_scale_f and rounded <= 65535:
            return rounded, 0
    for shift in range(16):
        scaled = inv_scale_f * float(1 << shift)
        if scaled > 65535.0:
            break
        mult = int(scaled + 0.5)
        if mult == 0:
            mult = 1
        approx = float(mult) / float(1 << shift)
        err = abs(approx - inv_scale_f)
        if err < best_err or (err == best_err and shift > best_shift):
            best_err = err
            best_mult = mult
            best_shift = shift
    if math.isinf(best_err):
        return 65535, 0
    return best_mult, best_shift


def _round_shift_right_signed(value: int, shift: int) -> int:
    if shift <= 0:
        return value
    if shift >= 63:
        return -1 if value < 0 else 0
    if value >= 0:
        return (value + (1 << (shift - 1))) >> shift
    return -((-value + (1 << (shift - 1))) >> shift)


def _quantize_fp16_lane_to_i16(fp16: int, multiplier: int, shift: int) -> int:
    sign = (fp16 >> 15) & 1
    exp_bits = (fp16 >> 10) & 0x1F
    frac_bits = fp16 & 0x3FF

    if multiplier == 0:
        return 0
    if exp_bits == 0x1F:
        return -32768 if sign else 32767
    if exp_bits == 0 and frac_bits == 0:
        return 0

    if exp_bits == 0:
        mant = frac_bits
        exp2 = -24
    else:
        mant = 1024 + frac_bits
        exp2 = exp_bits - 25

    scaled = mant * multiplier
    if exp2 >= shift:
        left_shift = exp2 - shift
        qvalue = (1 << 63) - 1 if left_shift >= 47 else scaled << left_shift
    else:
        qvalue = _round_shift_right_signed(scaled, shift - exp2)
    if sign:
        qvalue = -qvalue
    return max(-32768, min(32767, qvalue))


def quantize_fp16_to_i16_xform(source: np.ndarray, *, scale: float) -> np.ndarray:
    inv_scale = 1.0 / float(scale)
    multiplier, shift = choose_xform_q_f16_i16_scale_params(inv_scale)
    bits = float32_to_fp16_bits(np.asarray(source, dtype=np.float32)).reshape(-1)
    out = np.empty(bits.shape, dtype=np.int16)
    for idx, fp16 in enumerate(bits):
        out[idx] = np.int16(_quantize_fp16_lane_to_i16(int(fp16), multiplier, shift))
    return out.reshape(np.asarray(source).shape)


def quantize_fp16_bits_to_i16_xform(source: np.ndarray, *, scale: float) -> np.ndarray:
    inv_scale = 1.0 / float(scale)
    multiplier, shift = choose_xform_q_f16_i16_scale_params(inv_scale)
    bits = np.asarray(source, dtype=np.int16).view(np.uint16).reshape(-1)
    out = np.empty(bits.shape, dtype=np.int16)
    for idx, fp16 in enumerate(bits):
        out[idx] = np.int16(_quantize_fp16_lane_to_i16(int(fp16), multiplier, shift))
    return out.reshape(np.asarray(source).shape)


def exp_approx_scalar(x: float) -> float:
    exp_neg_int = (
        1.0,
        0.36787945,
        0.13533528,
        0.049787067,
        0.018315639,
        0.0067379470,
        0.0024787522,
        0.00091188195,
        0.00033546263,
        0.00012340980,
        0.000045399930,
        0.000016701700,
        0.0000061442124,
        0.0000022603294,
        0.00000083152872,
        0.00000030590232,
        0.00000011253518,
    )
    if x == 0.0:
        return 1.0
    if x > 0.0:
        if x >= 16.0:
            return 8.8861100e6
        return recip_approx_scalar(exp_approx_scalar(-x))
    if x <= -16.0:
        return 0.0
    k = int(-x)
    r = x + float(k)
    poly = 1.0 + r * (1.0 + r * (0.5 + r * (0.16666667 + r * (0.04166667 + r * 0.0083333333))))
    return exp_neg_int[k] * max(poly, 0.0)


def recip_approx_scalar(x: float) -> float:
    if x <= 0.0:
        raise ValueError(f"reciprocal input must be positive, got {x}.")

    exp2 = 0
    while x > 1.0:
        x *= 0.5
        exp2 += 1
    while x < 0.5:
        x *= 2.0
        exp2 -= 1

    y = 2.8235295 - 1.8823529 * x
    y = y * (2.0 - x * y)
    y = y * (2.0 - x * y)

    while exp2 > 0:
        y *= 0.5
        exp2 -= 1
    while exp2 < 0:
        y *= 2.0
        exp2 += 1
    return y


def rsqrt_approx_scalar(x: float) -> float:
    if x <= 0.0:
        raise ValueError(f"rsqrt input must be positive, got {x}.")

    scale = 1.0
    while x > 2.0:
        x *= 0.25
        scale *= 0.5
    while x < 0.5:
        x *= 4.0
        scale *= 2.0

    y = 1.25 - 0.25 * x
    for _ in range(4):
        y = y * (1.5 - 0.5 * x * y * y)
    return y * scale


def sigmoid_approx(source: np.ndarray) -> np.ndarray:
    source_f32 = np.asarray(source, dtype=np.float32)
    flat = source_f32.reshape(-1)
    out = np.empty_like(flat)
    for idx, value in enumerate(flat):
        denom = 1.0 + exp_approx_scalar(-float(value))
        out[idx] = np.float32(recip_approx_scalar(denom))
    return out.reshape(source_f32.shape)


def silu_approx(source: np.ndarray) -> np.ndarray:
    source_f32 = np.asarray(source, dtype=np.float32)
    flat = source_f32.reshape(-1)
    out = np.empty_like(flat)
    for idx, value in enumerate(flat):
        denom = 1.0 + exp_approx_scalar(-float(value))
        out[idx] = np.float32(float(value) * recip_approx_scalar(denom))
    return out.reshape(source_f32.shape)


def softmax_f16_approx(source: np.ndarray, *, axis: int = -1) -> np.ndarray:
    source_f32 = np.asarray(source, dtype=np.float32)
    if source_f32.ndim == 0:
        raise ValueError("softmax_f16 expects rank >= 1 input.")

    moved = np.moveaxis(source_f32, axis, -1)
    rows = moved.reshape(-1, moved.shape[-1])
    out_rows = np.empty_like(rows)
    for row_idx, row in enumerate(rows):
        max_value = float(np.max(row))
        exp_values = np.empty_like(row)
        sum_value = np.float32(0.0)
        for idx, value in enumerate(row):
            exp_value = np.float32(exp_approx_scalar(float(value) - max_value))
            exp_values[idx] = exp_value
            sum_value = np.float32(sum_value + exp_value)
        inv_sum = np.float32(recip_approx_scalar(float(sum_value)))
        out_rows[row_idx] = fp16_roundtrip(exp_values * inv_sum)
    out = out_rows.reshape(moved.shape)
    return np.moveaxis(out, -1, axis).astype(np.float32)


def rmsnorm_approx(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x_f = np.asarray(x, dtype=np.float32)
    weight_f = np.asarray(weight, dtype=np.float32).reshape(-1)
    hidden = x_f.shape[-1]
    if weight_f.size != hidden:
        raise ValueError(f"rmsnorm weight size mismatch: hidden={hidden}, weight={weight_f.size}.")

    outer = x_f.reshape(-1, hidden)
    out = np.empty_like(outer)
    inv_hidden = np.float32(recip_approx_scalar(float(hidden)))
    for row_idx, row in enumerate(outer):
        mean_sq = np.float32(np.sum(row * row, dtype=np.float32) * inv_hidden)
        inv_rms = np.float32(rsqrt_approx_scalar(float(mean_sq + np.float32(eps))))
        out[row_idx] = row * inv_rms * weight_f
    return out.reshape(x_f.shape).astype(np.float32)


def layernorm_approx(x: np.ndarray, weight_bias: np.ndarray, eps: float) -> np.ndarray:
    x_f = np.asarray(x, dtype=np.float32)
    wb_f = np.asarray(weight_bias, dtype=np.float32).reshape(-1)
    hidden = x_f.shape[-1]
    if wb_f.size != hidden * 2:
        raise ValueError(f"layernorm weight/bias size mismatch: hidden={hidden}, weight_bias={wb_f.size}.")
    weight = wb_f[:hidden]
    bias = wb_f[hidden:]

    outer = x_f.reshape(-1, hidden)
    out = np.empty_like(outer)
    inv_hidden = np.float32(recip_approx_scalar(float(hidden)))
    for row_idx, row in enumerate(outer):
        mean = np.float32(np.sum(row, dtype=np.float32) * inv_hidden)
        centered = row - mean
        var = np.float32(np.sum(centered * centered, dtype=np.float32) * inv_hidden)
        inv_std = np.float32(rsqrt_approx_scalar(float(var + np.float32(eps))))
        out[row_idx] = centered * inv_std * weight + bias
    return out.reshape(x_f.shape).astype(np.float32)
