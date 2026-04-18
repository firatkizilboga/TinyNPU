#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPU_BASE 0x30000000u
#define TB_TIMER_CTRL_BASE 0x15000000u
#define TB_TIMER_COUNT_REG 0x15001000u
#define TINY_IM_BASE_ADDR 0x9000u
#define NPU_SHARED_UB_BASE 0x31000000u
#define NPU_SHARED_IM_BASE 0x32000000u
#define TINY_ARRAY_SIZE __TINY_ARRAY_SIZE__
#define TINY_BUFFER_WORDS_32 __TINY_BUFFER_WORDS_32__
#define TINY_MMVR_BYTES (TINY_BUFFER_WORDS_32 * 4)

#ifndef TINYNPU_USE_SHARED_SRAM
#define TINYNPU_USE_SHARED_SRAM 1
#endif

#ifndef TINYNPU_SHARED_PACKED_READ_MMIO_FALLBACK
#define TINYNPU_SHARED_PACKED_READ_MMIO_FALLBACK 0
#endif

enum {
    REG_STATUS = 0x00,
    REG_CMD = 0x04,
    REG_ADDR = 0x08,
    REG_ARG = 0x0C,
    REG_MMVR = 0x10,
};

enum {
    CMD_WRITE_MEM = 0x01,
    CMD_READ_MEM = 0x02,
    CMD_RUN = 0x03,
};

enum {
    STATUS_BUSY = 0x01,
    STATUS_DATA_VALID = 0x02,
    STATUS_HALTED = 0xFF,
};

static int g_tinynpu_force_mmio_transfers = 0;

static void tinynpu_set_force_mmio(int enabled)
{
    g_tinynpu_force_mmio_transfers = enabled ? 1 : 0;
}

typedef enum {
    TINY_DTYPE_INT4 = 0,
    TINY_DTYPE_INT8 = 1,
    TINY_DTYPE_INT16 = 2,
    TINY_DTYPE_INT32 = 3,
    TINY_DTYPE_FLOAT32 = 4,
} TinyDType;

enum {
    HOST_ACT_NONE = 0,
    HOST_ACT_RELU = 1,
    HOST_ACT_SIGMOID = 2,
    HOST_ACT_H_GELU = 3,
};

typedef struct {
    const char *name;
    void *data;
    TinyDType dtype;
    int rank;
    int shape[4];
    int elem_count;
} TinyTensor;

static volatile uint8_t *const npu = (volatile uint8_t *)NPU_BASE;
static volatile uint32_t *const tb_timer_ctrl = (volatile uint32_t *)TB_TIMER_CTRL_BASE;
static volatile uint32_t *const tb_timer_value = (volatile uint32_t *)(TB_TIMER_CTRL_BASE + 4u);
static volatile uint32_t *const tb_timer_count = (volatile uint32_t *)TB_TIMER_COUNT_REG;

volatile uint32_t runtime_cycle_start __attribute__((section(".noinit")));
volatile uint32_t runtime_cycle_post_bss __attribute__((section(".noinit")));
volatile uint32_t runtime_cycle_post_init __attribute__((section(".noinit")));
volatile uint32_t runtime_cycle_pre_main __attribute__((section(".noinit")));

static inline uint32_t read_mcycle32(void)
{
    return *tb_timer_count;
}

static void print_cycle_delta32(const char *label, uint32_t start, uint32_t end)
{
    /* The testbench timer counts down from 0xFFFFFFFF, so elapsed cycles are start - end. */
    printf("%s cycles=%lu\n", label, (unsigned long)(start - end));
}

static void print_startup_cycle_report(void)
{
    print_cycle_delta32("startup.bss_clear", runtime_cycle_start, runtime_cycle_post_bss);
    print_cycle_delta32("startup.init_array", runtime_cycle_post_bss, runtime_cycle_post_init);
    print_cycle_delta32("startup.to_main", runtime_cycle_post_init, runtime_cycle_pre_main);
    print_cycle_delta32("startup.total", runtime_cycle_start, runtime_cycle_pre_main);
}

static void tb_timer_reset_counter(void)
{
    *tb_timer_ctrl = 0u;
    *tb_timer_value = 0xFFFFFFFFu;
    while (*tb_timer_count == 0u) {
    }
}

static inline void npu_write8(uint32_t reg, uint8_t value)
{
    npu[reg] = value;
}

static inline uint8_t npu_read8(uint32_t reg)
{
    return npu[reg];
}

static void npu_write16(uint32_t reg, uint16_t value)
{
    npu_write8(reg + 0u, (uint8_t)((value >> 0) & 0xFFu));
    npu_write8(reg + 1u, (uint8_t)((value >> 8) & 0xFFu));
}

static void npu_write32(uint32_t reg, uint32_t value)
{
    npu_write8(reg + 0u, (uint8_t)((value >> 0) & 0xFFu));
    npu_write8(reg + 1u, (uint8_t)((value >> 8) & 0xFFu));
    npu_write8(reg + 2u, (uint8_t)((value >> 16) & 0xFFu));
    npu_write8(reg + 3u, (uint8_t)((value >> 24) & 0xFFu));
}

static uint32_t npu_read32(uint32_t reg)
{
    uint32_t value = 0u;
    value |= (uint32_t)npu_read8(reg + 0u) << 0;
    value |= (uint32_t)npu_read8(reg + 1u) << 8;
    value |= (uint32_t)npu_read8(reg + 2u) << 16;
    value |= (uint32_t)npu_read8(reg + 3u) << 24;
    return value;
}

static void runtime_fail(const char *message)
{
    printf("runtime failure: %s\n", message);
    exit(EXIT_FAILURE);
}

static void runtime_assert(int condition, const char *message)
{
    if (!condition) {
        runtime_fail(message);
    }
}

static float host_absf(float x);
static int32_t host_round_to_i32(float x);
static int64_t host_round_to_i64(float x);

static int32_t *tensor_i32(const TinyTensor *tensor)
{
    runtime_assert(tensor->dtype != TINY_DTYPE_FLOAT32, "expected integer tensor");
    return (int32_t *)tensor->data;
}

static float *tensor_f32(const TinyTensor *tensor)
{
    runtime_assert(tensor->dtype == TINY_DTYPE_FLOAT32, "expected float tensor");
    return (float *)tensor->data;
}

static int tensor_extent(const TinyTensor *tensor, int axis)
{
    if (axis < tensor->rank) {
        return tensor->shape[axis];
    }
    return 1;
}

static void tensor_unravel(const TinyTensor *tensor, int linear, int idx[4])
{
    int tmp = linear;
    for (int axis = tensor->rank - 1; axis >= 0; --axis) {
        int dim = tensor_extent(tensor, axis);
        idx[axis] = tmp % dim;
        tmp /= dim;
    }
    for (int axis = tensor->rank; axis < 4; ++axis) {
        idx[axis] = 0;
    }
}

static int tensor_ravel_from_dims(const int dims[4], int rank, const int idx[4])
{
    int linear = 0;
    for (int axis = 0; axis < rank; ++axis) {
        linear = (linear * dims[axis]) + idx[axis];
    }
    return linear;
}

static int tensor_ravel(const TinyTensor *tensor, const int idx[4])
{
    return tensor_ravel_from_dims(tensor->shape, tensor->rank, idx);
}

static void tensor_zero(TinyTensor *tensor)
{
    if (tensor->dtype == TINY_DTYPE_FLOAT32) {
        memset(tensor_f32(tensor), 0, (size_t)tensor->elem_count * sizeof(float));
    } else {
        memset(tensor_i32(tensor), 0, (size_t)tensor->elem_count * sizeof(int32_t));
    }
}

static float tensor_get_float(const TinyTensor *tensor, int linear)
{
    if (tensor->dtype == TINY_DTYPE_FLOAT32) {
        return tensor_f32(tensor)[linear];
    }
    return (float)tensor_i32(tensor)[linear];
}

static int32_t tensor_get_i32(const TinyTensor *tensor, int linear)
{
    if (tensor->dtype == TINY_DTYPE_FLOAT32) {
        float value = tensor_f32(tensor)[linear];
        float rounded = (float)host_round_to_i32(value);
        if (host_absf(value - rounded) > 1e-6f) {
            printf("non-integral float reached NPU boundary in tensor %s\n", tensor->name);
            exit(EXIT_FAILURE);
        }
        return (int32_t)rounded;
    }
    return tensor_i32(tensor)[linear];
}

static void tensor_set_i32(TinyTensor *tensor, int linear, int32_t value)
{
    tensor_i32(tensor)[linear] = value;
}

static void tensor_set_float(TinyTensor *tensor, int linear, float value)
{
    tensor_f32(tensor)[linear] = value;
}

static int32_t clip_for_dtype(int64_t value, TinyDType dtype)
{
    if (dtype == TINY_DTYPE_INT4) {
        if (value < -8) {
            return -8;
        }
        if (value > 7) {
            return 7;
        }
        return (int32_t)value;
    }
    if (dtype == TINY_DTYPE_INT8) {
        if (value < -128) {
            return -128;
        }
        if (value > 127) {
            return 127;
        }
        return (int32_t)value;
    }
    if (dtype == TINY_DTYPE_INT16) {
        if (value < -32768) {
            return -32768;
        }
        if (value > 32767) {
            return 32767;
        }
        return (int32_t)value;
    }
    return (int32_t)value;
}

static int32_t clip_for_output_dtype(int64_t value, TinyDType dtype)
{
    if (dtype == TINY_DTYPE_INT32) {
        if (value < INT32_MIN) {
            return INT32_MIN;
        }
        if (value > INT32_MAX) {
            return INT32_MAX;
        }
        return (int32_t)value;
    }
    return clip_for_dtype(value, dtype);
}

static void host_alias(TinyTensor *dst, const TinyTensor *src)
{
    runtime_assert(dst->elem_count == src->elem_count, "alias size mismatch");
    if (dst->dtype == TINY_DTYPE_FLOAT32) {
        runtime_assert(src->dtype == TINY_DTYPE_FLOAT32, "alias dtype mismatch");
        memcpy(dst->data, src->data, (size_t)src->elem_count * sizeof(float));
    } else {
        runtime_assert(src->dtype != TINY_DTYPE_FLOAT32, "alias dtype mismatch");
        memcpy(dst->data, src->data, (size_t)src->elem_count * sizeof(int32_t));
    }
}

static void host_relu(TinyTensor *dst, const TinyTensor *src)
{
    runtime_assert(dst->elem_count == src->elem_count, "relu size mismatch");
    if (src->dtype == TINY_DTYPE_FLOAT32) {
        runtime_assert(dst->dtype == TINY_DTYPE_FLOAT32, "relu dtype mismatch");
        for (int i = 0; i < src->elem_count; ++i) {
            float value = tensor_get_float(src, i);
            tensor_set_float(dst, i, value > 0.0f ? value : 0.0f);
        }
        return;
    }
    runtime_assert(dst->dtype != TINY_DTYPE_FLOAT32, "relu dtype mismatch");
    for (int i = 0; i < src->elem_count; ++i) {
        int32_t value = tensor_get_i32(src, i);
        tensor_set_i32(dst, i, value > 0 ? value : 0);
    }
}

static void host_slice_row(TinyTensor *dst, const TinyTensor *src, int row_index)
{
    runtime_assert(src->rank >= 2, "slice_row expects rank >= 2");
    runtime_assert(row_index >= 0 && row_index < src->shape[0], "slice_row row_index out of range");
    const int row_width = src->elem_count / src->shape[0];
    runtime_assert(dst->elem_count == row_width, "slice_row output size mismatch");
    if (src->dtype == TINY_DTYPE_FLOAT32) {
        runtime_assert(dst->dtype == TINY_DTYPE_FLOAT32, "slice_row dtype mismatch");
        for (int i = 0; i < row_width; ++i) {
            tensor_set_float(dst, i, tensor_get_float(src, row_index * row_width + i));
        }
        return;
    }
    runtime_assert(dst->dtype != TINY_DTYPE_FLOAT32, "slice_row dtype mismatch");
    for (int i = 0; i < row_width; ++i) {
        tensor_set_i32(dst, i, tensor_get_i32(src, row_index * row_width + i));
    }
}

static float host_exp_approx(float x);
static float host_recip_approx(float x);
static float host_log_approx(float x);
static float host_rsqrt_approx(float x);
static float host_sin_approx(float x);
static float host_cos_approx(float x);
static float host_absf(float x);
static int32_t host_round_to_i32(float x);
static int64_t host_round_to_i64(float x);
static float host_erf_approx(float x);

static void host_sigmoid(TinyTensor *dst, const TinyTensor *src)
{
    runtime_assert(dst->dtype == TINY_DTYPE_FLOAT32, "sigmoid expects float output");
    runtime_assert(dst->elem_count == src->elem_count, "sigmoid size mismatch");
    for (int i = 0; i < src->elem_count; ++i) {
        float value = tensor_get_float(src, i);
        float denom = 1.0f + host_exp_approx(-value);
        tensor_set_float(dst, i, host_recip_approx(denom));
    }
}

static void host_silu(TinyTensor *dst, const TinyTensor *src)
{
    runtime_assert(dst->dtype == TINY_DTYPE_FLOAT32, "silu expects float output");
    runtime_assert(dst->elem_count == src->elem_count, "silu size mismatch");
    for (int i = 0; i < src->elem_count; ++i) {
        float value = tensor_get_float(src, i);
        float denom = 1.0f + host_exp_approx(-value);
        float sigma = host_recip_approx(denom);
        tensor_set_float(dst, i, value * sigma);
    }
}

static float host_exp_approx_neg_unit(float x)
{
    /* 5th-order Taylor on [-1, 0]. */
    float y = 1.0f + x * (
        1.0f + x * (
            0.5f + x * (
                0.16666667f + x * (
                    0.04166667f + x * 0.0083333333f))));
    return y > 0.0f ? y : 0.0f;
}

static float host_absf(float x)
{
    return x < 0.0f ? -x : x;
}

static int32_t host_round_to_i32(float x)
{
    return x >= 0.0f ? (int32_t)(x + 0.5f) : (int32_t)(x - 0.5f);
}

static int64_t host_round_to_i64(float x)
{
    return x >= 0.0f ? (int64_t)(x + 0.5f) : (int64_t)(x - 0.5f);
}

static float host_exp_approx(float x)
{
    static const float exp_neg_int[17] = {
        1.0f,
        0.36787945f,
        0.13533528f,
        0.049787067f,
        0.018315639f,
        0.0067379470f,
        0.0024787522f,
        0.00091188195f,
        0.00033546263f,
        0.00012340980f,
        0.000045399930f,
        0.000016701700f,
        0.0000061442124f,
        0.0000022603294f,
        0.00000083152872f,
        0.00000030590232f,
        0.00000011253518f,
    };

    if (x == 0.0f) {
        return 1.0f;
    }
    if (x > 0.0f) {
        return host_recip_approx(host_exp_approx(-x));
    }
    if (x <= -16.0f) {
        return 0.0f;
    }

    int k = (int)(-x);
    float r = x + (float)k; /* r in [-1, 0] */
    return exp_neg_int[k] * host_exp_approx_neg_unit(r);
}

static float host_recip_approx(float x)
{
    runtime_assert(x > 0.0f, "reciprocal input must be positive");

    int exp2 = 0;
    while (x > 1.0f) {
        x *= 0.5f;
        exp2 += 1;
    }
    while (x < 0.5f) {
        x *= 2.0f;
        exp2 -= 1;
    }

    /* Linear seed for 1/x on [0.5, 1.0]. */
    float y = 2.8235295f - 1.8823529f * x;
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);

    while (exp2 > 0) {
        y *= 0.5f;
        exp2 -= 1;
    }
    while (exp2 < 0) {
        y *= 2.0f;
        exp2 += 1;
    }
    return y;
}

static float host_log_approx(float x)
{
    const float ln2 = 0.69314718f;
    runtime_assert(x > 0.0f, "log input must be positive");

    int exp2 = 0;
    while (x > 1.5f) {
        x *= 0.5f;
        exp2 += 1;
    }
    while (x < 0.75f) {
        x *= 2.0f;
        exp2 -= 1;
    }

    {
        float y = (x - 1.0f) * host_recip_approx(x + 1.0f);
        float y2 = y * y;
        float y3 = y * y2;
        float y5 = y3 * y2;
        float y7 = y5 * y2;
        float ln_m = 2.0f * (y + y3 * 0.33333333f + y5 * 0.2f + y7 * 0.14285715f);
        return ((float)exp2 * ln2) + ln_m;
    }
}

static float host_rsqrt_approx(float x)
{
    runtime_assert(x > 0.0f, "rsqrt input must be positive");

    float scale = 1.0f;
    while (x > 2.0f) {
        x *= 0.25f;
        scale *= 0.5f;
    }
    while (x < 0.5f) {
        x *= 4.0f;
        scale *= 2.0f;
    }

    float y = 1.25f - 0.25f * x;
    for (int i = 0; i < 4; ++i) {
        y = y * (1.5f - 0.5f * x * y * y);
    }
    return y * scale;
}

static float host_wrap_pi(float x)
{
    const float pi = 3.14159265f;
    const float two_pi = 6.28318531f;
    while (x > pi) {
        x -= two_pi;
    }
    while (x < -pi) {
        x += two_pi;
    }
    return x;
}

static float host_sin_approx(float x)
{
    const float half_pi = 1.57079633f;
    x = host_wrap_pi(x);
    if (x > half_pi) {
        x = 3.14159265f - x;
    } else if (x < -half_pi) {
        x = -3.14159265f - x;
    }

    {
        float x2 = x * x;
        return x * (1.0f + x2 * (-0.16666667f + x2 * (0.0083333333f + x2 * -0.00019841270f)));
    }
}

static float host_cos_approx(float x)
{
    const float half_pi = 1.57079633f;
    float sign = 1.0f;
    x = host_wrap_pi(x);
    if (x > half_pi) {
        x = 3.14159265f - x;
        sign = -1.0f;
    } else if (x < -half_pi) {
        x = -3.14159265f - x;
        sign = -1.0f;
    }

    {
        float x2 = x * x;
        float y = 1.0f + x2 * (-0.5f + x2 * (0.04166667f + x2 * -0.0013888889f));
        return sign * y;
    }
}

static float host_erf_approx(float x)
{
    const float p = 0.3275911f;
    const float a1 = 0.25482959f;
    const float a2 = -0.28449672f;
    const float a3 = 1.4214138f;
    const float a4 = -1.4531521f;
    const float a5 = 1.0614054f;
    float sign = 1.0f;
    if (x < 0.0f) {
        sign = -1.0f;
        x = -x;
    }
    {
        float t = host_recip_approx(1.0f + p * x);
        float poly = (((((a5 * t) + a4) * t + a3) * t + a2) * t + a1) * t;
        return sign * (1.0f - poly * host_exp_approx(-(x * x)));
    }
}

static uint16_t host_float32_to_fp16_bits(float value);
static void host_quantize_fp16bits(TinyTensor *dst, const TinyTensor *src, float inv_scale, int zero_point);

static void host_gelu(TinyTensor *dst, const TinyTensor *src)
{
    runtime_assert(dst->elem_count == src->elem_count, "gelu size mismatch");
    for (int i = 0; i < src->elem_count; ++i) {
        float value = tensor_get_float(src, i);
        float erf_term = host_erf_approx(value * 0.70710678f);
        float out = 0.5f * value * (1.0f + erf_term);
        if (dst->dtype == TINY_DTYPE_FLOAT32) {
            tensor_set_float(dst, i, out);
        } else {
            tensor_set_i32(dst, i, (int16_t)host_float32_to_fp16_bits(out));
        }
    }
}

static void host_quantize(TinyTensor *dst, const TinyTensor *src, float inv_scale, int zero_point)
{
    runtime_assert(inv_scale > 0.0f, "quantize inv_scale must be positive");
    runtime_assert(dst->dtype != TINY_DTYPE_FLOAT32, "quantize output must be integer");
    runtime_assert(dst->elem_count == src->elem_count, "quantize size mismatch");
    for (int i = 0; i < src->elem_count; ++i) {
        float source = tensor_get_float(src, i);
        int64_t quantized = host_round_to_i64(source * inv_scale) + (int64_t)zero_point;
        tensor_set_i32(dst, i, clip_for_dtype(quantized, dst->dtype));
    }
}

static void host_quantize_fp16bits_attr(TinyTensor *dst, const TinyTensor *src, float inv_scale, int zero_point)
{
    host_quantize_fp16bits(dst, src, inv_scale, zero_point);
}

static void host_quantize_fp16bits(TinyTensor *dst, const TinyTensor *src, float inv_scale, int zero_point)
{
    runtime_assert(inv_scale > 0.0f, "quantize inv_scale must be positive");
    runtime_assert(dst->dtype != TINY_DTYPE_FLOAT32, "quantize output must be integer");
    runtime_assert(dst->elem_count == src->elem_count, "quantize size mismatch");
    for (int i = 0; i < src->elem_count; ++i) {
        uint16_t bits = (uint16_t)tensor_get_i32(src, i);
        union {
            uint32_t u;
            float f;
        } out;
        uint32_t sign = ((uint32_t)bits & 0x8000u) << 16;
        uint32_t exp = ((uint32_t)bits >> 10) & 0x1Fu;
        uint32_t mant = (uint32_t)bits & 0x03FFu;
        if (exp == 0u) {
            if (mant == 0u) {
                out.u = sign;
            } else {
                exp = 1u;
                while ((mant & 0x0400u) == 0u) {
                    mant <<= 1;
                    exp -= 1u;
                }
                mant &= 0x03FFu;
                out.u = sign | ((exp + 112u) << 23) | (mant << 13);
            }
        } else if (exp == 0x1Fu) {
            out.u = sign | 0x7F800000u | (mant << 13);
        } else {
            out.u = sign | ((exp + 112u) << 23) | (mant << 13);
        }
        {
            int64_t quantized = host_round_to_i64(out.f * inv_scale) + (int64_t)zero_point;
            tensor_set_i32(dst, i, clip_for_dtype(quantized, dst->dtype));
        }
    }
}

static void host_dequantize(TinyTensor *dst, const TinyTensor *src, float scale, int zero_point)
{
    runtime_assert(scale > 0.0f, "dequantize scale must be positive");
    runtime_assert(dst->dtype == TINY_DTYPE_FLOAT32, "dequantize output must be float");
    runtime_assert(dst->elem_count == src->elem_count, "dequantize size mismatch");
    for (int i = 0; i < src->elem_count; ++i) {
        float source = (float)tensor_get_i32(src, i);
        tensor_set_float(dst, i, (source - (float)zero_point) * scale);
    }
}

static void host_requantize(TinyTensor *dst, const TinyTensor *src, float scale, int zero_point)
{
    runtime_assert(dst->dtype != TINY_DTYPE_FLOAT32, "requantize output must be integer");
    runtime_assert(dst->elem_count == src->elem_count, "requantize size mismatch");
    for (int i = 0; i < src->elem_count; ++i) {
        float source = tensor_get_float(src, i);
        int64_t quantized = host_round_to_i64(source * scale) + (int64_t)zero_point;
        tensor_set_i32(dst, i, clip_for_dtype(quantized, dst->dtype));
    }
}

static void host_reshape(TinyTensor *dst, const TinyTensor *src)
{
    host_alias(dst, src);
}

static void host_transpose(TinyTensor *dst, const TinyTensor *src, const int *axes, int axis_count)
{
    int mapping[4] = {0, 1, 2, 3};
    if (axis_count == 0) {
        for (int i = 0; i < src->rank; ++i) {
            mapping[i] = src->rank - 1 - i;
        }
    } else {
        runtime_assert(axis_count == src->rank, "transpose axis rank mismatch");
        for (int i = 0; i < axis_count; ++i) {
            mapping[i] = axes[i] < 0 ? axes[i] + src->rank : axes[i];
        }
    }

    int out_idx[4] = {0, 0, 0, 0};
    int in_idx[4] = {0, 0, 0, 0};
    if (dst->dtype == TINY_DTYPE_FLOAT32) {
        runtime_assert(src->dtype == TINY_DTYPE_FLOAT32, "transpose dtype mismatch");
        for (int linear = 0; linear < dst->elem_count; ++linear) {
            tensor_unravel(dst, linear, out_idx);
            for (int axis = 0; axis < src->rank; ++axis) {
                in_idx[axis] = 0;
            }
            for (int axis = 0; axis < src->rank; ++axis) {
                in_idx[mapping[axis]] = out_idx[axis];
            }
            tensor_set_float(dst, linear, tensor_get_float(src, tensor_ravel(src, in_idx)));
        }
        return;
    }

    runtime_assert(src->dtype != TINY_DTYPE_FLOAT32, "transpose dtype mismatch");
    for (int linear = 0; linear < dst->elem_count; ++linear) {
        tensor_unravel(dst, linear, out_idx);
        for (int axis = 0; axis < src->rank; ++axis) {
            in_idx[axis] = 0;
        }
        for (int axis = 0; axis < src->rank; ++axis) {
            in_idx[mapping[axis]] = out_idx[axis];
        }
        tensor_set_i32(dst, linear, tensor_get_i32(src, tensor_ravel(src, in_idx)));
    }
}

static void host_softmax(TinyTensor *dst, const TinyTensor *src, int axis)
{
    runtime_assert(dst->dtype == TINY_DTYPE_FLOAT32, "softmax output must be float");
    runtime_assert(dst->elem_count == src->elem_count, "softmax size mismatch");
    if (axis < 0) {
        axis += src->rank;
    }
    runtime_assert(axis >= 0 && axis < src->rank, "softmax axis out of range");

    int outer = 1;
    int inner = 1;
    int extent = src->shape[axis];
    for (int i = 0; i < axis; ++i) {
        outer *= src->shape[i];
    }
    for (int i = axis + 1; i < src->rank; ++i) {
        inner *= src->shape[i];
    }

    for (int outer_idx = 0; outer_idx < outer; ++outer_idx) {
        for (int inner_idx = 0; inner_idx < inner; ++inner_idx) {
            float max_value = -1.0e30f;
            for (int axis_idx = 0; axis_idx < extent; ++axis_idx) {
                int linear = ((outer_idx * extent) + axis_idx) * inner + inner_idx;
                float value = tensor_get_float(src, linear);
                if (value > max_value) {
                    max_value = value;
                }
            }

            float sum = 0.0f;
            for (int axis_idx = 0; axis_idx < extent; ++axis_idx) {
                int linear = ((outer_idx * extent) + axis_idx) * inner + inner_idx;
                float exp_value = host_exp_approx(tensor_get_float(src, linear) - max_value);
                tensor_set_float(dst, linear, exp_value);
                sum += exp_value;
            }
            runtime_assert(sum != 0.0f, "softmax sum is zero");
            {
                float inv_sum = host_recip_approx(sum);
                for (int axis_idx = 0; axis_idx < extent; ++axis_idx) {
                    int linear = ((outer_idx * extent) + axis_idx) * inner + inner_idx;
                    float normalized = tensor_get_float(dst, linear) * inv_sum;
                    tensor_set_float(dst, linear, normalized);
                }
            }
        }
    }
}

static uint16_t host_float32_to_fp16_bits(float value)
{
    union {
        float f;
        uint32_t u;
    } v = {value};
    uint32_t sign = (v.u >> 16) & 0x8000u;
    uint32_t exp = (v.u >> 23) & 0xFFu;
    uint32_t mant = v.u & 0x7FFFFFu;
    int32_t half_exp;

    if (exp == 0xFFu) {
        if (mant == 0u) {
            return (uint16_t)(sign | 0x7C00u);
        }
        mant >>= 13;
        if (mant == 0u) {
            mant = 1u;
        }
        return (uint16_t)(sign | 0x7C00u | mant);
    }

    half_exp = (int32_t)exp - 127 + 15;
    if (half_exp >= 0x1F) {
        return (uint16_t)(sign | 0x7C00u);
    }
    if (half_exp <= 0) {
        uint32_t shift;
        uint32_t rounded;
        if (half_exp < -10) {
            return (uint16_t)sign;
        }
        mant |= 0x800000u;
        shift = (uint32_t)(14 - half_exp);
        rounded = mant + ((1u << (shift - 1)) - 1u) + ((mant >> shift) & 1u);
        return (uint16_t)(sign | (rounded >> shift));
    }

    {
        uint32_t half_mant = mant >> 13;
        uint32_t round_bits = mant & 0x1FFFu;
        if (round_bits > 0x1000u || (round_bits == 0x1000u && (half_mant & 1u))) {
            half_mant += 1u;
            if (half_mant == 0x400u) {
                half_mant = 0u;
                half_exp += 1;
                if (half_exp >= 0x1F) {
                    return (uint16_t)(sign | 0x7C00u);
                }
            }
        }
        return (uint16_t)(sign | ((uint32_t)half_exp << 10) | half_mant);
    }
}

static void host_softmax_f16(TinyTensor *dst, const TinyTensor *src, int axis)
{
    runtime_assert(dst->dtype != TINY_DTYPE_FLOAT32, "softmax_f16 output must be int16 fp16-bits");
    runtime_assert(dst->elem_count == src->elem_count, "softmax_f16 size mismatch");
    if (axis < 0) {
        axis += src->rank;
    }
    runtime_assert(axis >= 0 && axis < src->rank, "softmax_f16 axis out of range");

    int outer = 1;
    int inner = 1;
    int extent = src->shape[axis];
    for (int i = 0; i < axis; ++i) {
        outer *= src->shape[i];
    }
    for (int i = axis + 1; i < src->rank; ++i) {
        inner *= src->shape[i];
    }

    for (int outer_idx = 0; outer_idx < outer; ++outer_idx) {
        for (int inner_idx = 0; inner_idx < inner; ++inner_idx) {
            float max_value = -1.0e30f;
            for (int axis_idx = 0; axis_idx < extent; ++axis_idx) {
                int linear = ((outer_idx * extent) + axis_idx) * inner + inner_idx;
                float value = tensor_get_float(src, linear);
                if (value > max_value) {
                    max_value = value;
                }
            }

            float sum = 0.0f;
            for (int axis_idx = 0; axis_idx < extent; ++axis_idx) {
                int linear = ((outer_idx * extent) + axis_idx) * inner + inner_idx;
                float exp_value = host_exp_approx(tensor_get_float(src, linear) - max_value);
                sum += exp_value;
            }
            runtime_assert(sum != 0.0f, "softmax_f16 sum is zero");
            {
                float inv_sum = host_recip_approx(sum);
                for (int axis_idx = 0; axis_idx < extent; ++axis_idx) {
                    int linear = ((outer_idx * extent) + axis_idx) * inner + inner_idx;
                    float exp_value = host_exp_approx(tensor_get_float(src, linear) - max_value);
                    float normalized = exp_value * inv_sum;
                    uint16_t fp16_bits = host_float32_to_fp16_bits(normalized);
                    tensor_set_i32(dst, linear, (int16_t)fp16_bits);
                }
            }
        }
    }
}

static void host_mean(
    TinyTensor *dst,
    const TinyTensor *src,
    const int *dims,
    int dim_count,
    int keepdim,
    int has_input_quant,
    float input_scale,
    int input_zero_point)
{
    runtime_assert(dst->dtype == TINY_DTYPE_FLOAT32, "mean output must be float");

    int reduce_mask[4] = {0, 0, 0, 0};
    if (dim_count == 0) {
        for (int axis = 0; axis < src->rank; ++axis) {
            reduce_mask[axis] = 1;
        }
    } else {
        for (int i = 0; i < dim_count; ++i) {
            int axis = dims[i] < 0 ? dims[i] + src->rank : dims[i];
            runtime_assert(axis >= 0 && axis < src->rank, "mean axis out of range");
            reduce_mask[axis] = 1;
        }
    }

    float *out = tensor_f32(dst);
    int counts[dst->elem_count];
    for (int i = 0; i < dst->elem_count; ++i) {
        out[i] = 0.0f;
        counts[i] = 0;
    }

    int src_idx[4] = {0, 0, 0, 0};
    int dst_idx[4] = {0, 0, 0, 0};
    for (int linear = 0; linear < src->elem_count; ++linear) {
        tensor_unravel(src, linear, src_idx);
        int out_axis = 0;
        for (int axis = 0; axis < src->rank; ++axis) {
            if (reduce_mask[axis]) {
                if (keepdim) {
                    dst_idx[out_axis++] = 0;
                }
            } else {
                dst_idx[out_axis++] = src_idx[axis];
            }
        }
        if (keepdim) {
            while (out_axis < dst->rank) {
                dst_idx[out_axis++] = 0;
            }
        }
        float value = has_input_quant
            ? ((float)tensor_get_i32(src, linear) - (float)input_zero_point) * input_scale
            : tensor_get_float(src, linear);
        int out_linear = tensor_ravel(dst, dst_idx);
        out[out_linear] += value;
        counts[out_linear] += 1;
    }

    for (int i = 0; i < dst->elem_count; ++i) {
        runtime_assert(counts[i] > 0, "mean produced empty output bucket");
        out[i] /= (float)counts[i];
    }
}

static void host_rmsnorm(TinyTensor *dst, const TinyTensor *src, const TinyTensor *weight, float eps)
{
    runtime_assert(eps > 0.0f, "rmsnorm eps must be positive");
    runtime_assert(dst->dtype == TINY_DTYPE_FLOAT32, "rmsnorm output must be float");
    runtime_assert(src->rank >= 1, "rmsnorm expects rank >= 1");
    runtime_assert(dst->elem_count == src->elem_count, "rmsnorm size mismatch");

    const int hidden = src->shape[src->rank - 1];
    runtime_assert(weight->elem_count == hidden, "rmsnorm weight size mismatch");
    const int outer = src->elem_count / hidden;

    for (int outer_idx = 0; outer_idx < outer; ++outer_idx) {
        float mean_sq = 0.0f;
        const int base = outer_idx * hidden;
        for (int i = 0; i < hidden; ++i) {
            float value = tensor_get_float(src, base + i);
            mean_sq += value * value;
        }
        mean_sq *= host_recip_approx((float)hidden);
        {
            float inv_rms = host_rsqrt_approx(mean_sq + eps);
            for (int i = 0; i < hidden; ++i) {
                float value = tensor_get_float(src, base + i);
                float scale = tensor_get_float(weight, i);
                tensor_set_float(dst, base + i, value * inv_rms * scale);
            }
        }
    }
}

static void host_layernorm(TinyTensor *dst, const TinyTensor *src, const TinyTensor *weight_bias, float eps)
{
    runtime_assert(eps > 0.0f, "layernorm eps must be positive");
    runtime_assert(src->rank >= 1, "layernorm expects rank >= 1");
    runtime_assert(dst->elem_count == src->elem_count, "layernorm size mismatch");

    const int hidden = src->shape[src->rank - 1];
    runtime_assert(weight_bias->elem_count == hidden * 2, "layernorm weight/bias size mismatch");
    const int outer = src->elem_count / hidden;

    for (int outer_idx = 0; outer_idx < outer; ++outer_idx) {
        float mean = 0.0f;
        float var = 0.0f;
        const int base = outer_idx * hidden;
        for (int i = 0; i < hidden; ++i) {
            mean += tensor_get_float(src, base + i);
        }
        mean *= host_recip_approx((float)hidden);
        for (int i = 0; i < hidden; ++i) {
            float centered = tensor_get_float(src, base + i) - mean;
            var += centered * centered;
        }
        var *= host_recip_approx((float)hidden);
        {
            float inv_std = host_rsqrt_approx(var + eps);
            for (int i = 0; i < hidden; ++i) {
                float centered = tensor_get_float(src, base + i) - mean;
                float scale = tensor_get_float(weight_bias, i);
                float bias = tensor_get_float(weight_bias, hidden + i);
                float out = centered * inv_std * scale + bias;
                if (dst->dtype == TINY_DTYPE_FLOAT32) {
                    tensor_set_float(dst, base + i, out);
                } else {
                    tensor_set_i32(dst, base + i, (int16_t)host_float32_to_fp16_bits(out));
                }
            }
        }
    }
}

static void host_mul(TinyTensor *dst, const TinyTensor *lhs, const TinyTensor *rhs)
{
    runtime_assert(dst->dtype == TINY_DTYPE_FLOAT32, "mul expects float output");
    runtime_assert(lhs->elem_count == rhs->elem_count, "mul input size mismatch");
    runtime_assert(dst->elem_count == lhs->elem_count, "mul output size mismatch");
    for (int i = 0; i < lhs->elem_count; ++i) {
        tensor_set_float(dst, i, tensor_get_float(lhs, i) * tensor_get_float(rhs, i));
    }
}

static void host_add(TinyTensor *dst, const TinyTensor *lhs, const TinyTensor *rhs)
{
    runtime_assert(dst->dtype == TINY_DTYPE_FLOAT32, "add expects float output");
    runtime_assert(lhs->elem_count == rhs->elem_count, "add input size mismatch");
    runtime_assert(dst->elem_count == lhs->elem_count, "add output size mismatch");
    for (int i = 0; i < lhs->elem_count; ++i) {
        tensor_set_float(dst, i, tensor_get_float(lhs, i) + tensor_get_float(rhs, i));
    }
}

static void host_causal_mask(TinyTensor *dst, const TinyTensor *src, int past_kv_len, float fill_value)
{
    runtime_assert(dst->elem_count == src->elem_count, "causal_mask size mismatch");
    runtime_assert(src->rank >= 2, "causal_mask expects rank >= 2");
    runtime_assert(past_kv_len >= 0, "causal_mask past_kv_len must be non-negative");

    const int q_len = src->shape[src->rank - 2];
    const int k_len = src->shape[src->rank - 1];
    const int outer = src->elem_count / (q_len * k_len);
    if (src->dtype == TINY_DTYPE_FLOAT32) {
        runtime_assert(dst->dtype == TINY_DTYPE_FLOAT32, "float causal_mask output must be float");
        for (int outer_idx = 0; outer_idx < outer; ++outer_idx) {
            const int base = outer_idx * q_len * k_len;
            for (int row = 0; row < q_len; ++row) {
                const int max_col = past_kv_len + row;
                for (int col = 0; col < k_len; ++col) {
                    const int linear = base + row * k_len + col;
                    if (col > max_col) {
                        tensor_set_float(dst, linear, fill_value);
                    } else {
                        tensor_set_float(dst, linear, tensor_get_float(src, linear));
                    }
                }
            }
        }
        return;
    }

    runtime_assert(dst->dtype != TINY_DTYPE_FLOAT32, "integer causal_mask output must be integer");
    {
        const int32_t fill_i32 = (int32_t)host_round_to_i32(fill_value);
        for (int outer_idx = 0; outer_idx < outer; ++outer_idx) {
            const int base = outer_idx * q_len * k_len;
            for (int row = 0; row < q_len; ++row) {
                const int max_col = past_kv_len + row;
                for (int col = 0; col < k_len; ++col) {
                    const int linear = base + row * k_len + col;
                    if (col > max_col) {
                        tensor_set_i32(dst, linear, fill_i32);
                    } else {
                        tensor_set_i32(dst, linear, tensor_get_i32(src, linear));
                    }
                }
            }
        }
    }
}

static void host_concat_lastdim2(TinyTensor *dst, const TinyTensor *lhs, const TinyTensor *rhs)
{
    runtime_assert(lhs->rank == rhs->rank, "concat_lastdim2 rank mismatch");
    runtime_assert(dst->rank == lhs->rank, "concat_lastdim2 output rank mismatch");
    for (int axis = 0; axis < lhs->rank - 1; ++axis) {
        runtime_assert(lhs->shape[axis] == rhs->shape[axis], "concat_lastdim2 prefix shape mismatch");
        runtime_assert(dst->shape[axis] == lhs->shape[axis], "concat_lastdim2 output prefix mismatch");
    }
    runtime_assert(dst->shape[lhs->rank - 1] == lhs->shape[lhs->rank - 1] + rhs->shape[rhs->rank - 1], "concat_lastdim2 last-dim mismatch");
    runtime_assert(lhs->dtype == rhs->dtype, "concat_lastdim2 dtype mismatch");
    runtime_assert(dst->dtype == lhs->dtype, "concat_lastdim2 output dtype mismatch");

    const int lhs_last = lhs->shape[lhs->rank - 1];
    const int rhs_last = rhs->shape[rhs->rank - 1];
    const int outer = lhs->elem_count / lhs_last;
    for (int outer_idx = 0; outer_idx < outer; ++outer_idx) {
        const int lhs_base = outer_idx * lhs_last;
        const int rhs_base = outer_idx * rhs_last;
        const int dst_base = outer_idx * (lhs_last + rhs_last);
        if (dst->dtype == TINY_DTYPE_FLOAT32) {
            for (int i = 0; i < lhs_last; ++i) {
                tensor_set_float(dst, dst_base + i, tensor_get_float(lhs, lhs_base + i));
            }
            for (int i = 0; i < rhs_last; ++i) {
                tensor_set_float(dst, dst_base + lhs_last + i, tensor_get_float(rhs, rhs_base + i));
            }
        } else {
            for (int i = 0; i < lhs_last; ++i) {
                tensor_set_i32(dst, dst_base + i, tensor_get_i32(lhs, lhs_base + i));
            }
            for (int i = 0; i < rhs_last; ++i) {
                tensor_set_i32(dst, dst_base + lhs_last + i, tensor_get_i32(rhs, rhs_base + i));
            }
        }
    }
}

static float host_float_from_bits(int32_t bits)
{
    union {
        int32_t i;
        float f;
    } value;
    value.i = bits;
    return value.f;
}

#ifndef TNPU_ROPE_CACHE_MAX_POS
#define TNPU_ROPE_CACHE_MAX_POS 256
#endif

#ifndef TNPU_ROPE_CACHE_MAX_HALF
#define TNPU_ROPE_CACHE_MAX_HALF 64
#endif

static int g_rope_cache_valid = 0;
static int g_rope_cache_head_dim = 0;
static float g_rope_cache_theta = 0.0f;
static int g_rope_cache_max_pos = -1;
static float g_rope_cache_inv_freq[TNPU_ROPE_CACHE_MAX_HALF] __attribute__((section(".noinit")));
static float g_rope_cache_cos[TNPU_ROPE_CACHE_MAX_POS][TNPU_ROPE_CACHE_MAX_HALF] __attribute__((section(".noinit")));
static float g_rope_cache_sin[TNPU_ROPE_CACHE_MAX_POS][TNPU_ROPE_CACHE_MAX_HALF] __attribute__((section(".noinit")));

static int host_rope_cache_prepare(int head_dim, float theta, int max_pos)
{
    const int half = head_dim / 2;
    if (half > TNPU_ROPE_CACHE_MAX_HALF || max_pos >= TNPU_ROPE_CACHE_MAX_POS) {
        return 0;
    }

    if (!g_rope_cache_valid || g_rope_cache_head_dim != head_dim || g_rope_cache_theta != theta) {
        const float rope_base = powf(theta, -1.0f / (float)half);
        float inv_freq = 1.0f;
        for (int i = 0; i < half; ++i) {
            g_rope_cache_inv_freq[i] = inv_freq;
            inv_freq *= rope_base;
        }
        g_rope_cache_valid = 1;
        g_rope_cache_head_dim = head_dim;
        g_rope_cache_theta = theta;
        g_rope_cache_max_pos = -1;
    }

    for (int pos = g_rope_cache_max_pos + 1; pos <= max_pos; ++pos) {
        for (int i = 0; i < half; ++i) {
            float angle = (float)pos * g_rope_cache_inv_freq[i];
            g_rope_cache_cos[pos][i] = cosf(angle);
            g_rope_cache_sin[pos][i] = sinf(angle);
        }
    }
    if (max_pos > g_rope_cache_max_pos) {
        g_rope_cache_max_pos = max_pos;
    }
    return 1;
}

static void host_rope_precomputed(
    TinyTensor *dst,
    const TinyTensor *src,
    int head_dim,
    int position,
    const int *inv_freq_bits,
    int inv_freq_len)
{
    runtime_assert(dst->dtype == TINY_DTYPE_FLOAT32, "rope output must be float");
    runtime_assert(src->rank >= 2, "rope expects rank >= 2");
    runtime_assert(head_dim > 0 && (head_dim % 2) == 0, "rope head_dim must be positive and even");
    runtime_assert(src->shape[src->rank - 1] == head_dim, "rope last dimension mismatch");
    runtime_assert(dst->elem_count == src->elem_count, "rope size mismatch");

    const int half = head_dim / 2;
    runtime_assert(inv_freq_bits != NULL, "rope precomputed inv_freq must not be null");
    runtime_assert(inv_freq_len >= half, "rope precomputed inv_freq length mismatch");

    int outer = 1;
    int seq_len = 1;
    if (src->rank == 2) {
        outer = src->shape[0];
        seq_len = 1;
    } else {
        for (int axis = 0; axis < src->rank - 2; ++axis) {
            outer *= src->shape[axis];
        }
        seq_len = src->shape[src->rank - 2];
    }

    float *dst_data = tensor_f32(dst);
    float *src_data = tensor_f32(src);
    for (int outer_idx = 0; outer_idx < outer; ++outer_idx) {
        for (int seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
            const int logical_pos = position + seq_idx;
            const int base = ((outer_idx * seq_len) + seq_idx) * head_dim;
            for (int i = 0; i < half; ++i) {
                const float inv_freq = host_float_from_bits(inv_freq_bits[i]);
                const float angle = (float)logical_pos * inv_freq;
                const float c = cosf(angle);
                const float s = sinf(angle);
                const float first = src_data[base + i];
                const float second = src_data[base + half + i];
                dst_data[base + i] = first * c - second * s;
                dst_data[base + half + i] = second * c + first * s;
            }
        }
    }
}

/* Write a quantised INT16 key vector into a K-cache slot via the shared UB.
 *
 * src            : [1, d_head] INT16 key (the RoPE-encoded key after quantise)
 * scatter_addrs  : d_head UB word indices — one per key element
 * token_lane     : INT16 lane within each 128-bit UB word  (token_index % 8)
 *
 * Each 128-bit UB word holds 8 INT16 values.  Element i of the key is written to
 *   ((volatile int16_t *)NPU_SHARED_UB_BASE)[ scatter_addrs[i] * 8 + token_lane ]
 * which is the lane owned by this token inside the K-cache block.
 *
 * Requires TINYNPU_USE_SHARED_SRAM=1 so that the CPU has a direct window into the UB.
 */
static void host_k_cache_scatter_write(
    const TinyTensor *src,
    const int *scatter_addrs,
    int token_lane)
{
    runtime_assert(src->dtype == TINY_DTYPE_INT16, "k_cache_scatter_write: input must be INT16");
    runtime_assert(token_lane >= 0 && token_lane < 8, "k_cache_scatter_write: token_lane out of [0,8)");
#if TINYNPU_USE_SHARED_SRAM
    const int d_head = src->elem_count;
    volatile int16_t *ub = (volatile int16_t *)((uintptr_t)NPU_SHARED_UB_BASE);
    for (int i = 0; i < d_head; i++) {
        ub[scatter_addrs[i] * 8 + token_lane] = (int16_t)tensor_get_i32(src, i);
    }
#else
    (void)scatter_addrs;
    (void)token_lane;
    runtime_assert(0, "k_cache_scatter_write requires TINYNPU_USE_SHARED_SRAM=1");
#endif
}

static void host_v_cache_scatter_write(
    const TinyTensor *src,
    const int *scatter_addrs,
    int scatter_count)
{
    runtime_assert(src->dtype == TINY_DTYPE_INT16, "v_cache_scatter_write: input must be INT16");
    runtime_assert(scatter_count > 0, "v_cache_scatter_write: scatter_count must be positive");
#if TINYNPU_USE_SHARED_SRAM
    volatile int16_t *ub = (volatile int16_t *)((uintptr_t)NPU_SHARED_UB_BASE);
    for (int tile = 0; tile < scatter_count; ++tile) {
        const int base = tile * 8;
        for (int lane = 0; lane < 8; ++lane) {
            const int elem = base + lane;
            if (elem < src->elem_count) {
                ub[scatter_addrs[tile] * 8 + lane] = (int16_t)tensor_get_i32(src, elem);
            }
        }
    }
#else
    (void)scatter_addrs;
    (void)scatter_count;
    runtime_assert(0, "v_cache_scatter_write requires TINYNPU_USE_SHARED_SRAM=1");
#endif
}

static void host_k_cache_scatter_matrix(TinyTensor *dst, const TinyTensor *src, int base_addr)
{
    runtime_assert(dst->dtype == TINY_DTYPE_INT16, "k_cache_scatter_matrix: output must be INT16");
    runtime_assert(src->dtype == TINY_DTYPE_INT16, "k_cache_scatter_matrix: input must be INT16");
    runtime_assert(src->rank >= 2, "k_cache_scatter_matrix expects rank >= 2 input");
    runtime_assert(dst->rank >= 2, "k_cache_scatter_matrix expects rank >= 2 output");
    const int token_count = src->shape[0];
    const int d_head = src->shape[1];
    runtime_assert(dst->shape[0] == d_head, "k_cache_scatter_matrix output row mismatch");
    runtime_assert(dst->shape[1] >= token_count, "k_cache_scatter_matrix output token capacity too small");
    for (int token = 0; token < token_count; ++token) {
        for (int row = 0; row < d_head; ++row) {
            tensor_set_i32(dst, row * dst->shape[1] + token, tensor_get_i32(src, token * d_head + row));
        }
    }
#if TINYNPU_USE_SHARED_SRAM
    {
        volatile int16_t *ub = (volatile int16_t *)((uintptr_t)NPU_SHARED_UB_BASE);
        const int k_tiles = (d_head + 7) / 8;
        const int block_words = k_tiles * 8;
        for (int token = 0; token < token_count; ++token) {
            const int token_block = token / 8;
            const int token_lane = token % 8;
            const int block_base = base_addr + token_block * block_words;
            for (int row = 0; row < d_head; ++row) {
                const int k_tile = row / 8;
                const int row_in_tile = row % 8;
                const int word_addr = block_base + k_tile * 8 + row_in_tile;
                ub[word_addr * 8 + token_lane] = (int16_t)tensor_get_i32(src, token * d_head + row);
            }
        }
    }
#else
    (void)base_addr;
    runtime_assert(0, "k_cache_scatter_matrix requires TINYNPU_USE_SHARED_SRAM=1");
#endif
}

static void host_v_cache_scatter_matrix(TinyTensor *dst, const TinyTensor *src, int base_addr)
{
    runtime_assert(dst->dtype == TINY_DTYPE_INT16, "v_cache_scatter_matrix: output must be INT16");
    runtime_assert(src->dtype == TINY_DTYPE_INT16, "v_cache_scatter_matrix: input must be INT16");
    runtime_assert(src->rank >= 2, "v_cache_scatter_matrix expects rank >= 2 input");
    runtime_assert(dst->rank >= 2, "v_cache_scatter_matrix expects rank >= 2 output");
    const int token_count = src->shape[0];
    const int d_head = src->shape[1];
    runtime_assert(dst->shape[0] >= token_count, "v_cache_scatter_matrix output token capacity too small");
    runtime_assert(dst->shape[1] == d_head, "v_cache_scatter_matrix output column mismatch");
    for (int token = 0; token < token_count; ++token) {
        for (int col = 0; col < d_head; ++col) {
            tensor_set_i32(dst, token * d_head + col, tensor_get_i32(src, token * d_head + col));
        }
    }
#if TINYNPU_USE_SHARED_SRAM
    {
        volatile int16_t *ub = (volatile int16_t *)((uintptr_t)NPU_SHARED_UB_BASE);
        const int n_tiles = (d_head + 7) / 8;
        const int block_words = n_tiles * 8;
        for (int token = 0; token < token_count; ++token) {
            const int token_block = token / 8;
            const int row_in_block = token % 8;
            const int block_base = base_addr + token_block * block_words;
            for (int n_tile = 0; n_tile < n_tiles; ++n_tile) {
                const int word_addr = block_base + n_tile * 8 + row_in_block;
                for (int lane = 0; lane < 8; ++lane) {
                    const int col = n_tile * 8 + lane;
                    if (col < d_head) {
                        ub[word_addr * 8 + lane] = (int16_t)tensor_get_i32(src, token * d_head + col);
                    }
                }
            }
        }
    }
#else
    (void)base_addr;
    runtime_assert(0, "v_cache_scatter_matrix requires TINYNPU_USE_SHARED_SRAM=1");
#endif
}

static void host_commit_k_cache_slot(TinyTensor *base, const TinyTensor *src, int token_index)
{
    runtime_assert(base->dtype == TINY_DTYPE_INT16, "k_cache base must be INT16");
    runtime_assert(src->dtype == TINY_DTYPE_INT16, "k_cache slot must be INT16");
    runtime_assert(base->rank >= 2, "k_cache base expects rank >= 2");
    runtime_assert(src->elem_count == base->shape[0], "k_cache slot size mismatch");
    runtime_assert(token_index >= 0 && token_index < base->shape[1], "k_cache token index out of range");
    const int d_head = base->shape[0];
    for (int row = 0; row < d_head; ++row) {
        tensor_set_i32(base, row * base->shape[1] + token_index, tensor_get_i32(src, row));
    }
}

static void host_commit_v_cache_slot(TinyTensor *base, const TinyTensor *src, int token_index)
{
    runtime_assert(base->dtype == TINY_DTYPE_INT16, "v_cache base must be INT16");
    runtime_assert(src->dtype == TINY_DTYPE_INT16, "v_cache slot must be INT16");
    runtime_assert(base->rank >= 2, "v_cache base expects rank >= 2");
    runtime_assert(src->elem_count == base->shape[1], "v_cache slot size mismatch");
    runtime_assert(token_index >= 0 && token_index < base->shape[0], "v_cache token index out of range");
    const int d_head = base->shape[1];
    const int base_offset = token_index * d_head;
    for (int col = 0; col < d_head; ++col) {
        tensor_set_i32(base, base_offset + col, tensor_get_i32(src, col));
    }
}

static void host_materialize_k_cache_view(TinyTensor *dst, const TinyTensor *base, int token_index)
{
    runtime_assert(dst->dtype == TINY_DTYPE_INT16, "k_cache view must be INT16");
    runtime_assert(base->dtype == TINY_DTYPE_INT16, "k_cache base must be INT16");
    runtime_assert(base->rank >= 2, "k_cache base expects rank >= 2");
    const int d_head = base->shape[0];
    const int token_capacity = base->shape[1];
    if (token_index >= 0) {
        runtime_assert(dst->elem_count == d_head, "k_cache slot view size mismatch");
        runtime_assert(token_index < token_capacity, "k_cache slot token index out of range");
        for (int row = 0; row < d_head; ++row) {
            tensor_set_i32(dst, row, tensor_get_i32(base, row * token_capacity + token_index));
        }
        return;
    }
    runtime_assert(dst->rank >= 2, "k_cache valid view expects rank >= 2");
    runtime_assert(dst->shape[0] == d_head, "k_cache valid view row mismatch");
    const int valid_tokens = dst->shape[1];
    runtime_assert(valid_tokens <= token_capacity, "k_cache valid view width exceeds base");
    for (int row = 0; row < d_head; ++row) {
        for (int col = 0; col < valid_tokens; ++col) {
            tensor_set_i32(dst, row * valid_tokens + col, tensor_get_i32(base, row * token_capacity + col));
        }
    }
}

static void host_materialize_v_cache_view(TinyTensor *dst, const TinyTensor *base, int token_index)
{
    runtime_assert(dst->dtype == TINY_DTYPE_INT16, "v_cache view must be INT16");
    runtime_assert(base->dtype == TINY_DTYPE_INT16, "v_cache base must be INT16");
    runtime_assert(base->rank >= 2, "v_cache base expects rank >= 2");
    const int token_capacity = base->shape[0];
    const int d_head = base->shape[1];
    if (token_index >= 0) {
        runtime_assert(dst->elem_count == d_head, "v_cache slot view size mismatch");
        runtime_assert(token_index < token_capacity, "v_cache slot token index out of range");
        const int base_offset = token_index * d_head;
        for (int col = 0; col < d_head; ++col) {
            tensor_set_i32(dst, col, tensor_get_i32(base, base_offset + col));
        }
        return;
    }
    runtime_assert(dst->rank >= 2, "v_cache valid view expects rank >= 2");
    runtime_assert(dst->shape[1] == d_head, "v_cache valid view column mismatch");
    const int valid_tokens = dst->shape[0];
    runtime_assert(valid_tokens <= token_capacity, "v_cache valid view height exceeds base");
    for (int row = 0; row < valid_tokens; ++row) {
        for (int col = 0; col < d_head; ++col) {
            tensor_set_i32(dst, row * d_head + col, tensor_get_i32(base, row * d_head + col));
        }
    }
}

static void host_rope(TinyTensor *dst, const TinyTensor *src, int head_dim, int position, float theta)
{
    runtime_assert(dst->dtype == TINY_DTYPE_FLOAT32, "rope output must be float");
    runtime_assert(src->rank >= 2, "rope expects rank >= 2");
    runtime_assert(theta > 0.0f, "rope theta must be positive");
    runtime_assert(head_dim > 0 && (head_dim % 2) == 0, "rope head_dim must be positive and even");
    runtime_assert(src->shape[src->rank - 1] == head_dim, "rope last dimension mismatch");
    runtime_assert(dst->elem_count == src->elem_count, "rope size mismatch");

    const int half = head_dim / 2;
    int outer = 1;
    int seq_len = 1;
    if (src->rank == 2) {
        outer = src->shape[0];
        seq_len = 1;
    } else {
        for (int axis = 0; axis < src->rank - 2; ++axis) {
            outer *= src->shape[axis];
        }
        seq_len = src->shape[src->rank - 2];
    }

    float *dst_data = tensor_f32(dst);
    float *src_data = tensor_f32(src);
    if (host_rope_cache_prepare(head_dim, theta, position + seq_len - 1)) {
        for (int outer_idx = 0; outer_idx < outer; ++outer_idx) {
            for (int seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
                int logical_pos = position + seq_idx;
                int base = ((outer_idx * seq_len) + seq_idx) * head_dim;
                const float *cos_row = g_rope_cache_cos[logical_pos];
                const float *sin_row = g_rope_cache_sin[logical_pos];
                for (int i = 0; i < half; ++i) {
                    float c = cos_row[i];
                    float s = sin_row[i];
                    float first = src_data[base + i];
                    float second = src_data[base + half + i];
                    dst_data[base + i] = first * c - second * s;
                    dst_data[base + half + i] = second * c + first * s;
                }
            }
        }
        return;
    }

    {
        const float rope_base = powf(theta, -1.0f / (float)half);
        float inv_freqs[half];
        float delta_cos[half];
        float delta_sin[half];
        float cur_cos[half];
        float cur_sin[half];
        float inv_freq = 1.0f;
        for (int i = 0; i < half; ++i) {
            inv_freqs[i] = inv_freq;
            delta_cos[i] = cosf(inv_freq);
            delta_sin[i] = sinf(inv_freq);
            inv_freq *= rope_base;
        }
        for (int outer_idx = 0; outer_idx < outer; ++outer_idx) {
            {
                const int initial_pos = position;
                for (int i = 0; i < half; ++i) {
                    float angle = (float)initial_pos * inv_freqs[i];
                    cur_cos[i] = cosf(angle);
                    cur_sin[i] = sinf(angle);
                }
            }
            for (int seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
                int base = ((outer_idx * seq_len) + seq_idx) * head_dim;
                for (int i = 0; i < half; ++i) {
                    float c = cur_cos[i];
                    float s = cur_sin[i];
                    float first = src_data[base + i];
                    float second = src_data[base + half + i];
                    dst_data[base + i] = first * c - second * s;
                    dst_data[base + half + i] = second * c + first * s;
                }
                if (src->rank != 2 && seq_idx + 1 < seq_len) {
                    for (int i = 0; i < half; ++i) {
                        float c = cur_cos[i];
                        float s = cur_sin[i];
                        cur_cos[i] = (c * delta_cos[i]) - (s * delta_sin[i]);
                        cur_sin[i] = (s * delta_cos[i]) + (c * delta_sin[i]);
                    }
                }
            }
        }
    }
}

static void host_layout_restore(
    TinyTensor *dst,
    const TinyTensor *src,
    int layout_is_chw,
    int original_rank,
    int out_h,
    int out_w,
    int out_channels)
{
    runtime_assert(src->dtype != TINY_DTYPE_FLOAT32, "layout_restore expects integer input");
    runtime_assert(dst->dtype != TINY_DTYPE_FLOAT32, "layout_restore expects integer output");
    runtime_assert(src->elem_count == out_h * out_w * out_channels, "layout_restore size mismatch");

    for (int h = 0; h < out_h; ++h) {
        for (int w = 0; w < out_w; ++w) {
            for (int c = 0; c < out_channels; ++c) {
                int src_linear = ((h * out_w) + w) * out_channels + c;
                int dst_linear;
                if (layout_is_chw) {
                    dst_linear = ((c * out_h) + h) * out_w + w;
                    if (original_rank == 4) {
                        dst_linear = dst_linear;
                    }
                } else {
                    dst_linear = src_linear;
                }
                tensor_set_i32(dst, dst_linear, tensor_get_i32(src, src_linear));
            }
        }
    }
}

static void host_im2col(
    TinyTensor *dst,
    const TinyTensor *src,
    int kernel_size,
    int stride,
    int padding,
    int input_layout_is_chw)
{
    runtime_assert(src->dtype != TINY_DTYPE_FLOAT32, "im2col expects integer input");
    runtime_assert(dst->dtype != TINY_DTYPE_FLOAT32, "im2col expects integer output");
    runtime_assert(kernel_size > 0 && stride > 0 && padding >= 0, "im2col attrs invalid");

    int h = 0;
    int w = 0;
    int c = 0;
    int has_batch = (src->rank == 4);
    if (input_layout_is_chw) {
        if (has_batch) {
            runtime_assert(src->shape[0] == 1, "im2col only supports batch size 1");
            c = src->shape[1];
            h = src->shape[2];
            w = src->shape[3];
        } else {
            c = src->shape[0];
            h = src->shape[1];
            w = src->shape[2];
        }
    } else {
        runtime_assert(src->rank == 3, "hwc im2col expects rank-3 input");
        h = src->shape[0];
        w = src->shape[1];
        c = src->shape[2];
    }

    const int out_h = ((h + (2 * padding) - kernel_size) / stride) + 1;
    const int out_w = ((w + (2 * padding) - kernel_size) / stride) + 1;
    runtime_assert(dst->elem_count == out_h * out_w * kernel_size * kernel_size * c, "im2col output shape mismatch");

    int patch_index = 0;
    for (int y = 0; y <= (h + 2 * padding - kernel_size); y += stride) {
        for (int x = 0; x <= (w + 2 * padding - kernel_size); x += stride) {
            int out_linear_base = patch_index * (kernel_size * kernel_size * c);
            int out_linear = out_linear_base;
            for (int channel = 0; channel < c; ++channel) {
                for (int ky = 0; ky < kernel_size; ++ky) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        int in_y = y + ky - padding;
                        int in_x = x + kx - padding;
                        int32_t value = 0;
                        if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
                            int src_linear;
                            if (input_layout_is_chw) {
                                src_linear = channel * h * w + in_y * w + in_x;
                            } else {
                                src_linear = (in_y * w + in_x) * c + channel;
                            }
                            value = tensor_get_i32(src, src_linear);
                        }
                        tensor_set_i32(dst, out_linear++, value);
                    }
                }
            }
            patch_index += 1;
        }
    }
}

static void host_im2col_matrix(
    TinyTensor *dst,
    const TinyTensor *src,
    int matrix_h,
    int matrix_w,
    int matrix_c,
    int kernel_size,
    int stride,
    int padding)
{
    runtime_assert(src->dtype != TINY_DTYPE_FLOAT32, "im2col_matrix expects integer input");
    runtime_assert(dst->dtype != TINY_DTYPE_FLOAT32, "im2col_matrix expects integer output");
    runtime_assert(kernel_size > 0 && stride > 0 && padding >= 0, "im2col_matrix attrs invalid");
    runtime_assert(src->rank == 2, "im2col_matrix expects rank-2 [H*W, C] input");
    runtime_assert(matrix_h > 0 && matrix_w > 0 && matrix_c > 0, "im2col_matrix requires positive matrix dims");
    runtime_assert(src->shape[0] == matrix_h * matrix_w, "im2col_matrix H*W mismatch");
    runtime_assert(src->shape[1] == matrix_c, "im2col_matrix C mismatch");

    const int out_h = ((matrix_h + (2 * padding) - kernel_size) / stride) + 1;
    const int out_w = ((matrix_w + (2 * padding) - kernel_size) / stride) + 1;
    runtime_assert(dst->elem_count == out_h * out_w * kernel_size * kernel_size * matrix_c, "im2col_matrix output shape mismatch");

    int patch_index = 0;
    for (int y = 0; y <= (matrix_h + 2 * padding - kernel_size); y += stride) {
        for (int x = 0; x <= (matrix_w + 2 * padding - kernel_size); x += stride) {
            int out_linear_base = patch_index * (kernel_size * kernel_size * matrix_c);
            int out_linear = out_linear_base;
            for (int channel = 0; channel < matrix_c; ++channel) {
                for (int ky = 0; ky < kernel_size; ++ky) {
                    for (int kx = 0; kx < kernel_size; ++kx) {
                        int in_y = y + ky - padding;
                        int in_x = x + kx - padding;
                        int32_t value = 0;
                        if (in_y >= 0 && in_y < matrix_h && in_x >= 0 && in_x < matrix_w) {
                            int src_linear = (in_y * matrix_w + in_x) * matrix_c + channel;
                            value = tensor_get_i32(src, src_linear);
                        }
                        tensor_set_i32(dst, out_linear++, value);
                    }
                }
            }
            patch_index += 1;
        }
    }
}

typedef union {
    int64_t s64;
    struct {
        uint32_t lo;
        int32_t hi;
    } words;
} HostI64Bits;

static int32_t host_floor_div_i32(int32_t numer, int32_t denom)
{
    runtime_assert(denom != 0, "division by zero");
    int32_t quot = numer / denom;
    int32_t rem = numer % denom;
    if (rem != 0 && ((rem < 0) != (denom < 0))) {
        quot -= 1;
    }
    return quot;
}

static int32_t host_round_shift_i32(int32_t value, int32_t shift)
{
    if (shift <= 0) {
        return value;
    }
    runtime_assert(shift < 31, "32-bit rounded shift requires shift < 31");

    uint32_t magnitude = value < 0 ? (uint32_t)(-(value + 1)) + 1u : (uint32_t)value;
    magnitude += (uint32_t)(1u << (shift - 1));
    magnitude >>= shift;
    if (value >= 0) {
        runtime_assert(magnitude <= 0x7fffffffu, "32-bit rounded shift overflow");
        return (int32_t)magnitude;
    }
    if (magnitude == 0x80000000u) {
        return INT32_MIN;
    }
    runtime_assert(magnitude <= 0x7fffffffu, "32-bit rounded shift overflow");
    return -(int32_t)magnitude;
}

static int32_t host_round_div_signed_u32(uint32_t numer_abs, uint32_t denom, int negative)
{
    runtime_assert(denom > 0u, "round_div requires positive denom");
    uint32_t rounded = (numer_abs + (denom / 2u)) / denom;
    if (!negative) {
        runtime_assert(rounded <= 0x7fffffffu, "rounded division overflow");
        return (int32_t)rounded;
    }
    if (rounded == 0x80000000u) {
        return INT32_MIN;
    }
    runtime_assert(rounded <= 0x7fffffffu, "rounded division overflow");
    return -(int32_t)rounded;
}

static int32_t host_round_shift_i64_to_i32(int64_t value, int32_t shift)
{
    if (shift <= 0) {
        runtime_assert(value >= (int64_t)INT32_MIN && value <= (int64_t)INT32_MAX, "requantized value overflow");
        return (int32_t)value;
    }
    runtime_assert(shift < 64, "64-bit rounded shift requires shift < 64");
    runtime_assert(value != INT64_MIN, "64-bit rounded shift does not support INT64_MIN");

    int negative = value < 0;
    HostI64Bits bits = {.s64 = negative ? -value : value};
    uint32_t lo = bits.words.lo;
    uint32_t hi = (uint32_t)bits.words.hi;
    int round_shift = shift - 1;

    if (round_shift < 32) {
        uint32_t add_lo = 1u << round_shift;
        uint32_t prev_lo = lo;
        lo += add_lo;
        if (lo < prev_lo) {
            hi += 1u;
        }
    } else {
        hi += 1u << (round_shift - 32);
    }

    uint32_t out_lo = 0u;
    uint32_t out_hi = 0u;
    if (shift < 32) {
        out_lo = (lo >> shift) | (hi << (32 - shift));
        out_hi = hi >> shift;
    } else if (shift == 32) {
        out_lo = hi;
    } else {
        out_lo = hi >> (shift - 32);
    }

    if (!negative) {
        runtime_assert(out_hi == 0u && out_lo <= 0x7fffffffu, "requantized value overflow");
        return (int32_t)out_lo;
    }
    runtime_assert(out_hi == 0u && out_lo <= 0x80000000u, "requantized value overflow");
    if (out_lo == 0x80000000u) {
        return INT32_MIN;
    }
    return -(int32_t)out_lo;
}

static int32_t host_di_exp(int32_t x_in, int32_t m_i, int32_t k_i)
{
    runtime_assert(m_i > 0, "di_exp requires positive m_i");
    runtime_assert(k_i >= 0 && k_i < 31, "di_exp fast path requires 0 <= k_i < 31");

    int32_t m_f = m_i + (m_i >> 1) - (m_i >> 4);
    runtime_assert(m_f > 0, "di_exp derived m_f must be positive");

    uint32_t s_i_u = ((1u << k_i) + (uint32_t)(m_f / 2)) / (uint32_t)m_f;
    runtime_assert(s_i_u <= 0x7fffffffu, "di_exp derived s_i overflow");
    int32_t s_i = (int32_t)s_i_u;
    runtime_assert(s_i > 0, "di_exp derived s_i must be positive");

    int32_t t = -s_i;
    int32_t q_i = host_floor_div_i32(x_in, t);
    runtime_assert(q_i >= 0, "di_exp expects non-positive input");
    int32_t r_i = x_in - (q_i * t);
    int32_t unshifted_exp = (r_i >> 1) - t;
    if (q_i >= 31) {
        return 0;
    }
    int32_t result = unshifted_exp >> q_i;
    return result > 0 ? result : 0;
}

static int32_t host_int_div_prob(int32_t numer, int32_t denom, int32_t p_out)
{
    runtime_assert(p_out > 0, "probability width must be positive");
    runtime_assert(denom > 0, "probability denominator must be positive");
    int32_t scale = (1 << (p_out - 1)) - 1;
    runtime_assert(scale == 0 || numer <= (INT32_MAX / scale), "sigmoid scaling overflow");
    return (int32_t)(((numer * scale) + (denom / 2)) / denom);
}

static int32_t host_di_sigmoid(int32_t x_in, int32_t m_i, int32_t k_i, int32_t p_out, int32_t alpha_smooth)
{
    runtime_assert(alpha_smooth > 0, "sigmoid smoothing must be positive");

    int32_t x_smoothed = host_floor_div_i32(x_in, alpha_smooth);
    int32_t exp_zero = host_di_exp(0, m_i, k_i);
    int32_t exp_term = 0;
    int32_t numer = 0;

    if (x_smoothed >= 0) {
        exp_term = host_di_exp(-x_smoothed, m_i, k_i);
        numer = exp_zero;
    } else {
        exp_term = host_di_exp(x_smoothed, m_i, k_i);
        numer = exp_term;
    }

    return host_int_div_prob(numer, exp_zero + exp_term, p_out);
}

static int32_t host_h_gelu_i32(int32_t x_in, int32_t x_scale_shift)
{
    runtime_assert(x_scale_shift >= 0, "h_gelu scale shift must be non-negative");
    const int32_t slope_num = 218;
    const int32_t slope_shift = 7;
    runtime_assert(x_scale_shift < 28, "h_gelu scale shift too large for 32-bit denom");
    const int32_t scale_denom = 1 << x_scale_shift;
    const int32_t three_int = 3 * scale_denom;
    const int32_t six_int = 6 * scale_denom;

    int32_t slope_term = host_round_shift_i32(x_in * slope_num, slope_shift);
    int32_t gate_int = slope_term + three_int;
    if (gate_int < 0) {
        gate_int = 0;
    }
    if (gate_int > six_int) {
        gate_int = six_int;
    }

    uint32_t abs_x = x_in < 0 ? (uint32_t)(-(x_in + 1)) + 1u : (uint32_t)x_in;
    uint64_t numer_check = (uint64_t)abs_x * (uint64_t)(uint32_t)gate_int;
    runtime_assert(numer_check <= 0xffffffffu, "h_gelu fast path overflow");
    return host_round_div_signed_u32((uint32_t)numer_check, (uint32_t)six_int, x_in < 0);
}

static int32_t host_apply_ppu(
    int64_t acc,
    int32_t bias,
    int32_t multiplier,
    int32_t shift,
    int32_t activation,
    int32_t h_gelu_x_scale_shift,
    TinyDType out_dtype)
{
    int64_t value = acc + (int64_t)((int32_t)bias);
    value *= (int64_t)(multiplier & 0xFFFF);
    int32_t requantized = host_round_shift_i64_to_i32(value, shift);

    if (activation == HOST_ACT_RELU) {
        if (requantized < 0) {
            requantized = 0;
        }
    } else if (activation == HOST_ACT_SIGMOID) {
        int32_t p_out = out_dtype == TINY_DTYPE_INT4 ? 4 : out_dtype == TINY_DTYPE_INT8 ? 8 : 16;
        int32_t clamped = clip_for_output_dtype(requantized, TINY_DTYPE_INT16);
        return host_di_sigmoid(clamped, multiplier, shift, p_out, 1);
    } else if (activation == HOST_ACT_H_GELU) {
        int32_t clamped = clip_for_output_dtype(requantized, TINY_DTYPE_INT16);
        requantized = host_h_gelu_i32(clamped, h_gelu_x_scale_shift);
    }

    return clip_for_output_dtype(requantized, out_dtype);
}

static int tensor_matrix_rows(const TinyTensor *tensor)
{
    return tensor->rank == 1 ? 1 : tensor->shape[0];
}

static int tensor_matrix_cols(const TinyTensor *tensor)
{
    return tensor->rank == 1 ? tensor->shape[0] : tensor->shape[1];
}

static void host_matmul(
    TinyTensor *dst,
    const TinyTensor *lhs,
    const TinyTensor *rhs,
    const TinyTensor *bias,
    int32_t multiplier,
    int32_t shift,
    int32_t activation,
    int32_t h_gelu_x_scale_shift)
{
    runtime_assert(dst->dtype != TINY_DTYPE_FLOAT32, "host_matmul expects integer output");
    runtime_assert(lhs->dtype != TINY_DTYPE_FLOAT32, "host_matmul expects integer lhs");
    runtime_assert(rhs->dtype != TINY_DTYPE_FLOAT32, "host_matmul expects integer rhs");
    runtime_assert(bias == NULL || bias->dtype != TINY_DTYPE_FLOAT32, "host_matmul expects integer bias");

    const int rows = tensor_matrix_rows(lhs);
    const int inner = tensor_matrix_cols(lhs);
    const int rhs_rows = tensor_matrix_rows(rhs);
    const int cols = tensor_matrix_cols(rhs);

    runtime_assert(inner == rhs_rows, "host_matmul dimension mismatch");
    runtime_assert(tensor_matrix_rows(dst) == rows, "host_matmul output row mismatch");
    runtime_assert(tensor_matrix_cols(dst) == cols, "host_matmul output col mismatch");
    if (bias != NULL) {
        runtime_assert(bias->elem_count == cols, "host_matmul bias width mismatch");
    }

    int32_t *dst_data = tensor_i32(dst);
    const int32_t *lhs_data = tensor_i32(lhs);
    const int32_t *rhs_data = tensor_i32(rhs);
    const int32_t *bias_data = bias == NULL ? NULL : tensor_i32(bias);

    for (int row = 0; row < rows; ++row) {
        const int32_t *lhs_row = lhs_data + (row * inner);
        for (int col = 0; col < cols; ++col) {
            int64_t acc = 0;
            for (int k = 0; k < inner; ++k) {
                acc += (int64_t)lhs_row[k] * (int64_t)rhs_data[k * cols + col];
            }
            int32_t bias_value = bias_data == NULL ? 0 : bias_data[col];
            dst_data[row * cols + col] =
                host_apply_ppu(acc, bias_value, multiplier, shift, activation, h_gelu_x_scale_shift, dst->dtype);
        }
    }
}

static void npu_write_mmvr(const uint32_t chunks[TINY_BUFFER_WORDS_32])
{
    for (int part = 0; part < TINY_BUFFER_WORDS_32; ++part) {
        npu_write32(REG_MMVR + (uint32_t)(part * 4), chunks[part]);
    }
}

static void npu_doorbell(void)
{
    npu_write8(REG_MMVR + (uint32_t)(TINY_MMVR_BYTES - 1), 0);
}

#if TINYNPU_USE_SHARED_SRAM
static inline uintptr_t npu_shared_base_for_addr(uint16_t addr)
{
    return (addr >= TINY_IM_BASE_ADDR) ? (uintptr_t)NPU_SHARED_IM_BASE : (uintptr_t)NPU_SHARED_UB_BASE;
}

static inline uint32_t npu_shared_rel_word_for_addr(uint16_t addr)
{
    return (addr >= TINY_IM_BASE_ADDR) ? (uint32_t)(addr - TINY_IM_BASE_ADDR) : (uint32_t)addr;
}

static void npu_shared_write_word(uint16_t addr, const uint32_t chunks[TINY_BUFFER_WORDS_32])
{
    const uintptr_t base = npu_shared_base_for_addr(addr);
    const uintptr_t word_addr = base + (uintptr_t)(npu_shared_rel_word_for_addr(addr) * (uint32_t)TINY_MMVR_BYTES);
    for (int part = 0; part < TINY_BUFFER_WORDS_32; ++part) {
        volatile uint32_t *slot = (volatile uint32_t *)(word_addr + (uintptr_t)(part * 4u));
        *slot = chunks[part];
    }
}

static void npu_shared_write_image(
    uint16_t base_addr,
    const uint32_t image[][TINY_BUFFER_WORDS_32],
    int word_count)
{
    if (word_count <= 0) {
        return;
    }
    const uintptr_t base = npu_shared_base_for_addr(base_addr);
    const uint32_t rel_word = npu_shared_rel_word_for_addr(base_addr);
    volatile uint32_t *dst =
        (volatile uint32_t *)(base + (uintptr_t)(rel_word * (uint32_t)TINY_MMVR_BYTES));

    for (int i = 0; i < word_count; ++i) {
        for (int part = 0; part < TINY_BUFFER_WORDS_32; ++part) {
            dst[part] = image[i][part];
        }
        dst += TINY_BUFFER_WORDS_32;
    }
}

static void npu_shared_read_word(uint16_t addr, uint32_t chunks[TINY_BUFFER_WORDS_32])
{
    const uintptr_t base = npu_shared_base_for_addr(addr);
    const uintptr_t word_addr = base + (uintptr_t)(npu_shared_rel_word_for_addr(addr) * (uint32_t)TINY_MMVR_BYTES);
    for (int part = 0; part < TINY_BUFFER_WORDS_32; ++part) {
        volatile uint32_t *slot = (volatile uint32_t *)(word_addr + (uintptr_t)(part * 4u));
        chunks[part] = *slot;
    }
}
#endif

static void npu_write_mem_word_mmio(uint16_t addr, const uint32_t chunks[TINY_BUFFER_WORDS_32])
{
    npu_write16(REG_ADDR, addr);
    npu_write8(REG_CMD, CMD_WRITE_MEM);
    npu_write_mmvr(chunks);
}

static void npu_write_mem_word(uint16_t addr, const uint32_t chunks[TINY_BUFFER_WORDS_32])
{
#if TINYNPU_USE_SHARED_SRAM
    if (!g_tinynpu_force_mmio_transfers) {
        npu_shared_write_word(addr, chunks);
        return;
    }
    npu_write_mem_word_mmio(addr, chunks);
#else
    npu_write_mem_word_mmio(addr, chunks);
#endif
}

static int npu_read_mem_word_mmio(uint16_t addr, uint32_t chunks[TINY_BUFFER_WORDS_32])
{
    uint8_t status = 0;
    npu_write16(REG_ADDR, addr);
    npu_write8(REG_CMD, CMD_READ_MEM);
    npu_doorbell();

    for (int poll = 0; poll < 200000; ++poll) {
        status = npu_read8(REG_STATUS);
        if (status == STATUS_DATA_VALID) {
            for (int part = 0; part < TINY_BUFFER_WORDS_32; ++part) {
                chunks[part] = npu_read32(REG_MMVR + (uint32_t)(part * 4u));
            }
            return 0;
        }
    }

    printf("NPU read timeout at 0x%04x status=0x%02x\n", addr, (unsigned)status);
    return -1;
}

static int npu_read_mem_word(uint16_t addr, uint32_t chunks[TINY_BUFFER_WORDS_32], int precision)
{
#if TINYNPU_USE_SHARED_SRAM
    if (g_tinynpu_force_mmio_transfers) {
        return npu_read_mem_word_mmio(addr, chunks);
    }
#if TINYNPU_SHARED_PACKED_READ_MMIO_FALLBACK
    if (precision != 2) {
        return npu_read_mem_word_mmio(addr, chunks);
    }
#else
    (void)precision;
#endif
    npu_shared_read_word(addr, chunks);
    return 0;
#else
    (void)precision;
    return npu_read_mem_word_mmio(addr, chunks);
#endif
}

static int npu_run(uint32_t start_addr)
{
    uint8_t status = 0;
    npu_write32(REG_ARG, start_addr);
    npu_write8(REG_CMD, CMD_RUN);
    npu_doorbell();

    for (int poll = 0; poll < 200000; ++poll) {
        status = npu_read8(REG_STATUS);
        if (status == STATUS_HALTED) {
            return 0;
        }
    }

    printf("NPU run timeout status=0x%02x\n", (unsigned)status);
    return -1;
}

static void lanes_to_chunks(const uint16_t lanes[TINY_ARRAY_SIZE], uint32_t chunks[TINY_BUFFER_WORDS_32])
{
    for (int i = 0; i < TINY_BUFFER_WORDS_32; ++i) {
        chunks[i] = 0u;
    }
    for (int lane = 0; lane < TINY_ARRAY_SIZE; ++lane) {
        int word_index = lane / 2;
        int shift = (lane & 1) ? 16 : 0;
        chunks[word_index] |= ((uint32_t)lanes[lane]) << shift;
    }
}

static int16_t quantize_f32_to_int16(float value, float inv_scale, int zero_point)
{
    int64_t quantized = host_round_to_i64(value * inv_scale) + (int64_t)zero_point;
    return (int16_t)clip_for_dtype(quantized, TINY_DTYPE_INT16);
}

static void pack_tensor_word(
    const TinyTensor *tensor,
    char role_kind,
    int precision,
    int tile0,
    int tile1,
    int lane_selector,
    uint32_t out_chunks[TINY_BUFFER_WORDS_32])
{
    const int p = 1 << (2 - precision);
    const int bits = 16 / p;
    const int mask = (1 << bits) - 1;
    const int rows = tensor->rank == 1 ? 1 : tensor->shape[0];
    const int cols = tensor->rank == 1 ? tensor->shape[0] : tensor->shape[1];
    uint16_t lanes[TINY_ARRAY_SIZE];
    for (int lane = 0; lane < TINY_ARRAY_SIZE; ++lane) {
        uint16_t subword = 0;
        if (role_kind == 'A') {
            int row = tile0 * TINY_ARRAY_SIZE + lane;
            int start_k = (tile1 * TINY_ARRAY_SIZE + lane_selector) * p;
            for (int bit_idx = 0; bit_idx < p; ++bit_idx) {
                int col = start_k + bit_idx;
                int32_t value = 0;
                if (row < rows && col < cols) {
                    value = tensor_get_i32(tensor, row * cols + col);
                }
                subword |= (uint16_t)(((uint32_t)value & (uint32_t)mask) << (bit_idx * bits));
            }
        } else if (role_kind == 'B') {
            int col = tile1 * TINY_ARRAY_SIZE + lane;
            int start_k = (tile0 * TINY_ARRAY_SIZE + lane_selector) * p;
            for (int bit_idx = 0; bit_idx < p; ++bit_idx) {
                int row = start_k + bit_idx;
                int32_t value = 0;
                if (row < rows && col < cols) {
                    value = tensor_get_i32(tensor, row * cols + col);
                }
                subword |= (uint16_t)(((uint32_t)value & (uint32_t)mask) << (bit_idx * bits));
            }
        } else if (role_kind == 'C') {
            int row_start = tile0 * (TINY_ARRAY_SIZE * p) + lane_selector;
            int col = tile1 * TINY_ARRAY_SIZE + lane;
            for (int bit_idx = 0; bit_idx < p; ++bit_idx) {
                int row = row_start + bit_idx * TINY_ARRAY_SIZE;
                int32_t value = 0;
                if (row < rows && col < cols) {
                    value = tensor_get_i32(tensor, row * cols + col);
                }
                subword |= (uint16_t)(((uint32_t)value & (uint32_t)mask) << (bit_idx * bits));
            }
        } else {
            runtime_fail("unsupported pack role");
        }
        lanes[lane] = subword;
    }
    lanes_to_chunks(lanes, out_chunks);
}

static void write_tensor_to_npu(
    const TinyTensor *tensor,
    uint16_t base_addr,
    const char *role,
    int precision,
    int word_count)
{
    const int p = 1 << (2 - precision);
    const int rows = tensor->rank == 1 ? 1 : tensor->shape[0];
    const int cols = tensor->rank == 1 ? tensor->shape[0] : tensor->shape[1];
    const int m_tiles = (rows + TINY_ARRAY_SIZE - 1) / TINY_ARRAY_SIZE;
    uint16_t addr = base_addr;

    if (strcmp(role, "A") == 0) {
        const int k_tiles = ((cols / p) + TINY_ARRAY_SIZE - 1) / TINY_ARRAY_SIZE;
        runtime_assert(word_count == m_tiles * k_tiles * TINY_ARRAY_SIZE, "role A word count mismatch");
        for (int mt = 0; mt < m_tiles; ++mt) {
            for (int kt = 0; kt < k_tiles; ++kt) {
                for (int lane_selector = 0; lane_selector < TINY_ARRAY_SIZE; ++lane_selector) {
                    uint32_t chunks[TINY_BUFFER_WORDS_32];
                    pack_tensor_word(tensor, 'A', precision, mt, kt, lane_selector, chunks);
                    npu_write_mem_word(addr++, chunks);
                }
            }
        }
        return;
    }

    if (strcmp(role, "B") == 0) {
        const int k_tiles = ((rows / p) + TINY_ARRAY_SIZE - 1) / TINY_ARRAY_SIZE;
        const int n_tiles = (cols + TINY_ARRAY_SIZE - 1) / TINY_ARRAY_SIZE;
        runtime_assert(word_count == k_tiles * n_tiles * TINY_ARRAY_SIZE, "role B word count mismatch");
        for (int kt = 0; kt < k_tiles; ++kt) {
            for (int nt = 0; nt < n_tiles; ++nt) {
                for (int lane_selector = 0; lane_selector < TINY_ARRAY_SIZE; ++lane_selector) {
                    uint32_t chunks[TINY_BUFFER_WORDS_32];
                    pack_tensor_word(tensor, 'B', precision, kt, nt, lane_selector, chunks);
                    npu_write_mem_word(addr++, chunks);
                }
            }
        }
        return;
    }

    if (strcmp(role, "C") == 0) {
        const int n_tiles = (cols + TINY_ARRAY_SIZE - 1) / TINY_ARRAY_SIZE;
        const int mt_phys = (m_tiles + p - 1) / p;
        runtime_assert(word_count == mt_phys * n_tiles * TINY_ARRAY_SIZE, "role C word count mismatch");
        for (int mt_phys_idx = 0; mt_phys_idx < mt_phys; ++mt_phys_idx) {
            for (int nt = 0; nt < n_tiles; ++nt) {
                for (int lane_selector = 0; lane_selector < TINY_ARRAY_SIZE; ++lane_selector) {
                    uint32_t chunks[TINY_BUFFER_WORDS_32];
                    pack_tensor_word(tensor, 'C', precision, mt_phys_idx, nt, lane_selector, chunks);
                    npu_write_mem_word(addr++, chunks);
                }
            }
        }
        return;
    }

    if (strcmp(role, "BIAS") == 0) {
        const int n_tiles = (cols + TINY_ARRAY_SIZE - 1) / TINY_ARRAY_SIZE;
        runtime_assert(word_count == n_tiles * 2, "bias word count mismatch");
        const int32_t *values = tensor_i32(tensor);
        for (int nt = 0; nt < n_tiles; ++nt) {
            uint32_t chunks0[TINY_BUFFER_WORDS_32] = {0};
            uint32_t chunks1[TINY_BUFFER_WORDS_32] = {0};
            for (int j = 0; j < 4; ++j) {
                int idx0 = nt * TINY_ARRAY_SIZE + j;
                int idx1 = nt * TINY_ARRAY_SIZE + 4 + j;
                uint32_t value0 = (idx0 < cols) ? (uint32_t)values[idx0] : 0u;
                uint32_t value1 = (idx1 < cols) ? (uint32_t)values[idx1] : 0u;
                chunks0[j] = value0;
                chunks1[j] = value1;
            }
            npu_write_mem_word(addr++, chunks0);
            npu_write_mem_word(addr++, chunks1);
        }
        return;
    }

    runtime_fail("unsupported NPU tensor role");
}

static void write_tensor_to_npu_quantized_a_int16_from_float(
    const TinyTensor *tensor,
    uint16_t base_addr,
    int word_count,
    float inv_scale,
    int zero_point)
{
    runtime_assert(tensor->dtype == TINY_DTYPE_FLOAT32, "quantized A write expects float32 input");
    runtime_assert(inv_scale > 0.0f, "quantized A write expects positive inv_scale");
    const int rows = tensor->rank == 1 ? 1 : tensor->shape[0];
    const int cols = tensor->rank == 1 ? tensor->shape[0] : tensor->shape[1];
    const int m_tiles = (rows + TINY_ARRAY_SIZE - 1) / TINY_ARRAY_SIZE;
    const int k_tiles = (cols + TINY_ARRAY_SIZE - 1) / TINY_ARRAY_SIZE;
    runtime_assert(word_count == m_tiles * k_tiles * TINY_ARRAY_SIZE, "quantized A word count mismatch");

    uint16_t addr = base_addr;
    for (int mt = 0; mt < m_tiles; ++mt) {
        for (int kt = 0; kt < k_tiles; ++kt) {
            for (int lane_selector = 0; lane_selector < TINY_ARRAY_SIZE; ++lane_selector) {
                uint16_t lanes[TINY_ARRAY_SIZE];
                uint32_t chunks[TINY_BUFFER_WORDS_32];
                for (int lane = 0; lane < TINY_ARRAY_SIZE; ++lane) {
                    const int row = mt * TINY_ARRAY_SIZE + lane;
                    const int col = kt * TINY_ARRAY_SIZE + lane_selector;
                    int16_t value = 0;
                    if (row < rows && col < cols) {
                        value = quantize_f32_to_int16(
                            tensor_get_float(tensor, row * cols + col),
                            inv_scale,
                            zero_point);
                    }
                    lanes[lane] = (uint16_t)value;
                }
                lanes_to_chunks(lanes, chunks);
                npu_write_mem_word(addr++, chunks);
            }
        }
    }
}

static void read_role_c_tensor(TinyTensor *dst, uint16_t addr, int precision)
{
    runtime_assert(dst->dtype != TINY_DTYPE_FLOAT32, "NPU readback expects integer output tensor");
    const int p = 1 << (2 - precision);
    const int bits = 16 / p;
    const int mask = (1 << bits) - 1;
    const int rows = dst->rank == 1 ? 1 : dst->shape[0];
    const int cols = dst->rank == 1 ? dst->shape[0] : dst->shape[1];
    const int m_tiles = (rows + TINY_ARRAY_SIZE - 1) / TINY_ARRAY_SIZE;
    const int n_tiles = (cols + TINY_ARRAY_SIZE - 1) / TINY_ARRAY_SIZE;
    const int mt_phys = (m_tiles + p - 1) / p;
    uint32_t chunks[TINY_BUFFER_WORDS_32];

    for (int mtp = 0; mtp < mt_phys; ++mtp) {
        for (int nt = 0; nt < n_tiles; ++nt) {
            uint16_t tile_addr = (uint16_t)(addr + (mtp * n_tiles * TINY_ARRAY_SIZE) + (nt * TINY_ARRAY_SIZE));
            for (int row_in_tile = 0; row_in_tile < TINY_ARRAY_SIZE; ++row_in_tile) {
                runtime_assert(npu_read_mem_word((uint16_t)(tile_addr + row_in_tile), chunks, precision) == 0, "readback failed");
                for (int lane = 0; lane < TINY_ARRAY_SIZE; ++lane) {
                    uint32_t lane_word = chunks[lane / 2];
                    uint16_t packed_lane = (lane & 1) ? (uint16_t)(lane_word >> 16) : (uint16_t)(lane_word & 0xFFFFu);
                    int col_idx = nt * TINY_ARRAY_SIZE + lane;
                    for (int bit_idx = 0; bit_idx < p; ++bit_idx) {
                        int mt = mtp * p + bit_idx;
                        int row_idx = mt * TINY_ARRAY_SIZE + row_in_tile;
                        if (row_idx < rows && col_idx < cols) {
                            int32_t value = (packed_lane >> (bit_idx * bits)) & mask;
                            if (value & (1 << (bits - 1))) {
                                value -= (1 << bits);
                            }
                            tensor_set_i32(dst, row_idx * cols + col_idx, value);
                        }
                    }
                }
            }
        }
    }
}

static void read_role_a_tensor(TinyTensor *dst, uint16_t addr, int precision)
{
    runtime_assert(dst->dtype != TINY_DTYPE_FLOAT32, "NPU readback expects integer output tensor");
    const int p = 1 << (2 - precision);
    const int bits = 16 / p;
    const int mask = (1 << bits) - 1;
    const int rows = dst->rank == 1 ? 1 : dst->shape[0];
    const int cols = dst->rank == 1 ? dst->shape[0] : dst->shape[1];
    const int m_tiles = (rows + TINY_ARRAY_SIZE - 1) / TINY_ARRAY_SIZE;
    const int k_words = (cols + p - 1) / p;
    const int k_tiles = (k_words + TINY_ARRAY_SIZE - 1) / TINY_ARRAY_SIZE;
    uint32_t chunks[TINY_BUFFER_WORDS_32];

    for (int mt = 0; mt < m_tiles; ++mt) {
        for (int kt = 0; kt < k_tiles; ++kt) {
            uint16_t tile_addr = (uint16_t)(addr + (mt * k_tiles * TINY_ARRAY_SIZE) + (kt * TINY_ARRAY_SIZE));
            for (int col_in_tile = 0; col_in_tile < TINY_ARRAY_SIZE; ++col_in_tile) {
                runtime_assert(npu_read_mem_word((uint16_t)(tile_addr + col_in_tile), chunks, precision) == 0, "readback failed");
                for (int row_in_tile = 0; row_in_tile < TINY_ARRAY_SIZE; ++row_in_tile) {
                    uint32_t lane_word = chunks[row_in_tile / 2];
                    uint16_t packed_lane = (row_in_tile & 1) ? (uint16_t)(lane_word >> 16) : (uint16_t)(lane_word & 0xFFFFu);
                    int row_idx = mt * TINY_ARRAY_SIZE + row_in_tile;
                    for (int bit_idx = 0; bit_idx < p; ++bit_idx) {
                        int col_idx = ((kt * TINY_ARRAY_SIZE) + col_in_tile) * p + bit_idx;
                        if (row_idx < rows && col_idx < cols) {
                            int32_t value = (packed_lane >> (bit_idx * bits)) & mask;
                            if (value & (1 << (bits - 1))) {
                                value -= (1 << bits);
                            }
                            tensor_set_i32(dst, row_idx * cols + col_idx, value);
                        }
                    }
                }
            }
        }
    }
}

static void read_role_b_tensor(TinyTensor *dst, uint16_t addr, int precision)
{
    runtime_assert(dst->dtype != TINY_DTYPE_FLOAT32, "NPU readback expects integer output tensor");
    const int p = 1 << (2 - precision);
    const int bits = 16 / p;
    const int mask = (1 << bits) - 1;
    const int rows = dst->rank == 1 ? 1 : dst->shape[0];
    const int cols = dst->rank == 1 ? dst->shape[0] : dst->shape[1];
    const int k_words = (rows + p - 1) / p;
    const int k_tiles = (k_words + TINY_ARRAY_SIZE - 1) / TINY_ARRAY_SIZE;
    const int n_tiles = (cols + TINY_ARRAY_SIZE - 1) / TINY_ARRAY_SIZE;
    uint32_t chunks[TINY_BUFFER_WORDS_32];

    for (int kt = 0; kt < k_tiles; ++kt) {
        for (int nt = 0; nt < n_tiles; ++nt) {
            uint16_t tile_addr = (uint16_t)(addr + (kt * n_tiles * TINY_ARRAY_SIZE) + (nt * TINY_ARRAY_SIZE));
            for (int row_word = 0; row_word < TINY_ARRAY_SIZE; ++row_word) {
                /* B-layout host readback is mainly a verification/debug path. Use MMIO reads here
                 * to avoid shared-window lane selection issues in the example testbench wrapper. */
                runtime_assert(npu_read_mem_word_mmio((uint16_t)(tile_addr + row_word), chunks) == 0, "readback failed");
                for (int col_in_tile = 0; col_in_tile < TINY_ARRAY_SIZE; ++col_in_tile) {
                    uint32_t lane_word = chunks[col_in_tile / 2];
                    uint16_t packed_lane = (col_in_tile & 1) ? (uint16_t)(lane_word >> 16) : (uint16_t)(lane_word & 0xFFFFu);
                    int col_idx = nt * TINY_ARRAY_SIZE + col_in_tile;
                    for (int bit_idx = 0; bit_idx < p; ++bit_idx) {
                        int row_idx = ((kt * TINY_ARRAY_SIZE) + row_word) * p + bit_idx;
                        if (row_idx < rows && col_idx < cols) {
                            int32_t value = (packed_lane >> (bit_idx * bits)) & mask;
                            if (value & (1 << (bits - 1))) {
                                value -= (1 << bits);
                            }
                            tensor_set_i32(dst, row_idx * cols + col_idx, value);
                        }
                    }
                }
            }
        }
    }
}

static void read_role_k_tensor(TinyTensor *dst, uint16_t addr, int precision)
{
    runtime_assert(precision == 2, "K-cache readback currently supports INT16 only");
    runtime_assert(dst->dtype != TINY_DTYPE_FLOAT32, "NPU readback expects integer output tensor");
    const int rows = dst->rank == 1 ? 1 : dst->shape[0];
    const int cols = dst->rank == 1 ? dst->shape[0] : dst->shape[1];
    const int k_tiles = (rows + TINY_ARRAY_SIZE - 1) / TINY_ARRAY_SIZE;
    const int token_blocks = (cols + TINY_ARRAY_SIZE - 1) / TINY_ARRAY_SIZE;
    const int block_word_count = k_tiles * TINY_ARRAY_SIZE;
    uint32_t chunks[TINY_BUFFER_WORDS_32];

    for (int token_block = 0; token_block < token_blocks; ++token_block) {
        uint16_t block_base = (uint16_t)(addr + (token_block * block_word_count));
        for (int k_tile = 0; k_tile < k_tiles; ++k_tile) {
            uint16_t tile_addr = (uint16_t)(block_base + (k_tile * TINY_ARRAY_SIZE));
            for (int row_in_tile = 0; row_in_tile < TINY_ARRAY_SIZE; ++row_in_tile) {
                runtime_assert(npu_read_mem_word_mmio((uint16_t)(tile_addr + row_in_tile), chunks) == 0, "readback failed");
                int row_idx = k_tile * TINY_ARRAY_SIZE + row_in_tile;
                if (row_idx >= rows) {
                    continue;
                }
                for (int token_lane = 0; token_lane < TINY_ARRAY_SIZE; ++token_lane) {
                    int col_idx = token_block * TINY_ARRAY_SIZE + token_lane;
                    if (col_idx >= cols) {
                        continue;
                    }
                    uint32_t lane_word = chunks[token_lane / 2];
                    uint16_t packed_lane = (token_lane & 1) ? (uint16_t)(lane_word >> 16) : (uint16_t)(lane_word & 0xFFFFu);
                    int16_t value = (int16_t)packed_lane;
                    tensor_set_i32(dst, row_idx * cols + col_idx, (int32_t)value);
                }
            }
        }
    }
}

static void read_tensor_from_npu(TinyTensor *dst, uint16_t addr, const char *role, int precision)
{
    if (strcmp(role, "A") == 0) {
        read_role_a_tensor(dst, addr, precision);
        return;
    }
    if (strcmp(role, "B") == 0) {
        read_role_b_tensor(dst, addr, precision);
        return;
    }
    if (strcmp(role, "C") == 0) {
        read_role_c_tensor(dst, addr, precision);
        return;
    }
    if (strcmp(role, "K") == 0) {
        read_role_k_tensor(dst, addr, precision);
        return;
    }
    runtime_fail("unsupported NPU readback role");
}

static void load_ub_image(uint16_t base_addr, const uint32_t image[][TINY_BUFFER_WORDS_32], int word_count)
{
#if TINYNPU_USE_SHARED_SRAM
    if (!g_tinynpu_force_mmio_transfers) {
        npu_shared_write_image(base_addr, image, word_count);
        return;
    }
#endif
    for (int i = 0; i < word_count; ++i) {
        npu_write_mem_word((uint16_t)(base_addr + i), image[i]);
    }
}

static void load_im_image(uint16_t base_addr, const uint32_t image[][TINY_BUFFER_WORDS_32], int word_count)
{
#if TINYNPU_USE_SHARED_SRAM
    if (!g_tinynpu_force_mmio_transfers) {
        npu_shared_write_image(base_addr, image, word_count);
        return;
    }
#endif
    for (int i = 0; i < word_count; ++i) {
        npu_write_mem_word((uint16_t)(base_addr + i), image[i]);
    }
}

static void print_float_scalar(float value)
{
    if (value < 0.0f) {
        putchar('-');
        value = -value;
    }
    int whole = (int)value;
    int frac = (int)((value - (float)whole) * 1000.0f + 0.5f);
    printf("%d.%03d", whole, frac);
}

static void print_tensor(const TinyTensor *tensor)
{
    printf("%s shape=(", tensor->name);
    for (int axis = 0; axis < tensor->rank; ++axis) {
        if (axis) {
            printf(", ");
        }
        printf("%d", tensor->shape[axis]);
    }
    printf(")\n");

    if (tensor->rank == 2) {
        const int rows = tensor->shape[0];
        const int cols = tensor->shape[1];
        for (int row = 0; row < rows; ++row) {
            printf("  row %d:", row);
            for (int col = 0; col < cols; ++col) {
                int linear = row * cols + col;
                putchar(' ');
                if (tensor->dtype == TINY_DTYPE_FLOAT32) {
                    print_float_scalar(tensor_get_float(tensor, linear));
                } else {
                    printf("%ld", (long)tensor_get_i32(tensor, linear));
                }
            }
            printf("\n");
        }
        return;
    }

    printf("  values:");
    for (int i = 0; i < tensor->elem_count; ++i) {
        putchar(' ');
        if (tensor->dtype == TINY_DTYPE_FLOAT32) {
            print_float_scalar(tensor_get_float(tensor, i));
        } else {
            printf("%ld", (long)tensor_get_i32(tensor, i));
        }
    }
    printf("\n");
}

static int tensor_matches_expected(const TinyTensor *actual, const TinyTensor *expected)
{
    if (actual->dtype != expected->dtype || actual->elem_count != expected->elem_count) {
        return 0;
    }
    if (actual->dtype == TINY_DTYPE_FLOAT32) {
        for (int i = 0; i < actual->elem_count; ++i) {
            if (host_absf(tensor_get_float(actual, i) - tensor_get_float(expected, i)) > 1e-3f) {
                return 0;
            }
        }
        return 1;
    }
    for (int i = 0; i < actual->elem_count; ++i) {
        if (tensor_get_i32(actual, i) != tensor_get_i32(expected, i)) {
            return 0;
        }
    }
    return 1;
}

__GENERATED_DECLS__

int main(void)
{
__GENERATED_MAIN__
}
