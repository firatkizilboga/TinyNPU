#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPU_BASE 0x30000000u
#define TB_TIMER_CTRL_BASE 0x15000000u
#define TB_TIMER_COUNT_REG 0x15001000u
#define TINY_ARRAY_SIZE __TINY_ARRAY_SIZE__
#define TINY_BUFFER_WORDS_32 __TINY_BUFFER_WORDS_32__
#define TINY_MMVR_BYTES (TINY_BUFFER_WORDS_32 * 4)

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
        float rounded = roundf(value);
        if (fabsf(value - rounded) > 1e-6f) {
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

static void host_sigmoid(TinyTensor *dst, const TinyTensor *src)
{
    runtime_assert(dst->dtype == TINY_DTYPE_FLOAT32, "sigmoid expects float output");
    runtime_assert(dst->elem_count == src->elem_count, "sigmoid size mismatch");
    for (int i = 0; i < src->elem_count; ++i) {
        float value = tensor_get_float(src, i);
        tensor_set_float(dst, i, 1.0f / (1.0f + expf(-value)));
    }
}

static void host_gelu(TinyTensor *dst, const TinyTensor *src)
{
    runtime_assert(dst->dtype == TINY_DTYPE_FLOAT32, "gelu expects float output");
    runtime_assert(dst->elem_count == src->elem_count, "gelu size mismatch");
    for (int i = 0; i < src->elem_count; ++i) {
        float value = tensor_get_float(src, i);
        float erf_term = erff(value / sqrtf(2.0f));
        tensor_set_float(dst, i, 0.5f * value * (1.0f + erf_term));
    }
}

static void host_quantize(TinyTensor *dst, const TinyTensor *src, float scale, int zero_point)
{
    runtime_assert(scale > 0.0f, "quantize scale must be positive");
    runtime_assert(dst->dtype != TINY_DTYPE_FLOAT32, "quantize output must be integer");
    runtime_assert(dst->elem_count == src->elem_count, "quantize size mismatch");
    for (int i = 0; i < src->elem_count; ++i) {
        float source = tensor_get_float(src, i);
        int64_t quantized = (int64_t)lrintf(source / scale) + (int64_t)zero_point;
        tensor_set_i32(dst, i, clip_for_dtype(quantized, dst->dtype));
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
        int64_t quantized = (int64_t)lrintf(source * scale) + (int64_t)zero_point;
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
                float exp_value = expf(tensor_get_float(src, linear) - max_value);
                tensor_set_float(dst, linear, exp_value);
                sum += exp_value;
            }
            runtime_assert(sum != 0.0f, "softmax sum is zero");
            for (int axis_idx = 0; axis_idx < extent; ++axis_idx) {
                int linear = ((outer_idx * extent) + axis_idx) * inner + inner_idx;
                tensor_set_float(dst, linear, tensor_get_float(dst, linear) / sum);
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

static int64_t floor_div_i64(int64_t numer, int64_t denom)
{
    runtime_assert(denom != 0, "division by zero");
    int64_t quot = numer / denom;
    int64_t rem = numer % denom;
    if (rem != 0 && ((rem < 0) != (denom < 0))) {
        quot -= 1;
    }
    return quot;
}

static int32_t host_di_exp(int32_t x_in, int32_t m_i, int32_t k_i)
{
    runtime_assert(m_i > 0, "di_exp requires positive m_i");
    runtime_assert(k_i >= 0, "di_exp requires non-negative k_i");

    int64_t m_f = (int64_t)m_i + ((int64_t)m_i >> 1) - ((int64_t)m_i >> 4);
    runtime_assert(m_f > 0, "di_exp derived m_f must be positive");

    int64_t s_i = ((((int64_t)1) << k_i) + (m_f / 2)) / m_f;
    runtime_assert(s_i > 0, "di_exp derived s_i must be positive");

    int64_t t = -s_i;
    int64_t q_i = floor_div_i64((int64_t)x_in, t);
    int64_t r_i = (int64_t)x_in - (q_i * t);
    int64_t unshifted_exp = (r_i >> 1) - t;
    int64_t result = unshifted_exp >> q_i;
    return (int32_t)(result > 0 ? result : 0);
}

static int32_t host_int_div_prob(int32_t numer, int32_t denom, int32_t p_out)
{
    runtime_assert(p_out > 0, "probability width must be positive");
    runtime_assert(denom > 0, "probability denominator must be positive");
    int32_t scale = (1 << (p_out - 1)) - 1;
    return (int32_t)((((int64_t)numer * (int64_t)scale) + (denom / 2)) / denom);
}

static int32_t host_di_sigmoid(int32_t x_in, int32_t m_i, int32_t k_i, int32_t p_out, int32_t alpha_smooth)
{
    runtime_assert(alpha_smooth > 0, "sigmoid smoothing must be positive");

    int32_t x_smoothed = (int32_t)floor_div_i64((int64_t)x_in, (int64_t)alpha_smooth);
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

static int32_t host_round_shift_signed(int64_t value, int32_t shift)
{
    if (shift <= 0) {
        return (int32_t)value;
    }
    int64_t rounder = ((int64_t)1) << (shift - 1);
    if (value >= 0) {
        return (int32_t)((value + rounder) >> shift);
    }
    return (int32_t)(-(((-value) + rounder) >> shift));
}

static int32_t host_round_div_signed(int64_t numer, int32_t denom)
{
    runtime_assert(denom > 0, "round_div requires positive denom");
    int32_t half = denom / 2;
    if (numer >= 0) {
        return (int32_t)((numer + half) / denom);
    }
    return (int32_t)(-(((-numer) + half) / denom));
}

static int32_t host_h_gelu_i32(int32_t x_in, int32_t x_scale_shift)
{
    runtime_assert(x_scale_shift >= 0, "h_gelu scale shift must be non-negative");
    const int32_t slope_num = 218;
    const int32_t slope_shift = 7;
    const int32_t scale_denom = 1 << x_scale_shift;
    const int32_t three_int = 3 * scale_denom;
    const int32_t six_int = 6 * scale_denom;

    int32_t slope_term = host_round_shift_signed((int64_t)x_in * (int64_t)slope_num, slope_shift);
    int32_t gate_int = slope_term + three_int;
    if (gate_int < 0) {
        gate_int = 0;
    }
    if (gate_int > six_int) {
        gate_int = six_int;
    }
    return host_round_div_signed((int64_t)x_in * (int64_t)gate_int, six_int);
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
    if (shift > 0) {
        value = (value + (((int64_t)1) << (shift - 1))) >> shift;
    }

    if (activation == HOST_ACT_RELU) {
        if (value < 0) {
            value = 0;
        }
    } else if (activation == HOST_ACT_SIGMOID) {
        int32_t p_out = out_dtype == TINY_DTYPE_INT4 ? 4 : out_dtype == TINY_DTYPE_INT8 ? 8 : 16;
        int32_t clamped = clip_for_output_dtype(value, TINY_DTYPE_INT16);
        return host_di_sigmoid(clamped, multiplier, shift, p_out, 1);
    } else if (activation == HOST_ACT_H_GELU) {
        int32_t clamped = clip_for_output_dtype(value, TINY_DTYPE_INT16);
        value = host_h_gelu_i32(clamped, h_gelu_x_scale_shift);
    }

    return clip_for_output_dtype(value, out_dtype);
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

    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int64_t acc = 0;
            for (int k = 0; k < inner; ++k) {
                int32_t lhs_value = tensor_get_i32(lhs, row * inner + k);
                int32_t rhs_value = tensor_get_i32(rhs, k * cols + col);
                acc += (int64_t)lhs_value * (int64_t)rhs_value;
            }
            int32_t bias_value = bias == NULL ? 0 : tensor_get_i32(bias, col);
            tensor_set_i32(
                dst,
                row * cols + col,
                host_apply_ppu(acc, bias_value, multiplier, shift, activation, h_gelu_x_scale_shift, dst->dtype));
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

static void npu_write_mem_word(uint16_t addr, const uint32_t chunks[TINY_BUFFER_WORDS_32])
{
    npu_write16(REG_ADDR, addr);
    npu_write8(REG_CMD, CMD_WRITE_MEM);
    npu_write_mmvr(chunks);
}

static int npu_read_mem_word(uint16_t addr, uint8_t out[TINY_MMVR_BYTES])
{
    uint8_t status = 0;
    npu_write16(REG_ADDR, addr);
    npu_write8(REG_CMD, CMD_READ_MEM);
    npu_doorbell();

    for (int poll = 0; poll < 200000; ++poll) {
        status = npu_read8(REG_STATUS);
        if (status == STATUS_DATA_VALID) {
            for (int i = 0; i < TINY_MMVR_BYTES; ++i) {
                out[i] = npu_read8(REG_MMVR + (uint32_t)i);
            }
            return 0;
        }
    }

    printf("NPU read timeout at 0x%04x status=0x%02x\n", addr, (unsigned)status);
    return -1;
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
    uint8_t bytes[TINY_MMVR_BYTES];

    for (int mtp = 0; mtp < mt_phys; ++mtp) {
        for (int nt = 0; nt < n_tiles; ++nt) {
            uint16_t tile_addr = (uint16_t)(addr + (mtp * n_tiles * TINY_ARRAY_SIZE) + (nt * TINY_ARRAY_SIZE));
            for (int row_in_tile = 0; row_in_tile < TINY_ARRAY_SIZE; ++row_in_tile) {
                runtime_assert(npu_read_mem_word((uint16_t)(tile_addr + row_in_tile), bytes) == 0, "readback failed");
                for (int lane = 0; lane < TINY_ARRAY_SIZE; ++lane) {
                    uint16_t packed_lane = (uint16_t)bytes[lane * 2] | ((uint16_t)bytes[lane * 2 + 1] << 8);
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
    uint8_t bytes[TINY_MMVR_BYTES];

    for (int mt = 0; mt < m_tiles; ++mt) {
        for (int kt = 0; kt < k_tiles; ++kt) {
            uint16_t tile_addr = (uint16_t)(addr + (mt * k_tiles * TINY_ARRAY_SIZE) + (kt * TINY_ARRAY_SIZE));
            for (int col_in_tile = 0; col_in_tile < TINY_ARRAY_SIZE; ++col_in_tile) {
                runtime_assert(npu_read_mem_word((uint16_t)(tile_addr + col_in_tile), bytes) == 0, "readback failed");
                for (int row_in_tile = 0; row_in_tile < TINY_ARRAY_SIZE; ++row_in_tile) {
                    uint16_t packed_lane = (uint16_t)bytes[row_in_tile * 2] | ((uint16_t)bytes[row_in_tile * 2 + 1] << 8);
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
    uint8_t bytes[TINY_MMVR_BYTES];

    for (int kt = 0; kt < k_tiles; ++kt) {
        for (int nt = 0; nt < n_tiles; ++nt) {
            uint16_t tile_addr = (uint16_t)(addr + (kt * n_tiles * TINY_ARRAY_SIZE) + (nt * TINY_ARRAY_SIZE));
            for (int row_word = 0; row_word < TINY_ARRAY_SIZE; ++row_word) {
                runtime_assert(npu_read_mem_word((uint16_t)(tile_addr + row_word), bytes) == 0, "readback failed");
                for (int col_in_tile = 0; col_in_tile < TINY_ARRAY_SIZE; ++col_in_tile) {
                    uint16_t packed_lane = (uint16_t)bytes[col_in_tile * 2] | ((uint16_t)bytes[col_in_tile * 2 + 1] << 8);
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
    runtime_fail("unsupported NPU readback role");
}

static void load_ub_image(uint16_t base_addr, const uint32_t image[][TINY_BUFFER_WORDS_32], int word_count)
{
    for (int i = 0; i < word_count; ++i) {
        npu_write_mem_word((uint16_t)(base_addr + i), image[i]);
    }
}

static void load_im_image(uint16_t base_addr, const uint32_t image[][TINY_BUFFER_WORDS_32], int word_count)
{
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
            if (fabsf(tensor_get_float(actual, i) - tensor_get_float(expected, i)) > 1e-5f) {
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
