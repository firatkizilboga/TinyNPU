#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPU_BASE 0x30000000u
#define TB_TIMER_CTRL_BASE 0x15000000u
#define TB_TIMER_COUNT_REG 0x15001000u
#define TINY_IM_BASE_ADDR 0x8000u
#define NPU_SHARED_UB_BASE 0x31000000u
#define NPU_SHARED_IM_BASE 0x32000000u
#define TINY_ARRAY_SIZE 8
#define TINY_BUFFER_WORDS_32 4
#define TINY_MMVR_BYTES (TINY_BUFFER_WORDS_32 * 4)

#ifndef TINYNPU_USE_SHARED_SRAM
#define TINYNPU_USE_SHARED_SRAM 0
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

static void npu_shared_read_word(uint16_t addr, uint8_t out[TINY_MMVR_BYTES])
{
    const uintptr_t base = npu_shared_base_for_addr(addr);
    const uintptr_t word_addr = base + (uintptr_t)(npu_shared_rel_word_for_addr(addr) * (uint32_t)TINY_MMVR_BYTES);
    for (int part = 0; part < TINY_BUFFER_WORDS_32; ++part) {
        volatile uint32_t *slot = (volatile uint32_t *)(word_addr + (uintptr_t)(part * 4u));
        const uint32_t value = *slot;
        out[part * 4 + 0] = (uint8_t)((value >> 0) & 0xFFu);
        out[part * 4 + 1] = (uint8_t)((value >> 8) & 0xFFu);
        out[part * 4 + 2] = (uint8_t)((value >> 16) & 0xFFu);
        out[part * 4 + 3] = (uint8_t)((value >> 24) & 0xFFu);
    }
}
#endif

static void npu_write_mem_word(uint16_t addr, const uint32_t chunks[TINY_BUFFER_WORDS_32])
{
#if TINYNPU_USE_SHARED_SRAM
    npu_shared_write_word(addr, chunks);
#else
    npu_write16(REG_ADDR, addr);
    npu_write8(REG_CMD, CMD_WRITE_MEM);
    npu_write_mmvr(chunks);
#endif
}

static int npu_read_mem_word(uint16_t addr, uint8_t out[TINY_MMVR_BYTES])
{
#if TINYNPU_USE_SHARED_SRAM
    npu_shared_read_word(addr, out);
    return 0;
#else
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

static float x_data[64] __attribute__((section(".data"))) = {
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.019607843831181526f, 0.07058823853731155f, 0.027450980618596077f, 0.01568627543747425f, 0.01568627543747425f, 0.003921568859368563f, 0.0f, 0.0f, 0.125490203499794f, 0.4470588266849518f, 0.4588235318660736f, 0.4588235318660736f, 0.4901960790157318f, 0.11764705926179886f, 0.0f, 0.0f, 0.007843137718737125f, 0.027450980618596077f, 0.07058823853731155f, 0.1725490242242813f, 0.545098066329956f, 0.09019608050584793f, 0.0f, 0.0f, 0.0f, 0.0f, 0.003921568859368563f, 0.3176470696926117f, 0.32549020648002625f, 0.003921568859368563f, 0.0f, 0.0f, 0.0f, 0.0f, 0.13725490868091583f, 0.501960813999176f, 0.07058823853731155f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0313725508749485f, 0.49803921580314636f, 0.2549019753932953f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.09019608050584793f, 0.501960813999176f, 0.0784313753247261f, 0.0f, 0.0f, 0.0f
};

static TinyTensor x = {"x", x_data, TINY_DTYPE_FLOAT32, 2, {1, 64, 1, 1}, 64};

static int32_t q_in_data[64] __attribute__((section(".noinit")));

static TinyTensor q_in = {"q_in", q_in_data, TINY_DTYPE_INT16, 2, {1, 64, 1, 1}, 64};

static int32_t inner_fc1_weight_t_data[4096] __attribute__((section(".data"))) = {
    -9092, 13465, -3367, 12262, 10092, 12914, 11523, -1684, 14478, -1058, 2848, -1783, -3965, -8361, 5690, -9516, 1180, 8721, 5287, -8024, 226, -2824, -758, -5490, -7571, -383, 10253, 493, 11165, -4256, 719, 6951, 3991, 9012, -8997, 7644, 1, -7967, -7919, -10045, -8542, 12162, 12706, -1390, 5244, -7056, 12152, 8589, 749, 12083, 12349, 11729, -12864, 7698, -6599, 10906, 9218, 13184, -749, 4178, -10437, -3002, -5335, -3061, 15991, 18358, 4608, -589, 5451, 1614, 5509, 3931, 6070, 13605, 14764, -3033, -4835, 11656, 1786, 18816, 7938, -4042, -5509, 16199, 11679, 7456, -16949, -3886, 15752, -2315, -767, 17719, 12518, 400, 15077, -3223, 13224, 7513, 641, 17264, -1583, -6053, -6775, 11601, -3354, -7088, 7685, 12805, 6845, 337, 820, 8558, -4332, 854, 5897, 175, 591, 13302, -413, 2354, 12192, 6119, 13167, 12977, 7742, 8672, 13708, -1801, 1276, 3030, 691, -2451, 2629, 15351, -3286, 2539, -19, 5528, 911, -4427, -199, -3077, -2, -1677, 4392, -8330, 11259, -4823, 1845, 14062, -4953, 4880, -5203, -101, -11835, 6388, 17105, -3485, 17419, -7561, 17081, 13028, 19878, 1635, 14078, 7238, -4759, 13962, 1846, 884, 7693, 2254, 6586, -217, -1652, -1207, -3158, 12449, 5733, -821, -6326, 8242, 10553, 20803, 7297, -2235, -2121, 9151, -10583, 12960, -4922, 4990, 7539, 10355, 6653, -4125, 9016, -1375, 10235, -5157, 15823, 11518, 18481, -502, 14926, -2138, -3559, 5157, 9157, -1652, 12359, -2724, 18205, 10351, -18374, 5941, -4171, -2783, 8801, 14931, -5140, 11189, 11888, 6021, 6759, 12524, 8326, 14413, 3493, 5819, -6125, 13670, 392, 9921, 2295, 9343, 5577, 3733, 9940, 8610, 18286, 2197, 10648, -3191, 10925, 6963, 10213, 19486, 12573, -2440, 15622, 15289, -9099, 89, 7107, -9554, 4330, 10797, 11836, 22607, -4879, 9424, -4417, -10933, 14160, 25610, 15618, -8471, 13258, 10973, -9503, 19171, -10221, 3204, 4326, 11705, 19650, 9277, -14948, 4465, -8294, 4445, 2781, 9395, 2866, 5996, 14307, 5566, 23090, 6902, 8634, 14250, 20255, 17730, -7495, 11402, 1260, -6876, 5969, 15018, 11296, 8098, 14630, 20667, 4312, 3982, 13524, 11063, -16501, 5179, -7704, 15924, -10643, 8824, 3169, 17727, -679, 16923, 10602, -12097, 5937, 21286, 23448, 32767, 5114, 5909, -10181, -173, -7512, 11122, 18867, -13100, 12674, 12455, -9921, 15529, -1977, -11048, -4658, 18704, 6163, 2110, -3077, 21920, -3449, 11048, -6995, 23982, 12726, 23673, 19765, 24385, 13739, 23854, 8512, 14042, 27225, 10177, 4737, -1870, 17775, 4917, 8026, 11545, 11672, 8748, 12714, 10602, 26423, 16297, 29024, 10295, -21906, 8561, -2294, 29407, -7117, 15814, 20835, 16202, -10299, 11314, -940, 11, 11979, 17312, 16079, 28595, -7466, 7972, -11723, 13728, -742, 6209, 3653, 5541, 7057, 12942, -8784, 20683, -4429, -8018, -9118, 19409, 12826, -8297, -20972, 15148, 7574, 15366, -6751, 3667, 935, 18051, 3386, -1862, 2448, 22518, 23388, 6238, 3318, 4727, 2911, -1296, 9988, -2557, 10216, 16166, 15003, 22817, 13797, 9365, 21935, -129, 1104, 5465, -11994, -3188, 1120, 19988, 3807, 8539, -16, 2250, -4937, 17034, -1598, 4789, 52, 15145, 8570, 5940, 3273, -602, 4746, 7534, 8611, 11939, 3497, -2121, 4434, 7018, -336, 16606, -10306, -9300, -8405, 7582, 1399, 578, -4852, 13154, -3456, 13967, 8128, 15083, 12699, 17208, 5463, 7343, 9561, 10374, 20051, 20463, 6017, 3467, 11015, 771, 14584, -8211, 15738, 20209, 6755, 7616, 8075, 19712, 76, 13202, 5655, -4361, -13851, -4910, 8793, 16008, 9913, -2620, 5753, 10493, -2750, 20547, 8830, -5332, 1431, 5050, 2456, 4037, 4235, 9765, 11873, 11497, 19515, 19944, 17298, 7493, 3691, 1056, 6703, 10728, 8671, 10961, -6203, 19004, 6006, 14287, 1233, -958, 13084, 16347, 6554, -1532, 6415, 12777, 18258, 586, 2429, 2222, 14047, 21526, 500, -1399, -8741, -2002, 3109, 13048, 876, 11600, 14087, 12809, -4689, 18381, 680, -4372, 15807, -3494, -3963, -2665, 7396, 8845, -3863, 4620, 11787, 17351, 760, 17400, -3185, 713, 5807, -1156, 6646, -5863, 249, 13930, 5630, 11741, -1070, 14465, -3849, -8884, -3337, 6687, -1938, 9323, -1688, 7007, 1939, -7477, 647, 13434, 1381, 11683, 19210, 1042, 6380, -257, 13024, 7684, -1640, -6770, 11670, 11496, 14056, 5703, -2441, 18168, -10802, 16385, 1312, 9055, 2795, 2421, -308, 6592, 5426, 7495, -3528, 330, -3021, -5863, 3456, -1769, 2255, 3634, -454, 8199, 11175, 8194, -9762, -468, 5238, 2408, -6188, 1516, 2446, -10450, -2771, -4039, -2061, 15949, 15900, -8883, 8456, 5253, 16930, 7833, -8312, -2975, -11325, 8212, 14579, -233, 14826, 10717, 4364, 14345, 13111, 14250, 2193, 10689, -2179, 8443, 9298, 9541, 13738, -7633, 17752, -3327, 12362, -4106, 10853, 818, -8031, 6632, 14292, 3177, 15909, -2212, -3720, -461, 662, -4643, 5872, 10822, 8307, 10402, -138, 17363, -1513, -1664, 12869, -3091, 11308, 1393, 8739, -8779, -6293, 1894, 3327, -7524, 5053, -13188, 6145, -5512, 18389, -10482, 8201, 8507, 8107, -5368, 3836, 4739, -419, 6060, 6787, 729, -8455, 9322, 11411, -5961, 5174, 10997, 6052, 11524, -9803, -6016, -7740, 5904, -10579, -4127, 6384, 12939, -5873, -4060, -629, 3116, -14720, -2377, 7556, 6364, 10517, 14464, 1595, -3936, 7331, -5398, 13888, 4598, 6524, 6849, 5113, -1207, 4368, 7323, 7868, 10823, 6199, -1962, 1344, -164, -9738, -6732, -6167, 10716, -5309, -1252, -6275, -7004, 1188, -6693, -3212, -3055, 577, -4891, -6670, -994, -11750, -8875, 10266, -7710, -7327, -7316, 17351, 9671, 11866, -8362, 4675, 4205, -3779, 10265, 1410, -7099, -10388, -12850, 11035, 9090, 10883, 3677, 2787, 2351, -1116, 8008, -2150, 6418, 18114, -6004, -2008, 10471, 6277, -4544, -6486, -1015, 12804, -6581, -516, 4879, -6624, -4453, -45, -446, -5081, -5873, 4616, -10473, -5000, -195, 8099, 3882, -10269, 3453, -5500, 4943, 4340, 192, -10347, 3768, 1307, -6668, 5037, -1406, -6984, 4839, 16270, 6806, 1807, 10731, 16526, -6621, -11821, 640, -2507, 12777, 1461, -9835, -8071, 6520, -7497, -7673, 6615, 15101, 10452, 10472, 3517, -16367, 12870, 1738, 629, -2154, 5344, 12063, 7257, 8646, -5792, 5482, 2836, 6908, -6370, 8741, -4458, -6743, 872, -7010, -12856, 6184, -2592, -2837, -9501, 2159, 13561, 10056, 11745, 12539, 2311, 4994, -7190, 17721, -9555, 1187, -6845, 11749, -6620, 3627, 1863, 1827, 10524, -2984, -1132, 10999, 5614, 8196, 6252, -5239, -5700, -8846, 7452, 10243, 12329, -2638, 14604, -10429, -11301, 2612, 11927, -2715, 450, 7077, 10349, 9786, 8222, -4419, -8139, 12307, 4562, -4996, -9751, -934, 3389, -4307, 12075, -4099, 8253, -3562, 11510, 6959, 4126, -105, -1609, 10201, -5997, 9799, 10361, -5225, 25228, 10298, 21304, 5518, 455, 8109, 1637, 32663, 7925, 10459, 21297, -8010, 9797, 24631, -5881, 15730, 10409, 5423, -18178, 18013, 3091, 8106, -2559, 22555, 934, 1839, -6920, 27424, 9651, 10278, 7068, 14962, 6247, 23943, 15758, 15920, 12000, 4274, 7725, -1975, 32469, -5952, 5113, 12181, 4965, 7931, 17122, 6778, 26258, 22274, 25330, 14603, -18502, 14663, -3256, 13334, 13868, 6930, 6930, 18216, -9497, 24416, 10651, -8856, 13139, 5933, 12339, -9375, 7796, 2065, 2295, 15558, 3076, 13763, 9419, -726, 15166, 11756, 3225, 18420, 4521, -1876, 1119, -1593, 12224, 7833, -14554, 8823, 12453, 7057, -10780, 22025, 3390, 5027, 7548, 13515, -3490, -1615, 15719, 18229, 10871, 8363, 10679, -2699, 4391, -1418, 2736, 17463, 5270, 20854, 10056, 8869, 17911, 2849, 2676, 2517, -4901, 2500, 12339, -776, 11190, -4785, 16328, 12232, 2436, 5701, 3887, -7704, 10412, 17591, -4872, 380, 7118, 10196, -4840, 3890, 11441, 16004, -656, 8414, 13301, 4503, 6364, -1695, 2264, -11688, 14519, 1117, -307, 11346, -1314, -1926, 622, 7483, -7670, -2974, 3204, -221, -179, 2301, 1314, -907, 2102, 4911, 11528, -2643, -1424, -3093, 9778, 4254, 6449, 16016, -5146, 10110, 9742, 7258, 4484, 3182, 17342, 1689, -8825, -4910, 8597, 7132, -800, 1644, 8390, 8784, -2715, 5367, -4974, 1369, 12165, 18359, 5420, -4549, 9216, 9953, 3978, 4712, -744, -4971, -3491, 14197, 14479, 9854, -4181, 6280, 1658, -12928, 8891, -996, 16246, 14022, -149, -12, 13238, -10033, 356, 7874, 7630, 2628, 1168, -3856, -4650, 7668, 11970, 3333, 2172, 10535, -348, 10342, 2324, -10305, -7187, 2849, -1733, -243, 9140, 11820, 8347, -7651, -17, -94, 15440, 13883, -123, 5859, 14740, 2734, -8521, 12986, 11854, -3160, -9590, -11352, -7347, -2229, -5362, 2562, 8261, -7383, -7671, 2107, -1465, -2755, -1930, 13351, -10359, -6769, -1834, 7686, 2844, -8355, 602, -1228, 4559, 6669, -1197, 1039, -2248, 11628, 2005, 1931, -4752, 4257, 4034, 408, -1300, -1935, -2339, 1405, 4203, -3471, -4587, -8395, -5806, 9692, 5505, -8316, -9137, 13945, 2408, -7776, 779, 9599, -1537, 5910, 15634, -4652, -1215, 14088, -7108, 4717, -1002, -5608, -945, -8747, 7545, 1417, -4323, -5731, -5398, 4412, 11240, -4424, -10132, 8881, -927, -5097, 5645, 13415, 2960, 6258, 2739, 6517, -13006, -7906, 20021, -9470, -8086, -2351, 14820, -2348, -6492, -10530, -5383, -9475, -12393, -2339, 5076, -17812, -2064, -12888, 1559, -9685, -11107, 6036, -3783, -7588, 4090, -278, 935, -8217, -3312, -7898, 8235, 4170, -6948, -11323, -4606, -15658, 6320, 11403, 15668, -7722, 3003, -3245, 3321, -819, 4137, -2719, -293, 6934, 5099, -9193, -7865, -17632, 8496, -6612, 726, -1252, 12345, 4263, 1830, 21141, -9284, 7895, 3381, -11100, -2304, -7564, 21660, 5363, 2030, 66, 2378, -17636, -4701, -12293, -7316, -10636, 5658, 289, -12639, 7083, 6556, -4564, -4111, 2382, -7399, -2056, -9787, -3650, -3138, -3787, -11187, -11277, 5156, 4155, 6017, -9794, 8457, -5027, -12867, -16316, 13104, 2566, -2809, -15149, 11931, 8961, 4137, -14142, 17221, -145, 9305, -10815, 3505, 3256, -90, -12423, 6996, 10893, -3231, 6664, 13822, -11156, 5179, 3576, 14580, 13457, -7976, 308, -745, 2497, 6157, -3147, 5405, 13265, 11893, -7276, 8590, -12077, -3240, 1684, -4334, -10427, -4701, 5087, 7632, -4785, -3430, 2509, 8523, -8659, 1004, 9589, 2933, -14445, 9075, -452, -16552, 4288, -6826, 3329, -2772, 8719, 2110, -4685, 12678, 3645, -6171, -2747, 1470, 1734, 3827, 5713, 663, 4121, 9369, 1054, 8354, 20502, 10081, 1324, -12209, 5743, -564, 27789, 18815, 6825, 12687, 5088, 6134, 12209, 263, 9291, -3884, 11335, 796, 1627, 2266, -1359, 6303, 8921, -6456, 9997, -4937, 18898, 3459, 16145, 8804, -1007, 15983, 16195, 18126, 6522, 17877, 14679, -7174, 600, 23987, -15310, 20462, 12768, 2670, 18078, 2239, 11249, 1501, 13508, 19216, 2949, -4018, -1279, -1142, 14175, -3017, -5783, 8752, 14698, -17509, 1921, 1039, 8717, 6390, 19948, -1998, -4943, 6389, 21180, -5282, 6792, -837, 15612, 7535, 3977, 4193, 3586, -8475, 12423, -9083, -10387, 1619, 18964, 16738, -6985, 805, -1440, 10510, 18146, 2997, 3113, 19576, -1475, -1489, -1665, 11316, 17023, 862, 570, 9994, 9650, 9525, 20593, 22072, 3687, 21191, 10631, -7058, 19351, 17700, 4767, 22091, 15185, 3093, 9672, -7624, -2453, 178, 4976, -5735, 621, -2267, 8618, -8965, 20858, 8092, 161, -3922, -1044, 4631, 8559, 12215, 14558, 7712, -4718, -19571, 4112, 269, 96, 953, 10323, 10946, -207, -1079, -5068, 7961, 324, 850, 9102, 13666, 7559, 5228, 5265, 1746, -14385, -3502, 6685, 12267, 8001, 8302, 605, -4364, -11484, -7867, 2539, -9123, 6388, 9591, -15245, -2534, -6641, -2461, 8553, -3947, -2895, 4778, -3345, 1062, -10617, 15973, -399, -1309, -11547, 10265, 8017, -904, -9030, 9206, 9676, -2385, 6271, -1834, 5289, -5344, -10874, -6968, -6056, -4187, 729, -6987, 14335, 9008, 3116, 9825, -8442, -9789, -6260, -11291, -6245, 1051, 7110, 13157, 8390, 2767, 6715, 2386, -12305, 3752, -3970, -2841, 3155, -1319, -11639, -4585, 7874, 10890, 240, -2903, -3271, -4531, 7917, 9394, 6532, -10710, 5114, -9241, 6559, 10378, -4586, -2659, 2390, 1876, -13554, 6390, 2854, 1044, 70, 10814, 10206, 10156, -8388, 15132, 2816, 9000, -2933, -241, 9496, -935, 841, -1365, 5082, -1360, 259, -16300, -7767, 4765, 6908, 2243, -9623, -11756, 10120, -12672, 3719, 3103, -5777, 11071, 4614, -1452, -5641, -5633, -4948, 1917, 214, 3759, -4782, 5908, 905, -4100, -11053, -3380, 8934, 5930, -543, -10180, -3202, -3308, 17615, -3531, 1171, 73, 1668, -547, 6912, 8577, -18597, 10645, -6466, 19141, 4203, 15113, 7286, 6514, -1308, -12231, -3343, 14869, 5007, 8936, 4010, 21004, 10200, 3592, 594, 14219, 542, 6326, 6176, 1312, 8224, 13531, 3857, -274, -1267, 708, 16797, -6301, 5335, -11977, 11011, 17133, 7913, -23757, 3982, 2193, 17426, -8459, 5867, 14947, 17722, 1790, 12456, 15455, -2559, 21809, 17532, 6217, 3305, 3176, 7549, 9678, -796, 12770, 10485, 15297, 9144, 17800, 342, 7673, -6011, 14314, 14996, -4196, 17944, 3349, 12754, 10454, 16522, -1319, 16107, -4255, 5129, 14793, 5158, 8996, 4995, 2147, -14062, 13080, 5043, -12749, -8443, 9809, 483, -2076, 4629, 5862, 9787, -9464, 8425, 2403, -2716, 5473, -1579, -13075, 645, 15942, 1350, -4391, 8240, -12730, -5632, -5013, -9296, -10270, 9562, 5457, -7404, -3964, -1027, -1670, 8971, -5761, 8123, -14680, -4075, 7016, -8005, -2927, 11053, 7322, 445, -10898, -9567, 4182, -6187, 574, -5910, -2514, 4420, -1102, -3808, -8433, 8585, 1175, 729, 741, -3868, -8859, -7117, -5038, 3983, -460, -8411, 10860, 2100, -15375, 1608, -3375, 13081, 9194, 10579, -11136, 3140, -2531, 10532, 2194, -15669, -3827, 5215, 18355, -5866, 530, 7597, -11368, -3794, 2654, 6538, 4093, -17133, -3521, -8910, -4238, -9314, -13187, 27, -5002, -4172, -14677, -18831, 6362, 8458, -1510, 2896, -14343, -14773, -10644, -8600, -16762, -10698, -2360, -4720, -2676, -9289, -14700, -7672, 1691, 1109, 12245, -15946, 4155, -11829, 2640, 2301, 2354, -9217, -736, -14916, 7079, -1833, -1799, 4065, -2493, 11222, 645, -4536, -8788, -8717, 6028, -4171, -2091, -9499, -11969, 3627, 5323, -5265, -2069, 10154, 7665, 2815, -4696, 3121, -4218, 1763, -14693, 14616, -9428, 8426, -9092, -3867, 9723, 4838, -14655, -15396, 2556, 6379, -16768, -32, -12813, -14008, -4317, 10366, 5673, 1490, -2160, -17382, -103, -4839, -12954, -3407, 6529, -11414, -4164, -3352, 1373, 6441, 3376, 12203, 19237, 4512, -12218, 13848, 465, 5203, 14064, 950, 1019, 7359, 2898, 491, -7142, 3592, -5964, 5313, 4836, 1632, 1139, 7745, 9089, -4356, -14344, -927, -9945, 16945, 8739, 6192, 972, 11297, 365, 786, 13544, 6616, -2444, -4334, -9519, -1812, -1340, 3933, 1290, 8955, 14755, 14290, -10797, 17014, 13414, 10909, 12034, 8078, -11793, 10192, -9495, 2767, 6174, -5076, 3335, 820, -12823, 15748, 1689, -7654, -8939, -9203, 3743, -9402, 5227, -8382, 3425, -2316, -16963, 12845, -11535, 6639, -7956, -7032, 7360, -8809, 1675, -11815, 9827, 7885, -4355, 7955, 10943, 5463, 6371, 1129, 6266, 1414, -956, 1800, -10699, 7857, -9911, 9953, 7987, -11453, 7983, 1247, -7256, -1061, -2472, -8391, 2846, -3296, -2023, -3817, -1983, -11233, -9901, -705, -10063, 7101, -753, -13901, 3167, 1811, -6323, 359, -12455, 3884, 6089, 5590, -5824, -1353, -11381, -13020, -2184, 10770, 1969, 3435, 5468, -6456, -2589, -1858, -8962, 21513, 763, 2989, -8831, -17020, -3811, -10372, -708, -13409, -18116, -5681, 20684, 2874, 8247, -2994, 619, -14149, -15102, -3473, 3383, 6561, -6473, -53, 8, -17190, -7163, -3002, 2952, -13765, -5408, 4052, -8665, -13485, -1702, -12661, 1515, -5167, -10912, 7659, -21349, 1330, 21583, 8487, 10505, -17357, -1978, 7637, 349, 2208, 17359, -10903, -9005, -4300, -3628, 16087, -3325, 12651, 4612, 389, -9770, -313, 18511, 3024, 7077, -288, 11865, 15169, 5281, 11080, -5605, -4833, -1921, 343, 11017, 3063, -7787, 16694, 3438, -6214, -2581, 894, -8284, 1627, -1330, 16183, -6347, 16037, -260, -7615, 16185, 444, -5406, 1901, 1791, 8438, 10867, -2974, 1023, 10615, 6097, 5666, 13482, -763, -4155, 9792, -3900, -3147, -2093, 998, 1978, -5140, -922, 13207, 4819, 14164, 7914, -7430, 19713, 3882, 18924, -2749, -4169, 251, -810, 914, 5392, 15959, 14099, -2139, 2680, 14887, 7265, 13747, 4877, -4186, -12670, 14503, 19569, 1840, -23174, 17064, -125, 18422, -1526, 13015, 7638, 10604, 7942, -1841, 15736, 14256, 26868, 14162, 21478, 8370, -7112, 5902, 5222, 4179, 25038, 12727, -14482, 17047, 4439, 5230, 7471, 25269, 14543, 6039, -12640, 3552, 9376, 7791, -7259, -3991, 19425, 16773, -3682, 12026, 6257, 3530, -11511, 14151, 2275, 6677, 7931, 931, -7423, -8280, -14542, 14558, 6892, 12733, -6296, -7690, -9068, -8448, 10066, 2629, 10865, 8165, -4099, -3802, -3202, 9232, 9611, 8924, 407, -4808, 3230, 4631, -10963, 2589, 9630, -5102, -7143, -6594, -10772, -6797, 4221, -47, -7247, 5950, -10690, -3671, -212, 877, -6415, -349, 9170, 2948, 7124, 1374, 10181, 9583, 2678, 1325, -4677, 7967, -9039, -3163, 16066, -4514, 5363, -9069, 1154, 1727, 2771, 13900, 322, -14476, 52, -18624, -7997, -2845, -2793, 1995, -11982, 1142, 6879, 664, -1958, -8002, 8933, -869, -6542, -138, 17538, 5505, -8608, 256, -9972, -6146, 2432, -5215, -6756, -1888, -727, 640, -10445, -5440, -7782, 3755, -7923, 11345, -11232, -9751, 8557, -1968, -1208, -9060, -3443, 7297, -9288, -6626, -4284, 3155, 15013, -15356, -4052, 9018, -11675, -15824, -8769, -4878, 9756, -416, -1291, 11636, -14899, 10932, 5712, -6109, -14524, -8207, -7784, 3540, -9509, 3374, -8468, 1525, -7221, -14115, -847, 10596, -10414, -4991, 11016, -9053, 5350, -2808, 4094, 12721, -1481, -11780, 477, 5736, -14111, -8428, -13493, -2155, -8810, -11585, -9234, 3061, -7108, 5461, -8582, 4406, 2630, -7631, -3617, -10884, 3370, 4925, 1206, 5160, 8167, -8326, 10336, 2697, 13204, -9747, 8992, -10261, -12445, -8697, -9711, 3627, 4994, 4119, -2551, 4167, -175, 11614, 11127, -5444, -10611, -4528, 4969, -2120, 22102, 2292, -7707, 3733, 5255, 10964, 9432, 9400, 10444, 3823, -16947, 19632, 11659, 7436, -15095, 1215, -730, -3588, -5504, 4432, 7166, -2998, 12462, 19782, -4126, -9241, 9309, 16400, 14918, 5294, 144, -3783, 18067, -11121, 10130, -466, 15818, 11997, 12861, 13763, 10286, 4806, 25960, 13026, 558, -8403, -5147, 17469, -4941, 12934, 3296, 13847, 3447, 3661, 10074, -9344, -3121, 9569, -335, -3488, 7602, -13157, -496, -14245, 14166, 9567, 4447, 5474, 1042, -4564, -6120, 4679, -3032, -9802, -6504, -3368, -6695, -6116, -3104, 241, 6828, 7644, -8522, -4195, -11469, -5786, -2513, 11531, 3608, -3040, 6362, 12366, 10026, -12306, -1362, -13222, -3860, -5927, 2729, -4972, 23218, 12558, -7727, 182, 4807, 3161, -2106, -576, -210, -11532, 7887, 11357, 4116, 2945, -544, 649, 4933, 8739, -5105, 9696, 3006, 534, -7895, 2497, 5767, -14799, 6675, -21756, 2810, -5325, -5861, 21088, -6429, -16217, -9273, -1883, 9757, -9475, 13654, -11001, -3942, -10941, 9180, 1418, 11980, -6670, -1985, 4369, -144, -4905, -15804, -5800, -6159, 2697, -139, 8446, 9368, -3220, -2100, -12794, -13764, 249, -14474, 4998, 13495, -12097, -4622, -13050, -14511, 7768, 4144, 4031, 19442, 116, 7232, -11539, 11551, -14091, -7220, 214, 7126, 4573, -14255, -2326, -769, 11497, 5455, -6477, 4312, -11255, 6375, 3939, 790, -11166, 4237, 247, 1823, 4888, -6345, 5398, 6982, 2308, 10467, 8702, -8083, 10510, -1388, -16999, 14699, 7347, 7257, -1173, -2215, -598, 6492, -6448, 4046, 9010, 1819, 9410, -66, 2931, -11002, -802, -109, 9873, -2696, 7517, -12026, 6043, 4759, 2869, 3866, 12399, -13966, 3836, 11505, 13191, -1563, -7624, -2058, 4107, -1410, -3260, 17585, 8394, 9539, -2602, 10635, 11127, -10926, 9686, 1237, 4236, -2603, -7668, -9721, 9797, -1551, -667, 1576, 14870, -9943, 1409, 9635, -1934, 5669, 4998, 5895, 5412, 9604, -1783, 11744, -7520, -8290, 18130, 14096, -7193, 11161, -9766, 3565, 6421, 4821, -2661, 8859, -147, 3101, 7448, 4867, 8856, 13642, 17410, 8415, -324, -1552, 4778, 3388, 8514, -6071, -1960, 14810, 12457, -2967, -471, -7369, 13250, 15361, 14318, -2136, -6976, -9465, -4725, -1761, -524, 10672, 3824, 3248, -7055, -6478, -10525, -3425, 14987, 231, 9624, -2330, -12224, -8192, -7542, -9633, 459, 9575, 4651, 11306, -4136, 10733, 7284, -2293, 1207, 2781, -1869, 4914, -4339, -12570, 3022, 7596, 2286, -3807, 9005, -3731, 11952, 5830, -12843, 10671, -9780, 13101, 9913, 2104, 3553, -853, 5582, 1845, -1416, -2895, 9064, 299, -3172, -9859, 6994, -1603, -3849, 5028, 9744, 7236, -2421, 6361, -2942, -5691, 5857, 1856, 15883, 4078, 2715, -5974, -15849, 19190, -2912, -3265, -1655, 12818, 3827, 4924, 10680, 29, -5310, 12713, 12519, -3928, 7952, 16083, 4404, -9500, -6503, 7058, 5520, 12173, -850, 4671, 7437, 839, -4250, 3186, -343, 3528, -9795, 6854, -8168, -6903, -19213, 81, 3234, -4569, -1221, -8674, 4180, 192, -5286, 7942, -2062, 5829, 5911, 176, 4148, 1297, 5589, 4072, 11270, -2986, 2206, -6686, -2275, -2081, 5926, -7403, 15886, 7418, -7569, 7224, -11234, 2686, -1578, 9698, -6684, -4704, 3, -7390, -5246, 1948, 1281, -9781, 10490, -456, 9550, 10555, 20797, 4108, 3002, -10332, -1976, -3956, 869, -4497, 8434, 2337, 6638, 9078, 19624, 5927, 43, 6693, -127, -9483, 3113, 577, 6988, 9924, -1751, 10983, 12004, 3125, 4681, 17593, -2825, 6752, 1703, -8352, 10649, 13621, -6978, 10432, 8673, 3379, 2289, -4686, -8288, 15963, 6449, -1726, -2389, -9160, 4603, -11350, 14436, 8123, 2304, 11534, -5531, -3237, 20065, -1925, 8046, -243, 10231, -9644, 22442, 7751, 12238, 325, 7693, -7282, 16223, -479, 21030, 2495, -3692, 19341, -5018, 16513, 19795, 4396, 8064, -948, 10785, 9417, 3699, 20840, 5215, 17687, 22629, 7899, 6464, 13629, 5843, 21392, 18059, 13085, 1609, 3373, 5645, 4449, 9270, 100, -2500, 17619, 20520, -12426, 12663, 987, 8116, -6231, 4982, 4877, -10677, -10209, -6564, -2933, -4047, 7318, -12633, 7242, -4799, -3209, -7506, 131, 8014, 2689, 5085, 8241, -1619, -4917, -6579, 11842, 4767, 4209, 15766, 2311, 9294, 4240, -5051, -8565, 1406, 7474, 5396, 9685, -2267, 9447, -4832, -8562, 2937, -3261, -12762, 12675, 4341, -4240, 7956, 6098, 781, 6492, -407, 925, 7809, 2913, -4862, 11203, 15504, -79, -2286, 11441, 12780, -3500, 7095, 3872, 6889, 10854, -7144, -593, 3759, 2553, 3389, -7334, 13067, 13204, -10964, -7106, 6710, -5838, -11343, 9644, 11894, -3812, -10233, -63, -6387, 3958, 4780, 16842, 7067, -10375, -7441, -7120, -6606, -9267, -5311, -13544, -5540, 5493, -8182, -1496, -7169, -1901, -7309, -989, 7371, -11324, 3882, -4395, -7206, -5409, -5247, -439, -4361, 8129, -4200, 10602, 11844, 2415, 9228, 6327, 673, -570, -7814, 1839, 2874, 6844, 9207, 5981, -3604, -9836, -8240, -4635, -2252, 9473, -8504, 9478, 7595, 640, -9607, 7534, 6402, -6110, -1390, -624, -7284, 4141, -8266, 2016, -1085, -5942, 10248, 13381, -14857, -1874, 6151, -9806, -93, 2417, 7438, -6779, -5027, -475, -2240, -10768, -8907, 2779, 2551, 3279, -7494, 4479, -807, -6172, -10617, -6378, -5697, -1556, -13134, -7921, -16177, 4356, -1677, 20237, 1006, 5714, 8919, 8315, -10136, -8523, -9856, 17753, -13086, -1681, 10107, 7622, -4404, 9731, -5212, 9184, 10269, -306, 10324, -2669, -1527, 6391, 6456, -9199, -7350, 6652, 2346, 7684, -5996, 7991, -1361, -6185, 14246, 8870, -11696, 7917, 8151, 1740, -6720, -128, -8293, -13519, -11863, -4908, -12733, -3982, 2341, -4353, 10565, -193, -4905, 382, -8972, 10104, 12197, 9117, 1339, 11086, 10123, 76, 11953, -4548, -14362, 18599, 6144, 733, -2209, 3355, -1390, -694, -5887, 2732, 7134, 1491, 403, 3735, 7121, -2778, 12931, 6724, 9280, 775, -604, -5647, 5709, 12784, 8861, 12125, 10086, -4490, 13415, -12626, 7571, 6942, 1945, 15676, 5002, -582, -1569, 1358, -7048, 1599, -1881, 6694, -8243, 6258, 10926, -3280, -10998, 1175, -840, -621, -526, -8068, -9914, 9188, -728, -1692, 11993, 7173, 3515, 7623, 6684, 8648, -1042, 4031, 6295, -298, -6336, 6634, -7438, -7853, 931, -11101, 5921, -2489, 8037, 7578, 195, 7224, 17310, 3521, -3276, 11766, -2866, -7336, 5760, 2904, 5770, 5585, 7106, 11559, -10176, -2698, 4067, 9339, 6069, 3944, -2728, 17422, -2391, -5154, -5838, 9906, 15025, -3062, 3568, 2530, 5367, -10923, 8747, 5306, -650, -6018, 12522, 17534, -2458, -6663, -7927, 17502, -4146, -5836, 1079, 2614, 2172, -3280, -6590, 3915, 15054, 10275, 11784, 12872, 13631, 1023, -4696, 16351, 2903, -1521, 13879, -4510, 3461, 1525, -1380, 5194, 10400, 3284, -4252, 11955, 7361, -10558, 7461, 17862, -4006, 16762, -2797, 1080, 15340, -2062, -3616, -11187, 7602, 11424, -1963, 5804, 13078, -1779, -1939, -185, 8077, -11299, -4961, -4033, 5738, -1228, -4922, 11709, -4292, 18510, 15844, 11300, 4132, 4778, 2901, 11053, 7303, 14739, 15951, -7219, -6541, 7542, 12669, -4328, 14310, 12502, -7078, 4211, -4823, 10993, 17710, -3317, 16142, 9682, 2151, 12117, 779, 8627, -2201, 10853, 9654, 1393, 5519, 6067, 4524, 5723, 11396, 18664, 3636, 10175, 4922, 15646, 9264, 2396, 11795, -9342, 5615, -862, 7023, -1733, -784, -6529, 13877, -7878, 5955, -12311, 6100, 11599, 12437, 7259, 16254, -1905, -2137, 7419, 7241, 14716, 16663, 1468, 14733, -2858, 3683, 608, 17746, 2585, 20137, 7523, 7475, 10413, -2790, 7197, 5785, 4817, 18357, -3198, 15704, -56, 12340, 5932, 2731, -8843, 6030, 1432, 3987, 749, 21574, 16727, -16743, -251, 4788, -10983, 21885, 19101, 15544, 9870, -2007, 9214, 4416, 5282, 11648, 4846, 6655, 2689, 949, 16817, 18362, -3993, -2167, 7315, 5012, 1007, 12533, 15219, 3098, 476, -1368, 7515, 16975, 7088, 5979, 529, 10363, 9379, 13989, 6567, -7768, 16683, 3097, -124, 13688, 5237, -1985, -2206, 10526, 22738, -574, 1420, 4335, 11711, 20223, -2744, 15844, 3565, -1058, -2330, 7187, 5379, -10009, 2901, 22532, 12706, -260, 2244, 587, -5766, 17819, 8669, 18671, 4425, 5508, 25479, 20348, 2707, 14941, 3316, 1710, 2252, 21611, 24760, 5887, 2762, 3350, -5551, 7660, 6091, 22739, 7147, 2448, 23281, -3578, 15308, 16647, 11834, 4099, 22058, 1890, 1261, 29741, 16205, 6653, 4849, 13902, -1459, 18131, 13806, 3766, 4301, 24004, 7984, -6466, -8899, 6377, 12982, 12358, 7896, 1377, 12354, 4461, -1775, 22161, 9395, -3376, 3750, 10290, 18118, -7609, -6797, 8349, -5799, 24411, -2379, 16243, -2404, -13290, 24851, 5417, -9622, 17652, 5160, 924, 10064, 10017, 5071, 14435, -8706, 40, 3831, 9909, -13201, 19562, 19721, 10807, 22413, -3794, 11360, 15560, 19066, 5029, 5077, -3401, -10375, 21407, 715, -6830, 23704, 20595, -5571, 2940, 592, 14017, 15041, 14689, -1072, -5020, 6900, 7991, 6263, 16341, 13969, 1351, 11914, 10266, -3208, 19173, -5401, -12079, 5899, 16222, -5068, -1271, 9044, 2939, -8399, 14599, -4059, 14601, 18666, 11705, 19828, 25493, 936, 21365, 3189, 5818, -2922, -3345, 832, 14444, -6138, 4672, -1757, 16026, 5892, 9390, 6269, 8045, 5409, 2922, 17590, 16691, 21778, -4259, 4166, 1944, -9647, 11759, 8742, -3812, 15083, 4106, -6630, 19673, 903, -311, 5195, 23943, 7883, -9105, -11324, 1271, 3654, 16036, 13967, 15542, 15430, 951, 9379, 18477, 7945, 9951, 17332, 3147, 12367, 10235, -2632, 9059, 4444, -320, 1560, 18232, -1117, -5496, 7484, 13135, 10091, 17962, 584, 10519, -6192, 15047, 21284, 10973, -15149, 14327, 5523, 19542, -6773, -7242, 578, 14780, 9278, 10126, 7888, 3409, 996, 12891, 580, 18114, -8963, -1457, 5463, -4928, 10869, -2132, -3965, 21745, 10825, 8019, 467, 18242, 5959, 5746, -11189, 6029, 15242, 16705, 10860, 10521, 14016, 20824, -9175, 15380, 2731, -6401, 3002, 12425, 12362, 10381, 6256, -3307, 10874, 7731, 3676, 19270, 8353, -2826, 14749, 7774, 761, 19745, -7024, 8715, -5707, 14341, 4812, 2649, -13130, 12250, 3430, 3395, 8062, 8017, 21470, 3371, 12303, 4068, 19432, 9459, 18998, 4616, 14074, 7277, 8016, -1810, 7791, 268, 23503, 6494, 1508, 15118, -3604, 5270, 9612, 21822, 13795, 8178, -8775, 10382, 15483, 17387, 5772, 19334, 50, 13460, -3108, 19618, 6077, -3713, -4435, -3848, 6882, 14758, 2376, 7863, 10932, 2407, 8678, -5058, -2781, 764, 7962, 6628, 4919, -1094, 1342, -8830, -9474, -9010, 14413, -4316, -2907, 12382, 12936, 5665, 8397, -8576, 10149, 5703, -5490, 15001, -5915, 2880, -1325, 414, 6550, 6121, -2724, 13187, 12016, 5738, 1780, 7876, -11180, -2496, 9090, -452, 3781, 2310, -2660, 5926, 945, 3035, 3514, 11434, 10684, -5510, -5837, 9806, 7554, -2075, -8224, 4436
};

static TinyTensor inner_fc1_weight_t = {"inner_fc1_weight_t", inner_fc1_weight_t_data, TINY_DTYPE_INT16, 2, {64, 64, 1, 1}, 4096};

static int32_t inner_fc1_bias_data[64] __attribute__((section(".data"))) = {
    480049632, 57049736, 357721632, -139508448, 546901696, 459639488, -386810144, 212823856, -161551040, -67401976, 328944864, 174454032, -64522216, -25373116, -384566496, 97907528, -429843968, -9410098, 300999552, 338405504, 51215856, 27373194, 426100416, 155971312, 385580608, 139037344, -259172624, 404875040, 285912512, 486514528, 467701568, -136331936, 407854848, 475309312, 104622560, -175098128, 95158048, 240248720, -92968008, 356113984, 348529696, -258990704, 227735136, 346507072, -268436000, -131190320, 387788192, 478999488, 420506112, 71625880, 52434520, 254451280, 417585728, 278151840, 457591328, 430240864, 534813504, 308470848, 97099384, -8299818, 13045965, 286624864, 514691872, -337914496
};

static TinyTensor inner_fc1_bias = {"inner_fc1_bias", inner_fc1_bias_data, TINY_DTYPE_INT32, 2, {1, 64, 1, 1}, 64};

static int32_t inner_fc1_data[64] __attribute__((section(".noinit")));

static TinyTensor inner_fc1 = {"inner_fc1", inner_fc1_data, TINY_DTYPE_INT16, 2, {1, 64, 1, 1}, 64};

static int32_t relu_data[64] __attribute__((section(".noinit")));

static TinyTensor relu = {"relu", relu_data, TINY_DTYPE_INT16, 2, {1, 64, 1, 1}, 64};

static int32_t inner_fc2_weight_t_data[4096] __attribute__((section(".data"))) = {
    2380, 11613, 10844, -5125, 3241, -1965, 5835, 441, 16345, 1042, 16221, 16454, -8046, 13491, 4364, -6284, 15526, 7620, -6955, 5322, 12870, 965, -82, 747, 4974, -11219, 11482, 12140, 10418, 7623, -10582, 5755, 4972, -3753, 13842, 7821, -7600, -6400, 3852, -7699, 9177, 3745, 14679, 2290, -12375, 6101, -5143, -7376, -5461, 655, 7757, -6523, 12123, 5557, -985, 9444, -2003, 1604, 1559, 10003, 4418, -6787, 14405, -5622, -8869, 8574, 5753, 4971, 18962, 3578, 11682, 3540, 6514, 10117, 6508, 9464, -13127, 11692, -3978, 3790, 181, -4101, -9571, 13938, 15778, 4876, 17469, 9701, -1242, 8601, 11480, 16643, 11768, 163, -1008, 2525, -10664, 18908, 17342, -1243, 12965, 17982, 5506, 7405, 10511, 12269, 5744, 12485, -5225, 590, 9649, 3795, 3856, -9915, -3751, 7271, 2436, -8295, 514, 9198, 16753, 92, -1339, 1733, -11713, -5268, -112, -10906, -4025, 7799, 17640, -9071, 16858, -8668, 9904, 2142, 16871, -9838, 18141, 5557, 8563, -3351, 14404, 1330, 7630, 11311, 1663, 6100, -557, 14559, -879, 10327, 13705, 8189, -4477, -452, 2283, -7868, -2205, -4546, 10119, 11516, -4750, -6806, 17581, 4427, 11325, -4636, 9012, 3574, 3871, 10286, -13255, 3789, 830, -1417, 3044, 10712, -887, 1891, 4509, -6641, -14426, 9119, 9438, 9113, 10716, 6090, -12162, 6031, -368, 7594, 7778, 12432, 3022, 4320, 8708, -11393, 12556, 12515, 1547, 197, 8079, -1088, -5641, 1721, 5440, 8830, 10932, -5462, -22557, 7121, 6661, 19563, 20682, 11277, -1546, -8697, 12034, 16597, -8281, -5559, -14864, 16506, -3930, 22548, 2258, 11893, 1954, -595, 21851, 1489, 1053, 16005, 13230, 12105, -19298, 7829, 7135, -5916, 11823, -2951, 14605, -8251, 7612, -9672, -7669, 3944, 8087, -5428, 1058, 21775, 6006, -2644, 14107, 1906, -10427, 4061, -7029, -213, -6070, 6345, 10291, -7098, -2802, 932, 10694, 13175, -10847, 12241, 9235, 10702, -9717, 6757, 18858, -9179, -5785, -12749, 1151, 9711, -4632, -3347, 3832, 3476, 4139, -8349, 13603, -1373, -10245, 5186, 10879, 9566, -807, 6723, -9078, 10643, 6198, 645, 9912, -2386, 12858, -4123, 5674, -4785, -7382, 1329, 1735, 7616, 11293, -7581, 17753, -4970, -11508, -77, 6193, 2563, -766, -5867, -1935, 1373, 2688, 10719, 12111, 3907, 12534, -5217, 2699, 4393, 7205, -4455, 9290, 13356, 8726, -1055, -4813, 5593, -3530, 5343, -471, 4152, 11003, 10223, 6966, 6382, 4651, 5839, 304, -4018, -2012, 167, -3903, 1766, 5943, 9309, 9284, 12894, 13103, 7111, -6485, 8907, -6887, 6002, 11256, -7394, 2378, -1778, 6419, 62, 14274, 3870, 15324, -2494, 13076, 9736, -7680, -5295, 11438, 7022, 12893, 12732, -9606, -11263, -6921, 10884, 12759, -379, -11487, -588, 5052, 11795, -9305, -9488, -4226, -3358, 1345, 617, 2622, -6102, -9259, -10987, 9452, -8550, 16, -8225, 8740, 7742, 1524, -11686, 1440, -9766, -7342, 6735, -11612, -10881, 7080, 11515, -2663, 6932, -4223, 6164, -12165, 8806, 3134, 14192, -7909, -7247, 4179, -1060, -1810, -253, 5839, -11374, 5269, -5199, 7658, 5758, -12880, 220, -3818, 6902, 5512, 5063, 7199, 10624, 11611, -2044, 9936, 3003, -3569, 18959, 18259, -83, 21593, -19744, 18765, 11843, 17310, 8458, 10564, 20513, 7176, 11393, 11550, 24719, 9503, 7146, -912, 25343, 9526, 23079, 6176, 15063, 17914, -156, 18451, 9135, -10372, 109, -14297, 10456, 5342, 18700, 20768, -799, 5053, 18899, 9746, 13130, -4902, 21229, 19721, 13911, -2687, 17423, 23442, 9513, 21078, -10000, 21701, -4648, 11207, -7554, -19397, 10314, 17507, 1143, 11450, 4158, -6611, -9156, 6015, 187, -9863, 10399, 8722, -7805, 18798, -20036, 18078, 11810, 7433, -4107, 8975, 24341, 2685, 21758, 12330, 24198, 5145, 5809, -17189, 9202, 9333, 4263, 19382, 7298, 25359, 2594, 12457, 2549, -7180, 11423, -19685, 6326, -7195, 10767, 11533, 4082, -7543, 17804, 8539, -9, -9747, 10246, 24437, 20051, -24707, 25420, 14618, -6138, 21482, -435, 2750, 1171, 11435, -11590, -19332, 25273, 5604, 542, 5422, 3548, -11816, -5904, 6392, 19661, 765, 12075, 6056, -314, 23924, -23133, 10945, 8907, 24727, 7843, 25067, 11607, -926, 17659, 26679, 15432, 20600, -10101, -10913, 21897, 25499, 3853, 11136, 9732, 5783, 11487, 26847, 22842, 1030, 2516, -13523, 22469, 3885, 4601, 10008, 16115, 11100, 7191, 26526, 17936, 7401, 5751, 11384, 6538, -17111, 27809, 7243, 9230, 21983, -6101, 11520, -10769, 21676, 5930, -4698, 22411, 20650, 2914, 15483, 28054, 5491, 3001, 5003, 4855, 3151, -5857, -5711, 6460, 11158, -748, 13114, -4100, 7500, -8612, 10979, 8926, 2532, 10824, 4138, 985, 8702, -6800, 12420, -3379, 9938, -363, 7621, 11672, 10999, 8555, 12510, -3621, 13492, -2937, 8604, -10313, -4324, 8190, 6539, -1494, 11479, 5190, 12384, -8812, -7797, 5576, 4612, -3209, 924, -7603, 8187, 7535, -2294, -890, -3384, -11871, 6786, 4115, 1130, -5541, 462, 2353, -801, 2014, 10604, 2686, 7644, 2255, -6459, 6459, -2457, 1573, 4495, 22645, 159, -11457, 4544, 5526, -6681, -11896, -3697, 8684, -7866, -1861, 458, -3664, 8814, 4629, -9806, -4269, -6193, -809, 9241, 3290, 8061, -9695, 19935, -2855, 14381, 2813, 7389, -4850, -3311, 7489, 4392, 3937, -4402, 6109, -5464, -9803, 8757, -8380, 13698, -8599, 2066, -11253, -478, -11876, 1947, -4267, -9435, 5641, 15931, -7574, 2627, -7258, -6023, -7124, -4819, -3132, -86, -8131, -2435, 14505, 10153, -2031, 1730, -1594, -2783, 11253, 3811, 6754, -4003, 3046, 7314, 11311, 14483, 7391, 2646, 7179, 8571, 14666, -9325, 8114, -6820, -7845, 6809, -12081, 11190, -7316, 17051, 11499, 13424, 9889, 1063, 8614, 11549, 12925, 6485, -572, 1507, 6362, 8816, 8365, 7795, 11971, 13024, -4452, -7953, -6303, -9098, -9873, 8951, -3514, -1283, -6808, 5730, 5110, 6769, -4654, 10533, -4512, -8640, 7383, -2802, 2427, 5834, -546, 9160, -9123, 4984, 4581, 8472, 10767, -4972, -7787, 13500, 6046, -2504, 16011, 4750, -61, 3290, 1729, -1201, 3787, -1586, 9138, 9812, 17117, 5099, -8705, 9834, 10054, -362, 11231, 1216, -6173, 10175, -981, 3678, 17793, -6487, 7738, 8461, -9105, -501, -297, -774, 15504, -5326, 10901, 16102, -2175, -3529, -1986, 11196, 3274, 1313, -4347, -16404, 11265, 13187, -256, 7535, 3996, -5437, -5418, 7171, -4118, -8827, 10037, -9764, -1104, -4131, 1711, -1397, 2284, -477, 9772, -399, 2637, -2961, -3476, 10385, 11815, -6709, 196, 4987, -9794, -4900, -9124, -956, 2839, 5720, -730, 6356, -5591, -10986, 3155, 2169, 4907, -3401, -11970, 1217, 10012, 4932, -1141, -4682, -9865, 2966, 16, 573, 5233, -2530, -6550, -2242, -8833, -644, 9018, 2572, 9647, -8233, 1228, -7877, 985, 4559, -3463, 2552, 6069, 344, -4671, -506, -5321, -6865, 13109, 9501, -6878, 21152, -3114, 6977, 11864, 2566, 15319, 9120, 19041, -12969, -3704, -2192, 13273, 16018, -3121, 5530, 19473, 6754, 15877, -677, 9662, -270, -2470, 6635, 635, -1757, 10512, -12438, 16965, -5717, 12289, 3474, 4207, 3146, 5577, 14852, -721, 1571, 17915, 4205, 17360, -14267, 10477, 1972, -841, 16916, 1942, 682, 1393, -784, -7872, 3342, -428, 14942, 8490, 9213, 1897, -4243, 2944, 17217, 7343, -2219, 11722, -6268, -542, 6587, -11514, 5060, 15986, 10054, -7597, -8172, -4117, 9505, -1465, 4617, -9550, -2462, 10476, -13523, -10855, 10280, 2685, -5270, -769, 737, 7306, -5466, 1940, -10482, 11669, -3183, -13068, 5776, -907, -6688, -8127, 335, 13492, 9763, 5055, 1465, 3181, -3492, -1389, -9596, 3360, -7652, -7581, 9581, -2297, 10573, -10283, 1648, -4752, 4183, -3575, -9884, 4171, 7074, 1995, -2275, -11682, -4485, -7858, 5106, -7853, 166, 1572, 4819, -11617, -4652, 6496, -1661, -7343, -7585, -1564, -2773, -3055, -2275, 8456, 946, -4240, 10282, 6388, -10232, -3388, -5383, -1814, -2798, 2669, 8496, 643, -2538, -6478, 7449, -1305, 5682, -4193, -6989, 11204, 275, 3348, -5579, 9030, 592, 8005, -452, -6444, -9326, -7479, -4813, 6083, 2685, -10143, 7583, -6900, -580, 10013, 9898, -4402, -5595, 4193, 5463, -354, 1828, -126, 7641, 5295, 1171, -11878, -3284, 9428, -6164, 18561, -1603, 5464, -7354, -8659, 4945, -935, 5671, -16924, 5397, -17225, 5200, -6944, 18762, -1133, -153, 4127, 1707, -11641, -797, -1976, -8183, -11332, 16426, -11067, 11413, -2461, -11852, -12540, -2498, -2489, 10366, 1499, 4295, 9179, -11701, -7582, 3976, 3461, 24413, 3911, -9104, -11911, 3531, -4474, 4777, -8422, -404, -3173, 20308, 2854, -2660, 8094, -8855, -11275, -3542, 2993, -2799, 10894, -13648, 963, 14866, -7868, 7186, -17785, 7833, -186, 13817, 7830, -1158, 13972, -11799, -1576, 21904, 13346, 15307, -6740, -4129, 5755, 4346, -787, -2680, 14696, 16209, 952, 12035, 70, 2018, -5914, -3662, -6051, -11072, 9755, -1241, 9772, -2432, 12089, 3327, 6541, 1454, 12776, -2321, 2980, -4892, 20078, 19893, -4683, 15282, -7951, 10005, 8903, 21655, -9540, 834, 847, -2039, 5523, 17377, 9410, -1645, 9040, -1126, -9820, -11158, 7950, 16341, 8922, 23820, -11467, 23329, 17046, 17064, -5555, 1736, 10255, 10076, 3792, 17288, 12536, 18075, 8564, -16819, 4075, 24004, 7411, 17775, 1435, 5320, -3814, 14441, 21480, -2650, -10637, -20780, 21597, 9254, 3848, 6108, 2835, 6095, 23918, 24164, 18491, 3114, 3866, 21877, 16884, -20534, 20219, 20695, -11478, 18103, -10869, 20449, -2455, 9345, -9450, -9047, 3276, 8586, 4553, 17521, 12576, -5181, -11744, 19696, -5493, -4375, -2904, 14940, -11901, 6160, 12561, 815, 465, 1451, 2034, -3630, -3169, -13142, 6091, -4657, -6101, 11051, -7718, 16212, 11575, 12388, 1728, -8784, 12765, 1893, 143, 1994, 2792, 21054, -2271, 3523, -5876, 1604, 4443, -7864, 8107, -1762, 2790, 12931, 8080, 10771, 1107, 2661, 8744, 7210, 1731, 8805, 4184, 6344, -7423, 4992, 670, 9050, -7108, 12303, 7749, 9214, 4183, -8095, -4108, 5645, 10635, 112, 8680, -52, -15749, -13794, -7933, -1485, 19071, -11790, -4488, -11095, -4040, -10634, -4922, -2, -2305, -4579, -3854, -13312, 724, 15222, -7747, -3025, 2235, -7947, -19641, -12338, -32, -19061, -5636, 6796, -573, 26067, -10774, 7005, -12481, -13413, -1624, 19680, -16688, -18830, 7841, 7395, -9107, -17835, -14714, 24575, -16423, -11529, -3826, -17146, -11859, -11574, 3052, 4250, -3033, 27120, -15086, -17675, -5421, -6771, -18156, -10059, -9403, -10703, 1865, 8195, 16385, 18524, -4690, 12832, -27782, 26330, 15761, 22954, 16403, 25535, 4002, 2725, 12583, 9642, 7092, 16275, 7660, -8390, 17360, 7285, 19353, 21379, 15098, 21695, -1728, 4799, 340, -5915, 6195, -12728, 14811, 4165, 15179, 16706, 10664, -4250, 8881, 25683, 18002, -7082, 6979, 12150, 7166, -25596, 18704, 19073, 9428, 18960, -11864, 19768, 9042, 16508, -4477, -15130, 4538, 7721, -6393, 9024, 19489, 128, -2670, 18881, 3502, -11913, 8445, 7688, -8447, -8656, 10194, 3268, 9667, 2683, 2251, -8415, -2382, 3539, 6570, -9334, 3654, -2727, -2313, -391, -8461, 8547, 2750, 121, -8199, 7151, -4312, -5957, 407, -1644, -11775, 205, -6019, -7687, 6853, 11630, 1791, 15902, -2065, -2566, -5616, -3116, 4145, -9098, -974, 18359, 12129, 11705, 10981, 7830, -9818, -9189, 9762, 1053, -6160, 14666, 3094, -4418, -4943, 10612, 8993, -5493, -5836, 665, -2366, -7933, 12882, 10742, 1799, 7351, -9584, 3711, 12760, -7487, 2445, -7285, 3318, -10949, 15693, 7926, 7889, 1654, -9615, -12719, 5537, -4821, 15571, 139, -5526, -4019, -1235, 15887, -3127, -7865, -10994, 4510, 2573, -9834, 11768, -2206, -7096, -2744, 946, 7810, -7732, 3169, -1764, 12428, 3912, 2085, 6564, -6134, 9010, 14129, 3118, 9555, -795, 7636, -7209, -9245, -664, -3551, -2511, -1579, -6736, -11555, 1522, 8070, -8816, 4674, -5054, -2295, -3, 6236, 4220, 11081, -4768, 275, 11365, 3059, -4880, 858, 5905, 9847, 14356, 1831, -8484, -2665, -1678, 3878, -6700, 3840, 802, 1288, -12025, 1820, 7074, 12257, -11403, -3884, 7261, 10892, 11628, -3039, -9301, -3771, 938, -267, 7282, -10187, 5577, -6687, -5649, -13254, 2461, 6625, -6509, -8568, -11036, 160, -10843, -3352, 10969, -13243, -1894, -126, 2926, -5492, 11626, -4471, -11711, -6699, -1753, 10104, 16756, 21879, 5092, 16871, -1521, 6094, 12707, 10804, -3588, 7126, 20037, 10318, 1483, 3632, 4971, 20407, 6833, -16028, 16514, 4111, -1253, 23729, 4308, 16126, -5592, 14858, 13848, 16678, 5212, -10591, 15328, 11689, 21857, 11662, 19722, 978, 10441, 7036, -4658, 4466, 12706, 12789, 3525, -1864, 20300, 22153, -108, 7199, 3703, 11588, -8679, 2715, 7498, -15337, 9800, 3287, 2965, 4733, 3252, 7607, 9323, 12367, 6725, 5234, 17401, 14277, 6839, 13090, -131, 5948, 10785, 7628, -10123, -576, 11089, -332, -665, -3130, 13967, 13132, 9788, -1040, -1567, 11899, 20192, 9751, 13407, 9247, -4625, 13157, 14718, -1371, -9811, -9143, -892, 356, 12051, 8105, 2851, -8303, 15022, -4408, 8644, 10075, 8349, 9541, -2876, 2498, 7485, 6710, 7958, 1615, -6864, -1768, -5732, 8407, 3129, -6899, -3931, -3836, -4344, 14773, 19019, -11844, 6242, -1824, -5071, 8731, 7523, 5761, -1932, -4261, -250, 13129, -7692, 8746, -929, -5556, 5089, -9033, 4300, 3481, 14617, 6049, 9197, 12737, 6820, -611, -8172, 12691, 4479, 11044, 9602, 1316, 3518, 12569, -2217, 2314, -10144, -10499, -3868, 8611, -4871, 250, 5445, 14419, 13896, 11131, 12109, 3195, -3399, 10300, 15500, 6848, 10224, 11530, 8821, 14547, -4558, 13755, -2773, -4294, -4050, 10678, 7915, -4845, 16394, -3704, -1401, 6666, -5536, -6845, -3762, 14554, -511, -6396, -8556, 7222, 1868, 10908, 416, 8341, 1423, 3066, -6639, 8962, 16440, 3380, -9809, -6722, -100, 13236, -5837, -6962, 9278, 12644, -5982, 13101, 4077, -5584, 6770, 5925, 4796, -11689, 1466, 15713, -4794, 10225, 13876, -2327, 1307, 1768, 7257, 16868, -3307, -6397, -5096, 4492, -9053, 12348, 8791, 15807, -4487, 5603, -6882, -7719, 16109, 4515, -10945, 12270, 15373, -2654, -1398, -4861, -12304, 7054, 9915, -1788, -6989, -4157, -7282, 964, 13274, 11299, -6532, 17562, -3637, -8693, 12165, 9353, 13642, 6263, -1479, -10615, 8095, 13277, 11278, 257, 18903, 5875, -2615, 6955, 14080, 1864, -2893, -1057, 15822, 350, 4522, 9078, 15250, 8596, 7576, 12587, -1613, -4346, 7908, 19339, 16199, -9934, 9603, -1500, 2099, -415, 5592, 6048, -9268, 13300, 2762, -9642, 15230, 13129, 193, 2184, 8814, -4086, -11663, 10210, 3455, -8392, -3877, 3943, 9488, -1850, 800, -4810, 834, 4493, -637, 7981, 9237, -6246, 13859, -5165, 7711, -7286, 692, 434, 15541, -8066, 2703, 9576, 11709, 13074, -9644, -1375, 10058, 3520, -6591, -8011, 3114, 529, 2850, 4724, 5857, -4236, 11298, -4866, -8855, 9745, 9459, 13574, 9814, 2011, 13807, 5479, -6781, 8318, 10121, 4817, 1563, -2483, -3719, 9061, 6686, 10139, -9609, -4190, -5466, 5370, -12901, 2863, 6455, -9084, 10260, 20825, 10478, 20029, -8395, 10639, 8368, 21343, 5659, 4000, 9053, 10085, 6656, 5729, 22343, 9854, 3786, -17455, 10595, 8683, 16010, 23733, 9227, 12098, -3812, 2337, 21315, -1353, -8021, -13820, -701, 10291, 25207, 13062, 5127, -2679, 13521, 15739, 2675, 3577, 19653, 15778, 10028, -23597, 21912, 13610, -8829, 8800, 7, 9611, 1576, 17695, -3162, -8452, 20154, 5985, -178, 8306, 13879, -8708, 9551, 4084, 15556, 9392, 16459, 11353, -4588, 1333, -12651, 11183, 1392, 21079, -1833, 6836, 15619, -3973, 20722, 17473, 5441, 3956, 8647, 1391, 15749, 12161, 18897, 14594, 20075, 18638, -2454, 22263, 23973, 1224, 1488, -11311, 6491, -10403, 4531, 10125, -1105, -4864, 18239, 727, -5767, -9472, 4769, 19209, 20430, -14899, 6736, 13188, 1999, 10076, -6454, 17200, -8730, 5656, -12000, -1844, 19993, 22457, -9726, 22701, 21006, 9259, 5591, 22820, 14072, 8193, 10351, 4346, -521, 6847, -15889, -5327, 1902, -330, 4255, 17820, -3745, 9073, 9482, -3966, 11621, 14086, 1643, 5084, 10288, 9120, 1411, 16582, 16211, -5063, -8256, 11908, 4915, -12901, 5368, -7327, -1594, -6923, 94, 10955, 9718, -6175, 6524, 17470, 5471, -6136, -1389, -2412, 13868, -14128, 6597, 15167, -8665, 14716, -6914, 8533, -9267, 533, 5579, 6117, 16598, 2257, -3374, 12356, 16897, 5552, -12488, 1914, 14540, -3791, 8423, 14084, -9686, -425, -8994, -173, 11718, 3458, -4276, 7747, 5353, -5195, 13991, 17444, 14786, 9090, 8252, -3002, -4168, 16601, 12120, 6036, 13194, 14941, 11844, 16154, 16766, -3382, 1930, -6712, 11867, -4992, 5648, 3687, 17131, 8358, -4982, 12463, 13806, -11023, 12823, 16461, 9348, 7588, -2056, 4685, -9340, -1499, 2430, 4188, -12642, -4703, 7732, -10869, 4237, 15210, 838, -3216, 8360, -9510, 2872, 5950, 7674, -7925, -7136, 4009, -9861, -7267, 13296, 13502, 4707, -3310, 6079, 2130, 5769, -7091, 11505, 4410, -1619, 8177, -3304, 1151, 8943, 224, -6189, 6628, -2957, 4756, -9782, -2756, -5282, 1859, 8131, 1050, 4286, 10528, 10699, -2124, 7515, 6363, 4227, 4520, -6865, -11774, 5237, -4283, 13853, 6112, -7853, -1269, -4888, 4247, -10214, 2338, 1599, 13032, -942, 11062, 14373, 6806, -2467, 9599, 2792, -10628, 9962, 8384, 11290, -2894, -2829, 10463, -8375, 9092, -9553, 3079, -1379, -8863, 8894, -11847, 10999, 2052, -9099, -6202, 9436, 7164, -8263, 1411, -480, 1570, 4147, 9362, -2185, -5163, 4160, -11168, 2547, -10530, -7720, -8753, -2520, -11410, 2975, -4476, -565, 5545, 7561, 3940, 6682, 11549, 8589, 4441, -7133, -5579, 11746, 329, 4461, 3228, -3802, -5293, -6107, -11190, -4259, 9086, -5441, 9719, 2464, -1628, 2347, 3610, -3709, 8351, 3359, 8335, 21712, 13194, 2142, 20209, -8404, 6909, 5869, 24072, -1744, 15115, -400, -8938, 16182, 16340, 9171, 5959, -8621, -8854, 17582, 14299, 14723, 881, 20846, -1769, 1942, 16063, 7167, -2207, -1794, -14012, 14057, -3160, 16999, -45, 3240, 6312, 7108, 11187, 8885, 6984, 8759, 15876, 15350, -25337, 18195, 26774, 2969, 11188, -9464, 4167, -6550, 3259, -1026, -7403, 15016, 14794, 2828, 24283, 3960, -11858, -9543, 9467, 6725, 7334, 16715, 3297, -12316, 17018, -18519, 20037, 3046, 6926, -443, 8399, 24, -280, -2276, 10802, 19433, 2843, 10556, -993, 7821, 9377, 7721, 17231, 3162, 6254, 3049, 8642, -3431, -9851, -8789, -5706, 18587, 3194, 11759, 19695, -3527, 691, 4625, 1222, 10438, -4455, 7235, 19289, 14508, 44, 18967, -945, -4767, 3444, -8633, 7381, -4266, 792, -3154, -11961, 12232, -3304, -3932, 9523, 8887, -2996, 7516, 18371, 6888, -4305, -14331, 5634, -1134, 2108, 12921, 3318, -5297, -1576, 6013, -15420, -4903, 3926, -4249, 1928, 6309, -1160, -4591, 16509, -13018, -7455, -10869, -1371, -3301, -13990, -6019, -11806, -5742, 4490, 896, 6796, -13337, 7404, -321, -2023, 2513, -1469, -2157, 8098, -7734, -11010, -3970, 4088, 7258, 18881, -7812, 7806, -10625, 6942, -2683, -10300, 9429, 5610, 7767, 4239, 7098, -5195, -5314, -9351, -6181, 7111, -9746, 2872, -5874, -10809, 8364, -941, 3938, 18295, -7954, -2153, 10376, 16111, 7914, -358, 6483, 4096, -5953, 11170, 859, 1640, 3967, -17460, 18969, 2454, 7336, 5678, 17617, -5976, -9046, 1695, 5261, 1115, 1355, -12278, 7745, -9584, 17253, 2274, -1338, 10546, 15093, 11525, 8453, 9384, 10120, 4025, 521, -3371, 8754, 8136, 9578, 14979, -1598, -221, 3332, 16302, 2451, 4925, 4527, 3565, -6032, -1515, 13331, -444, -8232, 4316, -4283, 1754, 8613, -2629, -2727, -1102, -1102, 11235, 10454, 20045, 12507, 14748, 8619, -7443, 15349, 21733, 18058, 15940, 9495, -14664, 312, 1558, 15575, 914, 15777, 17878, 7886, 21002, 17891, 13381, -2966, 2463, 3724, -4507, 18338, 11760, 17895, 9715, 11466, 5476, -2003, 5740, -2774, 12417, 16271, -2310, 20230, 19408, -10678, 18705, 7824, 13842, -10996, 17330, 6445, 834, 15844, 1673, -7416, 2465, 10001, 841, 380, 5838, 14854, -12588, 19885, 24370, 7585, 24419, -27241, 4239, 24480, 4108, -8906, 5374, 5283, -4778, 18526, 20424, 3535, 22666, -6709, -7647, 20229, 21096, 372, 17731, 1927, 7114, 11648, 3811, 2858, -18840, -11774, -12963, 4885, 9850, 180, 2735, 10777, -822, 1337, 881, 18369, 2338, 3785, 4840, 9447, -7901, 21529, 16681, 7579, 15924, -2776, 7844, -2348, 15446, -3648, -26825, 21411, 2861, 7829, 13527, 8724, -5030, 10036, 17008, 21059, -1751, 8892, 17131, -9178, -824, -4132, 5759, 6980, 13853, 5335, 4561, 12215, 5355, 19411, 7927, 11846, 20615, 2503, -9619, 15133, -1820, 14431, 14256, 13189, 15449, 9487, 4803, 14830, 9121, -2922, -4392, 1880, -2384, 17183, 20184, 13837, 1351, 17931, 15973, -2665, -5002, 992, 18682, 16898, 121, 4788, 13083, 3143, 4377, -218, 16911, -12119, 13108, -10697, 3752, 1004, -2218, 7362, 13763, 16431, -2554, 2352, 4873, 10514, 7801, 1526, 5157, 7676, 7118, 4456, 6859, -5264, 12097, 6132, -4405, 10806, 7818, -3303, -8136, 4631, 3211, -10744, 16475, 9064, 13212, -1281, 7272, 11621, -7270, 3139, 899, 5562, -2176, 86, 17088, 9500, 2505, -983, 3649, 8772, -5161, 8503, -6034, 6120, -2974, -3858, 10943, 3028, -1938, -5535, 5606, 10687, -3712, -4705, -848, -13294, 6536, -10756, -4636, -3995, 10096, -14192, 2296, 15357, -6313, -8503, -8028, 2518, 918, -2393, 10710, 876, -637, -9327, -5079, 13932, -1088, 2406, 7324, 357, -12208, 9772, 12628, 15361, 9398, 4270, -9431, 15210, 1816, -2508, 10853, 15434, 14121, -11870, 1852, -3559, 15692, 10099, -3285, 14686, -1273, 14785, 11212, 10218, 2782, 1848, 6435, -4223, -11555, 8781, 4126, 9751, 1700, 14551, 15290, -702, 111, 8093, 5129, 10463, 15206, 1343, -10373, 4861, 5609, 1775, 7012, 16278, 2366, -4722, 5871, -9617, 2126, 11154, 8406, 2825, 7919, -10642, 11794, -2178, 321, -1320, -1168, 2414, -9851, 7103, 13583, 16634, 14624, -10614, 19, 10746, 16342, 6031, 16295, 15812, 4857, 5569, 17774, 15566, 5118, -6970, -5138, 8134, 1903, -6278, -2562, -5632, 8076, 6010, 6119, 1510, 3666, 7602, 17587, -3504, 8592, 5434, 17762, 10919, -1292, -12075, 12359, 8510, 15841, -5796, -4668, 16470, 9379, 5354, -2858, 17600, 1175, 7702, 2752, 20, 4911, 30363, 15594, -1832, 32767, -25029, 11517, 17029, 25481, -679, 23717, 17811, 10775, 10389, 11645, 23350, 11282, -6545, -24010, 20834, 29721, 15091, 11524, 31445, 8144, 4573, 20832, 16849, -5519, 7725, -19226, 30973, -10088, 11749, 31365, 17639, 2589, 19144, 14651, 22326, -10676, 22006, 31724, 14065, -24433, 16986, 14301, -832, 11079, 5306, 28030, 2165, 30102, 1061, -26688, 17071, 30220, 7508, 28091, 18659, -298, -7157, 27557, 2372, 4143, 12227, 1280, 507, 13271, -8024, 14246, 6027, -147, 5908, 8247, 14800, 5119, 19190, 14766, 515, 12468, -2135, 2372, 5410, 5790, 21572, 20601, 7263, 15848, -1349, 9854, -1880, 2385, 5040, -22084, 11758, 7757, 7619, 23659, -56, 9110, 20760, 11410, 18614, -6656, 13904, 2165, 3801, -20655, 8534, 4879, -2669, 10750, -5805, 20095, 6980, 2512, -10286, -12871, 5248, 22797, -12232, 22702, 22474, 2087, -2846, 13997, 6505, 9784, -41, 10552, 2039, 644, -8666, 14817, 8381, 2741, -196, 992, 10971, 4452, 1618, 18090, -4576, 5556, -8549, -22268, 4140, 6734, 17206, -1023, -3677, 1291, 2141, 12868, 12641, -6709, -9857, -20264, 16290, -12526, 11744, -2461, 8842, -5143, 262, 8718, 486, 2986, 15749, -2810, 11067, -14886, 2939, 23986, -7352, 8228, -3666, 10217, -9878, 12702, -2792, -27596, 8293, 3383, -2866, 8787, -3409, -1725, -7963, 12553, 19811, -10486, -7382, 8996, -10953, -8920, 25326, -7022, 6148, -1531, 5420, 5055, -3983, -9485, -5246, -1426, -370, -7437, -3358, 9895, 8447, -6880, 4418, -7175, -6428, -14721, -653, 3358, -2629, 17280, 7726, 17263, 6627, -12819, -3251, 2511, -11260, 21322, -6427, 7750, -5889, -12028, -9584, -10672, 3292, 5846, -3788, 477, -9611, -11476, -6361, -4733, -1291, -8272, 6329, 14078, 3661, 1286, -9938, -10124, 8485, 4727, -4963, 1293, 12205, 2076, 9328, -7480, -7519, -1147, 6993, 12130, 12945, -8592, 633, 12251, 9008, -9253, 3619, -3867, -9490, 2465, -11684, 17617, 11164, 10457, -4350, 5145, 8403, -6303, -9737, -1531, 8312, 9127, -169, 3882, -7976, 3956, -2626, 4912, 5099, 4589, -2954, 9155, -2075, -8529, 12471, 11868, -5831, 8786, 12955, 783, -5244, 2095, 3664, 11126, -13097, 8259, -9884, 10080, -3566, 793, 4452, 12113, -5855, -7774, 1011, -7873, -8047, -6859, 4888, -1435, 1491, 4650, 21653, 6223, 840, -3721, 8206, -2113, -6749, -7946, 12208, 7460, 10612, 13534, 3638, 15440, 5829, -942, -11609, -4525, 10575, 758, 1882, -3490, -7136, 8333, 10963, 20600, 3144, 3675, -3843, -4154, 11222, 6603, 6890, 11682, -4993, 4238, 4499, -4644, 11944, 4349, 10864, 12085, -8894, 9997, -1724, -3723, -9607, 1742, -897, 2767, 1909, 7473, 2760, -1292, -4402, -3489, 8488, -62, 10564, -8222, 21473, 25304, 6244, 23580, -18280, 20908, 14064, 10231, 9229, 20383, 20751, -7090, 14715, 3989, 5158, 14471, 8371, 399, 19295, 22089, 14801, 9182, 8299, 4742, 10659, 20412, 20449, 10517, -5, -413, 1018, -10001, 11565, 17496, 23326, 1020, 15801, 7830, 7546, -10635, 24005, 23640, 6548, -18818, 4101, 4240, -254, 16167, -8342, 12401, -6096, 21964, -3873, -9988, 2035, 8577, -10160, 14207, 12866, -9066, 9043, 14376, -6397, 3778, 8323, 4869, -3517, -1754, 9732, 7058, 9341, -6727, 4481, -2427, 7852, -12059, 816, -3740, 1231, 6241, -6684, 343, 35, 816, -6789, -690, 1230, -2051, -10984, 7935, 2923, 11187, 1096, 16375, -6828, 8187, -197, 9095, 4598, 18550, 7051, -3939, -6078, -7693, 8919, -1667, 14021, 18951, 1893, 12893, -6180, 4216, 2702, -7309, -13863, -3324, -5932, 11069, 6918, 8178, -13605, 7407, 2119, -8773, 6200, 6941, 2026, -3062, 4510, 7053, -11983, -1732, -258, 5581, 7271, -7488, 9780, 11299, 12193, -6867, 13095, -1350, 4193, -3318, 1224, -442, 8618, 9779, 5218, 12329, 12818, 2086, 5751, -7499, -2489, 10030, 6328, -2764, -6045, 4009, 107, 5063, -2201, 2874, 4006, -1976, 6132, -929, 3246, 9159, -8586, 10432, -7923, -2295, 7021, -4763, -7434, 6931, -10007, -1201, -8153, 10212, 5447, -4273, 7410, 5375, 3856, 1277, -10074, 1182, -7819, -10123, 15388, 1374, -911, 2249, 5274, 246, 2338, -1833, -3450, -7229, 568, -6265, -6608, -757, 15413, 7590, 3870, 2042, 7905, 10763, 11548, 8347, -2122, 6743, -1157, 3924, -1015, -7837, -1806, -8113, 12155, -7441, 12136, 6181, -5313, -7205, -2394, 4488, 11594, -10516, 1063, 17195, 9812, -10545, 32, 12640, 4729, 16135, 7602, -3329, 890, 1902, 1838, -2520, 11342, -1721, -12359, -4751, 16973, 8393, 4924, 8740, -1975, 7222, 8146, 8924, -8224, 13663, -1004, 10942, 9293, 20749, 13638, 1088, -1870, 2730, 16368, 12565, 6050, 6224, 4820, 3222, -783, 6117, 7047, 6429, -1576, 18197, 7183, 1075, 16677, -773, 825, -14880, 14474, -4027, 8886, 20961, -2170, 9881, 13946, 6361, 9415, -10687, 5113, 22278, 20648, -20931, 19556, 16421, 933, 284, -11054, 20022, 3769, 19013, 3103, -12120, 11029, 10694, -9649, 18426, 13347, -3826, 10263, -284, 9121, 5491, 3622, -5859, 7050, -7802, 21106, -3684, -15363, -8858, -10001, 1158, 1704, 5927, -1929, -11324, 1711, 1860, -2351, 8787, 2344, -12124, 3832, -3165, 2616, 6700, -7140, -13182, 6985, 8143, -8422, 18430, 2208, 894, 5485, -6333, 163, 7165, 2308, 3702, -10547, 3904, -6436, -9155, -4760, 3050, 5968, -10680, -11479, 7837, -6848, -10648, -13137, 1113, -6623, 22979, 185, -7713, 4200, 418, -721, 7244, 4287, -2537, -9618, -8537, 10324, -2047, 8421, 658, 8539, 15151, 5359, -1201, 5873, 2544, 8271, -8255, 12023, 947, 17347, 5052, 1084, -4859, 2500, 559, 3249, 8990, 4098, 4251, -4589, -5502, -1390, -6508, 10464, -8170, -8097, 2324, 12205, 12941, 54, 13427, 4924, -1312, 2972, -2859, -2233, 7649, -2820, 3521, 3214, 14222, 8328, -4564, 2038, -1090, -5960, -5933, -4684, -5764, -3480, -1278, -10839, 8099, 10164, 2243, 8906, 16824, 4441, 2924, -4392, 6875, -9406, 2221, 15052, -2436, -8309, 3523, -12531, 3894, -6581, -5679, -4666, 11362, -4163, 13423, -1776, 13471, -4560, 5971, 11825, 4539, 5372, -7734, 7330, 1633, 829, 5279, -12327, -4409, 2842, -11473, -2698, 9625, 15045, 239, 3568, -2389, -1474, -1418, 12264, -4184, 4088, 9930, -4262, -2594, -11608, 6988, 1282, 7373, 5038, -8378, -9744, 6131, 11495, 7455, -6829, 4746, 3334, 5004, 2975, 11077, 5627, -11117, -5635, 7926, 3604, 525, -2072, 301, -2340, -4562, -7749, -10113, -2284, -6614, 3179, -5960, 494, 11320, -3415, 6640, -8408, -4873, -5695, -8614, -889, 1782, -785, 12872, -8468, -11634, -472, -8938, -5060, 8499, -4361, -2240, -10495, 8104, -6739, -4711, 11054, -7482, 10187, 898, -9793, 8298, -10890, 11626, 11435, -699, 3003, 8686, 2131, -1135, 241, -12205, 1857, -10947, 10686, -4124, -3058, -3914, -8379, -2994, -3177
};

static TinyTensor inner_fc2_weight_t = {"inner_fc2_weight_t", inner_fc2_weight_t_data, TINY_DTYPE_INT16, 2, {64, 64, 1, 1}, 4096};

static int32_t inner_fc2_bias_data[64] __attribute__((section(".data"))) = {
    179937008, 210937312, 59117716, 14448044, 209766800, 14735078, 58234536, 111743720, 281295360, 81780984, -138242368, -23219204, 112143528, 208443104, 116425424, 128237376, -55362768, -231460240, -43176424, -36355800, 235278640, -16820038, 201229696, 168180864, 223517648, -179240896, 276532416, 260782960, 35631548, -143736880, 169599984, 90499944, -30624002, 206310384, 49523260, 101566424, -4070902, 61370672, 137102560, -73837640, -228591664, 260511472, 141094064, 233189632, 85956504, 166612848, -65338496, -30198618, 263150112, -129281752, 167067984, 141059264, -102534856, 9128, -48345920, 86864672, 221065408, 138784064, 212888320, 151327264, -63797356, -201192160, 258971616, -158100304
};

static TinyTensor inner_fc2_bias = {"inner_fc2_bias", inner_fc2_bias_data, TINY_DTYPE_INT32, 2, {1, 64, 1, 1}, 64};

static int32_t inner_fc2_data[64] __attribute__((section(".noinit")));

static TinyTensor inner_fc2 = {"inner_fc2", inner_fc2_data, TINY_DTYPE_INT16, 2, {1, 64, 1, 1}, 64};

static int32_t relu_1_data[64] __attribute__((section(".noinit")));

static TinyTensor relu_1 = {"relu_1", relu_1_data, TINY_DTYPE_INT16, 2, {1, 64, 1, 1}, 64};

static int32_t inner_fc3_weight_t_data[4096] __attribute__((section(".data"))) = {
    -6115, 7402, -12883, 5866, -5729, -1678, -10936, 169, 2757, 8382, 3710, 9945, 1356, 2230, 860, -2024, 5919, -5057, 7084, -5236, 8278, -7735, -3064, 8075, -12471, -10205, 7544, 6645, -937, 3596, 2898, 208, 12624, 10830, -8679, 6815, 711, 1424, 10445, -11295, 4840, -10207, 1280, 8264, 7929, 4007, -2268, 3918, 5573, -7753, -1168, 3815, 5789, 1278, 7432, -4066, -2229, -11695, 13678, 2975, 5000, -8756, -11440, 8669, 5478, -13696, -16870, -3602, -7021, -2558, 5852, -16713, -8791, -2607, 1493, -10810, 8124, -5407, 7517, 1277, 13490, 6945, 11103, 11751, -529, 16100, -15667, -11141, 4985, 1443, -2327, 96, 1876, -9686, 4657, -16261, 7431, 5863, -17594, 15038, 7491, 6536, -11542, 4985, 8367, 10031, 1486, -1928, 1716, 15632, 1864, -11587, -4523, 5472, 13200, 1089, 7319, 6552, -12001, -17692, -15113, -10131, 2196, 17330, -4178, -1475, -12056, -17763, 4764, -5276, -15577, -11672, -1367, -10978, 7787, -7807, 4112, -712, 3624, -9643, 19525, 7796, -13681, 8369, -1432, -2321, 1134, 20082, 4096, 15041, -16405, -1433, -6104, 14563, 4764, 17911, 5017, 2909, 7156, -14507, 18699, 8378, -385, 8963, 910, 846, -7119, -17661, 5452, 15670, -6634, 3577, 7303, 16487, -9387, 1966, 6258, -17447, 16395, -3532, -1395, 9576, -3447, -7356, -15549, -16177, 9938, 10394, 6583, 12017, -14541, -1338, -73, -9860, -13578, 1884, -11987, -14032, -9055, 7188, -7000, -1707, -149, 10555, -5565, -389, -10878, 2879, -124, 7848, -10598, -5102, -6759, 657, 10329, 9765, 7575, -2711, -326, 283, -12079, 7426, -901, 5876, -6229, 11562, -2431, -10346, 8897, -9548, 9539, 11393, 11683, -6800, -13440, -5532, 9744, 8464, 7565, 11534, -5113, 4472, 4152, -11685, -9815, 10867, 880, 3710, -5037, 6423, 633, 6775, 8587, 13525, -2294, -10907, -7737, -2532, -6352, -12658, -19535, -5570, -4421, 847, 2691, 1125, 22002, -18818, 468, 4888, -16041, 22830, 4062, 2047, 20719, 2708, -8196, 19503, -8002, -525, -14382, 19233, 2, 22409, 4909, -13106, 17782, -2634, 23735, 17428, -15317, 18536, 11629, 13639, -8098, -9416, 3874, 5810, -6515, -25804, -6756, 7005, 5306, -3195, 16810, -5671, 16570, -4373, 544, 6794, -720, -7637, -14057, -9071, -1520, 11487, -15201, 17037, 6002, -5694, 30111, 25280, 2867, 15403, 18726, 23064, 11854, 13254, 9764, 14154, -13377, 25326, -20923, -6324, 29224, -22922, -5627, 21039, -715, -16898, 14870, -19960, 12561, 21979, 17352, -1893, 29748, -15994, 11632, 19942, -21440, 23430, -1597, -11471, 21098, -10134, -7681, -748, 8287, 18318, -7173, -10459, 28916, 22935, 23152, -731, 26939, 31125, -13818, 13247, -22950, 22440, 32767, -3883, 26127, 13664, 26803, 19337, -14865, -22104, 19481, -16159, 5067, 9874, 3871, 9115, -9076, -952, -1522, -6484, -11202, -6876, -17728, -8156, 3940, -10402, -2638, 16136, 10814, 5120, 1278, 1159, 2944, 15696, 8162, 8777, -11839, -12913, -5868, -3765, 4698, -4773, -16063, 6800, 14102, -2152, 11883, 12769, 7971, 17271, 1003, 3891, -4024, -11074, 13112, 15495, 6775, 7300, -14167, 5089, 1632, -15674, 8692, -13708, 17091, 2001, 5373, 5067, 3786, -10782, 7556, 7494, 16160, 12415, 7367, 12011, -7105, -5238, 7452, 5319, -8622, -127, -12706, -7341, -17935, -17922, -14150, -4805, 15605, -9236, 4529, 2815, -7916, -1894, 1275, -4168, 10599, 9475, -426, 6387, -10396, -14498, 7948, -1823, -543, 3434, -13483, 5158, 5617, 3270, 13533, 4548, -16085, 3489, 10129, 2112, -15993, 7370, -2068, 2243, -11594, -11146, -13439, 10419, 4080, 7480, 12453, 2914, -2648, -10130, 417, -9712, -272, -16552, -7860, -8566, 16632, 4596, -13219, 4367, 4906, -7939, -13765, -11246, 5077, -6800, 1895, 3097, -14006, -1282, -20348, 8189, -581, -16074, 20332, 11159, -3579, 9327, 16961, 1274, 15881, 18208, 3026, 13265, 879, -7139, -13501, -132, -8233, 19706, 6203, -6741, 3067, 4094, 4300, 2966, 5173, 10492, 394, 10233, 1960, 3382, 2504, 286, -459, 3637, -14349, 24624, -4780, 6185, 21147, 5028, 13983, -5884, -17334, -115, -13836, -14270, 1074, -4612, 5674, 13165, -15663, 23891, -14271, -18879, 1005, 3669, -6413, -12661, -11530, -12709, -574, 12212, -3693, 7260, -7401, -2236, -75, 8278, 2726, -6996, 10342, 7324, 10963, -4806, -14006, 1410, 9516, 9852, 10063, -4795, -6665, -3076, -11438, -7022, 14160, -2907, 4318, -2118, -5140, -8068, 5675, -3601, 1511, 3670, -2485, -12244, -9789, -885, 2919, -8113, -8384, -4581, 6057, 10290, -4677, -2954, 15599, 2737, 3833, 8784, 2550, 2335, 7021, 6188, 722, 4063, 5723, 15245, -1219, -17205, -16469, -12282, 7638, -10805, -4074, -17955, 5694, 5442, 5831, 3029, 2291, 10049, 9280, -5044, 4575, -1448, -2970, -6168, -7834, 16301, -7131, -12254, -15971, 6717, -4797, 2782, -8725, -9276, 11596, -18214, 5511, 2771, -12608, 15433, 6811, 2182, -5616, -7906, -2298, 6345, -3527, 6892, -14629, -6639, -9498, 3290, 15729, -7158, 3951, -9322, 1563, 4208, 8285, 1938, -6604, -14635, 18382, 6448, 1070, -978, 2991, -5892, 701, 2823, -4179, -12665, -1949, -10279, -13538, -15979, -6376, -17022, 15480, 2095, 18339, 2411, -810, 112, 14251, -8627, 11316, 5228, -11006, -2119, -5035, 2572, -3542, 17405, -13141, 16945, -14386, 4284, -4363, -13863, 9442, -860, 1614, -1857, -2245, -6317, -13856, -2947, 8447, 6741, 4501, -2031, -8207, -5341, -11012, -9584, -2157, -13025, -5130, 4024, -9227, 6612, -2593, -639, -254, 2709, 17809, 14694, -16215, 17538, -13112, -11424, -9312, -4870, 11675, 2958, -201, 8597, 11535, -778, -10746, -10151, -7917, -4624, 11729, 4855, 3159, 10172, -3005, -3535, -10200, -11028, 9893, 8491, -11763, 8858, -12234, -5478, -9833, -8758, -7016, -11254, 1846, 9866, -3762, 12738, 9269, 2703, -8235, -42, 1237, -1841, 2551, -6316, 5820, 2433, 9833, -6700, -702, -6276, -8731, -14035, -2144, -9807, 11753, -6134, -11936, 8796, -504, 4358, 1978, -11988, -6222, 7581, -9656, 1258, -4551, 5086, -13443, -9332, -13823, -11850, -7157, -11868, 9560, -9302, -1429, 7563, 9318, 8246, 9309, -5263, 9547, 3079, 13150, 11819, -13300, -7897, 738, -7681, -1824, 16264, -8302, 10933, 4821, 2092, 9668, -14576, 8199, 16865, -3213, 8140, -1136, 18666, -1060, -13543, 1035, -8400, 3737, 100, 12333, 16152, 5948, 185, 3749, 4746, -6610, -11861, 13681, 9099, 5605, 6591, -7658, -6824, -6765, -5678, 11807, 13170, -4119, -8870, -6374, -3326, -11223, -2241, -20430, 4675, -16925, 873, -23642, -18804, 21728, 2858, 3552, 17512, -9385, 14747, 9350, -7669, 9776, 17381, -2466, 23102, 1957, -10103, -13631, 22608, -12169, 15965, -8913, -21876, 1278, -2664, 23881, 14595, -15152, 6577, 142, 24950, -8346, -10783, 22139, 12980, 705, -6359, -22160, 21738, -19465, -20107, 13708, -7384, 5162, -8756, -22754, 10921, -1762, -15815, 687, -2412, 9764, 17265, -2381, 23703, -13739, -1366, -11698, -12129, -4503, -6812, -28, -15463, 5529, -4339, -16685, -13381, -7014, 7994, -2133, -3294, -202, 15569, 13025, 3013, 8091, 13957, -8639, -1371, -11518, -6537, -7176, 10694, 8687, 5212, -7605, 7353, -4077, -14337, -1759, 1656, -15346, 9689, 10133, -991, -5181, -11268, 11075, 492, 8928, 7900, 6608, 7933, 1552, 6767, 14787, -4062, 11226, 9146, -9884, 2808, -6677, 859, -4124, 4944, -4184, 17408, 5049, 12884, -12140, 10157, -5601, -14413, 1167, -8913, -13529, -14770, -9380, 2253, 6593, 6048, 19108, -9132, 6491, -2812, 587, 6581, 9379, -15849, -1032, 7997, -13832, 19072, -12729, -15350, -11279, -6530, 4270, -4818, -9527, 4887, 13693, 6487, 20001, -337, -3756, -486, 10121, 15528, -13313, 100, -565, 3476, -10714, -1407, -13318, 20788, -8895, -17763, 18858, -4319, 8149, -10047, -3452, 2016, -16696, -10512, -7304, -7888, -143, -223, -16736, 16692, -11505, 3613, -12370, 11764, 4085, -11960, -8359, -5387, 13628, -9048, 10966, -4660, -3339, -3074, -10615, -6475, -745, -7251, 7720, 210, -6054, 3111, 10732, -5645, -108, -2056, 5833, 8996, 9867, -8766, -356, -6654, 505, 2024, 3141, -1059, 11613, -11484, -4119, 2186, 7350, -762, 13633, -5216, 6741, 5893, 8226, -11692, 5031, 7013, -6712, 1560, 6925, 6700, 3375, -6119, -11495, -4253, -11476, 6878, -2425, 12805, -5293, 1375, -3885, -4065, 11039, 1338, 3409, 23444, 26107, 15770, 4572, 19994, 8030, 11395, -5136, 23869, -12930, -16638, 24602, 3854, -19725, 17870, -10291, -2948, 7572, -2680, 23986, 15243, 15679, -12300, 26432, -16247, 23450, 12890, -1444, 18136, -10913, -17180, 19515, 6518, -5022, -2752, 24796, 27165, -10956, -779, 10647, 15087, 20244, -5086, 13583, 25011, 668, 25854, -519, 22317, 7722, -16449, 25229, 17976, 1067, 6852, 2332, -15267, 6277, -14199, 4734, 26970, 7534, -6345, -10074, -8741, 7757, 19046, 14433, -5043, 7867, 5915, 4749, -2355, 15977, 5391, 4320, 18061, 21722, 4918, 8322, 1329, 761, 20705, 11843, 6028, -13356, 15818, -6845, 7122, 6231, -7879, 19998, 3427, 12566, 8019, -8971, 4724, 19307, 8321, 8584, -2340, 23106, 9972, 6487, -20287, 7060, 24600, 2801, -715, 1160, -17890, 1980, -14220, -3096, -338, -5633, 5224, -601, -15946, 2897, 15589, -20016, 18753, 8918, -22247, -10450, -16753, -15189, 2561, -1914, -5568, -6129, -3497, -8457, -929, 5431, -9537, 19044, 16753, -7598, 12084, 5428, -7909, 8637, -2761, 1549, -2949, 3356, 7054, 4164, 2781, -5121, 1321, -6840, 714, -815, -12337, 2224, 13694, 1741, -1964, 5843, -5885, -4456, 9415, 13024, 15491, 6653, -4823, -11246, 8201, -12742, -16515, -2025, -5937, 9841, 7025, 3186, -2510, -13392, -12915, -618, -4023, 6829, 12686, -823, -1919, 7515, -16170, 2695, 8156, 4242, -9002, 8928, 4257, -5624, 5036, 2013, -14680, 16093, -4017, 5823, 9557, 2615, -1010, 763, -8261, 21602, 7215, 6992, 5321, -4410, -9531, -12226, 11314, -5599, -1964, 2687, -4679, 8531, -5276, 4710, 704, -10497, 13936, 6749, 16679, 9765, -14980, -593, -137, 5269, -5475, 5672, 18270, 6064, -5903, 1902, -12153, 11361, -10734, 1665, 12455, 7439, -8401, -8038, -1218, 20497, 7748, -9393, 12708, -13344, -16761, 3143, -6199, 5571, 782, 6017, -16064, -13882, 5193, 3006, -6958, -1456, 8331, 7689, 20058, -3522, 11083, 7367, -4626, 20541, 20316, -419, 19944, -13482, -4383, -11044, -857, 135, 6926, 7525, -9770, 17484, -12527, -5719, -5811, 5997, 6485, 17717, -5396, -8970, -13826, 4181, 7422, -15466, 2532, -8406, 12699, -13594, 6947, 7298, 5731, 15799, 1827, -1602, 7574, 5917, 5514, -12762, 4599, 5810, 17532, -3891, 18781, -12123, 675, -11757, 8860, -5489, -10621, -12099, 9492, -7171, -4925, -9511, 6812, -453, -14633, 16950, 1021, -9643, 12264, 15136, -16971, 1176, 16433, -15099, -4302, -12687, -11168, 8046, 9558, 3779, 10786, -14591, -8092, 6726, -8508, -4725, 15584, -7393, 10712, 10553, 9407, -3656, 5486, 9508, 10836, -8264, -8222, 8050, 11420, -706, 6857, 9977, 6788, 4014, -17106, -12912, -10163, 3491, 3800, 795, -17795, -1428, 1875, -8313, 11326, -12361, -6482, 1056, -3749, -11129, 8070, -6031, -2257, 8538, -6, 4755, -16490, 5640, -15626, 6220, 15552, -3750, 19871, 8631, 4428, 12849, 2702, -818, -4582, 7576, 1435, -14747, 4520, -15372, 18151, -12861, 5348, 4826, -7269, 17935, -1132, 2897, 2199, 9605, 10054, 790, 3340, 19181, 10699, 1230, -11930, -5072, 18651, -9474, 7997, 18394, -17114, -1249, 5353, -6865, 9083, 6553, -9335, 9309, 497, 4102, -4048, -14196, 5819, -10652, 1793, -7562, 8933, -11658, 3366, -3280, 3485, -12573, 13100, 4530, -2711, 8672, 9211, 2776, 5199, 9551, -12846, 3979, 10767, -5219, -6460, 2459, 2725, 12728, 8953, 3412, -10931, 9026, 2327, 8020, -5642, -6405, 5619, -12084, -3185, -9377, -5064, -5320, -5883, 4604, 11013, 2654, 2577, 6458, 858, 7695, -12352, 10470, 8442, 6716, 1562, -11201, -11508, -8029, 1808, -4206, 5121, -11983, -3818, 4739, 5894, 5970, -7685, 8753, 9884, 5564, -741, 507, 1376, -5571, -8271, -9620, -10818, 3218, -17891, 10198, -3509, 1092, 624, -3479, 20303, -3485, -12092, 20542, 19513, -9567, 16649, -17, -4499, -1363, 7425, -14857, 824, 427, -12485, 17114, -11857, 10696, 19882, -2872, 1247, 13984, 13342, 5388, -425, 2870, 3722, -4982, -10883, -8021, -3421, -14751, 7101, 12339, 1664, 17914, 2429, -569, -4089, -14151, -12949, -14157, -9857, 14706, 13320, -4243, 11771, -15145, -15170, -2255, 1888, -1991, 4837, -182, -9480, -4055, -5912, -14759, 6637, 543, 8266, 6973, 10494, 1362, -1336, 3637, 1433, 15954, 21783, -4748, 21794, 2387, 9371, -2901, 5563, -4754, 3070, -8795, -16586, 5925, -4183, -3202, 5808, -178, 16553, 23567, 18743, -8154, -85, 20836, 10897, -12594, 4083, -702, 13620, 6159, -1235, 17832, -12750, 3172, 7048, -5624, 14130, -8569, -13583, 2980, 5107, 4429, -2057, -4188, 10311, 9195, -11141, 7519, 11841, 6422, 11141, 16074, 17777, 2934, 11825, 7105, -3824, 13064, 3522, 13202, -582, -1291, 14902, 1332, -4538, 9824, 5588, 3866, -183, 21186, 16636, 8907, -1136, 13600, -6026, 19345, 7422, 6757, 11164, 15623, -1736, 9834, -1865, -511, -7386, 19651, 412, 14861, 7964, 13771, 19720, 18572, -3389, 10494, -2862, 12228, 8456, 12499, 1630, 9008, 10546, 10425, -5300, 19860, 1035, 2223, 554, 15326, -8636, 7859, 907, 1259, -580, 1338, 4027, -3778, 4389, -3042, 7361, 5484, -5535, -12418, -878, 9315, -9925, 12450, 3469, 5703, 8712, -10286, -9921, 2309, -2579, -8319, -4284, -8414, -9214, 10676, 5281, 3584, -13869, 10986, 7469, 4463, -5314, 1635, -3447, 5228, -8024, 9109, -12023, 4926, 11508, 12488, -11311, -3187, -9994, -2021, -9820, 8524, 9560, 6061, -11710, -1460, -11518, -7498, 324, -198, -4440, 3465, 347, 8577, -4601, -11312, 3670, 14808, 16573, 3790, 6490, 24700, 60, 22716, 10868, 10837, 19355, -12371, 17999, -16410, -14368, 26097, -20142, -18028, 26702, -14121, -3622, 12247, -8661, 20185, 20024, 15102, -7725, 8586, -6022, 24369, 29062, -1745, 14300, -7779, 2317, 19161, -5693, -8174, -15156, 21965, 5155, -21965, -25311, 11004, 11946, 31042, -12482, 13887, 24811, -3771, 24100, -11244, 21185, 8414, -26282, 22031, 8686, 21949, 24761, -17190, -6311, 26403, -18474, 20841, 31560, -1702, 4286, -13374, -10459, 6686, 1386, 6951, -14932, -7357, -8793, 3229, -6657, 1162, 694, -8270, -3046, 9541, 5537, 5220, 1732, -12093, -3762, 2235, -15787, -18710, 3764, 1337, 18265, -1939, -11120, 18449, 3297, 15471, 799, -2889, 16368, 3780, 1137, 5399, -8891, 665, 309, -4533, -5673, -4002, 19335, -5419, -13053, 14859, -8888, 15586, -9971, -20128, 1345, 5697, -7033, -2905, -2055, -5712, 19951, 1555, 4730, -1650, 4517, 6542, -12892, -567, -8389, 7462, -8998, 8343, 1926, 10213, -4799, 11275, -10455, 3890, 2222, 7590, -6213, 5054, 3931, 3179, -8502, 6064, -10547, -7920, 7107, -7932, 8439, 9499, -11407, 7636, 11535, -490, -7343, 2857, -2893, -8966, 2952, -9626, 1534, 10458, -8505, -10476, -4239, 2015, -6833, 5, 1866, -11821, 1010, -6004, -337, -3178, -3562, -1077, -1916, -7063, 6078, -7059, -13788, -11786, -891, -5365, -9160, -5058, -7662, 3743, -13393, -921, 858, 3684, -15851, -12215, -709, -832, -14404, -5858, -10604, 838, 1382, -7032, 3385, 3222, 2356, -55, 9452, -10100, 13319, 7112, -3922, -3543, 14459, -14304, 20223, -12357, 6122, 6669, -8152, 12574, 16648, 7996, -4025, 2976, 11754, 2499, 879, 17473, -3409, -14044, -1687, 3347, -1830, -9839, -16121, 8312, 3515, -2991, -9093, -181, -9241, -9670, -16918, -10519, -3098, 18637, 5913, 7634, -5842, -8320, 7563, 1854, -5989, -5521, -15802, -5365, -14821, 7426, -2628, 6023, -13321, 15207, -7956, 19051, 9333, 2887, 2924, 12591, 3932, -5593, 15365, 5093, 9437, -6601, -15695, -13949, 4841, 843, 594, -12997, -12167, 11632, -2856, 14235, 12722, -15170, 9473, 3909, 18333, -1188, -1619, 14008, 11469, -11910, -14830, -1337, 18052, 4127, -13080, -2655, -15112, 1908, -3217, -8678, 12683, -14257, -2654, 7322, -8981, 15374, 11029, -987, -673, 2913, -16890, 2821, -7641, 5501, -13527, -4361, 11090, 6791, 7641, 141, -1228, -52, -7516, 7864, 7176, 239, -2325, -7533, -8346, 5766, 14908, -8106, 16382, 850, 591, 2128, 10016, 2077, 16822, -7577, 5514, 10392, 6832, 13211, 5910, -9119, 16912, 7493, -2781, -13918, -11360, -2754, -5814, 5215, 12368, -1265, 10187, -153, 1904, -3327, -14620, -3835, 5930, 4970, 120, -5910, -977, -7213, -2100, -8392, 8739, -4779, 5979, -6419, 290, -5849, 13070, 14377, 19769, 11185, 19349, 1726, 12484, -1577, 20145, -2772, 16319, 9593, 11409, 14839, 5638, 13602, -2571, -5305, 3651, 6545, 3299, 21237, 1267, -4626, -9215, 1329, 11297, 4504, 15293, -6952, 19236, 5135, -4513, 17482, 12863, 13320, 10812, 16632, 964, 9412, 14466, -4793, 2574, 6319, 17734, 10038, 8069, 659, 10251, 5205, -4979, 16211, 3, 2836, -4175, 7832, 16084, 1374, 8342, 8592, 16353, 15170, -1747, -14707, -14482, -5059, 5227, -15448, -6527, -1251, 8823, -6783, 6575, 6119, -11344, 15540, 178, 8809, -2520, 8550, -8691, -6167, 1651, -6738, 12752, -15140, 1208, 5354, 1306, -4573, 4539, -11456, -7288, -6270, -1448, -2550, 10534, 4818, 6624, 10940, -5419, -13800, -2379, -3999, -3576, -12697, 2677, -2197, 11252, -3734, 6313, 10989, -15295, -6927, 2362, 9533, -4906, 1199, 5992, 1823, -9470, 2332, 16683, -4445, 3612, 608, -7239, -3130, 6046, -15030, -15902, 3703, 887, -13834, -14613, -7465, 4846, 308, -12743, -209, 12473, -6102, 11065, 14463, 1723, 6586, -6182, 5785, 5714, -13382, -1890, 3152, -2733, -6025, 18372, -15063, -2535, 8080, 7780, 13174, -3591, -2830, -23, 10548, 5629, -11922, -11007, 8777, 17778, 4640, -16051, 11279, 9466, -415, -2566, 13611, -213, 11123, 9443, -461, 3972, -6643, 889, 5047, -12915, 8161, 12178, -11385, 734, -3031, 9398, -1919, -4422, 8435, -13524, -4006, 751, -7317, 6612, -7903, -17428, -1708, -17979, -3483, 12962, -14708, 13853, 9960, -926, 2166, 5220, -15253, 8919, -6780, 1664, -16952, -5094, -835, 20956, -13580, -14898, 12250, 7193, 14528, 10047, -15400, -1372, 18458, 19442, 8302, 5765, -5213, 15389, 8175, -3329, -6450, 15933, -17684, -15401, 19062, -6138, 11551, -17840, 4737, 16976, -5410, -6622, -15991, -16113, 11051, 19626, -170, 18413, -5188, -16830, -1734, -2732, 8661, 2835, 6040, -1002, -2676, -3553, -593, 11217, -3945, 7421, -368, -7562, 7955, 13351, -6235, -13837, 6550, -2293, -8548, 1091, -10728, 6484, -1472, 3443, 5645, 4764, 9037, -10026, -7842, 725, -9173, 13283, 8911, 8151, 354, 8405, 7281, -4706, 3977, 10792, -12438, -12501, -3718, -2296, -7096, 6028, -1266, 4822, -10353, -11725, 5132, -7217, 6602, 11387, 10954, -5083, -4096, 8383, -6751, -4141, 10295, -53, 7267, -1459, -13417, -3399, 6864, -7847, 6292, 7675, -4894, 2563, -1487, -15219, 1550, 9708, -11436, 18173, -943, -12693, 11757, 11627, -7805, 8119, -5325, -15261, -1911, 14695, 7953, 16957, -4910, -11114, 10996, -4856, -692, 7384, 6638, 10590, 3970, 1156, -16047, -11177, -3774, 15666, 4389, 4929, -8491, 14161, -14745, -10710, -5172, -4617, -6065, -11198, -9888, 12707, -6936, -13731, 7638, -2384, -2126, -2772, 6357, 20122, -7240, 2233, 7823, -571, -7031, -10569, 4822, 1061, -14812, -9031, -15432, -2365, 11217, -14533, 9281, -1132, -12981, 15215, -800, -2819, 14369, 18033, -12256, 18565, 5122, -14507, -9990, 7129, -11440, 11719, -9990, -6113, -3356, -9038, 9515, 4974, -6768, 3073, 9690, 14279, 2851, -16001, 9124, 13516, -3270, -5635, -3950, 2864, -14600, -15730, -1131, -8351, 1775, -8448, 5740, 13452, -13168, -3609, -13762, -7220, 13239, 16527, 8229, 1418, -67, -11993, -5913, -668, -12086, 1448, 3478, -12118, -14657, -7282, -8378, -16958, 16800, 5088, 10499, 12685, -9574, -495, 8491, -10955, 4327, 11109, -17084, 10173, -2286, 3488, 8216, 17894, -2490, 20937, -9952, 6761, 11229, 5335, 7382, 11667, 7534, 18588, 7201, 3594, -689, 1819, 1444, 13863, -10668, 6011, 10180, 13811, 5130, 7427, 6149, -2018, 20998, 3170, -4215, -2399, 4725, -7425, -13724, -228, 12001, 10877, 5889, -4731, 6782, 6057, 14578, 16970, 20836, 22921, 13683, 17606, 20042, 10937, 7318, 12471, -6313, 5299, -14902, -4992, 21828, 3206, -1581, 13261, -2086, -11053, 9251, -9735, 9599, 23788, 22730, -14867, 12381, -3939, 8910, 19299, -21391, 3155, -19514, -18899, 17702, -17652, -20689, -18438, 23272, 26097, -4252, -8064, 28052, 19756, 30110, -16492, 2605, 7612, -7942, 14160, -19537, 26128, 7183, -8632, 25041, 23696, 19150, 12342, 2063, -5524, 26829, -14179, 16174, 18923, -12979, -9756, -1424, -426, 914, -41, -14400, 4284, -9094, 4878, 5240, -10135, 21940, 19815, -15677, 10060, 7089, -16471, 5102, 14216, -13982, -2525, -11593, 4280, -8232, 10181, -180, 9352, 94, -18014, 1009, 5141, 16019, 12215, -9452, 19775, 23367, 21858, -15375, -3952, 13485, -3375, -11108, -13024, -12779, -157, -18865, -18009, 18760, 837, 7622, -3399, 4966, -902, -13158, -7367, -11368, -12894, 5706, 2960, -12225, 7382, -1807, -6975, 2458, -1383, 2099, 7233, -9732, -11292, -4558, 566, -7699, -3454, 1506, -17442, 19545, 8907, -9788, 13877, -3393, -20958, 1827, 8134, -5261, -3815, -14367, -1707, -9289, -3687, -16305, 9654, 7009, 3158, 6103, -7197, 21996, 21894, -9697, 14835, 20672, 14587, -14218, -14952, -2017, 39, -10756, -9992, -18255, -3066, -2586, -14506, -3861, -2347, 19096, -7730, -20882, 15161, 2962, -13722, -10094, -6019, 15875, 2661, -14177, 13146, 5928, -3973, 3197, -10557, -10220, -1133, -10435, -12522, 9063, 5716, -12789, 2625, 1723, 1209, -4412, -993, -5413, -3385, -11707, 4531, -4302, 4771, 1545, 8392, 1285, 11792, -9060, 7384, -1522, 2033, -6125, 12022, -8416, 6708, -11117, -2304, -919, -12341, -12109, -6421, 11778, 11196, 3225, 10904, -11232, -3497, 7437, 9233, -2385, -8333, -9239, 9840, 9794, 5592, 12243, -6642, 12089, -12394, -9563, 519, 7031, 9437, 2381, 1901, 13146, 3285, -14641, -275, 4479, -12672, -15790, -9943, -5773, 2531, 5487, -10435, 17101, 8842, -5143, 19148, 731, 12618, 18438, -13176, 18294, 69, -8094, 7845, -8303, -6533, 276, -2700, 762, 15216, 964, 7020, 17260, 4194, 6295, 13677, 542, 14686, 702, 6205, -6382, -14511, 1414, 8371, -4883, 2438, 3677, -6168, -12592, -11803, -1900, -6617, 3674, -10375, -7897, -2342, -10428, 2585, 3353, 4308, 8509, 11609, 400, 18943, -6560, -443, 6866, -1121, -11778, 5154, -9993, 4806, 5087, -7715, -5820, -12717, -1441, -8361, 2024, -13011, 10625, -2264, -4994, 4853, -1530, 9399, 2610, 11419, 12127, -1061, 5598, -11680, -7745, -9833, 8708, -2978, -7058, -4954, 8433, -8086, 11014, 47, 5843, -3881, 9992, -1238, 2892, 4406, 11700, 4265, 6160, -6474, -10059, 5021, -6674, -8177, 13016, -11198, 4684, -4625, -2222, 5251, 8023, -12891, -12101, -6875, 11528, 9121, 12057, -11980, -17849, 4127, -5516, -13160, -6835, -17158, -6915, -4249, -3956, -947, 1400, -6250, -1294, 16688, 1145, 7940, 11434, 2042, 16559, 14098, -6896, 12665, 3208, -6774, -18902, -5417, -7678, 10188, 2613, -13986, 19821, -10930, 8063, 15916, -11732, 17957, 15976, 13029, 1863, -9868, -2460, 11943, -15939, -453, -1022, 11576, -11091, 219, 16014, 5822, 5006, -17224, -14806, -4849, -9054, -13356, -12396, -4667, -947, 7528, -124, 16821, -1224, -22281, 4439, -451, 2418, 4010, -10411, -12994, -1630, 6981, -5777, -8097, -4942, 1226, 3304, 2329, 4183, 2652, 1085, -7495, 8614, 13888, 11673, -8479, 4774, -4341, 775, 1007, -13829, -3789, -1331, 10826, 5433, 2447, -9117, -3336, 11343, -1745, -6954, 14033, -3540, -6376, -1083, -3918, 11468, -2029, -2251, 5155, 1183, 3358, 13245, 4399, -222, 3545, -4656, 2152, 7014, 700, 11807, 7029, 6158, -2410, -6109, 166, -10028, 3742, 6534, 3832, -3995, -16658, -16903, -4770, 6777, -2286, -14094, 6664, 9943, -5660, 8065, 11908, -1148, 12265, 1891, -14241, 2891, -3904, 5031, 18853, 7761, -2813, -582, 12869, 7336, 16323, -4994, 1159, 1293, 168, -2535, 15539, 3949, 17855, 4788, 12480, -9120, 2601, 8747, 17024, 4828, 1789, 8473, 19964, -3357, -2279, 18702, -20440, 16503, 1329, -15741, -4588, -14329, -2906, 8494, -7095, 15105, 5196, -1196, 17117, -14325, -18361, -10510, 12586, -5906, 5619, -10859, 5348, -7433, -8480, -10661, 1594, 2883, 8645, -6432, -12897, -7917, 7521, 7037, -8289, 8715, -4279, -1070, -5246, -2484, -2479, -9806, -8390, 9094, 6015, -2592, -654, -12877, -11013, -2730, -3683, 4057, 816, 3662, -5607, -11674, -10585, 1158, 5822, -11517, 5458, -5553, 8847, -6713, -12254, 7596, -3697, 5013, -9461, 10535, 1575, -1484, -1029, -9724, 5781, 8404, 3614, -7334, -10252, -9177, 4920, 16923, 32355, 6723, 20407, 14435, 21692, 10813, 24356, 30215, 14578, -7196, 6456, 1684, -16339, 20488, -13808, -23342, 10757, -24455, -9215, 3257, 707, 17463, 11485, 27395, -18039, 27653, -18454, 18819, 18849, -6467, 3011, -13476, -8782, 25462, -19687, -10687, -17647, 9496, 9693, -16310, -1221, 10915, 29499, 25351, -20482, 16142, 22796, -6190, 10713, -20243, 31456, 27799, -3833, 9330, 25828, 5532, 17397, -15257, -10823, 24169, -2976, 24378, 9744, -8944, -9078, -4270, -4851, -5248, 6429, 5483, 4704, -5624, -7927, 15082, -6280, -5686, 19787, -11076, 20029, 11290, 6690, 19682, -596, 5531, 8130, -2992, 628, -9640, 5760, -7371, 3675, -4617, -11694, 2052, -2513, -1117, 18673, -14838, 18222, -2569, 1442, 197, -14262, 13460, 6444, -17106, 608, -8461, 17409, 3784, -11791, 1645, -3428, 10438, -6598, -1953, 1114, -15728, -6782, 8342, 4025, 19433, 9430, -14182, 5492, -11711, -133, -754, -277, -8637, 4033, 4043, -8831, -11222, 6122, 3112, -2111, 8393, 5740, 18562, 7130, -10380, 5839, -3747, -18108, 6735, 17690, 6150, 12346, -10121, -9570, -15824, -4347, -469, 9643, -7691, 8685, 13257, -6276, 5603, 7251, -1099, -2943, -1616, 20375, -5490, -7127, -730, 11437, 2451, -4130, -16869, 18343, -873, -13303, 14638, -2007, 3364, -4256, 6923, -5625, -15843, -536, -2291, -2012, 11572, 1623, -1621, 18040, -6223, -17729, 3650, 5912, -5417, -6242, -10896, -3898, -9841, -12267, -3138, -15164, 14169, 7406, 3269, -9325, -11845, -9403, -1540, -9431, -9787, -6458, -9446, -7502, -15100, -4026, 3072, -9158, 4542, 11924, 1241, 8674, 5273, -9119, 1769, 1289, -3080, 3414, -7719, 14476, -2892, -4374, 10223, 168, 382, -4327, 516, 7801, -13115, -350, -6968, 9714, 8491, 718, -361, -14227, 2007, 10171, -11600, -12320, -7451, -2581, 6416, 8559, -3869, -13820, 2945, 1932, -16223, -7757, -15515, -7181, -15666, -1027, -10659, -18, 12604, -15116, 20706, 21083, -5720, 2997, 2345, 2159, 18818, 18898, 2899, -1034, 9807, -4347, 1684, 7153, -14135, 20061, -6158, 1374, 9954, -16736, 15477, 3347, -12559, 14319, 13141, 13109, -5081, 1241, -2819, 3647, -4858, -16740, -13475, 12197, 5500, -5374, -152, -6448, 6342, -517, -14736, 14329, 5436, 2154, 7221, -2253, 1637, 9741, 2683, -2498, -8595, -3626, -11031, 6905, -2540, -6201, 2319, -59, -6773, -10058, -4044, -18335, 15334, -2504, 12337, 16822, -17450, 19804, 5565, -16933, 8127, 19825, 4148, 9957, -2760, -1988, -16585, 13215, -7241, 19681, -12825, -7915, -4538, -1520, 20218, -687, -9077, 10348, 7548, 6122, -15677, 597, 15543, -5989, -14650, 5076, -1433, 18645, -8556, -3923, 4865, 631, -4115, 1118, -14545, 9773, -13191, -14271, -15783, 5932, -3945, 5182, 6792, 7271, 3144, -10196, 3903, -10370, 4120, -8682, -11107, -11379, -3172, 8411, 190, -5117, 13235, 2677, -7899, 294, -10559, -11816, -2464, -5999, -1778, -1832, 5405, 6314, 1671, 10126, 11537, 606, -7142, 4596, -6956, -6783, -2321, 9203, -4984, -4271, 10224, -11956, 11443, -10677, 10813, -4410, 3613, 10101, 5617, 5708, 8411, -3034, 840, 10425, -10416, -9598, 863, 4347, 11485, 10094, 6402, -12504, 5537, -6377, -10309, -2501, 6574, 7599, 9608, 3269, -7344, 375, 4028, 11424, -2608, -9125, 6567, 5681, 11983, -1142, -9062, -10845, 8846, 9185, 11101, 3717, -9434, 11238, -1518, -7482, 9481, -9676, 9230, -7576, -3347, -438, -1105, 11523, -1213, -6538, -12070, 10819, -10032, -3382, 11305, -9162, 4200, 709, 1138, -12122, -8686, 11456, -6753, 8899, -8262, -11945, 6133, -4435, 1445, -13344, -299, -10752, 9928, 2702, 3617, 8384, 2403, -6059, 2584, 7905, -4177, -3043, -562, 4551, -12078, -7132, -2213, -16082, -4174, -893, -1938, 457, -9044, 9124, 9192, 235, -3414, 17912, -4416, 16841, -810, 2302, 19881, -3226, -4453, -3870, -6564, -15054, 10130, 4880, 9327, 536, -13605, -12681, 12298, -158, 13289, 10977, 4568, -3921, 18908, 18640, -1246, -4246, 1651, 5969, 10327, -13327, 1799, 985, -15164, -3156, 19383, 7121, 3098, -10107, -7525, -9263, 3639, 10022, 5928, -12839, 10732, 17365, -11878, 2650, 1175, 6780, -14038, 7802, 593, -6910, 9063, -5799, 4042, -2396, -13669, -3234, 14349, -9080, -2149, 16764, 1231, -3807, 18559, -1252, 17462, -108, -4648, 10805, -8905, -12469, 267, 12147, -665, 19622, -11595, 4026, -3159, 3487, 7652, 10180, -9054, -1064, -6134, -4537, 2397, -4638, 3857, 2953, 3386, -6245, 251, 15769, -1564, -15969, -1141, 3225, 11389, 4669, 7407, 2556, -11556, 537, 10406, 8114, -2884, 13040, -5824, -5079, -14640, -1829
};

static TinyTensor inner_fc3_weight_t = {"inner_fc3_weight_t", inner_fc3_weight_t_data, TINY_DTYPE_INT16, 2, {64, 64, 1, 1}, 4096};

static int32_t inner_fc3_bias_data[64] __attribute__((section(".data"))) = {
    -188012336, 82871720, 21039334, 91570384, 29765662, -77325704, 28586268, -96789288, 36476844, -111916736, -58995248, 104828968, 95226408, 197361536, 28447680, 203790192, 281179264, 3737236, 94765192, -49971540, -37691544, 155223328, -220748768, 35797460, 4255052, 207762560, -190152784, 19521278, 88582184, 198669536, -56329008, 6724672, -78239008, 1431036, 113832440, 124513016, 97854048, -74233896, -10992419, -47784044, 178327408, 150191984, -105991480, -31138014, -74792128, -76763944, 147061344, 101436512, -24578788, -1829596, 283555232, -43515276, -160728496, 157219248, 131361984, 175711744, -189093552, 148471104, 198879808, 139477744, -106224224, 170221552, -86272064, 269726240
};

static TinyTensor inner_fc3_bias = {"inner_fc3_bias", inner_fc3_bias_data, TINY_DTYPE_INT32, 2, {1, 64, 1, 1}, 64};

static int32_t inner_fc3_data[64] __attribute__((section(".noinit")));

static TinyTensor inner_fc3 = {"inner_fc3", inner_fc3_data, TINY_DTYPE_INT16, 2, {1, 64, 1, 1}, 64};

static int32_t gelu_data[64] __attribute__((section(".noinit")));

static TinyTensor gelu = {"gelu", gelu_data, TINY_DTYPE_INT16, 2, {1, 64, 1, 1}, 64};

static int32_t inner_fc4_weight_t_data[64] __attribute__((section(".data"))) = {
    14707, 21232, 6302, 17320, 22030, 21960, 24475, 23648, 11217, 25196, -21405, 20946, -24838, -21457, 15921, -26489, -25428, 18141, -14158, -16183, 20168, -24591, 19179, 16817, 18881, -25622, 10193, -15288, 23744, 19867, -16028, 23796, -21156, -14343, 11878, -19102, -13407, -25250, 24805, 22862, -22922, -10300, 28784, 17936, 32767, -20362, 13766, 18207, -18623, 14539, -29984, 26951, 13189, -4554, 16644, 14195, 22127, 10016, -26281, -22363, 25789, -17863, 18800, 12066
};

static TinyTensor inner_fc4_weight_t = {"inner_fc4_weight_t", inner_fc4_weight_t_data, TINY_DTYPE_INT16, 2, {64, 1, 1, 1}, 64};

static int32_t inner_fc4_bias_data[1] __attribute__((section(".data"))) = {
    -44765060
};

static TinyTensor inner_fc4_bias = {"inner_fc4_bias", inner_fc4_bias_data, TINY_DTYPE_INT32, 2, {1, 1, 1, 1}, 1};

static int32_t inner_fc4_data[1] __attribute__((section(".noinit")));

static TinyTensor inner_fc4 = {"inner_fc4", inner_fc4_data, TINY_DTYPE_INT16, 2, {1, 1, 1, 1}, 1};

static int32_t sigmoid_data[1] __attribute__((section(".noinit")));

static TinyTensor sigmoid = {"sigmoid", sigmoid_data, TINY_DTYPE_INT16, 2, {1, 1, 1, 1}, 1};

static float dq_out_data[1] __attribute__((section(".noinit")));

static TinyTensor dq_out = {"dq_out", dq_out_data, TINY_DTYPE_FLOAT32, 2, {1, 1, 1, 1}, 1};

static float dq_out_expected_data[1] __attribute__((section(".data"))) = {
    0.13794365525245667f
};

static TinyTensor dq_out_expected = {"dq_out", dq_out_expected_data, TINY_DTYPE_FLOAT32, 2, {1, 1, 1, 1}, 1};

static uint32_t tinynpu_static_ub_image[1650][TINY_BUFFER_WORDS_32] __attribute__((section(".data"))) = {
    {0x1c9cf9e0u, 0x03668288u, 0x15526620u, 0xf7af4520u},
    {0x20990ec0u, 0x1b658ac0u, 0xe8f1bee0u, 0x0caf6f30u},
    {0xf65eed40u, 0xfbfb8708u, 0x139b4ce0u, 0x0a65f510u},
    {0xfc277818u, 0xfe7cd644u, 0xe913fb20u, 0x05d5f348u},
    {0xe6611a00u, 0xff7069ceu, 0x11f0e380u, 0x142ba880u},
    {0x030d7df0u, 0x01a1ae8au, 0x1965c6c0u, 0x094beef0u},
    {0x16fb7e40u, 0x08498aa0u, 0xf08d56f0u, 0x1821e720u},
    {0x110aadc0u, 0x1cff9f60u, 0x1be08f40u, 0xf7dfbd60u},
    {0x184f5f00u, 0x1c54a500u, 0x063c69e0u, 0xf59036f0u},
    {0x05abff20u, 0x0e51e790u, 0xfa756bb8u, 0x1539de40u},
    {0x14c62420u, 0xf0901d90u, 0x0d92f660u, 0x14a74740u},
    {0xeffffde0u, 0xf82e31d0u, 0x171d2da0u, 0x1c8cf3c0u},
    {0x19106a00u, 0x0444ec98u, 0x03201658u, 0x0f2a9e50u},
    {0x18e3da40u, 0x109442a0u, 0x1b464a20u, 0x19a4f460u},
    {0x1fe09b40u, 0x1262e440u, 0x05c99e78u, 0xff815ad6u},
    {0x00c710cdu, 0x11158c60u, 0x1ead9320u, 0xebdbd580u},
    {0x3499dc7cu, 0x2fe6f2d9u, 0x3272276cu, 0xf96c2d03u},
    {0x47b63e77u, 0xfdb31200u, 0x064e154bu, 0x0f5b1585u},
    {0x0bd604fcu, 0xf66d02b3u, 0x3bf70a45u, 0x09ebf32au},
    {0x28731d73u, 0xefe319fdu, 0xfaa12338u, 0xebdb27fbu},
    {0x2a2d10eau, 0x584f2e3cu, 0x24d0ecf1u, 0xd54beebfu},
    {0x53261731u, 0x7fff5b98u, 0x171513fau, 0xff53d83bu},
    {0x43a02ecbu, 0x6fb33ecfu, 0x1f24e2d6u, 0x35a0d235u},
    {0x3b290034u, 0x1734217au, 0xfda60cc9u, 0x1d6e128au},
    {0xfbde388eu, 0xf9090b20u, 0xdf57f083u, 0xdad4163au},
    {0x352517b6u, 0xf42739acu, 0x2d88ed1du, 0x498006fau},
    {0x1598ffedu, 0xeeb5038fu, 0xf3fbff39u, 0xf973fffeu},
    {0x2cfe3dcfu, 0xfe0a4831u, 0xf7a63a4eu, 0x1425f219u},
    {0x640a3750u, 0xdee93d02u, 0x2add33cau, 0x4ae3dae1u},
    {0x2b72e2a8u, 0xccd449b3u, 0x30a73182u, 0x3ca9d93fu},
    {0x1841fd1au, 0x15a50e45u, 0x328e1b91u, 0x50cbddb0u},
    {0x2ea321a3u, 0xf7b70da9u, 0x1b6a1152u, 0x40defeb0u},
    {0x2211049cu, 0xe0a814a7u, 0xf4f800e2u, 0xea8efd0au},
    {0xf0361f02u, 0x3f47ea7bu, 0x1d202d9fu, 0xf0d2bdcbu},
    {0xdf761128u, 0xed292bfbu, 0x36ee0735u, 0x1310eca7u},
    {0xf98c23c5u, 0xf55c3047u, 0x286f471du, 0x1735b83au},
    {0x0c84d813u, 0x2db910e6u, 0x243d4cc2u, 0x1171c59cu},
    {0xd4d8f847u, 0x4910edceu, 0x083e1813u, 0x55a0f3fbu},
    {0xe0aeeeb3u, 0x4bd1dc62u, 0xdf97321au, 0x3b2cae14u},
    {0xdbacd7beu, 0x1d9edf2bu, 0x02420577u, 0x3362ed0cu},
    {0xfe81e26du, 0x01ed280du, 0xef602b9du, 0x1b2702cfu},
    {0xf6f53d88u, 0x4537fd01u, 0x019030e6u, 0xf3693ae5u},
    {0xff9bebadu, 0x18f4d1c5u, 0xf26342d1u, 0xe277440bu},
    {0xf521efb5u, 0x3a532261u, 0x2bb5ebecu, 0x17852e70u},
    {0x115ddf9au, 0x24b30addu, 0x176c0b32u, 0x15be37e3u},
    {0x2b28f287u, 0x5daee4adu, 0x5c7931b6u, 0x5f414d35u},
    {0x3c061d96u, 0x0e53e5a1u, 0x468303a7u, 0xf8ba0d3au},
    {0x368ff280u, 0x3aeb1fc0u, 0x4338319bu, 0x1caf1557u},
    {0x23340f97u, 0x1ddcdcdbu, 0xe0e10001u, 0xd8c3e111u},
    {0x1d5933a8u, 0x43700281u, 0xe85bf9d1u, 0x2d51e589u},
    {0x32e442b9u, 0x06634da6u, 0x1c4636feu, 0x368aed69u},
    {0x30ec1a67u, 0x384d2086u, 0x16bb0da5u, 0x3566e813u},
    {0x1af65a32u, 0x37aa21bau, 0x45424f1fu, 0x2c8ae2b9u},
    {0x5d2e35abu, 0x36da2140u, 0x27c16a59u, 0xf8b21281u},
    {0x57f60990u, 0x185e5b5cu, 0x12770cf6u, 0xfaf00b5fu},
    {0x28862559u, 0x4fef4e53u, 0x0d8b1781u, 0x03032b07u},
    {0x2f82dea2u, 0xfa9231a2u, 0xe470147cu, 0x218d2f78u},
    {0xe450f2e6u, 0x32051e05u, 0x01511abdu, 0x216e0334u},
    {0x03740736u, 0x08ce1e0du, 0xff2719bau, 0xfb49f98cu},
    {0x26c10188u, 0x247f08f7u, 0x0e9515c9u, 0x21a226d4u},
    {0xe52404ecu, 0x3aaa1751u, 0x1fa22c20u, 0x50bb3926u},
    {0x1335456fu, 0x2d191f5au, 0x222c2d98u, 0x296a31aau},
    {0xf6032704u, 0x3f2627e8u, 0x59213a9bu, 0x249535e5u},
    {0xdfed38f8u, 0x4ef13d7au, 0x1dc01a63u, 0x4d001f8bu},
    {0x2f3302edu, 0x2dd1303du, 0x1e12cdc0u, 0x2a9ae639u},
    {0x0356ef14u, 0x00af1709u, 0x33f6024fu, 0x0932fe63u},
    {0x30a1f3aau, 0xfccb1665u, 0x2032e74au, 0x51432939u},
    {0x0895476eu, 0xf3892998u, 0x1b332aadu, 0x4c1e27e5u},
    {0x0f8e10d8u, 0x2b3734d4u, 0x143bbf8bu, 0x3e34e1e8u},
    {0x3fa96737u, 0x28377160u, 0x2171aa6eu, 0x72dff70au},
    {0xff7f55afu, 0x15590450u, 0xf38cd126u, 0x4e140460u},
    {0x3392004cu, 0xeef71617u, 0xecd2c9e5u, 0x3e882259u},
    {0x33802402u, 0x1052fd13u, 0xf446d73bu, 0xf40beb29u},
    {0x17e72fa0u, 0x32b1336fu, 0x21e01e3eu, 0xf8f7358cu},
    {0xf7451c81u, 0x23bff7b7u, 0x32a0d6a9u, 0x137eecc6u},
    {0xf678311du, 0x3bb93d06u, 0x0059dc75u, 0xdaae1bc3u},
    {0x2278d66du, 0x453f0c61u, 0x421bfd59u, 0xd0bf296au},
    {0x3dc6e433u, 0x3f4a5163u, 0x2c32d7c5u, 0x000bfc54u},
    {0x215b0edfu, 0x08cafff0u, 0x428aecb7u, 0x12b5f9c2u},
    {0xf5c426b9u, 0x28fd1679u, 0x5043f542u, 0xeb2c227eu},
    {0x13ba0597u, 0x0fc50998u, 0x2625108bu, 0x2ce92e61u},
    {0xfb7c16afu, 0xe91919f6u, 0x366a00f9u, 0x2ddd15feu},
    {0x05ece7d4u, 0xd72e098eu, 0xf039f52du, 0x3e4df7f3u},
    {0x0766e76bu, 0xe29c0cffu, 0xcc7c13bdu, 0xea781801u},
    {0xe5b4d9f6u, 0x29dce7e9u, 0xfb1ceb43u, 0xe4a4e77du},
    {0xff3dec78u, 0x0f2a1fa3u, 0x0d7dd7e3u, 0x134fea84u},
    {0x274834f9u, 0x30fb2de1u, 0x13820907u, 0x4539e3eau},
    {0x283a628cu, 0x158e5338u, 0x1fad01c7u, 0x7f970665u},
    {0x4de84c3bu, 0x1d454392u, 0x04200e6bu, 0x29e81a2fu},
    {0x3881fbd2u, 0xdd4cf0f7u, 0x1a1ff2f7u, 0x246bf86eu},
    {0xdd4d3e1cu, 0x14852108u, 0x1e994222u, 0xf461df88u},
    {0xd70e47d5u, 0x213b2009u, 0xeb081fabu, 0x12830efcu},
    {0xe5db04a4u, 0xf411f374u, 0xece50241u, 0xfc1ee5f2u},
    {0x00c010f4u, 0x0eb8d795u, 0xe5f4051bu, 0xfa8213adu},
    {0x04a3daadu, 0x2de5e543u, 0x0e2be624u, 0x07230747u},
    {0x28db1ef5u, 0xe0b65331u, 0x60372645u, 0x3d72e907u},
    {0x2ad121dfu, 0x4a3ce7c5u, 0x37cf1776u, 0xfc4204d1u},
    {0x1b5ff968u, 0xe2cb0793u, 0x347a0287u, 0x2da30565u},
    {0x2014d3c3u, 0xff1738f3u, 0x29dd39eau, 0x3809110cu},
    {0x17acfe5du, 0x02d91a83u, 0x246adef9u, 0xe8b72c93u},
    {0xdd55d21au, 0xe1e2281au, 0xe36ce361u, 0x25c743c7u},
    {0x12e7e4b8u, 0x1a963f8eu, 0x29eb070fu, 0xe623408eu},
    {0xf458291cu, 0x2af7fb94u, 0x200415eeu, 0xeb89186cu},
    {0x152f28a9u, 0x465db8feu, 0x1faa0c13u, 0x581bf601u},
    {0x3fdb331cu, 0xfa04199au, 0x31e9190fu, 0x024a4752u},
    {0x04124b0au, 0xfeff18ecu, 0x1e0432e0u, 0xe58ef998u},
    {0x37aa3337u, 0x29c10891u, 0x20fbf77du, 0x25452452u},
    {0x2af51436u, 0x2d0417a4u, 0xe880d9b5u, 0x1710e1c4u},
    {0xdf562e5au, 0x106d1243u, 0x2819f13du, 0xe4450582u},
    {0x0280d1d3u, 0x31e9f635u, 0xd99505b5u, 0x1978e079u},
    {0xdd72e9bcu, 0x28031d1cu, 0xf5b23029u, 0xd743390cu},
    {0x072f03a6u, 0x6b20e4f8u, 0x282625b3u, 0x3a721b9cu},
    {0x08ae097du, 0x541636dfu, 0xfa8901f4u, 0xf82edddbu},
    {0x2ce82d96u, 0x164736e8u, 0x46f8f677u, 0x4001d5ceu},
    {0xe22f35aau, 0xf3014558u, 0xeff6304au, 0x03322a65u},
    {0xefe1d6adu, 0x328b18f0u, 0xf024e90fu, 0x0c2cfd8bu},
    {0xcdced76cu, 0x23822b1bu, 0x0e5d2a83u, 0x092f0ae3u},
    {0xe207e2b7u, 0x3afd19d7u, 0x28e828d4u, 0xc0110dbdu},
    {0x0a34d3dbu, 0xf5652e97u, 0x1ba501c2u, 0x263a286du},
    {0x5d871867u, 0x3e303d8eu, 0x10b22ee0u, 0xf8491e2du},
    {0x32f80c25u, 0x2d50036cu, 0x32093707u, 0x47cdedafu},
    {0x235f0520u, 0x09750aebu, 0x19c0feccu, 0x1d471532u},
    {0x19e8e0a1u, 0x0c6937d4u, 0xf75c3e25u, 0xfe33f178u},
    {0xf6b7c680u, 0x18dc1d84u, 0x38802915u, 0xf0a0063bu},
    {0x1f48fba4u, 0x1912f79au, 0xe88c46c2u, 0x28e7f828u},
    {0x06ca3246u, 0xf7960275u, 0x2f1f14e0u, 0x21c61c59u},
    {0xeebd201eu, 0x3013e035u, 0xec7c11d2u, 0xfc5ad9e9u},
    {0xe8c07ed5u, 0x2f9513f9u, 0x1efb1365u, 0x1a7a42e2u},
    {0xeeec02a8u, 0xf25a3dbfu, 0xf597f085u, 0x228d1ce4u},
    {0x014af238u, 0xe919f433u, 0xf9170d80u, 0x0e3208cfu},
    {0xeddd0296u, 0x2a4616f0u, 0x28a22073u, 0x43d3ff76u},
    {0xeaea1ca3u, 0x11f63640u, 0x1ac1197cu, 0xfb4913f9u},
    {0xee401885u, 0xfc09e6aau, 0xe64b3204u, 0x130ffdfcu},
    {0x156ae960u, 0x1afc0b14u, 0x2225e71eu, 0xe5a9ee96u},
    {0xef2d0d3du, 0xeffd2f2bu, 0xf216203du, 0x1b2f2cf6u},
    {0x57026692u, 0x390b62f2u, 0x3947b7bau, 0x3416f348u},
    {0x120cf0e9u, 0x43c72e0bu, 0x43f802f8u, 0x02c9f38fu},
    {0x2007fe3au, 0x20022ba7u, 0xfe2cd9deu, 0x09681476u},
    {0xf980fa17u, 0xf3ed3245u, 0x05712c2cu, 0xddb52223u},
    {0x1c9b1110u, 0x2a471ebcu, 0xf8561837u, 0xff5c0540u},
    {0xee9be620u, 0xfe42ffd3u, 0xe90fec27u, 0xd7171208u},
    {0xe49e0368u, 0x1828cdc8u, 0xf4ebf5e0u, 0x086fdae3u},
    {0xff97101eu, 0x27d9f9b7u, 0x2647e893u, 0xeb972879u},
    {0x1b12362cu, 0x47281b12u, 0x5f60dae7u, 0xdd68299bu},
    {0x172d3353u, 0xdb613033u, 0x08111e74u, 0x3cc608f7u},
    {0x44b728acu, 0x017cecf8u, 0x27d41bceu, 0x0f32ed18u},
    {0x47b72f85u, 0xee3b152cu, 0x26e12400u, 0x12680f8au},
    {0xf74be34du, 0x0a02eb0eu, 0xe3292045u, 0x083be209u},
    {0xe99def1du, 0x113ceaeau, 0xeeb82be8u, 0x22b1d86cu},
    {0xdc1713ebu, 0xbb20e147u, 0xe62c2130u, 0xfb1c02d6u},
    {0x0cb80db1u, 0xcf79ffa6u, 0x2a8d1b54u, 0x1a08f361u},
    {0x501620a2u, 0x052c2761u, 0x166fd04fu, 0x6c8dfdccu},
    {0x35c30c04u, 0xfd2a24cbu, 0x2dec3b3eu, 0x47f40c99u},
    {0x3e842cb1u, 0x20defd70u, 0x119733f5u, 0xf96118dcu},
    {0xec95fd18u, 0x3775f25du, 0x267e388fu, 0x1888efabu},
    {0xf53dfa47u, 0x3427f876u, 0xe58fd789u, 0x1e06f8d6u},
    {0xec17fc61u, 0x3467160du, 0x18720b90u, 0x19750ab3u},
    {0x10a73039u, 0x52950726u, 0x1ed7dbbcu, 0xd4a40d35u},
    {0xd46c35feu, 0x0df8143bu, 0x349138f4u, 0x0134e0d8u},
    {0x1aa9497fu, 0x13e0318fu, 0x2fb117f6u, 0x244b0107u},
    {0xf8ac11a9u, 0xf9c7045fu, 0x1e992fc0u, 0x2277c726u},
    {0xd25808d8u, 0x045d38b7u, 0x2c52fecdu, 0xf87afadeu},
    {0xcd80067au, 0xfc1c22bbu, 0x36c63f76u, 0xfff4ff6bu},
    {0xdf5d0b1cu, 0xfb34025au, 0x1a0d11cfu, 0x040ffb53u},
    {0xe11ecd32u, 0xdb024e35u, 0xf6d1e06au, 0xf6d439e4u},
    {0xe274f700u, 0x14f3549cu, 0x004207eeu, 0xbb1c094au},
    {0x09c1fd17u, 0xf3b5180du, 0x33d1151du, 0xe3942e75u},
    {0x2c47f0d4u, 0x065b031cu, 0xfab108dau, 0x22d9189fu},
    {0x1b9130a5u, 0x5609d5e4u, 0x13a30d3eu, 0x34cb1d7cu},
    {0x1d3b026eu, 0xf462e20au, 0xff230c84u, 0x08fdff4du},
    {0xd8cf33b6u, 0x1ec20164u, 0x0a441dceu, 0xf0f00490u},
    {0x2d6cf738u, 0x078b07d5u, 0x10a1ed70u, 0x01980fc2u},
    {0xd6dee6a4u, 0xdafdeaf9u, 0xf6ddcf97u, 0xba6c13d4u},
    {0xcffbeda3u, 0xd674e36cu, 0x0121161au, 0x1babcea1u},
    {0xd0d3218eu, 0x0694f358u, 0xd745ef12u, 0x13dfeda3u},
    {0x270de6c8u, 0x49d2ecb7u, 0x3f110d83u, 0xfc112264u},
    {0xf9b1f25eu, 0x47353d67u, 0x20ab2a77u, 0xf57529b7u},
    {0xfc750522u, 0x132f0836u, 0xf5ad2d08u, 0xf3ebfa70u},
    {0x1df4edd6u, 0x0d052ec2u, 0x2927087cu, 0x2866fea4u},
    {0xf871faecu, 0x057df6ddu, 0xf271106bu, 0xdf35ee15u},
    {0xcda8f7f0u, 0xda2b0617u, 0x1794d49du, 0xe25cf139u},
    {0xee2c199cu, 0x094eeff1u, 0xf7f8e319u, 0xf1bed9c5u},
    {0xed4f1dd0u, 0x09cdf29au, 0xde2d214bu, 0x257503ecu},
    {0x3f433e6fu, 0x197a46ceu, 0x395745d5u, 0x0258e3fau},
    {0xfa761127u, 0x44370ab0u, 0x51761496u, 0x22a52748u},
    {0x109e2632u, 0x3e901931u, 0x277eebe6u, 0x1c5a260eu},
    {0xd7bf0914u, 0x0b21e3edu, 0xff0df93bu, 0x2e2c23b4u},
    {0x25dce952u, 0xdf841581u, 0x3679dc4fu, 0xe1a00968u},
    {0xfeea0ffau, 0xdfe703a7u, 0xe126f310u, 0x104a202bu},
    {0xf135f3beu, 0xd3f3d44du, 0x103b1424u, 0xd9be1781u},
    {0xc7930b75u, 0xfe3c2373u, 0x10c0bf58u, 0x0d01e556u},
    {0xc4325db3u, 0x31e04feeu, 0x469e0a6eu, 0x2bf108bfu},
    {0x0b2145f7u, 0x09d50a74u, 0x09c4ecdbu, 0xfcf83033u},
    {0x0c6e1184u, 0x069943beu, 0xecd2dd87u, 0x1bdc2195u},
    {0xe21d209bu, 0xffa2ffefu, 0x363b3c50u, 0x16e3ff85u},
    {0x257f030bu, 0x1716f9ffu, 0xedd43d12u, 0x3708fb41u},
    {0xd3c5e4dcu, 0xc2d6ee02u, 0x2c8b18b0u, 0xe1d63d34u},
    {0xec5d2109u, 0xc044cdbdu, 0x0a063330u, 0xc4d3f507u},
    {0x220ff52cu, 0xedb3083eu, 0x0e3d3186u, 0xf545e7e5u},
    {0x34c405ddu, 0x0b854b10u, 0xfb01f04eu, 0x375ffb8au},
    {0xed4f2bb6u, 0x2fc83fc8u, 0x16450984u, 0xe1e80f2fu},
    {0x066cfce0u, 0x225020c6u, 0x14f7f565u, 0x0559ec92u},
    {0x0aae3994u, 0x32badeb7u, 0xf3a82e4eu, 0xd3a8da8au},
    {0x126de43cu, 0xea18fc16u, 0xddd5fc4fu, 0x05891d79u},
    {0xf3530bbbu, 0xfccd0cf9u, 0xf5611029u, 0x1b16fedbu},
    {0x23012e9bu, 0xc8c21029u, 0xff6f4345u, 0xd5c12459u},
    {0x06c605beu, 0x16510ef3u, 0x10190297u, 0x041e2499u},
    {0xe969f437u, 0x396a2230u, 0x0781bb9bu, 0x220d040fu},
    {0x4dec18f6u, 0xecb1f832u, 0x52bc18f5u, 0x1a88eb5eu},
    {0xfbecf0aeu, 0x216f1217u, 0x38de2fb7u, 0xed921e20u},
    {0x14a9f8d6u, 0xd586eb20u, 0xe858e4c8u, 0x02d9efa5u},
    {0x2518ff0fu, 0x0349fc59u, 0x13dafaabu, 0x0103fab0u},
    {0x27d8520cu, 0x02520e08u, 0x021e378bu, 0x182018b6u},
    {0x13832324u, 0xc9120863u, 0x13b33318u, 0xdf05ce33u},
    {0xe433dd65u, 0x0f8fec52u, 0xdf25fe34u, 0x08342a6cu},
    {0x08fd0a50u, 0xdbff0932u, 0xc5bcfd20u, 0xf8d71ba7u},
    {0x3cfcfcbbu, 0x0f891d6fu, 0x0e021061u, 0x3087dee5u},
    {0x1010b38du, 0x0060010du, 0x285303b9u, 0xff312ac2u},
    {0x37ffe4b5u, 0x0c2c2330u, 0xdf062661u, 0xe78cd9c3u},
    {0xe1a9c054u, 0x1afc129du, 0xda6908c3u, 0x2788d214u},
    {0x20200520u, 0x0f1134dbu, 0xfb0dfeeeu, 0x419d02c4u},
    {0x01e32651u, 0x1215f7e4u, 0x263b16e6u, 0x20e9db08u},
    {0x0648c3f1u, 0x3319f2d1u, 0x295323eau, 0x0c44d480u},
    {0x0fe1f8f9u, 0x2bd6f643u, 0xee480285u, 0xddf3ddacu},
    {0xd76ddc85u, 0x4a140653u, 0xe4b74162u, 0xfa600325u},
    {0xec34fbc9u, 0x01441f19u, 0x238e0352u, 0x1d873562u},
    {0xe79bd3e5u, 0x1bc6041bu, 0x20c63365u, 0x1a3b0acfu},
    {0x0e87ce80u, 0xe96f0c1fu, 0x12062b3fu, 0xe9f7fa54u},
    {0x14d7e763u, 0x2b03d137u, 0x1ee942edu, 0x0f8ea333u},
    {0xf5640963u, 0xf9d51561u, 0x0285ccedu, 0x05463e46u},
    {0x2924f61du, 0xc2cb0892u, 0x145ff10du, 0xe91647b3u},
    {0xefb5178cu, 0xdae5f7d5u, 0x0e2bd13fu, 0xeb6f14cbu},
    {0x46e2290eu, 0x0c290bb5u, 0xfa3d4c78u, 0xf97ffa2fu},
    {0x1491146cu, 0xc7cf06d2u, 0x1a1df252u, 0x1f412febu},
    {0xcfef0952u, 0xf07e0ea8u, 0x0c53f4e7u, 0xd289fad9u},
    {0xecace9ffu, 0x00d6077du, 0xed520eafu, 0x03891714u},
    {0x44120891u, 0x16ebdef5u, 0x453a3a63u, 0x30a806feu},
    {0x2030eed9u, 0xea00ce46u, 0xdbb0ec6bu, 0x255ad7e2u},
    {0x1dad0212u, 0xf12ed398u, 0x198a0a5eu, 0xbd130ffdu},
    {0x27aaf7ebu, 0x0aff1df1u, 0x0c31eda8u, 0x06e3ef86u},
    {0x427f2c34u, 0x023a035eu, 0x25b2270au, 0x50712535u},
    {0x025d206eu, 0xd324eef4u, 0x09ebe145u, 0x18f4dc5du},
    {0x1ec2ee17u, 0x00f02a8au, 0xf339f4a9u, 0x1eedee4du},
    {0xd4d3effcu, 0x22e6f2ccu, 0xfde1172au, 0xf37ed83cu},
    {0xf6013c5fu, 0x447c5531u, 0x0ce91849u, 0x1d7d0c68u},
    {0xe3141551u, 0xfbfdf084u, 0x230bf97au, 0x1fbbe97fu},
    {0xdd32f23fu, 0xdb9eef72u, 0x001bcc7du, 0xefb4ec76u},
    {0x3918c69bu, 0x20eadb2cu, 0xf0e5dc7cu, 0x12e625fbu},
    {0x0e675638u, 0x298752c7u, 0x4b97e46eu, 0x129f4524u},
    {0xc4732577u, 0xe60ff61au, 0x2169f663u, 0xf4b1f095u},
    {0x198424b2u, 0x13fad62au, 0x199fdbe7u, 0xee16288au},
    {0x44cff314u, 0x0493f235u, 0x06840049u, 0x1b00fdddu},
    {0xfce425ceu, 0x28f531e2u, 0x23b83bc1u, 0x01564588u},
    {0xf015c6a8u, 0xe0bb1b68u, 0x2b2df491u, 0x01bd1c9au},
    {0xb671c6abu, 0x210a18dau, 0x0b50fa1au, 0xc64bc7f9u},
    {0xc3dcc6c1u, 0x18eb09fcu, 0xffe0be80u, 0xc948cdf3u},
    {0x3b51564bu, 0x25c80c15u, 0xf66be238u, 0x137000b2u},
    {0xf2ef12aau, 0xd6870426u, 0xfe713e65u, 0xd2e5fae3u},
    {0x0956f59du, 0xcb0e0754u, 0x0b2618f6u, 0x00460414u},
    {0xb75b2181u, 0xe6be2995u, 0x106b4ac5u, 0x1c763b09u},
    {0xe8851df9u, 0x3a9437eau, 0x4618ef9cu, 0x31d20d15u},
    {0xdaa1d56eu, 0xe7d51056u, 0xe8ea023eu, 0x1144f62eu},
    {0xde68d66cu, 0xd636be86u, 0xed90f6c8u, 0xdbb7f58cu},
    {0x287eef23u, 0x05d21629u, 0xbc1af790u, 0xed19ff99u},
    {0x026de999u, 0x21aaf725u, 0x517adcfbu, 0x00a11f9cu},
    {0x1f512819u, 0xdcbafc78u, 0x25cc23f6u, 0x187ff6afu},
    {0x27de2a3eu, 0xdf3c27acu, 0x0b003b1cu, 0xf48b2328u},
    {0xfae41972u, 0xf2f1d039u, 0x138f3a15u, 0x0faa22e8u},
    {0x408a28d6u, 0x3eebfad9u, 0x1409ef61u, 0x142639c9u},
    {0xf120fbb2u, 0x2189df0fu, 0x02d90497u, 0xf0e402e5u},
    {0xe208c694u, 0x0455069bu, 0xc1b62fd5u, 0xd1cb103bu},
    {0xf2b1cd66u, 0xd36a1981u, 0xf2e8efbcu, 0x1929055du},
    {0x2fab0d30u, 0x11a04b25u, 0x3618d046u, 0x145301d1u},
    {0xdc0ddd15u, 0xdb460e9fu, 0xdf42146bu, 0xf6f40d61u},
    {0xcd24d38bu, 0x2a12f778u, 0x0d6b07b1u, 0xe6c8155cu},
    {0x3ed7f1d4u, 0x316bf303u, 0x01851204u, 0xfec7d9d6u},
    {0x0f2a4d01u, 0xf54349ecu, 0x00fbefb7u, 0x0392fcd6u},
    {0x3747d309u, 0x1a1508e3u, 0x03a31efbu, 0xdfa8e301u},
    {0x06bf0482u, 0x364c0ad3u, 0xc7740142u, 0xb7400034u},
    {0x2ab4c5cdu, 0xe8231650u, 0xdff1c744u, 0x0dd4e198u},
    {0x03b636f0u, 0x1cbf03fbu, 0x01eb0b52u, 0x0e08e41au},
    {0x322dbdbdu, 0x19efd2f1u, 0xe488e0ecu, 0xdd971cc0u},
    {0xf8bef5e3u, 0x5409dcfeu, 0x0bad02fbu, 0xbd84dd81u},
    {0x0bd0484fu, 0xfee01ba5u, 0x3b412e59u, 0x2b4814a1u},
    {0x3e571510u, 0xf7a53713u, 0x3a270a78u, 0x35b31c61u},
    {0x38dec732u, 0x31bd1aecu, 0xe1f6e768u, 0xdf00dc94u},
    {0xf4e3e0c3u, 0x07cbf517u, 0x0476d132u, 0x02981adfu},
    {0x0d2edadbu, 0x05f5deecu, 0xc8dde3cbu, 0x2964fcb1u},
    {0x14c1e8b4u, 0x066012e4u, 0x1e410473u, 0xeefc2381u},
    {0xd1d9068bu, 0x1ecd2663u, 0x1f13eefdu, 0x15572abfu},
    {0xd77cf11du, 0xcb9ffd3cu, 0xe9cfb93cu, 0x0b3a50ccu},
    {0xed1fea1bu, 0x0157f87fu, 0x0bf72b09u, 0x4136e195u},
    {0xefa6130du, 0x38a7ce82u, 0x07304c71u, 0x42a8a57au},
    {0x0a452752u, 0x1fe52a71u, 0xf126effdu, 0x2410f37eu},
    {0xe0bef85au, 0xfc9b22e5u, 0xff76e672u, 0x15814482u},
    {0xec81d752u, 0xdca32b08u, 0xf50814e6u, 0x31b10ffeu},
    {0xfc61c7f8u, 0x4231d927u, 0x18302223u, 0x2c2103ccu},
    {0x046918e3u, 0x0586187au, 0x0708fc44u, 0x1eb1d635u},
    {0xf44e2037u, 0xc8bb026bu, 0xf26fc502u, 0x19a10d37u},
    {0xe7ba0d6eu, 0x037ef5ebu, 0x065bdfa4u, 0x3f37faceu},
    {0x47f6ff83u, 0x32d7fa0au, 0x296c1dd6u, 0xf8cf1f06u},
    {0x22dc258bu, 0xed380197u, 0x12170c9eu, 0x0a1dd52du},
    {0x0100de60u, 0xe7fed90cu, 0xeba10980u, 0xf8a0e59cu},
    {0xd1fcfa37u, 0x166801ddu, 0xdf14c8e1u, 0xf795cb4bu},
    {0x0312016du, 0x19d834e8u, 0xef12f674u, 0xf8ecdad1u},
    {0x26e1d949u, 0xd3431f33u, 0x04df1f2fu, 0xfbdbe3a8u},
    {0xffcbe6b7u, 0xbcda0008u, 0xf446e405u, 0xca3b0b88u},
    {0x3ea5e735u, 0xe241fefcu, 0x01bc3f39u, 0x076deae2u},
    {0x37b03d78u, 0x375268f4u, 0x20b253e6u, 0x170ee438u},
    {0xec12259eu, 0xe63ee419u, 0xe573d5ecu, 0xffd1107du},
    {0x0280fd29u, 0xeac0d733u, 0x0eabe19au, 0x2c51e10du},
    {0xd2bfdd96u, 0x0bf5dbeeu, 0x1555e43cu, 0x1136de7au},
    {0x0f5dfac4u, 0x22fb050au, 0x37d239a3u, 0x4276d5d3u},
    {0xdf39f658u, 0xf3200b1eu, 0xf117f819u, 0xd41ff841u},
    {0x0fd4eae0u, 0xcb53de27u, 0xce8bf95au, 0xebd105ebu},
    {0x20f606ffu, 0xf4622a73u, 0x297703ffu, 0x162217d1u},
    {0x10531466u, 0x31b761ceu, 0x4297c76eu, 0x146e1157u},
    {0x173ee3b1u, 0xf1a9d63eu, 0x036dff2cu, 0xfea3e6f1u},
    {0xd9e9d420u, 0xf850216du, 0xdc9cfb48u, 0x1c81f28du},
    {0xe2310a46u, 0xd57cf1dfu, 0x133d0d2au, 0x142804b6u},
    {0x2a9d3466u, 0x1f8e2f02u, 0x27d0d1efu, 0x0acfdae9u},
    {0xfd3fd953u, 0x1bbdd8b1u, 0xc9b3fd0fu, 0x07130c5fu},
    {0x1debd560u, 0x0532ac9bu, 0x2127544fu, 0xbc332909u},
    {0xfd0534aau, 0x2640efc5u, 0xf3b5f0c4u, 0x03e6f7d3u},
    {0x62b51d2fu, 0x179738cfu, 0x0de0cea0u, 0x1e6f24a0u},
    {0x0b8423d2u, 0x055e1bd4u, 0x256f27c5u, 0x052d0a76u},
    {0xe61edbb8u, 0x0c53ef44u, 0xc4043aa5u, 0x233af02cu},
    {0xdf7a1fe7u, 0x0a892860u, 0xd9ed3394u, 0xd7eb2320u},
    {0xec2c181eu, 0x03340d07u, 0x3d84cde9u, 0xe21a0699u},
    {0x0167e74du, 0x0f2ccf59u, 0x15d617c9u, 0xfab7e940u},
    {0x1dd5f846u, 0x08a0015du, 0xd56943cfu, 0xef34dcd3u},
    {0xebec07bau, 0x3397fc66u, 0x375412d3u, 0xe2fa1eeau},
    {0xf069e3a5u, 0x41854be1u, 0x2efaf19eu, 0x0dca1871u},
    {0x1f1fedbbu, 0xf3a5dcb1u, 0xee5e3ec2u, 0xdc9314f3u},
    {0xc230d265u, 0xecf2ddbfu, 0xfe60261cu, 0x2d74faf5u},
    {0xde07cf63u, 0x0e2bda11u, 0x10171382u, 0x1047f609u},
    {0x2d5eff51u, 0xeabc2b77u, 0xee50d68du, 0xf7b81369u},
    {0x2561f3cfu, 0xf260feb1u, 0xcc9b1db2u, 0xc85bfe10u},
    {0x02160bbeu, 0x09c1e129u, 0xc6311687u, 0xab041a13u},
    {0x2ce9fcffu, 0xe6b3154fu, 0xd40910d8u, 0x0f6318e7u},
    {0x2b77298bu, 0x25d6d552u, 0x108c04d5u, 0xe20cf5d5u},
    {0xfdf4f91fu, 0x0ef029b0u, 0xe4710cb0u, 0xd6e3e6b2u},
    {0x16e1e9c5u, 0x3e0b0740u, 0x0a9b0feeu, 0xc217e8aau},
    {0x1726f7dfu, 0x3e0ee315u, 0xe26f1cfau, 0xd41e1c38u},
    {0x08f45656u, 0x0e95e1e5u, 0x2ad41487u, 0x24b824d8u},
    {0x255f3756u, 0x1562115fu, 0xee2c0412u, 0x1247e818u},
    {0xeb330afau, 0x5260e91bu, 0xc0a7e6e3u, 0xf8a5dbc7u},
    {0xd4620316u, 0x00f7108du, 0x1318071fu, 0x1516e737u},
    {0x2645da07u, 0xfd65f9f1u, 0x3a160628u, 0x0581d929u},
    {0x3a8bf29fu, 0x259800e7u, 0xd040f6e6u, 0xe28ae000u},
    {0xf4a04af6u, 0xf989f33fu, 0x0ef33212u, 0x29b8133cu},
    {0xf9d60a7eu, 0xe5e425e2u, 0x0003eda0u, 0xeb82e322u},
    {0x0eef28ccu, 0x4cb0bdcdu, 0x1d0c2d8bu, 0x04bfc509u},
    {0xd9b6f428u, 0xf2d8e698u, 0xe81ce5d9u, 0x00f1f3e0u},
    {0xdafd261du, 0xd5073556u, 0xd543f09au, 0x058a23dcu},
    {0x09041b46u, 0x21fe28e3u, 0x290ee06du, 0xbd99fa94u},
    {0xf87225a3u, 0x13861625u, 0x15241707u, 0xf9092584u},
    {0x01cbda5fu, 0x122b2567u, 0xefd82c2au, 0x1c7429edu},
    {0xeb42001du, 0x30e731a9u, 0x1f10f0a8u, 0x11343ed3u},
    {0x0501079cu, 0x28fad9cbu, 0x254efe38u, 0x513d293bu},
    {0xf1fcfd26u, 0x1150ea80u, 0xf44a1bfeu, 0x4d4630aeu},
    {0x1ddc1aacu, 0xef9ddeb6u, 0xe966d333u, 0x2d0bf62fu},
    {0xe5f22eccu, 0x1111f83fu, 0xecd7ff70u, 0xe958c244u},
    {0x1cb3396bu, 0xfb6b1c59u, 0xfdaaf759u, 0xe6d0195cu},
    {0xe2a02de0u, 0x46d2df9eu, 0xe3e73710u, 0xd9da2b99u},
    {0x04b7f70bu, 0xf8b30addu, 0xef0d1332u, 0x0bcecee6u},
    {0xe699dae4u, 0x15901b92u, 0xfcae2f8du, 0x1d0d123fu},
    {0x0bba100cu, 0xf848d7a4u, 0x0365f08cu, 0x20f2ee6fu},
    {0xdbe7efe2u, 0x4010245du, 0x14ae3a46u, 0xf1390090u},
    {0xf4200e18u, 0x304e18dau, 0xcfee272au, 0xcc5afaaeu},
    {0x0a89e7f1u, 0x20feff75u, 0xf36c2498u, 0xce06f7ccu},
    {0x23320fceu, 0x24c2071bu, 0x0b73ffbeu, 0xfcded506u},
    {0x19150dedu, 0xf59b12d5u, 0xff6d229bu, 0x1d180c1du},
    {0x08ee1dacu, 0x232df121u, 0x2eb0f16du, 0xcdd516c6u},
    {0xef660347u, 0xfea90c72u, 0xd9bd0dc8u, 0xe0181ac6u},
    {0x19ee0921u, 0x4ca82376u, 0x002b1727u, 0xff811a25u},
    {0xd48f4693u, 0xfe2e2792u, 0x2edd3dcau, 0x35c3323du},
    {0xe8d9f0ecu, 0xec940aa9u, 0x310e5ab2u, 0x00b6e1d1u},
    {0x00f9ca3cu, 0x1386c776u, 0xd0bf34b7u, 0xcd06edf2u},
    {0x2691ff93u, 0x1d5df578u, 0x179bd106u, 0x0b351297u},
    {0x22981303u, 0x4402354au, 0xfebc20dfu, 0x12aaf9f0u},
    {0xd9cc29afu, 0x26b9332du, 0x0de10838u, 0x15cefcabu},
    {0xb4f3e509u, 0x0ca20051u, 0xfb3bee27u, 0x1054de1eu},
    {0x0c29daf5u, 0x1b4c0241u, 0xf92926c4u, 0x2ee42ae7u},
    {0x12c6282eu, 0x32e26568u, 0xdf2d022eu, 0x443debe5u},
    {0x0c5912c7u, 0xfdc0f7c6u, 0xd2f4ff2eu, 0x2c5d1ecfu},
    {0x1e58c751u, 0x0fbf1030u, 0x00744bf2u, 0xd2ed1c40u},
    {0x306f0f1au, 0x0efcc972u, 0x33872cf1u, 0xe238f9e5u},
    {0x21420d3cu, 0xf858e849u, 0x30a939dau, 0xfe29f469u},
    {0xfa780735u, 0x2368f4b1u, 0xf39c012bu, 0x1b52d97du},
    {0xeb5a00c0u, 0xf7f21f06u, 0x171716c5u, 0x103400b0u},
    {0x12490c35u, 0xf4f744b9u, 0x06a71a60u, 0x2999df60u},
    {0x3286ecb3u, 0x36170ce0u, 0x0e4d0d77u, 0xdb80275au},
    {0x0b811014u, 0x0289fde0u, 0x22231345u, 0x25e0ec0fu},
    {0xc8f52d1fu, 0x00d6e3ccu, 0x11dd1bd6u, 0xf6eac851u},
    {0x100bf7f6u, 0xf344fa7eu, 0x20ca44b1u, 0xf5d62543u},
    {0x33c2e337u, 0x37ee3c01u, 0xe4c0f7a8u, 0xed8bdb07u},
    {0xf0f7f9bdu, 0x261013a4u, 0xf68b1c44u, 0xf48218d9u},
    {0x15d50511u, 0x2c060fe8u, 0x089ef456u, 0xf71de5e2u},
    {0xe4be3535u, 0x21e128c0u, 0x08f10d33u, 0xdfa0edb2u},
    {0x19313e5bu, 0xf6abf942u, 0x11fbdc38u, 0x3864d3aau},
    {0x1376e7a9u, 0xd64b130du, 0xe65cd81fu, 0xf031f48bu},
    {0xe4182a66u, 0x0eaffdafu, 0x0d3d09f9u, 0x330be35au},
    {0xdfd0d994u, 0xf734ede5u, 0xdec82501u, 0x1dab2506u},
    {0xeecc1dc6u, 0xeba42603u, 0x281d23e0u, 0x2854feceu},
    {0x1bd10e97u, 0x3283f526u, 0x24401a44u, 0xfda40307u},
    {0x439e1c38u, 0xf3340dc1u, 0xf4ce2df6u, 0x1680e358u},
    {0x28a0144au, 0xef640cd4u, 0x1cc12eb3u, 0x1d25d6c2u},
    {0x09001fbbu, 0xea652d0eu, 0x4e61f35bu, 0x1f6ef87bu},
    {0xcea71c96u, 0xed411c4au, 0xe2aef377u, 0x1f4e0083u},
    {0xd52c3394u, 0x1a36e43eu, 0xd3b1e932u, 0x2e7625acu},
    {0xda790280u, 0x19021d6eu, 0xfa92e822u, 0xe38cfd90u},
    {0xfa09f593u, 0x193818f7u, 0xe34adc11u, 0x092a19fcu},
    {0x164de9f1u, 0x229d31f0u, 0x27662f5du, 0x3467ee76u},
    {0x168a0b58u, 0x1bc215d1u, 0xd8402d27u, 0x0fe3f576u},
    {0xf05a45c6u, 0xf513417au, 0x3bec0438u, 0xf1e0f7f2u},
    {0x27f7ff0du, 0x57aada54u, 0x2fce1e47u, 0x1e0d0145u},
    {0x13dd0a81u, 0xf9ad2031u, 0xe64deccbu, 0x129f2e42u},
    {0xd807f11cu, 0xe70dffc1u, 0x12ac0f76u, 0x1b9b41cau},
    {0xdfb6102du, 0xfbc307e0u, 0x2808e8cau, 0xc5f73445u},
    {0xe8941e04u, 0xfaaf1f37u, 0x37a6e7d7u, 0xd25022a6u},
    {0x1d93ceaeu, 0x07991b1eu, 0x138a3d3cu, 0xf9dffdbau},
    {0x17b5247bu, 0xf5580f68u, 0xf6a9440eu, 0xe932ebdeu},
    {0x1db2d44du, 0xf8552ca0u, 0x331616acu, 0xf86df90du},
    {0x3f5fe38eu, 0x5226fe21u, 0xf19409bfu, 0xec664b8du},
    {0x3d961071u, 0x244e0907u, 0xec451090u, 0x057ede8bu},
    {0xe2efd779u, 0xe632e430u, 0xeb41dbcdu, 0xea5ccb18u},
    {0x1807f8aeu, 0xffa3d9b2u, 0x1d0e0971u, 0xec5de585u},
    {0x1fd71eedu, 0xe5c006ccu, 0xdf9bff80u, 0xd1a9cb31u},
    {0xe478054eu, 0xf8a7063fu, 0xdfcd1a26u, 0x2aae1872u},
    {0x3ab126b2u, 0x0df0f40au, 0x14f709e2u, 0x222bd555u},
    {0x1f8dff47u, 0xec9fd3ddu, 0x166af03fu, 0xecc6fb34u},
    {0x4d534081u, 0x1f80112cu, 0x2a21fc4cu, 0x0e7324c9u},
    {0x15141d32u, 0xf72525d5u, 0xed2024e7u, 0x0b79de8eu},
    {0xe00a1575u, 0xe3fffa28u, 0xe373f893u, 0x1ccbfc23u},
    {0xf740fe25u, 0xdd35d5f0u, 0x09f70adbu, 0xe2ba0ccfu},
    {0xce43ecd4u, 0x0925f072u, 0x2945eeffu, 0xecd7ff3fu},
    {0xd50af330u, 0xfcb80497u, 0xfdf2fd93u, 0xd946e07cu},
    {0xfd7614bau, 0x30eae87eu, 0xf666447eu, 0xe109e5f9u},
    {0xef3c2dbdu, 0x3de4484eu, 0x10242c24u, 0x0b5512aau},
    {0x145f5168u, 0x58654517u, 0x19401edbu, 0x16d3353du},
    {0xce26f343u, 0x10f53183u, 0x1f14ef70u, 0x030d17d2u},
    {0x0f2ad3c4u, 0xe3daeed5u, 0xeb81eadfu, 0xeef7fe49u},
    {0xfcd9117fu, 0xd687e7e4u, 0xe9bfe716u, 0xccb2f9ecu},
    {0xdcf4017eu, 0x2fa52778u, 0x053b239du, 0x278b2b4eu},
    {0xfd2823e4u, 0x2ed9f964u, 0x0dbb1c05u, 0x1a1c1dc7u},
    {0xefce445eu, 0x0437e934u, 0x087c0a36u, 0xe642f330u},
    {0x1c872b2du, 0x3e4f3993u, 0xe673e3cdu, 0x317d1d76u},
    {0x468b5390u, 0x0649331du, 0x160d0d2du, 0x24361161u},
    {0xfe69195cu, 0x1e81039du, 0xed020b61u, 0x3c902bc3u},
    {0xef981fc1u, 0x2e44296au, 0x240c096fu, 0x02a118b7u},
    {0xc0cfe10fu, 0xf9731104u, 0x03ee4f0du, 0x22d71652u},
    {0x2eb1004cu, 0xc7e6ee3cu, 0x180048a7u, 0xf75f02ddu},
    {0xfbee21c8u, 0x18970fbfu, 0xe740fed6u, 0xe2f219eau},
    {0x3ace0f4bu, 0x2e082823u, 0x353f3248u, 0xeda803ffu},
    {0x37e6ef18u, 0xe45a30d6u, 0xed291073u, 0x452e2af1u},
    {0xf63c0064u, 0x502844d3u, 0x3177cf76u, 0x1fb403dbu},
    {0xf712ffb1u, 0x31ec2cb1u, 0x1bb7f254u, 0x1ae90f20u},
    {0xe17afdc6u, 0x0b3a072fu, 0x23f71abcu, 0xf1ec175du},
    {0xd868207bu, 0xd980deb5u, 0xcce24559u, 0x277bf96fu},
    {0xfa920d1bu, 0xe901fd4au, 0x1bde0aacu, 0x019305d3u},
    {0x03a3e153u, 0x1721d4a3u, 0x1f65f647u, 0x00c31d9au},
    {0x0b573fdfu, 0x3637fa0fu, 0x0d85ee62u, 0xfa9c05f5u},
    {0x3f0ef30bu, 0x086725d2u, 0x030b2f55u, 0xf76721b3u},
    {0x25b62a65u, 0x158f0571u, 0x11ac17b3u, 0x2c84165bu},
    {0x544602edu, 0xbe994157u, 0x12b4ff05u, 0x557dd519u},
    {0x58040b55u, 0xfefc31a2u, 0x024b08c4u, 0x459be97au},
    {0x28320ea6u, 0xe24746c6u, 0x209de573u, 0x5f5be959u},
    {0x3f5e170bu, 0xfb09ec34u, 0x0b7b2354u, 0x3907df31u},
    {0x0c4b43b4u, 0x27fb304fu, 0x2363f5b8u, 0xfec0115cu},
    {0x30890bbau, 0x288d304au, 0xf3151870u, 0x1e332a7au},
    {0xf0f8eeadu, 0x39a61ae2u, 0x1eb70948u, 0x09672ab4u},
    {0x0e3448e8u, 0x133a27bfu, 0x24303d1eu, 0x2e13095cu},
    {0x3cb84a9du, 0xf829268eu, 0x114023feu, 0x2d8014a2u},
    {0x48ef21ddu, 0x15841149u, 0x4f7c6387u, 0x3a5d0a93u},
    {0x3f73f6b5u, 0xcc16f69cu, 0x15296113u, 0x44f4da6au},
    {0x3909f025u, 0x2db948eau, 0x63954d74u, 0x537503a8u},
    {0x47380618u, 0xea88fba3u, 0x334f1d3cu, 0x462a276bu},
    {0x4b460e5cu, 0xf4f620a1u, 0x1e5e399du, 0x4d2102f9u},
    {0xec3e21e6u, 0x02fcf523u, 0x19e41f1au, 0xfbba1337u},
    {0x15efdb82u, 0x1b6ffca2u, 0xfcf0f93bu, 0x3635e67fu},
    {0x19ff12eeu, 0x03b50a81u, 0x47ba41b1u, 0xf789f067u},
    {0x06ae0cf4u, 0x546b08ccu, 0x16ff60b8u, 0x0d160acau},
    {0x039c1428u, 0x27212750u, 0x386313cfu, 0x0028ddfeu},
    {0x16ba0c75u, 0xf2eff496u, 0x386c0340u, 0x1240e806u},
    {0x29170248u, 0x3ac7e7d0u, 0x2add5324u, 0x37f7c4d3u},
    {0x220be490u, 0x3805e9b5u, 0x0a5912ccu, 0x2fdaccb6u},
    {0xdd82053eu, 0xdccedafeu, 0xef24384du, 0x305ef4a5u},
    {0x1743e13au, 0x17d4cfe9u, 0x30952d4fu, 0x3f7e1c5bu},
    {0x13941c93u, 0x30f503efu, 0x0c1a3b73u, 0xfaa801dcu},
    {0x1decea51u, 0x58d317cbu, 0x09901bebu, 0xf2065af1u},
    {0x26b50ef7u, 0x4c6acc6fu, 0x2a374d09u, 0xf12e578du},
    {0x3e9af923u, 0x24ae1704u, 0x1f6d187du, 0x0b6a1521u},
    {0x4c561593u, 0xe3b6e58bu, 0x39bc0242u, 0x278e243eu},
    {0x0d430d66u, 0x1f511f7eu, 0x0d2b53deu, 0x0fe4300fu},
    {0x16213288u, 0xde8020cdu, 0x164727a5u, 0x3a99ea8eu},
    {0xf7a7f88fu, 0x1c491cfbu, 0x4117397cu, 0x398d05bcu},
    {0x424f1d5bu, 0x175b1bb0u, 0x287b0211u, 0x36a524a3u},
    {0x41073bccu, 0x10032e3au, 0x0762562au, 0x742d04edu},
    {0x3cc82c60u, 0x13a54a7au, 0xf2b713d5u, 0x539fd779u},
    {0x413344b6u, 0xef5d5512u, 0x07981046u, 0x2defda51u},
    {0x0d511ed0u, 0x325b03e4u, 0x46c20244u, 0xfa4fdcfdu},
    {0x24f34be8u, 0x12084a36u, 0x1c6d36fau, 0xf8ee1f50u},
    {0x0b40e8e5u, 0x019efad3u, 0x17e91996u, 0x3383f55cu},
    {0x0e63f4d6u, 0x45520260u, 0x4ea90a19u, 0x1d331d63u},
    {0xe1a819a7u, 0x0c19412bu, 0x3578ff84u, 0xf83f1475u},
    {0x19fd3f4du, 0x364e12f1u, 0x46d3fa4du, 0x0eb635eeu},
    {0xe55202cbu, 0x50735c98u, 0x0b7cea3du, 0x36c10250u},
    {0xf11c2226u, 0x100a3aebu, 0x4cd9e61au, 0xfec90387u},
    {0xecc01557u, 0xf7ac2a75u, 0x54f1f083u, 0x1f532a49u},
    {0x010c1e6fu, 0x195e5bcfu, 0x3b0e05e4u, 0x1496f1ecu},
    {0x166a2ef0u, 0x1ec406f4u, 0xf640d454u, 0xfe3c2382u},
    {0xf51a28adu, 0x16991c1du, 0x47b512d1u, 0x3d58f382u},
    {0x291ef762u, 0xfdc258d2u, 0x10ef058cu, 0x4eff2dbfu},
    {0x5dc410cdu, 0xe6be1f30u, 0x18e9dd3du, 0x304632b6u},
    {0x39613ac1u, 0xec64fbd0u, 0x1f371af4u, 0x3fd51877u},
    {0x5d87144bu, 0xdc6f1ecbu, 0x04f7d3c4u, 0x3ea40e46u},
    {0x474201d3u, 0x16721747u, 0x178dd44bu, 0x41413b8au},
    {0x553e258cu, 0x1ff235e3u, 0x288eddb9u, 0x43eb3c7bu},
    {0x09060ec5u, 0x1726f59cu, 0x0bdb03b1u, 0x2caa0dbau},
    {0x3034ffc8u, 0x0aab172cu, 0x178edd75u, 0x0f930598u},
    {0x3de4f548u, 0xfbde0dedu, 0x1c13f6e6u, 0xd8e71503u},
    {0x05611ed8u, 0x116d3042u, 0x5691f911u, 0xf2d024b3u},
    {0x05473691u, 0x281a2e8au, 0x4ae5f378u, 0xd0d1eae7u},
    {0x3cb6368fu, 0x03b73c46u, 0x482d24a3u, 0x26df1f09u},
    {0x29192a6cu, 0x515836c0u, 0x3c14dc29u, 0xe6ff0aabu},
    {0x4b86168cu, 0x34940032u, 0x4ca2f3dcu, 0xf17f17bdu},
    {0xea7a29bcu, 0x264ee933u, 0xf7e51d82u, 0x1154dfe0u},
    {0x0ab99ef0u, 0x0c92a5e0u, 0x03861094u, 0x00dc75acu},
    {0x0c80c990u, 0x00e0d6e6u, 0x037896a8u, 0x06a912e8u},
    {0x10c43a00u, 0x04dfe0f8u, 0xf7c296c0u, 0xfe9db3fcu},
    {0x06af2ca8u, 0x0c6c96e0u, 0x06f082d0u, 0x07a4bf40u},
    {0xfcb33b30u, 0xf2343270u, 0xfd6d2e18u, 0xfdd54128u},
    {0x0e061130u, 0xfeff58bau, 0x0bfe8580u, 0x0a063c80u},
    {0x0d529bd0u, 0xf5510040u, 0x107b8cc0u, 0x0f8b3b70u},
    {0x021fb1bcu, 0xf76ebfd0u, 0x0a1be3f0u, 0x0564eb68u},
    {0xfe2cb6feu, 0x0c4c0bf0u, 0x02f3aa3cu, 0x060dc7d8u},
    {0xffc1e20au, 0x03a87130u, 0x082c04e0u, 0xfb9953b8u},
    {0xf25ff7d0u, 0x0f8716f0u, 0x0868ecb0u, 0x0de63100u},
    {0x051f9798u, 0x09ee4f70u, 0xfc1b0380u, 0xfe3334a6u},
    {0x0faf5a20u, 0xf84b5128u, 0x09f54150u, 0x086864c0u},
    {0xf9e37138u, 0x000023a8u, 0xfd1e4cc0u, 0x052d7320u},
    {0x0d2d30c0u, 0x0845ad40u, 0x0cb06b00u, 0x09051220u},
    {0xfc328794u, 0xf4020d20u, 0x0f6f97e0u, 0xf69394b0u},
    {0x2d5d094cu, 0xebfb2a5cu, 0xf8530ca9u, 0x01b916cbu},
    {0x217edd5bu, 0x136b1679u, 0x0dfa4a12u, 0x0dd42da2u},
    {0x1e77f047u, 0xdc9144e8u, 0xde2441dau, 0x085e26b0u},
    {0x30901e62u, 0x10e00bceu, 0xd37f2204u, 0x30e3310cu},
    {0x0fddd745u, 0xff2be48bu, 0x18c9e84au, 0xe4462833u},
    {0x29df0a80u, 0x0f432f4fu, 0xeb9f30f6u, 0x11290a8bu},
    {0xfe8531d7u, 0xfdb4d321u, 0x2e1313bcu, 0xdaf0dba7u},
    {0x4a0ff20fu, 0xffad4753u, 0xb2e05459u, 0x2e43494du},
    {0x04123fd9u, 0x40463f5du, 0x34b3e092u, 0xe774110cu},
    {0x27851972u, 0x24f8196cu, 0x2dacccb9u, 0x0ecef076u},
    {0xd99241e7u, 0x15b546ddu, 0xf2e92173u, 0x05323844u},
    {0x00c5060bu, 0xfbc01f8fu, 0x06b9e9f7u, 0x227e1540u},
    {0x03a4f50eu, 0x337729c6u, 0x2fd1d5a1u, 0x29ce2413u},
    {0xee991c25u, 0x342c244au, 0xfbe12216u, 0x15d9ed33u},
    {0xf2e2ef7eu, 0x02690541u, 0xe82a0a3eu, 0xd515dbd5u},
    {0x210a439eu, 0x50212944u, 0x2c811c08u, 0x608f2d1eu},
    {0x1dc43ca6u, 0x14cae4d5u, 0x03c53246u, 0x02ebffaeu},
    {0xeffb00b5u, 0x3672da9du, 0x130c3da2u, 0x25e5443du},
    {0x2c2f1dceu, 0x17d4067fu, 0x38dffdd3u, 0x2857fc91u},
    {0xeaaa2ab4u, 0x1bd1a7e3u, 0x4c6b1a05u, 0x2c0d50cau},
    {0x1a65da0bu, 0xdc2549aau, 0xce33e967u, 0x25ef047fu},
    {0x14dff236u, 0x1038fe29u, 0x27ef2afbu, 0x18ee1b36u},
    {0xde9a24ecu, 0xdfdf0010u, 0x1e3e2224u, 0xd25a05f4u},
    {0x1bea251fu, 0x62fffc70u, 0x5a272536u, 0x3ad71820u},
    {0xd42d136eu, 0x2f6c2cdau, 0x1dc728b2u, 0x167bd6aau},
    {0x2199fb26u, 0x41032cd8u, 0x00a32df8u, 0x09ddfc10u},
    {0x1ffd3589u, 0xfe3cee83u, 0xe14408ebu, 0xee3ef763u},
    {0xde07f9f6u, 0x40d52f02u, 0xea49dfa7u, 0x407ac5f0u},
    {0xf2edede8u, 0x0d940ef8u, 0xdf63102bu, 0xfaa33523u},
    {0x16cf122bu, 0xf04e0130u, 0x00a7f824u, 0x06e6f0c1u},
    {0xd9da05a0u, 0x1a4fe352u, 0xd57fd2a4u, 0x2cfb1ba8u},
    {0xff6445fau, 0x23af4813u, 0x006dd77cu, 0x28d8c827u},
    {0xf157136cu, 0x1e8d3612u, 0xe700e250u, 0xe1ed0f0cu},
    {0x49dcd658u, 0xfb2543beu, 0x463e32a5u, 0x1ced1582u},
    {0x2cfc2787u, 0xe56aed72u, 0x114b44adu, 0xede42c3du},
    {0x5814f0a6u, 0x2e7508d2u, 0xfdad07a2u, 0x05d1555bu},
    {0x1442d7fbu, 0x255e2a7fu, 0x1a43fcd9u, 0x2993dc8au},
    {0x245d1737u, 0x325e2444u, 0x1bc7332fu, 0x22cbe6abu},
    {0x1b14f599u, 0x1814ef81u, 0x2266d07bu, 0x37700c3eu},
    {0x490c14deu, 0xfce15120u, 0x49d313bdu, 0x334a2612u},
    {0x0ea123d9u, 0x08f23957u, 0x17d5cfa9u, 0xe330ebe9u},
    {0x2fed290fu, 0x30c51670u, 0x024eeb97u, 0x0ed325b1u},
    {0x0df62334u, 0x282e0f1fu, 0x0ecdcc39u, 0xfa77033eu},
    {0x3e85041du, 0x2f4933aeu, 0x1e95b49eu, 0xe8e41bdfu},
    {0x02851836u, 0xf6ae26b8u, 0xefe5323au, 0xed4f162au},
    {0x1772e519u, 0xe31e2bf8u, 0xf90e094au, 0x003e1913u},
    {0xe3b1e11bu, 0xfbdc1053u, 0xff03f8eeu, 0xd39216cfu},
    {0x52edecdau, 0x36574d09u, 0x440ff581u, 0x25295b92u},
    {0x028feaabu, 0xe6851e4du, 0x15b52f5bu, 0x24e4fc27u},
    {0xd9450f10u, 0x1c67f159u, 0xdf990984u, 0x23ee0202u},
    {0x29d80be4u, 0x0763fc89u, 0xe60f119du, 0x239fc7a6u},
    {0xf4792e2fu, 0xdfc5390du, 0xda381dbcu, 0x0f68e20bu},
    {0x0531e32au, 0x1dc006c7u, 0xe2632c1du, 0xec964559u},
    {0x0f1e37c2u, 0xf6423bdcu, 0x26083314u, 0xeb51e200u},
    {0xebb11495u, 0x167e1deau, 0x00dccdb0u, 0x1af6f116u},
    {0xd8f05256u, 0xedd854c5u, 0xe27e2bc7u, 0x284ab43bu},
    {0x0644f82du, 0x27130617u, 0xe57d1142u, 0xea0a3845u},
    {0x005c4171u, 0x06c5fac5u, 0xeb6cd23fu, 0xd566ff90u},
    {0x239924deu, 0x17ca29dcu, 0x178fd07eu, 0x1daafe90u},
    {0xeacc1f97u, 0x550f0422u, 0xf5ac1776u, 0x0772371bu},
    {0xffb3d30cu, 0x0a031831u, 0xe915fd02u, 0x055df871u},
    {0x1b6e2caeu, 0x31bc325du, 0xd401da7au, 0x2a84e4f7u},
    {0x13c71588u, 0x29801c1fu, 0xf8042d5bu, 0x0bbb26d0u},
    {0x04774463u, 0x103e2cbau, 0xdc3ce62du, 0x00bb177fu},
    {0x289fd979u, 0xe1832212u, 0xb1bc496eu, 0x2e22469eu},
    {0x2f2b02fdu, 0xfec617a8u, 0xa5a35d74u, 0x22cb2ac1u},
    {0xe91f0c4fu, 0x193ce9b1u, 0xfd142b96u, 0xeffc333au},
    {0x193be6c5u, 0x0625f667u, 0x5875118fu, 0xd33f009fu},
    {0x38a9f67du, 0xf81127a9u, 0xf9c606c2u, 0x2bf5f521u},
    {0xfdde16cau, 0xdc5d23c8u, 0x11e51378u, 0x2a0f2118u},
    {0x2735dd85u, 0xfbb0d9dcu, 0x06afefddu, 0x08ecfa8bu},
    {0x3335e52fu, 0xe522251du, 0xf3d652a0u, 0x2e581b41u},
    {0xeff51d09u, 0x5f15230fu, 0x54fe0a7du, 0x5e86302au},
    {0x1ea36097u, 0x2d5761ebu, 0x44fbfc62u, 0x3c486837u},
    {0xde5c1d4cu, 0x22de2ae3u, 0x2a4809e4u, 0x03d9102au},
    {0x159611c0u, 0xd188e5e7u, 0x21ecf18fu, 0xf8bbe146u},
    {0x1a620ee3u, 0x0be6f05du, 0x2c2f1c92u, 0x1cdf3893u},
    {0xe195ec94u, 0x179e34bcu, 0x3e8bf638u, 0xffc3128eu},
    {0x262cfe23u, 0x0a4dfe71u, 0xf26cf46fu, 0x2e272891u},
    {0x3bd70a06u, 0x4a6123a0u, 0xf188cd57u, 0x33d9f770u},
    {0x16b11419u, 0x23f2bcdbu, 0x10a72475u, 0x1c824bb6u},
    {0xd88b5078u, 0x5589d55fu, 0x0f0d639bu, 0x26042b80u},
    {0xe57021feu, 0xf2cd3084u, 0xfe9526d2u, 0x2d981dc5u},
    {0xf1b001cau, 0x1215226eu, 0xef53d9b2u, 0xfcd7e7cfu},
    {0x1c0b0a56u, 0x394a217bu, 0x1fb2db93u, 0xe15be55cu},
    {0x06c10cdau, 0x0ecbfb4fu, 0x23b2f9ceu, 0x42dd2654u},
    {0x00c4e5cbu, 0xd9be137bu, 0xdc5cecdcu, 0x0b17fc44u},
    {0xf3cf3e92u, 0x4c11159au, 0x3e051a62u, 0x25befd5bu},
    {0x0a22630fu, 0x09f530a9u, 0x2c9fe3f4u, 0x18b6b31bu},
    {0x2cdf1697u, 0x593a68dfu, 0x09d40406u, 0x57c5cb2du},
    {0x216b2af7u, 0xf1db30deu, 0xf48734b4u, 0xd7b7219cu},
    {0x0cda2419u, 0xda211f7du, 0xf4d94ddfu, 0x0afd382du},
    {0xd0cf1a99u, 0xe36c2bb6u, 0x2ceb429bu, 0x26a13470u},
    {0xddff13ebu, 0x2746266au, 0x2bdffe96u, 0xe7e304c0u},
    {0xfd261658u, 0xea2918d4u, 0x0c53d516u, 0x132b0879u},
    {0xf65afef2u, 0x027b19ebu, 0x2910f923u, 0x4245cf6au},
    {0x2a0fe3e5u, 0x0ff22d0du, 0x458ce289u, 0xfff7215bu},
    {0x11f90f2du, 0x3ef32718u, 0x1c172b5cu, 0x4610679eu},
    {0x1ffeef1cu, 0xfa2a198bu, 0x14462cd7u, 0xdd943060u},
    {0xed0e1cddu, 0x1d41f311u, 0x0f611128u, 0x17ddeeceu},
    {0x21a60427u, 0x327d2d1du, 0xfdc41955u, 0x18da05e3u},
    {0xfc2b27bfu, 0x45810e5eu, 0x1e3ae6a9u, 0xdc6f210du},
    {0xd13ef2b7u, 0x271c04c1u, 0xfb8b1344u, 0xd977edb6u},
    {0x3001e9abu, 0x106f0d92u, 0x15c90c4au, 0xfd2f3a04u},
    {0x2806d9edu, 0x4e535f75u, 0x634c9f7du, 0xe806391au},
    {0x16771ce9u, 0x198a2c78u, 0x6ca1bd29u, 0x240e1c4bu},
    {0x15c8e18bu, 0xf3771204u, 0xe24d039cu, 0x1d6f1ffbu},
    {0xd9b5eaa8u, 0xdf442235u, 0xde693582u, 0xd40b0812u},
    {0x20ad2270u, 0x2ec31e73u, 0xee9c32e0u, 0xe761e0efu},
    {0xfed7fe0bu, 0x3c90fcfau, 0x2a95eb32u, 0xf7813ee6u},
    {0x00100b96u, 0x1471023du, 0xe66af61eu, 0xdd7ff73eu},
    {0x45fb0623u, 0x43d0106du, 0x28edc845u, 0xfcb707b4u},
    {0xfe4d53eau, 0x04930abeu, 0xd2ba2cabu, 0x62b9b47cu},
    {0xe82b55dfu, 0xd5ef2d00u, 0x172a54acu, 0x578beda6u},
    {0xfc86f70au, 0xd1a1f2c8u, 0x10131a82u, 0xea5b046au},
    {0xd19cfe22u, 0xef55079bu, 0x1609db25u, 0xe26a3e3bu},
    {0xd96fdc76u, 0xf24622f7u, 0xe568fafdu, 0x13f61662u},
    {0xf83ef237u, 0x0cca2bbcu, 0xef050521u, 0x2c01bfecu},
    {0x233afd7cu, 0x25af0a0cu, 0x04ccdfd7u, 0x03d9e13bu},
    {0x07964214u, 0x057102aau, 0xe140fcf0u, 0xfe540d0eu},
    {0x021e15e4u, 0x0ddc152eu, 0xe8f0d1d8u, 0x4ccd18f8u},
    {0x0b6250aau, 0x6d963c7bu, 0x0bb91573u, 0x12f7138bu},
    {0x093101ceu, 0x07defcdfu, 0x0a7e296cu, 0x08cf1ddcu},
    {0xe3a60a43u, 0xe42ce879u, 0xf3c4ed2du, 0xe03dffaau},
    {0xedd21a71u, 0xee602925u, 0x1cd7de40u, 0x097bf50eu},
    {0xff003383u, 0x0f9c1d6fu, 0xead6eac3u, 0xefea1c03u},
    {0xf27911cfu, 0x17b509f8u, 0xedc10158u, 0xeb37fe06u},
    {0x212a3a5eu, 0x076923fdu, 0x0b80ef6du, 0x1caf4341u},
    {0x2dcaf755u, 0xfde2e784u, 0xd30619bbu, 0x3e7213c4u},
    {0xe15313f2u, 0x062400a6u, 0xd29f12d3u, 0x1960edd4u},
    {0xd19a0493u, 0x24d4f32cu, 0x4881e7ecu, 0x1558f9bdu},
    {0x03c3cab0u, 0xe1443a12u, 0xba871c12u, 0xff461e99u},
    {0x1f0ed46au, 0x22da3fd5u, 0xd3355d0cu, 0x42965b21u},
    {0xf4a8eee9u, 0xd1833a5cu, 0x31111810u, 0x01d1032fu},
    {0xc27bffccu, 0xe103ca1eu, 0x4a7ffa33u, 0xee78d1f2u},
    {0x40012003u, 0xedae485cu, 0x937a3220u, 0x3d9166dau},
    {0xe2532746u, 0xefebe014u, 0xfa472521u, 0xdab21209u},
    {0xe351f983u, 0xf9e4e25fu, 0xf411f52bu, 0x2108f71du},
    {0xde2de346u, 0xfc591351u, 0xbde41627u, 0xbcb71515u},
    {0x1e9635f9u, 0x3694fb7au, 0xf9d8d1e9u, 0x34225590u},
    {0xea4d42a8u, 0x280f06c8u, 0x0ed0275cu, 0x30f84388u},
    {0x07f205abu, 0xf39ff1d2u, 0x17cbccaau, 0xe82bedcfu},
    {0xf038d4a9u, 0xecc6d676u, 0xf6fffffeu, 0xf0f2ee1du},
    {0x401359aau, 0x0fa263bfu, 0x31270aa5u, 0x1bb425aau},
    {0x28ecf662u, 0xd599cb2du, 0x0a7d2828u, 0xfcffeb6au},
    {0xef7003b2u, 0x18f4282au, 0xf2c4d808u, 0xf8eaeaf9u},
    {0xe4e01450u, 0xfb93494au, 0x101fff67u, 0xd28706abu},
    {0xe5ac3bcbu, 0x167befdfu, 0xfced10fau, 0x3968f588u},
    {0x2174469bu, 0x0febbe4du, 0x1cf35dc4u, 0x059b456fu},
    {0xe1da2b2bu, 0x2d373f54u, 0x06c03064u, 0x31ddddb0u},
    {0x02d4cc00u, 0xe1bd3b76u, 0x08bbf42fu, 0xb347e0f5u},
    {0x1dec3f93u, 0x43d0df3au, 0x4b991c75u, 0x3afa5383u},
    {0x1c8a02e1u, 0x0794eaa6u, 0x2d95d70eu, 0xccf4f391u},
    {0x0a6df512u, 0x02832130u, 0xe6b2f616u, 0xfae71d19u},
    {0xf848fce3u, 0xd3bce009u, 0xd4c5402au, 0xf6632c95u},
    {0x03b83f51u, 0x00462f03u, 0xe8e607e2u, 0xe85df1b2u},
    {0xf11a14c8u, 0x53e83869u, 0xd673f5a6u, 0x545daed4u},
    {0x008f0765u, 0x0ae807cau, 0xf721523eu, 0xe90c0dc3u},
    {0xffe0cfceu, 0xe9fcb58bu, 0xfdc31a8cu, 0xd5ea65d3u},
    {0xf94054bfu, 0x015412bfu, 0x1833e8e5u, 0x39dbce48u},
    {0xfc751690u, 0xe041e5e0u, 0x34b4014fu, 0x13bf2623u},
    {0xef9f1632u, 0x2bc4e4b3u, 0x0d140113u, 0x2346ea35u},
    {0xcf04d1b4u, 0xf647f63eu, 0x05db287eu, 0x23db10c7u},
    {0x261bd4c0u, 0x262cfb27u, 0x2f39f680u, 0x198d0cffu},
    {0x0f082426u, 0x0b1317dcu, 0x5d6e17cfu, 0x483b5e64u},
    {0x115b0644u, 0x1fabe148u, 0x0ae6f91eu, 0x1f903283u},
    {0xcf3f1b5du, 0xf9a8cb9bu, 0xbed04ce0u, 0x1ea1b672u},
    {0x3b4b1045u, 0x29a84142u, 0x22b1ef66u, 0x46526453u},
    {0x0c6d05b9u, 0xfa93f25cu, 0x0d20da84u, 0xe263e21cu},
    {0x1f450250u, 0xe6d4fe3cu, 0xe2c9db92u, 0x17c3ed33u},
    {0xe262d24bu, 0x0d850f88u, 0x0f475f5du, 0xd179dc70u},
    {0x31e805aeu, 0x0ba4f6efu, 0x4e6eece4u, 0xedb54db5u},
    {0x0f1a0c2au, 0x41f45575u, 0x4efbafcau, 0xd32a50d7u},
    {0x04532a13u, 0x22280a65u, 0x06c31c2au, 0x10582265u},
    {0xdc6d1ce3u, 0xc686ba55u, 0xbfd95fffu, 0xf10ed2f7u},
    {0x1b43e456u, 0x1bfe2f76u, 0x49109c04u, 0x24d44a81u},
    {0xf707256du, 0xd7d5294du, 0xed700670u, 0xf2091057u},
    {0xd8610a7du, 0xe50c1d9fu, 0x271dfdbcu, 0xeece26aau},
    {0xee860dcbu, 0xdf1a12a9u, 0xf39bfe6cu, 0x0b264f54u},
    {0xe0f13bb2u, 0x22c72715u, 0xdabc5497u, 0x034f0342u},
    {0xd58b46b7u, 0xf6694fe1u, 0xdb162481u, 0x0cccdca9u},
    {0xe30118c8u, 0x029e1380u, 0xe43c235au, 0x1e45300fu},
    {0xd1adbd06u, 0x0becd2cau, 0xf427109au, 0xc51269f0u},
    {0xd1a84a10u, 0x23524d38u, 0xee83407cu, 0x11bac4e6u},
    {0x104bd964u, 0x07cb1ba2u, 0xd25ef71du, 0xe14eee7bu},
    {0x1061ea25u, 0xfe9e1557u, 0xff820724u, 0x14af1dd9u},
    {0x1f9ef59cu, 0xd3f5dd69u, 0x0bb1f22au, 0x2a8ef511u},
    {0x1593f809u, 0x24c243e1u, 0x2350f993u, 0xd9a4fb9au},
    {0x11c9218au, 0x31204471u, 0xd220ebc3u, 0xea8b4cf0u},
    {0x105723feu, 0xeff4e061u, 0x298b160du, 0x21e80070u},
    {0xead3baf5u, 0xb914e58du, 0xdb45d8b5u, 0x0749d631u},
    {0xe7071e29u, 0x4c212340u, 0xf5920080u, 0x0dae49c1u},
    {0x20fdd177u, 0xdf011e08u, 0x27d2de30u, 0x25c30cc4u},
    {0x3252e103u, 0x070729f6u, 0xda901cb7u, 0x31d80e7fu},
    {0xec421242u, 0xfffdf709u, 0x107c185cu, 0xed602b49u},
    {0x41742778u, 0x13e45577u, 0xfa0f41e7u, 0x31a317ceu},
    {0x43f91472u, 0x1ab737c5u, 0xff7d3322u, 0x2a21173cu},
    {0x1d63221bu, 0xf8741681u, 0xff06ef5bu, 0xe1f43349u},
    {0xf14ee543u, 0xfe0138dau, 0xde94e704u, 0x074c1c36u},
    {0x26bb1b8eu, 0xe4b3f904u, 0xe38eefc3u, 0x33da03c4u},
    {0x08cb0a7bu, 0xf6b2df21u, 0x19aa0dd3u, 0x0e46db8au},
    {0x098de2c1u, 0x0cf6e38bu, 0x3d4dd53bu, 0x1ed11ef6u},
    {0x2c650113u, 0xecf00bf3u, 0x1711035au, 0x38142677u},
    {0xf1fc2a34u, 0x4e451bd6u, 0x05cb284eu, 0x136b0e30u},
    {0xd8751dccu, 0x2b51fdc0u, 0xfd67feb4u, 0x368ff3c6u},
    {0xfc5f222au, 0x13e1ea4cu, 0x10ccdcb7u, 0x39190d99u},
    {0x01a02a9cu, 0x058f2095u, 0xe6110bfau, 0x40382302u},
    {0xe67c2c23u, 0xf1cb449au, 0x2f85de0bu, 0x354a2489u},
    {0xf6f7f559u, 0xdef3fe79u, 0x0abe2163u, 0xdff90079u},
    {0xda710676u, 0x15a1ce51u, 0x3cd3ed2bu, 0xea6a008bu},
    {0xdedc0727u, 0xf972f597u, 0xe5d40f26u, 0x03220f00u},
    {0x1ab14fb7u, 0x4082c164u, 0xfb1b100fu, 0x10d45cb1u},
    {0x263c334cu, 0xf9e1fbf0u, 0x4ee02e7bu, 0x345f2617u},
    {0x23ed17a1u, 0x1aa431c1u, 0xe014fd9du, 0x117f3193u},
    {0xd9af0d34u, 0xff9ce5beu, 0xe93333b4u, 0x243ee4ceu},
    {0xfa391877u, 0x1f9fd689u, 0x2c0e33ddu, 0x49d70101u},
    {0xef281befu, 0x0197e8bbu, 0xd201f994u, 0xe87d00cdu},
    {0xfb2df04du, 0xf3c93e0fu, 0xd50ee147u, 0x0a0d119eu},
    {0xd1070508u, 0x1ba2071cu, 0xd3752fe1u, 0x1c5df0d4u},
    {0xea283efeu, 0x36183a0au, 0x145c4126u, 0x3be0d6a1u},
    {0xedef241fu, 0x397e3365u, 0xd9adfaa5u, 0xfc84dc49u},
    {0x25822b24u, 0x0dbe0524u, 0xf7573119u, 0xd860090au},
    {0xe8a23164u, 0x0fed332du, 0x1a72ea30u, 0x12bc1725u},
    {0xf5c916f3u, 0x37001b2bu, 0xf4b30748u, 0x3dcefbdfu},
    {0x1ac5e1f9u, 0x06ff2d6eu, 0xf7ef3e1eu, 0xea10f5fau},
    {0x2df8d996u, 0xe448f762u, 0x03b2f548u, 0xe1cc1e82u},
    {0x2d6c2a8cu, 0xdbabf421u, 0x03aaf145u, 0x1c72fef5u},
    {0x55612da9u, 0x4d0a2d8eu, 0x28c903d2u, 0xedce1b7cu},
    {0x2f130164u, 0x0b231fa9u, 0x3aaedf91u, 0x21c4eec8u},
    {0xf0e4d6fdu, 0xecf921a3u, 0x154500fau, 0x36483853u},
    {0x05bad257u, 0xed463d61u, 0x363427f1u, 0x051bf6e9u},
    {0x11aa015eu, 0x3b922376u, 0x1d982194u, 0xf9b3312bu},
    {0x1031f3d4u, 0xfc32dc76u, 0x2f6147b7u, 0x2ae52db9u},
    {0xf91c0c61u, 0x0f48308cu, 0x19a40825u, 0x2332e80au},
    {0x15c9d835u, 0xe9efe5e1u, 0x099dcc3au, 0xe69319e1u},
    {0x31a21172u, 0x0dc531f5u, 0x4f4cf8b8u, 0xff945689u},
    {0x209d275bu, 0xf4c42545u, 0x1d3d09c2u, 0x1f161a36u},
    {0x2f4d2b7bu, 0xf2b90c7bu, 0x3c8c283cu, 0x27f01ac0u},
    {0x1c5906e8u, 0xf31541e4u, 0xec18e703u, 0xdca3118cu},
    {0x1ee4ef06u, 0x3f474b8bu, 0x2583d932u, 0x0833fa24u},
    {0xd9a61e96u, 0x2622dc1bu, 0xe7f0041du, 0x0c16394au},
    {0x0c2e3731u, 0xfce52553u, 0xe3d71dd4u, 0xfd68dbe3u},
    {0xd4e4de88u, 0xd5a500a0u, 0x2ad9f2e8u, 0xf89acc45u},
    {0x0e771c1fu, 0xde192d44u, 0x1d4a0a9bu, 0x2648c417u},
    {0xe530064fu, 0xe99cf918u, 0x0c3920d7u, 0xf0a5e50du},
    {0x22752d0au, 0xee3238d3u, 0xf52b35bbu, 0xf02eef3au},
    {0x2257303cu, 0xee793dbfu, 0xe51e15e3u, 0x3eede1d9u},
    {0x15d8fe61u, 0xdbcc17a0u, 0x0aca33f4u, 0x3b7eda56u},
    {0xecb1eebeu, 0x23212974u, 0xe934ea8bu, 0xf6c20299u},
    {0xf631f221u, 0xe5b0f9d5u, 0x05f2d2ddu, 0xdd901f86u},
    {0x0b6eff82u, 0x2d6aea8cu, 0xd241ee89u, 0xf927e5d5u},
    {0x0b950cd7u, 0x0cb4127du, 0x246b1db7u, 0x1a45304fu},
    {0xef08f104u, 0x4a4b39b5u, 0x1862d1bcu, 0xec31f8e0u},
    {0x1eeb29b6u, 0x400aed13u, 0xfa87f188u, 0xea601a0au},
    {0xd53f11a3u, 0x3c0d2feeu, 0xfa8af5a2u, 0xcff0ed03u},
    {0x00c13349u, 0x226e0888u, 0xd271f00au, 0x0d7f27e2u},
    {0xf0dbdf38u, 0x25100f67u, 0x0320f8c6u, 0x0342ed36u},
    {0x2814dc84u, 0x28ee5159u, 0xdf354e3du, 0x20b0298fu},
    {0x404b24b0u, 0xee142c59u, 0xce950535u, 0x05702bafu},
    {0x286f2001u, 0xfdf710fau, 0xc1ef1abfu, 0x076eeb31u},
    {0x20e7f131u, 0xda2a3704u, 0xdcdefe57u, 0x2dc6ff53u},
    {0xe420e10bu, 0xd97b0fa9u, 0x33f0e39du, 0x126334beu},
    {0xf4f3f4b2u, 0xdf4928dfu, 0xdaaf2384u, 0xfa9d0c07u},
    {0x54d0208fu, 0x085e338au, 0xdf2c4ef1u, 0x16ed1afdu},
    {0xfd83118du, 0x24151f2du, 0x3623e79au, 0x1e1febd3u},
    {0x161b535fu, 0x235d0fa0u, 0x1a002765u, 0x57471661u},
    {0xf8d75257u, 0x3d031ab4u, 0x50f2f07bu, 0x15414441u},
    {0x109ffeb6u, 0xf15f459cu, 0x250a2371u, 0x2d65f082u},
    {0xef4c0d82u, 0x14e91e43u, 0x36a7ebb5u, 0x39c24424u},
    {0x17bff312u, 0x16890852u, 0x2cf1e44du, 0xf9ad113au},
    {0x22bedd61u, 0x2af7d1b9u, 0xdc750804u, 0x24dce7c6u},
    {0xf9305e08u, 0xfe703b0bu, 0x3f36dd16u, 0x23d33fd4u},
    {0x02b4e38au, 0x3cb501b2u, 0x0a8fe07eu, 0x2dbd2568u},
    {0x0eca267eu, 0x2963bbd1u, 0x3e8a21ebu, 0x240b5cb5u},
    {0x21c70f74u, 0x3d85056fu, 0x49d12f81u, 0x4e6b3902u},
    {0x066b3706u, 0x283013dcu, 0x058323a0u, 0x3f5340c6u},
    {0x203c2382u, 0xefb8f446u, 0x2f5840d9u, 0x338a1794u},
    {0xf3181ff1u, 0x22ef047fu, 0xe7d300e0u, 0xf47319e4u},
    {0xdfb91bfcu, 0xfe200583u, 0x10330622u, 0xf7772492u},
    {0xde531747u, 0x44aedd6au, 0x398337dbu, 0x516e0371u},
    {0xda543312u, 0x274afaa1u, 0xe6410dc0u, 0x0c2ae0b5u},
    {0xf11c2f42u, 0x53430921u, 0xe0abfab7u, 0xfd43ca04u},
    {0xf66a48ceu, 0x5da556f7u, 0x05d004c8u, 0x195bd3d1u},
    {0xdfc0ec39u, 0x13332e84u, 0x14f8cd9bu, 0xf9c6e361u},
    {0x2e443a5du, 0x417e3f1au, 0x078af2cau, 0x2e5be5c8u},
    {0xd9ca1294u, 0xeb5ef53cu, 0x1fc30743u, 0x10be041au},
    {0x1040ebd5u, 0x09f3d460u, 0xe1d8d6deu, 0xf628ddcfu},
    {0x0796f917u, 0x1bff3ebfu, 0xf8fef761u, 0x36e9c944u},
    {0x0b220211u, 0x16e11274u, 0x2c22ef74u, 0xdd69ecfeu},
    {0x62772833u, 0x14073306u, 0x34d1f589u, 0x0a733d7bu},
    {0x11b3d75du, 0xfbaf278du, 0x473fed00u, 0xe97902d7u},
    {0x005ee4f5u, 0x25f62acbu, 0x197ce7e1u, 0x155f443eu},
    {0x1610ec80u, 0x42eb0e67u, 0xec8a20a6u, 0x35ee30afu},
    {0x29cb2920u, 0x1d5bf7b4u, 0x108318dbu, 0xe52f11a8u},
    {0x0b9fd36eu, 0xfdcbee84u, 0x1d8915a9u, 0x1a1a0f64u},
    {0x4267f3a8u, 0x0ca8ffd3u, 0x1bc418a8u, 0x22b52bb3u},
    {0x24f32611u, 0x26563506u, 0x35ef07dbu, 0xe5831567u},
    {0x4cc50df9u, 0x272c3da2u, 0x5598a3d3u, 0xdd83352au},
    {0x12a1db00u, 0x4fce4b09u, 0x1a50c5cdu, 0x07cf3384u},
    {0xfa93e808u, 0x362cf694u, 0x19c5c8d0u, 0xde273b3fu},
    {0x3217d4f1u, 0x2484404du, 0xf7f81da4u, 0xdb84124du},
    {0x1475d202u, 0x361def45u, 0xe15317e0u, 0xece8fb0bu},
    {0x218d2d1du, 0xe4231159u, 0x2de2ea35u, 0x116d0149u},
    {0x22371b48u, 0x3bf63e04u, 0x47139d07u, 0x0b996896u},
    {0x2789207eu, 0x061b12d1u, 0xf179f64du, 0x1a1e2365u},
    {0x00072260u, 0x0628258bu, 0xf3a6451fu, 0x4ebadefcu},
    {0xe6ca275cu, 0xdde64330u, 0xd1201618u, 0x4e19f8ccu},
    {0xe4fe397cu, 0xdbcd2155u, 0x15cb0215u, 0x40d617e5u},
    {0x097efa25u, 0xce9e105cu, 0x1e34eda1u, 0x108dd58bu},
    {0xd81a1097u, 0x063f0922u, 0xfc5232e8u, 0x38252b36u},
    {0xf1260c9cu, 0xe825eb53u, 0xef5dd44au, 0xeabf237eu},
    {0xdb082bb4u, 0xe66a1047u, 0xfbfe0cbbu, 0x3aa8e315u},
    {0xda77279bu, 0xeaa6efa2u, 0xcd9b14fau, 0x19370b2fu},
    {0xff4e1761u, 0x36372072u, 0x254fddfcu, 0x3cc40ff4u},
    {0xda0257b9u, 0x520e58adu, 0x15d7242bu, 0x36f85924u},
    {0xf2d208d1u, 0x42013044u, 0xcf3815b0u, 0x38cc077au},
    {0x03463b6au, 0x20a8f370u, 0x0b38dadau, 0x1dfa173eu},
    {0xf65d1a96u, 0x0ae8257fu, 0x26ead67cu, 0x2c1a20c0u},
    {0x09a025f7u, 0x092bf9a4u, 0xf1830e1au, 0x0d1f209fu},
    {0x0b0c39cau, 0x0f785edbu, 0xdab9d1aeu, 0x1a4524fbu},
    {0x414b1ca6u, 0xcfe40ce1u, 0xb7a9427au, 0x0be64e45u},
    {0xc805ef2fu, 0xfb921602u, 0x3279083cu, 0xeb4f0cf6u},
    {0x20acd5c7u, 0x0f62fc53u, 0xe0ee4777u, 0x2888f797u},
    {0x21a506dau, 0xf559f5bbu, 0xfbb2fbb2u, 0x28d62be3u},
    {0x4dadced4u, 0x1da15f32u, 0x95975f63u, 0x5fa0108fu},
    {0x22bcf929u, 0xdc2642ebu, 0xefdcfcc8u, 0x1b44167fu},
    {0x05f61e79u, 0x1dfc1425u, 0x11681bceu, 0xeb701acbu},
    {0xf6a70396u, 0x036c29d6u, 0xdb91fd83u, 0x366cec29u},
    {0xfe451b0eu, 0x001820cfu, 0xf71cfee8u, 0x4be92a32u},
    {0x177df9d8u, 0xecd9c3c4u, 0xef670f56u, 0x18a50788u},
    {0x1eea3eefu, 0x1953fe9au, 0xe8bf1000u, 0x035b2ba2u},
    {0x30db4e4du, 0x21ab399cu, 0x3bf5e2edu, 0x468a54e5u},
    {0xdd36100cu, 0x14a314feu, 0x485eed56u, 0x0dcf4fc8u},
    {0x14d7361du, 0x2fb711d1u, 0x4bd314ebu, 0x2e461ef7u},
    {0x17f42f41u, 0x2a36eecbu, 0xf3191e8au, 0x1217e038u},
    {0x0966fbc0u, 0x01651c9cu, 0x262cd050u, 0x3c013154u},
    {0x293c0b1bu, 0x1e8dfc1fu, 0x1e2924a1u, 0x0c5a434fu},
    {0xee11fb78u, 0xcd26407du, 0xd58be2e1u, 0xf31bfaa5u},
    {0x0f7f0668u, 0x4a19bbccu, 0x1ca80996u, 0x44d1162eu},
    {0x25173e44u, 0x0138c6b8u, 0x3cd70616u, 0x3da10392u},
    {0xe5cb588au, 0x4f05e221u, 0x01745268u, 0x07874543u},
    {0x09c75087u, 0x3b1dda6du, 0x385ff8e4u, 0x338537b0u},
    {0xd6080c8bu, 0x2368405bu, 0xfaff339cu, 0x2d651c68u},
    {0x10ae24b6u, 0x3b6adb29u, 0xf6340718u, 0x3c4a2a65u},
    {0x0be9186eu, 0xf29921c2u, 0xddabd985u, 0x489be9b6u},
    {0xe87dc95au, 0xe992d1e2u, 0x0380118au, 0xcbe71a8cu},
    {0xdcaae8a8u, 0x148d069fu, 0x054b045bu, 0x1e41d00au},
    {0x1ece45d6u, 0x45e3520au, 0xf46a3445u, 0x0e8c099fu},
    {0x2d801bcau, 0x0b2a0ee3u, 0xd202b668u, 0x1315cd5du},
    {0x250f3c59u, 0x39ee12c3u, 0xf49623a1u, 0x0758eed8u},
    {0x0c43e39au, 0x15ba0383u, 0x0056f780u, 0x251c42c0u},
    {0xd1a23729u, 0xf219073cu, 0x27733d4cu, 0x395ef32bu},
    {0x2def0c7au, 0xf2394cefu, 0x121102b3u, 0x28c604c6u},
    {0xfebf1cecu, 0x09d1f819u, 0xf793fa43u, 0xe1ca1fa2u},
    {0x4365da90u, 0xfac608e2u, 0x3af52932u, 0x21052d05u},
    {0x47a2ee65u, 0x45e72df0u, 0x2cca25f3u, 0xf82d1564u},
    {0x00b4267au, 0x2a190aafu, 0x0539fccau, 0x47c10371u},
    {0x431ff6b0u, 0x360d4ed8u, 0x460b0547u, 0xf5973e65u},
    {0xfc2909c9u, 0x22440e41u, 0x2137ebd7u, 0x17e8e86eu},
    {0x39c1fb07u, 0x27ea2bccu, 0x07380adeu, 0xef811923u},
    {0x1c43ee99u, 0x38ac4b59u, 0x4a17002cu, 0xed61fc4fu},
    {0xf07ed4feu, 0x1c5a0ff8u, 0xe17c49c1u, 0xd67f1e7eu},
    {0x278824a8u, 0x02090fb9u, 0x2232f2d5u, 0x256a1fc8u},
    {0xf52a166cu, 0x3f8f3081u, 0x4f06f6fau, 0xd64a4bd0u},
    {0x0ec90922u, 0x24e712e8u, 0x5419e123u, 0x1d9b4129u},
    {0x03e0ec76u, 0x420248fau, 0x12b40079u, 0x0c47331bu},
    {0xf0eef462u, 0x0bd42abfu, 0xea61f86eu, 0x29bf15e6u},
    {0x224dd2ddu, 0x2617101eu, 0x38d706a4u, 0xfd423bbau},
    {0xde470d74u, 0xef561cd5u, 0xf3ae0318u, 0x2fc8d147u},
    {0xf5851b1eu, 0x24d5d7c4u, 0x1e5715eau, 0x1bba108fu},
    {0xf9c23a83u, 0x0d04ff23u, 0x09933faeu, 0x11af133du},
    {0x1e904911u, 0xd50c3612u, 0x192d43b2u, 0x3de40342u},
    {0xf5283e34u, 0xf6d41ea4u, 0xf1c03c56u, 0x53a39737u},
    {0xff261119u, 0xd0a9420fu, 0xd6373334u, 0x03ec0ea8u},
    {0xed9ff180u, 0xcc12fcb0u, 0xd5fc1988u, 0xf065ede4u},
    {0x1f9d006fu, 0x28df1409u, 0x053f3b66u, 0x12fdd77bu},
    {0xf0a4f318u, 0x22b72533u, 0x1d5cf44cu, 0x1ae847c3u},
    {0xeb3eebb5u, 0xe7dbdb79u, 0xd9ee1bc7u, 0xe90e0b38u},
    {0xe8700dedu, 0x3413fa15u, 0xdfd8fe44u, 0xef4510dcu},
    {0xe3080689u, 0x271109a1u, 0x017c0349u, 0x3a0616ceu},
    {0x1e950b2du, 0x221434d7u, 0x2734ec5au, 0x52434270u},
    {0x1cc2f756u, 0x402f35c3u, 0x0930f606u, 0x29121309u},
    {0xc8902770u, 0x3bfd08f8u, 0xdec9e757u, 0x09d6e0a4u},
    {0x06ef15e9u, 0x3f961b64u, 0xed8e093eu, 0xda6f16efu},
    {0x2b92084eu, 0x0b0920d6u, 0xd66e1eefu, 0xf77e2e12u},
    {0x769b132fu, 0xf8d83ceau, 0x9e3b7fffu, 0x42852cfdu},
    {0x2fc3102fu, 0x01fb0500u, 0xe0a833d7u, 0x178b37a6u},
    {0xffd72638u, 0x07f72938u, 0xde260284u, 0x20bd39e1u},
    {0xe32ad70au, 0xd5372324u, 0x62eedd28u, 0x1804e492u},
    {0x2470081cu, 0xe2a1e2c8u, 0x1b51fb85u, 0x32912f62u},
    {0x1318e535u, 0x05d3fa65u, 0x5495122au, 0x0348184fu},
    {0x53e1dfe2u, 0x186462d8u, 0xb8985c1cu, 0x36f051acu},
    {0xfad80141u, 0x096efb70u, 0x1bbfd985u, 0x40fa350fu},
    {0xfd596389u, 0x45935ca5u, 0x28952a17u, 0x5b362d7du},
    {0x1714ff6du, 0x39d02037u, 0x4af613ffu, 0x020339aeu},
    {0xff3c0ab5u, 0x2adb03e0u, 0x06521164u, 0xee2046aau},
    {0x152cfa05u, 0xf07113bfu, 0xeb82daf3u, 0xfe8efa6eu},
    {0x0279de70u, 0x23302fdbu, 0x0e23dbdbu, 0xdaeef0e5u},
    {0x200ef177u, 0xe5a3f7bfu, 0x2fb0e0f6u, 0x29741d24u},
    {0x240d27f7u, 0x510f4f9fu, 0x397be44eu, 0x14260f95u},
    {0xd68a3920u, 0x29fa0013u, 0x178f3fd6u, 0x3dc43fa7u},
    {0xe66f2c12u, 0x5162a236u, 0x3af37419u, 0x7ad52d04u},
    {0xf7a930b4u, 0x15220944u, 0x5444169eu, 0x1c5f5079u},
    {0xde9b15b4u, 0x102ca904u, 0x43361a4eu, 0xf1a3fc01u},
    {0xf2e2e2f3u, 0x20ff26a7u, 0x1142e520u, 0xe6e4e3f9u},
    {0xd25c09a1u, 0x2b9c44d1u, 0xef0228d9u, 0x20d31419u},
    {0x0e3634deu, 0x16c53c50u, 0xd2a7fc52u, 0x294fee53u},
    {0x20b33887u, 0x4b5f018fu, 0x39d15649u, 0x206b23deu},
    {0x15c112f9u, 0x3cce456eu, 0xe4c613feu, 0x1fc6ebeeu},
    {0x11dd1fd0u, 0x41d15160u, 0x1e2dea71u, 0x78fdb4e6u},
    {0xfabb3de8u, 0xf8a8267eu, 0x13b00951u, 0x2deea9bcu},
    {0x085d050bu, 0x31613244u, 0xd97fe5cbu, 0x3fa2b0d8u},
    {0xfd73c67fu, 0xf5bb0d1eu, 0x1e2e4380u, 0x19e3436fu},
    {0xd9f7e761u, 0x2078fa05u, 0xff5723a7u, 0xe0d80f2au},
    {0x075a02f6u, 0xe420f25eu, 0x2ad3208du, 0x0c485078u},
    {0x29a31286u, 0x4fe14fbcu, 0xfffb2915u, 0x03fafe63u},
    {0xe77a076fu, 0xea00f5feu, 0x177a1f8cu, 0x05e617e7u},
    {0x2de5d898u, 0x44e77a85u, 0x4ac80a1du, 0x5736393bu},
    {0x1dc31e4du, 0xffc85c6bu, 0x51182396u, 0x48b62c92u},
    {0x2de0cf12u, 0x228af663u, 0x0106ebe9u, 0x01e6220eu},
    {0xf34dcdedu, 0xd40409cfu, 0xe6e5534au, 0xe8ff1e46u},
    {0xf5be0f74u, 0x13eb1330u, 0xf47611edu, 0xf7e523c3u},
    {0xf0fd0e5bu, 0x2bd6efc6u, 0x1aea19cbu, 0xec7f2da2u},
    {0x2d2dd8efu, 0x5b1e4458u, 0x3db903fcu, 0x1d7a1e96u},
    {0x1db20e52u, 0xf25044b3u, 0x153a2190u, 0x2aa74562u},
    {0x55f6d64cu, 0x36f17becu, 0x425aa08fu, 0xfcc037ddu},
    {0x3650e600u, 0x0ed90875u, 0x2156af51u, 0xf593130fu},
    {0x3d850baau, 0x2b3bf506u, 0x0b7bc5dau, 0xe3485db2u},
    {0xda90d104u, 0x0cdcd650u, 0xf13416d6u, 0xda7501ddu},
    {0x30b7deafu, 0xe9392e5cu, 0x329b2252u, 0xeb84030fu},
    {0x1193108eu, 0x2ea8eddcu, 0x2a7010fdu, 0xdd422f35u},
    {0x5dc5d675u, 0x19945c58u, 0x1005b67eu, 0xff021090u},
    {0xd0d5faf4u, 0x213e3047u, 0xe95c3de1u, 0x4056edc4u},
    {0x14ba2b47u, 0x08756d7eu, 0x04257596u, 0x42af97c0u},
    {0xe95329feu, 0x1b444e7fu, 0xd7d209d0u, 0x1480cdb9u},
    {0xf1ae2024u, 0xd96a27e9u, 0xf518319eu, 0x20659434u},
    {0xe727d32cu, 0xfaf5ed83u, 0x18b9dfb0u, 0x0e4d36feu},
    {0x0e50082fu, 0xccd72b76u, 0xd9642043u, 0xf2122760u},
    {0xf944270du, 0xda79f175u, 0xfc7f06ceu, 0x07750acfu},
    {0xdf6a3f27u, 0xe8303071u, 0xf0df55ccu, 0x07f3d8fcu},
    {0x14ea24a3u, 0x44c0f4d6u, 0x1e160497u, 0x00140ac0u},
    {0x1d54760cu, 0x48e36dbbu, 0xe40bfed6u, 0x09446ba5u},
    {0xd038590du, 0x57ca58aeu, 0xf4e20827u, 0x196936adu},
    {0xf4ce0d37u, 0xf2af2253u, 0xe0e5f943u, 0x4d633109u},
    {0xd92e0506u, 0x2125d874u, 0xec9d1277u, 0x2fad050du},
    {0x11640319u, 0xe9212f51u, 0x03f3e1a2u, 0xe091e13fu},
    {0x0ac81d31u, 0xeecefaf4u, 0x2128f25fu, 0x2944ffc2u},
    {0xd8502181u, 0x3242377fu, 0x2353dc96u, 0xe7033828u},
    {0x20830ec2u, 0xf2431305u, 0x2604f926u, 0x247d1b92u},
    {0x119ef40au, 0xd1311b8du, 0xfefef93cu, 0x1c6715cdu},
    {0x3c1cd875u, 0xfc71055eu, 0x149a08c9u, 0x092200f6u},
    {0x1fd21c36u, 0xdfe022dcu, 0xfc14355fu, 0x244d2abeu},
    {0x0e261573u, 0x1b8ae91du, 0x5272e186u, 0xc3fdf19cu},
    {0x2854dea7u, 0x20e5f801u, 0x215b0292u, 0x14ef3b2fu},
    {0xeed80b6cu, 0xdb421adbu, 0x3acc08adu, 0xdf8bf67cu},
    {0xe9fdd493u, 0x0e141ef6u, 0xf7e8020du, 0xf6dc012du},
    {0x1181e5b9u, 0x1eacf685u, 0x0330d0e5u, 0x04cff164u},
    {0x2634e2c0u, 0x2fa12c23u, 0x3327e52du, 0x1061fabau},
    {0xf286f8d7u, 0x0238e3c3u, 0xe630e787u, 0x3c35fd0bu},
    {0x3546510du, 0xf8b20440u, 0x3ff00aaau, 0x17a23115u},
    {0xd8efdd66u, 0x06a80486u, 0xf8771727u, 0x06afd3c4u},
    {0x16f1fb4fu, 0x204f09f0u, 0x2ef7dfc1u, 0x43c303b3u},
    {0xcf0d0dc3u, 0xe64b0f36u, 0xedc6e9d1u, 0xefbd2c62u},
    {0xe1bbee2eu, 0xf714d87fu, 0x0c6be62au, 0x01eee8b8u},
    {0xe5e41861u, 0x00230157u, 0xe57b0330u, 0x04cefd4eu},
    {0x04c8f30au, 0x21aafe46u, 0x14622633u, 0x32123029u},
    {0x0f1e1da6u, 0x1ee107fau, 0x2d1c2a0bu, 0xf7b6209bu},
    {0x12d41850u, 0xfcf10c96u, 0x1b8717e5u, 0xf9d8191du},
    {0xf6d10744u, 0x09282253u, 0x0ef8d0a4u, 0x0a38f3a3u},
    {0x043c13bcu, 0x09c4ed05u, 0x0cb1022fu, 0x1002231eu},
    {0xf910346fu, 0xee30349fu, 0x2e311753u, 0x14fc11bbu},
    {0xf2a92c38u, 0xdf2819f0u, 0xe9c1ecf7u, 0xfc87de5au},
    {0xd518f7fdu, 0x0b6b1effu, 0x04482bb3u, 0xe5543ff7u},
    {0x16770826u, 0xf647e2b5u, 0x18b8272eu, 0xe863f534u},
    {0xfb7b1a57u, 0xfc090f54u, 0xf8f2e163u, 0x2f7be04fu},
    {0x1c0f4715u, 0x41250433u, 0x0339fcfbu, 0x388ac5e0u},
    {0xe41c1a2cu, 0x1b49cc82u, 0xdf1a1fcfu, 0x08a047feu},
    {0xee13109bu, 0xfa92ea82u, 0x28e0e694u, 0xe05fe016u},
    {0x1ca2e1cau, 0x033d0661u, 0xcfd9149fu, 0x0b1aeec7u},
    {0xfcef06f6u, 0xdeec3248u, 0xfe28d28eu, 0xec3cdd16u},
    {0xff3b1ffbu, 0x11f62387u, 0x1b8b4876u, 0xe842f09du},
    {0x006b0fa9u, 0xf76713c7u, 0x0fa60b3au, 0x17f4f848u},
    {0x2f68e2efu, 0xeb3f1825u, 0xf6a6e3dbu, 0x2d4a1188u},
    {0x22b6f045u, 0xf78651e1u, 0x367a2699u, 0x24c718d9u},
    {0x156d037eu, 0x00a3e743u, 0x09041bfdu, 0xd6cd0e76u},
    {0x2fad0914u, 0x0036328du, 0x133c3473u, 0x0b9cfae0u},
    {0xf576d32fu, 0x3ac52599u, 0x0df000efu, 0xfa3ef6abu},
    {0xeef72133u, 0xd701f740u, 0xe5ad1fa8u, 0x2b2eed99u},
    {0x22d7e1f3u, 0x36c5f97du, 0x07654a07u, 0xe7dc325du},
    {0x0caefc5fu, 0xde7623c7u, 0xe10d28c0u, 0x1b6df709u},
    {0x0427d6ecu, 0x2654432bu, 0x0020d6cfu, 0x12793160u},
    {0x13f9d641u, 0x50a85706u, 0x4c64ae3du, 0x03a54025u},
    {0xe6dc0f40u, 0xed68dc3du, 0x17500beau, 0xd329d648u},
    {0xf747f4d5u, 0xf4fc1de1u, 0x0c8e0dc1u, 0x2088378eu},
    {0x2fe8fa76u, 0x0ff8efa8u, 0xef5a26cau, 0xd2a8f5deu},
    {0x27cbe2c6u, 0xd9bf0382u, 0xd576206au, 0x2cab2d6au},
    {0x0a8e1078u, 0xc9d9e373u, 0xe8d4f304u, 0x1b062b3du},
    {0xe2f6ed65u, 0xd8e91b13u, 0xe027fb4fu, 0x154727e4u},
    {0x1db23f07u, 0x037af2ffu, 0x072e076eu, 0x2c4ef628u},
    {0xd4d2011cu, 0x0eb94e36u, 0x0c1f4a45u, 0x2b15d0a8u},
    {0xe5401e9du, 0xccafd668u, 0xe6210459u, 0x00b959c3u},
    {0x07f6ee2cu, 0xe8b8fbbeu, 0xedb4e8d3u, 0xf268e97cu},
    {0x05021b4cu, 0x13ae1ccdu, 0xd9f0df46u, 0x2ce717f3u},
    {0x0bbbfd45u, 0x085321eeu, 0x00f1fb91u, 0x0741d053u},
    {0xcadb1ff2u, 0x08471cefu, 0x1838ddbbu, 0x07ea1b1du},
    {0x1cf2ef4fu, 0x0f1014ffu, 0xd8a604fdu, 0xe175049eu},
    {0xcfb9f947u, 0x424ded71u, 0x133c20c9u, 0xf8492224u},
    {0xda4f29c6u, 0x342347fau, 0x2817f10eu, 0x23a1fee4u},
    {0x1068e1dfu, 0xfd2f01a2u, 0x10bf1c4cu, 0xda6ef617u},
    {0xd5a9fb02u, 0x27b41fa3u, 0x22ca08c3u, 0x115941b8u},
    {0xe5531d1fu, 0x0d06128au, 0x0b9f138cu, 0x15fb2b45u},
    {0x29bed53du, 0xf40eefe4u, 0xdf45f0b6u, 0xf397f44eu},
    {0xf4cb28d0u, 0x04f085a8u, 0x014108e6u, 0x057540d0u},
    {0x01c6301eu, 0xfb641a78u, 0x01b4311cu, 0xfa3b1cd8u},
    {0x022c97acu, 0xf9544940u, 0xfc7bcdd0u, 0x063f9028u},
    {0x05ad0a28u, 0x0bc37f80u, 0x01b213c0u, 0x0c259770u},
    {0x10c27480u, 0x00390694u, 0x05a60088u, 0xfd057eacu},
    {0xfdc0df68u, 0x09408520u, 0xf2d7a420u, 0x022239d4u},
    {0x0040ed4cu, 0x0c623480u, 0xf4aa7fb0u, 0x0129defeu},
    {0x0547a828u, 0x0bd774e0u, 0xfca47cd0u, 0x00669c40u},
    {0xfb562ae0u, 0x0015d5fcu, 0x06c8f1f8u, 0x076beaf8u},
    {0x05d52260u, 0xfb9347d8u, 0xff5844ddu, 0xfd26df94u},
    {0x0aa10f70u, 0x08f3bf70u, 0xf9aeb2c8u, 0xfe24df22u},
    {0xfb8ac340u, 0xfb6cacd8u, 0x08c3fa60u, 0x060bcc60u},
    {0xfe88f51cu, 0xffe41524u, 0x10e6b5a0u, 0xfd680274u},
    {0xf66b7a50u, 0x095ef9b0u, 0x07d46cc0u, 0x0a792600u},
    {0xf4baa950u, 0x08d97d40u, 0x0bdaaa40u, 0x085042f0u},
    {0xf9ab25a0u, 0x0a255ff0u, 0xfadb97c0u, 0x1013b220u},
    {0x1ceae81du, 0x16eacdadu, 0xf972e99fu, 0x00a9d548u},
    {0xca801566u, 0xf1eebe1au, 0xf602e493u, 0xbeb716dcu},
    {0xeb64129cu, 0xd268c327u, 0xd51efaa9u, 0xe1811e6bu},
    {0xd97cffb7u, 0x075ccaf6u, 0xc930d12du, 0x1c14dca1u},
    {0xf61ce1c7u, 0xce8ee730u, 0xea3eb3b1u, 0x034feebbu},
    {0x62c0759fu, 0x3c2b0b33u, 0x5a184926u, 0x33c62e4eu},
    {0x239b0f1fu, 0xfc48dc8cu, 0xe6acfa0eu, 0xe524d43eu},
    {0x14c71d1cu, 0xff81de52u, 0xe353ce5eu, 0xb9feb9f1u},
    {0x20be0ac5u, 0x26d90e7eu, 0x08b6054cu, 0xf818035cu},
    {0xf5d1dda9u, 0xd5c605d5u, 0xeae11fbcu, 0x04fd1d5du},
    {0xfd381010u, 0xda550e28u, 0x1e744c45u, 0x20b1ca8fu},
    {0xf955e4a8u, 0x293bff6bu, 0xfe7bea43u, 0x0b3fd582u},
    {0x04650a83u, 0xb67e55f2u, 0x131801d4u, 0x592ec157u},
    {0x374a2624u, 0x62eecbbfu, 0xe74cae45u, 0xa6767228u},
    {0xe024bac0u, 0xd75e0f64u, 0x3f08f5b2u, 0x14002a3eu},
    {0xed3bc8bau, 0xdbec3cf5u, 0x0aff11b1u, 0xf89ae114u},
    {0xec3f171fu, 0xeb8c1bacu, 0xe1c92056u, 0x1f8bf408u},
    {0x1b2134b2u, 0x2de72b5fu, 0x3ee4fdefu, 0xd47bc2cdu},
    {0xf6effa68u, 0x4e72046eu, 0x3ac11000u, 0xfa67bfebu},
    {0x1ea8ff84u, 0xec12d69au, 0x0291e599u, 0x26252859u},
    {0x07ff0fdeu, 0x0a9450efu, 0x4c2fdffcu, 0xfdf3e0beu},
    {0x522fea05u, 0xbdfefd35u, 0xb2083a16u, 0x55db3111u},
    {0x048704feu, 0x3d500b80u, 0x22491fe2u, 0xcd8fd1c1u},
    {0xefb804fbu, 0x25032967u, 0x18f3fe56u, 0xc75ed764u},
    {0xd823cf49u, 0x19f51d78u, 0x0e0cfc57u, 0x00d00b52u},
    {0x05a31379u, 0x0060f6e9u, 0xda2a0754u, 0xc07b1231u},
    {0x38e3e828u, 0x45f7129cu, 0x0b5d1399u, 0xc7551bf4u},
    {0xf5691d97u, 0x011bfebau, 0x1d02d0d1u, 0x16f4fc7bu},
    {0x4b21c7d2u, 0x57890002u, 0xccce132du, 0xf5b64576u},
    {0xf89b43c8u, 0xc1867434u, 0x4de62d70u, 0x5b86ac40u},
    {0xf14be914u, 0xed5b125au, 0x1a90c141u, 0xf7983716u},
    {0xf8e11f0cu, 0x0d6afde1u, 0x1426cb55u, 0x0cc615f1u},
    {0x2a4e3150u, 0x1a9fde19u, 0x059002c7u, 0xd3e128cdu},
    {0x16e71d07u, 0x3abebb46u, 0x19881d43u, 0x1379d2eau},
    {0x20ba490bu, 0x2303fe7fu, 0x034e038eu, 0xbb03e431u},
    {0x2d2ae7abu, 0xd796f681u, 0xdab422c1u, 0x2c812543u},
    {0x44145cb7u, 0x4868c42bu, 0x35472d6du, 0xdb38e05eu},
    {0xd331f9c3u, 0xd86a526au, 0xfd14e1ffu, 0x478e205fu},
    {0x31e12e6bu, 0x43771f23u, 0x0f3303ebu, 0xd4bef048u},
    {0x11c434ddu, 0x0da1c12bu, 0x08402791u, 0x1ccac187u},
    {0xd82112e8u, 0x20480500u, 0x0fa71ef9u, 0x0f4ef724u},
    {0x272f20afu, 0xf87805ceu, 0x3d1006b4u, 0xd2bd0748u},
    {0x3d36154cu, 0x0df9e616u, 0x40671c87u, 0x07aedb55u},
    {0xe5702da3u, 0xea64cb80u, 0x21102610u, 0x2d0e1d8du},
    {0x16b20f22u, 0x9b34e68du, 0x1b5de59cu, 0xf38514bau},
    {0xd725e3fbu, 0x599770f4u, 0xfd255a70u, 0x7995693bu},
    {0x3c873338u, 0x1c841a77u, 0x13e1c8a9u, 0xc2c60660u},
    {0x08c3f7ecu, 0xd476d2b6u, 0x28b3cb81u, 0x1d380ff0u},
    {0xe1b715c5u, 0x0ee7fb70u, 0x04fe169du, 0xf01e1d08u},
    {0x1560ee55u, 0x04413390u, 0x19981c97u, 0xbae4d11fu},
    {0xbbd91872u, 0xf234400bu, 0x2568fa8du, 0xe344f289u},
    {0x1178ec07u, 0xd25b1038u, 0x2a73d9a9u, 0x0e7e0370u},
    {0xe9d941aau, 0xeeeb40bau, 0x1a8a0220u, 0xe22bfd30u},
    {0x33bfca06u, 0x57a8a65au, 0xf0d57fffu, 0x3560660fu},
    {0xca7421f4u, 0x07d142c3u, 0x13cb14fdu, 0xd5e20ecau},
    {0x0b6230a5u, 0xd86ef5a8u, 0xda1001a1u, 0xbf58fef0u},
    {0xd251f74bu, 0x0b9f356eu, 0xddcc1388u, 0x21ddd350u},
    {0xd86dc4f7u, 0x43b20894u, 0xfa3defaeu, 0xba9dd0e8u},
    {0xc0cfc343u, 0x289a26d2u, 0x2ef119b7u, 0xfac6c733u},
    {0x1917ec53u, 0x1a770279u, 0x34d5218bu, 0xd565f70au},
    {0xdc91c917u, 0x2cdffa10u, 0x428dc49fu, 0xe9c21772u},
    {0x4b8968b3u, 0xa9a8c5efu, 0xc0e14c19u, 0x269213cbu},
    {0x1d461d84u, 0x307f3f20u, 0x2eeb1cc7u, 0xeb8ae43fu},
    {0xde8ae14cu, 0x11f440f8u, 0x110fcc5du, 0xe0fd132au},
    {0xd412ca3bu, 0xe57013d5u, 0x0c190767u, 0xfafec94au},
    {0x0e5503edu, 0xce8be6f3u, 0xce5bd2f6u, 0x2fb4fdc2u},
    {0xbccbfb3du, 0xd006bfabu, 0xd5cb1dd6u, 0xb9ddf016u},
    {0x0b0702bdu, 0xce87efadu, 0xd7d9f863u, 0xc195cb1eu},
    {0xecfadba0u, 0x0b8e2d9bu, 0x2195ff37u, 0xfcf62d0fu},
    {0x13deee39u, 0xdb8ccb7du, 0xd1b6ca01u, 0xd1a4e40bu},
    {0xf302e71au, 0xf73fd429u, 0x1243b032u, 0x0369bde3u},
    {0xd09fd24eu, 0xe564ee69u, 0xc399ffe4u, 0xef0d1599u},
    {0x1ffdb084u, 0xc136fdbbu, 0x2b974f6cu, 0x246ff205u},
    {0x1c5cf193u, 0xf744e317u, 0x2056ffb5u, 0xe4ac0aa6u},
    {0x1542163eu, 0x0bd516c7u, 0x274108f3u, 0xec4c2440u},
    {0xbd82e718u, 0x082f3c78u, 0x096b47a3u, 0x0070fcd6u},
    {0xd859d606u, 0xedf0e113u, 0x12f72dd1u, 0x27bc0c57u},
    {0xdbaa2558u, 0x1d8bfa6bu, 0x20362466u, 0xeb71245du},
    {0xb68ca3a6u, 0x0b2a54e0u, 0x44680de0u, 0x399bdb57u},
    {0xcbbbbed3u, 0x1f3ae49au, 0xf322f7abu, 0x3cd1ff36u},
    {0x04fa4241u, 0x47203e09u, 0x33d10bd2u, 0xe41d036fu},
    {0x1c9c2866u, 0xed3a2ad3u, 0x0582c94au, 0x267c252cu},
    {0xfa5811dfu, 0xe7e8f466u, 0x3fade166u, 0xd022e425u},
    {0xde4d37abu, 0x146c2c34u, 0xf7b9d502u, 0x0a0cec55u},
    {0xf231f443u, 0xd4ecd828u, 0x212b26a5u, 0x229ad20du},
    {0x0c07254bu, 0x2e2b335eu, 0xe127cc0cu, 0xe1ff02e2u},
    {0xe20b2486u, 0x43e52630u, 0x5a3ef65eu, 0xd88907a5u},
    {0x0bc532e1u, 0x36851f9bu, 0xfaa5de41u, 0xe677d302u},
    {0xff7ccb43u, 0x4cfadfd7u, 0xe5ab183bu, 0x0ffe0bfbu},
    {0xed45274fu, 0xf3fce5f7u, 0xe492d352u, 0xf4a53750u},
    {0x1a3dc19du, 0x0adeed43u, 0xdbc4ddebu, 0xb8da2d4cu},
    {0x43fdf22au, 0x4231ccabu, 0x10bcc7ceu, 0xc9d9eef5u},
    {0xea9ad036u, 0xddcad997u, 0xd40ae498u, 0x268a0736u},
    {0x3f88f8e0u, 0x2ab5df92u, 0x082c12d5u, 0xc71025c4u},
    {0x5850cac1u, 0x3e5dd077u, 0xaa8cdd2fu, 0xf59804feu},
    {0x29c6e3f8u, 0x145c21efu, 0x1cb9e24bu, 0xc7fff013u},
    {0x0b9610ccu, 0x28fc1435u, 0x27f9018au, 0x0d3607a8u},
    {0xf7ba10deu, 0xe07cebecu, 0xf1ef162bu, 0x0e5605e7u},
    {0x0ad31587u, 0x3c49cec0u, 0x08861a9bu, 0xe11eea10u},
    {0xfca424e2u, 0xf8bf064eu, 0xe753f73bu, 0xf47dc9e0u},
    {0x31c2f14eu, 0x0a8f2435u, 0xffd6dfd5u, 0xf8cf04d5u},
    {0x41e12007u, 0x1fccf373u, 0x48eafb90u, 0xcb19fbdcu},
    {0x39035d49u, 0x19b1c4d0u, 0x6176008eu, 0xd5e1df66u},
    {0x0678f921u, 0x25d9c40eu, 0xfc212795u, 0xd3fcebc3u},
    {0x011e09c8u, 0x0e35fe35u, 0x6030c7f3u, 0x1829ed54u},
    {0xd02cf64bu, 0xfc8bd9c3u, 0xe04f0b67u, 0xee1bdf40u},
    {0x18c9f706u, 0x1aecf239u, 0xe611c6dbu, 0x0cdadae6u},
    {0x1a5520ffu, 0xf8111195u, 0xeb23dff1u, 0xda90d4fcu},
    {0xe75409f7u, 0x098116bcu, 0xe5d42669u, 0xe77cfd42u},
    {0xdf30040bu, 0x00640e99u, 0x3f18302du, 0x00b9173cu},
    {0x32b4567bu, 0xe72902c1u, 0x54eaa970u, 0xb175b3f7u},
    {0x01ec2b43u, 0x1edc22e0u, 0x1efd19d0u, 0x1a6f0610u},
    {0x13a4529bu, 0xe904369fu, 0xff8dbc4au, 0xc842c9f4u},
    {0x283217a9u, 0xf476edbbu, 0x0ab13cefu, 0x22500ef9u},
    {0xe40a3d71u, 0xdb960f6fu, 0x1070061bu, 0x0792205du},
    {0xcd1ff793u, 0x0fb8ebf6u, 0x19d4dbf5u, 0xfd81f5dfu},
    {0xc92ddde5u, 0xd9b1f7a0u, 0xe80a2de9u, 0x225cd160u},
    {0x128a0ea5u, 0xd1abe62eu, 0x238b3571u, 0x19bf15e5u},
    {0xe328358cu, 0xddcc142au, 0x2aa9a71eu, 0xc239f91eu},
    {0xf02239c3u, 0x23ba2bdau, 0x0af8d964u, 0x035be5ebu},
    {0xedfc0432u, 0x336d162au, 0x5d53c2d1u, 0xb641c841u},
    {0x091f09f6u, 0x182c1b6du, 0x0fdf02d2u, 0x3b8d165bu},
    {0xc6d5e634u, 0x193047ceu, 0xfc2e042eu, 0xe8fc0bafu},
    {0x0a95ff02u, 0x39664591u, 0x4482c0a9u, 0xd360ccc8u},
    {0x1106fe08u, 0xd12c07bau, 0x1d9de7b2u, 0x04eada48u},
    {0xe558e216u, 0xe9d2e593u, 0x33722e1fu, 0xdd5aefe9u},
    {0xf69402afu, 0x43712624u, 0x5c97f6b3u, 0xfaaaca55u},
    {0x1350efe4u, 0x4400efa8u, 0x325413b9u, 0x27add094u},
    {0xc7b3ea1fu, 0xdd2f048fu, 0xc64ecb27u, 0x08cddb5cu},
    {0x2df4cfaeu, 0xd1480ff5u, 0xeaf5df59u, 0xdca8353cu},
    {0x053a2b1fu, 0x5b940d51u, 0x3d9a65fbu, 0x4e1a11dcu},
    {0xe7371d6eu, 0xdddbd8a6u, 0x4a661e4du, 0xec4d3861u},
    {0xbe8fd72eu, 0x0a01c4abu, 0xea40f886u, 0xf257e80fu},
    {0x1fdc0a87u, 0xdcd61092u, 0x10a122e0u, 0x13acea08u},
    {0xe7c90c47u, 0x030e15c3u, 0xc1401781u, 0x1449c9c6u},
    {0x229cd213u, 0xd683ea8fu, 0x2514d0bdu, 0xecc3e3fdu},
    {0x17a019c1u, 0xdc544aa4u, 0xf504195bu, 0x19b5024bu},
    {0xedcc2ad6u, 0xf3fef2f5u, 0xe6b5d689u, 0xe3adfd17u},
    {0x2c831f5eu, 0x5d3debf0u, 0xbf02cd7eu, 0x0f0e601au},
    {0x171b1ebbu, 0xf6cd128du, 0x150f3e69u, 0x468d10e0u},
    {0xfc5fdef7u, 0xdabf1537u, 0x41714a64u, 0x2f34e252u},
    {0xc6a807ddu, 0xf04f3eddu, 0x255516bfu, 0xfc0e0a37u},
    {0xe4d20bbeu, 0x208bfa50u, 0x4e5a1e09u, 0x2b4bf23eu},
    {0x1a9cdad9u, 0xc6d7fe3bu, 0x03fd4236u, 0x2fe8da55u},
    {0xc21724a3u, 0x1f3dfbf8u, 0x4a80c9f8u, 0xc40ace47u},
    {0x00d21e28u, 0x0c27e85au, 0xe9f329ecu, 0xf7f8ff94u},
    {0x45ceb2f3u, 0xf47cd7cdu, 0xf5881d94u, 0x3b8b5db2u},
    {0x133654dau, 0x05312082u, 0x50e102f9u, 0x178c2e43u},
    {0xe11b1534u, 0xf53721bdu, 0xf47b060du, 0x1b8e0d1cu},
    {0xdfbb02fbu, 0x1c2f5462u, 0x14c91b50u, 0xdac5eec6u},
    {0xedee1cc7u, 0x4f5c503du, 0x4de8fe5du, 0xeee1cb56u},
    {0xbdb53b20u, 0x40310498u, 0xef32c505u, 0xd460ce71u},
    {0xe67ed3f1u, 0xed2e10aeu, 0x1317dac9u, 0x1957357du},
    {0x232416c9u, 0xddc2268bu, 0xe602fe9cu, 0x07e801f9u},
    {0xcff43d3fu, 0xc0896740u, 0x325a5b9au, 0x46d8fa5cu},
    {0x3dcacbd4u, 0x1bd2e543u, 0xe1391857u, 0x0d634e1eu},
    {0x0add1044u, 0x0529ebffu, 0x02cae548u, 0xcfcffcd1u},
    {0x2c32d03eu, 0xf854ea21u, 0xedb90a7fu, 0xeb642153u},
    {0xfca7d4dcu, 0x1b0e0087u, 0xd9d61d65u, 0xcf11444cu},
    {0x25561f6eu, 0x2a220ec3u, 0xe064c701u, 0xdec41a46u},
    {0xfeaf4e21u, 0xfe1af154u, 0x3ca82789u, 0x0064cbffu},
    {0xfbdd0c45u, 0xd3242d5du, 0x088aefe9u, 0xfd061cb6u},
    {0xbce4d55fu, 0x19764c3bu, 0xf540ec62u, 0x6a1d60dcu},
    {0x1f533116u, 0x1274dcf5u, 0x20814b6bu, 0xf6dc2188u},
    {0x357e08b0u, 0xf85406cdu, 0xe90316d3u, 0x24c7ee98u},
    {0x02c01266u, 0x3670d6ffu, 0x41271a5du, 0xc57c2625u},
    {0xe94de9a9u, 0x1955176du, 0xeaec4535u, 0xc9fedcf6u},
    {0x3ce0ed8bu, 0x29d8e31fu, 0x24bf2939u, 0x156ef1b8u},
    {0x0d94fdcbu, 0xfa81d626u, 0x5134cbfau, 0xba9ddd41u},
    {0xeba03541u, 0x17051a55u, 0xd2542022u, 0x1b6513a7u},
    {0xfcf5d534u, 0x3aef2997u, 0xec224f14u, 0x61b3350fu},
    {0x26f45a42u, 0xb0c11957u, 0x60181b94u, 0xfd350af1u},
    {0x3c8332e0u, 0xed2919fdu, 0x2009d412u, 0xbf7dce3au},
    {0xff77fdafu, 0xea9d1495u, 0x475e1628u, 0xe8f117b0u},
    {0x1cfe1055u, 0x09e4c396u, 0x319bdf2au, 0x1b23cae6u},
    {0x2a542524u, 0xdfe2dfb8u, 0x2c9c1f72u, 0x1ac9fd3eu},
    {0xef2149aau, 0xd8c11fd5u, 0x07e0f284u, 0xd6f0bec8u},
    {0x0618e5c8u, 0x1a2c1b0du, 0xe8190d2fu, 0xef63d319u},
    {0x64fe029cu, 0x572dfdf9u, 0xbfbf1e2au, 0x4638628du},
    {0xba1e0488u, 0xc87407bcu, 0xfeaef3e8u, 0x1468e9ffu},
    {0xe8cff817u, 0x1b712671u, 0xf6320c72u, 0xcd8dcbb0u},
    {0xd087076eu, 0xd6122c61u, 0x30a70681u, 0xdf2f1d0fu},
    {0x16631c82u, 0x07233db7u, 0x1d96f9beu, 0x158a171du},
    {0x1a8426f9u, 0xbd2e0faeu, 0xd84dcd90u, 0x0ed80da3u},
    {0xe130e378u, 0xff21ff71u, 0x4134bea0u, 0x0e1dd30fu},
    {0x1aded32cu, 0x3205f687u, 0x055feb53u, 0xf01ff0d3u},
    {0x1ac4042bu, 0xc45d091cu, 0xc8891885u, 0x695a127eu},
    {0xc1b6fda7u, 0x3ce50b51u, 0x4941b1d0u, 0xa91922d6u},
    {0xf049fd96u, 0x318e1aadu, 0xf881fcc9u, 0xc0d61d5bu},
    {0xfb3ee09au, 0x1e445011u, 0x31a4db4fu, 0xbe87cbe0u},
    {0x11f7ce26u, 0x447c16b2u, 0x495df0cdu, 0x02a3d0a5u},
    {0xba7d031bu, 0x0753fa6cu, 0x2c3edf87u, 0xe6aecfb7u},
    {0xf15b0420u, 0x1f86d487u, 0xf72fe871u, 0xfffa215au},
    {0x22e5e276u, 0x0d26d276u, 0x0d9df330u, 0x332ccee3u},
    {0xfd1b15bcu, 0x056001fbu, 0xdfb1ea3du, 0xd5beda6cu},
    {0x0760f731u, 0x12e5f839u, 0xdaf8ff4au, 0xe8e8f029u},
    {0x2e411d5fu, 0x2b851916u, 0x45713ecau, 0x2e310b76u},
    {0xfdbc04ebu, 0x0fbb053au, 0x1125f13eu, 0x1cc1f41eu},
    {0x40bd39d8u, 0x195a0eceu, 0x003c607cu, 0x2a7458bcu},
    {0x10bef95au, 0xd725cbc2u, 0x056a1a1eu, 0xc5ac1b27u},
    {0xbf961293u, 0xc2f61608u, 0x3cc0184cu, 0x4d9ff15au},
    {0xf56911b2u, 0x23fb21e0u, 0x144f0ad8u, 0xcdd2254fu},
    {0xba1d0c92u, 0xf24b27d6u, 0x02700444u, 0x4f4ff269u},
    {0x19edc659u, 0x204a021fu, 0x28fe1b3du, 0xfac80552u},
    {0xf1101bc1u, 0x0dc23308u, 0xfdba3392u, 0x3a36faf5u},
    {0xea61156cu, 0xfc92cf7eu, 0xd93b2463u, 0x0d8d30a2u},
    {0x4b9b2a55u, 0x464fcfadu, 0xc7e0bfe6u, 0xb15265f1u},
    {0xdda7e343u, 0xe5ff0c9du, 0x02b6048au, 0xf41adfb2u},
    {0x114c21b7u, 0x0a8e3231u, 0xee1afcceu, 0x059b1d98u},
    {0x2a0f0f8bu, 0xe6c4eb9du, 0x0aa5099bu, 0x22f931b8u},
    {0xd0c4f263u, 0x4c39503eu, 0x4109daa1u, 0xee6dffefu},
    {0x05990e35u, 0x55173e52u, 0x5522ed74u, 0x249b0953u},
    {0xee460534u, 0x15d42660u, 0xff490f1au, 0x40fc52c2u},
    {0x22081647u, 0xd93fd7d2u, 0xf5ed0905u, 0xef44df81u},
    {0x684eb994u, 0xf1dac8d7u, 0xde2b2fd7u, 0x4e384ed9u},
    {0x15a12545u, 0x06c41464u, 0xf14ed0c3u, 0xc25508bbu},
    {0x11a8c665u, 0x46e7c3f4u, 0x14e4cdc3u, 0xe39b12dau},
    {0xd54d0d54u, 0x09172342u, 0xe9f61f54u, 0x15f3e6fbu},
    {0x1d01faadu, 0x0338c5f7u, 0xcf3b01abu, 0xd1af42dau},
    {0x15bbf4abu, 0x0bfeed6eu, 0xbf36dda5u, 0xefa91725u},
    {0xfb9022cbu, 0xe8763520u, 0x1cfe4b91u, 0x2b9c1a65u},
    {0xdc02df22u, 0x14a129b4u, 0xc9d30e00u, 0x1d2d2aeau},
    {0xe1d33afeu, 0xe87a218au, 0x71865f31u, 0x37dcf92fu},
    {0x0eb4b6eau, 0x47590539u, 0xd490f86du, 0x0ce14811u},
    {0xfb94460fu, 0x08970b51u, 0x27462585u, 0x0d0c0316u},
    {0xf38fd0ccu, 0xec38db5fu, 0xe905eb38u, 0x2b0511fcu},
    {0x4daa29c8u, 0x04dff4c8u, 0x341e36a0u, 0xfe57150cu},
    {0x16b0f37eu, 0x40a9ff4eu, 0x49375c0fu, 0xffabe026u},
    {0xf9383d07u, 0xf8b7266au, 0xe326fe01u, 0x019c4cc3u},
    {0xeb3e116fu, 0xf2890663u, 0xe0a8146cu, 0xd1092395u},
    {0x090de19du, 0xe9c34ad9u, 0xc4cce012u, 0x142355cdu},
    {0x031f3c6fu, 0x3ff0f4b7u, 0x04710ec4u, 0xdd451517u},
    {0x29cb4aedu, 0xd16604ceu, 0x48dbec30u, 0x1f3ddafeu},
    {0x0a110a5eu, 0x035a193au, 0xcfc01e0fu, 0x20fa28e6u},
    {0x0e8a0b36u, 0xd57dec8au, 0xf2a3e0abu, 0x1bbdc661u},
    {0x2a915164u, 0x0ff3ceceu, 0x3534fd42u, 0xfb2d180fu},
    {0x1f1c3a0du, 0x4d0835cbu, 0xf2c3488cu, 0xf4d228feu},
    {0x2cf4133eu, 0xd3d130c8u, 0xd8f6f38du, 0xd9a4f81bu},
    {0x9d21aa33u, 0x2eaa2afcu, 0xcf3e7942u, 0x60eb363fu},
    {0x01350299u, 0xe9d7ee4bu, 0x4b87f05eu, 0xcd03ead5u},
    {0xbd2647dau, 0x14e9fb1fu, 0x237be52fu, 0xdb891999u},
    {0x061a1a3cu, 0xd30cd43fu, 0x0710e0a3u, 0x1401ef92u},
    {0x06803033u, 0x097d45fau, 0xf007fdc7u, 0xcd6bc8b9u},
    {0xce3245a8u, 0x1b880c64u, 0x3732ea08u, 0xcaf1de87u},
    {0x21082fc4u, 0x065e30d3u, 0x29322330u, 0xeb4c28b9u},
    {0x2558214cu, 0xd24217adu, 0xd302fa4cu, 0x0144e2b6u},
    {0x5e24f145u, 0x52c1d414u, 0x995620deu, 0x21ee560fu},
    {0xdd483a0bu, 0xd90d3ce2u, 0x0541b160u, 0xe4871641u},
    {0x01f1245du, 0xf0301006u, 0x16bbc88cu, 0x0701d664u},
    {0xf116d131u, 0x17061283u, 0xe1fb1752u, 0x269c2231u},
    {0xd97fc8b3u, 0x34083972u, 0x2dfbef6du, 0xc4bec4d7u},
    {0x13f30ba4u, 0xf7f7114du, 0x2847efa4u, 0xd47b23ebu},
    {0x040b4d94u, 0x022a08afu, 0xde443bdeu, 0x038b1eb3u},
    {0xeea8ff3au, 0x015b0d89u, 0xee072181u, 0x0e56d3d0u},
    {0x60b955bdu, 0xe759bcdau, 0xb7d66723u, 0x7b485169u},
    {0xf7f9f4a7u, 0x4defe9b0u, 0x127a0613u, 0x11a5f98eu},
    {0xcda4198eu, 0xdf3bfdc9u, 0xdcda1d26u, 0x07862097u},
    {0xcbaf0e9fu, 0x035afc67u, 0xc2150e64u, 0xfd3bd049u},
    {0xe89b073eu, 0xc246ea6fu, 0xc61beb0bu, 0xf5bc1d02u},
    {0xe2270b05u, 0xcb29157du, 0x2b52eef7u, 0x1dd91a87u},
    {0x330ee927u, 0x4d393829u, 0x4b952bb1u, 0x30c406beu},
    {0xc76ec68du, 0x146bec3du, 0xe681c3a8u, 0x2277fb1du},
    {0x179ef3c6u, 0xc1e2c54au, 0x03770e77u, 0xc6ebc9f6u},
    {0xeebaf881u, 0xcb2c20f3u, 0x02eff05au, 0x19d4e36bu},
    {0xed4127e5u, 0xd7292c0bu, 0x08ae0f32u, 0xe7bb1da6u},
    {0xc7bcfcc0u, 0xd694e91eu, 0x05660346u, 0x0d39e488u},
    {0xcbf71787u, 0xe0ec3b67u, 0x24754a6bu, 0x0b6c0b47u},
    {0xfb34008du, 0xe2a4ffccu, 0x1c081eb8u, 0xf6eb00efu},
    {0x4eb1f9d7u, 0x3fbff52cu, 0x2c912579u, 0x160639f7u},
    {0x19afe581u, 0xd3b017e7u, 0x00b23cb4u, 0xf6282269u},
    {0x12eee2d7u, 0xce390134u, 0x30b9ff2fu, 0x2b39e82au},
    {0xbbece121u, 0xb9c5f954u, 0x32a2f265u, 0x361dc68cu},
    {0x0f5b13beu, 0xdeca0c6bu, 0xd6cd17b0u, 0x1bc3e110u},
    {0x09340c96u, 0x24ecffc9u, 0x3407d88cu, 0xf0ae1bc8u},
    {0x0f5c312fu, 0x3c05ea27u, 0x24dd13e5u, 0xc2b1e637u},
    {0xdf66e293u, 0x3a3c1686u, 0x3ffee056u, 0x024f0352u},
    {0xf5f53522u, 0x0e43eb47u, 0x0ce31991u, 0x04f352f5u},
    {0xde0d2166u, 0x0673e7e9u, 0x31d0e5aeu, 0x04b8c4dcu},
    {0x06bb387fu, 0xe7da19bau, 0x16521699u, 0xf89ecbbau},
    {0xfc6226e8u, 0x14640876u, 0x22d7c46bu, 0x0680e584u},
    {0x20f7e104u, 0xd371251bu, 0x2d0f1dd4u, 0xe351fe16u},
    {0x387bf229u, 0x4effc820u, 0x17eacfbbu, 0xe0281a0du},
    {0x12e9c983u, 0x0252034bu, 0xd079cd3bu, 0xf4d82d70u},
    {0x27200850u, 0x41b6081du, 0x158ae267u, 0x1ab02898u},
    {0xdc01edeeu, 0x2c210531u, 0x3bbd1198u, 0x4b24e4d8u},
    {0x051a14eau, 0x11bbee23u, 0xe388d340u, 0xfa58e782u},
    {0xf5530c50u, 0x47c4e877u, 0xf619c529u, 0x1e641f90u},
    {0xec1abdc8u, 0x51dcfcbdu, 0xc5cecaf4u, 0x1c192fdau},
    {0xf4b30b29u, 0x0b88dcfau, 0x05feda66u, 0xdec728dau},
    {0x4108311eu, 0xf0471f3cu, 0x2dea0ba0u, 0x036f09c3u},
    {0x31b2379bu, 0x2501c4beu, 0x479d0f45u, 0xf9adfb5cu},
    {0x1716339bu, 0x4210dc61u, 0xf5231d45u, 0xd3a0c9a2u},
    {0xee5f140fu, 0x323f444au, 0x2a3c3408u, 0x03c440f8u},
    {0x2926f60au, 0x19e012d2u, 0xead52abcu, 0xf6b5ca18u},
    {0xf1f93376u, 0xffe9f4f2u, 0x15fd2934u, 0xd501d16eu},
    {0x273f38c0u, 0xfaa4c3d8u, 0x4bf2481au, 0x1685206eu},
    {0xef71d714u, 0xe54f07dfu, 0x074a0005u, 0x03f2d1d3u},
    {0xf2af4441u, 0xf969c924u, 0xf8da0d13u, 0xc107d991u},
    {0x2ccd36b8u, 0xc612d17au, 0x4684fac7u, 0xcce8101fu},
    {0xe94af53eu, 0x3050145fu, 0x27cbfb0fu, 0x0770ff67u},
    {0x388224c4u, 0x0a0eed47u, 0x454618afu, 0x1f852736u},
    {0xf208f061u, 0x0a75ce67u, 0x2bf4f76bu, 0x18a9f16au},
    {0x45722249u, 0xc14d1220u, 0x24fa2c0fu, 0xf5fafe61u},
    {0x3c1deba3u, 0xf2ff1fefu, 0x3e3de6ceu, 0xc3d7baecu},
    {0xfeafe88cu, 0xf216f396u, 0xf884fbcbu, 0x17bee469u},
    {0x0dbb2078u, 0xdc7bf451u, 0xdbe7ff4bu, 0xbdeada3au},
    {0xc4f8f5a1u, 0xf36f0774u, 0x318bde1au, 0xf5a2c84fu},
    {0xc6e4f301u, 0x172af105u, 0x0078136au, 0xfc2fe8eau},
    {0x280b0293u, 0xec8d1455u, 0x00033f53u, 0xefb10b14u},
    {0xc4412aedu, 0x093ae4f1u, 0xecd6253du, 0x176804afu},
    {0xff2b352bu, 0x24e32b73u, 0x0f84fe33u, 0x0379e60du},
    {0xe8064a76u, 0xba502d1fu, 0x42501281u, 0xe622eadeu},
    {0xca24e46du, 0xfc85d1f6u, 0xdc38eb0bu, 0xe212ec3eu},
    {0xf3e6d6e9u, 0x171948cdu, 0xe92e1dd2u, 0x1d8bdf80u},
    {0xdceb1c9au, 0x2b153c0eu, 0xfd5ffc25u, 0xbe060b61u},
    {0xf7cce3d3u, 0x2223df38u, 0x175bed55u, 0x0122e6edu},
    {0x3ed41e98u, 0x2096055eu, 0x3fe12190u, 0xf92d3b42u},
    {0xdb02071fu, 0x412b091cu, 0x0e1ceea3u, 0xe3b90260u},
    {0xcd8d13b7u, 0x2f921fe1u, 0x02ded387u, 0x24b6f429u},
    {0xc10fc189u, 0x4caa2b2bu, 0x47edff56u, 0xbe42ebbcu},
    {0xf554f93au, 0x0b1321d5u, 0xfc161798u, 0xf21ff58cu},
    {0xfa4d1c63u, 0xf2b9cb97u, 0xe1591ad0u, 0x1dfb1894u},
    {0xfdc51e8fu, 0xd6b7e489u, 0x042512d6u, 0xdcb9c624u},
    {0xfd64e8e7u, 0x05a8d0cau, 0xd0aa0d96u, 0xe38ec6bfu},
    {0x424a38f2u, 0x59895164u, 0x44c63573u, 0x2ab94e4au},
    {0xd9e4cd4du, 0xfe56fa70u, 0xffd70392u, 0x10bcc7c0u},
    {0xfa99099au, 0x1c410833u, 0xd3e4d9fcu, 0x0236ee32u},
    {0xd6c30c7du, 0xfb93d814u, 0xcf16d73du, 0x16542367u},
    {0x2bd1fdafu, 0x1cfdf097u, 0xe276fe90u, 0x34271f13u},
    {0x0a03ece2u, 0xc48dfa31u, 0x25ec060eu, 0x46fdd354u},
    {0xf6c3c3b8u, 0xc73b2bd1u, 0xfb942441u, 0x3b6fcd4bu},
    {0xbdc2df46u, 0x13e041a0u, 0x318d2903u, 0xfe11da9au},
    {0x30b71c96u, 0x14b3e757u, 0xec80c5cau, 0x0c865544u},
    {0x130edc7au, 0xd8691478u, 0x4d6755b4u, 0x274cc2c3u},
    {0xf282e1edu, 0xbbde05e2u, 0x22cb4c59u, 0x3635d9c4u},
    {0x0a41ce0bu, 0x04b906bbu, 0xfc1feec4u, 0xf2c7eadbu},
    {0xc9f3e7a5u, 0xf70b1996u, 0x0443de9cu, 0x1954d618u},
    {0xce6bfc51u, 0x2d6b2dedu, 0x1fb7e183u, 0xc463eb33u},
    {0xf4fdfce0u, 0x46713821u, 0x4885d020u, 0xc7551402u},
    {0xd535212bu, 0x2b6510e7u, 0x27bdbd44u, 0x0da0f712u},
    {0x33cdf9d3u, 0xd4d3f7dau, 0xd9f92423u, 0x5cec257fu},
    {0xbfa91bb1u, 0x378813eeu, 0xf623c962u, 0x10b8d2b7u},
    {0xae22f2bfu, 0x1fc60723u, 0xf119eb73u, 0xf955c7e1u},
    {0x11b3d245u, 0x12a3ef32u, 0x20c80609u, 0x2e100505u},
    {0x0d73fa40u, 0x129c160du, 0xd8d6234du, 0x02d5e15eu},
    {0x3967f889u, 0x423d1f11u, 0xd496ecd2u, 0xed082af4u},
    {0x1bd9d8fau, 0x2dc7d350u, 0xe81fd8fau, 0xdcb2f2e4u},
    {0x45e62018u, 0x51c9f646u, 0x1a69d920u, 0x14d72bddu},
    {0xc5ed58cau, 0xf09d305du, 0x4b6322ceu, 0x0c53ac71u},
    {0x27c5dfd8u, 0x2488ff4cu, 0xb9a2005eu, 0x141503f1u},
    {0xf199dbb7u, 0x25b6c04fu, 0x0c561b61u, 0xe3e317d7u},
    {0x1cd8dc9cu, 0x07f1fa0eu, 0x2ef6e813u, 0x1a34df20u},
    {0x33e3dc2bu, 0x1fd722cfu, 0x20d50162u, 0xed9e1c71u},
    {0x1cd8fd4cu, 0x295e19eeu, 0x04840f82u, 0xd457c151u},
    {0x136e252bu, 0x0c01e590u, 0x37c725dau, 0xc17f0b23u},
    {0x2d931cd6u, 0x489c1d6eu, 0x0e0a1c21u, 0x071bfd4fu},
    {0xb62db3c6u, 0xbb0c4526u, 0xb7faaf2fu, 0x65f15ae8u},
    {0x2fb73e93u, 0x4d3fdb14u, 0x55625b47u, 0xf090c3f1u},
    {0x558655ecu, 0x39f3da1fu, 0x38fb50c0u, 0xc598c876u},
    {0xf700d493u, 0xcfcbfc69u, 0xe6ebd0b3u, 0x2bbc2e02u},
    {0x2a280f89u, 0xcf2bcf6au, 0xf708f17au, 0x178ce448u},
    {0x3d32f142u, 0x13411125u, 0x3751ded5u, 0xd62ac667u},
    {0x34cc23a4u, 0xe9fdf33au, 0x0b30f092u, 0xc28ec6f8u},
    {0x362705a4u, 0x177bd654u, 0x35f327c4u, 0x1d03140au},
    {0xe080ef64u, 0x4d2c6d94u, 0xbf94759eu, 0x1dbc0a2du},
    {0xf2d134adu, 0xcd20d49cu, 0xff63ce15u, 0xb9a7b64fu},
    {0x0027f81fu, 0xd8f8d5fcu, 0xf406b8b1u, 0xc756f5e6u},
    {0x2a980c99u, 0xf257d420u, 0x24111d0du, 0xdf73f6afu},
    {0x12d6fb0eu, 0xd233d78fu, 0xe3cf140cu, 0x2c7b19cau},
    {0xedf7ebccu, 0xd442e84fu, 0x31a3d960u, 0xca5de4e8u},
    {0xdf61fb95u, 0xdf0006efu, 0x348c166cu, 0xf1e7cc90u},
    {0xf81e1805u, 0x0c625206u, 0xf6a1ef89u, 0xe2ff1275u},
    {0x3750e0fau, 0x6610b3afu, 0xde481c0fu, 0x5c9061d1u},
    {0x03454948u, 0xf2b91dc6u, 0xfc7a1366u, 0xe339cc9au},
    {0xf6d5f0ebu, 0xe1ce4a98u, 0x3b39ae6eu, 0xca660b92u},
    {0x2670dbe9u, 0x15d82642u, 0xe60e2fd3u, 0xcf962f39u},
    {0xec252acau, 0x20bff000u, 0xefd3e5a1u, 0xffcb2837u},
    {0xf6b01dd6u, 0xf52cf7b2u, 0x4e9a18d5u, 0x08b9e3b8u},
    {0xe3ccca3eu, 0x408f33b7u, 0x058a2025u, 0xd127ffbdu},
    {0xff1cca64u, 0x2a7d2ee1u, 0xed851701u, 0x17a91a7eu},
    {0x30364aceu, 0xea6c080fu, 0xc89d68cdu, 0x49eb3f2eu},
    {0xcda2d398u, 0x0b90164au, 0x1cd6d03fu, 0xe4c1f8f1u},
    {0xe87dd892u, 0x0a653e03u, 0x335ac89fu, 0xf07b1728u},
    {0x0207daa5u, 0x24dd1b77u, 0x076d094du, 0x0cd5335au},
    {0xfeedc6cfu, 0xce80117fu, 0xd929c252u, 0x09e3e973u},
    {0xfb9f1ad2u, 0x1422d1feu, 0x12c6d8f7u, 0xe1dd13dfu},
    {0x101fba47u, 0xcc98ea74u, 0xbcfae54du, 0xef67e4fdu},
    {0xfe3d1157u, 0x0faa0972u, 0xcd3ed755u, 0x1b45f9a2u},
    {0x0ef81986u, 0xbeeef065u, 0xed5ebdf9u, 0xf7121a79u},
    {0x312ad6f2u, 0x15f3e8eeu, 0x14e4d595u, 0xdee0e2f7u},
    {0x7e63421bu, 0x4fb71a43u, 0x54bc3863u, 0x5f242a3du},
    {0xdc8add10u, 0xed0def52u, 0x191deb80u, 0x1260156bu},
    {0xd73d156fu, 0x228a42cdu, 0x4accebe9u, 0x314a02dbu},
    {0xce53e944u, 0xdf57fa5fu, 0xcd2d07e8u, 0xf7282981u},
    {0xfc4df08cu, 0xe7960578u, 0x4130faf2u, 0x1f040479u},
    {0xe05fe96fu, 0x04caecb2u, 0x09190ce8u, 0x0a5c1057u},
    {0x1a08c8f2u, 0xe9e426d7u, 0x2e841f81u, 0x2fe9fb84u},
    {0x063ad65bu, 0x21c50b43u, 0xcd9fe6e0u, 0x1d61e113u},
    {0x38f27607u, 0x1938e3e4u, 0xc02d0694u, 0xca105008u},
    {0xe109ea08u, 0xe7783aeau, 0x4d4be9cau, 0x4e3dd4bcu},
    {0xcc884806u, 0x00454776u, 0x1ea5e062u, 0xe67bdf91u},
    {0x12f5ec7eu, 0x24b7fa06u, 0x2c9b0a32u, 0xfbdb2f5fu},
    {0x07fa2caau, 0x371240afu, 0x3179e510u, 0xe58a0c88u},
    {0xe2b9043du, 0x364021a6u, 0xdee12d99u, 0xef0b12a6u},
    {0xc85f0763u, 0xf0c00b4bu, 0x49a513a7u, 0xf5031e51u},
    {0xdf9f1b7du, 0xef49220bu, 0xeb82fbd2u, 0xf651f64cu},
    {0x2a05a4d2u, 0xdc01a079u, 0x02c30cb9u, 0x2cdd4437u},
    {0x1a222c1au, 0xfdac4ce2u, 0x1fc2159bu, 0x0274f450u},
    {0xf5740114u, 0x3b7002fau, 0x1b6c03c4u, 0x1062436cu},
    {0xd26015deu, 0xd997e1bfu, 0xf45e2204u, 0xeca6e46eu},
    {0xead7b62au, 0x27cce202u, 0xc95e0a35u, 0xd54e4d6du},
    {0x03ef0307u, 0xf133c9fbu, 0x2a4afacdu, 0x098f1539u},
    {0x3245fdbau, 0x3fc31ca8u, 0x0487ec7eu, 0x00a8050du},
    {0xdf3ad9b2u, 0x177f2386u, 0xfd72f5e0u, 0xd4fbcdb3u},
    {0xb9896b03u, 0xb7ea6c05u, 0x49a14983u, 0x0bc3e6bdu},
    {0x1680da58u, 0x0e5be335u, 0xd252edf7u, 0xf62f0804u},
    {0x356d1897u, 0x395e021eu, 0x183d02beu, 0xc751e712u},
    {0xe06a20f1u, 0x002f2b06u, 0xf0d716d3u, 0xfb2a2708u},
    {0x3e2c1f7fu, 0x4625d22cu, 0x32e53e68u, 0xd9740747u},
    {0xf2f8dc63u, 0xf92f2c4fu, 0x36d1e4d6u, 0xe718f22cu},
    {0x3cb3f619u, 0x45bf0f6du, 0x30c012b4u, 0x0a29dc60u},
    {0xf19df556u, 0x03300fd9u, 0xea190e4eu, 0xd6a7d266u},
    {0xddb2cb5cu, 0xb3196376u, 0xbb11d641u, 0x25dd2518u},
    {0x48f1fba3u, 0x472ec60au, 0x05a2f5f7u, 0xc84a00c5u},
    {0x20b30586u, 0x0986ecedu, 0xe7e80e5du, 0xd1e5ced0u},
    {0x11360b4cu, 0x10a92db4u, 0xe6b61810u, 0x139dd8b5u},
    {0x2ea7f664u, 0xfe3bc1bdu, 0x2d38fc02u, 0x00dbd4adu},
    {0xf0b2fbc5u, 0xf8132cccu, 0x1423f735u, 0x0d1e049fu},
    {0x4280222bu, 0x06fd12dcu, 0x4dfc2119u, 0xf719f2e3u},
    {0x16be0486u, 0x1552d303u, 0x228fea4fu, 0xd022e5c7u},
    {0xfb3bc04au, 0x733b2aa3u, 0xaffe6307u, 0x590c3f0eu},
    {0x192c3494u, 0x0260bd2eu, 0x4401def3u, 0xd1f10ec8u},
    {0xe627f894u, 0xd7790e5au, 0xf6dae127u, 0x0a19d744u},
    {0xe00fe5eeu, 0xd44232d8u, 0xedef124cu, 0x1483f752u},
    {0x16be3e8eu, 0xbcb8138eu, 0xed0fc62au, 0xcbd4dca2u},
    {0x112f33bdu, 0x0dd9ff22u, 0x0868edd0u, 0x02bc1b66u},
    {0xb028490eu, 0x05314077u, 0xee14c283u, 0xf4a6c807u},
    {0xf18f1dacu, 0xdb0b1395u, 0x06272927u, 0xfbfbfa34u},
    {0x29d9e7d2u, 0x7ae0b0edu, 0xf1076c97u, 0x64e42472u},
    {0xf29c066du, 0xe63a28c6u, 0x045af85fu, 0xe582c290u},
    {0x10d40d19u, 0x2d59213du, 0x49ff0190u, 0xfe45e660u},
    {0xcda51f57u, 0xe525d0bbu, 0x23a12d08u, 0xd1342f19u},
    {0xedc5cf94u, 0x1d68fc4du, 0x41b5ff84u, 0xa8f7fb38u},
    {0x1b752e1fu, 0xf696180eu, 0x00a6e823u, 0x0e9ed8d4u},
    {0xe449212eu, 0x144c3b01u, 0x42ddfb54u, 0xb847c80bu},
    {0x1695da04u, 0x0e1e20d4u, 0xd7f4e35au, 0x1338dc27u},
    {0x43f5159cu, 0xd5b9c467u, 0xf4605e69u, 0x26105f3au},
    {0x0fb92096u, 0x24d64be9u, 0x1574c89au, 0xff7bd241u},
    {0xfeebfd0eu, 0x0fc1de43u, 0xdd810fcbu, 0x17ead42au},
    {0x17180e42u, 0xe79eead7u, 0xf0c6d570u, 0xd015d98fu},
    {0x078c0b81u, 0xe1b3c0a1u, 0xe3f3c365u, 0xfbfdc2ceu},
    {0x1af9d4e9u, 0xe7c7f614u, 0xffc5090fu, 0xd8b6e58bu},
    {0xd77e0f3fu, 0xde161018u, 0xd38dd49du, 0x20dbf39cu},
    {0x0177e350u, 0x2ca00fbcu, 0xdc5bf5d0u, 0x163119a7u},
    {0xe424d0d2u, 0xc12ef75bu, 0xfc83efb2u, 0x01c9f86eu},
    {0x1e7ac92au, 0xe5020251u, 0xe9592367u, 0xf6a40fcau},
    {0xf7c10c28u, 0x166c20c9u, 0x1bda4882u, 0x16cfd774u},
    {0xc4c4f3beu, 0x1cee3759u, 0xdb930cc5u, 0xdb45d1bbu},
    {0xffeed65du, 0xc4f4313cu, 0x525b50e2u, 0x0bb5e9a8u},
    {0xb861f034u, 0xf6383be6u, 0x41b63031u, 0x4d5cbbd6u},
    {0xec0300beu, 0x0a7533b3u, 0x0126e125u, 0xd1d8d6c1u},
    {0xfb8a2ecfu, 0xd5a3dc9au, 0x23e1228eu, 0x0e852b5du},
    {0x23a4dcacu, 0x00eb23e8u, 0x45f8f2aau, 0x41c9eec0u},
    {0xf35eca9bu, 0xdc88380du, 0x417cf79bu, 0xf12104cfu},
    {0xb944f15du, 0x451a1a4fu, 0x303a1806u, 0xda9ed877u},
    {0xdb29f9fcu, 0xe6c6d9c5u, 0xe2b2db1au, 0xf046c504u},
    {0x086f0929u, 0x49d24982u, 0xfbf60b53u, 0xef05264fu},
    {0xbddb15bdu, 0x4d711fbfu, 0x26e51034u, 0xf83cf538u},
    {0xe891f660u, 0xf8d8f90eu, 0x18aa151du, 0x278e0687u},
    {0x2be6db26u, 0xe2c6fa12u, 0xda342509u, 0xe268240eu},
    {0x08fefcd6u, 0xf3664da9u, 0xf0e2ee9bu, 0xc532e65cu},
    {0xfb1c487fu, 0xff944436u, 0x2a35edd8u, 0xcf4bdd37u},
    {0xef05c230u, 0x25abfe2bu, 0x21ede1f5u, 0xe77c33c9u},
    {0xdc3a0c00u, 0x2e9411beu, 0x21e204d9u, 0xdc611499u},
    {0x1bf10694u, 0x4e5dc8c9u, 0x055ee7f2u, 0xbea026e2u},
    {0x339fbf37u, 0x4ce1e3b7u, 0xe115cde7u, 0xfa10ee46u},
    {0x025e2d11u, 0x11f4e41au, 0xe581e4d4u, 0x23f3f6efu},
    {0xfe4af2edu, 0x2d03fbafu, 0xe676fb43u, 0x2a43d0dau},
    {0x13102792u, 0x0218246fu, 0xce77cadbu, 0xff62300au},
    {0x2f73010bu, 0x4ca6fd67u, 0x0fbad2b5u, 0x0d9ff3a9u},
    {0x1c5315e3u, 0xf481fbb5u, 0x4f97f9b0u, 0xe429ea8eu},
    {0x050906e9u, 0x0d56f3f8u, 0x388ce1d9u, 0xeeeaf4b4u},
    {0x0d133c75u, 0x37efcef1u, 0x33353355u, 0x04d9ec27u},
    {0xfd514efau, 0x286cdc8bu, 0x17ea1d7cu, 0x0255c2c3u},
    {0xef51ec88u, 0xd14c27f0u, 0xd64b2cb3u, 0xeec62a3du},
    {0xf2cad8d0u, 0xdc362c29u, 0x02c51068u, 0xd0a60472u},
    {0x2ae133e9u, 0xf0af11d8u, 0x48d049dcu, 0xef6afb22u},
    {0x27c41de4u, 0xfbd8dca2u, 0xee47e80au, 0xede2095du},
    {0x2cadfd26u, 0xefde0993u, 0x47a7be1bu, 0xcc09fc97u},
    {0x00a827efu, 0xef19017eu, 0x1e790204u, 0xfea2ccc5u},
    {0x0e3ff4fdu, 0xbe9ced06u, 0x2fa5cb5du, 0xeb02157cu},
    {0xe89b3cb7u, 0x13d4c6c6u, 0x48d5fa67u, 0xf0adde94u},
    {0x27750e1du, 0x164c15f1u, 0xf42620dbu, 0x28b90348u},
    {0x2cc0de12u, 0x22c3e59fu, 0xd157dfbau, 0xeead17f5u},
    {0x17510673u, 0xcbf12857u, 0x03d90707u, 0xf3acc4c4u},
    {0x0b890f11u, 0xe79b0d3au, 0x3d9900fbu, 0xc19ff9e4u},
    {0xf829392eu, 0xef600d24u, 0xea071b0bu, 0xfde8c21du},
    {0x25f2e4c8u, 0x02ce212bu, 0xc86dfe97u, 0x27bb07d7u},
    {0xe6d0ff68u, 0xfdfb18c6u, 0x37f9c670u, 0x086a153cu},
    {0x02771301u, 0x045eefedu, 0x262dc72fu, 0xc841cc79u},
    {0xda82d750u, 0x10fb035fu, 0x276e2cddu, 0xcf281902u},
    {0xcbe005a5u, 0xd600fed5u, 0x0a8e26c8u, 0x20c00e21u},
    {0x1bd14bb7u, 0xd8850c1au, 0xdbd1e29bu, 0x27260e37u},
    {0x0c99fb8bu, 0x123d2c7du, 0x09fc1cefu, 0x0219d2dcu},
    {0xf824f70du, 0x06572d34u, 0x4678f9abu, 0xbabfe7b1u},
    {0xcfe0d2b0u, 0xf5ebe2e5u, 0x216f1910u, 0xca04f0e3u},
    {0xf7331c35u, 0x260d0665u, 0xf63e0a7bu, 0xf1d6de6du},
    {0x172cc259u, 0x143ef097u, 0x1c671a88u, 0xd82c0c48u},
    {0xe71715a1u, 0xf63bd7bbu, 0x1daf19aeu, 0x0cc52588u},
    {0xe8550963u, 0x1ee10a18u, 0xf41defafu, 0x11c7fdceu},
    {0xcdd91728u, 0x43d529ecu, 0x0a5ad19au, 0x1a7c0497u},
    {0x1fb228a6u, 0x32f0f4bcu, 0xec29e940u, 0xf8dbc6d0u},
    {0xfd54f07cu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00003973u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x000052f0u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000189eu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x000043a8u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000560eu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x000055c8u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00005f9bu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00005c60u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00002bd1u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000626cu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000ac63u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x000051d2u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00009efau, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000ac2fu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00003e31u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00009887u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00009cacu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x000046ddu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000c8b2u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000c0c9u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00004ec8u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00009ff1u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00004aebu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x000041b1u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x000049c1u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00009beau, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x000027d1u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000c448u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00005cc0u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00004d9bu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000c164u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00005cf4u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000ad5cu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000c7f9u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00002e66u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000b562u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000cba1u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00009d5eu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x000060e5u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000594eu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000a676u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000d7c4u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00007070u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00004610u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00007fffu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000b076u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x000035c6u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000471fu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000b741u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x000038cbu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00008ae0u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00006947u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00003385u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000ee36u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00004104u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00003773u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000566fu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00002720u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00009957u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000a8a5u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x000064bdu, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x0000ba39u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00004970u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00002f22u, 0x00000000u, 0x00000000u, 0x00000000u}
};

static uint32_t im_segment_000[10][TINY_BUFFER_WORDS_32] __attribute__((section(".data"))) = {
    {0x00000000u, 0x00000000u, 0x01881d00u, 0x0021ca71u},
    {0x08000800u, 0x00000100u, 0x1006f200u, 0x2006b200u},
    {0x00000000u, 0x00000000u, 0x01881d00u, 0x101f4b37u},
    {0x08000802u, 0x00000100u, 0x20073200u, 0x2006f202u},
    {0x00000000u, 0x00000000u, 0x03883100u, 0x20229cb9u},
    {0x08000804u, 0x00000100u, 0x30067200u, 0x20073204u},
    {0x00000000u, 0x00000000u, 0x02881c00u, 0x301d7917u},
    {0x08000106u, 0x00000100u, 0x3206b200u, 0x20067206u},
    {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00000000u, 0x00000000u, 0x00000000u, 0x10000000u}
};

int main(void)
{
    uint32_t cycle_t0 = 0;
    uint32_t cycle_t1 = 0;
    uint32_t cycle_segment_t0 = 0;
    printf("TinyNPU bare-metal program: cv32e40p_iszero_mlp_demo\n");
    tb_timer_reset_counter();
    cycle_t0 = read_mcycle32();
    load_ub_image(0u, tinynpu_static_ub_image, 1650);
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("preload.ub_image", cycle_t0, cycle_t1);
    cycle_t0 = read_mcycle32();
    load_im_image(0x8000u, im_segment_000, 10);
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("preload.im_segment_000", cycle_t0, cycle_t1);
    printf("HostOp quantize: q_in\n");
    cycle_t0 = read_mcycle32();
    host_quantize(&q_in, &x, 3.0279148631962016e-05f, 0);
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("hostop.q_in", cycle_t0, cycle_t1);
    printf("NpuSegment: segment_000\n");
    cycle_segment_t0 = read_mcycle32();
    cycle_t0 = read_mcycle32();
    write_tensor_to_npu(&q_in, 0x06b2u, "A", 2, 64);
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("segment.segment_000.stage", cycle_t0, cycle_t1);
    cycle_t0 = read_mcycle32();
    if (npu_run(0x8000u) != 0) return EXIT_FAILURE;
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("segment.segment_000.run", cycle_t0, cycle_t1);
    cycle_t0 = read_mcycle32();
    read_tensor_from_npu(&sigmoid, 0x06b2u, "C", 2);
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("segment.segment_000.readback", cycle_t0, cycle_t1);
    print_cycle_delta32("segment.segment_000.npu", cycle_segment_t0, cycle_t1);
    printf("HostOp dequantize: dq_out\n");
    cycle_t0 = read_mcycle32();
    host_dequantize(&dq_out, &sigmoid, 3.051850947599719e-05f, 0);
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("hostop.dq_out", cycle_t0, cycle_t1);
    printf("Final outputs:\n");
    print_tensor(&dq_out);
    if (!tensor_matches_expected(&dq_out, &dq_out_expected)) {
        printf("verification failed: dq_out (dq_out)\n");
        printf("meta actual dtype=%d elems=%d expected dtype=%d elems=%d\n", dq_out.dtype, dq_out.elem_count, dq_out_expected.dtype, dq_out_expected.elem_count);
        print_tensor(&dq_out);
        print_tensor(&dq_out_expected);
        return EXIT_FAILURE;
    }
    printf("All outputs matched expected tensors\n");
    return EXIT_SUCCESS;
}
