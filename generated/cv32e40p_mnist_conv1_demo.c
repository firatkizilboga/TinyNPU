#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NPU_BASE 0x30000000u
#define TINY_ARRAY_SIZE 8
#define TINY_BUFFER_WORDS_32 4
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

typedef struct {
    const char *name;
    void *data;
    TinyDType dtype;
    int rank;
    int shape[4];
    int elem_count;
} TinyTensor;

static volatile uint8_t *const npu = (volatile uint8_t *)NPU_BASE;

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
    int saw_busy = 0;
    npu_write32(REG_ARG, start_addr);
    npu_write8(REG_CMD, CMD_RUN);
    npu_doorbell();

    for (int poll = 0; poll < 200000; ++poll) {
        status = npu_read8(REG_STATUS);
        if (status == STATUS_BUSY) {
            saw_busy = 1;
        }
        if (saw_busy && status == STATUS_HALTED) {
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

static int32_t x_data[784] __attribute__((section(".data"))) = {
    -126, -39, -21, -93, -41, -126, -67, 69, -24, 78, 61, 10, -16, 19, 39, 4, 81, 44, 122, -48, 28, -90, 89, -23, -78, -128, 123, -101, -102, -8, 73, -70, 127, -95, 1, -117, -65, -113, -24, -17, 55, 71, -57, -48, -112, -85, 120, 0, -11, 100, -40, -37, -107, -115, 16, 8, -45, -61, -77, -70, 39, -40, -60, 44, 127, 22, 75, -36, 121, 5, 51, -118, 18, 20, -21, 76, -42, 123, 58, 108, -93, 103, -11, 7, -54, 1, 7, 31, -21, 2, 10, -85, -30, -122, 54, -91, 102, 65, -32, -99, 26, -61, -79, 98, -11, 117, 38, -1, -59, -50, 6, -5, 113, 97, -83, -81, -58, -13, 127, -90, -91, -41, 11, -80, 105, -76, 90, 99, 59, -32, -81, 34, 4, -108, 117, -72, 106, -40, -98, -122, 78, 96, 2, 79, 13, 108, -76, 124, -77, -72, 121, -29, -90, 18, 121, -17, 32, -120, 119, 91, -91, -84, -57, -110, -34, 65, 8, 82, -42, -72, 11, -104, -108, -127, 89, 86, -110, -62, 36, 100, 73, 84, 5, -29, -70, 67, 0, 58, 2, -74, 49, -97, -118, 12, 7, 21, 44, 96, 42, -43, -18, 102, 70, -55, -19, 81, 80, 61, 31, -54, 69, -32, -4, 20, 80, 14, -23, -85, -60, 86, -92, -110, -73, 38, -92, -36, 112, 115, 74, -27, -24, -53, 85, -76, 94, 59, -67, -60, 68, 13, -73, -111, -75, 103, 21, -1, -64, 50, 114, -14, -105, 23, 59, -97, -87, -108, 66, -26, 89, 76, 97, -26, -37, -84, 10, -29, 113, 51, 40, -62, 83, -15, -42, 12, -36, 93, -127, -94, -67, 27, -124, -126, 59, -95, -82, -61, 0, -19, 68, 104, -114, 112, -25, 121, 10, -12, -30, 50, 44, 127, -42, 12, -56, 108, 14, -14, 79, -98, 15, 89, -91, -2, -45, -102, -7, -14, 84, 4, -124, 117, -102, -43, -124, -63, 74, 104, 24, 127, -8, 107, 0, -121, 127, 58, 64, -55, -65, 19, -30, 106, 9, 72, 29, -68, -67, -65, 117, -74, -6, -20, 118, -18, 40, 47, 57, 29, 92, 33, -68, -70, 27, -8, -89, -19, -40, -85, 2, -9, 33, -33, -54, 3, 78, 30, -124, -35, 117, 21, -103, -5, -41, -45, -71, -11, 24, 92, 101, 38, 62, -124, 51, -8, 70, 94, -47, -29, -49, 98, 23, -46, -73, 58, 125, 122, 51, 83, -54, 59, 22, -28, 11, -51, -128, 35, 120, -20, -99, -8, 29, 68, 69, 109, -102, -119, 84, -88, 63, 113, 17, 38, 38, -18, -59, -79, -24, 10, 90, -48, -62, -69, -9, 46, -112, -99, -98, -128, -13, 7, 13, -19, -55, -72, -118, 79, 8, 4, -3, 23, 19, -62, 34, 38, 39, -125, -126, -26, -58, -19, -11, 68, -56, 16, 10, 126, -111, -41, 86, -97, -104, 17, -100, -90, -78, 88, 52, 57, 14, 110, 8, -33, -40, -9, 46, 36, 109, -113, -100, -62, -13, 66, -23, 69, 117, 61, -111, 53, -72, -29, -109, 55, -108, 57, 9, 32, -47, 31, -72, -57, 11, 73, 78, -51, 41, -106, 67, -37, -32, -115, 85, -61, 52, -49, -23, 15, 90, -113, 26, 116, -7, 73, 82, 115, -109, 105, 78, 125, 96, -47, -103, -45, -70, -96, 108, -75, 28, -11, 126, -93, 73, 94, -64, 21, -14, -115, -127, 59, 106, -104, -64, 6, -100, -39, -60, -15, -64, 37, -26, -47, 15, 104, -60, 10, 18, -24, -80, -16, 110, 11, -74, -105, -32, -46, -67, -54, -101, 106, -123, -36, 8, -44, 87, 29, 97, 2, -73, 26, -86, 93, -126, 28, 25, 20, -13, 66, -65, -89, -128, -122, -7, 70, -109, -13, 1, -77, -17, -94, -8, -15, -99, -55, 21, 32, -73, 9, 28, -35, 54, 104, 10, 15, 71, 122, 45, 97, -21, 110, -84, 110, 27, 80, 62, 125, 56, 125, 111, -7, 63, -79, -78, -58, -91, 2, 70, 79, 5, -5, 6, 69, -26, -126, 114, -36, 91, -93, 88, -31, 18, 42, -87, -126, -55, -94, -61, 88, -8, -110, -25, -2, -122, -68, -83, 80, -72, 16, 61, 46, 38, 69, 28, -31, -74, 69, 54, -49, -6, 70, -52, -86, 66, -122, -104, -11, 77, -112, -95, -20, 117, -10, 54, -65, 98, -96, -19, -36, -61, -1, 2, 63, 63, -112, -69, 36, -73, 26, 44, -85, -99, 12, 61, -74, -14, -46, -5, 42, -109, -82, 61, 101, 95, -28, -123, -23, 106, -109, -11, -113, -115, -75, -46, -108, 22, -66, 125, 49, 82, -56, 100, -122, -87, -64, -47, -123, -73, 78, 124, 22, 125, -122, -104, -5, 108, 9, -5, -110
};

static TinyTensor x = {"x", x_data, TINY_DTYPE_INT8, 3, {28, 28, 1, 1}, 784};

static int32_t im2col_for_npu_data[7056] __attribute__((section(".noinit")));

static TinyTensor im2col_for_npu = {"im2col_for_npu", im2col_for_npu_data, TINY_DTYPE_INT8, 2, {784, 9, 1, 1}, 7056};

static int32_t kernel_t_data[144] __attribute__((section(".data"))) = {
    -52, 2, -72, 69, 0, -103, -33, 50, 9, 60, 38, 7, 6, 36, 62, 57, 7, -52, 4, 36, 29, -76, 15, 38, -9, 22, -3, -21, -17, 16, 27, 81, 0, -43, 67, 17, 43, -89, -20, 42, -12, 53, 6, -48, 29, 52, -12, 67, 2, 2, -66, 51, -7, 22, -28, -16, -43, 3, 8, 1, 13, -11, 53, -3, -25, -39, 4, -3, 7, 27, -20, 20, 17, 44, 16, -57, 12, 27, 40, -21, 10, 43, 55, 11, 57, -5, 27, 30, 20, 56, 36, 32, 13, 26, 25, 24, 7, -52, -13, 49, -24, 59, -8, -106, -29, -32, 54, -27, -37, -14, 61, -91, -16, -53, 36, 26, 26, 38, 28, -68, 34, 29, 60, -16, 13, -15, 25, -112, -3, -9, -15, -33, 60, 16, 12, -127, 21, 53, 69, -58, -39, 1, 103, -35
};

static TinyTensor kernel_t = {"kernel_t", kernel_t_data, TINY_DTYPE_INT8, 2, {9, 16, 1, 1}, 144};

static int32_t npu_matmul_data[12544] __attribute__((section(".noinit")));

static TinyTensor npu_matmul = {"npu_matmul", npu_matmul_data, TINY_DTYPE_INT8, 2, {784, 16, 1, 1}, 12544};

static int32_t _tensor_constant0_data[16] __attribute__((section(".data"))) = {
    18435, 11042, -6, -8, 1904, 3220, 18664, 8333, 12136, -825, -5, 10707, 16475, 7780, -4, 684
};

static TinyTensor _tensor_constant0 = {"_tensor_constant0", _tensor_constant0_data, TINY_DTYPE_INT32, 2, {1, 16, 1, 1}, 16};

static int32_t add_bias_data[16] __attribute__((section(".data"))) = {
    18435, 11042, -6, -8, 1904, 3220, 18664, 8333, 12136, -825, -5, 10707, 16475, 7780, -4, 684
};

static TinyTensor add_bias = {"add_bias", add_bias_data, TINY_DTYPE_INT32, 2, {1, 16, 1, 1}, 16};

static int32_t add_data[12544] __attribute__((section(".noinit")));

static TinyTensor add = {"add", add_data, TINY_DTYPE_INT8, 2, {784, 16, 1, 1}, 12544};

static int32_t reshape_data[12544] __attribute__((section(".noinit")));

static TinyTensor reshape = {"reshape", reshape_data, TINY_DTYPE_INT8, 3, {28, 28, 16, 1}, 12544};

static int32_t reshape_expected_data[12544] __attribute__((section(".data"))) = {
    58, 50, 0, 0, 0, 0, 44, 32, 14, 0, 0, 48, 34, 12, 0, 36, 46, 42, 18, 0, 20, 0, 61, 28, 53, 7, 0, 27, 38, 23, 0, 23, 43, 13, 3, 2, 0, 9, 48, 27, 33, 0, 0, 30, 47, 11, 0, 0, 55, 30, 0, 0, 8, 10, 48, 0, 22, 0, 14, 16, 15, 12, 25, 0, 40, 12, 15, 0, 0, 0, 55, 41, 44, 0, 0, 36, 54, 9, 0, 0, 59, 29, 0, 3, 0, 8, 43, 0, 9, 0, 0, 35, 19, 7, 0, 2, 51, 57, 38, 0, 4, 0, 63, 91, 46, 0, 0, 66, 58, 26, 0, 43, 47, 35, 0, 0, 0, 0, 37, 66, 26, 0, 0, 29, 43, 29, 0, 36, 52, 66, 1, 2, 0, 0, 43, 102, 23, 0, 0, 64, 66, 28, 0, 63, 47, 51, 6, 0, 8, 0, 42, 75, 34, 9, 0, 33, 50, 36, 0, 51, 43, 42, 0, 0, 9, 0, 40, 62, 31, 12, 0, 31, 58, 27, 0, 33, 46, 30, 0, 0, 10, 9, 42, 10, 27, 5, 4, 19, 40, 19, 18, 5, 45, 25, 4, 0, 23, 13, 55, 0, 40, 14, 20, 20, 39, 18, 20, 0, 45, 15, 15, 15, 4, 21, 52, 17, 34, 0, 12, 29, 46, 19, 0, 0, 48, 24, 0, 11, 0, 15, 37, 29, 18, 0, 0, 26, 40, 22, 2, 4, 51, 53, 7, 6, 0, 0, 44, 86, 25, 0, 0, 56, 61, 28, 0, 43, 47, 48, 1, 0, 0, 0, 36, 88, 25, 0, 0, 39, 55, 34, 0, 51, 48, 61, 0, 0, 41, 0, 48, 36, 37, 40, 7, 24, 46, 36, 32, 46, 32, 6, 1, 1, 12, 18, 45, 24, 45, 21, 8, 7, 57, 22, 10, 0, 53, 21, 0, 33, 0, 29, 40, 0, 7, 0, 18, 30, 35, 11, 29, 0, 42, 14, 0, 0, 8, 12, 45, 0, 37, 1, 8, 1, 28, 18, 16, 0, 51, 36, 18, 15, 17, 9, 62, 21, 36, 4, 12, 50, 51, 15, 0, 0, 44, 9, 7, 0, 0, 19, 42, 20, 31, 0, 1, 12, 34, 25, 0, 0, 48, 35, 0, 11, 0, 0, 32, 62, 10, 0, 0, 44, 55, 13, 0, 27, 53, 44, 0, 0, 0, 0, 34, 74, 11, 0, 0, 52, 46, 12, 0, 47, 60, 83, 21, 0, 23, 0, 62, 71, 39, 3, 0, 65, 44, 30, 0, 73, 33, 17, 14, 0, 4, 0, 47, 51, 55, 8, 0, 7, 49, 29, 0, 11, 54, 35, 0, 19, 0, 11, 44, 5, 13, 0, 2, 41, 42, 9, 9, 0, 53, 66, 0, 0, 0, 28, 44, 26, 23, 0, 0, 64, 46, 4, 0, 0, 67, 59, 45, 0, 0, 30, 64, 56, 38, 0, 0, 53, 49, 15, 0, 12, 48, 46, 0, 0, 0, 29, 40, 55, 28, 0, 0, 40, 45, 9, 0, 10, 57, 85, 0, 0, 15, 22, 53, 29, 31, 1, 0, 60, 48, 16, 0, 18, 45, 28, 2, 0, 0, 67, 55, 19, 48, 0, 0, 26, 43, 0, 0, 0, 59, 64, 0, 3, 0, 58, 40, 3, 7, 0, 0, 63, 44, 0, 0, 0, 60, 28, 28, 0, 0, 25, 48, 8, 32, 0, 0, 9, 38, 13, 0, 0, 59, 25, 0, 0, 22, 11, 66, 0, 35, 0, 3, 21, 25, 2, 6, 0, 33, 0, 13, 7, 5, 0, 49, 0, 44, 2, 22, 9, 38, 20, 0, 0, 58, 0, 17, 8, 18, 0, 59, 0, 24, 0, 24, 9, 18, 16, 19, 0, 36, 7, 12, 15, 8, 0, 55, 33, 44, 7, 11, 27, 40, 27, 0, 5, 43, 29, 0, 6, 23, 3, 49, 0, 35, 19, 35, 15, 21, 26, 50, 0, 38, 19, 27, 4, 28, 13, 58, 19, 52, 30, 24, 20, 54, 28, 11, 0, 45, 0, 0, 17, 3, 29, 36, 0, 17, 0, 24, 0, 29, 19, 41, 0, 47, 18, 0, 29, 0, 0, 42, 46, 15, 0, 0, 44, 53, 12, 0, 8, 47, 20, 0, 0, 0, 0, 30, 27, 15, 0, 0, 14, 31, 28, 0, 38, 49, 27, 7, 0, 14, 0, 55, 46, 35, 0, 0, 31, 44, 21, 0, 48, 44, 26, 43, 18, 36, 0, 57, 57, 42, 33, 16, 34, 49, 49, 0, 48, 36, 4, 0, 0, 21, 0, 51, 17, 46, 25, 19, 0, 24, 33, 39, 14, 27, 25, 0, 39, 1, 0, 29, 35, 26, 19, 15, 32, 60, 26, 21, 0, 59, 39, 0, 0, 23, 46, 64, 0, 34, 0, 28, 24, 16, 10, 44, 0, 28, 4, 17, 0, 24, 13, 41, 0, 51, 35, 28, 0, 56, 32, 18, 0, 61, 0, 0, 14, 14, 51, 57, 0, 15, 0, 32, 0, 15, 0, 51, 0, 31, 12, 0, 24, 0, 19, 44, 9, 33, 0, 4, 41, 40, 2, 0, 0, 57, 46, 0, 0, 0, 62, 47, 0, 19, 0, 0, 34, 4, 0, 10, 0, 56, 42, 66, 0, 22, 7, 66, 23, 51, 0, 0, 39, 62, 19, 0, 0, 67, 9, 23, 0, 0, 50, 73, 0, 34, 0, 0, 21, 13, 1, 0, 0, 28, 42, 0, 13, 0, 0, 32, 29, 36, 13, 13, 34, 50, 27, 14, 0, 48, 47, 0, 0, 0, 22, 38, 13, 24, 0, 0, 37, 40, 10, 0, 0, 61, 25, 29, 0, 5, 6, 54, 20, 29, 0, 0, 22, 45, 14, 0, 8, 51, 24, 0, 0, 0, 6, 59, 1, 33, 0, 0, 32, 24, 4, 0, 0, 40, 30, 29, 8, 18, 0, 45, 39, 39, 18, 11, 31, 54, 39, 0, 18, 56, 11, 1, 0, 0, 25, 61, 11, 33, 0, 0, 20, 22, 8, 0, 0, 29, 41, 0, 14, 0, 0, 29, 22, 29, 3, 6, 35, 47, 21, 10, 2, 64, 51, 14, 0, 0, 50, 68, 24, 32, 0, 0, 64, 39, 0, 0, 0, 49, 71, 9, 0, 0, 35, 50, 39, 41, 0, 0, 52, 45, 22, 0, 0, 54, 56, 3, 0, 0, 67, 48, 62, 32, 0, 0, 51, 60, 4, 0, 0, 59, 71, 0, 0, 5, 46, 38, 13, 18, 0, 0, 37, 45, 15, 9, 4, 52, 32, 22, 0, 0, 36, 53, 56, 37, 0, 0, 34, 64, 10, 0, 0, 59, 44, 4, 2, 31, 9, 46, 0, 19, 17, 21, 21, 33, 29, 37, 17, 36, 0, 34, 0, 42, 0, 55, 26, 55, 46, 18, 0, 53, 39, 16, 3, 42, 9, 0, 50, 2, 20, 45, 0, 18, 4, 29, 30, 37, 14, 47, 0, 33, 21, 0, 16, 0, 19, 24, 33, 18, 0, 0, 32, 42, 12, 0, 0, 65, 77, 0, 0, 0, 39, 54, 8, 19, 0, 0, 63, 31, 0, 0, 5, 48, 63, 24, 0, 0, 41, 62, 44, 56, 0, 0, 56, 53, 8, 0, 0, 62, 28, 31, 0, 0, 24, 36, 39, 13, 0, 0, 23, 57, 23, 0, 4, 65, 39, 15, 0, 37, 0, 63, 24, 30, 13, 0, 22, 31, 25, 11, 48, 20, 15, 0, 14, 9, 0, 41, 45, 51, 29, 12, 19, 55, 31, 13, 2, 54, 22, 12, 19, 42, 6, 48, 0, 22, 31, 45, 8, 34, 33, 55, 0, 38, 0, 22, 4, 33, 17, 65, 8, 55, 33, 30, 4, 40, 27, 33, 0, 33, 24, 0, 48, 8, 29, 35, 0, 24, 22, 46, 27, 42, 24, 64, 0, 44, 20, 0, 0, 0, 65, 36, 6, 22, 0, 0, 25, 39, 0, 0, 0, 62, 85, 0, 0, 0, 55, 51, 19, 18, 0, 0, 78, 45, 0, 0, 0, 53, 43, 33, 0, 1, 43, 50, 29, 45, 0, 0, 19, 49, 20, 0, 0, 63, 33, 9, 0, 9, 31, 52, 19, 20, 0, 0, 30, 49, 9, 0, 0, 45, 26, 0, 2, 5, 3, 46, 24, 32, 1, 2, 26, 41, 22, 3, 5, 44, 29, 0, 0, 16, 42, 63, 0, 46, 3, 32, 26, 32, 2, 26, 0, 50, 19, 13, 4, 0, 69, 59, 0, 35, 0, 10, 40, 37, 0, 0, 0, 60, 48, 0, 0, 0, 58, 44, 16, 16, 0, 0, 49, 37, 4, 0, 0, 56, 50, 14, 0, 0, 13, 40, 64, 25, 0, 0, 36, 58, 21, 0, 27, 59, 48, 0, 0, 2, 10, 53, 46, 26, 0, 0, 42, 44, 13, 0, 30, 40, 47, 0, 0, 19, 13, 52, 0, 47, 16, 16, 24, 32, 18, 27, 0, 44, 7, 18, 2, 0, 29, 46, 20, 33, 0, 0, 25, 59, 10, 0, 0, 67, 11, 17, 18, 0, 0, 37, 35, 0, 0, 0, 27, 39, 23, 0, 31, 44, 23, 0, 0, 0, 0, 41, 87, 27, 0, 0, 29, 40, 31, 0, 83, 39, 53, 0, 11, 25, 0, 45, 75, 35, 28, 0, 47, 52, 43, 0, 79, 38, 12, 4, 0, 0, 0, 49, 63, 41, 0, 0, 27, 43, 22, 0, 20, 48, 47, 0, 13, 38, 0, 39, 18, 23, 36, 24, 24, 42, 43, 34, 47, 43, 6, 49, 0, 27, 0, 67, 71, 56, 24, 0, 22, 57, 35, 0, 17, 31, 7, 0, 36, 8, 0, 17, 0, 12, 25, 33, 0, 30, 37, 71, 12, 41, 23, 0, 0, 5, 28, 56, 0, 39, 0, 10, 23, 29, 0, 27, 0, 42, 28, 4, 9, 11, 30, 54, 0, 40, 2, 42, 30, 35, 11, 22, 0, 59, 0, 37, 0, 0, 47, 60, 0, 32, 0, 0, 8, 32, 7, 0, 0, 50, 24, 0, 18, 0, 5, 38, 29, 11, 0, 0, 41, 38, 8, 0, 6, 53, 55, 25, 0, 27, 0, 52, 47, 35, 16, 0, 39, 44, 38, 0, 51, 41, 17, 16, 0, 15, 2, 58, 43, 51, 12, 0, 16, 43, 25, 0, 11, 41, 32, 10, 44, 12, 0, 31, 69, 17, 23, 8, 41, 68, 44, 2, 33, 52, 9, 13, 0, 38, 0, 46, 23, 30, 32, 8, 0, 29, 44, 37, 46, 24, 6, 0, 47, 23, 0, 32, 83, 32, 47, 7, 21, 75, 46, 11, 45, 44, 10, 0, 20, 0, 11, 40, 0, 16, 0, 22, 9, 7, 11, 62, 0, 28, 26, 3, 15, 6, 0, 37, 44, 40, 18, 5, 33, 66, 31, 0, 6, 68, 14, 23, 4, 0, 16, 59, 28, 15, 0, 0, 38, 30, 10, 0, 7, 36, 49, 0, 9, 0, 0, 31, 103, 27, 0, 0, 52, 55, 36, 0, 63, 52, 57, 0, 0, 3, 0, 42, 67, 26, 0, 0, 39, 47, 27, 0, 59, 37, 22, 18, 0, 37, 36, 55, 0, 54, 40, 43, 4, 44, 29, 44, 0, 44, 0, 0, 18, 0, 56, 38, 0, 20, 0, 19, 0, 41, 7, 30, 0, 52, 9, 0, 19, 0, 18, 38, 0, 9, 0, 19, 10, 25, 7, 35, 0, 46, 16, 7, 0, 4, 4, 61, 0, 41, 0, 4, 30, 33, 9, 0, 0, 45, 20, 0, 0, 15, 13, 54, 0, 39, 2, 30, 14, 21, 16, 27, 0, 50, 27, 45, 8, 20, 15, 67, 33, 47, 9, 13, 43, 54, 24, 0, 0, 40, 7, 0, 0, 0, 48, 36, 0, 28, 0, 20, 0, 11, 12, 53, 0, 42, 37, 0, 12, 0, 28, 42, 18, 24, 0, 0, 53, 61, 0, 0, 0, 68, 45, 0, 0, 0, 61, 58, 0, 17, 0, 0, 57, 19, 0, 0, 0, 52, 60, 21, 0, 23, 6, 52, 8, 46, 7, 0, 24, 40, 25, 0, 13, 55, 14, 18, 0, 0, 49, 70, 5, 45, 0, 0, 29, 41, 0, 0, 0, 50, 39, 15, 31, 0, 0, 37, 35, 15, 0, 10, 47, 52, 32, 0, 7, 55, 3, 25, 0, 10, 0, 45, 47, 29, 0, 0, 0, 37, 33, 0, 42, 39, 48, 0, 19, 41, 0, 52, 18, 37, 43, 28, 36, 39, 28, 52, 38, 30, 27, 10, 8, 13, 34, 61, 11, 61, 21, 34, 33, 44, 21, 22, 0, 48, 21, 0, 10, 5, 59, 35, 0, 20, 4, 29, 7, 40, 21, 39, 0, 56, 23, 0, 4, 0, 48, 47, 39, 18, 0, 0, 38, 53, 2, 0, 0, 47, 49, 0, 0, 0, 31, 33, 0, 17, 0, 1, 24, 19, 4, 30, 0, 52, 37, 48, 0, 43, 7, 67, 0, 54, 24, 13, 23, 55, 21, 0, 0, 56, 0, 36, 12, 0, 33, 64, 6, 32, 0, 6, 19, 35, 12, 0, 0, 37, 7, 0, 35, 0, 0, 12, 54, 2, 0, 0, 16, 47, 36, 0, 39, 62, 57, 22, 3, 24, 0, 58, 96, 24, 6, 0, 59, 52, 35, 0, 94, 24, 25, 0, 2, 0, 0, 30, 112, 41, 8, 0, 29, 56, 43, 0, 62, 53, 77, 0, 11, 1, 0, 42, 65, 16, 0, 0, 68, 48, 20, 0, 54, 37, 44, 0, 0, 7, 18, 47, 30, 53, 9, 0, 17, 43, 23, 0, 3, 51, 23, 0, 0, 0, 38, 43, 0, 21, 0, 0, 25, 47, 0, 0, 0, 57, 12, 0, 0, 0, 28, 53, 0, 24, 0, 4, 14, 18, 0, 2, 0, 50, 28, 21, 0, 9, 3, 66, 3, 43, 0, 7, 41, 37, 11, 0, 0, 49, 13, 31, 20, 25, 0, 44, 84, 26, 27, 0, 18, 55, 50, 0, 62, 30, 31, 0, 24, 3, 0, 32, 58, 27, 21, 0, 25, 39, 32, 39, 48, 34, 68, 0, 21, 0, 8, 38, 70, 32, 2, 0, 69, 64, 19, 0, 12, 57, 53, 1, 0, 0, 51, 40, 51, 21, 0, 0, 46, 49, 11, 0, 0, 62, 79, 12, 0, 0, 16, 44, 97, 20, 0, 0, 68, 68, 22, 0, 47, 48, 26, 9, 0, 20, 0, 30, 48, 27, 18, 0, 0, 49, 42, 0, 55, 53, 45, 0, 6, 46, 0, 74, 14, 43, 29, 18, 45, 41, 13, 33, 18, 17, 9, 0, 23, 0, 35, 33, 19, 43, 0, 14, 28, 45, 15, 5, 0, 67, 49, 0, 0, 0, 61, 32, 0, 0, 0, 0, 49, 35, 0, 0, 0, 66, 74, 44, 0, 29, 6, 69, 44, 45, 1, 0, 51, 47, 23, 0, 34, 40, 21, 5, 0, 18, 38, 59, 0, 57, 12, 10, 4, 32, 15, 16, 0, 40, 1, 0, 17, 10, 21, 33, 0, 21, 8, 29, 0, 46, 15, 32, 0, 63, 8, 9, 0, 17, 43, 76, 0, 33, 0, 27, 22, 12, 0, 27, 0, 31, 19, 33, 21, 30, 0, 50, 15, 51, 41, 46, 20, 50, 44, 24, 0, 52, 0, 26, 13, 12, 10, 41, 21, 19, 6, 4, 0, 43, 33, 11, 0, 41, 20, 0, 49, 8, 0, 31, 71, 10, 17, 0, 33, 53, 38, 16, 60, 31, 20, 0, 2, 0, 0, 32, 27, 30, 5, 0, 10, 29, 26, 23, 35, 45, 67, 0, 8, 17, 5, 64, 26, 44, 9, 12, 69, 48, 12, 3, 0, 44, 22, 40, 0, 25, 37, 54, 0, 52, 21, 28, 4, 38, 33, 16, 0, 48, 0, 0, 5, 0, 42, 40, 0, 19, 0, 0, 1, 45, 9, 13, 0, 48, 34, 0, 25, 0, 29, 39, 8, 9, 0, 0, 54, 34, 0, 0, 0, 50, 40, 16, 0, 4, 0, 40, 15, 29, 0, 0, 22, 40, 27, 0, 18, 61, 19, 17, 0, 0, 13, 68, 46, 34, 0, 0, 44, 40, 0, 0, 7, 42, 43, 0, 21, 0, 0, 25, 76, 13, 0, 0, 51, 52, 35, 0, 53, 64, 56, 27, 0, 19, 0, 67, 68, 39, 0, 0, 45, 35, 25, 0, 68, 25, 53, 0, 0, 3, 0, 45, 53, 54, 18, 0, 41, 50, 25, 4, 13, 52, 60, 0, 0, 0, 72, 55, 0, 34, 0, 11, 48, 39, 0, 13, 0, 52, 32, 18, 0, 0, 67, 48, 0, 35, 0, 0, 19, 48, 10, 0, 0, 47, 0, 7, 4, 27, 0, 49, 0, 30, 16, 16, 0, 34, 27, 21, 11, 39, 4, 1, 26, 23, 0, 59, 0, 40, 18, 32, 21, 31, 23, 30, 0, 31, 0, 0, 26, 0, 0, 33, 18, 28, 0, 12, 7, 40, 30, 4, 0, 55, 15, 0, 11, 0, 0, 42, 18, 10, 0, 0, 29, 29, 10, 0, 23, 39, 17, 0, 0, 16, 0, 43, 15, 36, 10, 0, 8, 33, 31, 0, 39, 52, 31, 22, 0, 36, 0, 84, 0, 56, 12, 25, 43, 25, 12, 12, 0, 26, 3, 40, 25, 19, 0, 41, 35, 49, 37, 37, 10, 55, 52, 11, 0, 55, 0, 0, 9, 0, 38, 38, 4, 6, 0, 0, 8, 25, 3, 15, 0, 34, 59, 0, 7, 0, 0, 35, 0, 27, 0, 7, 46, 36, 7, 24, 4, 56, 23, 62, 0, 38, 12, 67, 0, 51, 14, 17, 13, 45, 25, 0, 0, 56, 0, 36, 2, 36, 30, 75, 0, 45, 15, 35, 8, 25, 19, 33, 0, 25, 21, 0, 32, 8, 30, 42, 0, 42, 27, 48, 21, 36, 23, 64, 0, 47, 29, 6, 16, 2, 49, 40, 12, 23, 2, 14, 28, 56, 20, 12, 0, 56, 6, 18, 0, 25, 13, 38, 5, 17, 15, 6, 0, 42, 33, 21, 10, 42, 9, 0, 24, 9, 0, 48, 41, 27, 7, 0, 26, 48, 19, 9, 19, 33, 18, 0, 27, 0, 0, 28, 20, 20, 0, 9, 22, 37, 26, 18, 12, 55, 55, 0, 0, 8, 22, 67, 15, 40, 0, 0, 55, 31, 5, 0, 0, 34, 20, 24, 0, 25, 1, 40, 4, 47, 32, 24, 0, 49, 40, 12, 0, 64, 0, 42, 11, 37, 10, 62, 5, 24, 13, 13, 6, 45, 24, 11, 0, 30, 9, 0, 38, 0, 0, 41, 22, 31, 7, 21, 25, 30, 23, 41, 0, 33, 45, 0, 17, 0, 22, 27, 35, 20, 0, 0, 50, 47, 7, 0, 0, 67, 73, 7, 0, 0, 41, 54, 36, 23, 0, 0, 66, 45, 3, 0, 3, 51, 49, 0, 0, 0, 51, 54, 0, 44, 0, 0, 23, 28, 0, 0, 0, 52, 41, 0, 0, 0, 45, 48, 15, 24, 0, 0, 55, 56, 0, 0, 0, 74, 41, 40, 0, 0, 15, 54, 38, 13, 0, 0, 51, 40, 12, 0, 14, 55, 46, 25, 0, 1, 0, 52, 97, 35, 0, 0, 42, 48, 32, 0, 71, 39, 44, 0, 0, 25, 0, 37, 70, 34, 32, 0, 24, 54, 45, 0, 70, 43, 20, 4, 3, 18, 0, 52, 44, 38, 15, 0, 23, 47, 26, 1, 25, 50, 61, 0, 0, 0, 24, 46, 46, 31, 0, 0, 49, 43, 21, 0, 14, 45, 46, 4, 0, 28, 26, 45, 11, 42, 26, 7, 10, 47, 27, 19, 1, 47, 16, 0, 0, 14, 67, 59, 0, 40, 0, 23, 10, 31, 0, 39, 0, 43, 24, 0, 8, 0, 80, 54, 0, 35, 0, 44, 28, 24, 0, 42, 0, 61, 44, 22, 0, 0, 103, 72, 0, 40, 0, 19, 50, 31, 0, 0, 0, 57, 26, 43, 0, 8, 53, 44, 8, 29, 0, 4, 12, 52, 27, 0, 0, 61, 2, 15, 0, 4, 7, 40, 47, 9, 0, 0, 6, 53, 23, 0, 26, 41, 41, 0, 14, 5, 0, 45, 10, 25, 3, 7, 35, 23, 13, 41, 28, 33, 58, 5, 3, 0, 13, 54, 46, 54, 6, 6, 63, 59, 19, 0, 0, 61, 29, 27, 0, 22, 53, 51, 0, 30, 3, 17, 2, 30, 23, 21, 0, 47, 7, 33, 0, 47, 4, 53, 18, 40, 43, 20, 0, 58, 35, 22, 2, 41, 0, 3, 51, 12, 0, 38, 31, 17, 18, 21, 9, 49, 35, 30, 5, 36, 4, 0, 42, 0, 0, 20, 60, 7, 7, 0, 14, 46, 40, 15, 51, 45, 45, 0, 9, 31, 0, 49, 47, 32, 29, 5, 34, 39, 35, 24, 64, 26, 19, 0, 1, 10, 0, 47, 33, 52, 21, 11, 19, 47, 28, 11, 0, 49, 34, 0, 20, 0, 38, 43, 10, 17, 0, 0, 47, 40, 2, 0, 0, 54, 43, 21, 0, 0, 2, 39, 65, 21, 0, 0, 43, 56, 27, 0, 24, 62, 52, 21, 0, 7, 0, 52, 89, 25, 0, 0, 47, 52, 29, 0, 64, 34, 38, 0, 0, 32, 0, 34, 50, 40, 44, 4, 6, 48, 46, 25, 56, 41, 12, 0, 11, 19, 2, 51, 18, 36, 17, 9, 18, 50, 18, 15, 0, 40, 13, 0, 21, 0, 27, 40, 0, 22, 0, 13, 26, 25, 3, 23, 0, 50, 33, 3, 0, 0, 5, 41, 27, 22, 0, 0, 41, 48, 13, 0, 0, 65, 54, 15, 0, 14, 26, 72, 8, 39, 0, 0, 46, 23, 9, 0, 8, 26, 34, 0, 0, 0, 44, 48, 0, 58, 4, 16, 13, 32, 7, 23, 0, 57, 18, 0, 0, 0, 71, 48, 0, 17, 0, 1, 29, 43, 0, 0, 0, 71, 17, 26, 0, 0, 35, 58, 6, 14, 0, 0, 42, 30, 0, 0, 0, 54, 51, 14, 0, 11, 0, 53, 44, 34, 0, 0, 40, 34, 29, 0, 47, 41, 32, 17, 0, 16, 0, 55, 53, 50, 14, 0, 27, 50, 29, 0, 23, 51, 11, 20, 20, 1, 0, 48, 49, 21, 0, 0, 35, 55, 23, 0, 22, 53, 34, 23, 18, 10, 0, 51, 80, 25, 3, 0, 45, 41, 42, 0, 66, 33, 56, 0, 8, 6, 0, 40, 125, 42, 25, 0, 51, 61, 47, 0, 75, 38, 58, 0, 0, 19, 2, 31, 48, 30, 32, 1, 23, 51, 36, 33, 38, 43, 15, 13, 2, 17, 1, 37, 48, 29, 20, 0, 11, 70, 30, 0, 11, 54, 0, 0, 17, 23, 0, 43, 0, 13, 9, 17, 0, 27, 25, 40, 16, 30, 11, 0, 13, 15, 0, 53, 0, 46, 18, 25, 18, 31, 18, 35, 0, 40, 28, 0, 14, 25, 23, 57, 0, 44, 20, 53, 25, 29, 18, 49, 0, 44, 0, 30, 1, 2, 31, 46, 0, 35, 0, 12, 0, 45, 20, 0, 0, 61, 18, 12, 35, 17, 0, 47, 25, 7, 1, 9, 31, 41, 28, 14, 27, 34, 18, 19, 17, 15, 0, 41, 99, 39, 29, 0, 24, 56, 53, 0, 63, 35, 20, 0, 20, 1, 0, 22, 56, 15, 14, 0, 12, 45, 35, 22, 55, 41, 41, 0, 10, 0, 0, 43, 46, 26, 0, 0, 48, 46, 12, 0, 26, 47, 41, 18, 0, 33, 0, 53, 0, 42, 24, 24, 20, 35, 33, 18, 11, 45, 4, 27, 0, 28, 26, 68, 0, 54, 16, 19, 7, 38, 17, 13, 0, 34, 1, 0, 30, 0, 26, 33, 0, 21, 0, 32, 7, 31, 13, 46, 0, 53, 11, 0, 8, 0, 19, 44, 6, 16, 0, 0, 30, 40, 3, 0, 0, 55, 20, 5, 0, 10, 0, 47, 0, 23, 0, 0, 14, 25, 21, 0, 22, 48, 32, 42, 0, 35, 0, 74, 49, 54, 22, 7, 44, 47, 30, 0, 27, 33, 13, 0, 17, 0, 2, 40, 34, 37, 8, 14, 18, 38, 33, 17, 0, 45, 46, 0, 4, 0, 30, 39, 16, 22, 0, 0, 37, 40, 10, 17, 0, 46, 44, 0, 0, 0, 39, 44, 19, 31, 0, 0, 41, 47, 2, 0, 0, 63, 39, 29, 0, 27, 20, 55, 0, 28, 2, 8, 19, 38, 21, 0, 0, 52, 29, 38, 0, 9, 25, 74, 60, 49, 0, 0, 52, 53, 16, 0, 0, 34, 47, 0, 18, 0, 37, 18, 41, 14, 0, 0, 42, 40, 16, 10, 0, 58, 72, 0, 0, 0, 46, 34, 50, 13, 0, 0, 56, 55, 0, 0, 9, 68, 67, 26, 0, 11, 15, 58, 29, 30, 0, 0, 47, 47, 12, 0, 23, 53, 25, 36, 0, 16, 8, 65, 34, 48, 0, 0, 24, 44, 20, 0, 6, 45, 47, 0, 0, 0, 22, 23, 66, 14, 0, 0, 44, 50, 13, 0, 14, 65, 52, 3, 0, 7, 0, 38, 40, 12, 0, 0, 26, 48, 20, 0, 59, 52, 8, 41, 0, 7, 0, 57, 85, 35, 0, 0, 30, 59, 28, 0, 59, 48, 30, 8, 33, 8, 0, 44, 84, 20, 5, 0, 47, 39, 45, 0, 85, 30, 37, 0, 0, 0, 0, 39, 105, 43, 12, 0, 35, 48, 42, 0, 75, 42, 66, 0, 6, 12, 0, 41, 61, 31, 17, 0, 52, 52, 30, 6, 45, 45, 45, 14, 0, 11, 26, 55, 44, 46, 8, 0, 35, 51, 23, 0, 0, 47, 30, 13, 1, 33, 15, 40, 11, 29, 33, 22, 7, 51, 37, 32, 2, 45, 4, 0, 0, 21, 42, 55, 0, 37, 13, 22, 0, 30, 9, 47, 0, 33, 7, 0, 49, 0, 0, 27, 27, 16, 0, 13, 30, 63, 23, 1, 0, 64, 2, 0, 0, 9, 3, 46, 0, 11, 0, 0, 0, 12, 18, 18, 16, 34, 38, 0, 10, 13, 0, 62, 49, 51, 12, 1, 54, 49, 18, 0, 18, 38, 25, 0, 0, 1, 35, 46, 0, 40, 0, 34, 11, 14, 13, 42, 0, 54, 30, 49, 0, 35, 27, 65, 3, 45, 19, 21, 30, 57, 22, 0, 0, 52, 0, 24, 10, 13, 32, 52, 2, 29, 3, 19, 2, 33, 26, 22, 0, 37, 36, 0, 28, 0, 13, 36, 33, 23, 6, 8, 37, 45, 20, 30, 4, 41, 44, 0, 0, 0, 48, 39, 0, 30, 0, 0, 29, 33, 2, 17, 0, 58, 62, 8, 0, 11, 65, 65, 0, 40, 0, 10, 50, 42, 0, 0, 0, 52, 1, 44, 0, 30, 48, 54, 0, 43, 14, 28, 0, 35, 23, 17, 0, 55, 0, 10, 24, 6, 24, 55, 0, 20, 0, 8, 15, 43, 7, 5, 0, 40, 20, 0, 31, 0, 0, 38, 5, 21, 0, 22, 26, 27, 25, 32, 5, 41, 35, 0, 0, 0, 9, 49, 44, 38, 0, 0, 41, 45, 14, 0, 2, 52, 48, 9, 0, 30, 7, 49, 0, 33, 20, 20, 23, 37, 29, 24, 8, 41, 0, 32, 0, 18, 0, 47, 18, 40, 13, 0, 0, 52, 28, 0, 0, 50, 2, 0, 44, 0, 0, 40, 0, 2, 0, 6, 29, 24, 5, 24, 3, 35, 42, 0, 0, 0, 0, 53, 0, 45, 0, 11, 39, 18, 9, 11, 0, 49, 42, 36, 0, 0, 18, 58, 36, 44, 0, 0, 54, 57, 14, 0, 0, 64, 24, 19, 0, 0, 32, 52, 10, 22, 0, 0, 19, 27, 16, 0, 0, 51, 25, 0, 0, 0, 17, 56, 0, 27, 0, 0, 30, 16, 0, 11, 0, 44, 57, 12, 0, 0, 22, 66, 23, 50, 0, 3, 75, 45, 4, 0, 0, 57, 35, 0, 0, 0, 77, 49, 0, 31, 0, 0, 25, 19, 0, 0, 0, 59, 49, 8, 0, 0, 19, 43, 33, 19, 0, 0, 44, 59, 6, 0, 3, 68, 31, 22, 0, 0, 11, 61, 47, 22, 0, 0, 49, 37, 4, 0, 19, 44, 48, 8, 0, 8, 0, 39, 69, 31, 6, 0, 36, 48, 40, 0, 65, 50, 20, 29, 0, 22, 0, 53, 69, 38, 12, 0, 20, 51, 35, 0, 55, 39, 18, 6, 20, 43, 0, 45, 32, 35, 47, 27, 10, 41, 48, 39, 50, 33, 13, 9, 24, 33, 0, 57, 38, 50, 42, 30, 22, 45, 37, 35, 10, 27, 5, 0, 39, 0, 16, 17, 38, 14, 0, 0, 23, 48, 17, 8, 0, 64, 80, 0, 0, 15, 31, 53, 0, 18, 0, 6, 52, 24, 6, 29, 12, 36, 26, 43, 0, 0, 22, 55, 64, 61, 2, 0, 30, 70, 23, 0, 0, 70, 45, 0, 14, 0, 44, 47, 12, 3, 0, 0, 45, 35, 11, 9, 0, 38, 31, 28, 0, 4, 0, 31, 114, 34, 16, 0, 25, 75, 49, 0, 58, 57, 30, 0, 14, 0, 0, 32, 82, 2, 0, 0, 30, 50, 31, 0, 78, 33, 39, 0, 1, 23, 0, 36, 62, 36, 33, 0, 21, 46, 41, 14, 74, 39, 34, 0, 0, 33, 0, 63, 0, 52, 28, 28, 28, 35, 19, 33, 0, 35, 3, 24, 16, 23, 14, 47, 0, 43, 28, 41, 5, 47, 32, 28, 0, 57, 8, 13, 10, 42, 39, 63, 0, 32, 25, 48, 1, 22, 21, 66, 0, 22, 0, 0, 27, 0, 26, 38, 1, 42, 19, 31, 7, 47, 20, 39, 0, 53, 20, 0, 17, 10, 44, 40, 0, 12, 0, 36, 11, 25, 9, 55, 0, 46, 9, 11, 0, 2, 33, 60, 0, 43, 0, 5, 20, 39, 6, 0, 0, 54, 29, 27, 31, 2, 0, 44, 44, 17, 0, 4, 45, 55, 31, 0, 7, 50, 33, 0, 0, 14, 26, 55, 1, 40, 2, 2, 9, 13, 18, 34, 13, 24, 51, 0, 3, 6, 33, 43, 0, 50, 20, 23, 39, 55, 7, 28, 0, 64, 31, 18, 0, 7, 101, 66, 0, 33, 0, 32, 26, 25, 0, 21, 0, 49, 0, 17, 0, 0, 58, 43, 0, 28, 0, 0, 8, 46, 7, 0, 0, 66, 33, 0, 6, 0, 22, 48, 0, 6, 0, 0, 37, 31, 6, 0, 3, 48, 75, 0, 0, 0, 55, 50, 8, 38, 0, 1, 54, 37, 8, 3, 0, 51, 26, 22, 0, 0, 51, 40, 20, 31, 0, 0, 12, 60, 12, 0, 0, 71, 38, 0, 0, 8, 54, 64, 0, 17, 0, 0, 31, 17, 0, 18, 0, 31, 34, 0, 0, 0, 31, 55, 0, 56, 0, 13, 36, 42, 7, 0, 0, 59, 22, 0, 0, 0, 80, 51, 0, 22, 0, 8, 17, 18, 0, 6, 0, 60, 33, 25, 0, 7, 41, 66, 0, 38, 0, 4, 35, 37, 1, 0, 0, 55, 17, 26, 0, 0, 33, 59, 12, 34, 0, 0, 32, 36, 9, 0, 0, 53, 41, 20, 7, 6, 0, 43, 58, 22, 0, 0, 39, 51, 34, 0, 34, 47, 19, 24, 0, 13, 0, 38, 92, 27, 14, 0, 16, 59, 46, 0, 67, 37, 29, 0, 1, 0, 0, 37, 7, 24, 0, 0, 16, 21, 10, 41, 29, 37, 51, 29, 23, 24, 0, 50, 59, 43, 29, 14, 60, 74, 36, 0, 14, 58, 0, 15, 0, 0, 35, 51, 1, 25, 0, 0, 0, 17, 14, 0, 0, 37, 36, 0, 29, 7, 0, 25, 60, 15, 16, 0, 36, 62, 35, 0, 56, 51, 2, 1, 0, 15, 0, 57, 0, 35, 0, 0, 0, 16, 17, 7, 17, 32, 21, 0, 22, 0, 0, 52, 18, 41, 0, 12, 42, 43, 14, 0, 0, 53, 33, 16, 0, 10, 20, 62, 0, 39, 0, 23, 33, 21, 17, 11, 0, 43, 26, 39, 0, 36, 9, 57, 19, 52, 34, 23, 13, 49, 37, 12, 0, 44, 0, 6, 14, 27, 20, 45, 0, 29, 26, 28, 0, 41, 28, 43, 0, 42, 6, 16, 42, 44, 0, 43, 20, 26, 51, 41, 8, 53, 46, 50, 16, 34, 0, 0, 21, 23, 7, 46, 0, 36, 29, 36, 0, 25, 28, 67, 0, 30, 29, 0, 45, 25, 0, 40, 3, 33, 41, 49, 30, 51, 30, 59, 0, 47, 7, 3, 9, 0, 51, 48, 8, 29, 0, 5, 19, 39, 12, 4, 0, 46, 17, 0, 18, 0, 0, 15, 41, 0, 0, 0, 18, 54, 23, 0, 20, 68, 56, 25, 0, 34, 0, 68, 51, 28, 6, 0, 50, 39, 26, 0, 66, 23, 27, 13, 0, 22, 0, 49, 67, 62, 41, 11, 21, 52, 46, 7, 25, 45, 23, 1, 33, 5, 0, 30, 55, 14, 12, 0, 27, 63, 35, 6, 18, 49, 30, 0, 1, 0, 17, 39, 37, 17, 0, 0, 28, 33, 12, 5, 21, 38, 61, 0, 1, 0, 0, 36, 44, 30, 5, 0, 49, 55, 21, 0, 21, 54, 32, 17, 11, 2, 0, 43, 65, 18, 0, 0, 41, 53, 30, 0, 44, 47, 37, 0, 0, 0, 0, 51, 48, 34, 0, 0, 34, 30, 19, 0, 38, 37, 67, 0, 3, 16, 0, 46, 55, 43, 24, 5, 54, 59, 31, 0, 23, 52, 23, 14, 0, 8, 49, 55, 2, 39, 0, 0, 12, 36, 14, 3, 0, 47, 50, 0, 11, 0, 47, 49, 15, 29, 0, 6, 51, 48, 6, 12, 0, 49, 39, 7, 0, 0, 46, 41, 25, 28, 0, 0, 32, 50, 17, 0, 0, 59, 32, 0, 0, 0, 36, 42, 17, 17, 0, 0, 22, 42, 7, 0, 0, 49, 25, 0, 0, 17, 0, 44, 7, 27, 6, 0, 14, 43, 22, 0, 16, 49, 4, 12, 0, 12, 0, 62, 0, 38, 0, 6, 18, 31, 12, 0, 0, 44, 49, 0, 27, 0, 13, 60, 41, 37, 0, 6, 74, 39, 12, 0, 0, 40, 25, 9, 0, 0, 1, 20, 50, 24, 0, 0, 5, 51, 39, 0, 18, 69, 49, 0, 0, 21, 25, 66, 0, 24, 0, 0, 34, 28, 0, 9, 21, 24, 33, 0, 1, 4, 10, 50, 5, 58, 14, 25, 34, 47, 18, 8, 0, 66, 46, 16, 0, 0, 83, 66, 0, 27, 0, 0, 62, 36, 0, 0, 0, 47, 49, 0, 0, 0, 58, 36, 16, 30, 0, 0, 26, 39, 13, 0, 0, 64, 50, 25, 0, 11, 23, 47, 46, 21, 0, 0, 39, 66, 17, 0, 15, 57, 15, 38, 0, 20, 0, 48, 61, 25, 9, 0, 16, 51, 39, 0, 43, 39, 27, 0, 28, 12, 0, 38, 90, 25, 22, 0, 33, 54, 43, 4, 67, 33, 25, 0, 23, 21, 0, 24, 81, 24, 40, 0, 14, 59, 56, 14, 73, 46, 30, 0, 14, 33, 0, 53, 55, 34, 33, 5, 27, 46, 36, 24, 51, 23, 9, 0, 23, 38, 0, 31, 23, 41, 61, 40, 0, 50, 53, 55, 21, 42, 0, 0, 19, 22, 14, 48, 0, 28, 16, 30, 0, 30, 14, 53, 0, 33, 14, 0, 31, 0, 31, 47, 0, 31, 0, 35, 32, 22, 0, 44, 0, 49, 20, 33, 0, 19, 15, 49, 0, 35, 7, 32, 13, 39, 26, 8, 0, 62, 0, 53, 3, 4, 7, 62, 47, 28, 0, 0, 20, 48, 21, 0, 3, 35, 13, 0, 27, 0, 0, 20, 30, 10, 8, 3, 6, 29, 38, 34, 54, 41, 42, 0, 1, 9, 0, 57, 50, 42, 3, 0, 50, 46, 16, 0, 32, 39, 31, 0, 1, 0, 3, 46, 14, 38, 0, 10, 31, 37, 23, 0, 0, 42, 18, 0, 0, 8, 25, 48, 0, 37, 0, 19, 11, 35, 9, 15, 0, 59, 16, 21, 3, 0, 38, 66, 0, 31, 0, 5, 39, 35, 0, 0, 0, 47, 13, 21, 5, 1, 0, 38, 27, 25, 0, 2, 15, 41, 35, 0, 11, 55, 29, 13, 2, 12, 0, 53, 64, 27, 0, 0, 36, 47, 27, 0, 47, 31, 21, 0, 5, 23, 0, 32, 27, 35, 35, 15, 2, 39, 43, 33, 39, 43, 6, 0, 11, 17, 0, 52, 14, 34, 10, 7, 16, 45, 19, 8, 2, 42, 23, 0, 26, 0, 9, 49, 9, 28, 0, 12, 42, 30, 11, 10, 0, 46, 57, 0, 0, 0, 21, 49, 45, 36, 0, 0, 56, 45, 18, 0, 0, 52, 54, 0, 0, 5, 47, 49, 12, 38, 0, 0, 26, 39, 14, 0, 0, 45, 6, 4, 0, 21, 20, 38, 0, 31, 13, 8, 0, 48, 19, 12, 0, 60, 2, 0, 5, 21, 39, 74, 0, 32, 0, 32, 17, 13, 0, 38, 0, 31, 21, 19, 20, 25, 15, 61, 0, 57, 28, 63, 27, 32, 26, 41, 0, 49, 0, 18, 0, 17, 66, 56, 0, 40, 4, 38, 0, 25, 14, 41, 0, 42, 0, 0, 31, 0, 10, 28, 9, 11, 0, 5, 4, 56, 20, 4, 0, 61, 19, 0, 12, 9, 0, 54, 0, 14, 0, 6, 25, 17, 12, 22, 16, 26, 21, 3, 2, 14, 0, 46, 37, 52, 25, 12, 21, 47, 35, 0, 14, 52, 7, 7, 12, 0, 2, 47, 12, 21, 0, 0, 20, 38, 17, 0, 0, 45, 13, 0, 7, 3, 0, 41, 26, 24, 0, 0, 17, 35, 27, 0, 27, 45, 27, 6, 11, 5, 0, 50, 55, 31, 0, 0, 40, 46, 26, 0, 37, 44, 26, 21, 17, 0, 0, 39, 92, 25, 0, 0, 39, 56, 42, 0, 58, 49, 43, 5, 0, 29, 0, 46, 70, 30, 27, 0, 26, 41, 45, 10, 80, 27, 43, 0, 0, 44, 0, 54, 30, 60, 59, 30, 20, 45, 36, 52, 20, 32, 19, 0, 23, 1, 56, 44, 0, 38, 10, 32, 24, 50, 12, 36, 0, 58, 27, 6, 19, 0, 53, 37, 14, 8, 0, 0, 33, 50, 14, 0, 0, 53, 22, 0, 0, 0, 18, 33, 31, 14, 0, 0, 13, 37, 13, 0, 18, 51, 71, 0, 0, 9, 3, 57, 28, 33, 0, 0, 64, 42, 7, 0, 21, 40, 31, 18, 0, 11, 33, 56, 0, 55, 5, 14, 15, 36, 19, 0, 0, 57, 18, 13, 0, 5, 48, 55, 0, 27, 0, 7, 23, 42, 5, 0, 0, 40, 34, 1, 0, 29, 50, 61, 0, 54, 25, 40, 18, 34, 16, 46, 0, 41, 3, 1, 17, 0, 62, 38, 2, 25, 0, 6, 19, 56, 7, 0, 0, 68, 32, 0, 13, 0, 36, 36, 28, 0, 0, 0, 46, 39, 3, 0, 3, 49, 50, 0, 0, 12, 0, 43, 42, 33, 2, 0, 23, 38, 28, 0, 55, 49, 49, 29, 0, 17, 0, 68, 72, 48, 6, 0, 62, 61, 21, 0, 22, 41, 30, 0, 0, 1, 28, 40, 0, 34, 3, 17, 12, 25, 24, 33, 0, 43, 28, 0, 0, 20, 32, 46, 0, 37, 14, 13, 9, 45, 12, 24, 0, 52, 0, 16, 9, 8, 24, 51, 0, 25, 0, 17, 6, 40, 10, 4, 0, 55, 10, 17, 17, 22, 0, 60, 0, 28, 4, 26, 21, 25, 23, 23, 0, 35, 37, 0, 6, 21, 16, 63, 13, 57, 25, 32, 35, 32, 24, 37, 0, 35, 40, 0, 11, 8, 51, 41, 12, 40, 20, 29, 29, 54, 25, 29, 0, 59, 12, 29, 5, 10, 37, 35, 31, 12, 2, 0, 6, 61, 30, 0, 0, 56, 27, 1, 24, 12, 0, 42, 75, 12, 7, 0, 33, 54, 32, 2, 54, 28, 42, 0, 12, 0, 0, 27, 62, 28, 1, 0, 35, 43, 23, 14, 36, 47, 68, 0, 0, 8, 10, 43, 34, 30, 4, 0, 51, 55, 18, 0, 13, 55, 29, 23, 0, 21, 47, 64, 0, 47, 1, 7, 12, 32, 12, 5, 0, 42, 21, 0, 11, 0, 41, 46, 6, 31, 0, 5, 32, 49, 6, 1, 0, 58, 42, 0, 2, 0, 49, 46, 7, 16, 0, 0, 47, 34, 1, 0, 0, 52, 48, 2, 0, 0, 20, 41, 51, 27, 0, 0, 42, 50, 13, 0, 10, 58, 35, 0, 0, 0, 6, 41, 32, 17, 0, 0, 29, 42, 9, 0, 23, 56, 40, 30, 0, 31, 0, 59, 35, 36, 11, 0, 34, 45, 30, 0, 44, 44, 12, 61, 15, 29, 0, 59, 91, 46, 29, 0, 30, 60, 51, 0, 47, 35, 8, 0, 37, 0, 0, 20, 89, 11, 8, 0, 16, 48, 46, 10, 64, 36, 39, 0, 24, 0, 0, 15, 110, 6, 0, 0, 41, 63, 35, 0, 92, 52, 72, 0, 0, 0, 0, 50, 99, 24, 0, 0, 83, 45, 10, 0, 66, 41, 73, 9, 0, 3, 0, 37, 81, 41, 7, 0, 46, 56, 40, 0, 57, 62, 45, 36, 0, 26, 13, 66, 51, 41, 4, 0, 33, 51, 21, 0, 28, 35, 24, 0, 10, 6, 1, 38, 31, 33, 15, 8, 19, 47, 28, 17, 5, 46, 12, 35, 15, 35, 0, 50, 54, 35, 36, 12, 13, 54, 46, 12, 33, 32, 2, 0, 29, 0, 0, 27, 47, 18, 0, 0, 13, 41, 23, 19, 18, 40, 47, 0, 23, 0, 0, 23, 45, 5, 0, 0, 54, 47, 7, 0, 25, 65, 77, 27, 0, 0, 2, 64, 77, 34, 0, 0, 82, 49, 12, 0, 37, 45, 29, 19, 0, 11, 0, 37, 40, 40, 8, 0, 0, 42, 39, 0, 38, 57, 33, 18, 5, 34, 0, 65, 39, 33, 15, 0, 40, 54, 19, 0, 31, 34, 25, 4, 19, 17, 0, 52, 22, 47, 25, 32, 27, 34, 35, 34, 0, 37, 32, 0, 6, 0, 40, 38, 14, 33, 4, 10, 24, 45, 17, 24, 0, 52, 34, 3, 2, 13, 29, 40, 6, 21, 5, 8, 21, 51, 21, 12, 0, 55, 4, 40, 0, 33, 0, 51, 26, 29, 21, 6, 2, 50, 36, 3, 17, 39, 0, 6, 40, 15, 0, 34, 63, 19, 23, 2, 7, 53, 45, 10, 48, 35, 8, 0, 34, 10, 0, 29, 42, 16, 19, 6, 11, 34, 38, 34, 61, 30, 13, 0, 17, 17, 0, 39, 41, 35, 25, 6, 16, 46, 36, 9, 44, 44, 18, 0, 10, 8, 0, 59, 0, 36, 0, 14, 30, 24, 13, 16, 0, 36, 25, 16, 20, 0, 0, 41, 59, 34, 0, 0, 43, 57, 31, 0, 6, 63, 41, 23, 3, 0, 0, 47, 82, 15, 0, 0, 47, 47, 30, 0, 53, 36, 48, 0, 0, 0, 0, 29, 92, 30, 5, 0, 31, 51, 34, 0, 69, 45, 51, 0, 0, 0, 0, 35, 69, 21, 0, 0, 48, 60, 18, 0, 41, 55, 42, 0, 0, 0, 9, 50, 30, 26, 0, 0, 43, 34, 7, 0, 15, 48, 65, 0, 0, 0, 25, 58, 25, 42, 0, 0, 61, 36, 1, 0, 0, 52, 49, 29, 0, 18, 32, 55, 0, 45, 3, 7, 25, 43, 20, 0, 0, 56, 0, 39, 0, 2, 14, 47, 25, 24, 0, 0, 0, 53, 19, 0, 0, 53, 13, 0, 24, 0, 0, 41, 6, 6, 0, 0, 27, 21, 9, 14, 26, 32, 46, 0, 0, 0, 0, 50, 25, 46, 0, 0, 50, 32, 8, 0, 6, 49, 31, 0, 0, 0, 32, 53, 0, 39, 0, 7, 17, 21, 0, 0, 0, 63, 20, 52, 0, 0, 24, 72, 19, 37, 0, 0, 50, 49, 2, 0, 0, 56, 17, 23, 0, 0, 0, 44, 34, 21, 0, 0, 21, 30, 32, 0, 26, 43, 39, 0, 0, 14, 0, 47, 69, 36, 13, 0, 32, 48, 32, 0, 57, 44, 0, 28, 24, 10, 0, 46, 55, 28, 8, 0, 18, 59, 33, 0, 17, 46, 25, 0, 31, 0, 0, 35, 51, 8, 0, 0, 47, 28, 12, 0, 31, 35, 79, 0, 0, 0, 0, 35, 44, 34, 0, 0, 60, 38, 9, 0, 25, 58, 42, 33, 0, 0, 14, 47, 53, 29, 0, 0, 43, 64, 8, 0, 2, 78, 59, 23, 0, 0, 19, 71, 49, 21, 0, 0, 71, 31, 3, 0, 31, 31, 53, 14, 0, 0, 0, 32, 107, 45, 14, 0, 37, 64, 48, 0, 59, 60, 27, 16, 0, 35, 0, 45, 35, 20, 19, 0, 4, 46, 34, 3, 56, 35, 8, 0, 15, 33, 0, 54, 31, 44, 36, 17, 14, 45, 32, 26, 25, 32, 5, 0, 34, 20, 0, 42, 0, 34, 29, 43, 9, 34, 32, 51, 0, 43, 4, 10, 17, 18, 3, 49, 5, 33, 15, 23, 11, 40, 27, 23, 0, 39, 5, 0, 9, 11, 13, 43, 0, 29, 8, 28, 0, 23, 17, 49, 0, 42, 30, 0, 15, 8, 24, 58, 0, 40, 1, 24, 41, 40, 8, 16, 0, 49, 30, 4, 0, 0, 56, 53, 0, 35, 0, 12, 35, 32, 8, 2, 0, 54, 46, 4, 0, 0, 45, 45, 23, 26, 0, 0, 38, 48, 13, 0, 0, 52, 22, 0, 0, 0, 28, 37, 19, 19, 0, 0, 15, 43, 9, 0, 0, 54, 21, 0, 0, 3, 0, 42, 9, 16, 0, 0, 20, 40, 14, 0, 22, 51, 25, 10, 0, 12, 0, 70, 6, 44, 0, 2, 38, 25, 10, 0, 5, 32, 24, 0, 0, 4, 15, 52, 0, 51, 2, 30, 17, 20, 13, 27, 0, 52, 30, 11, 1, 0, 54, 60, 6, 33, 0, 0, 55, 47, 0, 0, 0, 62, 41, 7, 0, 0, 37, 41, 29, 13, 0, 0, 44, 36, 10, 0, 0, 61, 62, 30, 0, 10, 0, 50, 90, 29, 0, 0, 49, 59, 31, 0, 63, 44, 43, 0, 0, 0, 10, 46, 54, 36, 0, 0, 35, 40, 13, 0, 26, 42, 70, 0, 0, 0, 32, 46, 0, 35, 0, 5, 51, 40, 5, 14, 0, 56, 44, 21, 0, 0, 80, 66, 0, 46, 0, 0, 42, 42, 0, 0, 0, 62, 60, 3, 0, 7, 92, 65, 0, 37, 0, 19, 41, 26, 0, 20, 0, 43, 12, 1, 0, 4, 80, 46, 0, 45, 0, 14, 0, 38, 3, 15, 0, 66, 9, 3, 4, 7, 67, 60, 0, 16, 0, 20, 19, 32, 0, 16, 0, 50, 7, 22, 3, 3, 25, 61, 0, 37, 0, 20, 22, 27, 13, 1, 0, 43, 23, 0, 0, 0, 9, 32, 19, 23, 0, 0, 5, 30, 22, 11, 18, 45, 44, 0, 3, 0, 6, 44, 43, 23, 0, 0, 57, 55, 0, 0, 6, 59, 58, 1, 0, 0, 27, 57, 13, 29, 0, 0, 58, 28, 4, 0, 0, 49, 53, 0, 0, 0, 42, 51, 52, 39, 0, 0, 56, 46, 0, 0, 0, 62, 52, 0, 0, 0, 15, 37, 5, 14, 0, 0, 28, 40, 11, 0, 13, 64, 25, 41, 0, 0, 0, 69, 63, 36, 0, 0, 47, 51, 6, 0, 22, 48, 50, 0, 1, 17, 0, 50, 24, 33, 9, 12, 38, 26, 33, 15, 39, 37, 31, 28, 0, 27, 0, 58, 50, 59, 30, 3, 23, 55, 34, 0, 10, 45, 18, 0, 19, 17, 21, 41, 0, 24, 16, 25, 13, 42, 26, 38, 0, 44, 15, 2, 16, 5, 9, 43, 34, 27, 5, 2, 22, 51, 25, 5, 0, 49, 45, 0, 12, 14, 17, 49, 16, 28, 10, 16, 36, 35, 22, 33, 7, 35, 26, 9, 3, 1, 9, 37, 57, 38, 12, 0, 24, 64, 31, 0, 0, 59, 22, 0, 11, 0, 12, 34, 37, 3, 0, 0, 25, 45, 17, 0, 21, 45, 35, 0, 0, 8, 0, 40, 54, 26, 4, 0, 28, 45, 29, 0, 52, 43, 44, 0, 0, 8, 10, 60, 5, 45, 0, 5, 39, 27, 8, 14, 0, 38, 51, 0, 0, 5, 59, 60, 0, 54, 0, 32, 40, 32, 2, 25, 0, 58, 27, 31, 0, 0, 90, 58, 0, 35, 0, 6, 30, 47, 0, 0, 0, 67, 41, 12, 0, 10, 71, 58, 0, 24, 0, 10, 28, 30, 7, 16, 0, 38, 18, 0, 0, 0, 33, 33, 47, 29, 0, 0, 25, 60, 12, 0, 0, 64, 40, 0, 0, 0, 28, 37, 0, 0, 0, 0, 29, 24, 0, 4, 1, 49, 30, 35, 0, 8, 0, 60, 49, 45, 0, 0, 38, 54, 20, 0, 19, 56, 26, 3, 0, 0, 6, 58, 19, 28, 0, 0, 39, 25, 11, 0, 6, 41, 54, 12, 4, 0, 0, 43, 86, 37, 0, 0, 57, 60, 33, 0, 36, 53, 33, 0, 0, 12, 9, 43, 20, 29, 1, 0, 7, 33, 26, 5, 26, 43, 28, 31, 16, 31, 0, 48, 75, 35, 33, 0, 34, 74, 40, 0, 40, 50, 25, 0, 10, 19, 17, 59, 7, 34, 11, 23, 23, 18, 24, 48, 7, 20, 45, 0, 20, 2, 11, 31, 53, 44, 30, 15, 38, 66, 32, 20, 0, 60, 27, 0, 0, 0, 59, 39, 0, 11, 0, 0, 17, 41, 9, 7, 0, 58, 47, 0, 3, 7, 4, 64, 17, 29, 0, 0, 54, 27, 7, 3, 17, 26, 38, 0, 0, 0, 28, 46, 0, 57, 2, 16, 20, 29, 12, 17, 0, 59, 32, 18, 0, 0, 55, 51, 9, 21, 0, 0, 49, 57, 0, 0, 0, 66, 30, 0, 0, 0, 54, 52, 0, 17, 0, 0, 26, 11, 0, 0, 0, 47, 77, 0, 0, 20, 24, 66, 0, 54, 5, 10, 54, 37, 6, 2, 0, 47, 11, 18, 0, 0, 83, 61, 0, 49, 0, 0, 17, 35, 0, 0, 0, 68, 29, 21, 0, 18, 41, 52, 0, 15, 0, 23, 21, 32, 11, 14, 0, 51, 0, 48, 0, 39, 7, 68, 0, 51, 22, 18, 0, 34, 28, 9, 0, 34, 0, 0, 36, 37, 0, 45, 0, 34, 43, 56, 0, 33, 33, 73, 0, 35, 0, 0, 41, 0, 3, 42, 26, 26, 0, 12, 21, 48, 20, 5, 0, 49, 20, 0, 20, 23, 0, 33, 0, 12, 18, 26, 5, 26, 38, 46, 31, 36, 0, 8, 0, 30, 0, 59, 18, 52, 27, 9, 2, 41, 25, 12, 10, 33, 5, 0, 29, 10, 11, 45, 0, 30, 9, 51, 10, 18, 10, 66, 0, 44, 14, 10, 0, 18, 41, 66, 0, 49, 2, 44, 18, 24, 5, 31, 0, 50, 12, 30, 11, 4, 49, 60, 0, 38, 0, 26, 28, 39, 13, 4, 0, 54, 21, 21, 15, 0, 20, 40, 45, 18, 0, 0, 28, 50, 30, 0, 3, 47, 26, 0, 3, 0, 0, 28, 70, 13, 0, 0, 19, 51, 31, 0, 52, 47, 43, 5, 18, 16, 0, 36, 98, 19, 18, 0, 43, 67, 42, 0, 82, 41, 28, 0, 0, 0, 0, 42, 75, 27, 0, 0, 41, 37, 14, 0, 41, 41, 92, 0, 0, 7, 3, 44, 12, 36, 6, 7, 61, 34, 14, 23, 21, 49, 36, 38, 0, 0, 63, 65, 27, 57, 0, 0, 38, 57, 1, 0, 0, 66, 33, 0, 0, 0, 61, 42, 0, 7, 0, 0, 25, 27, 0, 3, 0, 51, 19, 18, 0, 0, 0, 48, 29, 32, 0, 0, 20, 49, 15, 0, 7, 59, 37, 8, 0, 35, 2, 69, 0, 36, 9, 25, 30, 20, 17, 29, 7, 26, 0, 26, 10, 0, 0, 44, 50, 51, 14, 5, 13, 60, 34, 0, 0, 62, 12, 2, 24, 25, 0, 40, 0, 2, 8, 19, 6, 28, 32, 39, 29, 30, 0, 9, 1, 40, 0, 51, 31, 52, 47, 18, 0, 42, 42, 26, 33, 33, 3, 0, 48, 9, 0, 44, 1, 27, 15, 34, 24, 37, 22, 42, 0, 49, 49, 3, 0, 3, 11, 39, 20, 27, 0, 0, 31, 46, 24, 0, 6, 62, 43, 30, 0, 18, 34, 72, 31, 43, 0, 0, 42, 43, 11, 0, 0, 32, 22, 0, 11, 0, 20, 23, 47, 22, 0, 0, 29, 52, 16, 0, 0, 69, 78, 0, 0, 0, 36, 47, 55, 4, 0, 0, 84, 44, 0, 0, 21, 52, 77, 39, 0, 30, 0, 53, 61, 51, 21, 0, 36, 52, 42, 0, 48, 49, 26, 3, 0, 25, 51, 66, 0, 52, 11, 2, 9, 37, 7, 17, 0, 38, 26, 1, 42, 0, 22, 36, 24, 23, 7, 22, 41, 66, 23, 8, 0, 66, 22, 25, 0, 0, 30, 49, 34, 15, 0, 0, 24, 36, 24, 0, 11, 38, 51, 0, 7, 29, 0, 38, 80, 35, 42, 0, 30, 60, 47, 15, 61, 36, 16, 0, 10, 0, 11, 30, 63, 22, 0, 0, 25, 56, 16, 0, 10, 56, 67, 0, 19, 3, 0, 37, 50, 9, 0, 0, 60, 51, 25, 0, 44, 49, 49, 29, 0, 9, 10, 61, 79, 52, 2, 0, 42, 51, 27, 0, 29, 42, 64, 0, 5, 7, 24, 38, 47, 30, 15, 2, 45, 54, 26, 18, 12, 50, 44, 19, 0, 9, 34, 44, 58, 32, 9, 0, 32, 65, 28, 0, 4, 50, 14, 0, 0, 9, 12, 29, 22, 13, 6, 0, 0, 47, 28, 16, 18, 45, 8, 0, 19, 4, 0, 37, 43, 16, 0, 0, 19, 53, 22, 0, 34, 46, 28, 0, 18, 0, 0, 48, 35, 23, 0, 0, 47, 30, 14, 0, 26, 37, 36, 0, 0, 0, 0, 39, 37, 37, 0, 0, 28, 41, 28, 0, 26, 54, 32, 0, 0, 2, 30, 64, 0, 39, 0, 0, 27, 19, 0, 0, 0, 44, 41, 40, 9, 16, 1, 59, 27, 46, 10, 18, 51, 58, 26, 0, 0, 57, 0, 11, 0, 0, 45, 51, 0, 27, 0, 0, 0, 20, 12, 0, 0, 42, 54, 0, 14, 6, 0, 43, 25, 27, 6, 4, 49, 48, 16, 14, 13, 48, 37, 0, 0, 0, 50, 59, 7, 43, 0, 0, 39, 34, 2, 0, 0, 55, 51, 32, 2, 8, 16, 44, 48, 25, 1, 0, 46, 63, 30, 0, 6, 56, 15, 0, 0, 0, 33, 46, 33, 24, 0, 0, 9, 34, 11, 0, 10, 41, 59, 0, 10, 33, 0, 40, 16, 29, 36, 21, 35, 49, 31, 35, 32, 48, 16, 43, 0, 30, 32, 75, 10, 59, 17, 19, 21, 41, 20, 4, 0, 35, 10, 0, 28, 0, 32, 30, 0, 22, 6, 27, 9, 40, 23, 42, 0, 55, 50, 2, 4, 0, 0, 42, 66, 17, 0, 0, 56, 51, 20, 0, 35, 51, 37, 23, 0, 8, 0, 41, 80, 29, 1, 0, 25, 52, 39, 0, 65, 44, 37, 0, 0, 0, 0, 45, 58, 26, 0, 0, 44, 40, 6, 0, 34, 39, 50, 0, 1, 0, 0, 28, 48, 20, 0, 0, 52, 50, 14, 0, 21, 73, 46, 64, 0, 18, 0, 65, 66, 29, 0, 0, 46, 49, 28, 0, 56, 46, 37, 33, 0, 16, 0, 58, 111, 46, 16, 0, 44, 52, 45, 0, 72, 25, 18, 0, 4, 11, 0, 14, 36, 25, 34, 4, 0, 42, 45, 41, 49, 49, 15, 6, 26, 1, 0, 46, 70, 17, 0, 0, 44, 68, 18, 0, 35, 49, 15, 11, 21, 5, 0, 38, 56, 16, 0, 0, 23, 32, 45, 0, 67, 34, 34, 0, 0, 0, 0, 48, 63, 42, 0, 0, 35, 34, 19, 0, 51, 34, 42, 0, 15, 0, 0, 30, 51, 27, 4, 0, 42, 60, 31, 0, 27, 65, 5, 35, 0, 23, 0, 61, 1, 30, 0, 0, 0, 24, 22, 0, 23, 33, 2, 15, 24, 35, 0, 48, 40, 41, 40, 19, 12, 50, 43, 12, 40, 37, 0, 0, 24, 27, 0, 45, 0, 31, 27, 34, 0, 23, 37, 45, 15, 34, 15, 0, 38, 16, 0, 56, 16, 40, 19, 30, 36, 35, 23, 33, 3, 32, 28, 0, 17, 0, 26, 40, 17, 34, 0, 7, 40, 34, 12, 5, 0, 55, 53, 0, 0, 0, 46, 41, 0, 21, 0, 0, 36, 36, 2, 0, 0, 61, 55, 20, 0, 0, 43, 64, 39, 36, 0, 0, 61, 51, 0, 0, 0, 56, 63, 1, 0, 12, 52, 59, 0, 41, 0, 11, 36, 25, 12, 16, 0, 42, 14, 36, 0, 0, 20, 38, 62, 36, 0, 0, 19, 78, 26, 0, 0, 73, 33, 0, 5, 0, 34, 50, 0, 0, 0, 0, 33, 16, 1, 17, 12, 27, 48, 3, 0, 20, 0, 44, 56, 55, 33, 4, 33, 58, 37, 0, 27, 55, 13, 8, 0, 5, 34, 54, 0, 28, 0, 2, 15, 35, 10, 3, 0, 42, 4, 3, 15, 3, 0, 33, 23, 20, 2, 1, 9, 49, 29, 0, 11, 55, 25, 0, 0, 9, 6, 63, 0, 30, 0, 4, 31, 17, 6, 17, 7, 30, 41, 26, 18, 18, 0, 47, 60, 50, 32, 19, 44, 63, 43, 0, 11, 58, 0, 34, 3, 2, 4, 41, 56, 16, 0, 0, 6, 48, 35, 0, 25, 41, 36, 0, 22, 8, 0, 31, 68, 16, 15, 0, 30, 46, 34, 18, 71, 54, 50, 0, 0, 0, 0, 28, 67, 6, 0, 0, 53, 48, 12, 0, 37, 64, 66, 10, 0, 0, 0, 56, 67, 30, 0, 0, 54, 41, 15, 0, 64, 42, 56, 0, 0, 4, 8, 61, 21, 55, 0, 0, 41, 33, 8, 0, 6, 46, 48, 0, 0, 0, 60, 52, 14, 36, 0, 0, 63, 46, 0, 0, 0, 68, 44, 17, 0, 0, 42, 43, 0, 13, 0, 0, 31, 36, 7, 0, 0, 64, 18, 41, 0, 12, 0, 57, 46, 30, 0, 0, 16, 48, 20, 0, 40, 50, 50, 0, 17, 5, 0, 66, 61, 37, 0, 0, 71, 38, 18, 0, 36, 24, 35, 0, 0, 0, 10, 25, 36, 39, 0, 0, 22, 38, 22, 0, 0, 67, 59, 0, 0, 0, 27, 47, 43, 10, 0, 0, 63, 58, 0, 0, 9, 61, 57, 8, 0, 0, 25, 63, 30, 37, 0, 0, 50, 28, 7, 0, 11, 41, 54, 0, 0, 0, 18, 45, 38, 42, 0, 0, 42, 51, 13, 0, 0, 66, 58, 21, 0, 0, 45, 61, 39, 27, 0, 0, 64, 49, 4, 0, 0, 50, 48, 10, 0, 0, 22, 41, 56, 31, 0, 0, 34, 48, 25, 0, 15, 56, 65, 0, 0, 15, 28, 53, 40, 32, 4, 0, 45, 49, 17, 0, 19, 40, 27, 7, 0, 15, 19, 41, 22, 40, 19, 6, 10, 54, 28, 7, 0, 54, 13, 0, 12, 0, 33, 43, 17, 12, 0, 0, 29, 45, 2, 0, 0, 50, 52, 0, 2, 0, 7, 45, 18, 23, 0, 0, 47, 32, 14, 3, 12, 43, 28, 9, 0, 2, 11, 48, 22, 43, 0, 0, 17, 45, 18, 0, 0, 61, 29, 38, 10, 25, 0, 56, 33, 24, 6, 2, 36, 53, 28, 0, 19, 39, 0, 0, 0, 0, 4, 44, 6, 35, 0, 0, 0, 20, 18, 17, 2, 38, 46, 0, 17, 23, 0, 45, 0, 33, 23, 32, 37, 42, 18, 39, 0, 50, 2, 46, 0, 19, 31, 68, 0, 50, 2, 20, 12, 37, 18, 0, 0, 46, 13, 0, 19, 3, 28, 46, 0, 24, 0, 22, 18, 31, 16, 32, 0, 42, 32, 0, 0, 4, 35, 48, 0, 36, 0, 17, 23, 31, 9, 29, 0, 48, 35, 8, 0, 29, 47, 59, 0, 45, 16, 38, 18, 35, 12, 36, 0, 47, 0, 34, 0, 18, 36, 50, 0, 35, 7, 27, 0, 42, 20, 14, 0, 53, 0, 0, 24, 9, 16, 49, 0, 13, 0, 29, 0, 15, 6, 48, 0, 35, 17, 0, 25, 9, 0, 54, 5, 41, 8, 27, 34, 37, 21, 13, 0, 51, 49, 0, 0, 0, 41, 50, 0, 28, 0, 0, 37, 18, 0, 5, 0, 57, 46, 40, 0, 0, 44, 72, 10, 47, 0, 0, 61, 49, 0, 0, 0, 66, 39, 26, 0, 14, 57, 66, 0, 38, 0, 15, 19, 12, 10, 10, 0, 37, 18, 0, 0, 0, 45, 52, 0, 49, 0, 0, 16, 46, 3, 0, 0, 59, 32, 0, 2, 0, 64, 53, 0, 18, 0, 24, 31, 21, 0, 27, 0, 52, 31, 17, 0, 5, 66, 72, 0, 53, 0, 16, 29, 26, 0, 0, 0, 48, 0, 17, 0, 16, 41, 45, 0, 32, 2, 32, 0, 32, 15, 24, 0, 62, 6, 3, 0, 8, 61, 78, 0, 35, 0, 20, 27, 17, 0, 19, 0, 28, 13, 0, 2, 0, 37, 41, 0, 42, 0, 47, 5, 18, 8, 47, 0, 63, 13, 30, 0, 22, 63, 72, 0, 39, 0, 30, 14, 26, 0, 11, 0, 49, 0, 19, 3, 0, 50, 64, 0, 40, 0, 25, 19, 26, 3, 6, 0, 45, 0, 0, 5, 0, 13, 34, 0, 19, 0, 18, 0, 29, 19, 20, 0, 57, 11, 17, 9, 0, 0, 57, 35, 23, 0, 0, 38, 45, 11, 0, 11, 44, 25, 14, 19, 0, 0, 38, 70, 23, 0, 0, 35, 44, 43, 0, 57, 46, 39, 8, 0, 9, 0, 48, 94, 34, 7, 0, 39, 49, 37, 0, 73, 32, 45, 0, 9, 0, 0, 30, 81, 26, 0, 0, 48, 52, 21, 0, 40, 53, 63, 0, 0, 0, 0, 40, 44, 22, 0, 0, 47, 48, 21, 0, 35, 54, 46, 16, 0, 19, 27, 70, 12, 52, 0, 0, 32, 34, 11, 0, 0, 36, 11, 7, 0, 20, 10, 40, 0, 39, 23, 27, 0, 47, 28, 23, 0, 62, 19, 2, 7, 6, 58, 72, 0, 31, 0, 16, 37, 24, 0, 17, 0, 33, 37, 5, 15, 10, 14, 39, 15, 41, 23, 32, 28, 48, 35, 24, 0, 58, 12, 20, 0, 12, 41, 49, 12, 25, 0, 0, 5, 45, 19, 2, 0, 43, 22, 0, 16, 21, 0, 41, 5, 25, 20, 19, 14, 41, 25, 39, 7, 42, 24, 9, 16, 15, 8, 54, 25, 39, 14, 16, 32, 48, 24, 12, 0, 46, 26, 23, 27, 5, 0, 38, 68, 23, 10, 1, 35, 59, 40, 0, 19, 47, 37, 0, 0, 21, 14, 41, 18, 28, 20, 5, 8, 29, 28, 47, 30, 31, 56, 0, 15, 13, 23, 49, 44, 47, 26, 12, 54, 67, 18, 14, 0, 51, 20, 0, 0, 0, 65, 36, 0, 19, 0, 9, 11, 37, 13, 15, 0, 46, 62, 5, 0, 0, 63, 61, 17, 45, 0, 6, 71, 54, 1, 0, 0, 57, 23, 0, 0, 0, 82, 38, 0, 24, 0, 0, 0, 24, 5, 6, 0, 66, 43, 43, 0, 31, 18, 65, 32, 29, 5, 0, 46, 65, 16, 0, 4, 45, 14, 0, 0, 1, 35, 55, 2, 39, 0, 13, 15, 21, 17, 25, 0, 35, 52, 0, 12, 15, 20, 38, 13, 35, 25, 24, 34, 53, 25, 34, 0, 57, 22, 31, 0, 2, 48, 54, 30, 30, 0, 0, 29, 55, 16, 0, 0, 53, 53, 0, 18, 0, 29, 41, 65, 16, 0, 0, 57, 50, 20, 0, 17, 43, 64, 0, 0, 0, 11, 29, 83, 27, 4, 0, 39, 61, 33, 0, 40, 58, 75, 0, 0, 0, 34, 51, 81, 27, 0, 0, 65, 64, 15, 0, 28, 46, 63, 4, 0, 2, 23, 36, 72, 30, 9, 0, 41, 62, 33, 0, 23, 53, 38, 0, 0, 16, 36, 41, 14, 26, 8, 0, 7, 45, 18, 17, 8, 46, 45, 0, 8, 8, 39, 57, 12, 36, 2, 9, 48, 50, 5, 13, 0, 44, 24, 0, 2, 0, 57, 34, 14, 21, 0, 0, 34, 43, 3, 0, 0, 67, 49, 0, 0, 0, 26, 38, 22, 5, 0, 0, 40, 41, 3, 0, 14, 57, 40, 21, 0, 6, 0, 57, 49, 36, 0, 0, 36, 45, 18, 0, 37, 46, 34, 0, 0, 0, 0, 49, 59, 29, 0, 0, 55, 42, 6, 0, 14, 52, 64, 0, 0, 0, 0, 40, 61, 21, 0, 0, 59, 44, 21, 0, 40, 59, 58, 44, 0, 11, 0, 59, 96, 40, 0, 0, 51, 59, 31, 0, 58, 45, 48, 0, 0, 27, 0, 49, 38, 40, 25, 2, 23, 39, 33, 22, 36, 34, 22, 6, 14, 8, 0, 38, 60, 36, 21, 0, 25, 70, 32, 0, 6, 57, 7, 1, 17, 11, 0, 35, 15, 6, 0, 2, 5, 38, 30, 18, 25, 40, 21, 0, 7, 29, 0, 51, 36, 39, 29, 8, 18, 40, 32, 22, 40, 32, 21, 0, 23, 27, 0, 48, 13, 43, 36, 36, 21, 42, 34, 39, 0, 40, 0, 7, 13, 16, 7, 41, 0, 31, 17, 25, 0, 40, 30, 29, 0, 44, 0, 0, 22, 14, 0, 34, 0, 14, 9, 16, 0, 35, 25, 30, 5, 48, 17, 19, 33, 40, 0, 65, 19, 36, 29, 32, 34, 36, 33, 27, 28, 23, 0, 13, 24, 11, 0, 34, 46, 42, 31, 20, 0, 45, 51, 16, 17, 47, 20, 0, 33, 6, 0, 38, 28, 11, 2, 4, 28, 37, 22, 28, 32, 41, 8, 0, 0, 9, 0, 27, 19, 24, 12, 0, 0, 41, 37, 12, 21, 56, 41, 0, 9, 31, 7, 70, 9, 35, 11, 12, 46, 38, 8, 22, 9, 23, 0, 14, 19, 1, 0, 30, 33, 41, 21, 21, 5, 54, 43, 4, 0, 66, 15, 2, 13, 9, 8, 49, 16, 7, 0, 0, 22, 34, 17, 7, 21, 31, 25, 3, 10, 36, 0, 42, 43, 43, 48, 20, 12, 47, 48, 27, 45, 37, 0, 0, 23, 2, 0, 39, 40, 27, 6, 0, 8, 50, 28, 0, 11, 44, 1, 0, 40, 0, 0, 22, 53, 0, 0, 0, 23, 44, 30, 0, 50, 49, 29, 0, 0, 19, 0, 50, 48, 29, 6, 0, 26, 28, 33, 0, 81, 28, 7, 20, 18, 1, 0, 41, 102, 41, 12, 0, 32, 64, 44, 0, 63, 53, 12, 0, 27, 0, 0, 33, 82, 4, 0, 0, 32, 39, 38, 0, 85, 40, 60, 0, 1, 24, 0, 52, 109, 43, 28, 0, 54, 45, 47, 0, 110, 24, 26, 0, 0, 30, 0, 38, 53, 52, 49, 11, 5, 51, 49, 19, 43, 45, 11, 0, 23, 0, 23, 43, 31, 18, 0, 0, 40, 47, 0, 0, 0, 50, 63, 0, 10, 0, 14, 35, 35, 11, 0, 0, 69, 34, 4, 0, 8, 59, 74, 17, 0, 0, 22, 58, 53, 42, 0, 0, 55, 45, 10, 0, 21, 51, 45, 0, 0, 0, 49, 54, 0, 41, 0, 0, 20, 28, 0, 0, 0, 54, 31, 11, 0, 0, 41, 59, 0, 34, 0, 0, 41, 46, 0, 0, 0, 63, 1, 30, 0, 2, 24, 57, 0, 26, 0, 3, 4, 20, 10, 0, 0, 53, 22, 43, 12, 24, 0, 62, 56, 37, 10, 0, 40, 52, 33, 0, 32, 35, 1, 0, 1, 11, 0, 38, 5, 35, 16, 16, 0, 20, 34, 39, 18, 37, 35, 0, 32, 0, 0, 46, 57, 30, 4, 0, 55, 62, 19, 0, 13, 52, 33, 10, 0, 0, 7, 46, 27, 26, 0, 1, 29, 32, 28, 0, 12, 44, 35, 20, 0, 19, 0, 43, 75, 35, 22, 0, 26, 62, 39, 0, 44, 47, 30, 0, 15, 2, 0, 43, 60, 22, 1, 0, 35, 48, 26, 1, 32, 39, 63, 0, 3, 0, 17, 43, 40, 34, 3, 0, 50, 41, 18, 18, 13, 43, 37, 23, 0, 17, 11, 36, 38, 34, 19, 0, 16, 65, 35, 0, 3, 63, 9, 0, 0, 14, 36, 58, 0, 22, 0, 0, 8, 28, 6, 16, 0, 31, 28, 0, 25, 5, 0, 40, 18, 35, 14, 20, 33, 47, 23, 20, 0, 53, 35, 4, 20, 0, 7, 65, 47, 31, 0, 0, 64, 44, 8, 0, 5, 36, 38, 8, 9, 0, 0, 26, 68, 28, 11, 0, 28, 52, 48, 0, 34, 57, 32, 0, 0, 8, 9, 48, 41, 24, 0, 0, 18, 41, 16, 0, 37, 36, 41, 0, 20, 0, 0, 38, 62, 29, 5, 0, 49, 61, 25, 0, 25, 57, 33, 22, 0, 19, 0, 52, 25, 29, 4, 1, 24, 36, 30, 0, 23, 37, 25, 0, 0, 5, 14, 50, 19, 44, 5, 0, 19, 37, 15, 13, 0, 41, 42, 0, 14, 0, 49, 39, 5, 21, 0, 0, 58, 42, 0, 0, 0, 68, 74, 17, 0, 0, 52, 55, 30, 23, 0, 0, 72, 43, 1, 0, 0, 57, 49, 0, 0, 0, 52, 50, 23, 36, 0, 0, 28, 34, 0, 0, 0, 55, 58, 0, 0, 0, 40, 51, 0, 27, 0, 0, 45, 39, 0, 0, 0, 62, 26, 57, 0, 1, 16, 64, 22, 36, 0, 0, 39, 50, 13, 0, 0, 63, 28, 30, 0, 13, 3, 63, 31, 31, 0, 0, 31, 29, 25, 0, 24, 27, 34, 0, 0, 0, 3, 34, 53, 37, 0, 0, 35, 45, 14, 0, 8, 53, 55, 0, 0, 0, 28, 36, 4, 13, 0, 0, 48, 41, 0, 0, 0, 65, 43, 34, 0, 9, 24, 69, 2, 41, 0, 0, 36, 34, 6, 0, 0, 53, 53, 24, 0, 0, 33, 71, 46, 48, 0, 0, 69, 45, 9, 0, 0, 43, 50, 0, 0, 0, 47, 30, 42, 25, 0, 0, 38, 45, 16, 0, 0, 65, 77, 0, 0, 7, 55, 53, 13, 26, 0, 0, 47, 44, 2, 0, 2, 44, 18, 23, 0, 24, 26, 48, 0, 46, 15, 7, 0, 49, 21, 0, 0, 62, 19, 11, 6, 32, 51, 76, 0, 35, 2, 38, 25, 23, 1, 42, 0, 24, 0, 0, 17, 0, 25, 34, 0, 40, 0, 26, 2, 41, 21, 17, 0, 66, 21, 3, 25, 0, 16, 40, 29, 0, 0, 0, 43, 47, 9, 0, 5, 50, 33, 7, 0, 1, 0, 42, 70, 26, 0, 0, 29, 39, 34, 0, 67, 38, 32, 0, 0, 25, 0, 45, 35, 40, 25, 0, 15, 40, 32, 9, 50, 41, 13, 24, 24, 6, 0, 52, 59, 35, 4, 0, 40, 59, 28, 0, 16, 49, 17, 0, 13, 0, 0, 40, 32, 18, 0, 0, 20, 27, 32, 8, 34, 39, 53, 0, 10, 1, 0, 47, 102, 37, 8, 0, 60, 63, 31, 0, 54, 43, 43, 0, 0, 0, 0, 33, 47, 26, 0, 0, 26, 43, 30, 0, 31, 44, 29, 0, 0, 0, 0, 20, 60, 14, 0, 0, 20, 52, 27, 0, 27, 65, 43, 1, 0, 0, 0, 46, 69, 10, 0, 0, 44, 51, 16, 0, 68, 45, 55, 5, 0, 0, 0, 55, 97, 40, 0, 0, 67, 49, 27, 0, 61, 38, 44, 0, 0, 0, 0, 33, 46, 34, 0, 0, 24, 39, 27, 0, 32, 56, 44, 26, 0, 12, 0, 52, 65, 28, 0, 0, 47, 64, 23, 0, 35, 50, 29, 0, 0, 26, 7, 59, 0, 41, 13, 17, 12, 18, 24, 33, 11, 25, 22, 0, 0, 0, 29, 46, 0, 49, 8, 19, 21, 44, 7, 23, 0, 55, 5, 0, 1, 0, 48, 43, 0, 15, 0, 24, 5, 27, 1, 19, 0, 63, 27, 19, 0, 0, 47, 78, 0, 37, 0, 0, 54, 27, 0, 0, 0, 39, 55, 0, 0, 2, 43, 51, 0, 47, 0, 30, 32, 20, 14, 32, 0, 51, 25, 23, 0, 0, 65, 50, 29, 35, 0, 0, 35, 64, 0, 0, 0, 72, 32, 8, 4, 0, 18, 37, 30, 0, 0, 0, 37, 41, 13, 0, 19, 51, 55, 0, 0, 0, 0, 54, 83, 35, 0, 0, 56, 41, 13, 0, 54, 39, 79, 0, 0, 0, 6, 40, 57, 36, 0, 0, 68, 50, 12, 0, 18, 63, 65, 32, 0, 0, 38, 55, 65, 32, 0, 0, 61, 59, 10, 0, 5, 58, 26, 16, 0, 0, 0, 29, 71, 11, 0, 0, 16, 55, 28, 0, 47, 57, 40, 0, 0, 12, 0, 51, 36, 21, 0, 0, 31, 33, 15, 0, 64, 32, 35, 18, 0, 48, 0, 61, 12, 62, 49, 36, 23, 41, 39, 23, 18, 44, 5, 28, 8, 26, 38, 71, 0, 54, 17, 41, 18, 32, 18, 31, 0, 35, 5, 3, 24, 38, 19, 35, 0, 33, 49, 63, 0, 38, 42, 79, 0, 47, 0, 0, 26, 0, 51, 52, 23, 24, 0, 0, 28, 49, 1, 0, 0, 39, 35, 0, 33, 0, 13, 8, 29, 0, 0, 0, 42, 39, 10, 0, 1, 68, 74, 6, 0, 0, 7, 58, 49, 26, 0, 0, 60, 42, 5, 0, 41, 50, 65, 27, 0, 35, 12, 73, 15, 64, 19, 8, 39, 36, 22, 0, 6, 36, 0, 5, 0, 0, 47, 45, 0, 43, 0, 8, 0, 46, 11, 0, 0, 64, 22, 0, 18, 0, 39, 48, 0, 5, 0, 6, 34, 32, 0, 9, 0, 45, 11, 17, 0, 0, 0, 41, 57, 28, 0, 0, 27, 49, 22, 0, 16, 59, 46, 0, 0, 1, 0, 48, 43, 17, 0, 0, 45, 32, 19, 0, 54, 53, 35, 0, 0, 0, 14, 56, 4, 24, 0, 0, 48, 25, 0, 0, 0, 40, 53, 0, 0, 0, 18, 50, 8, 44, 0, 0, 50, 35, 7, 0, 0, 60, 25, 23, 0, 0, 37, 53, 0, 29, 0, 0, 22, 38, 3, 0, 0, 64, 44, 10, 0, 2, 26, 69, 6, 32, 0, 0, 50, 30, 1, 0, 0, 37, 25, 6, 0, 0, 3, 42, 22, 42, 0, 0, 20, 42, 24, 0, 0, 64, 40, 34, 0, 28, 7, 63, 24, 28, 5, 3, 39, 46, 23, 0, 15, 37, 24, 0, 0, 9, 16, 53, 32, 48, 13, 9, 21, 37, 26, 17, 0, 39, 55, 0, 14, 0, 45, 40, 12, 29, 5, 15, 44, 47, 13, 31, 0, 50, 25, 1, 0, 0, 56, 39, 0, 26, 0, 0, 13, 48, 10, 0, 0, 67, 36, 28, 0, 24, 17, 57, 18, 20, 0, 0, 34, 49, 20, 0, 9, 37, 5, 0, 0, 0, 15, 43, 19, 36, 0, 0, 9, 31, 14, 3, 0, 43, 43, 0, 15, 0, 8, 33, 3, 14, 0, 0, 45, 42, 4, 0, 0, 60, 36, 16, 0, 0, 29, 64, 1, 37, 0, 0, 39, 29, 0, 0, 0, 50, 35, 9, 0, 0, 13, 53, 11, 36, 0, 0, 35, 37, 11, 0, 0, 52, 15, 9, 0, 0, 6, 47, 10, 27, 0, 0, 18, 37, 11, 0, 0, 54, 38, 0, 0, 0, 16, 60, 3, 29, 0, 0, 50, 24, 0, 0, 0, 45, 67, 8, 0, 0, 24, 62, 14, 51, 0, 5, 60, 37, 12, 0, 0, 58, 37, 60, 0, 26, 38, 60, 22, 46, 11, 4, 21, 52, 30, 0, 0, 52, 0, 24, 5, 24, 8, 41, 32, 22, 20, 2, 0, 52, 36, 13, 14, 43, 17, 3, 43, 39, 0, 40, 54, 21, 47, 21, 19, 54, 48, 40, 53, 24, 0, 0, 24, 0, 0, 28, 36, 29, 12, 3, 0, 38, 29, 31, 17, 40, 47, 0, 23, 0, 16, 35, 0, 13, 0, 0, 55, 32, 0, 15, 0, 54, 56, 19, 0, 0, 31, 60, 3, 42, 0, 0, 50, 38, 5, 0, 0, 65, 31, 73, 0, 32, 13, 66, 27, 42, 6, 0, 23, 49, 31, 0, 5, 45, 16, 0, 0, 4, 25, 54, 23, 37, 0, 0, 20, 32, 16, 18, 0, 31, 43, 0, 22, 0, 11, 26, 19, 24, 8, 13, 35, 51, 21, 25, 0, 59, 30, 0, 0, 0, 66, 54, 0, 25, 0, 0, 31, 29, 0, 0, 0, 53, 54, 1, 0, 0, 25, 50, 3, 29, 0, 0, 49, 43, 7, 0, 0
};

static TinyTensor reshape_expected = {"reshape", reshape_expected_data, TINY_DTYPE_INT8, 3, {28, 28, 16, 1}, 12544};

static uint32_t tinynpu_static_ub_image[20][TINY_BUFFER_WORDS_32] __attribute__((section(".data"))) = {
    {0x00004803u, 0x00002b22u, 0xfffffffau, 0xfffffff8u},
    {0x00000770u, 0x00000c94u, 0x000048e8u, 0x0000208du},
    {0x00002f68u, 0xfffffcc7u, 0xfffffffbu, 0x000029d3u},
    {0x0000405bu, 0x00001e64u, 0xfffffffcu, 0x000002acu},
    {0xcc0207ccu, 0x244504b8u, 0xb4991d00u, 0x26320fdfu},
    {0x02d50200u, 0x3311be43u, 0x16a7f92bu, 0xf02ae4ecu},
    {0x2bd90ae7u, 0x0bfd3704u, 0xfb1b3907u, 0x1e141becu},
    {0xcbccf007u, 0x1a3124f3u, 0x263b1ae8u, 0xbc961cf8u},
    {0x00f700fdu, 0x00df00f1u, 0x0010003cu, 0x0081000cu},
    {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x163cf709u, 0xeb07fd26u, 0x1024ef06u, 0x51391b3eu},
    {0x0335d5f4u, 0x01d00806u, 0xf5340d1du, 0xfd4335f4u},
    {0x382c1411u, 0x20c72410u, 0x1a1b0d0cu, 0x18eb1928u},
    {0x1de022e3u, 0xf0e53c36u, 0xf1f20ddbu, 0x90a5193du},
    {0x00350015u, 0x00c60045u, 0x000100d9u, 0x00dd0067u},
    {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u}
};

static uint32_t im_segment_000[4][TINY_BUFFER_WORDS_32] __attribute__((section(".data"))) = {
    {0x00000000u, 0x00000000u, 0x01440000u, 0x00175337u},
    {0x01000200u, 0x00006200u, 0x04001400u, 0x20032400u},
    {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00000000u, 0x00000000u, 0x00000000u, 0x10000000u}
};

int main(void)
{
    printf("TinyNPU bare-metal program: cv32e40p_mnist_conv1_demo\n");
    load_ub_image(0u, tinynpu_static_ub_image, 20);
    load_im_image(0x8000u, im_segment_000, 4);
    printf("HostOp im2col: im2col_for_npu\n");
    host_im2col(&im2col_for_npu, &x, 3, 1, 1, 0);
    printf("NpuSegment: segment_000\n");
    write_tensor_to_npu(&im2col_for_npu, 0x0324u, "A", 1, 784);
    write_tensor_to_npu(&add_bias, 0x0000u, "BIAS", 2, 4);
    if (npu_run(0x8000u) != 0) return EXIT_FAILURE;
    read_role_c_tensor(&add, 0x0014u, 1);
    printf("HostOp reshape: reshape\n");
    host_reshape(&reshape, &add);
    printf("Final outputs:\n");
    print_tensor(&reshape);
    if (!tensor_matches_expected(&reshape, &reshape_expected)) {
        printf("verification failed: reshape (reshape)\n");
        printf("meta actual dtype=%d elems=%d expected dtype=%d elems=%d\n", reshape.dtype, reshape.elem_count, reshape_expected.dtype, reshape_expected.elem_count);
        print_tensor(&reshape);
        print_tensor(&reshape_expected);
        return EXIT_FAILURE;
    }
    printf("All outputs matched expected tensors\n");
    return EXIT_SUCCESS;
}
