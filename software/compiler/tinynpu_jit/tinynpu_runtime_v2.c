#include "tinynpu_runtime_v2.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define __TINY_ARRAY_SIZE__ 8
#define __TINY_BUFFER_WORDS_32__ TNPU_MMVR_WORDS_32
#define __GENERATED_DECLS__
#define __GENERATED_MAIN__ return EXIT_SUCCESS;
#define main tinynpu_runtime_v2_template_main
#include "templates/cv32e40p_runtime_template.c"
#undef main
#undef __GENERATED_MAIN__
#undef __GENERATED_DECLS__
#undef __TINY_BUFFER_WORDS_32__
#undef __TINY_ARRAY_SIZE__

#define TNPU_ARRAY_SIZE 8

#ifndef TNPU_RUNTIME_V2_DUMP_FINAL_OUTPUTS
#define TNPU_RUNTIME_V2_DUMP_FINAL_OUTPUTS 1
#endif

static const char *tnpu_role_or_default(const char *role, const char *fallback)
{
    if (role == NULL || role[0] == '\0') {
        return fallback;
    }
    return role;
}

static inline char tnpu_role_code(const char *role, char fallback)
{
    if (role == NULL || role[0] == '\0') {
        return fallback;
    }
    return role[0];
}

static inline void tnpu_lanes_to_chunks_u16(const uint16_t lanes[8], uint32_t chunks[TNPU_MMVR_WORDS_32])
{
    chunks[0] = (uint32_t)lanes[0] | ((uint32_t)lanes[1] << 16);
    chunks[1] = (uint32_t)lanes[2] | ((uint32_t)lanes[3] << 16);
    chunks[2] = (uint32_t)lanes[4] | ((uint32_t)lanes[5] << 16);
    chunks[3] = (uint32_t)lanes[6] | ((uint32_t)lanes[7] << 16);
}

static inline int tnpu_precision_to_packed_count(int precision)
{
    if (precision == 2) {
        return 1;  // int16
    }
    if (precision == 1) {
        return 2;  // int8
    }
    if (precision == 0) {
        return 4;  // int4
    }
    return 0;
}

#define TNPU_XFORM_SCRATCH_IM_ADDR 0x87F8u
#define TNPU_ISA_OP_XFORM 0x4u
#define TNPU_ISA_OP_HALT 0x2u
#define TNPU_XFORM_MODE_Q_F16_I16 0x1u

static inline uint16_t tnpu_float32_to_fp16_bits(float value)
{
    union {
        float f;
        uint32_t u;
    } v = {value};
    const uint32_t sign = (v.u >> 16) & 0x8000u;
    const uint32_t exp = (v.u >> 23) & 0xFFu;
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
        const uint32_t round_bits = mant & 0x1FFFu;
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

static void tnpu_write_tensor_a_f16bits_from_float(
    const TinyTensor *tensor,
    uint16_t base_addr,
    int word_count)
{
    const int rows = tensor->rank == 1 ? 1 : tensor->shape[0];
    const int cols = tensor->rank == 1 ? tensor->shape[0] : tensor->shape[1];
    const int m_tiles = (rows + TNPU_ARRAY_SIZE - 1) / TNPU_ARRAY_SIZE;
    const int k_tiles = (cols + TNPU_ARRAY_SIZE - 1) / TNPU_ARRAY_SIZE;
    uint16_t addr = base_addr;
    uint16_t lanes[TNPU_ARRAY_SIZE];
    uint32_t chunks[TNPU_MMVR_WORDS_32];

    runtime_assert(word_count == m_tiles * k_tiles * TNPU_ARRAY_SIZE, "role A fp16 write word count mismatch");
    for (int mt = 0; mt < m_tiles; ++mt) {
        for (int kt = 0; kt < k_tiles; ++kt) {
            for (int lane_selector = 0; lane_selector < TNPU_ARRAY_SIZE; ++lane_selector) {
                for (int lane = 0; lane < TNPU_ARRAY_SIZE; ++lane) {
                    const int row = mt * TNPU_ARRAY_SIZE + lane;
                    const int col = kt * TNPU_ARRAY_SIZE + lane_selector;
                    uint16_t value = 0u;
                    if (row < rows && col < cols) {
                        value = tnpu_float32_to_fp16_bits(tensor_get_float(tensor, row * cols + col));
                    }
                    lanes[lane] = value;
                }
                tnpu_lanes_to_chunks_u16(lanes, chunks);
                npu_write_mem_word(addr++, chunks);
            }
        }
    }
}

static inline void tnpu_pack_field_u32(uint32_t words[8], uint16_t lsb, uint8_t width, uint32_t value)
{
    uint16_t bit = lsb;
    uint8_t remaining = width;
    uint32_t v = value;
    while (remaining > 0) {
        const uint16_t idx = bit >> 5;
        const uint8_t off = (uint8_t)(bit & 31u);
        const uint8_t take = (uint8_t)((remaining < (uint8_t)(32u - off)) ? remaining : (uint8_t)(32u - off));
        const uint32_t mask = (take == 32u) ? 0xFFFFFFFFu : ((1u << take) - 1u);
        words[idx] |= (v & mask) << off;
        v >>= take;
        bit = (uint16_t)(bit + take);
        remaining = (uint8_t)(remaining - take);
    }
}

static void tnpu_pack_xform_q_f16_i16_words(
    uint32_t words[8],
    uint16_t src,
    uint16_t dst,
    uint16_t count,
    uint16_t multiplier,
    uint8_t shift)
{
    memset(words, 0, sizeof(uint32_t) * 8u);
    tnpu_pack_field_u32(words, 252, 4, TNPU_ISA_OP_XFORM);
    tnpu_pack_field_u32(words, 248, 4, TNPU_XFORM_MODE_Q_F16_I16);
    tnpu_pack_field_u32(words, 232, 16, src);
    tnpu_pack_field_u32(words, 216, 16, dst);
    tnpu_pack_field_u32(words, 200, 16, count);
    tnpu_pack_field_u32(words, 184, 16, multiplier);
    tnpu_pack_field_u32(words, 176, 8, shift);
}

static void tnpu_pack_halt_words(uint32_t words[8])
{
    memset(words, 0, sizeof(uint32_t) * 8u);
    tnpu_pack_field_u32(words, 252, 4, TNPU_ISA_OP_HALT);
}

static void tnpu_run_xform_q_f16_i16(
    uint16_t src_addr,
    uint16_t dst_addr,
    uint16_t count,
    uint16_t multiplier,
    uint8_t shift)
{
    uint32_t xform_words[8];
    uint32_t halt_words[8];

    tnpu_pack_xform_q_f16_i16_words(xform_words, src_addr, dst_addr, count, multiplier, shift);
    tnpu_pack_halt_words(halt_words);
    npu_write_mem_word(TNPU_XFORM_SCRATCH_IM_ADDR + 0u, &xform_words[0]);
    npu_write_mem_word(TNPU_XFORM_SCRATCH_IM_ADDR + 1u, &xform_words[4]);
    npu_write_mem_word(TNPU_XFORM_SCRATCH_IM_ADDR + 2u, &halt_words[0]);
    npu_write_mem_word(TNPU_XFORM_SCRATCH_IM_ADDR + 3u, &halt_words[4]);
    runtime_assert(npu_run((uint32_t)TNPU_XFORM_SCRATCH_IM_ADDR) == 0, "xform execution failed");
}

static void tnpu_choose_xform_scale_params(float inv_scale, uint16_t *multiplier, uint8_t *shift)
{
    float best_err = 3.402823466e+38f;
    uint16_t best_mult = 1u;
    uint8_t best_shift = 0u;
    if (inv_scale >= 1.0f) {
        const uint32_t rounded = (uint32_t)(inv_scale + 0.5f);
        if ((float)rounded == inv_scale && rounded <= 65535u) {
            *multiplier = (uint16_t)rounded;
            *shift = 0u;
            return;
        }
    }
    for (uint8_t s = 0; s <= 15u; ++s) {
        const float scaled = inv_scale * (float)(1u << s);
        uint32_t m;
        float approx;
        float err;
        if (scaled > 65535.0f) {
            break;
        }
        m = (uint32_t)(scaled + 0.5f);
        if (m == 0u) {
            m = 1u;
        }
        approx = (float)m / (float)(1u << s);
        err = (approx >= inv_scale) ? (approx - inv_scale) : (inv_scale - approx);
        if (err < best_err || (err == best_err && s > best_shift)) {
            best_err = err;
            best_mult = (uint16_t)m;
            best_shift = s;
        }
    }
    if (best_err >= 3.402823466e+38f) {
        best_mult = 65535u;
        best_shift = 0u;
    }
    *multiplier = best_mult;
    *shift = best_shift;
}

static inline int64_t tnpu_round_shift_right_signed(int64_t value, int shift)
{
    int64_t abs_v;
    int64_t rounded;
    if (shift <= 0) {
        return value;
    }
    if (shift >= 63) {
        return value < 0 ? -1 : 0;
    }
    if (value >= 0) {
        return (value + ((int64_t)1 << (shift - 1))) >> shift;
    }
    abs_v = -value;
    rounded = (abs_v + ((int64_t)1 << (shift - 1))) >> shift;
    return -rounded;
}

static inline int16_t tnpu_clip_i16(int64_t value)
{
    if (value > 32767) {
        return 32767;
    }
    if (value < -32768) {
        return -32768;
    }
    return (int16_t)value;
}

static int16_t tnpu_quantize_fp16_lane_to_i16(uint16_t fp16, uint16_t multiplier, uint8_t shift)
{
    int sign = (fp16 >> 15) & 1;
    int exp_bits = (fp16 >> 10) & 0x1F;
    int frac_bits = fp16 & 0x3FF;
    int64_t mant;
    int64_t scaled;
    int64_t qvalue;
    int exp2;

    if (multiplier == 0u) {
        return 0;
    }
    if (exp_bits == 0x1F) {
        return sign ? -32768 : 32767;
    }
    if (exp_bits == 0 && frac_bits == 0) {
        return 0;
    }

    if (exp_bits == 0) {
        mant = (int64_t)frac_bits;
        exp2 = -24;
    } else {
        mant = (int64_t)(1024 + frac_bits);
        exp2 = exp_bits - 25;
    }

    scaled = mant * (int64_t)multiplier;
    if (exp2 >= (int)shift) {
        int left_shift = exp2 - (int)shift;
        if (left_shift >= 47) {
            qvalue = 0x7FFFFFFFFFFFFFFFLL;
        } else {
            qvalue = scaled << left_shift;
        }
    } else {
        int right_shift = (int)shift - exp2;
        qvalue = tnpu_round_shift_right_signed(scaled, right_shift);
    }
    if (sign) {
        qvalue = -qvalue;
    }
    return tnpu_clip_i16(qvalue);
}

static void tnpu_write_tensor_a_qf16_to_i16_fast(
    const TinyTensor *tensor,
    uint16_t base_addr,
    uint16_t word_count,
    uint16_t multiplier,
    uint8_t shift)
{
    const int rows = tensor->rank == 1 ? 1 : tensor->shape[0];
    const int cols = tensor->rank == 1 ? tensor->shape[0] : tensor->shape[1];
    const int m_tiles = (rows + TNPU_ARRAY_SIZE - 1) / TNPU_ARRAY_SIZE;
    const int k_tiles = (cols + TNPU_ARRAY_SIZE - 1) / TNPU_ARRAY_SIZE;
    const int32_t *src = tensor_i32(tensor);
    uint16_t addr = base_addr;
    uint16_t lanes[TNPU_ARRAY_SIZE];
    uint32_t chunks[TNPU_MMVR_WORDS_32];

    runtime_assert(word_count == (uint16_t)(m_tiles * k_tiles * TNPU_ARRAY_SIZE), "qf16->i16 role A word count mismatch");

    for (int mt = 0; mt < m_tiles; ++mt) {
        for (int kt = 0; kt < k_tiles; ++kt) {
            for (int lane_selector = 0; lane_selector < TNPU_ARRAY_SIZE; ++lane_selector) {
                for (int lane = 0; lane < TNPU_ARRAY_SIZE; ++lane) {
                    const int row = mt * TNPU_ARRAY_SIZE + lane;
                    const int col = kt * TNPU_ARRAY_SIZE + lane_selector;
                    int16_t value = 0;
                    if (row < rows && col < cols) {
                        uint16_t fp16_bits = (uint16_t)src[row * cols + col];
                        value = tnpu_quantize_fp16_lane_to_i16(fp16_bits, multiplier, shift);
                    }
                    lanes[lane] = (uint16_t)value;
                }
                tnpu_lanes_to_chunks_u16(lanes, chunks);
                npu_write_mem_word(addr++, chunks);
            }
        }
    }
}

static void tnpu_write_tensor_a_quantized_via_xform(
    const TinyTensor *tensor,
    uint16_t base_addr,
    int word_count,
    float inv_scale)
{
    uint16_t multiplier;
    uint8_t shift;
    runtime_assert(inv_scale > 0.0f, "xform quantized write expects positive inv_scale");
    runtime_assert(word_count <= 0xFFFF, "xform quantized write expects <= 65535 words");
    tnpu_write_tensor_a_f16bits_from_float(tensor, base_addr, word_count);
    tnpu_choose_xform_scale_params(inv_scale, &multiplier, &shift);
    tnpu_run_xform_q_f16_i16(base_addr, base_addr, (uint16_t)word_count, multiplier, shift);
}

static void tnpu_write_tensor_a_fast(
    const TinyTensor *tensor,
    uint16_t base_addr,
    int precision,
    uint16_t word_count)
{
    const int p = tnpu_precision_to_packed_count(precision);
    const int bits = 16 / p;
    const int mask = (1 << bits) - 1;
    const int rows = tensor->rank == 1 ? 1 : tensor->shape[0];
    const int cols = tensor->rank == 1 ? tensor->shape[0] : tensor->shape[1];
    const int m_tiles = (rows + TNPU_ARRAY_SIZE - 1) / TNPU_ARRAY_SIZE;
    const int k_tiles = ((cols / p) + TNPU_ARRAY_SIZE - 1) / TNPU_ARRAY_SIZE;
    const int32_t *src = tensor_i32(tensor);
    uint16_t addr = base_addr;
    uint16_t lanes[TNPU_ARRAY_SIZE];
    uint32_t chunks[TNPU_MMVR_WORDS_32];

    runtime_assert(p != 0, "unsupported precision in role A fast path");
    runtime_assert(word_count == (uint16_t)(m_tiles * k_tiles * TNPU_ARRAY_SIZE), "role A word count mismatch");

    for (int mt = 0; mt < m_tiles; ++mt) {
        for (int kt = 0; kt < k_tiles; ++kt) {
            for (int lane_selector = 0; lane_selector < TNPU_ARRAY_SIZE; ++lane_selector) {
                for (int lane = 0; lane < TNPU_ARRAY_SIZE; ++lane) {
                    const int row = mt * TNPU_ARRAY_SIZE + lane;
                    const int start_k = (kt * TNPU_ARRAY_SIZE + lane_selector) * p;
                    uint16_t subword = 0u;
                    for (int bit_idx = 0; bit_idx < p; ++bit_idx) {
                        const int col = start_k + bit_idx;
                        int32_t value = 0;
                        if (row < rows && col < cols) {
                            value = src[row * cols + col];
                        }
                        subword |= (uint16_t)(((uint32_t)value & (uint32_t)mask) << (bit_idx * bits));
                    }
                    lanes[lane] = subword;
                }
                tnpu_lanes_to_chunks_u16(lanes, chunks);
                npu_write_mem_word(addr++, chunks);
            }
        }
    }
}

static void tnpu_read_tensor_c_int16_fast(
    TinyTensor *dst,
    uint16_t addr)
{
    const int rows = dst->rank == 1 ? 1 : dst->shape[0];
    const int cols = dst->rank == 1 ? dst->shape[0] : dst->shape[1];
    const int m_tiles = (rows + TNPU_ARRAY_SIZE - 1) / TNPU_ARRAY_SIZE;
    const int n_tiles = (cols + TNPU_ARRAY_SIZE - 1) / TNPU_ARRAY_SIZE;
    int32_t *out = tensor_i32(dst);
    uint32_t chunks[TNPU_MMVR_WORDS_32];
    for (int mt = 0; mt < m_tiles; ++mt) {
        for (int nt = 0; nt < n_tiles; ++nt) {
            const uint16_t tile_addr =
                (uint16_t)(addr + (mt * n_tiles * TNPU_ARRAY_SIZE) + (nt * TNPU_ARRAY_SIZE));
            for (int row_in_tile = 0; row_in_tile < TNPU_ARRAY_SIZE; ++row_in_tile) {
                runtime_assert(
                    npu_read_mem_word((uint16_t)(tile_addr + row_in_tile), chunks, 2) == 0,
                    "readback failed");
                for (int lane = 0; lane < TNPU_ARRAY_SIZE; ++lane) {
                    const int row_idx = mt * TNPU_ARRAY_SIZE + row_in_tile;
                    const int col_idx = nt * TNPU_ARRAY_SIZE + lane;
                    const uint16_t packed_lane = (lane & 1)
                        ? (uint16_t)(chunks[lane / 2] >> 16)
                        : (uint16_t)(chunks[lane / 2] & 0xFFFFu);
                    if (row_idx >= rows || col_idx >= cols) {
                        continue;
                    }
                    out[row_idx * cols + col_idx] = (int16_t)packed_lane;
                }
            }
        }
    }
}

static void tnpu_read_tensor_c_int16_dequantize_float_fast(
    TinyTensor *dst,
    uint16_t addr,
    float scale,
    int zero_point)
{
    const int rows = dst->rank == 1 ? 1 : dst->shape[0];
    const int cols = dst->rank == 1 ? dst->shape[0] : dst->shape[1];
    const int m_tiles = (rows + TNPU_ARRAY_SIZE - 1) / TNPU_ARRAY_SIZE;
    const int n_tiles = (cols + TNPU_ARRAY_SIZE - 1) / TNPU_ARRAY_SIZE;
    float *out = tensor_f32(dst);
    uint32_t chunks[TNPU_MMVR_WORDS_32];

    runtime_assert(dst->dtype == TINY_DTYPE_FLOAT32, "dequantize read expects float32 destination");
    runtime_assert(scale > 0.0f, "dequantize read expects positive scale");

    for (int mt = 0; mt < m_tiles; ++mt) {
        for (int nt = 0; nt < n_tiles; ++nt) {
            const uint16_t tile_addr =
                (uint16_t)(addr + (mt * n_tiles * TNPU_ARRAY_SIZE) + (nt * TNPU_ARRAY_SIZE));
            for (int row_in_tile = 0; row_in_tile < TNPU_ARRAY_SIZE; ++row_in_tile) {
                runtime_assert(
                    npu_read_mem_word((uint16_t)(tile_addr + row_in_tile), chunks, 2) == 0,
                    "readback failed");
                for (int lane = 0; lane < TNPU_ARRAY_SIZE; ++lane) {
                    const int row_idx = mt * TNPU_ARRAY_SIZE + row_in_tile;
                    const int col_idx = nt * TNPU_ARRAY_SIZE + lane;
                    const uint16_t packed_lane = (lane & 1)
                        ? (uint16_t)(chunks[lane / 2] >> 16)
                        : (uint16_t)(chunks[lane / 2] & 0xFFFFu);
                    if (row_idx >= rows || col_idx >= cols) {
                        continue;
                    }
                    out[row_idx * cols + col_idx] = ((float)((int32_t)(int16_t)packed_lane - zero_point)) * scale;
                }
            }
        }
    }
}

static int tnpu_bind_inputs(
    TinyTensor *runtime_tensors,
    const TnpuProgram *program,
    const TnpuTensor *const *inputs)
{
    for (uint32_t i = 0; i < program->input_count; ++i) {
        const TnpuTensor *provided;
        uint16_t tensor_idx;
        if (inputs == NULL || inputs[i] == NULL) {
            printf("runtime v2: missing input %lu\n", (unsigned long)i);
            return 1;
        }
        provided = inputs[i];
        tensor_idx = program->input_tensor_indices[i];
        if ((uint32_t)tensor_idx >= program->tensor_count) {
            printf("runtime v2: bad input tensor index %u\n", (unsigned)tensor_idx);
            return 1;
        }
        runtime_tensors[tensor_idx].data = provided->data;
        if (provided->elem_count > 0 && runtime_tensors[tensor_idx].elem_count != provided->elem_count) {
            printf(
                "runtime v2: input elem_count mismatch (%s): expected %d got %d\n",
                runtime_tensors[tensor_idx].name,
                runtime_tensors[tensor_idx].elem_count,
                provided->elem_count);
            return 1;
        }
    }
    return 0;
}

static int tnpu_bind_outputs(
    TinyTensor *runtime_tensors,
    const TnpuProgram *program,
    const TnpuTensor *const *outputs)
{
    for (uint32_t i = 0; i < program->output_count; ++i) {
        const TnpuTensor *provided;
        uint16_t tensor_idx;
        if (outputs == NULL || outputs[i] == NULL) {
            printf("runtime v2: missing output %lu\n", (unsigned long)i);
            return 1;
        }
        provided = outputs[i];
        tensor_idx = program->output_tensor_indices[i];
        if ((uint32_t)tensor_idx >= program->tensor_count) {
            printf("runtime v2: bad output tensor index %u\n", (unsigned)tensor_idx);
            return 1;
        }
        runtime_tensors[tensor_idx].data = provided->data;
        if (provided->elem_count > 0 && runtime_tensors[tensor_idx].elem_count != provided->elem_count) {
            printf(
                "runtime v2: output elem_count mismatch (%s): expected %d got %d\n",
                runtime_tensors[tensor_idx].name,
                runtime_tensors[tensor_idx].elem_count,
                provided->elem_count);
            return 1;
        }
    }
    return 0;
}

static int tnpu_execute_host_op(TinyTensor *runtime_tensors, const TnpuHostOp *op)
{
    TinyTensor *out;
    const TinyTensor *in;
    const TinyTensor *in1 = NULL;
    if (op->input_idx >= 0xFFFFu || op->output_idx >= 0xFFFFu) {
        printf("runtime v2: invalid host tensor indices\n");
        return 1;
    }
    out = &runtime_tensors[op->output_idx];
    in = &runtime_tensors[op->input_idx];
    if (op->input1_idx < 0xFFFFu) {
        in1 = &runtime_tensors[op->input1_idx];
    }

    switch (op->kind) {
        case TNPU_HOST_ALIAS:
            host_alias(out, in);
            return 0;
        case TNPU_HOST_RELU:
            host_relu(out, in);
            return 0;
        case TNPU_HOST_SIGMOID:
            host_sigmoid(out, in);
            return 0;
        case TNPU_HOST_GELU:
            host_gelu(out, in);
            return 0;
        case TNPU_HOST_QUANTIZE:
            if (op->attrs_i32[1] != 0) {
                host_quantize_fp16bits(out, in, op->attrs_f32[0], op->attrs_i32[0]);
            } else {
                host_quantize(out, in, op->attrs_f32[0], op->attrs_i32[0]);
            }
            return 0;
        case TNPU_HOST_DEQUANTIZE:
            host_dequantize(out, in, op->attrs_f32[0], op->attrs_i32[0]);
            return 0;
        case TNPU_HOST_REQUANTIZE:
            host_requantize(out, in, op->attrs_f32[0], op->attrs_i32[0]);
            return 0;
        case TNPU_HOST_RESHAPE:
            host_reshape(out, in);
            return 0;
        case TNPU_HOST_SLICE_ROW:
            host_slice_row(out, in, op->attrs_i32[0]);
            return 0;
        case TNPU_HOST_TRANSPOSE:
            host_transpose(out, in, op->arr0, (int)op->arr0_len);
            return 0;
        case TNPU_HOST_SOFTMAX:
            host_softmax(out, in, op->attrs_i32[0]);
            return 0;
        case TNPU_HOST_SOFTMAX_F16:
            host_softmax_f16(out, in, op->attrs_i32[0]);
            return 0;
        case TNPU_HOST_MEAN:
            host_mean(
                out,
                in,
                op->arr0,
                (int)op->arr0_len,
                op->attrs_i32[0],
                op->attrs_i32[1],
                op->attrs_f32[0],
                op->attrs_i32[2]);
            return 0;
        case TNPU_HOST_IM2COL:
            if (op->attrs_i32[3] == 2) {
                host_im2col_matrix(
                    out,
                    in,
                    op->attrs_i32[4],
                    op->attrs_i32[5],
                    op->attrs_i32[6],
                    op->attrs_i32[0],
                    op->attrs_i32[1],
                    op->attrs_i32[2]);
            } else {
                host_im2col(out, in, op->attrs_i32[0], op->attrs_i32[1], op->attrs_i32[2], op->attrs_i32[3]);
            }
            return 0;
        case TNPU_HOST_LAYOUT_RESTORE:
            host_layout_restore(
                out,
                in,
                op->attrs_i32[0],
                op->attrs_i32[1],
                op->attrs_i32[2],
                op->attrs_i32[3],
                op->attrs_i32[4]);
            return 0;
        case TNPU_HOST_RMSNORM:
            if (in1 == NULL) {
                printf("runtime v2: rmsnorm missing weight input\n");
                return 1;
            }
            host_rmsnorm(out, in, in1, op->attrs_f32[0]);
            return 0;
        case TNPU_HOST_LAYERNORM:
            if (in1 == NULL) {
                printf("runtime v2: layernorm missing weight/bias input\n");
                return 1;
            }
            host_layernorm(out, in, in1, op->attrs_f32[0]);
            return 0;
        case TNPU_HOST_ROPE:
            if (op->arr0 != NULL && op->arr0_len > 0u) {
                host_rope_precomputed(
                    out,
                    in,
                    op->attrs_i32[0],
                    op->attrs_i32[1],
                    op->arr0,
                    (int)op->arr0_len);
            } else {
                host_rope(out, in, op->attrs_i32[0], op->attrs_i32[1], op->attrs_f32[0]);
            }
            return 0;
        case TNPU_HOST_SILU:
            host_silu(out, in);
            return 0;
        case TNPU_HOST_MUL:
            if (in1 == NULL) {
                printf("runtime v2: mul missing rhs input\n");
                return 1;
            }
            host_mul(out, in, in1);
            return 0;
        case TNPU_HOST_ADD:
            if (in1 == NULL) {
                printf("runtime v2: add missing rhs input\n");
                return 1;
            }
            host_add(out, in, in1);
            return 0;
        case TNPU_HOST_K_CACHE_SCATTER_WRITE:
            host_k_cache_scatter_write(in, (const int *)op->arr0, op->attrs_i32[0]);
            return 0;
        case TNPU_HOST_V_CACHE_SCATTER_WRITE:
            host_v_cache_scatter_write(in, (const int *)op->arr0, (int)op->arr0_len);
            return 0;
        case TNPU_HOST_K_CACHE_SCATTER_MATRIX:
            host_k_cache_scatter_matrix(out, in, op->attrs_i32[0]);
            return 0;
        case TNPU_HOST_V_CACHE_SCATTER_MATRIX:
            host_v_cache_scatter_matrix(out, in, op->attrs_i32[0]);
            return 0;
        case TNPU_HOST_CAUSAL_MASK:
            host_causal_mask(out, in, op->attrs_i32[0], op->attrs_f32[0]);
            return 0;
        case TNPU_HOST_CONCAT_LASTDIM2:
            if (in1 == NULL) {
                printf("runtime v2: concat_lastdim2 missing rhs input\n");
                return 1;
            }
            host_concat_lastdim2(out, in, in1);
            return 0;
        default:
            printf("runtime v2: unsupported host op kind=%u (%s)\n", (unsigned)op->kind, op->name ? op->name : "?");
            return 1;
    }
}

static int tnpu_execute_segment(TinyTensor *runtime_tensors, const TnpuSegment *segment, int verbose)
{
    uint32_t cycle_t0;
    uint32_t cycle_t1;
    uint32_t cycle_segment_t0;
    char label[96];

    if (verbose) {
        printf("NpuSegment: %s\n", segment->name ? segment->name : "segment");
    }

    cycle_segment_t0 = read_mcycle32();
    cycle_t0 = read_mcycle32();
    for (uint32_t i = 0; i < segment->write_count; ++i) {
        const TnpuTensorWrite *write = &segment->writes[i];
        TinyTensor *tensor = &runtime_tensors[write->tensor_idx];
        char role = tnpu_role_code(write->role, 'A');
        if (write->transform == TNPU_WRITE_QUANTIZE_F32_TO_INT16) {
            if (role != 'A' || (int)write->precision != 2) {
                printf("runtime v2: quantized write only supports role A INT16\n");
                return 1;
            }
            /* Keep TNPU_WRITE_QUANTIZE_F32_TO_INT16 numerically exact.
             * The XFORM fast path is available via TNPU_WRITE_XFORM_Q_F16_I16.
             */
            write_tensor_to_npu_quantized_a_int16_from_float(
                tensor,
                write->addr,
                (int)write->word_count,
                write->attrs_f32[0],
                write->attrs_i32[0]);
        } else if (write->transform == TNPU_WRITE_XFORM_Q_F16_I16) {
            uint16_t multiplier;
            uint8_t shift;
            if (role != 'A' || (int)write->precision != 2) {
                printf("runtime v2: xform write only supports role A INT16\n");
                return 1;
            }
            if (write->attrs_i32[0] != 0) {
                printf("runtime v2: xform write currently requires zero_point=0\n");
                return 1;
            }
            tnpu_choose_xform_scale_params(write->attrs_f32[0], &multiplier, &shift);
            tnpu_write_tensor_a_qf16_to_i16_fast(tensor, write->addr, write->word_count, multiplier, shift);
        } else if (role == 'A' && (int)write->precision == 2) {
            tnpu_write_tensor_a_fast(tensor, write->addr, (int)write->precision, write->word_count);
        } else {
            write_tensor_to_npu(
                tensor,
                write->addr,
                tnpu_role_or_default(write->role, "A"),
                (int)write->precision,
                (int)write->word_count);
        }
    }
    cycle_t1 = read_mcycle32();
    if (verbose) {
        snprintf(label, sizeof(label), "segment.%s.stage", segment->name ? segment->name : "segment");
        print_cycle_delta32(label, cycle_t0, cycle_t1);
    }

    cycle_t0 = read_mcycle32();
    if (npu_run((uint32_t)segment->im_start_addr) != 0) {
        return 1;
    }
    cycle_t1 = read_mcycle32();
    if (verbose) {
        snprintf(label, sizeof(label), "segment.%s.run", segment->name ? segment->name : "segment");
        print_cycle_delta32(label, cycle_t0, cycle_t1);
    }

    cycle_t0 = read_mcycle32();
    for (uint32_t i = 0; i < segment->read_count; ++i) {
        const TnpuTensorRead *read = &segment->reads[i];
        TinyTensor *tensor = &runtime_tensors[read->tensor_idx];
        char role = tnpu_role_code(read->role, 'C');
        if (read->transform == TNPU_READ_DEQUANTIZE_INT16_TO_FLOAT32) {
            if (role != 'C' || (int)read->precision != 2) {
                printf("runtime v2: dequantize read only supports role C INT16 source\n");
                return 1;
            }
            tnpu_read_tensor_c_int16_dequantize_float_fast(tensor, read->addr, read->attrs_f32[0], read->attrs_i32[0]);
        } else if (role == 'C' && (int)read->precision == 2) {
            tnpu_read_tensor_c_int16_fast(tensor, read->addr);
        } else {
            read_tensor_from_npu(
                tensor,
                read->addr,
                tnpu_role_or_default(read->role, "C"),
                (int)read->precision);
        }
    }
    cycle_t1 = read_mcycle32();
    if (verbose) {
        snprintf(label, sizeof(label), "segment.%s.readback", segment->name ? segment->name : "segment");
        print_cycle_delta32(label, cycle_t0, cycle_t1);
        snprintf(label, sizeof(label), "segment.%s.npu", segment->name ? segment->name : "segment");
        print_cycle_delta32(label, cycle_segment_t0, cycle_t1);
    }

    return 0;
}

static int tnpu_execute_verify(TinyTensor *runtime_tensors, const TnpuVerifyOp *verify)
{
    const TinyTensor *actual = &runtime_tensors[verify->actual_tensor_idx];
    const TinyTensor *expected = &runtime_tensors[verify->expected_tensor_idx];
    const float float_atol = verify->float_atol > 0.0f ? verify->float_atol : 1.0e-3f;
    int matches = 1;
    if (actual->dtype == TINY_DTYPE_FLOAT32 && expected->dtype == TINY_DTYPE_FLOAT32) {
        if (actual->elem_count != expected->elem_count) {
            matches = 0;
        } else {
            for (int i = 0; i < actual->elem_count; ++i) {
                if (fabsf(tensor_get_float(actual, i) - tensor_get_float(expected, i)) > float_atol) {
                    matches = 0;
                    break;
                }
            }
        }
    } else {
        matches = tensor_matches_expected(actual, expected);
    }
    if (!matches) {
        printf(
            "verification failed: %s (%s)\n",
            verify->label ? verify->label : actual->name,
            actual->name);
        printf(
            "meta actual dtype=%d elems=%d expected dtype=%d elems=%d\n",
            actual->dtype,
            actual->elem_count,
            expected->dtype,
            expected->elem_count);
        if (actual->dtype == TINY_DTYPE_FLOAT32 && expected->dtype == TINY_DTYPE_FLOAT32) {
            printf("float_atol=");
            print_float_scalar(float_atol);
            printf("\n");
        }
        print_tensor(actual);
        print_tensor(expected);
        return 1;
    }
    return 0;
}

static int tnpu_find_tensor_by_name(const TnpuProgram *program, const char *name)
{
    if (name == NULL || name[0] == '\0') {
        return -1;
    }
    for (uint32_t i = 0; i < program->tensor_count; ++i) {
        const char *candidate = program->tensors[i].name;
        if (candidate != NULL && strcmp(candidate, name) == 0) {
            return (int)i;
        }
    }
    return -1;
}

static int tnpu_autoverify_outputs(TinyTensor *runtime_tensors, const TnpuProgram *program)
{
    for (uint32_t i = 0; i < program->output_count; ++i) {
        char expected_name[128];
        uint16_t out_idx = program->output_tensor_indices[i];
        const TinyTensor *actual;
        const TinyTensor *expected;
        int expected_idx;

        if ((uint32_t)out_idx >= program->tensor_count) {
            continue;
        }
        actual = &runtime_tensors[out_idx];
        if (actual->name == NULL || actual->name[0] == '\0') {
            continue;
        }
        (void)snprintf(expected_name, sizeof(expected_name), "%s_expected", actual->name);
        expected_idx = tnpu_find_tensor_by_name(program, expected_name);
        if (expected_idx < 0) {
            continue;
        }
        expected = &runtime_tensors[expected_idx];
        if (!tensor_matches_expected(actual, expected)) {
            int first_mismatch = -1;
            printf("autoverify failed: %s\n", actual->name);
            printf(
                "meta actual dtype=%d elems=%d expected dtype=%d elems=%d\n",
                actual->dtype,
                actual->elem_count,
                expected->dtype,
                expected->elem_count);
            if (actual->dtype == TINY_DTYPE_FLOAT32) {
                for (int linear = 0; linear < actual->elem_count; ++linear) {
                    float a = tensor_get_float(actual, linear);
                    float b = tensor_get_float(expected, linear);
                    if (fabsf(a - b) > 1e-5f) {
                        first_mismatch = linear;
                        printf("first mismatch @%d: actual=", linear);
                        print_float_scalar(a);
                        printf(" expected=");
                        print_float_scalar(b);
                        printf("\n");
                        break;
                    }
                }
            } else {
                for (int linear = 0; linear < actual->elem_count; ++linear) {
                    int32_t a = tensor_get_i32(actual, linear);
                    int32_t b = tensor_get_i32(expected, linear);
                    if (a != b) {
                        first_mismatch = linear;
                        printf("first mismatch @%d: actual=%ld expected=%ld\n", linear, (long)a, (long)b);
                        break;
                    }
                }
            }
            if (first_mismatch < 0) {
                printf("autoverify mismatch reported, but no differing element found\n");
            }
            return 1;
        }
        printf("autoverify ok: %s\n", actual->name);
    }
    return 0;
}

static TinyTensor *tnpu_prepare_runtime_tensors(
    const TnpuProgram *program,
    const TnpuTensor *const *inputs,
    const TnpuTensor *const *outputs)
{
    TinyTensor *runtime_tensors;

    if (program == NULL || program->tensors == NULL) {
        printf("runtime v2: null program\n");
        return NULL;
    }

    runtime_tensors = (TinyTensor *)calloc(program->tensor_count, sizeof(TinyTensor));
    if (runtime_tensors == NULL) {
        printf("runtime v2: out of memory\n");
        return NULL;
    }

    for (uint32_t i = 0; i < program->tensor_count; ++i) {
        const TnpuTensorDesc *src = &program->tensors[i];
        TinyTensor *dst = &runtime_tensors[i];
        dst->name = src->name;
        dst->data = src->data;
        dst->dtype = (TinyDType)src->dtype;
        dst->rank = (int)src->rank;
        dst->shape[0] = src->shape[0];
        dst->shape[1] = src->shape[1];
        dst->shape[2] = src->shape[2];
        dst->shape[3] = src->shape[3];
        dst->elem_count = src->elem_count;
    }

    if (tnpu_bind_inputs(runtime_tensors, program, inputs) != 0) {
        free(runtime_tensors);
        return NULL;
    }
    if (tnpu_bind_outputs(runtime_tensors, program, outputs) != 0) {
        free(runtime_tensors);
        return NULL;
    }

    return runtime_tensors;
}

static int tnpu_execute_preloads(const TnpuProgram *program, uint32_t *preload_total_out)
{
    uint32_t preload_total = 0;

    for (uint32_t op_idx = 0; op_idx < program->op_count; ++op_idx) {
        const TnpuOp *op = &program->ops[op_idx];
        if (op->kind == TNPU_OP_PRELOAD_UB) {
            const TnpuImageLoad *load = &program->ub_preloads[op->index];
            uint32_t t0 = read_mcycle32();
            uint32_t t1;
            load_ub_image(load->base_addr, load->image, (int)load->word_count);
            t1 = read_mcycle32();
            preload_total += (t0 - t1);
            print_cycle_delta32(load->label ? load->label : "preload.ub_image", t0, t1);
        } else if (op->kind == TNPU_OP_PRELOAD_IM) {
            const TnpuImageLoad *load = &program->im_preloads[op->index];
            uint32_t t0 = read_mcycle32();
            uint32_t t1;
            load_im_image(load->base_addr, load->image, (int)load->word_count);
            t1 = read_mcycle32();
            preload_total += (t0 - t1);
            print_cycle_delta32(load->label ? load->label : "preload.im_image", t0, t1);
        }
    }

    if (preload_total_out != NULL) {
        *preload_total_out = preload_total;
    }
    return 0;
}

static int tnpu_execute_body(TinyTensor *runtime_tensors, const TnpuProgram *program, int verbose_steps)
{
    for (uint32_t op_idx = 0; op_idx < program->op_count; ++op_idx) {
        const TnpuOp *op = &program->ops[op_idx];
        if (op->kind == TNPU_OP_PRELOAD_UB || op->kind == TNPU_OP_PRELOAD_IM || op->kind == TNPU_OP_VERIFY) {
            continue;
        }
        if (op->kind == TNPU_OP_HOST) {
            const TnpuHostOp *host_op = &program->host_ops[op->index];
            uint32_t t0 = read_mcycle32();
            uint32_t t1;
            char label[96];
            if (verbose_steps) {
                printf("HostOp: %s\n", host_op->name ? host_op->name : "host");
            }
            if (tnpu_execute_host_op(runtime_tensors, host_op) != 0) {
                return 1;
            }
            t1 = read_mcycle32();
            if (verbose_steps) {
                snprintf(label, sizeof(label), "hostop.%s", host_op->name ? host_op->name : "host");
                print_cycle_delta32(label, t0, t1);
            }
        } else if (op->kind == TNPU_OP_SEGMENT) {
            if (tnpu_execute_segment(runtime_tensors, &program->segments[op->index], verbose_steps) != 0) {
                return 1;
            }
        } else {
            printf("runtime v2: unsupported op kind=%u\n", (unsigned)op->kind);
            return 1;
        }
    }
    return 0;
}

static int tnpu_execute_verifies(TinyTensor *runtime_tensors, const TnpuProgram *program)
{
    for (uint32_t op_idx = 0; op_idx < program->op_count; ++op_idx) {
        const TnpuOp *op = &program->ops[op_idx];
        if (op->kind == TNPU_OP_VERIFY) {
            if (tnpu_execute_verify(runtime_tensors, &program->verify_ops[op->index]) != 0) {
                return 1;
            }
        }
    }

    if (program->verify_op_count == 0u && tnpu_autoverify_outputs(runtime_tensors, program) != 0) {
        return 1;
    }

    return 0;
}

static void tnpu_dump_final_outputs(TinyTensor *runtime_tensors, const TnpuProgram *program)
{
#if TNPU_RUNTIME_V2_DUMP_FINAL_OUTPUTS
    if (program->output_count > 0) {
        printf("Final outputs:\n");
        for (uint32_t i = 0; i < program->output_count; ++i) {
            uint16_t idx = program->output_tensor_indices[i];
            print_tensor(&runtime_tensors[idx]);
        }
    }
#else
    (void)runtime_tensors;
    (void)program;
#endif
}

int tinynpu_run(
    const TnpuProgram *program,
    const TnpuTensor *const *inputs,
    const TnpuTensor *const *outputs,
    void *scratch,
    uint32_t scratch_words)
{
    TinyTensor *runtime_tensors;
    (void)scratch;
    (void)scratch_words;
    runtime_tensors = tnpu_prepare_runtime_tensors(program, inputs, outputs);
    if (runtime_tensors == NULL) {
        return EXIT_FAILURE;
    }

    printf("TinyNPU runtime v2 program: %s\n", program->name ? program->name : "program_v2");
    tb_timer_reset_counter();

    if (tnpu_execute_preloads(program, NULL) != 0) {
        free(runtime_tensors);
        return EXIT_FAILURE;
    }
    if (tnpu_execute_body(runtime_tensors, program, 1) != 0) {
        free(runtime_tensors);
        return EXIT_FAILURE;
    }
    if (tnpu_execute_verifies(runtime_tensors, program) != 0) {
        free(runtime_tensors);
        return EXIT_FAILURE;
    }
    tnpu_dump_final_outputs(runtime_tensors, program);

    free(runtime_tensors);
    return EXIT_SUCCESS;
}

int tinynpu_run_repeat(
    const TnpuProgram *program,
    const TnpuTensor *const *inputs,
    const TnpuTensor *const *outputs,
    void *scratch,
    uint32_t scratch_words,
    uint32_t repeat_count)
{
    TinyTensor *runtime_tensors;
    uint32_t preload_total = 0;
    uint32_t cold_npu = 0;
    uint32_t warm_sum = 0;
    uint32_t warm_avg = 0;
    uint32_t e2e_npu_10 = 0;
    (void)scratch;
    (void)scratch_words;

    if (repeat_count == 0u) {
        printf("runtime v2: repeat_count must be > 0\n");
        return EXIT_FAILURE;
    }

    runtime_tensors = tnpu_prepare_runtime_tensors(program, inputs, outputs);
    if (runtime_tensors == NULL) {
        return EXIT_FAILURE;
    }

    printf("TinyNPU runtime v2 program: %s\n", program->name ? program->name : "program_v2");
    tb_timer_reset_counter();

    if (tnpu_execute_preloads(program, &preload_total) != 0) {
        free(runtime_tensors);
        return EXIT_FAILURE;
    }
    printf("preload.total cycles=%lu\n", (unsigned long)preload_total);

    for (uint32_t iter = 0; iter < repeat_count; ++iter) {
        uint32_t t0;
        uint32_t t1;
        uint32_t delta;

        printf("repeat.iter=%lu\n", (unsigned long)(iter + 1u));
        fflush(stdout);
        t0 = read_mcycle32();
        if (tnpu_execute_body(runtime_tensors, program, 0) != 0) {
            free(runtime_tensors);
            return EXIT_FAILURE;
        }
        t1 = read_mcycle32();
        delta = (t0 - t1);
        if (tnpu_execute_verifies(runtime_tensors, program) != 0) {
            free(runtime_tensors);
            return EXIT_FAILURE;
        }

        if (iter == 0u) {
            cold_npu = delta;
            printf("cold.npu cycles=%lu\n", (unsigned long)cold_npu);
            printf("cold.e2e.npu cycles=%lu\n", (unsigned long)(preload_total + cold_npu));
        } else {
            warm_sum += delta;
            printf("warm%lu.npu cycles=%lu\n", (unsigned long)iter, (unsigned long)delta);
        }
    }

    if (repeat_count > 1u) {
        warm_avg = (warm_sum + ((repeat_count - 1u) / 2u)) / (repeat_count - 1u);
        printf("warm.avg.npu cycles=%lu\n", (unsigned long)warm_avg);
    }
    e2e_npu_10 = preload_total + cold_npu + (9u * (repeat_count > 1u ? warm_avg : cold_npu));
    printf("extrapolated.10x.e2e.npu cycles=%lu\n", (unsigned long)e2e_npu_10);

    tnpu_dump_final_outputs(runtime_tensors, program);
    free(runtime_tensors);
    return EXIT_SUCCESS;
}
