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
    if (op->input_idx >= 0xFFFFu || op->output_idx >= 0xFFFFu) {
        printf("runtime v2: invalid host tensor indices\n");
        return 1;
    }
    out = &runtime_tensors[op->output_idx];
    in = &runtime_tensors[op->input_idx];

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
            host_quantize(out, in, op->attrs_f32[0], op->attrs_i32[0]);
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
        case TNPU_HOST_TRANSPOSE:
            host_transpose(out, in, op->arr0, (int)op->arr0_len);
            return 0;
        case TNPU_HOST_SOFTMAX:
            host_softmax(out, in, op->attrs_i32[0]);
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
        default:
            printf("runtime v2: unsupported host op kind=%u (%s)\n", (unsigned)op->kind, op->name ? op->name : "?");
            return 1;
    }
}

static int tnpu_execute_segment(TinyTensor *runtime_tensors, const TnpuSegment *segment)
{
    uint32_t cycle_t0;
    uint32_t cycle_t1;
    uint32_t cycle_segment_t0;
    char label[96];

    printf("NpuSegment: %s\n", segment->name ? segment->name : "segment");

    cycle_segment_t0 = read_mcycle32();
    cycle_t0 = read_mcycle32();
    for (uint32_t i = 0; i < segment->write_count; ++i) {
        const TnpuTensorWrite *write = &segment->writes[i];
        TinyTensor *tensor = &runtime_tensors[write->tensor_idx];
        char role = tnpu_role_code(write->role, 'A');
        if (role == 'A' && (int)write->precision == 2) {
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
    snprintf(label, sizeof(label), "segment.%s.stage", segment->name ? segment->name : "segment");
    print_cycle_delta32(label, cycle_t0, cycle_t1);

    cycle_t0 = read_mcycle32();
    if (npu_run((uint32_t)segment->im_start_addr) != 0) {
        return 1;
    }
    cycle_t1 = read_mcycle32();
    snprintf(label, sizeof(label), "segment.%s.run", segment->name ? segment->name : "segment");
    print_cycle_delta32(label, cycle_t0, cycle_t1);

    cycle_t0 = read_mcycle32();
    for (uint32_t i = 0; i < segment->read_count; ++i) {
        const TnpuTensorRead *read = &segment->reads[i];
        TinyTensor *tensor = &runtime_tensors[read->tensor_idx];
        char role = tnpu_role_code(read->role, 'C');
        if (role == 'C' && (int)read->precision == 2) {
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
    snprintf(label, sizeof(label), "segment.%s.readback", segment->name ? segment->name : "segment");
    print_cycle_delta32(label, cycle_t0, cycle_t1);
    snprintf(label, sizeof(label), "segment.%s.npu", segment->name ? segment->name : "segment");
    print_cycle_delta32(label, cycle_segment_t0, cycle_t1);

    return 0;
}

static int tnpu_execute_verify(TinyTensor *runtime_tensors, const TnpuVerifyOp *verify)
{
    const TinyTensor *actual = &runtime_tensors[verify->actual_tensor_idx];
    const TinyTensor *expected = &runtime_tensors[verify->expected_tensor_idx];
    if (!tensor_matches_expected(actual, expected)) {
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

    if (program == NULL || program->tensors == NULL) {
        printf("runtime v2: null program\n");
        return EXIT_FAILURE;
    }

    runtime_tensors = (TinyTensor *)calloc(program->tensor_count, sizeof(TinyTensor));
    if (runtime_tensors == NULL) {
        printf("runtime v2: out of memory\n");
        return EXIT_FAILURE;
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
        return EXIT_FAILURE;
    }
    if (tnpu_bind_outputs(runtime_tensors, program, outputs) != 0) {
        free(runtime_tensors);
        return EXIT_FAILURE;
    }

    tinynpu_set_force_mmio(0);

    printf("TinyNPU runtime v2 program: %s\n", program->name ? program->name : "program_v2");
    tb_timer_reset_counter();

    for (uint32_t op_idx = 0; op_idx < program->op_count; ++op_idx) {
        const TnpuOp *op = &program->ops[op_idx];
        if (op->kind == TNPU_OP_PRELOAD_UB) {
            const TnpuImageLoad *load = &program->ub_preloads[op->index];
            uint32_t t0 = read_mcycle32();
            uint32_t t1;
            load_ub_image(load->base_addr, load->image, (int)load->word_count);
            t1 = read_mcycle32();
            print_cycle_delta32(load->label ? load->label : "preload.ub_image", t0, t1);
        } else if (op->kind == TNPU_OP_PRELOAD_IM) {
            const TnpuImageLoad *load = &program->im_preloads[op->index];
            uint32_t t0 = read_mcycle32();
            uint32_t t1;
            load_im_image(load->base_addr, load->image, (int)load->word_count);
            t1 = read_mcycle32();
            print_cycle_delta32(load->label ? load->label : "preload.im_image", t0, t1);
        } else if (op->kind == TNPU_OP_HOST) {
            const TnpuHostOp *host_op = &program->host_ops[op->index];
            uint32_t t0 = read_mcycle32();
            uint32_t t1;
            char label[96];
            printf("HostOp: %s\n", host_op->name ? host_op->name : "host");
            if (tnpu_execute_host_op(runtime_tensors, host_op) != 0) {
                free(runtime_tensors);
                return EXIT_FAILURE;
            }
            t1 = read_mcycle32();
            snprintf(label, sizeof(label), "hostop.%s", host_op->name ? host_op->name : "host");
            print_cycle_delta32(label, t0, t1);
        } else if (op->kind == TNPU_OP_SEGMENT) {
            if (tnpu_execute_segment(runtime_tensors, &program->segments[op->index]) != 0) {
                free(runtime_tensors);
                return EXIT_FAILURE;
            }
        } else if (op->kind == TNPU_OP_VERIFY) {
            if (tnpu_execute_verify(runtime_tensors, &program->verify_ops[op->index]) != 0) {
                free(runtime_tensors);
                return EXIT_FAILURE;
            }
        } else {
            printf("runtime v2: unsupported op kind=%u\n", (unsigned)op->kind);
            free(runtime_tensors);
            return EXIT_FAILURE;
        }
    }

    if (program->verify_op_count == 0u && tnpu_autoverify_outputs(runtime_tensors, program) != 0) {
        free(runtime_tensors);
        return EXIT_FAILURE;
    }

#if TNPU_RUNTIME_V2_DUMP_FINAL_OUTPUTS
    if (program->output_count > 0) {
        printf("Final outputs:\n");
        for (uint32_t i = 0; i < program->output_count; ++i) {
            uint16_t idx = program->output_tensor_indices[i];
            print_tensor(&runtime_tensors[idx]);
        }
    }
#endif

    free(runtime_tensors);
    return EXIT_SUCCESS;
}
