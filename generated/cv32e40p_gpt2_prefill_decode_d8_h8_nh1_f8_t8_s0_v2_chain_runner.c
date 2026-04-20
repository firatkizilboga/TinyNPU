#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tinynpu_runtime_v2.h"

extern const TnpuProgram cv32e40p_gpt2_prefill_d8_h8_nh1_f8_t8_s0_v2;
extern const TnpuProgram cv32e40p_gpt2_decode_d8_h8_nh1_f8_t8_s0_v2;

static int run_program(const TnpuProgram *program) {
    TnpuTensor outs[8];
    const TnpuTensor *op[8];
    if (program->output_count > 8u) {
        return EXIT_FAILURE;
    }
    for (uint32_t i = 0; i < program->output_count; ++i) {
        uint16_t t = program->output_tensor_indices[i];
        outs[i].data = program->tensors[t].data;
        outs[i].desc = &program->tensors[t];
        outs[i].elem_count = program->tensors[t].elem_count;
        op[i] = &outs[i];
    }
    return tinynpu_run(program, NULL, op, NULL, 0u);
}

static TnpuTensorDesc *find_tensor_mut(const TnpuProgram *program, const char *name) {
    for (uint32_t i = 0; i < program->tensor_count; ++i) {
        if (strcmp(program->tensors[i].name, name) == 0) {
            return (TnpuTensorDesc *)&program->tensors[i];
        }
    }
    return NULL;
}

static int copy_cache_prefix(
    const TnpuProgram *src_prog,
    const char *src_name,
    const TnpuProgram *dst_prog,
    const char *dst_name,
    int elem_count)
{
    TnpuTensorDesc *src = find_tensor_mut(src_prog, src_name);
    TnpuTensorDesc *dst = find_tensor_mut(dst_prog, dst_name);
    if (src == NULL || dst == NULL) {
        printf("chain runner missing tensor: %s -> %s\n", src_name, dst_name);
        return EXIT_FAILURE;
    }
    if (src->dtype != dst->dtype) {
        printf("chain runner dtype mismatch: %s -> %s\n", src_name, dst_name);
        return EXIT_FAILURE;
    }
    memcpy(dst->data, src->data, (size_t)elem_count * sizeof(int32_t));
    return EXIT_SUCCESS;
}

int main(void) {
    int rc = run_program(&cv32e40p_gpt2_prefill_d8_h8_nh1_f8_t8_s0_v2);
    if (rc != EXIT_SUCCESS) {
        printf("prefill failed in chain runner\n");
        return rc;
    }

    rc = copy_cache_prefix(
        &cv32e40p_gpt2_prefill_d8_h8_nh1_f8_t8_s0_v2,
        "prefill_k_cache_h0",
        &cv32e40p_gpt2_decode_d8_h8_nh1_f8_t8_s0_v2,
        "k_cache_h0",
        64);
    if (rc != EXIT_SUCCESS) {
        return rc;
    }

    rc = copy_cache_prefix(
        &cv32e40p_gpt2_prefill_d8_h8_nh1_f8_t8_s0_v2,
        "prefill_v_cache_h0",
        &cv32e40p_gpt2_decode_d8_h8_nh1_f8_t8_s0_v2,
        "v_cache_h0",
        64);
    if (rc != EXIT_SUCCESS) {
        return rc;
    }

    rc = run_program(&cv32e40p_gpt2_decode_d8_h8_nh1_f8_t8_s0_v2);
    if (rc != EXIT_SUCCESS) {
        printf("decode failed in chain runner\n");
        return rc;
    }
    return EXIT_SUCCESS;
}
