#include <stddef.h>
#include <stdint.h>
#include "tinynpu_runtime_v2.h"

static int32_t xmat_data[140] __attribute__((section(".data"))) = {
    -8, -8, -8, -8, -8, -8, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7,
    7, 7, 7, 7, 7, -8, -8, -8, -8, -8, -8, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4,
    5, 6, 7, 7, 7, 7, 7, 7, 7, 7, -8, -8, -8, -8, -8, -8, -8, -7, -6, -5, -4, -3, -2, -1,
    0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, -8, -8, -8, -8, -8, -8, -8, -7, -6,
    -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 7, 7, 7, 7, -8, -8, -8, -8,
    -8, -8, -8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 7, 7
};

static int32_t cols_data[378] __attribute__((section(".noinit")));

static int32_t w_data[315] __attribute__((section(".data"))) = {
    -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7
};

static int32_t y_data[30] __attribute__((section(".noinit")));

static int32_t y_expected_data[30] __attribute__((section(".data"))) = {
    5, -8, -8, -8, -8, 7, 7, 7, 7, 7, -8, -8, -8, -8, -8,
    7, 7, 7, 7, 7, -8, -8, -8, -8, -8, 7, 7, 7, 7, 7
};

static const uint16_t cv32e40p_int4_packed_convstream_default_v2_input_indices[1] = {0};

static const uint16_t cv32e40p_int4_packed_convstream_default_v2_output_indices[1] = {3};

static uint32_t cv32e40p_int4_packed_convstream_default_v2_ub_image[16][TNPU_MMVR_WORDS_32] __attribute__((section(".data"))) = {
    {0xa4fa93e9u, 0xc61cb50bu, 0x0000d72du, 0x00000000u},
    {0xfa4fe93eu, 0x1c610b50u, 0x00002d72u, 0x00000000u},
    {0x4fa43e93u, 0x61c650b5u, 0x000072d7u, 0x00000000u},
    {0xa4fa93e9u, 0xc61cb50bu, 0x0000d72du, 0x00000000u},
    {0xfa4fe93eu, 0x1c610b50u, 0x00002d72u, 0x00000000u},
    {0x4fa43e93u, 0x61c650b5u, 0x000072d7u, 0x00000000u},
    {0xa4fa93e9u, 0xc61cb50bu, 0x0000d72du, 0x00000000u},
    {0xfa4fe93eu, 0x1c610b50u, 0x00002d72u, 0x00000000u},
    {0x4fa43e93u, 0x61c650b5u, 0x000072d7u, 0x00000000u},
    {0xa4fa93e9u, 0xc61cb50bu, 0x0000d72du, 0x00000000u},
    {0xfa4fe93eu, 0x1c610b50u, 0x00002d72u, 0x00000000u},
    {0x4fa43e93u, 0x61c650b5u, 0x000072d7u, 0x00000000u},
    {0xa4fa93e9u, 0xc61cb50bu, 0x0000d72du, 0x00000000u},
    {0xfa4fe93eu, 0x1c610b50u, 0x00002d72u, 0x00000000u},
    {0x4fa43e93u, 0x61c650b5u, 0x000072d7u, 0x00000000u},
    {0x04fa03e9u, 0x061c050bu, 0x0000072du, 0x00000000u}
};

static const TnpuImageLoad cv32e40p_int4_packed_convstream_default_v2_ub_preloads[1] = {
    {.label = "preload.ub_image", .base_addr = 0u, .image = cv32e40p_int4_packed_convstream_default_v2_ub_image, .word_count = 16}
};

static uint32_t cv32e40p_int4_packed_convstream_default_v2_im_seg_conv[4][TNPU_MMVR_WORDS_32] __attribute__((section(".data"))) = {
    {0x80020007u, 0x81008002u, 0x00001c81u, 0xff000001u},
    {0x020001ffu, 0x00000100u, 0x00002800u, 0x20001000u},
    {0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
    {0x00000000u, 0x00000000u, 0x00000000u, 0x10000000u}
};

static const TnpuImageLoad cv32e40p_int4_packed_convstream_default_v2_im_preloads[1] = {
    {.label = "preload.im_seg_conv", .base_addr = 0x8000u, .image = cv32e40p_int4_packed_convstream_default_v2_im_seg_conv, .word_count = 4}
};

static const TnpuTensorWrite cv32e40p_int4_packed_convstream_default_v2_seg_seg_conv_writes[1] = {
    {.tensor_idx = 0, .addr = 0x0010u, .word_count = 24u, .precision = 0, .role = "A"}
};

static const TnpuTensorRead cv32e40p_int4_packed_convstream_default_v2_seg_seg_conv_reads[1] = {
    {.tensor_idx = 3, .addr = 0x0028u, .precision = 0, .role = "C"}
};

static const TnpuSegment cv32e40p_int4_packed_convstream_default_v2_segments[1] = {
    {.name = "seg_conv", .im_start_addr = 0x8000u, .writes = cv32e40p_int4_packed_convstream_default_v2_seg_seg_conv_writes, .write_count = 1u, .reads = cv32e40p_int4_packed_convstream_default_v2_seg_seg_conv_reads, .read_count = 1u}
};

static const TnpuTensorDesc cv32e40p_int4_packed_convstream_default_v2_tensors[5] = {
    {.name = "xmat", .data = xmat_data, .dtype = TNPU_DTYPE_INT4, .rank = 2, .shape = {20, 7, 1, 1}, .elem_count = 140},
    {.name = "cols", .data = cols_data, .dtype = TNPU_DTYPE_INT4, .rank = 2, .shape = {6, 63, 1, 1}, .elem_count = 378},
    {.name = "w", .data = w_data, .dtype = TNPU_DTYPE_INT4, .rank = 2, .shape = {63, 5, 1, 1}, .elem_count = 315},
    {.name = "y", .data = y_data, .dtype = TNPU_DTYPE_INT4, .rank = 2, .shape = {6, 5, 1, 1}, .elem_count = 30},
    {.name = "y_expected", .data = y_expected_data, .dtype = TNPU_DTYPE_INT4, .rank = 2, .shape = {6, 5, 1, 1}, .elem_count = 30}
};

static const TnpuOp cv32e40p_int4_packed_convstream_default_v2_ops[3] = {
    {.kind = TNPU_OP_PRELOAD_UB, .index = 0u},
    {.kind = TNPU_OP_PRELOAD_IM, .index = 0u},
    {.kind = TNPU_OP_SEGMENT, .index = 0u}
};

const TnpuProgram cv32e40p_int4_packed_convstream_default_v2 = {
    .name = "cv32e40p_int4_packed_convstream_default_v2",
    .tensors = cv32e40p_int4_packed_convstream_default_v2_tensors,
    .tensor_count = 5u,
    .input_tensor_indices = cv32e40p_int4_packed_convstream_default_v2_input_indices,
    .input_count = 1u,
    .output_tensor_indices = cv32e40p_int4_packed_convstream_default_v2_output_indices,
    .output_count = 1u,
    .ub_preloads = cv32e40p_int4_packed_convstream_default_v2_ub_preloads,
    .ub_preload_count = 1u,
    .im_preloads = cv32e40p_int4_packed_convstream_default_v2_im_preloads,
    .im_preload_count = 1u,
    .segments = cv32e40p_int4_packed_convstream_default_v2_segments,
    .segment_count = 1u,
    .host_ops = NULL,
    .host_op_count = 0u,
    .verify_ops = NULL,
    .verify_op_count = 0u,
    .ops = cv32e40p_int4_packed_convstream_default_v2_ops,
    .op_count = 3u,
};
