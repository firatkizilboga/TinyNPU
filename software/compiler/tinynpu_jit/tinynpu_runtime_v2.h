#ifndef TINYNPU_RUNTIME_V2_H
#define TINYNPU_RUNTIME_V2_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TNPU_MMVR_WORDS_32 4

typedef enum {
    TNPU_DTYPE_INT4 = 0,
    TNPU_DTYPE_INT8 = 1,
    TNPU_DTYPE_INT16 = 2,
    TNPU_DTYPE_INT32 = 3,
    TNPU_DTYPE_FLOAT32 = 4,
} TnpuDType;

typedef enum {
    TNPU_HOST_ALIAS = 0,
    TNPU_HOST_RELU = 1,
    TNPU_HOST_SIGMOID = 2,
    TNPU_HOST_GELU = 3,
    TNPU_HOST_QUANTIZE = 4,
    TNPU_HOST_DEQUANTIZE = 5,
    TNPU_HOST_REQUANTIZE = 6,
    TNPU_HOST_RESHAPE = 7,
    TNPU_HOST_TRANSPOSE = 8,
    TNPU_HOST_SOFTMAX = 9,
    TNPU_HOST_MEAN = 10,
    TNPU_HOST_IM2COL = 11,
    TNPU_HOST_LAYOUT_RESTORE = 12,
    TNPU_HOST_RMSNORM = 13,
    TNPU_HOST_ROPE = 14,
    TNPU_HOST_SILU = 15,
    TNPU_HOST_MUL = 16,
    TNPU_HOST_ADD = 17,
    TNPU_HOST_SOFTMAX_F16 = 18,
    TNPU_HOST_K_CACHE_SCATTER_WRITE = 19,
    TNPU_HOST_CAUSAL_MASK = 20,
    TNPU_HOST_CONCAT_LASTDIM2 = 21,
} TnpuHostKind;

typedef enum {
    TNPU_OP_PRELOAD_UB = 0,
    TNPU_OP_PRELOAD_IM = 1,
    TNPU_OP_HOST = 2,
    TNPU_OP_SEGMENT = 3,
    TNPU_OP_VERIFY = 4,
} TnpuOpKind;

typedef struct {
    const char *name;
    void *data;
    uint8_t dtype;
    uint8_t rank;
    uint16_t reserved0;
    int shape[4];
    int elem_count;
} TnpuTensorDesc;

typedef struct {
    void *data;
    const TnpuTensorDesc *desc;
    int elem_count;
} TnpuTensor;

typedef struct {
    uint16_t tensor_idx;
    uint16_t addr;
    uint16_t word_count;
    uint8_t precision;
    uint8_t transform;
    int32_t attrs_i32[2];
    float attrs_f32[1];
    const char *role;
} TnpuTensorWrite;

typedef enum {
    TNPU_WRITE_TRANSFORM_NONE = 0,
    TNPU_WRITE_QUANTIZE_F32_TO_INT16 = 1,
    TNPU_WRITE_XFORM_Q_F16_I16 = 2,
} TnpuTensorWriteTransform;

typedef enum {
    TNPU_READ_TRANSFORM_NONE = 0,
    TNPU_READ_DEQUANTIZE_INT16_TO_FLOAT32 = 1,
} TnpuTensorReadTransform;

typedef struct {
    uint16_t tensor_idx;
    uint16_t addr;
    uint8_t precision;
    uint8_t transform;
    int32_t attrs_i32[2];
    float attrs_f32[1];
    const char *role;
} TnpuTensorRead;

typedef struct {
    const char *name;
    uint16_t im_start_addr;
    const TnpuTensorWrite *writes;
    uint32_t write_count;
    const TnpuTensorRead *reads;
    uint32_t read_count;
} TnpuSegment;

typedef struct {
    const char *label;
    uint16_t base_addr;
    const uint32_t (*image)[TNPU_MMVR_WORDS_32];
    uint32_t word_count;
} TnpuImageLoad;

typedef struct {
    const char *name;
    uint8_t kind;
    uint16_t input_idx;
    uint16_t input1_idx;
    uint16_t output_idx;
    int32_t attrs_i32[8];
    float attrs_f32[2];
    const int *arr0;
    uint32_t arr0_len;
} TnpuHostOp;

typedef struct {
    const char *label;
    uint16_t actual_tensor_idx;
    uint16_t expected_tensor_idx;
    uint8_t is_final_output;
} TnpuVerifyOp;

typedef struct {
    uint8_t kind;
    uint16_t index;
} TnpuOp;

typedef struct {
    const char *name;
    const TnpuTensorDesc *tensors;
    uint32_t tensor_count;

    const uint16_t *input_tensor_indices;
    uint32_t input_count;
    const uint16_t *output_tensor_indices;
    uint32_t output_count;

    const TnpuImageLoad *ub_preloads;
    uint32_t ub_preload_count;
    const TnpuImageLoad *im_preloads;
    uint32_t im_preload_count;

    const TnpuSegment *segments;
    uint32_t segment_count;
    const TnpuHostOp *host_ops;
    uint32_t host_op_count;
    const TnpuVerifyOp *verify_ops;
    uint32_t verify_op_count;

    const TnpuOp *ops;
    uint32_t op_count;
} TnpuProgram;

int tinynpu_run(
    const TnpuProgram *program,
    const TnpuTensor *const *inputs,
    const TnpuTensor *const *outputs,
    void *scratch,
    uint32_t scratch_words);

int tinynpu_run_repeat(
    const TnpuProgram *program,
    const TnpuTensor *const *inputs,
    const TnpuTensor *const *outputs,
    void *scratch,
    uint32_t scratch_words,
    uint32_t repeat_count);

#ifdef __cplusplus
}
#endif

#endif
