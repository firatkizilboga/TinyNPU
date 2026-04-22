import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "scripts"))

from tinynpu_jit import emit_cv32e40p_c, emit_cv32e40p_program_v2
from tinynpu_jit.golden import GoldenModel
from tinynpu_jit.host_ops import execute_host_op
from tinynpu_jit import (
    DType,
    describe_int16_k_cache_append,
    describe_int16_v_cache_append,
    ExecutionPlan,
    HostOp,
    MatMulOp,
    NpuSegment,
    TensorKind,
    TensorSpec,
    VerificationMode,
    VerifyTensor,
    b_slot_word_stride,
    compile_plan,
    make_b_cache_specs,
    make_kv_cache_specs,
    make_native_int16_k_cache_specs,
    make_native_int16_kv_cache_specs,
    make_native_int16_v_cache_specs,
    make_rope_cos_sin_table_q14,
    make_rope_cs_tensor_spec,
    run_host_emulation,
)
from tinynpu import TinyNPUProgram
from tinynpu.isa import BReadMode, OutputLayout, PrecisionMode, WritebackMode, XformMode
from run_cv32e40p_decode_attention_jit_demo import (
    build_artifact_legacy as build_decode_attention_artifact_legacy,
    build_artifact_via_builder as build_decode_attention_artifact_via_builder,
)
from tinynpu_jit.blocks.gpt2_block import (
    QGPT2Block,
    QGPT2BlockConfig,
    build_decode_artifact as build_gpt2_decode_artifact,
    build_prefill_artifact as build_gpt2_prefill_artifact,
    build_shared_state as build_gpt2_shared_state,
    extend_kv_cache as extend_gpt2_kv_cache,
    reference_decode as reference_gpt2_decode,
    reference_prefill as reference_gpt2_prefill,
)
from run_cv32e40p_prefill_transformer_block_jit_demo import build_artifact as build_prefill_transformer_block_artifact


def test_emit_cv32e40p_c_for_two_segment_relu_chain():
    x = np.array(
        [
            [1, -2, 3, -4],
            [5, -6, 7, -8],
            [-2, 1, -1, 2],
            [0, 3, -3, 1],
        ],
        dtype=np.int16,
    )
    w0 = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.int16,
    )
    w1 = np.array(
        [
            [1, 2, 0, 1],
            [0, 1, 1, 0],
            [2, 0, 1, 1],
            [1, 1, 0, 2],
        ],
        dtype=np.int16,
    )

    tensors = {
        "x": TensorSpec("x", x.shape, DType.INT16, TensorKind.INPUT),
        "w0": TensorSpec("w0", w0.shape, DType.INT16, TensorKind.CONSTANT, data=w0),
        "h": TensorSpec("h", (4, 4), DType.INT16, TensorKind.INTERMEDIATE),
        "h_relu": TensorSpec("h_relu", (4, 4), DType.INT16, TensorKind.INTERMEDIATE),
        "w1": TensorSpec("w1", w1.shape, DType.INT16, TensorKind.CONSTANT, data=w1),
        "y": TensorSpec("y", (4, 4), DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        NpuSegment("seg0", [MatMulOp("op0", "x", "w0", "h")], inputs=["x", "w0"], outputs=["h"]),
        HostOp("relu_h", "relu", inputs=["h"], outputs=["h_relu"]),
        NpuSegment("seg1", [MatMulOp("op1", "h_relu", "w1", "y")], inputs=["h_relu", "w1"], outputs=["y"]),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["x"], outputs=["y"])
    plan.add_verification_step("y", "final_y")

    h = np.clip(x.astype(np.int32) @ w0.astype(np.int32), -32768, 32767).astype(np.int16)
    y = np.clip(np.maximum(h, 0).astype(np.int32) @ w1.astype(np.int32), -32768, 32767).astype(np.int16)
    artifact = compile_plan(plan, {"y": y})

    source = emit_cv32e40p_c(artifact, {"x": x}, program_name="unit_test_demo")

    assert "load_ub_image" in source
    assert "load_im_image" in source
    assert "host_relu" in source
    assert "write_tensor_to_npu(&x" in source
    assert "write_tensor_to_npu(&h_relu" in source
    assert 'read_tensor_from_npu(&h, ' in source
    assert 'read_tensor_from_npu(&y, ' in source
    assert "All outputs matched expected tensors" in source


def test_emit_cv32e40p_c_with_cpu_baseline_for_segment():
    x = np.array([[1, -2], [3, 4]], dtype=np.int16)
    w = np.array([[2, 1], [0, -1]], dtype=np.int16)
    bias = np.array([[1, -3]], dtype=np.int32)

    tensors = {
        "x": TensorSpec("x", x.shape, DType.INT16, TensorKind.INPUT),
        "w": TensorSpec("w", w.shape, DType.INT16, TensorKind.CONSTANT, data=w),
        "y_bias": TensorSpec("y_bias", bias.shape, DType.INT32, TensorKind.CONSTANT, data=bias),
        "y": TensorSpec("y", (2, 2), DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        NpuSegment(
            "seg_cpu",
            [MatMulOp("op0", "x", "w", "y", bias="y_bias", multiplier=3, shift=1, activation="relu")],
            inputs=["x", "w", "y_bias"],
            outputs=["y"],
        ),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["x"], outputs=["y"])
    plan.add_verification_step("y", "final_y")

    y = np.array([[5, 0], [10, 0]], dtype=np.int16)
    artifact = compile_plan(plan, {"y": y})

    source = emit_cv32e40p_c(
        artifact,
        {"x": x},
        program_name="unit_test_cpu_baseline",
        emit_cpu_baseline=True,
        verify_cpu_baseline=True,
    )

    assert "host_matmul(&y__cpu_seg_cpu" in source
    assert 'print_cycle_delta32("segment.seg_cpu.cpu"' in source
    assert 'print_cycle_delta32("segment.seg_cpu.npu"' in source
    assert 'cpu baseline mismatch: segment seg_cpu output y' in source


def test_emit_cv32e40p_c_with_repeat_count_accumulates_warm_totals():
    x = np.array([[1, -2], [3, 4]], dtype=np.int16)
    w = np.array([[2, 1], [0, -1]], dtype=np.int16)

    tensors = {
        "x": TensorSpec("x", x.shape, DType.INT16, TensorKind.INPUT),
        "w": TensorSpec("w", w.shape, DType.INT16, TensorKind.CONSTANT, data=w),
        "y": TensorSpec("y", (2, 2), DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        NpuSegment("seg_repeat", [MatMulOp("op0", "x", "w", "y")], inputs=["x", "w"], outputs=["y"]),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["x"], outputs=["y"])
    plan.add_verification_step("y", "final_y")

    y = np.clip(x.astype(np.int32) @ w.astype(np.int32), -32768, 32767).astype(np.int16)
    artifact = compile_plan(plan, {"y": y})

    source = emit_cv32e40p_c(
        artifact,
        {"x": x},
        program_name="unit_test_repeat",
        emit_cpu_baseline=True,
        verify_cpu_baseline=True,
        repeat_count=10,
    )

    assert "for (int repeat_iter = 0; repeat_iter < 10; ++repeat_iter)" in source
    assert 'if (repeat_iter == 0) printf("NpuSegment: seg_repeat\\n");' in source
    assert "segment_npu_total += delta;" in source
    assert "segment_cpu_total += delta;" in source
    assert 'printf("repeat.program.npu.hot.total cycles=%lu\\n"' in source
    assert 'printf("repeat.program.cpu.hot.avg cycles=%lu\\n"' in source


def test_emit_cv32e40p_program_v2_structured_program():
    x = np.array(
        [
            [1, -2, 3, -4],
            [5, -6, 7, -8],
            [-2, 1, -1, 2],
            [0, 3, -3, 1],
        ],
        dtype=np.int16,
    )
    w0 = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.int16,
    )
    w1 = np.array(
        [
            [1, 2, 0, 1],
            [0, 1, 1, 0],
            [2, 0, 1, 1],
            [1, 1, 0, 2],
        ],
        dtype=np.int16,
    )

    tensors = {
        "x": TensorSpec("x", x.shape, DType.INT16, TensorKind.INPUT),
        "w0": TensorSpec("w0", w0.shape, DType.INT16, TensorKind.CONSTANT, data=w0),
        "h": TensorSpec("h", (4, 4), DType.INT16, TensorKind.INTERMEDIATE),
        "h_relu": TensorSpec("h_relu", (4, 4), DType.INT16, TensorKind.INTERMEDIATE),
        "w1": TensorSpec("w1", w1.shape, DType.INT16, TensorKind.CONSTANT, data=w1),
        "y": TensorSpec("y", (4, 4), DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        NpuSegment("seg0", [MatMulOp("op0", "x", "w0", "h")], inputs=["x", "w0"], outputs=["h"]),
        HostOp("relu_h", "relu", inputs=["h"], outputs=["h_relu"]),
        NpuSegment("seg1", [MatMulOp("op1", "h_relu", "w1", "y")], inputs=["h_relu", "w1"], outputs=["y"]),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["x"], outputs=["y"])
    plan.add_verification_step("y", "final_y")

    h = np.clip(x.astype(np.int32) @ w0.astype(np.int32), -32768, 32767).astype(np.int16)
    y = np.clip(np.maximum(h, 0).astype(np.int32) @ w1.astype(np.int32), -32768, 32767).astype(np.int16)
    artifact = compile_plan(plan, {"y": y})

    source = emit_cv32e40p_program_v2(artifact, {"x": x}, program_name="unit_test_program_v2")

    assert '#include "tinynpu_runtime_v2.h"' in source
    assert "const TnpuProgram unit_test_program_v2" in source
    assert "TNPU_OP_PRELOAD_UB" in source


def test_compile_plan_canonicalizes_runtime_input_to_fp16_xform():
    x_f = np.array(
        [
            [1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 2.5, -1.5],
            [0.0, 1.0, -1.0, 2.0, 3.0, -3.0, 4.0, -4.0],
            [2.0, 2.0, 2.0, 2.0, -2.0, -2.0, -2.0, -2.0],
            [1.5, -1.5, 0.0, 0.0, 1.0, -1.0, 2.0, -2.0],
            [3.0, 0.0, -3.0, 1.0, -1.0, 2.0, -2.0, 4.0],
            [0.25, -0.25, 0.75, -0.75, 1.25, -1.25, 1.75, -1.75],
            [4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0],
            [1.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )
    scale = 0.5
    x_q = np.clip(np.rint(x_f / scale), -32768, 32767).astype(np.int16)
    w = np.eye(8, dtype=np.int16)
    y = x_q.copy()

    tensors = {
        "x_f": TensorSpec("x_f", x_f.shape, DType.FLOAT32, TensorKind.INPUT),
        "x_q": TensorSpec("x_q", x_q.shape, DType.INT16, TensorKind.INTERMEDIATE),
        "w": TensorSpec("w", w.shape, DType.INT16, TensorKind.CONSTANT, data=w),
        "y": TensorSpec("y", y.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        HostOp("quant_x", "quantize", inputs=["x_f"], outputs=["x_q"], attrs={"scale": scale, "zero_point": 0}),
        NpuSegment("seg0", [MatMulOp("op0", "x_q", "w", "y")], inputs=["x_q", "w"], outputs=["y"]),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["x_f"], outputs=["y"])
    plan.add_verification_step("y", "final_y")
    artifact = compile_plan(plan, {"y": y})
    quant = next(step for step in artifact.plan.steps if isinstance(step, HostOp) and step.name == "quant_x")

    assert artifact.plan.tensors["x_f"].dtype == DType.INT16
    assert artifact.plan.tensors["x_f"].metadata.get("value_encoding") == "fp16_bits"
    assert quant.attrs.get("input_encoding") == "fp16_bits"
    assert quant.attrs.get("_npu_write_transform") == "xform_q_f16_i16"

    source = emit_cv32e40p_program_v2(artifact, {"x_f": x_f}, program_name="unit_test_v2_runtime_input_xform")

    assert "TNPU_WRITE_XFORM_Q_F16_I16" in source
    assert ".transform = TNPU_WRITE_XFORM_Q_F16_I16" in source
    assert "TNPU_HOST_QUANTIZE" not in source
    assert "TNPU_OP_SEGMENT" in source
    assert "TNPU_OP_VERIFY" in source


def test_emit_cv32e40p_program_v2_falls_back_to_float_quantize_for_generic_host_output():
    x_f = np.arange(64, dtype=np.float32).reshape(8, 8) / 16.0
    bias_f = np.ones((8, 8), dtype=np.float32)
    x_sum = x_f + bias_f
    x_q = np.clip(np.rint(x_sum / 0.5), -32768, 32767).astype(np.int16)
    w = np.eye(8, dtype=np.int16)
    y = x_q.copy()

    tensors = {
        "x_f": TensorSpec("x_f", x_f.shape, DType.FLOAT32, TensorKind.INPUT),
        "bias_f": TensorSpec("bias_f", bias_f.shape, DType.FLOAT32, TensorKind.CONSTANT, data=bias_f),
        "x_sum": TensorSpec("x_sum", x_f.shape, DType.FLOAT32, TensorKind.INTERMEDIATE),
        "x_q": TensorSpec("x_q", x_q.shape, DType.INT16, TensorKind.INTERMEDIATE),
        "w": TensorSpec("w", w.shape, DType.INT16, TensorKind.CONSTANT, data=w),
        "y": TensorSpec("y", y.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        HostOp("add_bias", "add", inputs=["x_f", "bias_f"], outputs=["x_sum"]),
        HostOp("quant_x", "quantize", inputs=["x_sum"], outputs=["x_q"], attrs={"scale": 0.5, "zero_point": 0}),
        NpuSegment("seg0", [MatMulOp("op0", "x_q", "w", "y")], inputs=["x_q", "w"], outputs=["y"]),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["x_f"], outputs=["y"])
    plan.add_verification_step("y", "final_y")
    artifact = compile_plan(plan, {"y": y})
    quant = next(step for step in artifact.plan.steps if isinstance(step, HostOp) and step.name == "quant_x")

    assert quant.attrs.get("_npu_write_transform") == "quantize_f32_i16"

    source = emit_cv32e40p_program_v2(artifact, {"x_f": x_f}, program_name="unit_test_v2_float_quant_fallback")
    assert "TNPU_WRITE_QUANTIZE_F32_TO_INT16" in source
    assert ".transform = TNPU_WRITE_QUANTIZE_F32_TO_INT16" in source


def test_emit_cv32e40p_program_v2_absorbs_f16_quantize_into_xform_write():
    scores = np.array([[1, 0, -1, 2, -2, 3, -3, 4]], dtype=np.int16)
    probs_f16_bits = np.zeros_like(scores, dtype=np.int16)
    attn_q = np.zeros_like(scores, dtype=np.int16)
    w = np.eye(8, dtype=np.int16)
    y = np.zeros_like(scores, dtype=np.int16)

    tensors = {
        "scores": TensorSpec("scores", scores.shape, DType.INT16, TensorKind.INPUT),
        "probs_f16": TensorSpec("probs_f16", probs_f16_bits.shape, DType.INT16, TensorKind.INTERMEDIATE),
        "attn_q": TensorSpec("attn_q", attn_q.shape, DType.INT16, TensorKind.INTERMEDIATE),
        "w": TensorSpec("w", w.shape, DType.INT16, TensorKind.CONSTANT, data=w),
        "y": TensorSpec("y", y.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        HostOp("softmax_scores_f16", "softmax_f16", inputs=["scores"], outputs=["probs_f16"], attrs={"axis": -1}),
        HostOp("quantize_probs", "quantize", inputs=["probs_f16"], outputs=["attn_q"], attrs={"scale": 0.5, "zero_point": 0}),
        NpuSegment("seg0", [MatMulOp("op0", "attn_q", "w", "y")], inputs=["attn_q", "w"], outputs=["y"]),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["scores"], outputs=["y"])
    plan.add_verification_step("y", "final_y")
    artifact = compile_plan(plan, {"y": y})

    source = emit_cv32e40p_program_v2(artifact, {"scores": scores}, program_name="unit_test_v2_f16_quant_absorb")

    assert "TNPU_WRITE_XFORM_Q_F16_I16" in source
    assert ".transform = TNPU_WRITE_XFORM_Q_F16_I16" in source
    assert "TNPU_HOST_SOFTMAX_F16" in source
    assert "TNPU_HOST_QUANTIZE" not in source


def test_compile_plan_canonicalizes_layernorm_boundary_to_fp16_xform():
    x_f = np.arange(64, dtype=np.float32).reshape(8, 8) / 16.0
    ln_wb = np.concatenate(
        [
            np.ones((8,), dtype=np.float32),
            np.zeros((8,), dtype=np.float32),
        ]
    ).astype(np.float32)
    x_norm_q = np.zeros((8, 8), dtype=np.int16)
    w = np.eye(8, dtype=np.int16)
    y = np.zeros((8, 8), dtype=np.int16)

    tensors = {
        "x_f": TensorSpec("x_f", x_f.shape, DType.FLOAT32, TensorKind.INPUT),
        "ln_wb": TensorSpec("ln_wb", ln_wb.shape, DType.FLOAT32, TensorKind.CONSTANT, data=ln_wb),
        "x_norm": TensorSpec("x_norm", x_f.shape, DType.FLOAT32, TensorKind.INTERMEDIATE),
        "x_norm_q": TensorSpec("x_norm_q", x_norm_q.shape, DType.INT16, TensorKind.INTERMEDIATE),
        "w": TensorSpec("w", w.shape, DType.INT16, TensorKind.CONSTANT, data=w),
        "y": TensorSpec("y", y.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        HostOp("ln", "layernorm", inputs=["x_f", "ln_wb"], outputs=["x_norm"], attrs={"eps": 1.0e-5}),
        HostOp("quant_x_norm", "quantize", inputs=["x_norm"], outputs=["x_norm_q"], attrs={"scale": 0.5, "zero_point": 0}),
        NpuSegment("seg0", [MatMulOp("op0", "x_norm_q", "w", "y")], inputs=["x_norm_q", "w"], outputs=["y"]),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["x_f"], outputs=["y"])
    plan.add_verification_step("y", "final_y")

    artifact = compile_plan(plan, {"y": y})
    layernorm = next(step for step in artifact.plan.steps if isinstance(step, HostOp) and step.name == "ln")
    quant = next(step for step in artifact.plan.steps if isinstance(step, HostOp) and step.name == "quant_x_norm")

    assert layernorm.attrs.get("output_encoding") == "fp16_bits"
    assert quant.attrs.get("input_encoding") == "fp16_bits"
    assert quant.attrs.get("_npu_write_transform") == "xform_q_f16_i16"

    source = emit_cv32e40p_program_v2(artifact, {"x_f": x_f}, program_name="unit_test_v2_ln_xform")
    assert "TNPU_WRITE_XFORM_Q_F16_I16" in source


def test_emit_cv32e40p_program_v2_absorbs_dequantize_into_segment_read():
    x_q = np.array(
        [
            [8, -4, 2, 0, 1, -1, 3, -3],
            [0, 2, -2, 4, -4, 6, -6, 8],
            [1, 1, 1, 1, -1, -1, -1, -1],
            [3, -3, 0, 0, 2, -2, 5, -5],
            [7, 0, -7, 2, -2, 4, -4, 6],
            [1, -1, 2, -2, 3, -3, 4, -4],
            [8, 6, 4, 2, -2, -4, -6, -8],
            [2, 0, 2, 0, -2, 0, -2, 0],
        ],
        dtype=np.int16,
    )
    w = np.eye(8, dtype=np.int16)
    y_q = x_q.copy()
    scale = 0.25
    y_f = (y_q.astype(np.float32) * scale).astype(np.float32)

    tensors = {
        "x_q": TensorSpec("x_q", x_q.shape, DType.INT16, TensorKind.INPUT),
        "w": TensorSpec("w", w.shape, DType.INT16, TensorKind.CONSTANT, data=w),
        "y_q": TensorSpec("y_q", y_q.shape, DType.INT16, TensorKind.INTERMEDIATE),
        "y_f": TensorSpec("y_f", y_f.shape, DType.FLOAT32, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        NpuSegment("seg0", [MatMulOp("op0", "x_q", "w", "y_q")], inputs=["x_q", "w"], outputs=["y_q"]),
        HostOp("dequant_y", "dequantize", inputs=["y_q"], outputs=["y_f"], attrs={"scale": scale, "zero_point": 0}),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["x_q"], outputs=["y_f"])
    plan.add_verification_step("y_f", "final_y")
    artifact = compile_plan(plan, {"y_f": y_f})

    source = emit_cv32e40p_program_v2(artifact, {"x_q": x_q}, program_name="unit_test_v2_dequant_absorb")

    assert "TNPU_READ_DEQUANTIZE_INT16_TO_FLOAT32" in source
    assert ".transform = TNPU_READ_DEQUANTIZE_INT16_TO_FLOAT32" in source
    assert "TNPU_HOST_DEQUANTIZE" not in source


def test_compile_plan_rejects_float32_npu_weight_constant():
    x = np.eye(8, dtype=np.int16)
    w_f = np.eye(8, dtype=np.float32)
    y = np.eye(8, dtype=np.int16)

    tensors = {
        "x": TensorSpec("x", x.shape, DType.INT16, TensorKind.INPUT),
        "w_f": TensorSpec("w_f", w_f.shape, DType.FLOAT32, TensorKind.CONSTANT, data=w_f),
        "y": TensorSpec("y", y.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        NpuSegment("seg0", [MatMulOp("op0", "x", "w_f", "y")], inputs=["x", "w_f"], outputs=["y"]),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["x"], outputs=["y"])

    with pytest.raises(ValueError, match="INT4/INT8/INT16|FLOAT32 constant"):
        compile_plan(plan, {"y": y})


def test_compile_plan_folds_leading_input_quantize_into_int16_input_contract():
    x_f = (np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3) / 8.0) - 0.5
    x_q = np.clip(np.rint(x_f / 0.25), -32768, 32767).astype(np.int16)
    x_im2col = x_q.reshape(9, 1)
    w = np.array([[3]], dtype=np.int16)
    y_q = (x_im2col.astype(np.int32) * 3).astype(np.int16)
    y_f = y_q.astype(np.float32) * 0.5

    tensors = {
        "x_f": TensorSpec("x_f", x_f.shape, DType.FLOAT32, TensorKind.INPUT),
        "x_q": TensorSpec("x_q", x_q.shape, DType.INT16, TensorKind.INTERMEDIATE),
        "x_im2col": TensorSpec("x_im2col", x_im2col.shape, DType.INT16, TensorKind.INTERMEDIATE),
        "w": TensorSpec("w", w.shape, DType.INT16, TensorKind.CONSTANT, data=w),
        "y_q": TensorSpec("y_q", y_q.shape, DType.INT16, TensorKind.INTERMEDIATE),
        "y_f": TensorSpec("y_f", y_f.shape, DType.FLOAT32, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        HostOp("q_in", "quantize", inputs=["x_f"], outputs=["x_q"], attrs={"scale": 0.25, "zero_point": 0}),
        HostOp(
            "im2col",
            "im2col",
            inputs=["x_q"],
            outputs=["x_im2col"],
            attrs={"kernel_size": 1, "stride": 1, "padding": 0, "input_layout": "chw"},
        ),
        NpuSegment("seg0", [MatMulOp("op0", "x_im2col", "w", "y_q")], inputs=["x_im2col", "w"], outputs=["y_q"]),
        HostOp("dq_out", "dequantize", inputs=["y_q"], outputs=["y_f"], attrs={"scale": 0.5, "zero_point": 0}),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["x_f"], outputs=["y_f"])
    plan.add_verification_step("y_f", "final_y")

    artifact = compile_plan(plan, {"y_f": y_f})
    x_spec = artifact.plan.tensors["x_f"]

    assert x_spec.dtype == DType.INT16
    assert x_spec.metadata.get("runtime_input_transform") == "quantize_f32_i16"
    assert float(x_spec.metadata.get("runtime_input_scale", 0.0)) == pytest.approx(0.25)
    assert int(x_spec.metadata.get("runtime_input_zero_point", 123)) == 0
    assert "x_q" not in artifact.plan.tensors
    assert all(not (isinstance(step, HostOp) and step.name == "q_in") for step in artifact.plan.steps)

    result = artifact.run_host_emulation({"x_f": x_f}, verification=VerificationMode.DEBUG)
    np.testing.assert_allclose(result.tensors["y_f"], y_f, rtol=1e-5, atol=1e-6)

    source = emit_cv32e40p_program_v2(artifact, {"x_f": x_f}, program_name="unit_test_input_contract_fold")
    assert "TNPU_HOST_QUANTIZE" not in source
    assert '"x_f", .data = x_f_data, .dtype = TNPU_DTYPE_INT16' in source


def test_compile_plan_fuses_layout_restore_relu_im2col_chain():
    x = np.eye(4, dtype=np.int16)
    w0 = np.eye(4, dtype=np.int16)
    w1 = np.eye(4, dtype=np.int16)
    y = np.eye(4, dtype=np.int16)

    tensors = {
        "x": TensorSpec("x", x.shape, DType.INT16, TensorKind.INPUT),
        "w0": TensorSpec("w0", w0.shape, DType.INT16, TensorKind.CONSTANT, data=w0),
        "mat0": TensorSpec("mat0", (4, 4), DType.INT16, TensorKind.INTERMEDIATE),
        "restored0": TensorSpec("restored0", (1, 1, 2, 2), DType.INT16, TensorKind.INTERMEDIATE),
        "relu0": TensorSpec("relu0", (1, 1, 2, 2), DType.INT16, TensorKind.INTERMEDIATE),
        "cols1": TensorSpec("cols1", (4, 4), DType.INT16, TensorKind.INTERMEDIATE),
        "w1": TensorSpec("w1", w1.shape, DType.INT16, TensorKind.CONSTANT, data=w1),
        "y": TensorSpec("y", y.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        NpuSegment("seg0", [MatMulOp("op0", "x", "w0", "mat0")], inputs=["x", "w0"], outputs=["mat0"]),
        HostOp(
            "restore0",
            "layout_restore",
            inputs=["mat0"],
            outputs=["restored0"],
            attrs={"layout": "chw", "original_shape": (1, 2, 2), "out_h": 2, "out_w": 2, "out_channels": 1},
        ),
        HostOp("relu0_step", "relu", inputs=["restored0"], outputs=["relu0"]),
        HostOp(
            "im2col1",
            "im2col",
            inputs=["relu0"],
            outputs=["cols1"],
            attrs={"kernel_size": 1, "stride": 1, "padding": 0, "input_layout": "chw", "input_channels": 1},
        ),
        NpuSegment("seg1", [MatMulOp("op1", "cols1", "w1", "y")], inputs=["cols1", "w1"], outputs=["y"]),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["x"], outputs=["y"])
    plan.add_verification_step("y", "final_y")

    artifact = compile_plan(plan, {"y": y})

    kinds = [getattr(step, "kind", type(step).__name__) for step in artifact.plan.steps]
    assert "relu" not in kinds
    assert "layout_restore" not in kinds
    im2col = next(step for step in artifact.plan.steps if isinstance(step, HostOp) and step.kind == "im2col")
    assert im2col.inputs == ["mat0"]
    assert im2col.attrs["input_layout"] == "matrix_hwc"
    assert im2col.attrs["matrix_h"] == 2
    assert im2col.attrs["matrix_w"] == 2
    assert im2col.attrs["matrix_c"] == 1
    seg0 = next(step for step in artifact.plan.steps if isinstance(step, NpuSegment) and step.name == "seg0")
    assert seg0.ops[0].activation == "relu"


def test_compile_plan_fuses_layout_restore_sigmoid_dequantize_tail():
    x = np.eye(4, dtype=np.int16)
    w = np.eye(4, dtype=np.int16)
    y = np.ones((1, 1, 2, 2), dtype=np.float32)

    tensors = {
        "x": TensorSpec("x", x.shape, DType.INT16, TensorKind.INPUT),
        "w": TensorSpec("w", w.shape, DType.INT16, TensorKind.CONSTANT, data=w),
        "mat": TensorSpec("mat", (4, 4), DType.INT16, TensorKind.INTERMEDIATE),
        "restored": TensorSpec("restored", (1, 1, 2, 2), DType.INT16, TensorKind.INTERMEDIATE),
        "sigm": TensorSpec("sigm", (1, 1, 2, 2), DType.INT16, TensorKind.INTERMEDIATE),
        "y": TensorSpec("y", y.shape, DType.FLOAT32, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        NpuSegment("seg0", [MatMulOp("op0", "x", "w", "mat")], inputs=["x", "w"], outputs=["mat"]),
        HostOp(
            "restore",
            "layout_restore",
            inputs=["mat"],
            outputs=["restored"],
            attrs={"layout": "chw", "original_shape": (1, 2, 2), "out_h": 2, "out_w": 2, "out_channels": 1},
        ),
        HostOp("sigmoid_step", "sigmoid", inputs=["restored"], outputs=["sigm"]),
        HostOp("dq", "dequantize", inputs=["sigm"], outputs=["y"], attrs={"scale": 1.0, "zero_point": 0}),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["x"], outputs=["y"])
    plan.add_verification_step("y", "final_y")

    artifact = compile_plan(plan, {"y": y})

    kinds = [getattr(step, "kind", type(step).__name__) for step in artifact.plan.steps]
    assert "sigmoid" not in kinds
    seg0 = next(step for step in artifact.plan.steps if isinstance(step, NpuSegment) and step.name == "seg0")
    assert seg0.ops[0].activation == "sigmoid"
    restore = next(step for step in artifact.plan.steps if isinstance(step, HostOp) and step.kind == "layout_restore")
    assert restore.outputs == ["sigm"]
    dequant = next(step for step in artifact.plan.steps if isinstance(step, HostOp) and step.kind == "dequantize")
    assert dequant.inputs == ["sigm"]


def test_compile_plan_marks_rhs_chaining_output_as_b_layout():
    lhs0 = np.arange(1, 65, dtype=np.int16).reshape(8, 8)
    rhs0 = np.eye(8, dtype=np.int16)
    lhs1 = np.flipud(np.eye(8, dtype=np.int16))
    expected_mid = np.clip(lhs0.astype(np.int32) @ rhs0.astype(np.int32), -32768, 32767).astype(np.int16)
    expected_y = np.clip(lhs1.astype(np.int32) @ expected_mid.astype(np.int32), -32768, 32767).astype(np.int16)

    tensors = {
        "lhs0": TensorSpec("lhs0", lhs0.shape, DType.INT16, TensorKind.INPUT),
        "rhs0": TensorSpec("rhs0", rhs0.shape, DType.INT16, TensorKind.CONSTANT, data=rhs0),
        "lhs1": TensorSpec("lhs1", lhs1.shape, DType.INT16, TensorKind.CONSTANT, data=lhs1),
        "mid": TensorSpec("mid", expected_mid.shape, DType.INT16, TensorKind.INTERMEDIATE),
        "y": TensorSpec("y", expected_y.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        NpuSegment(
            "seg_b_chain",
            [
                MatMulOp("op0", "lhs0", "rhs0", "mid"),
                MatMulOp("op1", "lhs1", "mid", "y"),
            ],
            inputs=["lhs0", "rhs0", "lhs1"],
            outputs=["y"],
        ),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["lhs0"], outputs=["y"])
    artifact = compile_plan(plan, {"y": expected_y})

    seg = artifact.plan.steps[0]
    assert seg.ops[0].output_layout == "b"
    assert seg.ops[1].output_layout == "c"
    assert artifact.segment_artifacts["seg_b_chain"].symbol_table["mid"]["role"] == "B"


def test_tinynpu_program_encodes_output_word_offset_in_matmul_instruction():
    program = TinyNPUProgram()
    lhs = np.eye(8, dtype=np.int16)
    rhs = np.eye(8, dtype=np.int16)
    program.declare_data("lhs", lhs, precision=PrecisionMode.INT16, role="A")
    program.declare_data("rhs", rhs, precision=PrecisionMode.INT16, role="B")
    program.matmul(
        "lhs",
        "rhs",
        "out",
        in_precision=PrecisionMode.INT16,
        out_precision=PrecisionMode.INT16,
        output_layout=OutputLayout.B,
        output_word_offset=13,
    )
    program.halt()
    binary = program.compile()

    inst = int(binary["im"][0])
    assert ((inst >> 184) & 0xFFFF) == 13
    assert ((inst >> 72) & 0x3) == int(OutputLayout.B)


def test_tinynpu_program_encodes_b_word_offset_in_matmul_instruction():
    program = TinyNPUProgram()
    lhs = np.eye(8, dtype=np.int16)
    rhs = np.eye(8, dtype=np.int16)
    program.declare_data("lhs", lhs, precision=PrecisionMode.INT16, role="A")
    program.declare_data("rhs", rhs, precision=PrecisionMode.INT16, role="B")
    program.matmul(
        "lhs",
        "rhs",
        "out",
        in_precision=PrecisionMode.INT16,
        out_precision=PrecisionMode.INT16,
        output_layout=OutputLayout.C,
        b_word_offset=9,
    )
    program.halt()
    binary = program.compile()

    inst = int(binary["im"][0])
    assert ((inst >> 56) & 0xFFFF) == 9
    assert ((inst >> 72) & 0x3) == int(OutputLayout.C)


def test_tinynpu_program_encodes_b_read_mode_in_matmul_instruction():
    program = TinyNPUProgram()
    lhs = np.eye(8, dtype=np.int16)
    rhs = np.eye(8, dtype=np.int16)
    program.declare_data("lhs", lhs, precision=PrecisionMode.INT16, role="A")
    program.declare_data("rhs", rhs, precision=PrecisionMode.INT16, role="B")
    program.matmul(
        "lhs",
        "rhs",
        "out",
        in_precision=PrecisionMode.INT16,
        out_precision=PrecisionMode.INT16,
        b_read_mode=BReadMode.K_CACHE_INT16,
    )
    program.halt()
    binary = program.compile()

    inst = int(binary["im"][0])
    assert ((inst >> 52) & 0xF) == int(BReadMode.K_CACHE_INT16)


def test_tinynpu_program_encodes_writeback_mode_in_matmul_instruction():
    program = TinyNPUProgram()
    lhs = np.ones((1, 8), dtype=np.int16)
    rhs = np.ones((8, 16), dtype=np.int16)
    program.declare_data("lhs", lhs, precision=PrecisionMode.INT16, role="A")
    program.declare_data("rhs", rhs, precision=PrecisionMode.INT16, role="B")
    program.declare_data("cache", np.zeros((16, 16), dtype=np.int16), precision=PrecisionMode.INT16, role="B")
    program.matmul(
        "lhs",
        "rhs",
        "cache",
        in_precision=PrecisionMode.INT16,
        out_precision=PrecisionMode.INT16,
        output_layout=OutputLayout.B,
        writeback_mode=WritebackMode.V_CACHE_APPEND_INT16,
        output_word_offset=17,
    )
    program.halt()
    binary = program.compile()

    inst = int(binary["im"][0])
    assert ((inst >> 248) & 0xF) == int(WritebackMode.V_CACHE_APPEND_INT16)
    assert ((inst >> 184) & 0xFFFF) == 17


def test_tinynpu_program_encodes_k_cache_append_mode_in_matmul_instruction():
    program = TinyNPUProgram()
    lhs = np.ones((1, 8), dtype=np.int16)
    rhs = np.ones((8, 16), dtype=np.int16)
    program.declare_data("lhs", lhs, precision=PrecisionMode.INT16, role="A")
    program.declare_data("rhs", rhs, precision=PrecisionMode.INT16, role="B")
    program.declare_data("cache", np.zeros((16, 16), dtype=np.int16), precision=PrecisionMode.INT16, role="B")
    program.matmul(
        "lhs",
        "rhs",
        "cache",
        in_precision=PrecisionMode.INT16,
        out_precision=PrecisionMode.INT16,
        output_layout=OutputLayout.B,
        writeback_mode=WritebackMode.K_CACHE_APPEND_INT16,
        output_word_offset=17,
    )
    program.halt()
    binary = program.compile()

    inst = int(binary["im"][0])
    assert ((inst >> 248) & 0xF) == int(WritebackMode.K_CACHE_APPEND_INT16)
    assert ((inst >> 184) & 0xFFFF) == 17


def test_tinynpu_program_encodes_xform_q_f16_i16_instruction():
    program = TinyNPUProgram()
    src = np.zeros((8, 8), dtype=np.int16)
    dst = np.zeros((8, 8), dtype=np.int16)
    program.declare_data("src_f16", src, precision=PrecisionMode.INT16, role="A")
    program.declare_data("dst_i16", dst, precision=PrecisionMode.INT16, role="A")
    program.xform_q_f16_i16("src_f16", "dst_i16", multiplier=123, shift=7)
    program.halt()
    binary = program.compile()

    inst = int(binary["im"][0])
    assert ((inst >> 252) & 0xF) == 0x4
    assert ((inst >> 248) & 0xF) == int(XformMode.Q_F16_I16)
    assert ((inst >> 200) & 0xFFFF) == 8
    assert ((inst >> 184) & 0xFFFF) == 123
    assert ((inst >> 176) & 0xFF) == 7


def test_tinynpu_program_encodes_xform_q_f16_i16_inplace_instruction():
    program = TinyNPUProgram()
    src = np.zeros((8, 8), dtype=np.int16)
    program.declare_data("src_f16", src, precision=PrecisionMode.INT16, role="A")
    program.xform_q_f16_i16("src_f16", multiplier=16, shift=1)
    program.halt()
    binary = program.compile()

    inst = int(binary["im"][0])
    src_addr = (inst >> 232) & 0xFFFF
    dst_addr = (inst >> 216) & 0xFFFF
    assert ((inst >> 252) & 0xF) == 0x4
    assert ((inst >> 248) & 0xF) == int(XformMode.Q_F16_I16)
    assert src_addr == dst_addr
    assert ((inst >> 200) & 0xFFFF) == 8
    assert ((inst >> 184) & 0xFFFF) == 16
    assert ((inst >> 176) & 0xFF) == 1


def test_compile_plan_preserves_output_word_offset_in_segment_binary():
    lhs = np.eye(8, dtype=np.int16)
    rhs = np.eye(8, dtype=np.int16)
    expected = np.eye(8, dtype=np.int16)

    tensors = {
        "lhs": TensorSpec("lhs", lhs.shape, DType.INT16, TensorKind.INPUT),
        "rhs": TensorSpec("rhs", rhs.shape, DType.INT16, TensorKind.CONSTANT, data=rhs),
        "out": TensorSpec("out", expected.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        NpuSegment(
            "seg_offset",
            [
                MatMulOp(
                    "op0",
                    "lhs",
                    "rhs",
                    "out",
                    output_layout="b",
                    output_word_offset=21,
                )
            ],
            inputs=["lhs", "rhs"],
            outputs=["out"],
        )
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["lhs"], outputs=["out"])
    artifact = compile_plan(plan, {"out": expected})

    assert artifact.plan.steps[0].ops[0].output_layout == "c"
    inst = int(artifact.segment_artifacts["seg_offset"].binary["im"][0])
    assert ((inst >> 184) & 0xFFFF) == 21
    assert ((inst >> 72) & 0x3) == int(OutputLayout.C)


def test_compile_plan_preserves_b_word_offset_in_segment_binary():
    lhs = np.eye(8, dtype=np.int16)
    rhs = np.eye(8, dtype=np.int16)
    expected = np.eye(8, dtype=np.int16)

    tensors = {
        "lhs": TensorSpec("lhs", lhs.shape, DType.INT16, TensorKind.INPUT),
        "rhs": TensorSpec("rhs", rhs.shape, DType.INT16, TensorKind.CONSTANT, data=rhs),
        "out": TensorSpec("out", expected.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        NpuSegment(
            "seg_b_input_offset",
            [
                MatMulOp(
                    "op0",
                    "lhs",
                    "rhs",
                    "out",
                    b_word_offset=17,
                )
            ],
            inputs=["lhs", "rhs"],
            outputs=["out"],
        )
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["lhs"], outputs=["out"])
    artifact = compile_plan(plan, {"out": expected})

    inst = int(artifact.segment_artifacts["seg_b_input_offset"].binary["im"][0])
    assert ((inst >> 56) & 0xFFFF) == 17
    assert ((inst >> 72) & 0x3) == int(OutputLayout.C)


def test_compile_plan_preserves_b_read_mode_in_segment_binary():
    lhs = np.eye(8, dtype=np.int16)
    rhs = np.eye(8, dtype=np.int16)
    expected = np.eye(8, dtype=np.int16)

    tensors = {
        "lhs": TensorSpec("lhs", lhs.shape, DType.INT16, TensorKind.INPUT),
        "rhs": TensorSpec("rhs", rhs.shape, DType.INT16, TensorKind.CONSTANT, data=rhs),
        "out": TensorSpec("out", expected.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        NpuSegment(
            "seg_k_read_mode",
            [
                MatMulOp(
                    "op0",
                    "lhs",
                    "rhs",
                    "out",
                    b_read_mode="k_cache_int16",
                )
            ],
            inputs=["lhs", "rhs"],
            outputs=["out"],
        )
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["lhs"], outputs=["out"])
    artifact = compile_plan(plan, {"out": expected})

    inst = int(artifact.segment_artifacts["seg_k_read_mode"].binary["im"][0])
    assert ((inst >> 52) & 0xF) == int(BReadMode.K_CACHE_INT16)


def test_compile_plan_preserves_writeback_mode_in_segment_binary():
    lhs = np.ones((1, 8), dtype=np.int16)
    rhs = np.ones((8, 16), dtype=np.int16)
    expected = np.zeros((16, 16), dtype=np.int16)

    tensors = {
        "lhs": TensorSpec("lhs", lhs.shape, DType.INT16, TensorKind.CONSTANT, data=lhs),
        "rhs": TensorSpec("rhs", rhs.shape, DType.INT16, TensorKind.CONSTANT, data=rhs),
        "cache": TensorSpec("cache", expected.shape, DType.INT16, TensorKind.INTERMEDIATE),
    }
    steps = [
        NpuSegment(
            "seg_v_append_mode",
            [
                MatMulOp(
                    "op0",
                    "lhs",
                    "rhs",
                    "cache",
                    output_layout="b",
                    writeback_mode="v_cache_append_int16",
                    output_word_offset=17,
                )
            ],
            inputs=[],
            outputs=[],
        )
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=[], outputs=[])
    artifact = compile_plan(plan, {})

    inst = int(artifact.segment_artifacts["seg_v_append_mode"].binary["im"][0])
    assert ((inst >> 248) & 0xF) == int(WritebackMode.V_CACHE_APPEND_INT16)
    assert ((inst >> 184) & 0xFFFF) == 17
    assert ((inst >> 72) & 0x3) == int(OutputLayout.B)


def test_compile_plan_preserves_k_cache_append_mode_in_segment_binary():
    lhs = np.ones((1, 8), dtype=np.int16)
    rhs = np.ones((8, 16), dtype=np.int16)
    expected = np.zeros((16, 16), dtype=np.int16)

    tensors = {
        "lhs": TensorSpec("lhs", lhs.shape, DType.INT16, TensorKind.CONSTANT, data=lhs),
        "rhs": TensorSpec("rhs", rhs.shape, DType.INT16, TensorKind.CONSTANT, data=rhs),
        "cache": TensorSpec("cache", expected.shape, DType.INT16, TensorKind.INTERMEDIATE),
    }
    steps = [
        NpuSegment(
            "seg_k_append_mode",
            [
                MatMulOp(
                    "op0",
                    "lhs",
                    "rhs",
                    "cache",
                    output_layout="b",
                    writeback_mode="k_cache_append_int16",
                    output_word_offset=17,
                )
            ],
            inputs=[],
            outputs=[],
        )
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=[], outputs=[])
    artifact = compile_plan(plan, {})

    inst = int(artifact.segment_artifacts["seg_k_append_mode"].binary["im"][0])
    assert ((inst >> 248) & 0xF) == int(WritebackMode.K_CACHE_APPEND_INT16)
    assert ((inst >> 184) & 0xFFFF) == 17
    assert ((inst >> 72) & 0x3) == int(OutputLayout.B)


def test_compile_plan_supports_b_cache_views_for_append_and_consume():
    lhs0 = np.eye(8, dtype=np.int16)
    rhs0 = np.eye(8, dtype=np.int16)
    lhs1 = np.flipud(np.eye(8, dtype=np.int16))
    rhs1 = np.rot90(np.eye(8, dtype=np.int16), 2).astype(np.int16)
    query = np.eye(8, dtype=np.int16)
    token0 = np.clip(lhs0.astype(np.int32) @ rhs0.astype(np.int32), -32768, 32767).astype(np.int16)
    token1 = np.clip(lhs1.astype(np.int32) @ rhs1.astype(np.int32), -32768, 32767).astype(np.int16)
    expected = np.clip(query.astype(np.int32) @ token1.astype(np.int32), -32768, 32767).astype(np.int16)

    tensors = {
        "lhs0": TensorSpec("lhs0", lhs0.shape, DType.INT16, TensorKind.CONSTANT, data=lhs0),
        "rhs0": TensorSpec("rhs0", rhs0.shape, DType.INT16, TensorKind.CONSTANT, data=rhs0),
        "lhs1": TensorSpec("lhs1", lhs1.shape, DType.INT16, TensorKind.CONSTANT, data=lhs1),
        "rhs1": TensorSpec("rhs1", rhs1.shape, DType.INT16, TensorKind.CONSTANT, data=rhs1),
        "query": TensorSpec("query", query.shape, DType.INT16, TensorKind.CONSTANT, data=query),
        "out": TensorSpec("out", expected.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    tensors.update(
        make_b_cache_specs(
            "cache",
            (8, 8),
            DType.INT16,
            slot_names=["cache_t0", "cache_t1"],
            cache_kind="K",
        )
    )
    steps = [
        NpuSegment(
            "seg_cache_views",
            [
                MatMulOp("op0", "lhs0", "rhs0", "cache_t0"),
                MatMulOp("op1", "lhs1", "rhs1", "cache_t1"),
                MatMulOp("op2", "query", "cache_t1", "out"),
            ],
            inputs=[],
            outputs=["out"],
        ),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=[], outputs=["out"])
    artifact = compile_plan(plan, {"out": expected})

    seg = artifact.segment_artifacts["seg_cache_views"]
    inst0 = int(seg.binary["im"][0])
    inst1 = int(seg.binary["im"][1])
    inst2 = int(seg.binary["im"][2])
    assert ((inst0 >> 184) & 0xFFFF) == 0
    assert ((inst1 >> 184) & 0xFFFF) == 8
    assert ((inst2 >> 56) & 0xFFFF) == 8
    assert ((inst0 >> 248) & 0xF) == int(WritebackMode.K_CACHE_APPEND_INT16)
    assert ((inst1 >> 248) & 0xF) == int(WritebackMode.K_CACHE_APPEND_INT16)
    assert ((inst2 >> 52) & 0xF) == int(BReadMode.K_CACHE_INT16)
    assert ((inst0 >> 72) & 0x3) == int(OutputLayout.B)
    assert ((inst1 >> 72) & 0x3) == int(OutputLayout.B)
    assert seg.symbol_table["cache_t1"]["base_name"] == "cache"
    assert seg.symbol_table["cache_t1"]["word_offset"] == 8
    memory_names = {entry.name for entry in seg.memory_plan.entries}
    assert "cache" in memory_names
    assert "cache_t0" not in memory_names
    assert "cache_t1" not in memory_names
    assert plan.tensors["cache"].metadata["cache_kind"] == "K"
    assert plan.tensors["cache_t1"].metadata["cache_kind"] == "K"
    assert plan.tensors["cache_t1"].metadata["cache_slot_stride_words"] == 8


def test_make_b_cache_specs_computes_slot_offsets():
    specs = make_b_cache_specs(
        "k_cache",
        (8, 8),
        DType.INT16,
        slot_names=["k_t0", "k_t1", "k_t2"],
        cache_kind="K",
    )

    assert specs["k_cache"].shape == (24, 8)
    assert specs["k_cache"].metadata["cache_kind"] == "K"
    assert specs["k_cache"].metadata["cache_slot_stride_words"] == b_slot_word_stride((8, 8), DType.INT16)
    assert specs["k_t0"].metadata["storage_view_of"] == "k_cache"
    assert specs["k_t0"].metadata["storage_word_offset"] == 0
    assert specs["k_t1"].metadata["storage_word_offset"] == b_slot_word_stride((8, 8), DType.INT16)
    assert specs["k_t2"].metadata["storage_word_offset"] == 2 * b_slot_word_stride((8, 8), DType.INT16)
    assert specs["k_t1"].metadata["cache_kind"] == "K"
    assert specs["k_t2"].metadata["cache_slot_index"] == 2


def test_make_kv_cache_specs_tags_k_and_v_slots_separately():
    specs = make_kv_cache_specs(
        k_base_name="k_cache",
        v_base_name="v_cache",
        k_slot_shape=(8, 8),
        v_slot_shape=(8, 8),
        dtype=DType.INT16,
        slot_suffixes=["t0", "t1"],
    )

    assert specs["k_cache"].shape == (16, 8)
    assert specs["v_cache"].shape == (16, 8)
    assert specs["k_cache"].metadata["cache_kind"] == "K"
    assert specs["v_cache"].metadata["cache_kind"] == "V"
    assert specs["k_cache_t1"].metadata["cache_kind"] == "K"
    assert specs["v_cache_t1"].metadata["cache_kind"] == "V"
    assert specs["k_cache_t1"].metadata["storage_word_offset"] == 8
    assert specs["v_cache_t1"].metadata["storage_word_offset"] == 8


def test_make_native_int16_k_cache_specs_uses_column_append_offsets():
    specs = make_native_int16_k_cache_specs(
        "k_cache",
        d_head=16,
        token_capacity=16,
        token_names=["k_t1", "k_t9"],
        token_indices=[1, 9],
    )

    assert specs["k_cache"].shape == (16, 16)
    assert specs["k_cache"].metadata["cache_kind"] == "K"
    assert specs["k_t1"].metadata["storage_word_offset"] == 1
    assert specs["k_t9"].metadata["storage_word_offset"] == 17
    assert specs["k_t9"].metadata["cache_scatter_word_addrs"] == tuple(range(16, 32))


def test_make_native_int16_v_cache_specs_uses_sparse_row_offsets():
    specs = make_native_int16_v_cache_specs(
        "v_cache",
        d_head=16,
        token_capacity=16,
        token_names=["v_t1", "v_t9"],
        token_indices=[1, 9],
    )

    assert specs["v_cache"].shape == (16, 16)
    assert specs["v_cache"].metadata["cache_kind"] == "V"
    assert specs["v_t1"].metadata["storage_word_offset"] == 1
    assert specs["v_t9"].metadata["storage_word_offset"] == 17
    assert specs["v_t9"].metadata["cache_scatter_word_addrs"] == (17, 25)


def test_compile_plan_supports_native_v_cache_append_and_consume():
    lhs0 = np.array([[1, 2, -1, 0, 3, -2, 1, 4]], dtype=np.int16)
    rhs0 = np.array(
        [
            [1, 0, 1, 0, -1, 2, 0, 1],
            [0, 1, 0, 1, 2, -1, 1, 0],
            [1, -1, 1, 0, 0, 1, 2, 1],
            [2, 0, -1, 1, 1, 0, -1, 2],
            [0, 2, 1, -1, 1, 1, 0, 0],
            [1, 1, 0, 2, -1, 0, 1, -1],
            [0, -1, 2, 1, 0, 1, 1, 2],
            [1, 0, 1, 1, 2, 0, -1, 1],
        ],
        dtype=np.int16,
    )
    lhs1 = np.array([[2, 1, 0, -1, 1, 2, 0, 1]], dtype=np.int16)
    rhs1 = np.array(
        [
            [0, 1, 1, 0, 2, 1, 0, -1],
            [1, 0, 2, 1, -1, 0, 1, 2],
            [2, 1, 0, -1, 1, 2, 1, 0],
            [1, -1, 1, 2, 0, 1, 2, 1],
            [0, 2, 1, 0, 1, -1, 1, 2],
            [1, 1, -1, 1, 2, 0, 0, 1],
            [2, 0, 1, 1, 0, 2, -1, 1],
            [1, 2, 0, 1, -1, 1, 2, 0],
        ],
        dtype=np.int16,
    )
    attn = np.zeros((1, 16), dtype=np.int16)
    attn[0, 1] = 2
    attn[0, 9] = -1
    token0 = np.clip(lhs0.astype(np.int32) @ rhs0.astype(np.int32), -32768, 32767).astype(np.int16)
    token1 = np.clip(lhs1.astype(np.int32) @ rhs1.astype(np.int32), -32768, 32767).astype(np.int16)
    full_cache = np.zeros((16, 8), dtype=np.int16)
    full_cache[1, :] = token0[0]
    full_cache[9, :] = token1[0]
    expected = np.clip(attn.astype(np.int32) @ full_cache.astype(np.int32), -32768, 32767).astype(np.int16)

    tensors = {
        "lhs0": TensorSpec("lhs0", lhs0.shape, DType.INT16, TensorKind.CONSTANT, data=lhs0),
        "rhs0": TensorSpec("rhs0", rhs0.shape, DType.INT16, TensorKind.CONSTANT, data=rhs0),
        "lhs1": TensorSpec("lhs1", lhs1.shape, DType.INT16, TensorKind.CONSTANT, data=lhs1),
        "rhs1": TensorSpec("rhs1", rhs1.shape, DType.INT16, TensorKind.CONSTANT, data=rhs1),
        "attn": TensorSpec("attn", attn.shape, DType.INT16, TensorKind.CONSTANT, data=attn),
        "out": TensorSpec("out", expected.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    tensors.update(
        make_native_int16_v_cache_specs(
            "v_cache",
            d_head=8,
            token_capacity=16,
            token_names=["v_cache_t1", "v_cache_t9"],
            token_indices=[1, 9],
        )
    )
    steps = [
        NpuSegment(
            "seg_v_cache_views",
            [
                MatMulOp("op0", "lhs0", "rhs0", "v_cache_t1"),
                MatMulOp("op1", "lhs1", "rhs1", "v_cache_t9"),
                MatMulOp("op2", "attn", "v_cache", "out"),
            ],
            inputs=[],
            outputs=["out"],
        ),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=[], outputs=["out"])
    artifact = compile_plan(plan, {"out": expected})

    seg = artifact.segment_artifacts["seg_v_cache_views"]
    inst0 = int(seg.binary["im"][0])
    inst1 = int(seg.binary["im"][1])
    inst2 = int(seg.binary["im"][2])
    assert ((inst0 >> 184) & 0xFFFF) == 1
    assert ((inst1 >> 184) & 0xFFFF) == 9
    assert ((inst0 >> 248) & 0xF) == int(WritebackMode.V_CACHE_APPEND_INT16)
    assert ((inst1 >> 248) & 0xF) == int(WritebackMode.V_CACHE_APPEND_INT16)
    assert ((inst2 >> 52) & 0xF) == int(BReadMode.NORMAL)
    assert plan.tensors["v_cache"].metadata["cache_kind"] == "V"


def test_compile_plan_supports_decode_attention_bridge():
    lhs_k0 = np.array([[1, 2, -1, 0, 3, -2, 1, 4]], dtype=np.int16)
    rhs_k0 = np.array(
        [
            [1, 0, 1, 0, -1, 2, 0, 1],
            [0, 1, 0, 1, 2, -1, 1, 0],
            [1, -1, 1, 0, 0, 1, 2, 1],
            [2, 0, -1, 1, 1, 0, -1, 2],
            [0, 2, 1, -1, 1, 1, 0, 0],
            [1, 1, 0, 2, -1, 0, 1, -1],
            [0, -1, 2, 1, 0, 1, 1, 2],
            [1, 0, 1, 1, 2, 0, -1, 1],
        ],
        dtype=np.int16,
    )
    lhs_k1 = np.array([[2, 1, 0, -1, 1, 2, 0, 1]], dtype=np.int16)
    rhs_k1 = np.array(
        [
            [0, 1, 1, 0, 2, 1, 0, -1],
            [1, 0, 2, 1, -1, 0, 1, 2],
            [2, 1, 0, -1, 1, 2, 1, 0],
            [1, -1, 1, 2, 0, 1, 2, 1],
            [0, 2, 1, 0, 1, -1, 1, 2],
            [1, 1, -1, 1, 2, 0, 0, 1],
            [2, 0, 1, 1, 0, 2, -1, 1],
            [1, 2, 0, 1, -1, 1, 2, 0],
        ],
        dtype=np.int16,
    )
    lhs_v0 = np.array([[1, 0, 2, -1, 1, 0, 1, 2]], dtype=np.int16)
    rhs_v0 = np.array(
        [
            [1, 2, 0, 1, -1, 0, 2, 1],
            [0, 1, 2, 0, 1, 2, -1, 1],
            [2, 0, 1, 2, 0, 1, 1, -1],
            [1, -1, 0, 1, 2, 1, 0, 2],
            [0, 2, 1, 0, 1, -1, 2, 1],
            [1, 0, -1, 2, 1, 0, 1, 2],
            [2, 1, 0, 1, -1, 2, 0, 1],
            [1, 2, 1, 0, 2, 1, -1, 0],
        ],
        dtype=np.int16,
    )
    lhs_v1 = np.array([[0, 1, 1, 2, -1, 1, 0, 1]], dtype=np.int16)
    rhs_v1 = np.array(
        [
            [2, 0, 1, -1, 2, 1, 0, 1],
            [1, 2, 0, 1, 0, -1, 2, 1],
            [0, 1, 2, 1, -1, 0, 1, 2],
            [1, -1, 1, 0, 2, 1, 2, 0],
            [2, 1, 0, 2, 1, 0, -1, 1],
            [0, 2, 1, 1, 0, 2, 1, -1],
            [1, 0, 2, 1, 1, -1, 0, 2],
            [2, 1, -1, 0, 1, 2, 1, 0],
        ],
        dtype=np.int16,
    )
    query = np.array([[1, -1, 2, 0, 1, 3, -2, 1]], dtype=np.int16)
    golden = GoldenModel()
    attn_scale = 1.0 / 256.0

    k0 = np.clip(lhs_k0.astype(np.int32) @ rhs_k0.astype(np.int32), -32768, 32767).astype(np.int16)
    k1 = np.clip(lhs_k1.astype(np.int32) @ rhs_k1.astype(np.int32), -32768, 32767).astype(np.int16)
    v0 = np.clip(lhs_v0.astype(np.int32) @ rhs_v0.astype(np.int32), -32768, 32767).astype(np.int16)
    v1 = np.clip(lhs_v1.astype(np.int32) @ rhs_v1.astype(np.int32), -32768, 32767).astype(np.int16)

    k_cache = np.zeros((8, 16), dtype=np.int16)
    k_cache[:, 1] = k0[0]
    k_cache[:, 9] = k1[0]
    scores = np.clip(query.astype(np.int32) @ k_cache.astype(np.int32), -32768, 32767).astype(np.int16)
    probs = golden.softmax(scores, axis=-1).astype(np.float32)
    attn_q = golden.quantize(probs, scale=attn_scale, out_dtype=DType.INT16)
    v_cache = np.zeros((16, 8), dtype=np.int16)
    v_cache[1, :] = v0[0]
    v_cache[9, :] = v1[0]
    expected = golden.matmul(attn_q, v_cache, shift=8, out_dtype=DType.INT16)

    tensors = {
        "lhs_k0": TensorSpec("lhs_k0", lhs_k0.shape, DType.INT16, TensorKind.CONSTANT, data=lhs_k0),
        "rhs_k0": TensorSpec("rhs_k0", rhs_k0.shape, DType.INT16, TensorKind.CONSTANT, data=rhs_k0),
        "lhs_k1": TensorSpec("lhs_k1", lhs_k1.shape, DType.INT16, TensorKind.CONSTANT, data=lhs_k1),
        "rhs_k1": TensorSpec("rhs_k1", rhs_k1.shape, DType.INT16, TensorKind.CONSTANT, data=rhs_k1),
        "lhs_v0": TensorSpec("lhs_v0", lhs_v0.shape, DType.INT16, TensorKind.CONSTANT, data=lhs_v0),
        "rhs_v0": TensorSpec("rhs_v0", rhs_v0.shape, DType.INT16, TensorKind.CONSTANT, data=rhs_v0),
        "lhs_v1": TensorSpec("lhs_v1", lhs_v1.shape, DType.INT16, TensorKind.CONSTANT, data=lhs_v1),
        "rhs_v1": TensorSpec("rhs_v1", rhs_v1.shape, DType.INT16, TensorKind.CONSTANT, data=rhs_v1),
        "query": TensorSpec("query", query.shape, DType.INT16, TensorKind.CONSTANT, data=query),
        "scores": TensorSpec("scores", scores.shape, DType.INT16, TensorKind.INTERMEDIATE),
        "probs": TensorSpec("probs", probs.shape, DType.FLOAT32, TensorKind.INTERMEDIATE),
        "attn_q": TensorSpec(
            "attn_q",
            attn_q.shape,
            DType.INT16,
            TensorKind.INTERMEDIATE,
            metadata={"storage_role": "A"},
        ),
        "out": TensorSpec("out", expected.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    tensors.update(
        make_native_int16_kv_cache_specs(
            k_base_name="k_cache",
            v_base_name="v_cache",
            d_head=8,
            token_capacity=16,
            token_names=["t1", "t9"],
            token_indices=[1, 9],
        )
    )
    steps = [
        NpuSegment(
            "seg_score",
            [
                MatMulOp("op_k0", "lhs_k0", "rhs_k0", "k_cache_t1"),
                MatMulOp("op_k1", "lhs_k1", "rhs_k1", "k_cache_t9"),
                MatMulOp("op_v0", "lhs_v0", "rhs_v0", "v_cache_t1"),
                MatMulOp("op_v1", "lhs_v1", "rhs_v1", "v_cache_t9"),
                MatMulOp("op_qk", "query", "k_cache", "scores"),
            ],
            inputs=[],
            outputs=["scores"],
        ),
        HostOp("softmax_scores", "softmax", inputs=["scores"], outputs=["probs"], attrs={"axis": -1}),
        HostOp(
            "quantize_probs",
            "quantize",
            inputs=["probs"],
            outputs=["attn_q"],
            attrs={"scale": attn_scale, "zero_point": 0, "dtype": DType.INT16},
        ),
        NpuSegment(
            "seg_value",
            [
                MatMulOp("op_av", "attn_q", "v_cache", "out", shift=8),
            ],
            inputs=["attn_q"],
            outputs=["out"],
        ),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=[], outputs=["out"])
    artifact = compile_plan(plan, {"out": expected})

    seg_score = artifact.segment_artifacts["seg_score"]
    inst_qk = int(seg_score.binary["im"][4])
    assert ((inst_qk >> 52) & 0xF) == int(BReadMode.K_CACHE_INT16)
    host_kinds = [step.kind for step in artifact.plan.steps if isinstance(step, HostOp)]
    assert host_kinds == ["softmax", "quantize"]
    seg_value = artifact.segment_artifacts["seg_value"]
    inst_av = int(seg_value.binary["im"][0])
    assert ((inst_av >> 52) & 0xF) == int(BReadMode.NORMAL)
    assert ((inst_av >> 112) & 0xFF) == 8


def test_decode_attention_builder_matches_legacy_artifact():
    params = {
        "d_model": 16,
        "n_heads": 1,
        "n_kv_heads": 1,
        "d_head": 16,
        "token_capacity": 32,
        "token_indices": [1, 9, 17, 25],
        "seed": 0,
    }
    legacy_artifact, legacy_expected, legacy_d_model = build_decode_attention_artifact_legacy(**params)
    builder_artifact, builder_expected, builder_d_model = build_decode_attention_artifact_via_builder(**params)

    assert legacy_d_model == builder_d_model
    np.testing.assert_array_equal(legacy_expected, builder_expected)
    assert legacy_artifact.plan.inputs == builder_artifact.plan.inputs
    assert legacy_artifact.plan.outputs == builder_artifact.plan.outputs
    assert legacy_artifact.expected_tensors.keys() == builder_artifact.expected_tensors.keys()
    for name in legacy_artifact.expected_tensors:
        np.testing.assert_array_equal(legacy_artifact.expected_tensors[name], builder_artifact.expected_tensors[name])

    def _step_signature(step):
        if isinstance(step, HostOp):
            return ("host", step.name, step.kind, tuple(step.inputs), tuple(step.outputs), dict(step.attrs))
        if isinstance(step, NpuSegment):
            return (
                "seg",
                step.name,
                tuple(step.inputs),
                tuple(step.outputs),
                tuple(
                    (
                        op.name,
                        op.lhs,
                        op.rhs,
                        op.out,
                        op.bias,
                        op.multiplier,
                        op.shift,
                        op.activation,
                        op.in_dtype,
                        op.out_dtype,
                        op.output_layout,
                        op.writeback_mode,
                        op.output_word_offset,
                        op.b_word_offset,
                        op.b_read_mode,
                        op.rope_cs_name,
                    )
                    for op in step.ops
                ),
            )
        if isinstance(step, VerifyTensor):
            return ("verify", step.tensor_name, step.label, step.is_final_output, step.float_atol)
        raise TypeError(step)

    assert [_step_signature(step) for step in legacy_artifact.plan.steps] == [
        _step_signature(step) for step in builder_artifact.plan.steps
    ]
    assert legacy_artifact.static_ub_image == builder_artifact.static_ub_image
    assert set(legacy_artifact.segment_artifacts) == set(builder_artifact.segment_artifacts)
    for seg_name in legacy_artifact.segment_artifacts:
        legacy_seg = legacy_artifact.segment_artifacts[seg_name]
        builder_seg = builder_artifact.segment_artifacts[seg_name]
        assert legacy_seg.binary["im"] == builder_seg.binary["im"]
        assert legacy_seg.binary["ub"] == builder_seg.binary["ub"]
        assert legacy_seg.symbol_table == builder_seg.symbol_table


def test_gpt2_block_module_builds_prefill_and_decode():
    prefill_artifact, _, prefill_ref = build_gpt2_prefill_artifact(
        d_model=32,
        d_head=8,
        n_heads=4,
        ffn_dim=128,
        prompt_len=8,
        seed=0,
    )
    decode_artifact, _, _, decode_ref = build_gpt2_decode_artifact(
        d_model=32,
        d_head=8,
        n_heads=4,
        ffn_dim=128,
        prompt_len=8,
        seed=0,
    )

    assert sorted(prefill_artifact.segment_artifacts.keys()) == [
        "seg_ffn_fc",
        "seg_ffn_proj",
        "seg_kv_cache",
        "seg_o_proj",
        "seg_q",
        "seg_score",
        "seg_value",
    ]
    assert sorted(decode_artifact.segment_artifacts.keys()) == [
        "seg_ffn_fc",
        "seg_ffn_proj",
        "seg_o_proj",
        "seg_qkv",
        "seg_score",
        "seg_value",
    ]
    assert prefill_artifact.plan.outputs == ["out"]
    assert decode_artifact.plan.outputs == ["out"]
    assert prefill_ref["out"].shape == (8, 32)
    assert decode_ref["out"].shape == (1, 32)


def test_qgpt2_block_matches_hf_style_fused_shapes():
    state = build_gpt2_shared_state(
        d_model=32,
        d_head=8,
        n_heads=4,
        ffn_dim=128,
        prompt_len=8,
        seed=0,
    )
    block = state["block"]

    assert isinstance(block, QGPT2Block)
    assert block.config == QGPT2BlockConfig(d_model=32, d_head=8, n_heads=4, ffn_dim=128)
    assert block.attn_c_attn_w.shape == (32, 96)
    assert block.attn_c_attn_b.shape == (1, 96)
    assert block.attn_c_proj_w.shape == (32, 32)
    assert block.attn_c_proj_b.shape == (1, 32)
    assert block.mlp_c_fc_w.shape == (32, 128)
    assert block.mlp_c_fc_b.shape == (1, 128)
    assert block.mlp_c_proj_w.shape == (128, 32)
    assert block.mlp_c_proj_b.shape == (1, 32)

    w_q, w_k, w_v = block.split_c_attn_weights()
    b_q, b_k, b_v = block.split_c_attn_biases_fp32()
    assert [w.shape for w in w_q] == [(32, 8)] * 4
    assert [w.shape for w in w_k] == [(32, 8)] * 4
    assert [w.shape for w in w_v] == [(32, 8)] * 4
    assert [b.shape for b in b_q] == [(1, 8)] * 4
    assert [b.shape for b in b_k] == [(1, 8)] * 4
    assert [b.shape for b in b_v] == [(1, 8)] * 4


def test_gpt2_two_block_prefill_decode_reuse_matches_full_sequence():
    d_model = 8
    d_head = 8
    n_heads = 1
    ffn_dim = 8
    prompt_len = 8
    act_scale = 1.0 / 32.0
    attn_scale = 1.0 / 256.0

    layer0 = build_gpt2_shared_state(
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        prompt_len=prompt_len,
        seed=0,
    )
    layer1 = build_gpt2_shared_state(
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        prompt_len=prompt_len,
        seed=1,
    )
    layers = [layer0, layer1]

    prompt_x0 = np.asarray(layer0["x_prompt_in"], dtype=np.float32)
    decode1_x0 = np.asarray(layer0["x_decode_in"], dtype=np.float32)
    decode2_x0 = np.random.default_rng(123).uniform(-0.25, 0.25, size=(1, d_model)).astype(np.float32)

    def run_stack_prefill(x_prompt: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
        caches: list[dict[str, object]] = []
        x = np.asarray(x_prompt, dtype=np.float32)
        for layer_state in layers:
            ref = reference_gpt2_prefill(
                layer_state,
                d_head=d_head,
                n_heads=n_heads,
                act_scale=act_scale,
                attn_scale=attn_scale,
                x_in=x,
            )
            caches.append(ref)
            x = np.asarray(ref["out"], dtype=np.float32)
        return x, caches

    def run_stack_decode(
        cache_refs: list[dict[str, object]],
        x_decode: np.ndarray,
    ) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]]]:
        decode_refs: list[dict[str, object]] = []
        updated_caches: list[dict[str, object]] = []
        x = np.asarray(x_decode, dtype=np.float32)
        for layer_state, cache_ref in zip(layers, cache_refs):
            ref = reference_gpt2_decode(
                layer_state,
                cache_ref,
                d_head=d_head,
                n_heads=n_heads,
                act_scale=act_scale,
                attn_scale=attn_scale,
                x_in=x,
            )
            decode_refs.append(ref)
            updated_caches.append(extend_gpt2_kv_cache(cache_ref, ref))
            x = np.asarray(ref["out"], dtype=np.float32)
        return x, decode_refs, updated_caches

    prompt_out, prompt_cache_refs = run_stack_prefill(prompt_x0)
    assert prompt_out.shape == (prompt_len, d_model)

    decode1_out, decode1_refs, decode1_cache_refs = run_stack_decode(prompt_cache_refs, decode1_x0)
    assert decode1_out.shape == (1, d_model)

    full_seq1_layer0 = reference_gpt2_prefill(
        layer0,
        d_head=d_head,
        n_heads=n_heads,
        act_scale=act_scale,
        attn_scale=attn_scale,
        x_in=np.concatenate([prompt_x0, decode1_x0], axis=0),
    )
    full_seq1_layer1 = reference_gpt2_prefill(
        layer1,
        d_head=d_head,
        n_heads=n_heads,
        act_scale=act_scale,
        attn_scale=attn_scale,
        x_in=np.asarray(full_seq1_layer0["out"], dtype=np.float32),
    )

    np.testing.assert_allclose(decode1_out, np.asarray(full_seq1_layer1["out"], dtype=np.float32)[-1:], atol=1.0e-5, rtol=1.0e-5)
    for head_idx in range(n_heads):
        np.testing.assert_array_equal(decode1_cache_refs[0]["k_heads"][head_idx], full_seq1_layer0["k_heads"][head_idx])
        np.testing.assert_array_equal(decode1_cache_refs[0]["v_heads"][head_idx], full_seq1_layer0["v_heads"][head_idx])
        np.testing.assert_array_equal(decode1_cache_refs[1]["k_heads"][head_idx], full_seq1_layer1["k_heads"][head_idx])
        np.testing.assert_array_equal(decode1_cache_refs[1]["v_heads"][head_idx], full_seq1_layer1["v_heads"][head_idx])

    decode2_out, decode2_refs, decode2_cache_refs = run_stack_decode(decode1_cache_refs, decode2_x0)
    assert decode2_out.shape == (1, d_model)
    assert decode2_refs[0]["out"].shape == (1, d_model)
    assert decode2_refs[1]["out"].shape == (1, d_model)

    full_seq2_layer0 = reference_gpt2_prefill(
        layer0,
        d_head=d_head,
        n_heads=n_heads,
        act_scale=act_scale,
        attn_scale=attn_scale,
        x_in=np.concatenate([prompt_x0, decode1_x0, decode2_x0], axis=0),
    )
    full_seq2_layer1 = reference_gpt2_prefill(
        layer1,
        d_head=d_head,
        n_heads=n_heads,
        act_scale=act_scale,
        attn_scale=attn_scale,
        x_in=np.asarray(full_seq2_layer0["out"], dtype=np.float32),
    )

    np.testing.assert_allclose(decode2_out, np.asarray(full_seq2_layer1["out"], dtype=np.float32)[-1:], atol=1.0e-5, rtol=1.0e-5)
    for head_idx in range(n_heads):
        np.testing.assert_array_equal(decode2_cache_refs[0]["k_heads"][head_idx], full_seq2_layer0["k_heads"][head_idx])
        np.testing.assert_array_equal(decode2_cache_refs[0]["v_heads"][head_idx], full_seq2_layer0["v_heads"][head_idx])
        np.testing.assert_array_equal(decode2_cache_refs[1]["k_heads"][head_idx], full_seq2_layer1["k_heads"][head_idx])
        np.testing.assert_array_equal(decode2_cache_refs[1]["v_heads"][head_idx], full_seq2_layer1["v_heads"][head_idx])


def test_describe_int16_k_cache_append_requires_lane_partial_writes():
    contract = describe_int16_k_cache_append(d_head=16, token_capacity=16, token_index=9)

    assert contract.cache_shape == (16, 16)
    assert contract.token_block == 1
    assert contract.token_lane == 1
    assert contract.k_tiles == 2
    assert contract.block_word_base == 16
    assert contract.block_word_count == 16
    assert contract.scatter_word_addrs == tuple(range(16, 32))
    assert contract.lane_partial_write is True


def test_describe_int16_v_cache_append_requires_sparse_word_writes():
    contract = describe_int16_v_cache_append(d_head=16, token_capacity=16, token_index=9)

    assert contract.cache_shape == (16, 16)
    assert contract.token_block == 1
    assert contract.row_in_block == 1
    assert contract.n_tiles == 2
    assert contract.block_word_base == 16
    assert contract.block_word_count == 16
    assert contract.scatter_word_addrs == (17, 25)
    assert contract.lane_partial_write is False


def test_tinynpu_program_preserves_predeclared_b_cache_shape_for_append():
    program = TinyNPUProgram()
    lhs0 = np.eye(8, dtype=np.int16)
    rhs0 = np.eye(8, dtype=np.int16)
    lhs1 = np.flipud(np.eye(8, dtype=np.int16))
    rhs1 = np.rot90(np.eye(8, dtype=np.int16), 2).astype(np.int16)

    program.declare_data("lhs0", lhs0, precision=PrecisionMode.INT16, role="A")
    program.declare_data("rhs0", rhs0, precision=PrecisionMode.INT16, role="B")
    program.declare_data("lhs1", lhs1, precision=PrecisionMode.INT16, role="A")
    program.declare_data("rhs1", rhs1, precision=PrecisionMode.INT16, role="B")
    program.declare_data("cache", np.zeros((16, 8), dtype=np.int16), precision=PrecisionMode.INT16, role="B")

    program.matmul(
        "lhs0",
        "rhs0",
        "cache",
        in_precision=PrecisionMode.INT16,
        out_precision=PrecisionMode.INT16,
        output_layout=OutputLayout.B,
        output_word_offset=0,
    )
    program.matmul(
        "lhs1",
        "rhs1",
        "cache",
        in_precision=PrecisionMode.INT16,
        out_precision=PrecisionMode.INT16,
        output_layout=OutputLayout.B,
        output_word_offset=8,
    )
    program.halt()
    binary = program.compile()

    cache = program.symbols["cache"]
    assert cache.shape == (16, 8)
    assert cache.storage_role == "B"
    assert cache.word_count == 16

    inst0 = int(binary["im"][0])
    inst1 = int(binary["im"][1])
    assert ((inst0 >> 184) & 0xFFFF) == 0
    assert ((inst1 >> 184) & 0xFFFF) == 8
    assert ((inst0 >> 72) & 0x3) == int(OutputLayout.B)
    assert ((inst1 >> 72) & 0x3) == int(OutputLayout.B)


def test_tinynpu_program_b_view_applies_output_and_rhs_offsets():
    program = TinyNPUProgram()
    lhs0 = np.eye(8, dtype=np.int16)
    rhs0 = np.eye(8, dtype=np.int16)
    lhs1 = np.flipud(np.eye(8, dtype=np.int16))
    rhs1 = np.rot90(np.eye(8, dtype=np.int16), 2).astype(np.int16)
    query = np.eye(8, dtype=np.int16)

    program.declare_data("lhs0", lhs0, precision=PrecisionMode.INT16, role="A")
    program.declare_data("rhs0", rhs0, precision=PrecisionMode.INT16, role="B")
    program.declare_data("lhs1", lhs1, precision=PrecisionMode.INT16, role="A")
    program.declare_data("rhs1", rhs1, precision=PrecisionMode.INT16, role="B")
    program.declare_data("query", query, precision=PrecisionMode.INT16, role="A")
    program.declare_data("cache", np.zeros((16, 8), dtype=np.int16), precision=PrecisionMode.INT16, role="B")
    program.declare_b_view("cache_t0", "cache", (8, 8), word_offset=0)
    program.declare_b_view("cache_t1", "cache", (8, 8), word_offset=8)

    program.matmul(
        "lhs0",
        "rhs0",
        "cache_t0",
        in_precision=PrecisionMode.INT16,
        out_precision=PrecisionMode.INT16,
        output_layout=OutputLayout.B,
    )
    program.matmul(
        "lhs1",
        "rhs1",
        "cache_t1",
        in_precision=PrecisionMode.INT16,
        out_precision=PrecisionMode.INT16,
        output_layout=OutputLayout.B,
    )
    program.matmul(
        "query",
        "cache_t1",
        "out",
        in_precision=PrecisionMode.INT16,
        out_precision=PrecisionMode.INT16,
        output_layout=OutputLayout.C,
    )
    program.halt()
    binary = program.compile()

    cache = program.symbols["cache"]
    cache_t1 = program.symbols["cache_t1"]
    assert cache_t1.addr == cache.addr
    assert cache_t1.word_offset == 8

    inst0 = int(binary["im"][0])
    inst1 = int(binary["im"][1])
    inst2 = int(binary["im"][2])
    assert ((inst0 >> 184) & 0xFFFF) == 0
    assert ((inst1 >> 184) & 0xFFFF) == 8
    assert ((inst2 >> 56) & 0xFFFF) == 8


def test_tinynpu_program_rejects_predeclared_output_cache_that_is_too_small():
    program = TinyNPUProgram()
    lhs = np.eye(8, dtype=np.int16)
    rhs = np.eye(8, dtype=np.int16)
    program.declare_data("lhs", lhs, precision=PrecisionMode.INT16, role="A")
    program.declare_data("rhs", rhs, precision=PrecisionMode.INT16, role="B")
    program.declare_data("cache", np.zeros((4, 8), dtype=np.int16), precision=PrecisionMode.INT16, role="B")

    with pytest.raises(ValueError, match="too small"):
        program.matmul(
            "lhs",
            "rhs",
            "cache",
            in_precision=PrecisionMode.INT16,
            out_precision=PrecisionMode.INT16,
            output_layout=OutputLayout.B,
            output_word_offset=0,
        )


def test_compile_plan_keeps_layout_restore_then_im2col():
    a = np.arange(20, dtype=np.int16).reshape(4, 5)
    b = np.ones((5, 3), dtype=np.int16)
    mat = np.clip(a.astype(np.int32) @ b.astype(np.int32), -32768, 32767).astype(np.int16)
    cols = np.zeros((1, 12), dtype=np.int16)
    w2 = np.ones((12, 2), dtype=np.int16)
    y = np.zeros((1, 2), dtype=np.int16)

    tensors = {
        "a": TensorSpec("a", a.shape, DType.INT16, TensorKind.CONSTANT, data=a),
        "b": TensorSpec("b", b.shape, DType.INT16, TensorKind.CONSTANT, data=b),
        "mat": TensorSpec("mat", mat.shape, DType.INT16, TensorKind.INTERMEDIATE),
        "restored": TensorSpec("restored", (2, 2, 3), DType.INT16, TensorKind.INTERMEDIATE),
        "cols": TensorSpec("cols", cols.shape, DType.INT16, TensorKind.INTERMEDIATE),
        "w2": TensorSpec("w2", w2.shape, DType.INT16, TensorKind.CONSTANT, data=w2),
        "y": TensorSpec("y", y.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        NpuSegment("seg0", [MatMulOp("op0", "a", "b", "mat")], inputs=["a", "b"], outputs=["mat"]),
        HostOp(
            "restore",
            "layout_restore",
            inputs=["mat"],
            outputs=["restored"],
            attrs={"layout": "hwc", "original_shape": (2, 2, 3), "out_h": 2, "out_w": 2, "out_channels": 3},
        ),
        HostOp(
            "im2col_next",
            "im2col",
            inputs=["restored"],
            outputs=["cols"],
            attrs={"kernel_size": 2, "stride": 1, "padding": 0, "input_layout": "hwc"},
        ),
        NpuSegment("seg1", [MatMulOp("op1", "cols", "w2", "y")], inputs=["cols", "w2"], outputs=["y"]),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=[], outputs=["y"])
    artifact = compile_plan(plan, {"y": y})

    kinds = [step.kind for step in artifact.plan.steps if isinstance(step, HostOp)]
    assert kinds == ["layout_restore", "im2col"]

    restore = next(step for step in artifact.plan.steps if isinstance(step, HostOp) and step.kind == "layout_restore")
    assert restore.inputs == ["mat"]
    assert restore.outputs == ["restored"]

    im2col = next(step for step in artifact.plan.steps if isinstance(step, HostOp) and step.kind == "im2col")
    assert im2col.inputs == ["restored"]
    assert im2col.attrs["input_layout"] == "hwc"

    source_v1 = emit_cv32e40p_c(artifact, {}, program_name="unit_test_matrix_hwc")
    assert "host_layout_restore" in source_v1
    assert "host_im2col" in source_v1

    source_v2 = emit_cv32e40p_program_v2(artifact, {}, program_name="unit_test_matrix_hwc_v2")
    assert source_v2.count(".kind = TNPU_HOST_LAYOUT_RESTORE") == 1
    assert source_v2.count(".kind = TNPU_HOST_IM2COL") == 1


def test_compile_plan_keeps_matrix_im2col_for_int8_by_default():
    xmat = np.arange(5 * 4 * 7, dtype=np.int8).reshape(20, 7)
    cols = np.zeros((6, 63), dtype=np.int8)
    w = np.ones((63, 5), dtype=np.int8)
    y = np.zeros((6, 5), dtype=np.int8)

    tensors = {
        "xmat": TensorSpec("xmat", xmat.shape, DType.INT8, TensorKind.INPUT),
        "cols": TensorSpec("cols", cols.shape, DType.INT8, TensorKind.INTERMEDIATE),
        "w": TensorSpec("w", w.shape, DType.INT8, TensorKind.CONSTANT, data=w),
        "y": TensorSpec("y", y.shape, DType.INT8, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        HostOp(
            "conv_im2col",
            "im2col",
            inputs=["xmat"],
            outputs=["cols"],
            attrs={
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
                "input_layout": "matrix_hwc",
                "matrix_h": 5,
                "matrix_w": 4,
                "matrix_c": 7,
            },
        ),
        NpuSegment(
            "seg_conv",
            [MatMulOp("op0", "cols", "w", "y", in_dtype=DType.INT8, out_dtype=DType.INT8)],
            inputs=["cols", "w"],
            outputs=["y"],
        ),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["xmat"], outputs=["y"])
    artifact = compile_plan(plan, {"y": y})

    host_im2col = [step for step in artifact.plan.steps if isinstance(step, HostOp) and step.kind == "im2col"]
    assert len(host_im2col) == 1

    seg = next(step for step in artifact.plan.steps if isinstance(step, NpuSegment))
    assert seg.ops[0].lhs == "cols"


def test_compile_plan_keeps_matrix_im2col_for_int4_by_default():
    xmat = np.arange(4 * 4 * 10, dtype=np.int16).reshape(16, 10)
    cols = np.zeros((4, 90), dtype=np.int16)
    w = np.ones((90, 9), dtype=np.int16)
    y = np.zeros((4, 9), dtype=np.int16)

    tensors = {
        "xmat": TensorSpec("xmat", xmat.shape, DType.INT4, TensorKind.INPUT),
        "cols": TensorSpec("cols", cols.shape, DType.INT4, TensorKind.INTERMEDIATE),
        "w": TensorSpec("w", w.shape, DType.INT4, TensorKind.CONSTANT, data=w),
        "y": TensorSpec("y", y.shape, DType.INT4, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        HostOp(
            "conv_im2col",
            "im2col",
            inputs=["xmat"],
            outputs=["cols"],
            attrs={
                "kernel_size": 3,
                "stride": 1,
                "padding": 0,
                "input_layout": "matrix_hwc",
                "matrix_h": 4,
                "matrix_w": 4,
                "matrix_c": 10,
            },
        ),
        NpuSegment(
            "seg_conv",
            [MatMulOp("op0", "cols", "w", "y", in_dtype=DType.INT4, out_dtype=DType.INT4)],
            inputs=["cols", "w"],
            outputs=["y"],
        ),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["xmat"], outputs=["y"])
    artifact = compile_plan(plan, {"y": y})

    host_im2col = [step for step in artifact.plan.steps if isinstance(step, HostOp) and step.kind == "im2col"]
    assert len(host_im2col) == 1

    seg = next(step for step in artifact.plan.steps if isinstance(step, NpuSegment))
    assert seg.ops[0].lhs == "cols"


@pytest.mark.parametrize(
    ("dtype", "dtype_enum", "precision"),
    [
        (DType.INT16, "TNPU_DTYPE_INT16", 2),
        (DType.INT8, "TNPU_DTYPE_INT8", 1),
        (DType.INT4, "TNPU_DTYPE_INT4", 0),
    ],
)
def test_emit_cv32e40p_program_v2_emits_precision_and_verify_for_all_integer_modes(dtype, dtype_enum, precision):
    x = np.array(
        [
            [1, -2, 3, -4],
            [2, 1, -3, 0],
            [-1, 4, -2, 3],
            [0, -1, 2, -3],
        ],
        dtype=np.int16,
    )
    w = np.array(
        [
            [1, 0, -1, 2],
            [2, -2, 1, 0],
            [-1, 3, 0, -2],
            [0, 1, 2, -1],
        ],
        dtype=np.int16,
    )
    y_i32 = x.astype(np.int32) @ w.astype(np.int32)
    if dtype == DType.INT4:
        y = np.clip(y_i32, -8, 7).astype(np.int16)
    elif dtype == DType.INT8:
        y = np.clip(y_i32, -128, 127).astype(np.int16)
    else:
        y = np.clip(y_i32, -32768, 32767).astype(np.int16)

    tensors = {
        "x": TensorSpec("x", x.shape, dtype, TensorKind.INPUT),
        "w": TensorSpec("w", w.shape, dtype, TensorKind.CONSTANT, data=w),
        "y": TensorSpec("y", y.shape, dtype, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        NpuSegment(
            "seg0",
            [MatMulOp("op0", "x", "w", "y", in_dtype=dtype, out_dtype=dtype)],
            inputs=["x", "w"],
            outputs=["y"],
        )
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["x"], outputs=["y"])
    plan.add_verification_step("y", "final_y")
    artifact = compile_plan(plan, {"y": y})

    source = emit_cv32e40p_program_v2(artifact, {"x": x}, program_name=f"unit_test_v2_{dtype.value}")

    assert dtype_enum in source
    assert "TNPU_OP_VERIFY" in source
    assert ".expected_tensor_idx" in source
    assert f".precision = {precision}" in source


def test_host_op_rmsnorm_emulation_and_v2_emit():
    x = np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
    weight = np.array([1.5, 0.5, 2.0, 1.0], dtype=np.float32)
    eps = 1.0e-6
    mean_sq = np.mean(np.square(x), axis=-1, keepdims=True)
    expected = (x / np.sqrt(mean_sq + eps)) * weight

    values = {"x": x, "w": weight, "y": np.zeros_like(x)}
    step = HostOp("rms0", "rmsnorm", inputs=["x", "w"], outputs=["y"], attrs={"eps": eps})
    execute_host_op(step, values, golden=GoldenModel())
    np.testing.assert_allclose(values["y"], expected, rtol=1e-5, atol=1e-5)

    tensors = {
        "x": TensorSpec("x", x.shape, DType.FLOAT32, TensorKind.CONSTANT, data=x),
        "w": TensorSpec("w", weight.shape, DType.FLOAT32, TensorKind.CONSTANT, data=weight),
        "y": TensorSpec("y", x.shape, DType.FLOAT32, TensorKind.OUTPUT, is_final_output=True),
    }
    plan = ExecutionPlan(tensors=tensors, steps=[step], inputs=[], outputs=["y"])
    plan.add_verification_step("y", "y")
    artifact = compile_plan(plan, {"y": expected.astype(np.float32)})
    source = emit_cv32e40p_program_v2(artifact, {}, program_name="unit_test_rmsnorm_v2")
    assert "TNPU_HOST_RMSNORM" in source
    assert ".input1_idx =" in source


def test_host_op_layernorm_emulation_and_v2_emit():
    x = np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
    weight = np.array([1.5, 0.5, 2.0, 1.0], dtype=np.float32)
    bias = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    weight_bias = np.stack([weight, bias], axis=0).astype(np.float32)
    eps = 1.0e-6
    mean = np.mean(x, axis=-1, keepdims=True)
    centered = x - mean
    var = np.mean(np.square(centered), axis=-1, keepdims=True)
    expected = ((centered / np.sqrt(var + eps)) * weight.reshape(1, -1) + bias.reshape(1, -1)).astype(np.float32)

    values = {"x": x, "wb": weight_bias, "y": np.zeros_like(x)}
    step = HostOp("ln0", "layernorm", inputs=["x", "wb"], outputs=["y"], attrs={"eps": eps})
    execute_host_op(step, values, golden=GoldenModel())
    np.testing.assert_allclose(values["y"], expected, rtol=1e-5, atol=1e-5)

    tensors = {
        "x": TensorSpec("x", x.shape, DType.FLOAT32, TensorKind.CONSTANT, data=x),
        "wb": TensorSpec("wb", weight_bias.shape, DType.FLOAT32, TensorKind.CONSTANT, data=weight_bias),
        "y": TensorSpec("y", x.shape, DType.FLOAT32, TensorKind.OUTPUT, is_final_output=True),
    }
    plan = ExecutionPlan(tensors=tensors, steps=[step], inputs=[], outputs=["y"])
    plan.add_verification_step("y", "y")
    artifact = compile_plan(plan, {"y": expected.astype(np.float32)})
    source_v2 = emit_cv32e40p_program_v2(artifact, {}, program_name="unit_test_layernorm_v2")
    source_cpu = emit_cv32e40p_c(artifact, {}, program_name="unit_test_layernorm_cpu", cpu_only_baseline=True)
    assert "TNPU_HOST_LAYERNORM" in source_v2
    assert ".input1_idx =" in source_v2
    assert "host_layernorm(" in source_cpu


def test_host_op_slice_row_emulation_and_emit():
    x = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    expected = np.array([[4, 5, 6]], dtype=np.int32)

    values = {"x": x, "y": np.zeros((1, 3), dtype=np.int32)}
    step = HostOp("slice1", "slice_row", inputs=["x"], outputs=["y"], attrs={"row_index": 1})
    execute_host_op(step, values, golden=GoldenModel())
    np.testing.assert_array_equal(values["y"], expected)

    tensors = {
        "x": TensorSpec("x", x.shape, DType.INT16, TensorKind.CONSTANT, data=x),
        "y": TensorSpec("y", (1, 3), DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    plan = ExecutionPlan(tensors=tensors, steps=[step], inputs=[], outputs=["y"])
    plan.add_verification_step("y", "y")
    artifact = compile_plan(plan, {"y": expected.astype(np.int16)})
    source_v2 = emit_cv32e40p_program_v2(artifact, {}, program_name="unit_test_slice_row_v2")
    source_cpu = emit_cv32e40p_c(artifact, {}, program_name="unit_test_slice_row_cpu", cpu_only_baseline=True)
    assert "TNPU_HOST_SLICE_ROW" in source_v2
    assert "host_slice_row(" in source_cpu


def test_host_op_rope_emulation_and_v2_emit():
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    head_dim = 4
    position = 3
    theta = 10000.0
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / np.float32(half)))
    angles = np.float32(position) * inv_freq
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    expected = np.array(x, copy=True)
    expected[..., :half] = x[..., :half] * cos - x[..., half:] * sin
    expected[..., half:] = x[..., half:] * cos + x[..., :half] * sin

    values = {"x": x, "y": np.zeros_like(x)}
    step = HostOp("rope0", "rope", inputs=["x"], outputs=["y"], attrs={"head_dim": head_dim, "position": position, "theta": theta})
    execute_host_op(step, values, golden=GoldenModel())
    np.testing.assert_allclose(values["y"], expected, rtol=1e-5, atol=1e-5)

    tensors = {
        "x": TensorSpec("x", x.shape, DType.FLOAT32, TensorKind.CONSTANT, data=x),
        "y": TensorSpec("y", x.shape, DType.FLOAT32, TensorKind.OUTPUT, is_final_output=True),
    }
    plan = ExecutionPlan(tensors=tensors, steps=[step], inputs=[], outputs=["y"])
    plan.add_verification_step("y", "y")
    artifact = compile_plan(plan, {"y": expected.astype(np.float32)})
    source = emit_cv32e40p_program_v2(artifact, {}, program_name="unit_test_rope_v2")
    assert "TNPU_HOST_ROPE" in source
    assert "_host_rope0_inv_freq_bits" in source
    assert ".arr0 =" in source


def test_host_op_silu_emulation_and_v2_emit():
    x = np.array([[-1.0, 0.0, 1.5, 3.0]], dtype=np.float32)
    expected = x / (1.0 + np.exp(-x))

    values = {"x": x, "y": np.zeros_like(x)}
    step = HostOp("silu0", "silu", inputs=["x"], outputs=["y"])
    execute_host_op(step, values, golden=GoldenModel())
    np.testing.assert_allclose(values["y"], expected.astype(np.float32), rtol=1e-5, atol=1e-5)

    tensors = {
        "x": TensorSpec("x", x.shape, DType.FLOAT32, TensorKind.CONSTANT, data=x),
        "y": TensorSpec("y", x.shape, DType.FLOAT32, TensorKind.OUTPUT, is_final_output=True),
    }
    plan = ExecutionPlan(tensors=tensors, steps=[step], inputs=[], outputs=["y"])
    plan.add_verification_step("y", "y")
    artifact = compile_plan(plan, {"y": expected.astype(np.float32)})
    source = emit_cv32e40p_program_v2(artifact, {}, program_name="unit_test_silu_v2")
    assert "TNPU_HOST_SILU" in source


def test_host_op_mul_add_emulation_and_v2_emit():
    x = np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
    y = np.array([[0.5, 2.0, -1.0, -0.5]], dtype=np.float32)
    mul_expected = x * y
    add_expected = x + y

    mul_values = {"x": x, "y": y, "z": np.zeros_like(x)}
    mul_step = HostOp("mul0", "mul", inputs=["x", "y"], outputs=["z"])
    execute_host_op(mul_step, mul_values, golden=GoldenModel())
    np.testing.assert_allclose(mul_values["z"], mul_expected.astype(np.float32), rtol=1e-5, atol=1e-5)

    add_values = {"x": x, "y": y, "z": np.zeros_like(x)}
    add_step = HostOp("add0", "add", inputs=["x", "y"], outputs=["z"])
    execute_host_op(add_step, add_values, golden=GoldenModel())
    np.testing.assert_allclose(add_values["z"], add_expected.astype(np.float32), rtol=1e-5, atol=1e-5)

    tensors = {
        "x": TensorSpec("x", x.shape, DType.FLOAT32, TensorKind.CONSTANT, data=x),
        "y": TensorSpec("y", y.shape, DType.FLOAT32, TensorKind.CONSTANT, data=y),
        "z": TensorSpec("z", x.shape, DType.FLOAT32, TensorKind.OUTPUT, is_final_output=True),
    }

    mul_plan = ExecutionPlan(tensors=tensors, steps=[mul_step], inputs=[], outputs=["z"])
    mul_plan.add_verification_step("z", "z")
    mul_artifact = compile_plan(mul_plan, {"z": mul_expected.astype(np.float32)})
    mul_source = emit_cv32e40p_program_v2(mul_artifact, {}, program_name="unit_test_mul_v2")
    assert "TNPU_HOST_MUL" in mul_source
    assert ".input1_idx =" in mul_source

    add_plan = ExecutionPlan(tensors=tensors, steps=[add_step], inputs=[], outputs=["z"])
    add_plan.add_verification_step("z", "z")
    add_artifact = compile_plan(add_plan, {"z": add_expected.astype(np.float32)})
    add_source = emit_cv32e40p_program_v2(add_artifact, {}, program_name="unit_test_add_v2")
    assert "TNPU_HOST_ADD" in add_source


def test_host_op_causal_mask_emulation_and_emitters():
    x = np.array(
        [
            [1, 2, 3, 4],
            [5, 6, 7, 8],
            [9, 10, 11, 12],
            [13, 14, 15, 16],
        ],
        dtype=np.int16,
    )
    expected = np.array(
        [
            [1, np.iinfo(np.int16).min, np.iinfo(np.int16).min, np.iinfo(np.int16).min],
            [5, 6, np.iinfo(np.int16).min, np.iinfo(np.int16).min],
            [9, 10, 11, np.iinfo(np.int16).min],
            [13, 14, 15, 16],
        ],
        dtype=np.int32,
    )

    values = {"x": x, "y": np.zeros_like(x, dtype=np.int32)}
    step = HostOp("mask0", "causal_mask", inputs=["x"], outputs=["y"], attrs={"fill_value": float(np.iinfo(np.int16).min)})
    execute_host_op(step, values, golden=GoldenModel())
    np.testing.assert_array_equal(values["y"], expected)

    tensors = {
        "x": TensorSpec("x", x.shape, DType.INT16, TensorKind.CONSTANT, data=x),
        "y": TensorSpec("y", x.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    plan = ExecutionPlan(tensors=tensors, steps=[step], inputs=[], outputs=["y"])
    plan.add_verification_step("y", "y")
    artifact = compile_plan(plan, {"y": expected.astype(np.int16)})
    source_v2 = emit_cv32e40p_program_v2(artifact, {}, program_name="unit_test_causal_mask_v2")
    source_cpu = emit_cv32e40p_c(artifact, {}, program_name="unit_test_causal_mask_cpu", cpu_only_baseline=True)
    assert "TNPU_HOST_CAUSAL_MASK" in source_v2
    assert "host_causal_mask(" in source_cpu


def test_host_op_concat_lastdim2_emulation_and_emitters():
    lhs = np.array([[1, 2], [3, 4]], dtype=np.int16)
    rhs = np.array([[5, 6, 7], [8, 9, 10]], dtype=np.int16)
    expected = np.array([[1, 2, 5, 6, 7], [3, 4, 8, 9, 10]], dtype=np.int32)

    values = {
        "lhs": lhs,
        "rhs": rhs,
        "y": np.zeros(expected.shape, dtype=np.int32),
    }
    step = HostOp("cat0", "concat_lastdim2", inputs=["lhs", "rhs"], outputs=["y"])
    execute_host_op(step, values, golden=GoldenModel())
    np.testing.assert_array_equal(values["y"], expected)

    tensors = {
        "lhs": TensorSpec("lhs", lhs.shape, DType.INT16, TensorKind.CONSTANT, data=lhs),
        "rhs": TensorSpec("rhs", rhs.shape, DType.INT16, TensorKind.CONSTANT, data=rhs),
        "y": TensorSpec("y", expected.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    plan = ExecutionPlan(tensors=tensors, steps=[step], inputs=[], outputs=["y"])
    plan.add_verification_step("y", "y")
    artifact = compile_plan(plan, {"y": expected.astype(np.int16)})
    source_v2 = emit_cv32e40p_program_v2(artifact, {}, program_name="unit_test_concat_lastdim2_v2")
    source_cpu = emit_cv32e40p_c(artifact, {}, program_name="unit_test_concat_lastdim2_cpu", cpu_only_baseline=True)
    assert "TNPU_HOST_CONCAT_LASTDIM2" in source_v2
    assert "host_concat_lastdim2(" in source_cpu
    assert ".input1_idx =" in source_v2


def test_prefill_transformer_block_builds_and_emits_new_host_ops():
    artifact, expected = build_prefill_transformer_block_artifact(
        d_model=16,
        d_head=8,
        n_heads=2,
        ffn_dim=8,
        token_count=8,
        seed=0,
    )

    assert "seg_q" in artifact.segment_artifacts
    assert "seg_kv_cache" in artifact.segment_artifacts
    assert "seg_ffn_proj" in artifact.segment_artifacts
    assert artifact.plan.outputs == ["out"]
    assert expected.shape == (8, 16)

    source_v2 = emit_cv32e40p_program_v2(artifact, {}, program_name="unit_test_prefill_transformer_block_v2")
    source_cpu = emit_cv32e40p_c(artifact, {}, program_name="unit_test_prefill_transformer_block_cpu", cpu_only_baseline=True)
    assert "TNPU_HOST_CAUSAL_MASK" in source_v2
    assert "TNPU_HOST_CONCAT_LASTDIM2" in source_v2
    assert "TNPU_HOST_LAYERNORM" in source_v2
    assert "host_causal_mask(" in source_cpu
    assert "host_concat_lastdim2(" in source_cpu
    assert "host_layernorm(" in source_cpu
    result = run_host_emulation(artifact, {}, verification=VerificationMode.FINAL)
    assert "prefill_transformer_block_out" in result.verified


# ---------------------------------------------------------------------------
# XFORM ROPE_K16
# ---------------------------------------------------------------------------

def _rope_rotate_half_q14(k_q14: np.ndarray, cs_q14: np.ndarray) -> np.ndarray:
    """Reference Python impl of rotate-half RoPE in INT16 Q14 arithmetic."""
    k = k_q14.astype(np.int32).reshape(-1)
    cs = cs_q14.astype(np.int32).reshape(-1)
    d = len(k)
    half = d // 2
    k_lo = k[:half]
    k_hi = k[half:]
    cos_q = cs[:half]
    sin_q = cs[half:]
    lo_rot = np.clip((k_lo * cos_q - k_hi * sin_q + (1 << 13)) >> 14, -32768, 32767)
    hi_rot = np.clip((k_hi * cos_q + k_lo * sin_q + (1 << 13)) >> 14, -32768, 32767)
    return np.concatenate([lo_rot, hi_rot]).astype(np.int16)


def test_make_rope_cos_sin_table_q14_shape_and_dtype():
    """make_rope_cos_sin_table_q14 returns an int16 array of length d_head."""
    for d_head in (8, 16, 64):
        table = make_rope_cos_sin_table_q14(d_head, position=3)
        assert table.shape == (d_head,), f"bad shape for d_head={d_head}"
        assert table.dtype == np.int16


def test_rope_cos_sin_table_values():
    """cos and sin halves should be cos(pos*freq) and sin(pos*freq) in Q14."""
    d_head = 16
    position = 5
    theta = 10000.0
    table = make_rope_cos_sin_table_q14(d_head, position=position, theta=theta)
    half = d_head // 2
    Q14 = 16384.0
    for i in range(half):
        freq = 1.0 / (theta ** (2.0 * i / d_head))
        angle = position * freq
        expected_cos = int(round(np.cos(angle) * Q14))
        expected_sin = int(round(np.sin(angle) * Q14))
        assert int(table[i]) == expected_cos, f"cos mismatch at i={i}"
        assert int(table[half + i]) == expected_sin, f"sin mismatch at i={i}"


def test_npu_program_xform_rope_k16_isa_encoding():
    """TinyNPUProgram.xform_rope_k16 produces a XFORM ROPE_K16 instruction."""
    from tinynpu.isa import XformMode, Opcode

    d_head = 16
    k_data = np.ones((1, d_head), dtype=np.int16) * 100
    cs_data = make_rope_cos_sin_table_q14(d_head, position=3).reshape(1, d_head)

    prog = TinyNPUProgram()
    prog.declare_data("k", k_data, role="C")
    prog.declare_data("cs", cs_data, role="C")
    prog.xform_rope_k16("k", "cs")
    prog.halt()
    binary = prog.compile()

    # The XFORM ROPE_K16 instruction should be the first instruction.
    instr = binary["im"][0]
    opcode = (instr >> 252) & 0xF
    mode = (instr >> 248) & 0xF
    assert opcode == int(Opcode.XFORM), f"expected XFORM opcode, got {opcode}"
    assert mode == int(XformMode.ROPE_K16), f"expected ROPE_K16 mode, got {mode}"

    # half_count should be d_head / 16 = 1 for d_head=16
    half_count = (instr >> 200) & 0xFFFF
    assert half_count == d_head // 16, f"half_count={half_count}, expected {d_head // 16}"


def test_npu_segment_with_rope_cs_python_simulation():
    """End-to-end test: matmul producing K, then XFORM ROPE_K16, using host emulation."""
    d_head = 16
    d_model = 8  # small for test speed
    rng = np.random.default_rng(42)

    x = rng.integers(-4, 5, size=(1, d_model), dtype=np.int16)
    w_k = rng.integers(-4, 5, size=(d_model, d_head), dtype=np.int16)
    position = 3
    theta = 10000.0

    # Expected: K = x @ w_k, then ROPE
    k_ref = np.clip(x.astype(np.int32) @ w_k.astype(np.int32), -32768, 32767).astype(np.int16)
    cs_ref = make_rope_cos_sin_table_q14(d_head, position, theta)
    k_rope_ref = _rope_rotate_half_q14(k_ref, cs_ref)

    cs_tensor_spec = make_rope_cs_tensor_spec("k_rope_cs", d_head, position, theta)

    tensors = {
        "x": TensorSpec("x", x.shape, DType.INT16, TensorKind.CONSTANT, data=x),
        "w_k": TensorSpec("w_k", w_k.shape, DType.INT16, TensorKind.CONSTANT, data=w_k),
        "k_out": TensorSpec("k_out", (1, d_head), DType.INT16, TensorKind.OUTPUT, is_final_output=True),
        "k_rope_cs": cs_tensor_spec,
    }

    seg = NpuSegment(
        "seg_k",
        [MatMulOp("op_k", "x", "w_k", "k_out", in_dtype=DType.INT16, out_dtype=DType.INT16,
                  rope_cs_name="k_rope_cs")],
        inputs=["x", "w_k"],
        outputs=["k_out"],
    )
    plan = ExecutionPlan(tensors=tensors, steps=[seg], inputs=["x"], outputs=["k_out"])
    plan.add_verification_step("k_out", "k_out")
    artifact = compile_plan(plan, {"k_out": k_rope_ref})

    seg_artifact = artifact.segment_artifacts["seg_k"]
    k_addr = int(seg_artifact.symbol_table["k_out"]["addr"])
    cs_addr = int(seg_artifact.symbol_table["k_rope_cs"]["addr"])
    w_addr = int(seg_artifact.symbol_table["w_k"]["addr"])

    assert cs_addr != w_addr, "RoPE cos/sin table must not alias the weight tensor in UB"
    assert cs_addr != k_addr, "RoPE cos/sin table must not alias the K output tensor in UB"

    result = run_host_emulation(artifact, {"x": x})
    k_sim = np.asarray(result.tensors["k_out"]).astype(np.int16).reshape(-1)

    # Allow ±1 rounding tolerance due to Q14 integer arithmetic
    diff = np.abs(k_sim.astype(np.int32) - k_rope_ref.astype(np.int32))
    assert diff.max() <= 1, (
        f"ROPE result mismatch: max diff={diff.max()}\n"
        f"  sim:      {k_sim}\n"
        f"  expected: {k_rope_ref}"
    )


def test_compile_plan_rewrites_host_rope_chain_to_xform():
    d_model = 8
    d_head = 16
    position = 3
    theta = 10000.0
    rng = np.random.default_rng(7)

    x = rng.integers(-4, 5, size=(1, d_model), dtype=np.int16)
    w_q = rng.integers(-4, 5, size=(d_model, d_head), dtype=np.int16)

    q_ref = np.clip(x.astype(np.int32) @ w_q.astype(np.int32), -32768, 32767).astype(np.int16)
    cs_ref = make_rope_cos_sin_table_q14(d_head, position, theta)
    q_rope_ref = _rope_rotate_half_q14(q_ref, cs_ref)

    tensors = {
        "x": TensorSpec("x", x.shape, DType.INT16, TensorKind.INPUT),
        "w_q": TensorSpec("w_q", w_q.shape, DType.INT16, TensorKind.CONSTANT, data=w_q),
        "q_int": TensorSpec("q_int", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE),
        "q_f": TensorSpec("q_f", (1, d_head), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "q_rope_f": TensorSpec("q_rope_f", (1, d_head), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "q_rope_q": TensorSpec("q_rope_q", (1, d_head), DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        NpuSegment("seg_q", [MatMulOp("op_q", "x", "w_q", "q_int")], inputs=["x", "w_q"], outputs=["q_int"]),
        HostOp("dequant_q", "dequantize", inputs=["q_int"], outputs=["q_f"], attrs={"scale": 1.0 / 32.0, "zero_point": 0}),
        HostOp("rope_q", "rope", inputs=["q_f"], outputs=["q_rope_f"], attrs={"head_dim": d_head, "position": position, "theta": theta}),
        HostOp(
            "quant_q_rope",
            "quantize",
            inputs=["q_rope_f"],
            outputs=["q_rope_q"],
            attrs={"scale": 1.0 / 32.0, "zero_point": 0, "dtype": DType.INT16},
        ),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["x"], outputs=["q_rope_q"])
    plan.add_verification_step("q_rope_q", "q_rope_q")

    artifact = compile_plan(plan, {"q_rope_q": q_rope_ref})

    seg = artifact.segment_artifacts["seg_q"]
    assert artifact.plan.steps[0].ops[0].out == "q_rope_q"
    assert artifact.plan.steps[0].ops[0].rope_cs_name == "q_rope_q__rope_cs"
    assert "q_rope_q__rope_cs" in artifact.plan.tensors
    assert all(not (isinstance(step, HostOp) and step.name in {"dequant_q", "rope_q", "quant_q_rope"}) for step in artifact.plan.steps)
    assert "q_rope_q__rope_cs" in seg.symbol_table

    result = run_host_emulation(artifact, {"x": x}, verification=VerificationMode.OFF)
    assert "q_rope_q" in result.tensors
    diff = np.abs(np.asarray(result.tensors["q_rope_q"]).astype(np.int32).reshape(-1) - q_rope_ref.astype(np.int32).reshape(-1))
    assert diff.max() <= 1
