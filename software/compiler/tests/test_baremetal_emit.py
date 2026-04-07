import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tinynpu_jit import emit_cv32e40p_c, emit_cv32e40p_program_v2
from tinynpu_jit import DType, ExecutionPlan, HostOp, MatMulOp, NpuSegment, TensorKind, TensorSpec, compile_plan


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
    assert "TNPU_OP_SEGMENT" in source
    assert "TNPU_OP_HOST" in source
    assert "TNPU_OP_VERIFY" in source
    assert "TNPU_HOST_RELU" in source


def test_compile_plan_fuses_layout_restore_into_matrix_im2col():
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
    assert "layout_restore" not in kinds
    if "im2col" in kinds:
        fused_im2col = next(step for step in artifact.plan.steps if isinstance(step, HostOp) and step.kind == "im2col")
        assert fused_im2col.inputs == ["mat"]
        assert fused_im2col.attrs["input_layout"] == "matrix_hwc"
        assert fused_im2col.attrs["matrix_h"] == 2
        assert fused_im2col.attrs["matrix_w"] == 2
        assert fused_im2col.attrs["matrix_c"] == 3

        source_v1 = emit_cv32e40p_c(artifact, {}, program_name="unit_test_matrix_hwc")
        assert "host_im2col_matrix(&cols, &mat, 2, 2, 3, 2, 1, 0);" in source_v1

        source_v2 = emit_cv32e40p_program_v2(artifact, {}, program_name="unit_test_matrix_hwc_v2")
        assert ".kind = TNPU_HOST_IM2COL" in source_v2
        assert ".attrs_i32 = {2, 1, 0, 2, 2, 2, 3, 0}" in source_v2
    else:
        seg = next(step for step in artifact.plan.steps if isinstance(step, NpuSegment) and step.name == "seg1")
        assert seg.ops[0].lhs == "mat"
        assert seg.ops[0].conv_stream is not None
        assert seg.ops[0].conv_stream["input_h"] == 2
        assert seg.ops[0].conv_stream["input_w"] == 2
        assert seg.ops[0].conv_stream["input_c"] == 3


def test_compile_plan_converts_matrix_im2col_to_conv_stream():
    xmat = np.arange(16 * 8, dtype=np.int16).reshape(16, 8)
    cols = np.zeros((9, 32), dtype=np.int16)
    w = np.ones((32, 4), dtype=np.int16)
    y = np.zeros((9, 4), dtype=np.int16)

    tensors = {
        "xmat": TensorSpec("xmat", xmat.shape, DType.INT16, TensorKind.INPUT),
        "cols": TensorSpec("cols", cols.shape, DType.INT16, TensorKind.INTERMEDIATE),
        "w": TensorSpec("w", w.shape, DType.INT16, TensorKind.CONSTANT, data=w),
        "y": TensorSpec("y", y.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    steps = [
        HostOp(
            "conv2_im2col",
            "im2col",
            inputs=["xmat"],
            outputs=["cols"],
            attrs={
                "kernel_size": 2,
                "stride": 1,
                "padding": 0,
                "input_layout": "matrix_hwc",
                "matrix_h": 4,
                "matrix_w": 4,
                "matrix_c": 8,
            },
        ),
        NpuSegment(
            "seg_conv",
            [MatMulOp("op0", "cols", "w", "y", in_dtype=DType.INT16, out_dtype=DType.INT16)],
            inputs=["cols", "w"],
            outputs=["y"],
        ),
    ]
    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=["xmat"], outputs=["y"])
    artifact = compile_plan(plan, {"y": y})

    host_kinds = [step.kind for step in artifact.plan.steps if isinstance(step, HostOp)]
    assert "im2col" not in host_kinds

    seg = next(step for step in artifact.plan.steps if isinstance(step, NpuSegment))
    assert seg.ops[0].lhs == "xmat"
    assert seg.ops[0].conv_stream is not None
    assert seg.ops[0].conv_stream["input_h"] == 4
    assert seg.ops[0].conv_stream["input_w"] == 4
    assert seg.ops[0].conv_stream["input_c"] == 8
    assert seg.ops[0].conv_stream["kernel_size"] == 2
    first_instr = artifact.segment_artifacts["seg_conv"].binary["im"][0]
    assert ((first_instr >> 71) & 0x1) == 1

    source = emit_cv32e40p_c(artifact, {"xmat": xmat}, program_name="unit_test_conv_stream")
    assert "HostOp im2col" not in source


def test_compile_plan_converts_matrix_im2col_to_conv_stream_int8_with_padding():
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

    host_kinds = [step.kind for step in artifact.plan.steps if isinstance(step, HostOp)]
    assert "im2col" not in host_kinds

    seg = next(step for step in artifact.plan.steps if isinstance(step, NpuSegment))
    assert seg.ops[0].lhs == "xmat"
    assert seg.ops[0].conv_stream is not None
    assert seg.ops[0].conv_stream["padding"] == 1
    assert seg.ops[0].conv_stream["input_c"] == 7


def test_compile_plan_converts_matrix_im2col_to_conv_stream_int4_irregular_channels():
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

    host_kinds = [step.kind for step in artifact.plan.steps if isinstance(step, HostOp)]
    assert "im2col" not in host_kinds

    seg = next(step for step in artifact.plan.steps if isinstance(step, NpuSegment))
    assert seg.ops[0].lhs == "xmat"
    assert seg.ops[0].conv_stream is not None
    assert seg.ops[0].conv_stream["input_c"] == 10
