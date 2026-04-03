import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tinynpu_jit import emit_cv32e40p_c
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
