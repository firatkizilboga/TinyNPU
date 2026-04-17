import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tinynpu_jit import (
    CompiledArtifact,
    DType,
    ExecutionPlan,
    HostOp,
    HostOpSpec,
    PrimitiveCounts,
    TensorKind,
    TensorSpec,
    compile_plan,
    get_host_op_spec,
    register_host_op,
    registered_host_op_kinds,
)
from tinynpu_jit.benchmark import estimate_host_op_counts
from tinynpu_jit.executor import HostEmulationExecutor


def _artifact_for_host_op(step: HostOp, *, input_dtype: DType = DType.FLOAT32, output_dtype: DType = DType.FLOAT32) -> CompiledArtifact:
    tensors = {
        "x": TensorSpec("x", (2, 2), input_dtype, TensorKind.INPUT),
        "y": TensorSpec("y", (2, 2), output_dtype, TensorKind.OUTPUT, is_final_output=True),
    }
    plan = ExecutionPlan(tensors=tensors, steps=[step], inputs=["x"], outputs=["y"])
    return CompiledArtifact(plan=plan, expected_tensors={}, segment_artifacts={})


def test_builtin_host_op_registry_drives_execution_and_benchmark():
    step = HostOp(name="sigmoid0", kind="sigmoid", inputs=["x"], outputs=["y"])
    artifact = _artifact_for_host_op(step)
    source = np.array([[-2.0, 0.0], [2.0, 4.0]], dtype=np.float32)

    result = HostEmulationExecutor().run(artifact, {"x": source})

    expected = 1.0 / (1.0 + np.exp(-source))
    assert np.allclose(result.tensors["y"], expected)
    assert "sigmoid" in registered_host_op_kinds()
    assert get_host_op_spec("sigmoid").quant_boundary_policy == "passthrough"

    bucket, counts = estimate_host_op_counts(step, {"x": source, "y": result.tensors["y"]})
    assert bucket == "host_intrinsic"
    assert counts.nonlinear == source.size
    assert counts.divs == source.size


def test_register_host_op_extends_dispatch():
    kind = "test_negate"
    if kind not in registered_host_op_kinds():
        register_host_op(
            HostOpSpec(
                kind=kind,
                evaluator=lambda step, values, golden: values.__setitem__(
                    step.outputs[0], -np.array(values[step.inputs[0]], copy=False)
                ),
                benchmark=lambda step, values: (
                    "host_intrinsic",
                    PrimitiveCounts(
                        reads=int(np.array(values[step.inputs[0]], copy=False).size),
                        writes=int(np.array(values[step.outputs[0]], copy=False).size),
                    ),
                ),
            )
        )

    step = HostOp(name="neg0", kind=kind, inputs=["x"], outputs=["y"])
    artifact = _artifact_for_host_op(step, input_dtype=DType.INT16, output_dtype=DType.INT16)
    source = np.array([[1, -2], [3, -4]], dtype=np.int16)

    result = HostEmulationExecutor().run(artifact, {"x": source})

    assert np.array_equal(result.tensors["y"], -source)
    bucket, counts = estimate_host_op_counts(step, {"x": source, "y": result.tensors["y"]})
    assert bucket == "host_intrinsic"
    assert counts.reads == source.size
    assert counts.writes == source.size


def test_unknown_host_op_fails_closed():
    step = HostOp(name="mystery0", kind="mystery", inputs=["x"], outputs=["y"])
    plan = _artifact_for_host_op(step).plan

    with pytest.raises(NotImplementedError, match="Unsupported host op 'mystery'"):
        compile_plan(plan, expected_tensors={})


def test_quantize_validation_fails_on_non_positive_scale():
    step = HostOp(name="q0", kind="quantize", inputs=["x"], outputs=["y"], attrs={"scale": 0.0})
    plan = _artifact_for_host_op(step).plan

    with pytest.raises(ValueError, match="scale > 0"):
        compile_plan(plan, expected_tensors={})


def test_mean_validation_requires_input_quantization_scale():
    step = HostOp(
        name="mean0",
        kind="mean",
        inputs=["x"],
        outputs=["y"],
        attrs={"input_quantization": {"zero_point": 0}},
    )
    plan = _artifact_for_host_op(step).plan

    with pytest.raises(ValueError, match="missing required attr 'scale'"):
        compile_plan(plan, expected_tensors={})


def test_transpose_validation_rejects_duplicate_axes():
    step = HostOp(name="t0", kind="transpose", inputs=["x"], outputs=["y"], attrs={"axes": (0, 0)})
    plan = _artifact_for_host_op(step).plan

    with pytest.raises(ValueError, match="axes must be unique"):
        compile_plan(plan, expected_tensors={})


def test_mean_benchmark_counts_embedded_dequantization_work():
    step = HostOp(
        name="mean0",
        kind="mean",
        inputs=["x"],
        outputs=["y"],
        attrs={"dim": [0], "input_quantization": {"scale": 0.25, "zero_point": 3}},
    )
    source = np.array([[1, 2], [3, 4]], dtype=np.int8)
    output = np.array([1.5, 2.5], dtype=np.float32)

    bucket, counts = estimate_host_op_counts(step, {"x": source, "y": output})

    assert bucket == "host_intrinsic"
    assert counts.reads == source.size * 2
    assert counts.muls == source.size
    assert counts.adds == source.size * 3


def test_causal_mask_execution_and_validation():
    source = np.array(
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

    step = HostOp("mask0", "causal_mask", inputs=["x"], outputs=["y"])
    artifact = _artifact_for_host_op(step, input_dtype=DType.INT16, output_dtype=DType.INT16)
    artifact.plan.tensors["x"] = TensorSpec("x", source.shape, DType.INT16, TensorKind.INPUT)
    artifact.plan.tensors["y"] = TensorSpec("y", source.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True)

    result = HostEmulationExecutor().run(artifact, {"x": source})
    np.testing.assert_array_equal(result.tensors["y"], expected)

    with pytest.raises(ValueError, match="past_kv_len >= 0"):
        compile_plan(
            ExecutionPlan(
                tensors={
                    "x": TensorSpec("x", source.shape, DType.INT16, TensorKind.INPUT),
                    "y": TensorSpec("y", source.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
                },
                steps=[HostOp("mask_bad", "causal_mask", inputs=["x"], outputs=["y"], attrs={"past_kv_len": -1})],
                inputs=["x"],
                outputs=["y"],
            ),
            expected_tensors={},
        )


def test_concat_lastdim2_execution():
    lhs = np.array([[1, 2], [3, 4]], dtype=np.int16)
    rhs = np.array([[5, 6, 7], [8, 9, 10]], dtype=np.int16)
    expected = np.array([[1, 2, 5, 6, 7], [3, 4, 8, 9, 10]], dtype=np.int32)

    tensors = {
        "lhs": TensorSpec("lhs", lhs.shape, DType.INT16, TensorKind.INPUT),
        "rhs": TensorSpec("rhs", rhs.shape, DType.INT16, TensorKind.INPUT),
        "y": TensorSpec("y", expected.shape, DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    step = HostOp("cat0", "concat_lastdim2", inputs=["lhs", "rhs"], outputs=["y"])
    artifact = CompiledArtifact(
        plan=ExecutionPlan(tensors=tensors, steps=[step], inputs=["lhs", "rhs"], outputs=["y"]),
        expected_tensors={},
        segment_artifacts={},
    )

    result = HostEmulationExecutor().run(artifact, {"lhs": lhs, "rhs": rhs})
    np.testing.assert_array_equal(result.tensors["y"], expected)


def test_layernorm_execution_and_validation():
    source = np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float32)
    weight = np.array([1.5, 0.5, 2.0, 1.0], dtype=np.float32)
    bias = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    weight_bias = np.stack([weight, bias], axis=0).astype(np.float32)
    mean = np.mean(source, axis=-1, keepdims=True)
    centered = source - mean
    var = np.mean(np.square(centered), axis=-1, keepdims=True)
    expected = ((centered / np.sqrt(var + 1.0e-6)) * weight.reshape(1, -1) + bias.reshape(1, -1)).astype(np.float32)

    tensors = {
        "x": TensorSpec("x", source.shape, DType.FLOAT32, TensorKind.INPUT),
        "wb": TensorSpec("wb", weight_bias.shape, DType.FLOAT32, TensorKind.INPUT),
        "y": TensorSpec("y", source.shape, DType.FLOAT32, TensorKind.OUTPUT, is_final_output=True),
    }
    step = HostOp("ln0", "layernorm", inputs=["x", "wb"], outputs=["y"], attrs={"eps": 1.0e-6})
    artifact = CompiledArtifact(
        plan=ExecutionPlan(tensors=tensors, steps=[step], inputs=["x", "wb"], outputs=["y"]),
        expected_tensors={},
        segment_artifacts={},
        )


def test_slice_row_execution_and_validation():
    source = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int16)
    expected = np.array([[4, 5, 6]], dtype=np.int32)

    tensors = {
        "x": TensorSpec("x", source.shape, DType.INT16, TensorKind.INPUT),
        "y": TensorSpec("y", (1, source.shape[1]), DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    step = HostOp("slice1", "slice_row", inputs=["x"], outputs=["y"], attrs={"row_index": 1})
    artifact = CompiledArtifact(
        plan=ExecutionPlan(tensors=tensors, steps=[step], inputs=["x"], outputs=["y"]),
        expected_tensors={},
        segment_artifacts={},
    )

    result = HostEmulationExecutor().run(artifact, {"x": source})
    np.testing.assert_array_equal(result.tensors["y"], expected)

    with pytest.raises(ValueError, match="row_index >= 0"):
        compile_plan(
            ExecutionPlan(
                tensors=tensors,
                steps=[HostOp("slice_bad", "slice_row", inputs=["x"], outputs=["y"], attrs={"row_index": -1})],
                inputs=["x"],
                outputs=["y"],
            ),
            expected_tensors={},
        )
