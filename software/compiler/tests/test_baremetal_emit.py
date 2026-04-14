import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tinynpu_jit import emit_cv32e40p_c, emit_cv32e40p_program_v2
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
    b_slot_word_stride,
    compile_plan,
    make_b_cache_specs,
    make_kv_cache_specs,
)
from tinynpu import TinyNPUProgram
from tinynpu.isa import OutputLayout, PrecisionMode


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
    assert ((inst0 >> 72) & 0x3) == int(OutputLayout.B)
    assert ((inst1 >> 72) & 0x3) == int(OutputLayout.B)
    assert seg.symbol_table["cache_t1"]["base_name"] == "cache"
    assert seg.symbol_table["cache_t1"]["word_offset"] == 8
    memory_names = {entry.name for entry in seg.memory_plan.entries}
    assert "cache" in memory_names
    assert "cache_t0" not in memory_names
    assert "cache_t1" not in memory_names
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
    assert specs["k_cache_t1"].metadata["cache_kind"] == "K"
    assert specs["v_cache_t1"].metadata["cache_kind"] == "V"
    assert specs["k_cache_t1"].metadata["storage_word_offset"] == 8
    assert specs["v_cache_t1"].metadata["storage_word_offset"] == 8


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
