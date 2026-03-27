import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tinynpu_jit import (
    DType,
    ExecutionPlan,
    HostOp,
    MatMulOp,
    NpuSegment,
    TensorKind,
    TensorSpec,
    compile_plan,
)


def test_manual_attention_block_lowers_via_host_and_npu_segments():
    tensors = {
        "q": TensorSpec("q", (2, 4), DType.INT16, TensorKind.INPUT),
        "k": TensorSpec(
            "k",
            (2, 4),
            DType.INT16,
            TensorKind.CONSTANT,
            data=np.array([[2, 1, 0, -1], [1, 0, 1, 2]], dtype=np.int16),
        ),
        "v": TensorSpec(
            "v",
            (2, 4),
            DType.INT16,
            TensorKind.CONSTANT,
            data=np.array([[4, 1, -2, 3], [0, 2, 1, -1]], dtype=np.int16),
        ),
        "k_t": TensorSpec("k_t", (4, 2), DType.INT16, TensorKind.INTERMEDIATE),
        "scores": TensorSpec("scores", (2, 2), DType.INT16, TensorKind.INTERMEDIATE),
        "scores_scaled": TensorSpec("scores_scaled", (2, 2), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "attn_probs": TensorSpec("attn_probs", (2, 2), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "attn_q": TensorSpec("attn_q", (2, 2), DType.INT8, TensorKind.INTERMEDIATE),
        "out": TensorSpec("out", (2, 4), DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    plan = ExecutionPlan(
        tensors=tensors,
        steps=[
            HostOp("k_t0", "transpose", ["k"], ["k_t"], {"axes": (1, 0)}),
            NpuSegment("qk", [MatMulOp("qk_mm", "q", "k_t", "scores")], inputs=["q", "k_t"], outputs=["scores"]),
            HostOp("scale_scores", "scale", ["scores"], ["scores_scaled"], {"scale": 0.5}),
            HostOp("softmax_scores", "softmax", ["scores_scaled"], ["attn_probs"], {"axis": 1}),
            HostOp("quant_attn", "quantize", ["attn_probs"], ["attn_q"], {"scale": 1.0 / 127.0, "dtype": DType.INT8}),
            NpuSegment("pv", [MatMulOp("pv_mm", "attn_q", "v", "out")], inputs=["attn_q", "v"], outputs=["out"]),
        ],
        inputs=["q"],
        outputs=["out"],
    )

    artifact = compile_plan(plan, expected_tensors={})
    q = np.array([[3, 1, 2, 0], [0, 2, 1, 3]], dtype=np.int16)
    result = artifact.run_host_emulation({"q": q})

    assert result.tensors["out"].shape == (2, 4)
    assert result.tensors["out"].dtype == np.int32
    row_sums = result.trace_tensors["attn_probs"].sum(axis=1)
    assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-5)
