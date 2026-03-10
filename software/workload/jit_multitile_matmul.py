from __future__ import annotations

import numpy as np

from software.compiler.tinynpu_jit import (
    DType,
    ExecutionPlan,
    MatMulOp,
    NpuSegment,
    TensorKind,
    TensorSpec,
    VerificationMode,
    VerifyTensor,
    compile_plan,
    run_host_emulation,
)


def build_multitile_matmul_artifact():
    """
    Build a single-segment matmul whose logical M/K/N all exceed ARRAY_SIZE=8.

    Shape choice:
    - lhs: 13 x 17
    - rhs: 17 x 19
    - out: 13 x 19

    This forces:
    - M tiles > 1
    - K tiles > 1
    - N tiles > 1
    """
    lhs = (
        (np.arange(13 * 17, dtype=np.int32).reshape(13, 17) % 9) - 4
    ).astype(np.int16)
    rhs = (
        ((np.arange(17 * 19, dtype=np.int32).reshape(17, 19) * 3) % 11) - 5
    ).astype(np.int16)
    expected = lhs.astype(np.int32) @ rhs.astype(np.int32)

    tensors = {
        "lhs": TensorSpec("lhs", lhs.shape, DType.INT16, TensorKind.CONSTANT, data=lhs),
        "rhs": TensorSpec("rhs", rhs.shape, DType.INT16, TensorKind.CONSTANT, data=rhs),
        "out": TensorSpec(
            "out",
            expected.shape,
            DType.INT16,
            TensorKind.OUTPUT,
            is_final_output=True,
        ),
    }
    plan = ExecutionPlan(
        tensors=tensors,
        steps=[
            NpuSegment(
                "segment_000",
                [MatMulOp("mm_multitile", "lhs", "rhs", "out")],
                ["lhs", "rhs"],
                ["out"],
            ),
            VerifyTensor("out", "out", is_final_output=True),
        ],
        inputs=[],
        outputs=["out"],
        metadata={
            "shape": {
                "lhs": list(lhs.shape),
                "rhs": list(rhs.shape),
                "out": list(expected.shape),
            }
        },
    )
    return compile_plan(plan, {"out": expected}), expected


def smoke_run_multitile_matmul():
    artifact, expected = build_multitile_matmul_artifact()
    result = run_host_emulation(artifact, {}, VerificationMode.DEBUG, debug=True)
    output = result.tensors[artifact.plan.outputs[0]]
    return {
        "expected": expected,
        "output": output,
        "verified": result.verified,
        "debug_kinds": [event["kind"] for event in result.debug_trace],
    }


if __name__ == "__main__":
    summary = smoke_run_multitile_matmul()
    print("verified", summary["verified"])
    print("output_shape", list(summary["output"].shape))
    print("debug_kinds", summary["debug_kinds"])
