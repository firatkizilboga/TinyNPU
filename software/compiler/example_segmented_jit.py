import numpy as np

from tinynpu_jit import (
    DType,
    ExecutionPlan,
    HostOp,
    MatMulOp,
    NpuSegment,
    TensorKind,
    TensorSpec,
    VerificationMode,
    VerifyTensor,
    compile_plan,
)


def build_demo_artifact():
    tensors = {
        "weight": TensorSpec(
            "weight",
            (2, 2),
            DType.INT16,
            TensorKind.CONSTANT,
            data=np.array([[1, 0], [0, 1]], dtype=np.int16),
        ),
        "input": TensorSpec("input", (2, 2), DType.INT16, TensorKind.INPUT),
        "scores": TensorSpec(
            "scores",
            (2, 2),
            DType.INT16,
            TensorKind.INTERMEDIATE,
            verify_label="scores",
        ),
        "probs": TensorSpec(
            "probs",
            (2, 2),
            DType.FLOAT32,
            TensorKind.OUTPUT,
            is_final_output=True,
        ),
    }

    plan = ExecutionPlan(
        tensors=tensors,
        steps=[
            NpuSegment(
                "segment_000",
                [MatMulOp("mm0", "weight", "input", "scores")],
                ["weight", "input"],
                ["scores"],
            ),
            VerifyTensor("scores", "scores"),
            HostOp("softmax0", "softmax", ["scores"], ["probs"], {"axis": 0}),
            VerifyTensor("probs", "probs", is_final_output=True),
        ],
        inputs=["input"],
        outputs=["probs"],
    )

    expected = {
        "scores": np.array([[1, 3], [2, 4]], dtype=np.int32),
        "probs": np.array(
            [[0.2689414, 0.2689414], [0.7310586, 0.7310586]],
            dtype=np.float32,
        ),
    }
    return compile_plan(plan, expected)


if __name__ == "__main__":
    artifact = build_demo_artifact()
    result = artifact.run_host_emulation(
        {"input": np.array([[1, 3], [2, 4]], dtype=np.int16)},
        verification=VerificationMode.DEBUG,
    )
    print("Verified:", result.verified)
    print("Output:\n", result.tensors["probs"])
