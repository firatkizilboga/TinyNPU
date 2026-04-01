from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "software" / "compiler"))

from tinynpu_jit import (  # noqa: E402
    DType,
    ExecutionPlan,
    HostOp,
    MatMulOp,
    NpuSegment,
    TensorKind,
    TensorSpec,
    VerificationMode,
    compile_plan,
    run_host_emulation,
    write_cv32e40p_c,
)


def build_demo() -> tuple[object, dict[str, np.ndarray], np.ndarray]:
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
        NpuSegment(
            "segment_000",
            [MatMulOp("op0", "x", "w0", "h")],
            inputs=["x", "w0"],
            outputs=["h"],
        ),
        HostOp("relu_h", "relu", inputs=["h"], outputs=["h_relu"]),
        NpuSegment(
            "segment_001",
            [MatMulOp("op1", "h_relu", "w1", "y")],
            inputs=["h_relu", "w1"],
            outputs=["y"],
        ),
    ]

    plan = ExecutionPlan(
        tensors=tensors,
        steps=steps,
        inputs=["x"],
        outputs=["y"],
        metadata={"frontend": "manual_demo"},
    )
    plan.add_verification_step("y", "final_y")

    lhs0 = x.astype(np.int32) @ w0.astype(np.int32)
    h = np.clip(lhs0, -32768, 32767).astype(np.int16)
    h_relu = np.maximum(h, 0).astype(np.int16)
    lhs1 = h_relu.astype(np.int32) @ w1.astype(np.int32)
    y = np.clip(lhs1, -32768, 32767).astype(np.int16)

    artifact = compile_plan(plan, {"y": y})
    inputs = {"x": x}
    return artifact, inputs, y


def main() -> None:
    artifact, inputs, expected = build_demo()
    generated_dir = ROOT / "generated"
    generated_dir.mkdir(exist_ok=True)
    output_path = generated_dir / "cv32e40p_relu_chain_demo.c"
    write_cv32e40p_c(
        artifact,
        inputs,
        output_path,
        program_name="cv32e40p_relu_chain_demo",
    )
    host_result = run_host_emulation(
        artifact,
        inputs,
        verification=VerificationMode.FINAL,
    )
    if not np.array_equal(host_result.tensors["y"], expected):
        raise AssertionError("Host-emulation result does not match expected demo output.")
    print(output_path)


if __name__ == "__main__":
    main()
