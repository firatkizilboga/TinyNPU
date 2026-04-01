import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from software.compiler.tinynpu_jit import (  # noqa: E402
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
    run_host_emulation,
    run_sim,
)


def build_artifact():
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

    plan = ExecutionPlan(
        tensors=tensors,
        steps=[
            NpuSegment("segment_000", [MatMulOp("mm0", "x", "w0", "h")], ["x", "w0"], ["h"]),
            HostOp("relu_h", "relu", inputs=["h"], outputs=["h_relu"]),
            NpuSegment("segment_001", [MatMulOp("mm1", "h_relu", "w1", "y")], ["h_relu", "w1"], ["y"]),
            VerifyTensor("y", "final_y", is_final_output=True),
        ],
        inputs=["x"],
        outputs=["y"],
    )

    h = np.clip(x.astype(np.int32) @ w0.astype(np.int32), -32768, 32767).astype(np.int16)
    h_relu = np.maximum(h, 0).astype(np.int16)
    y = np.clip(h_relu.astype(np.int32) @ w1.astype(np.int32), -32768, 32767).astype(np.int16)
    artifact = compile_plan(plan, {"y": y})
    return artifact, {"x": x}, y


@cocotb.test()
async def test_jit_relu_chain_runtime(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1

    artifact, inputs, expected_y = build_artifact()

    host_result = run_host_emulation(
        artifact,
        inputs,
        VerificationMode.DEBUG,
    )
    rtl_result = await run_sim(
        artifact,
        inputs,
        dut=dut,
        verification=VerificationMode.DEBUG,
    )

    final_name = artifact.plan.outputs[0]
    assert np.array_equal(host_result.tensors[final_name], expected_y)
    assert np.array_equal(rtl_result.tensors[final_name], expected_y)

