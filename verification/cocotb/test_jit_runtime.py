import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

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
    run_sim,
)


def build_artifact():
    tensors = {
        'w1': TensorSpec('w1', (2, 2), DType.INT16, TensorKind.CONSTANT, data=np.array([[1, 2], [0, 1]], dtype=np.int16)),
        'x': TensorSpec('x', (2, 1), DType.INT16, TensorKind.INPUT),
        'h': TensorSpec('h', (2, 1), DType.INT16, TensorKind.INTERMEDIATE, verify_label='hidden'),
        'w2': TensorSpec('w2', (2, 2), DType.INT16, TensorKind.CONSTANT, data=np.array([[2, 0], [1, 1]], dtype=np.int16)),
        'y': TensorSpec('y', (2, 1), DType.INT16, TensorKind.OUTPUT, is_final_output=True),
    }
    plan = ExecutionPlan(
        tensors=tensors,
        steps=[
            NpuSegment('segment_000', [MatMulOp('mm0', 'w1', 'x', 'h')], ['w1', 'x'], ['h']),
            VerifyTensor('h', 'hidden'),
            NpuSegment('segment_001', [MatMulOp('mm1', 'w2', 'h', 'y')], ['w2', 'h'], ['y']),
            VerifyTensor('y', 'y', is_final_output=True),
        ],
        inputs=['x'],
        outputs=['y'],
    )
    expected = {
        'h': np.array([[5], [2]], dtype=np.int32),
        'y': np.array([[10], [7]], dtype=np.int32),
    }
    return compile_plan(plan, expected)


@cocotb.test()
async def test_jit_runtime_two_segments(dut):
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1

    artifact = build_artifact()
    result = await run_sim(
        artifact,
        {'x': np.array([[1], [2]], dtype=np.int16)},
        dut=dut,
        verification=VerificationMode.DEBUG,
    )

    assert result.verified == ['hidden', 'y']
    assert np.array_equal(result.tensors['y'], np.array([[10], [7]], dtype=np.int32))
