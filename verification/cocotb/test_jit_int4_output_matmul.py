import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from software.compiler.tinynpu_jit import DType, VerificationMode, run_host_emulation, run_sim
from software.workload.jit_multitile_matmul import (
    JitMatmulBenchmarkCase,
    build_configured_matmul_artifact,
)


async def _reset(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1


@cocotb.test()
async def test_jit_int4_output_c_layout_matmul(dut):
    await _reset(dut)

    case = JitMatmulBenchmarkCase(
        name="int4_output_c_layout",
        m=25,
        k=17,
        n=19,
        in_dtype=DType.INT16,
        out_dtype=DType.INT4,
        lhs_runtime_input=False,
        seed=41,
    )
    artifact, inputs, expected = build_configured_matmul_artifact(case)

    host_result = run_host_emulation(artifact, inputs, VerificationMode.DEBUG, debug=True)
    rtl_result = await run_sim(
        artifact,
        inputs,
        dut=dut,
        verification=VerificationMode.DEBUG,
        debug=True,
        capture_vectors=True,
    )

    final_name = artifact.plan.outputs[0]
    assert host_result.verified == ["out"]
    assert rtl_result.verified == ["out"]
    assert np.array_equal(host_result.tensors[final_name], expected)
    assert np.array_equal(rtl_result.tensors[final_name], expected)
