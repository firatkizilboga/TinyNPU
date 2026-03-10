import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from software.compiler.tinynpu_jit import VerificationMode, run_sim
from software.workload.jit_hostop_chain import build_hostop_chain_artifact


@cocotb.test()
async def test_jit_hostop_chain(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1

    test_case = build_hostop_chain_artifact()
    artifact = test_case["artifact"]
    result = await run_sim(
        artifact,
        test_case["inputs"],
        dut=dut,
        verification=VerificationMode.DEBUG,
    )

    assert "scores" in result.verified
    assert "q_probs" in result.verified
    final_name = artifact.plan.outputs[0]
    assert np.array_equal(result.tensors[final_name], test_case["expected"][final_name])
