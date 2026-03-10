import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from software.compiler.tinynpu_jit import VerificationMode, run_sim
from software.workload.jit_test_gen import build_simple_chain_artifact


@cocotb.test()
async def test_jit_simple_chain_inspect(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1

    test_case = build_simple_chain_artifact(seed=7, dim=16)
    artifact = test_case["artifact"]
    result = await run_sim(
        artifact,
        test_case["inputs"],
        dut=dut,
        verification=VerificationMode.DEBUG,
        capture_vectors=True,
    )

    report = artifact.inspect(test_case["inputs"], execution_result=result)
    dut._log.info("\n" + report)
