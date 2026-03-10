import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from software.compiler.tinynpu_jit import VerificationMode, run_host_emulation, run_sim
from software.workload.jit_qat_compiler_ready import build_qat_compiler_ready_artifact


async def _reset(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1


@cocotb.test()
async def test_jit_qat_compiler_ready(dut):
    await _reset(dut)
    artifact, example = build_qat_compiler_ready_artifact()
    inputs = {artifact.plan.inputs[0]: example}

    host_result = run_host_emulation(
        artifact,
        inputs,
        VerificationMode.DEBUG,
        debug=True,
    )
    rtl_result = await run_sim(
        artifact,
        inputs,
        dut=dut,
        verification=VerificationMode.DEBUG,
        debug=True,
        capture_vectors=True,
    )

    final_name = artifact.plan.outputs[0]
    assert np.allclose(host_result.tensors[final_name], rtl_result.tensors[final_name], rtol=1e-5, atol=1e-6)
    assert any(event["kind"] == "host_quantize" for event in rtl_result.debug_trace)
    assert any(event["kind"] == "npu_segment" for event in rtl_result.debug_trace)
    assert any(event["kind"] == "host_mean" for event in rtl_result.debug_trace)
    assert any(event["kind"] == "host_dequantize" for event in rtl_result.debug_trace)
    dut._log.info(f"QAT compiler-ready final vector: {rtl_result.tensors[final_name].reshape(-1).tolist()}")
