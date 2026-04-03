import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

RUN_DIR = os.path.join(
    project_root,
    "runs",
    "tinynpu_issue27_backup_2026_03_21",
    "mnist_mlp_iszero_int16_smoke",
)

from software.workload.mnist_mlp_feature_benchmark import build_compiled_artifact_from_run  # noqa: E402
from software.compiler.tinynpu_jit import VerificationMode, run_host_emulation, run_sim  # noqa: E402


@cocotb.test()
async def test_jit_iszero_mlp_runtime(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1

    artifact, sample, label, _summary = build_compiled_artifact_from_run(
        RUN_DIR,
        dequantize_output=True,
    )
    inputs = {"x": np.array(sample, copy=True)}

    host_result = run_host_emulation(
        artifact,
        inputs,
        VerificationMode.DEBUG,
    )
    rtl_result = await run_sim(
        artifact,
        inputs,
        dut=dut,
        verification=VerificationMode.OFF,
        debug=True,
    )

    host_out = host_result.tensors["dq_out"]
    rtl_out = rtl_result.tensors["dq_out"]
    print("label", label)
    print("host dq_out", host_out)
    print("rtl dq_out", rtl_out)
    print("host debug trace", host_result.debug_trace)
    print("rtl debug trace", rtl_result.debug_trace)
    assert np.allclose(rtl_out, host_out, rtol=1e-5, atol=1e-6)
