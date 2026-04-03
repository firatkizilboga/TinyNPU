from __future__ import annotations

import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
compiler_root = os.path.join(project_root, "software", "compiler")
if compiler_root not in sys.path:
    sys.path.append(compiler_root)

RUN_DIR = os.path.join(
    project_root,
    "runs",
    "tinynpu_issue27_backup_2026_03_21",
    "mnist_mlp_iszero_int16_smoke",
)

from software.compiler.tinynpu_jit import VerificationMode, run_host_emulation, run_sim  # noqa: E402
from software.compiler.tinynpu_jit.simulator import SimulatorExecutor  # noqa: E402
from software.workload.mnist_mlp_feature_benchmark import (  # noqa: E402
    build_compiled_artifact_from_run,
    get_flat_mnist_loaders,
)


@cocotb.test()
async def test_jit_iszero_mlp_runtime_sweep(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1

    artifact, _sample0, _label0, _summary = build_compiled_artifact_from_run(
        RUN_DIR,
        sample_index=0,
        dequantize_output=True,
    )
    _train_loader, _test_loader, _test_loader_single, _train_ds, test_ds = get_flat_mnist_loaders(task="is_zero")
    executor = SimulatorExecutor()

    for sample_index in range(5):
        sample_image, label = test_ds[sample_index]
        inputs = {artifact.plan.inputs[0]: sample_image.unsqueeze(0).numpy()}
        host_result = run_host_emulation(
            artifact,
            inputs,
            verification=VerificationMode.OFF,
            debug=False,
        )
        rtl_result = await run_sim(
            artifact,
            inputs,
            dut=dut,
            verification=VerificationMode.OFF,
            reset=(sample_index == 0),
            debug=False,
            executor=executor,
        )

        host_out = host_result.tensors[artifact.plan.outputs[0]]
        rtl_out = rtl_result.tensors[artifact.plan.outputs[0]]
        host_score = float(host_out.reshape(-1)[0])
        rtl_score = float(rtl_out.reshape(-1)[0])
        host_pred = int(host_score >= 0.5)
        rtl_pred = int(rtl_score >= 0.5)

        dut._log.info(
            f"sample={sample_index} label={int(label)} "
            f"host={host_score:.8f} rtl={rtl_score:.8f} "
            f"host_pred={host_pred} rtl_pred={rtl_pred}"
        )

        assert np.allclose(rtl_out, host_out, rtol=1e-5, atol=1e-6)
