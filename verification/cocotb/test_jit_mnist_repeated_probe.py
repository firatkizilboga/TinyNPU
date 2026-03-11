from __future__ import annotations

import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)

from software.compiler.tinynpu_jit import VerificationMode, run_host_emulation, run_sim
from software.compiler.tinynpu_jit.simulator import SimulatorExecutor
from software.workload.mnist_tinynpu_pipeline import build_compiled_artifact_from_run, get_mnist_loaders


def _choose_run_dir() -> str:
    fresh_dir = os.path.join(project_root, 'runs', 'mnist_tinynpu_fresh')
    smoke_dir = os.path.join(project_root, 'runs', 'mnist_tinynpu_smoke')
    if os.path.exists(os.path.join(fresh_dir, 'qat.pt')):
        return fresh_dir
    return smoke_dir


async def _reset(dut):
    clock = Clock(dut.clk, 10, units='ns')
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    dut.host_wr_en.value = 0
    dut.host_addr.value = 0
    dut.host_wr_data.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


@cocotb.test()
async def test_jit_mnist_repeated_probe(dut):
    assert int(dut.PERF_ENABLE.value) == 1
    await _reset(dut)
    run_dir = _choose_run_dir()
    artifact, _, _ = build_compiled_artifact_from_run(
        run_dir,
        data_dir=os.path.join(project_root, 'data'),
        sample_index=0,
        dequantize_output=False,
    )
    _, _, _, _, test_ds = get_mnist_loaders(os.path.join(project_root, 'data'))
    executor = SimulatorExecutor()

    for sample_index in range(3):
        sample_image, label = test_ds[sample_index]
        inputs = {artifact.plan.inputs[0]: sample_image.unsqueeze(0).numpy()}
        host = run_host_emulation(artifact, inputs, verification=VerificationMode.OFF, benchmark=True)
        rtl = await run_sim(
            artifact,
            inputs,
            dut=dut,
            verification=VerificationMode.OFF,
            benchmark=True,
            reset=(sample_index == 0),
            executor=executor,
        )
        host_out = host.tensors[artifact.plan.outputs[0]]
        rtl_out = rtl.tensors[artifact.plan.outputs[0]]
        dut._log.info(f'sample={sample_index} label={int(label)}')
        dut._log.info(f'host={host_out.tolist()}')
        dut._log.info(f'rtl={rtl_out.tolist()}')
        dut._log.info(f'rtl_bench={rtl.benchmark.to_dict()["totals"]}')
        assert (host_out == rtl_out).all()
