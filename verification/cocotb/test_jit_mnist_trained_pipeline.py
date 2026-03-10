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
from software.workload.mnist_tinynpu_pipeline import build_compiled_artifact_from_run, get_mnist_loaders


def _choose_run_dir() -> str:
    env_run_dir = os.environ.get("TINYNPU_MNIST_RUN_DIR")
    if env_run_dir:
        return os.path.join(project_root, env_run_dir) if not os.path.isabs(env_run_dir) else env_run_dir
    fresh_dir = os.path.join(project_root, "runs", "mnist_tinynpu_fresh")
    smoke_dir = os.path.join(project_root, "runs", "mnist_tinynpu_smoke")
    if os.path.exists(os.path.join(fresh_dir, "qat.pt")):
        return fresh_dir
    return smoke_dir


async def _reset(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1


@cocotb.test()
async def test_jit_mnist_trained_pipeline(dut):
    await _reset(dut)
    run_dir = _choose_run_dir()
    sample_count = int(os.environ.get("TINYNPU_MNIST_COMPARE_SAMPLES", "3"))
    data_dir = os.path.join(project_root, "data")
    artifact, example, label = build_compiled_artifact_from_run(
        run_dir,
        data_dir=data_dir,
        sample_index=0,
        dequantize_output=False,
    )
    dut._log.info(f"run_dir={run_dir}")
    dut._log.info(f"sample_count={sample_count}")

    _, _, _, _, test_ds = get_mnist_loaders(data_dir)
    final_name = artifact.plan.outputs[0]
    checked = []
    last_logits = None
    last_label = None
    last_pred = None

    for sample_index in range(sample_count):
        sample_image, sample_label = test_ds[sample_index]
        inputs = {artifact.plan.inputs[0]: sample_image.unsqueeze(0).numpy()}

        host_result = run_host_emulation(
            artifact,
            inputs,
            VerificationMode.OFF,
            debug=False,
        )
        rtl_result = await run_sim(
            artifact,
            inputs,
            dut=dut,
            verification=VerificationMode.OFF,
            debug=False,
            capture_vectors=False,
        )

        checked = []
        for name in artifact.plan.tensors:
            if name not in host_result.trace_tensors or name not in rtl_result.trace_tensors:
                continue
            host_value = host_result.trace_tensors[name]
            rtl_value = rtl_result.trace_tensors[name]
            checked.append(name)
            if not np.array_equal(host_value, rtl_value):
                raise AssertionError(
                    f"sample={sample_index} first mismatch tensor={name}\n"
                    f"host={host_value.reshape(-1).tolist()}\n"
                    f"rtl={rtl_value.reshape(-1).tolist()}"
                )

        last_logits = rtl_result.tensors[final_name]
        last_label = int(sample_label)
        last_pred = int(np.argmax(last_logits.reshape(-1)))

    dut._log.info(f"last_label={last_label} last_pred={last_pred}")
    dut._log.info(f"checked={checked}")
    dut._log.info(f"last_raw_logits={last_logits.reshape(-1).tolist()}")
