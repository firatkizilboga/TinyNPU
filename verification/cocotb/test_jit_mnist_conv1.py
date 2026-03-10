import json
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
from software.workload.mnist_npu_compiler import compile_mnist_layer_jit, prepare_activation_for_hw


@cocotb.test()
async def test_jit_mnist_conv1(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1

    with open(os.path.join(project_root, "mnist_mixed_export", "manifest.json")) as f:
        manifest = json.load(f)
    conv1 = next(layer for layer in manifest["layers"] if layer["name"] == "conv1")
    export_dir = os.path.join(project_root, "mnist_mixed_export")

    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(28, 28, 1), dtype=np.int16)
    prepared = prepare_activation_for_hw(image, conv1["a_bits"]).astype(np.int8)
    artifact = compile_mnist_layer_jit("conv1", conv1, image, export_dir=export_dir)

    result = await run_sim(
        artifact,
        {"x": prepared},
        dut=dut,
        verification=VerificationMode.DEBUG,
    )

    final_name = artifact.plan.outputs[0]
    assert np.array_equal(result.tensors[final_name], artifact.expected_tensors[final_name])
