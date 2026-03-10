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

from software.compiler.tinynpu_jit import VerificationMode, run_host_emulation, run_sim
from software.workload.mnist_npu_compiler import (
    compile_mnist_layer_jit,
    global_average_pool_for_fc_input,
    prepare_activation_for_hw,
)


def _load_manifest():
    with open(os.path.join(project_root, "mnist_mixed_export", "manifest.json")) as f:
        manifest = json.load(f)
    return manifest["layers"], os.path.join(project_root, "mnist_mixed_export")


async def _reset(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1


def _prepare_layer_input(layer, current_data):
    if layer["type"] == "linear":
        current_data = global_average_pool_for_fc_input(current_data)
    prepared = prepare_activation_for_hw(current_data, layer["a_bits"]).astype(np.int8)
    return current_data, prepared


@cocotb.test()
async def test_jit_mnist_full_chain(dut):
    await _reset(dut)
    layers, export_dir = _load_manifest()
    rng = np.random.default_rng(7)
    current_data = rng.integers(0, 256, size=(28, 28, 1), dtype=np.int16)

    host_final = None
    rtl_final = None

    for layer in layers:
        layer_input, prepared = _prepare_layer_input(layer, current_data)
        artifact = compile_mnist_layer_jit(layer["name"], layer, layer_input, export_dir=export_dir)

        host_result = run_host_emulation(
            artifact,
            {artifact.plan.inputs[0]: prepared},
            VerificationMode.DEBUG,
        )
        rtl_result = await run_sim(
            artifact,
            {artifact.plan.inputs[0]: prepared},
            dut=dut,
            verification=VerificationMode.DEBUG,
        )

        final_name = artifact.plan.outputs[0]
        assert np.array_equal(rtl_result.tensors[final_name], host_result.tensors[final_name])
        current_data = rtl_result.tensors[final_name]
        host_final = host_result.tensors[final_name]
        rtl_final = rtl_result.tensors[final_name]

    assert host_final is not None
    assert rtl_final is not None
    assert np.array_equal(rtl_final, host_final)
    dut._log.info(f"RTL final vector: {rtl_final.reshape(-1).tolist()}")
    dut._log.info(f"RTL predicted class: {int(np.argmax(rtl_final.reshape(-1)))}")
