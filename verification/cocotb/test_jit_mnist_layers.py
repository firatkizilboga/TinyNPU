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


def _load_manifest():
    with open(os.path.join(project_root, "mnist_mixed_export", "manifest.json")) as f:
        manifest = json.load(f)
    return {layer["name"]: layer for layer in manifest["layers"]}


def _prepare_case(layer_name: str):
    manifest = _load_manifest()
    layer = manifest[layer_name]
    export_dir = os.path.join(project_root, "mnist_mixed_export")
    rng = np.random.default_rng(0)

    if layer_name == "conv2":
        host_input = rng.integers(0, 256, size=(28, 28, 16), dtype=np.int16)
    elif layer_name == "conv3":
        host_input = rng.integers(0, 256, size=(28, 28, 16), dtype=np.int16)
    elif layer_name == "fc":
        host_input = rng.integers(0, 256, size=(16, 1), dtype=np.int16)
    else:
        raise ValueError(f"Unsupported layer {layer_name!r}.")

    prepared = prepare_activation_for_hw(host_input, layer["a_bits"]).astype(np.int8)
    artifact = compile_mnist_layer_jit(layer_name, layer, host_input, export_dir=export_dir)
    return artifact, prepared


async def _reset(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1


@cocotb.test()
async def test_jit_mnist_conv2(dut):
    await _reset(dut)
    artifact, prepared = _prepare_case("conv2")
    result = await run_sim(
        artifact,
        {"x": prepared},
        dut=dut,
        verification=VerificationMode.DEBUG,
    )
    final_name = artifact.plan.outputs[0]
    assert np.array_equal(result.tensors[final_name], artifact.expected_tensors[final_name])


@cocotb.test()
async def test_jit_mnist_conv3(dut):
    await _reset(dut)
    artifact, prepared = _prepare_case("conv3")
    result = await run_sim(
        artifact,
        {"x": prepared},
        dut=dut,
        verification=VerificationMode.DEBUG,
    )
    final_name = artifact.plan.outputs[0]
    assert np.array_equal(result.tensors[final_name], artifact.expected_tensors[final_name])


@cocotb.test()
async def test_jit_mnist_fc(dut):
    await _reset(dut)
    artifact, prepared = _prepare_case("fc")
    result = await run_sim(
        artifact,
        {"x": prepared},
        dut=dut,
        verification=VerificationMode.DEBUG,
    )
    final_name = artifact.plan.outputs[0]
    assert np.array_equal(result.tensors[final_name], artifact.expected_tensors[final_name])
