import os
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from software.compiler.tinynpu_jit import VerificationMode, run_host_emulation, run_sim
from software.workload.mnist_mlp_feature_benchmark import build_compiled_artifact_from_run


def _choose_run_dir() -> str:
    env_run_dir = os.environ.get("TINYNPU_FEATURE_MLP_RUN_DIR")
    if env_run_dir:
        return os.path.join(project_root, env_run_dir) if not os.path.isabs(env_run_dir) else env_run_dir
    return os.path.join(project_root, "runs", "mnist_mlp_feature_benchmark_164816_smoke")


def _choose_data_dir() -> str:
    env_data_dir = os.environ.get("TINYNPU_MNIST_DATA_DIR")
    candidates = []
    if env_data_dir:
        candidates.append(Path(env_data_dir))
    candidates.append(Path(project_root) / "data")
    candidates.append(Path.home() / "compiler-optimization" / "data")

    required = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    for candidate in candidates:
        raw_dir = candidate / "MNIST" / "raw"
        if all((raw_dir / name).exists() for name in required):
            return str(candidate)
    return str(candidates[0])


async def _reset(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1


@cocotb.test()
async def test_jit_mlp_feature_benchmark(dut):
    await _reset(dut)
    run_dir = _choose_run_dir()
    data_dir = _choose_data_dir()
    artifact, example, label, summary = build_compiled_artifact_from_run(
        run_dir,
        data_dir=data_dir,
        sample_index=0,
        dequantize_output=False,
    )
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
        verification=VerificationMode.OFF,
        debug=True,
        capture_vectors=True,
    )

    checked = []
    final_name = artifact.plan.outputs[0]
    for name in artifact.plan.tensors:
        if name not in host_result.trace_tensors or name not in rtl_result.trace_tensors:
            continue
        checked.append(name)
        if not np.array_equal(host_result.trace_tensors[name], rtl_result.trace_tensors[name]):
            dut._log.error(
                f"mismatch tensor={name} host={host_result.trace_tensors[name].reshape(-1).tolist()} "
                f"rtl={rtl_result.trace_tensors[name].reshape(-1).tolist()}"
            )
            raise AssertionError(name)

    assert np.array_equal(host_result.tensors[final_name], rtl_result.tensors[final_name])

    assert any(event["kind"] == "host_quantize" for event in rtl_result.debug_trace)
    assert any(event["kind"] == "npu_segment" for event in rtl_result.debug_trace)
    if any(event["kind"] == "host_dequantize" for event in host_result.debug_trace):
        assert any(event["kind"] == "host_dequantize" for event in rtl_result.debug_trace)
    dut._log.info(f"run_dir={run_dir}")
    dut._log.info(f"label={label} checked={checked}")
    dut._log.info(f"summary_bits={summary['layer_bits']}")
    dut._log.info(f"final_vector={rtl_result.tensors[final_name].reshape(-1).tolist()}")
