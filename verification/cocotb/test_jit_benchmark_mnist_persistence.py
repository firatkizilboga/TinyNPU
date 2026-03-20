from __future__ import annotations

import inspect
import json
import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from software.compiler.tinynpu_jit import VerificationMode, run_sim
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


def _choose_data_dir() -> str:
    env_data_dir = os.environ.get("TINYNPU_MNIST_DATA_DIR")
    if env_data_dir:
        return env_data_dir
    return os.path.join(project_root, "data")


def _load_run_activations(run_dir: str) -> tuple[str, str]:
    for summary_name in ("summary.json", "gelu_summary.json", "sigmoid_summary.json"):
        summary_path = os.path.join(run_dir, summary_name)
        if not os.path.exists(summary_path):
            continue
        with open(summary_path, "r") as handle:
            summary = json.load(handle)
        return str(summary.get("activation", "relu")), str(summary.get("output_activation", "none"))
    return "relu", "none"


def _sum_totals(reports):
    keys = [
        "cpu_replaced_cycles",
        "npu_compute_cycles",
        "npu_overhead_cycles",
        "host_intrinsic_cycles",
        "host_remaining_cycles",
    ]
    summed = {key: 0 for key in keys}
    for report in reports:
        totals = report.to_dict()["totals"]
        for key in keys:
            summed[key] += int(totals[key])
    return summed


async def _reset(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    dut.host_wr_en.value = 0
    dut.host_addr.value = 0
    dut.host_wr_data.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


@cocotb.test()
async def test_jit_benchmark_mnist_persistence(dut):
    assert int(dut.PERF_ENABLE.value) == 1, "Run this test with -GPERF_ENABLE=1"

    await _reset(dut)
    run_dir = _choose_run_dir()
    data_dir = _choose_data_dir()
    activation, output_activation = _load_run_activations(run_dir)
    artifact, _, _ = build_compiled_artifact_from_run(
        run_dir,
        data_dir=data_dir,
        sample_index=0,
        dequantize_output=False,
        activation=activation,
        output_activation=output_activation,
    )
    sample_count = int(os.environ.get("TINYNPU_BENCH_RUNS", "3"))
    _, _, _, _, test_ds = get_mnist_loaders(data_dir)

    run_sim_params = inspect.signature(run_sim).parameters
    supports_executor = "executor" in run_sim_params
    executor = None
    if supports_executor:
        from software.compiler.tinynpu_jit.simulator import SimulatorExecutor

        executor = SimulatorExecutor()

    reports = []
    for sample_index in range(sample_count):
        sample_image, _ = test_ds[sample_index]
        inputs = {artifact.plan.inputs[0]: sample_image.unsqueeze(0).numpy()}
        kwargs = dict(
            artifact=artifact,
            inputs=inputs,
            dut=dut,
            verification=VerificationMode.OFF,
            debug=False,
            capture_vectors=False,
            benchmark=True,
            reset=(sample_index == 0),
        )
        if supports_executor:
            kwargs["executor"] = executor
        result = await run_sim(**kwargs)
        assert result.benchmark is not None
        reports.append(result.benchmark)

    first = reports[0].to_dict()["totals"]
    tail_total = _sum_totals(reports[1:]) if len(reports) > 1 else {k: 0 for k in _sum_totals(reports)}
    tail_count = max(len(reports) - 1, 1)
    tail_avg = {k: (tail_total[k] / tail_count if len(reports) > 1 else first[k]) for k in first if k.endswith('_cycles')}

    dut._log.info(f"run_dir={run_dir}")
    dut._log.info(f"data_dir={data_dir}")
    dut._log.info(f"activation={activation} output_activation={output_activation}")
    dut._log.info(f"supports_executor={supports_executor}")
    dut._log.info(f"runs={sample_count}")
    dut._log.info(f"first={first}")
    dut._log.info(f"tail_total={tail_total}")
    dut._log.info(f"tail_avg={tail_avg}")

    assert first["npu_compute_cycles"] > 0
    assert first["npu_overhead_cycles"] > 0
    if supports_executor and sample_count > 1:
        assert int(tail_avg["npu_compute_cycles"]) == int(first["npu_compute_cycles"])
