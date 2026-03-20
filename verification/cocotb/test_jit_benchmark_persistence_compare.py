from __future__ import annotations

import inspect
import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from software.compiler.tinynpu_jit import VerificationMode, run_sim
from software.compiler.tinynpu_jit.benchmark import BenchmarkReport
from software.workload.jit_qat_compiler_ready import build_qat_compiler_ready_artifact


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
async def test_jit_benchmark_persistence_compare(dut):
    assert int(dut.PERF_ENABLE.value) == 1, "Run this test with -GPERF_ENABLE=1"

    await _reset(dut)
    artifact, example = build_qat_compiler_ready_artifact()
    inputs = {artifact.plan.inputs[0]: example}
    runs = int(os.environ.get("TINYNPU_BENCH_RUNS", "5"))

    run_sim_params = inspect.signature(run_sim).parameters
    supports_executor = "executor" in run_sim_params
    executor = None
    if supports_executor:
        from software.compiler.tinynpu_jit.simulator import SimulatorExecutor

        executor = SimulatorExecutor()

    reports = []
    for run_idx in range(runs):
        kwargs = dict(
            artifact=artifact,
            inputs=inputs,
            dut=dut,
            verification=VerificationMode.OFF,
            debug=False,
            capture_vectors=False,
            benchmark=True,
            reset=(run_idx == 0),
        )
        if supports_executor:
            kwargs["executor"] = executor
        result = await run_sim(**kwargs)
        assert result.benchmark is not None
        reports.append(result.benchmark)

    summed = _sum_totals(reports)
    first = reports[0].to_dict()["totals"]
    tail = _sum_totals(reports[1:]) if len(reports) > 1 else {k: 0 for k in summed}
    tail_count = max(len(reports) - 1, 1)
    tail_avg = {k: (tail[k] / tail_count if len(reports) > 1 else first[k]) for k in summed}

    dut._log.info(f"supports_executor={supports_executor}")
    dut._log.info(f"runs={runs}")
    dut._log.info(f"first={first}")
    dut._log.info(f"tail_total={tail}")
    dut._log.info(f"tail_avg={tail_avg}")
    dut._log.info(f"all_total={summed}")

    assert summed["npu_compute_cycles"] > 0
    assert summed["npu_overhead_cycles"] > 0
    if supports_executor and runs > 1:
        assert int(tail_avg["npu_compute_cycles"]) == int(first["npu_compute_cycles"])
