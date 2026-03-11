from __future__ import annotations

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

from software.compiler.tinynpu_jit import (
    VerificationMode,
    five_stage_in_order_model,
    ideal_issue_1_model,
    run_sim,
    unpipelined_scalar_model,
)
from software.workload.jit_multitile_matmul import build_multitile_matmul_artifact


async def _reset(dut) -> None:
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
async def test_jit_benchmark_multitile_matmul(dut):
    assert int(dut.PERF_ENABLE.value) == 1, "Run this test with -GPERF_ENABLE=1"

    await _reset(dut)
    artifact, _ = build_multitile_matmul_artifact()
    result = await run_sim(
        artifact,
        {},
        dut=dut,
        verification=VerificationMode.OFF,
        debug=False,
        benchmark=True,
    )
    assert result.benchmark is not None
    report = result.benchmark.to_dict()
    totals = report["totals"]

    dut._log.info(artifact.format_benchmark_report(result))
    dut._log.info(
        artifact.format_benchmark_comparison(
            result,
            [unpipelined_scalar_model(), ideal_issue_1_model(), five_stage_in_order_model()],
        )
    )

    assert totals["cpu_replaced_cycles"] > 0
    assert totals["npu_compute_cycles"] > 0
    assert totals["npu_overhead_cycles"] > 0
    assert totals["host_intrinsic_cycles"] == 0
    assert totals["pure_acceleration_speedup"] is not None
    assert totals["integration_adjusted_speedup"] is not None
    assert len(
        result.benchmark.model_comparison(
            [unpipelined_scalar_model(), ideal_issue_1_model(), five_stage_in_order_model()]
        )
    ) == 3
