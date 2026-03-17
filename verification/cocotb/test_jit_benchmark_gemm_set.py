from __future__ import annotations

from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

from software.compiler.tinynpu_jit import VerificationMode, run_sim
from software.workload.jit_multitile_matmul import (
    JitMatmulBenchmarkCase,
    build_configured_matmul_artifact,
    default_gemm_benchmark_cases,
)


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


def _format_case_summary(case: JitMatmulBenchmarkCase, totals: dict[str, object], artifact) -> str:
    npu_compute = int(totals["npu_compute_cycles"])
    npu_overhead = int(totals["npu_overhead_cycles"])
    cpu_replaced = int(totals["cpu_replaced_cycles"])
    total_cycles = npu_compute + npu_overhead
    cycles_per_mac = npu_compute / case.total_macs if case.total_macs else 0.0
    macs_per_cycle = case.total_macs / npu_compute if npu_compute else 0.0
    overhead_fraction = npu_overhead / total_cycles if total_cycles else 0.0
    seg = artifact.segment_artifacts["segment_000"]
    return (
        f"{case.name}: "
        f"macs={case.total_macs} "
        f"cpu_replaced={cpu_replaced} "
        f"npu_compute={npu_compute} "
        f"npu_overhead={npu_overhead} "
        f"cycles_per_mac={cycles_per_mac:.6f} "
        f"macs_per_cycle={macs_per_cycle:.4f} "
        f"overhead_fraction={overhead_fraction:.4f} "
        f"ub_words={seg.ub_words} "
        f"im_words={seg.im_words}"
    )


@cocotb.test()
async def test_jit_benchmark_gemm_set(dut):
    assert int(dut.PERF_ENABLE.value) == 1, "Run this test with -GPERF_ENABLE=1"

    summaries: list[str] = []
    for case in default_gemm_benchmark_cases():
        await _reset(dut)
        artifact, inputs, _ = build_configured_matmul_artifact(case)
        result = await run_sim(
            artifact,
            inputs,
            dut=dut,
            verification=VerificationMode.FINAL,
            debug=False,
            benchmark=True,
        )
        assert result.benchmark is not None
        report = result.benchmark.to_dict()
        totals = report["totals"]
        assert totals["cpu_replaced_cycles"] > 0
        assert totals["npu_compute_cycles"] > 0
        assert totals["npu_overhead_cycles"] > 0
        assert "out" in result.verified
        summaries.append(_format_case_summary(case, totals, artifact))

    dut._log.info("GEMM benchmark set results:")
    for line in summaries:
        dut._log.info("  %s", line)
    Path("/tmp/gemm_set_summary.txt").write_text("\n".join(summaries) + "\n")
