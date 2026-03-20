from __future__ import annotations

import os
import sys
from pathlib import Path

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

from software.compiler.tinynpu_jit import VerificationMode, run_sim
from software.workload.mnist_conv_feature_benchmark import (
    TASK_IS_ZERO,
    build_compiled_artifact_from_run,
    get_conv_mnist_loaders,
)


def _choose_run_dir() -> str:
    env_run_dir = os.environ.get("TINYNPU_CONV_FEATURE_RUN_DIR")
    if env_run_dir:
        return os.path.join(project_root, env_run_dir) if not os.path.isabs(env_run_dir) else env_run_dir
    return os.path.join(project_root, "runs", "mnist_conv_feature_benchmark_int16_iszero")


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


def _choose_compare_samples() -> int:
    raw = os.environ.get("TINYNPU_CONV_FEATURE_COMPARE_SAMPLES")
    if not raw:
        return 30
    return max(1, int(raw))


def _balanced_binary_indices(test_ds, *, count: int) -> list[int]:
    raw = os.environ.get("TINYNPU_CONV_FEATURE_SAMPLE_INDICES")
    if raw:
        indices = [int(part.strip()) for part in raw.split(",") if part.strip()]
        return indices[:count]

    if count <= 1:
        return [0]

    positives_needed = count // 2
    negatives_needed = count - positives_needed
    positives: list[int] = []
    negatives: list[int] = []
    for idx, (_, label) in enumerate(test_ds):
        if int(label) == 1 and len(positives) < positives_needed:
            positives.append(idx)
        elif int(label) == 0 and len(negatives) < negatives_needed:
            negatives.append(idx)
        if len(positives) >= positives_needed and len(negatives) >= negatives_needed:
            break

    indices = positives + negatives
    if len(indices) < count:
        seen = set(indices)
        for idx in range(len(test_ds)):
            if idx in seen:
                continue
            indices.append(idx)
            if len(indices) >= count:
                break
    return indices[:count]


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
async def test_jit_benchmark_conv_persistence(dut):
    assert int(dut.PERF_ENABLE.value) == 1, "Run this test with -GPERF_ENABLE=1"

    await _reset(dut)
    run_dir = _choose_run_dir()
    data_dir = _choose_data_dir()
    compare_samples = _choose_compare_samples()
    artifact, _, _, summary = build_compiled_artifact_from_run(
        run_dir,
        data_dir=data_dir,
        sample_index=0,
        dequantize_output=False,
    )
    assert str(summary.get("task")) == TASK_IS_ZERO

    _, _, _, _, test_ds = get_conv_mnist_loaders(data_dir)
    sample_indices = _balanced_binary_indices(test_ds, count=compare_samples)

    from software.compiler.tinynpu_jit.simulator import SimulatorExecutor

    executor = SimulatorExecutor()
    reports = []
    for pos, sample_index in enumerate(sample_indices):
        sample_image, _ = test_ds[sample_index]
        inputs = {artifact.plan.inputs[0]: sample_image.unsqueeze(0).numpy()}
        result = await run_sim(
            artifact=artifact,
            inputs=inputs,
            dut=dut,
            verification=VerificationMode.OFF,
            debug=False,
            capture_vectors=False,
            benchmark=True,
            reset=(pos == 0),
            executor=executor,
        )
        assert result.benchmark is not None
        reports.append(result.benchmark)

    first = reports[0].to_dict()["totals"]
    tail_total = _sum_totals(reports[1:]) if len(reports) > 1 else {k: 0 for k in _sum_totals(reports)}
    tail_count = max(len(reports) - 1, 1)
    tail_avg = {
        key: (tail_total[key] / tail_count if len(reports) > 1 else int(first[key]))
        for key in first
        if key.endswith("_cycles")
    }
    all_total = _sum_totals(reports)

    dut._log.info(f"run_dir={run_dir}")
    dut._log.info(f"samples={sample_indices}")
    dut._log.info(f"summary_bits={summary['layer_bits']}")
    dut._log.info(f"first={first}")
    dut._log.info(f"tail_total={tail_total}")
    dut._log.info(f"tail_avg={tail_avg}")
    dut._log.info(f"all_total={all_total}")

    assert int(first["npu_compute_cycles"]) > 0
    assert int(first["npu_overhead_cycles"]) > 0
    assert int(first["host_remaining_cycles"]) > 0
    if len(reports) > 1:
        assert int(tail_avg["npu_compute_cycles"]) == int(first["npu_compute_cycles"])
