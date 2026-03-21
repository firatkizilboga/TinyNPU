from __future__ import annotations

import os
import sys
import torch
from torch.utils.data import DataLoader, TensorDataset

TESTS_DIR = os.path.dirname(__file__)
COMPILER_DIR = os.path.dirname(TESTS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(COMPILER_DIR))
sys.path.insert(0, PROJECT_ROOT)

from software.compiler.tinynpu_jit import VerificationMode, compile_module, run_host_emulation
from software.compiler.tinynpu_jit.ir import NpuSegment
from software.workload.mnist_mlp_feature_benchmark import (
    INPUT_DIM,
    TASK_IS_ZERO,
    TinyFeatureMLPFP32,
    build_compiler_ready_model,
    build_feature_coverage_configs,
    get_flat_mnist_loaders,
    initialize_qat_from_fp32,
    plan_stats,
)


def _tiny_flat_calib_loader() -> DataLoader:
    x = torch.zeros(16, INPUT_DIM, dtype=torch.float32)
    y = torch.zeros(16, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def test_feature_benchmark_mixed_precision_lowers_without_conv_host_ops():
    torch.manual_seed(0)
    fp32_model = TinyFeatureMLPFP32()
    layer_configs = build_feature_coverage_configs("mixed")
    qat_model = initialize_qat_from_fp32(
        fp32_model,
        layer_configs=layer_configs,
        calib_loader=_tiny_flat_calib_loader(),
        device="cpu",
    )
    compiler_ready = build_compiler_ready_model(qat_model)

    artifact = compile_module(compiler_ready, (torch.zeros(1, INPUT_DIM, dtype=torch.float32),))
    stats = plan_stats(artifact)

    assert stats["npu_segments"] >= 1
    assert "host_im2col" not in stats["host_ops"]
    assert "host_layout_restore" not in stats["host_ops"]
    assert "h_gelu" in stats["activations"]
    assert "sigmoid" in stats["activations"]

    inner = compiler_ready.inner
    assert inner.fc1.in_bits == 16
    assert inner.fc1.out_bits == 4
    assert inner.fc2.in_bits == 4
    assert inner.fc2.out_bits == 8
    assert inner.fc3.in_bits == 8
    assert inner.fc3.out_bits == 16
    assert inner.fc4.in_bits == 16
    assert inner.fc4.out_bits == 16


def test_feature_benchmark_is_zero_loader_and_single_output_compile():
    _, _, _, train_ds, _ = get_flat_mnist_loaders(task=TASK_IS_ZERO)
    _, label0 = train_ds[0]
    assert label0 in (0, 1)

    torch.manual_seed(0)
    fp32_model = TinyFeatureMLPFP32(num_outputs=1)
    layer_configs = build_feature_coverage_configs("mixed")
    qat_model = initialize_qat_from_fp32(
        fp32_model,
        layer_configs=layer_configs,
        calib_loader=_tiny_flat_calib_loader(),
        device="cpu",
    )
    compiler_ready = build_compiler_ready_model(qat_model)
    artifact = compile_module(compiler_ready, (torch.zeros(1, INPUT_DIM, dtype=torch.float32),))
    assert artifact.plan.outputs


def test_feature_benchmark_internal_outputs_are_marked_as_a_layout():
    torch.manual_seed(0)
    fp32_model = TinyFeatureMLPFP32(num_outputs=1)
    layer_configs = build_feature_coverage_configs("mixed")
    qat_model = initialize_qat_from_fp32(
        fp32_model,
        layer_configs=layer_configs,
        calib_loader=_tiny_flat_calib_loader(),
        device="cpu",
    )
    compiler_ready = build_compiler_ready_model(qat_model)
    artifact = compile_module(compiler_ready, (torch.zeros(1, INPUT_DIM, dtype=torch.float32),))

    segment = next(step for step in artifact.plan.steps if isinstance(step, NpuSegment))
    output_layouts = {op.out: op.output_layout for op in segment.ops}
    assert output_layouts["relu"] == "a"
    assert output_layouts["relu_1"] == "a"
    assert output_layouts["gelu"] == "a"
    assert output_layouts["sigmoid"] == "c"

    symbol_table = artifact.segment_artifacts[segment.name].symbol_table
    assert symbol_table["relu"]["role"] == "A"
    assert symbol_table["relu_1"]["role"] == "A"
    assert symbol_table["gelu"]["role"] == "A"
    assert symbol_table["sigmoid"]["role"] == "C"


def test_feature_benchmark_reports_end_to_end_host_remaining_metric():
    torch.manual_seed(0)
    fp32_model = TinyFeatureMLPFP32(num_outputs=1)
    layer_configs = build_feature_coverage_configs("mixed")
    qat_model = initialize_qat_from_fp32(
        fp32_model,
        layer_configs=layer_configs,
        calib_loader=_tiny_flat_calib_loader(),
        device="cpu",
    )
    compiler_ready = build_compiler_ready_model(qat_model)
    artifact = compile_module(compiler_ready, (torch.zeros(1, INPUT_DIM, dtype=torch.float32),))

    result = run_host_emulation(
        artifact,
        {artifact.plan.inputs[0]: torch.zeros(1, INPUT_DIM, dtype=torch.float32).numpy()},
        VerificationMode.OFF,
        benchmark=True,
    )
    totals = result.benchmark.to_dict()["totals"]

    assert "host_remaining_cycles" in totals
    assert "end_to_end_analytical_speedup" in totals
    assert "cpu_full_baseline_cycles" in totals
    assert "cpu_only_baseline_speedup" in totals
    assert totals["host_remaining_cycles"] == totals["host_intrinsic_cycles"]
    assert totals["cpu_full_baseline_cycles"] == totals["cpu_replaced_cycles"]

    expected = totals["cpu_replaced_cycles"] / (
        totals["npu_compute_cycles"] + totals["npu_overhead_cycles"] + totals["host_remaining_cycles"]
    )
    assert totals["end_to_end_analytical_speedup"] == expected
    assert totals["cpu_only_baseline_speedup"] == expected
