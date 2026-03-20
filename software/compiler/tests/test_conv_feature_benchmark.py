from __future__ import annotations

import os
import sys

import torch
from torch.utils.data import DataLoader, TensorDataset

TESTS_DIR = os.path.dirname(__file__)
COMPILER_DIR = os.path.dirname(TESTS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(COMPILER_DIR))
sys.path.insert(0, PROJECT_ROOT)

from software.compiler.tinynpu_jit import compile_module
from software.workload.mnist_conv_feature_benchmark import (
    TinyConvFeatureFP32,
    build_compiler_ready_model,
    build_conv_feature_configs,
    initialize_qat_from_fp32,
    plan_stats,
)


def _tiny_conv_calib_loader() -> DataLoader:
    x = torch.zeros(16, 1, 8, 8, dtype=torch.float32)
    y = torch.zeros(16, dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def test_conv_feature_benchmark_int16_plan_removes_mean_boundary():
    torch.manual_seed(0)
    fp32_model = TinyConvFeatureFP32()
    qat_model = initialize_qat_from_fp32(
        fp32_model,
        layer_configs=build_conv_feature_configs("int16"),
        calib_loader=_tiny_conv_calib_loader(),
        device="cpu",
    )
    compiler_ready = build_compiler_ready_model(qat_model, dequantize_output=False)

    artifact = compile_module(compiler_ready, (torch.zeros(1, 1, 8, 8, dtype=torch.float32),))
    stats = plan_stats(artifact)

    assert stats["npu_segments"] == 4
    assert stats["host_ops"] == [
        "quantize",
        "im2col",
        "layout_restore",
        "im2col",
        "layout_restore",
        "im2col",
        "layout_restore",
        "im2col",
        "layout_restore",
    ]
    assert stats["host_ops"].count("quantize") == 1
    assert "mean" not in stats["host_ops"]
    assert stats["activations"] == ["relu", "relu", "relu", "sigmoid"]


def test_conv_feature_benchmark_int16_layers_stay_w16a16():
    torch.manual_seed(0)
    fp32_model = TinyConvFeatureFP32()
    qat_model = initialize_qat_from_fp32(
        fp32_model,
        layer_configs=build_conv_feature_configs("int16"),
        calib_loader=_tiny_conv_calib_loader(),
        device="cpu",
    )
    compiler_ready = build_compiler_ready_model(qat_model, dequantize_output=False)
    inner = compiler_ready.inner

    assert inner.conv1.in_bits == 16
    assert inner.conv1.out_bits == 16
    assert inner.conv2.in_bits == 16
    assert inner.conv2.out_bits == 16
    assert inner.conv3.in_bits == 16
    assert inner.conv3.out_bits == 16
    assert inner.conv4.in_bits == 16
    assert inner.conv4.out_bits == 16
