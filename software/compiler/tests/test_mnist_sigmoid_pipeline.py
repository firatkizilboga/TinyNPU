import os
import sys

import torch


TESTS_DIR = os.path.dirname(__file__)
COMPILER_DIR = os.path.dirname(TESTS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(COMPILER_DIR))
sys.path.insert(0, PROJECT_ROOT)

from software.compiler.tinynpu_jit import compile_module
from software.compiler.tinynpu_jit.ir import DType
from software.workload.mnist_tinynpu_pipeline import (
    TinyMNISTQAT,
    apply_compiler_boundary_overrides,
    build_compiler_ready_mnist_model,
    build_signed_w8a8_configs,
)


def _small_calib_dataset():
    return [(torch.zeros(1, 8, 8), 0) for _ in range(4)]


def test_sigmoid_mnist_model_lowers_sigmoid_into_npu_segments():
    layer_configs = build_signed_w8a8_configs()
    model = TinyMNISTQAT(layer_configs, activation="relu", output_activation="sigmoid").eval()

    calib_dataset = [(torch.zeros(1, 28, 28), 0) for _ in range(4)]
    compiler_ready = build_compiler_ready_mnist_model(
        model,
        calib_dataset=calib_dataset,
        activation="relu",
        output_activation="sigmoid",
    )

    example = torch.zeros(1, 1, 28, 28)
    artifact = compile_module(compiler_ready, (example,))

    activations = []
    for step in artifact.plan.steps:
        if hasattr(step, "ops"):
            activations.extend(op.activation for op in step.ops)

    assert "sigmoid" in activations


def test_qat_sigmoid_head_can_use_di_sigmoid_approximation():
    layer_configs = build_signed_w8a8_configs()
    model = TinyMNISTQAT(
        layer_configs,
        activation="relu",
        output_activation="sigmoid",
        use_di_sigmoid_approx=True,
    ).eval()
    x = torch.zeros(1, 1, 28, 28)
    y = model(x)
    assert y.shape == (1, 10)
    assert torch.all(y >= 0)
    assert torch.all(y <= 1)


def test_gelu_mnist_model_lowers_to_h_gelu_backend_activation():
    layer_configs = build_signed_w8a8_configs()
    model = TinyMNISTQAT(layer_configs, activation="gelu", output_activation="gelu").eval()

    calib_dataset = _small_calib_dataset()
    compiler_ready = build_compiler_ready_mnist_model(
        model,
        calib_dataset=calib_dataset,
        activation="gelu",
        output_activation="gelu",
    )

    example = torch.zeros(1, 1, 8, 8)
    artifact = compile_module(compiler_ready, (example,))

    activations = []
    for step in artifact.plan.steps:
        if hasattr(step, "ops"):
            activations.extend(op.activation for op in step.ops)

    assert "h_gelu" in activations
    assert "gelu" not in activations


def test_gelu_policy_forces_int16_layer_precision_even_from_w8a8_defaults():
    layer_configs = build_signed_w8a8_configs()
    model = TinyMNISTQAT(layer_configs, activation="gelu", output_activation="none").eval()

    assert model.conv1.w_bits == 16
    assert model.conv1.a_bits == 16
    assert model.conv2.w_bits == 16
    assert model.conv2.a_bits == 16
    assert model.conv3.w_bits == 16
    assert model.conv3.a_bits == 16
    assert model.fc.w_bits == 16
    assert model.fc.a_bits == 16
    assert model.get_h_gelu_output_bits("conv1") == 16
    assert model.get_h_gelu_output_bits("conv2") == 16
    assert model.get_h_gelu_output_bits("conv3") == 16


def test_gelu_compiler_ready_conversion_clears_qat_fc_override_before_compile():
    layer_configs = build_signed_w8a8_configs()
    model = TinyMNISTQAT(layer_configs, activation="gelu", output_activation="none").eval()

    calib_dataset = _small_calib_dataset()
    apply_compiler_boundary_overrides(model, calib_dataset=calib_dataset, activation="gelu")
    assert model.fc_input_scale_override is not None

    compiler_ready = build_compiler_ready_mnist_model(
        model,
        calib_dataset=calib_dataset,
        activation="gelu",
        output_activation="none",
    )

    example = torch.zeros(1, 1, 8, 8)
    artifact = compile_module(compiler_ready, (example,))
    assert artifact.plan.steps


def test_gelu_compiler_ready_conversion_forces_int16_h_gelu_outputs():
    layer_configs = build_signed_w8a8_configs()
    model = TinyMNISTQAT(layer_configs, activation="gelu", output_activation="none").eval()

    calib_dataset = _small_calib_dataset()
    compiler_ready = build_compiler_ready_mnist_model(
        model,
        calib_dataset=calib_dataset,
        activation="gelu",
        output_activation="none",
    )

    assert compiler_ready.inner.conv1.out_bits == 16
    assert compiler_ready.inner.conv2.out_bits == 16
    assert compiler_ready.inner.conv3.out_bits == 16
    assert compiler_ready.inner.fc.in_bits == 16
    assert compiler_ready.inner.fc.out_bits == 16

    example = torch.zeros(1, 1, 8, 8)
    artifact = compile_module(compiler_ready, (example,))
    h_gelu_out_dtypes = [
        op.out_dtype
        for step in artifact.plan.steps
        if hasattr(step, "ops")
        for op in step.ops
        if op.activation == "h_gelu"
    ]
    assert h_gelu_out_dtypes
    assert all(dtype == DType.INT16 for dtype in h_gelu_out_dtypes)


def test_gelu_compiler_freezes_per_layer_h_gelu_shift_into_fused_ops():
    layer_configs = build_signed_w8a8_configs()
    model = TinyMNISTQAT(layer_configs, activation="gelu", output_activation="none").eval()
    with torch.no_grad():
        model.h_gelu_shift_params["conv1"].fill_(9.0)
        model.h_gelu_shift_params["conv2"].fill_(8.0)
        model.h_gelu_shift_params["conv3"].fill_(6.0)

    calib_dataset = _small_calib_dataset()
    compiler_ready = build_compiler_ready_mnist_model(
        model,
        calib_dataset=calib_dataset,
        activation="gelu",
        output_activation="none",
    )

    assert compiler_ready.inner.conv1.h_gelu_x_scale_shift == 9
    assert compiler_ready.inner.conv2.h_gelu_x_scale_shift == 8
    assert compiler_ready.inner.conv3.h_gelu_x_scale_shift == 6

    example = torch.zeros(1, 1, 8, 8)
    artifact = compile_module(compiler_ready, (example,))
    fused_shifts = [
        op.h_gelu_x_scale_shift
        for step in artifact.plan.steps
        if hasattr(step, "ops")
        for op in step.ops
        if op.activation == "h_gelu"
    ]
    assert fused_shifts[:3] == [9, 8, 6]
