import os
import sys

import torch


TESTS_DIR = os.path.dirname(__file__)
COMPILER_DIR = os.path.dirname(TESTS_DIR)
PROJECT_ROOT = os.path.dirname(os.path.dirname(COMPILER_DIR))
sys.path.insert(0, PROJECT_ROOT)

from software.compiler.tinynpu_jit import compile_module
from software.workload.mnist_tinynpu_pipeline import (
    TinyMNISTQAT,
    build_compiler_ready_mnist_model,
    build_signed_w8a8_configs,
)


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

    calib_dataset = [(torch.zeros(1, 28, 28), 0) for _ in range(4)]
    compiler_ready = build_compiler_ready_mnist_model(
        model,
        calib_dataset=calib_dataset,
        activation="gelu",
        output_activation="gelu",
    )

    example = torch.zeros(1, 1, 28, 28)
    artifact = compile_module(compiler_ready, (example,))

    activations = []
    for step in artifact.plan.steps:
        if hasattr(step, "ops"):
            activations.extend(op.activation for op in step.ops)

    assert "h_gelu" in activations
    assert "gelu" not in activations
