from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from software.compiler.tinynpu_jit import compile_module
from software.compiler.tinynpu_quant import (
    LayerQuantConfig,
    QLinear,
    build_layer_config_map,
    convert_qat_model_for_compiler,
    infer_chain_output_bits,
    infer_chain_output_scales,
)


DEFAULT_DATA_DIR = "/home/firatkizilboga/compiler-optimization/data"
LAYER_ORDER = ["fc1", "fc2", "fc3", "fc4"]
INPUT_DIM = 64
HIDDEN_DIM = 64
TASK_MULTICLASS = "multiclass"
TASK_IS_ZERO = "is_zero"
FEATURE_SIGMOID_P_OUT = 16
FEATURE_SIGMOID_OUTPUT_SCALE = 1.0 / 32767.0
HGELU_MIN_SHIFT = 4
HGELU_MAX_SHIFT = 12


def _round_h_gelu_shift_value(shift: float) -> int:
    rounded = int(round(float(shift)))
    return max(HGELU_MIN_SHIFT, min(HGELU_MAX_SHIFT, rounded))


def _scale_to_h_gelu_shift(scale: float) -> int:
    safe_scale = max(float(scale), 1e-8)
    return _round_h_gelu_shift_value(-torch.log2(torch.tensor(safe_scale)).item())


def get_flat_mnist_loaders(
    data_dir: str = DEFAULT_DATA_DIR,
    *,
    image_size: int = 8,
    batch_size: int = 128,
    download: bool = False,
    task: str = TASK_MULTICLASS,
):
    if task not in {TASK_MULTICLASS, TASK_IS_ZERO}:
        raise ValueError(f"Unsupported task {task!r}.")

    target_transform = None
    if task == TASK_IS_ZERO:
        target_transform = lambda y: int(y == 0)

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )
    train_ds = datasets.MNIST(
        data_dir,
        train=True,
        download=download,
        transform=transform,
        target_transform=target_transform,
    )
    test_ds = datasets.MNIST(
        data_dir,
        train=False,
        download=download,
        transform=transform,
        target_transform=target_transform,
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    test_loader_single = DataLoader(test_ds, batch_size=1, shuffle=False)
    return train_loader, test_loader, test_loader_single, train_ds, test_ds


def layer_configs_from_summary(summary: dict[str, object]) -> dict[str, LayerQuantConfig]:
    layer_bits = summary.get("layer_bits", {})
    overrides: dict[str, LayerQuantConfig] = {}
    if not isinstance(layer_bits, dict):
        raise ValueError("summary.layer_bits must be a mapping.")
    for name in LAYER_ORDER:
        spec = layer_bits.get(name)
        if not isinstance(spec, dict):
            continue
        overrides[name] = LayerQuantConfig(
            w_bits=int(spec["w_bits"]),
            a_bits=int(spec["a_bits"]),
            signed_activations=True,
        )
    return build_layer_config_map(LAYER_ORDER, overrides=overrides, default_signed_activations=True)


class TinyFeatureMLPQAT(nn.Module):
    def __init__(
        self,
        layer_configs: dict[str, LayerQuantConfig],
        *,
        num_outputs: int = 10,
        use_di_sigmoid_approx: bool = True,
        use_h_gelu_approx: bool = True,
    ):
        super().__init__()
        self.use_di_sigmoid_approx = bool(use_di_sigmoid_approx)
        self.use_h_gelu_approx = bool(use_h_gelu_approx)
        self.fc1 = QLinear(INPUT_DIM, HIDDEN_DIM, config=layer_configs["fc1"])
        self.fc2 = QLinear(HIDDEN_DIM, HIDDEN_DIM, config=layer_configs["fc2"])
        self.fc3 = QLinear(HIDDEN_DIM, HIDDEN_DIM, config=layer_configs["fc3"])
        self.fc4 = QLinear(HIDDEN_DIM, num_outputs, config=layer_configs["fc4"])
        self.h_gelu_shift_param = nn.Parameter(torch.tensor(7.0))

    def set_initial_h_gelu_shift_param(self) -> None:
        with torch.no_grad():
            self.h_gelu_shift_param.fill_(float(_scale_to_h_gelu_shift(float(self.fc4.a_scale.item()))))

    def get_h_gelu_x_scale_shift(self, *, rounded: bool = False) -> int | torch.Tensor:
        if rounded:
            return _round_h_gelu_shift_value(float(self.h_gelu_shift_param.detach().cpu().item()))
        return self.h_gelu_shift_param

    def get_h_gelu_output_scale(self) -> float:
        shift = int(self.get_h_gelu_x_scale_shift(rounded=True))
        return 1.0 / float(1 << shift)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.gelu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


def build_compiler_ready_model(
    qat_model: TinyFeatureMLPQAT,
    *,
    dequantize_output: bool = True,
):
    model = copy.deepcopy(qat_model).cpu().eval()
    model.use_di_sigmoid_approx = False
    model.use_h_gelu_approx = False
    output_scales = infer_chain_output_scales(model, LAYER_ORDER)
    output_bits = infer_chain_output_bits(model, LAYER_ORDER)
    output_scales["fc3"] = qat_model.get_h_gelu_output_scale()
    output_bits["fc3"] = 16
    output_scales["fc4"] = FEATURE_SIGMOID_OUTPUT_SCALE
    output_bits["fc4"] = 16
    compiler_ready = convert_qat_model_for_compiler(
        model,
        layer_order=LAYER_ORDER,
        output_scales=output_scales,
        output_bits=output_bits,
        dequantize_output=dequantize_output,
    )
    setattr(compiler_ready.inner.fc3, "h_gelu_x_scale_shift", int(qat_model.get_h_gelu_x_scale_shift(rounded=True)))
    return compiler_ready


def load_qat_model_from_run(run_dir: str, *, device: str = "cpu") -> tuple[TinyFeatureMLPQAT, dict[str, object]]:
    run_path = Path(run_dir)
    compat_qat_path = run_path / "qat.compat.pt"
    with open(run_path / "summary.json", "r") as handle:
        summary = json.load(handle)
    layer_configs = layer_configs_from_summary(summary)
    task = str(summary.get("task", TASK_MULTICLASS))
    num_outputs = 1 if task == TASK_IS_ZERO else 10
    qat_model = TinyFeatureMLPQAT(
        layer_configs=layer_configs,
        num_outputs=num_outputs,
        use_di_sigmoid_approx=True,
        use_h_gelu_approx=True,
    ).to(device)
    load_path = compat_qat_path if compat_qat_path.exists() else (run_path / "qat.pt")
    state_dict = torch.load(load_path, map_location=device)
    try:
        qat_model.load_state_dict(state_dict)
    except RuntimeError:
        if load_path == compat_qat_path:
            raise
        qat_model.load_state_dict(state_dict, strict=False)
        torch.save(qat_model.state_dict(), compat_qat_path)
    return qat_model.eval(), summary


def build_compiled_artifact_from_run(
    run_dir: str,
    *,
    data_dir: str = DEFAULT_DATA_DIR,
    sample_index: int = 0,
    dequantize_output: bool = True,
):
    qat_model, summary = load_qat_model_from_run(run_dir, device="cpu")
    task = str(summary.get("task", TASK_MULTICLASS))
    _, _, _, _, test_ds = get_flat_mnist_loaders(data_dir, task=task)
    sample_image, sample_label = test_ds[sample_index]
    compiler_ready = build_compiler_ready_model(qat_model, dequantize_output=dequantize_output)
    artifact = compile_module(compiler_ready, (sample_image.unsqueeze(0),))
    return artifact, sample_image.unsqueeze(0).numpy(), int(sample_label), summary
