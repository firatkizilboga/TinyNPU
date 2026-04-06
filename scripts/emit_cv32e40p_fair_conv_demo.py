from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
COMPILER_ROOT = ROOT / "software" / "compiler"
for path in (ROOT, COMPILER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from software.compiler.tinynpu_jit import VerificationMode, compile_module, run_host_emulation, write_cv32e40p_c  # noqa: E402
from software.compiler.tinynpu_quant import LayerQuantConfig, QConv2d, build_layer_config_map, convert_qat_model_for_compiler  # noqa: E402


RUN_DIR = ROOT / "runs" / "tinynpu_issue27_backup_2026_03_21" / "mnist_conv_feature_benchmark_int16_iszero"
DEFAULT_DATA_DIR = ROOT / "data"
LAYER_ORDER = ["conv1", "conv2", "conv3", "conv4"]


class FairFightConvQAT(nn.Module):
    def __init__(self, layer_configs: dict[str, LayerQuantConfig]):
        super().__init__()
        self.conv1 = QConv2d(1, 16, 3, padding=0, config=layer_configs["conv1"])
        self.conv2 = QConv2d(16, 16, 3, padding=0, config=layer_configs["conv2"])
        self.conv3 = QConv2d(16, 16, 3, padding=0, config=layer_configs["conv3"])
        self.conv4 = QConv2d(16, 1, 2, padding=0, config=layer_configs["conv4"])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x


def _load_summary() -> dict[str, object]:
    with open(RUN_DIR / "summary.json", "r") as handle:
        return json.load(handle)


def _layer_configs_from_summary(summary: dict[str, object]) -> dict[str, LayerQuantConfig]:
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


def _load_model() -> tuple[FairFightConvQAT, dict[str, object]]:
    summary = _load_summary()
    model = FairFightConvQAT(_layer_configs_from_summary(summary)).cpu().eval()
    state_dict = torch.load(RUN_DIR / "qat.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    return model.eval(), summary


def _get_test_sample(*, data_dir: Path, sample_index: int = 0) -> tuple[torch.Tensor, int]:
    transform = transforms.Compose(
        [
            transforms.Resize((8, 8)),
            transforms.ToTensor(),
        ]
    )
    test_ds = datasets.MNIST(
        str(data_dir),
        train=False,
        download=False,
        transform=transform,
        target_transform=lambda y: int(y == 0),
    )
    image, label = test_ds[sample_index]
    return image.unsqueeze(0), int(label)


def main() -> None:
    model, summary = _load_model()
    sample_image, sample_label = _get_test_sample(data_dir=DEFAULT_DATA_DIR)
    compiler_ready = convert_qat_model_for_compiler(
        model,
        layer_order=LAYER_ORDER,
        dequantize_output=True,
    )
    artifact = compile_module(compiler_ready, (sample_image,))
    inputs = {artifact.plan.inputs[0]: sample_image.numpy()}
    host_result = run_host_emulation(
        artifact,
        inputs,
        verification=VerificationMode.FINAL,
        debug=False,
    )

    generated_dir = ROOT / "generated"
    generated_dir.mkdir(exist_ok=True)
    output_path = generated_dir / "cv32e40p_fair_conv_demo.c"
    write_cv32e40p_c(
        artifact,
        inputs,
        output_path,
        program_name="cv32e40p_fair_conv_demo",
    )

    prediction = int(float(host_result.tensors[artifact.plan.outputs[0]].reshape(-1)[0]) >= 0.5)
    print(output_path)
    print(f"sample_label={sample_label}")
    print(f"sample_pred={prediction}")
    print(f"qat_acc={float(summary.get('qat_acc', 0.0))}")
    print(f"steps={[type(step).__name__ for step in artifact.plan.steps]}")


if __name__ == "__main__":
    main()
