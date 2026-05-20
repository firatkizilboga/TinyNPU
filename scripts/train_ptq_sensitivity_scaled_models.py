from __future__ import annotations

import argparse
import gzip
import json
import random
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
RUNS_DIR = REPO_ROOT / "runs"


def _read_idx_bytes(path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    gz_path = Path(str(path) + ".gz")
    if gz_path.exists():
        with gzip.open(gz_path, "rb") as handle:
            return handle.read()
    raise FileNotFoundError(f"Missing MNIST IDX file: {path}")


def load_mnist_images(data_dir: Path, *, train: bool, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    raw_dir = data_dir / "MNIST" / "raw"
    prefix = "train" if train else "t10k"
    image_blob = _read_idx_bytes(raw_dir / f"{prefix}-images-idx3-ubyte")
    label_blob = _read_idx_bytes(raw_dir / f"{prefix}-labels-idx1-ubyte")
    magic, count, rows, cols = struct.unpack_from(">IIII", image_blob, 0)
    label_magic, label_count = struct.unpack_from(">II", label_blob, 0)
    if magic != 2051 or label_magic != 2049 or count != label_count:
        raise ValueError("Invalid MNIST IDX files.")
    images_u8 = torch.frombuffer(image_blob, dtype=torch.uint8, offset=16)
    images = images_u8.reshape(count, 1, rows, cols).float() / 255.0
    if image_size != rows or image_size != cols:
        images = F.interpolate(images, size=(image_size, image_size), mode="bilinear", align_corners=False)
    labels = torch.frombuffer(label_blob, dtype=torch.uint8, offset=8).long()
    labels = (labels == 0).float().view(-1, 1)
    return images, labels


def balanced_indices(labels: torch.Tensor, *, seed: int, max_per_class: int | None = None) -> list[int]:
    rng = random.Random(seed)
    flat = labels.view(-1)
    pos = [int(i) for i in torch.nonzero(flat == 1, as_tuple=False).view(-1).tolist()]
    neg = [int(i) for i in torch.nonzero(flat == 0, as_tuple=False).view(-1).tolist()]
    rng.shuffle(pos)
    rng.shuffle(neg)
    n = min(len(pos), len(neg))
    if max_per_class is not None:
        n = min(n, int(max_per_class))
    indices = pos[:n] + neg[:n]
    rng.shuffle(indices)
    return indices


class Wide32ConvBinary(nn.Module):
    layer_names = ("conv1", "conv2", "conv3", "conv4")

    def __init__(self, channels: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, 3)
        self.conv2 = nn.Conv2d(channels, channels, 3)
        self.conv3 = nn.Conv2d(channels, channels, 3)
        self.conv4 = nn.Conv2d(channels, 1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x.flatten(1)


class MLPH256Binary(nn.Module):
    layer_names = ("fc1", "fc2", "fc3", "fc4")

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(64, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


def _qmax(bits: int, *, signed: bool = True) -> int:
    return (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1


def fake_quant(x: torch.Tensor, *, bits: int, scale: float, signed: bool = True) -> torch.Tensor:
    qmax = _qmax(bits, signed=signed)
    qmin = -qmax - 1 if signed else 0
    s = max(float(scale), 1.0e-8)
    return torch.clamp(torch.round(x / s), qmin, qmax) * s


@dataclass(frozen=True)
class PTQConfig:
    weight_bits: dict[str, int]
    activation_bits: dict[str, int]
    weight_scales: dict[str, float]
    activation_scales: dict[str, float]


class PTQWrapper(nn.Module):
    def __init__(self, model: nn.Module, config: PTQConfig):
        super().__init__()
        self.model = model
        self.config = config

    def _linear(self, layer: nn.Linear, name: str, x: torch.Tensor) -> torch.Tensor:
        xq = fake_quant(x, bits=self.config.activation_bits[name], scale=self.config.activation_scales[name])
        wq = fake_quant(layer.weight, bits=self.config.weight_bits[name], scale=self.config.weight_scales[name])
        return F.linear(xq, wq, layer.bias)

    def _conv(self, layer: nn.Conv2d, name: str, x: torch.Tensor) -> torch.Tensor:
        xq = fake_quant(x, bits=self.config.activation_bits[name], scale=self.config.activation_scales[name])
        wq = fake_quant(layer.weight, bits=self.config.weight_bits[name], scale=self.config.weight_scales[name])
        return F.conv2d(xq, wq, layer.bias, stride=layer.stride, padding=layer.padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.model
        if isinstance(m, Wide32ConvBinary):
            x = F.relu(self._conv(m.conv1, "conv1", x))
            x = F.relu(self._conv(m.conv2, "conv2", x))
            x = F.relu(self._conv(m.conv3, "conv3", x))
            x = self._conv(m.conv4, "conv4", x)
            return x.flatten(1)
        if isinstance(m, MLPH256Binary):
            x = x.flatten(1)
            x = F.relu(self._linear(m.fc1, "fc1", x))
            x = F.relu(self._linear(m.fc2, "fc2", x))
            x = F.relu(self._linear(m.fc3, "fc3", x))
            return self._linear(m.fc4, "fc4", x)
        raise TypeError(f"Unsupported model type: {type(m).__name__}")


@torch.no_grad()
def calibrate(model: nn.Module, loader: DataLoader, layer_names: Iterable[str], *, device: torch.device) -> dict[str, float]:
    model.eval()
    maxes = {name: 0.0 for name in layer_names}
    hooks = []
    modules = dict(model.named_modules())

    def make_hook(name: str):
        def hook(_module, inputs, _output):
            maxes[name] = max(maxes[name], float(inputs[0].detach().abs().max().item()))

        return hook

    for name in layer_names:
        hooks.append(modules[name].register_forward_hook(make_hook(name)))
    try:
        for x, _y in loader:
            model(x.to(device))
    finally:
        for hook in hooks:
            hook.remove()
    return {name: max(value, 1.0e-8) for name, value in maxes.items()}


def make_ptq_config(
    model: nn.Module,
    layer_names: tuple[str, ...],
    activation_maxes: dict[str, float],
    *,
    default_bits: int,
    overrides: dict[str, tuple[int, int]] | None = None,
) -> PTQConfig:
    overrides = overrides or {}
    weight_bits: dict[str, int] = {}
    activation_bits: dict[str, int] = {}
    weight_scales: dict[str, float] = {}
    activation_scales: dict[str, float] = {}
    modules = dict(model.named_modules())
    for name in layer_names:
        w_bits, a_bits = overrides.get(name, (default_bits, default_bits))
        weight_bits[name] = int(w_bits)
        activation_bits[name] = int(a_bits)
        module = modules[name]
        weight = module.weight
        weight_scales[name] = max(float(weight.detach().abs().max().item()) / float(_qmax(w_bits)), 1.0e-8)
        activation_scales[name] = max(float(activation_maxes[name]) / float(_qmax(a_bits)), 1.0e-8)
    return PTQConfig(weight_bits, activation_bits, weight_scales, activation_scales)


def evaluate(model: nn.Module, loader: DataLoader, *, device: torch.device) -> dict[str, float]:
    model.eval()
    total = 0
    correct = 0
    pos_total = 0
    pos_correct = 0
    neg_total = 0
    neg_correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss_sum += float(F.binary_cross_entropy_with_logits(logits, y, reduction="sum").item())
            pred = (torch.sigmoid(logits) >= 0.5).float()
            match = pred.eq(y)
            total += int(y.numel())
            correct += int(match.sum().item())
            pos = y.eq(1)
            neg = y.eq(0)
            pos_total += int(pos.sum().item())
            neg_total += int(neg.sum().item())
            pos_correct += int((match & pos).sum().item())
            neg_correct += int((match & neg).sum().item())
    pos_acc = pos_correct / max(pos_total, 1)
    neg_acc = neg_correct / max(neg_total, 1)
    return {
        "accuracy": correct / max(total, 1),
        "balanced_accuracy": 0.5 * (pos_acc + neg_acc),
        "pos_accuracy": pos_acc,
        "neg_accuracy": neg_acc,
        "loss": loss_sum / max(total, 1),
        "samples": float(total),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    *,
    device: torch.device,
    epochs: int,
    lr: float,
) -> dict[str, float]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1.0e-4)
    best_state = None
    best_bal = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(model(x), y)
            loss.backward()
            opt.step()
        metrics = evaluate(model, valid_loader, device=device)
        if metrics["balanced_accuracy"] > best_bal:
            best_bal = metrics["balanced_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"epoch={epoch} valid_acc={metrics['accuracy']:.4f} "
            f"valid_bal={metrics['balanced_accuracy']:.4f} loss={metrics['loss']:.4f}",
            flush=True,
        )
    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate(model, valid_loader, device=device)


def sensitivity(
    model: nn.Module,
    layer_names: tuple[str, ...],
    activation_maxes: dict[str, float],
    loader: DataLoader,
    *,
    device: torch.device,
    baseline_bits: int,
    trial_bits: tuple[int, ...],
) -> tuple[dict[str, dict[str, float]], dict[int, dict[str, float]]]:
    full: dict[int, dict[str, float]] = {}
    per_layer: dict[str, dict[str, float]] = {}
    for bits in trial_bits:
        cfg = make_ptq_config(model, layer_names, activation_maxes, default_bits=bits)
        full[bits] = evaluate(PTQWrapper(model, cfg).to(device), loader, device=device)
    baseline_cfg = make_ptq_config(model, layer_names, activation_maxes, default_bits=baseline_bits)
    baseline_metrics = evaluate(PTQWrapper(model, baseline_cfg).to(device), loader, device=device)
    base_bal = baseline_metrics["balanced_accuracy"]
    for name in layer_names:
        layer_result: dict[str, float] = {}
        for bits in trial_bits:
            if bits == baseline_bits:
                continue
            cfg = make_ptq_config(
                model,
                layer_names,
                activation_maxes,
                default_bits=baseline_bits,
                overrides={name: (bits, bits)},
            )
            metrics = evaluate(PTQWrapper(model, cfg).to(device), loader, device=device)
            layer_result[f"{bits}bit_balanced_accuracy"] = metrics["balanced_accuracy"]
            layer_result[f"{bits}bit_drop_vs_int{baseline_bits}"] = base_bal - metrics["balanced_accuracy"]
        per_layer[name] = layer_result
    return per_layer, full


def format_pct(value: float) -> str:
    return f"{100.0 * float(value):.2f}%"


def write_report(path: Path, result: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Trained PTQ Accuracy and Sensitivity", ""]
    lines.append("Dataset: MNIST binary `is_zero`, trained and evaluated on balanced zero/nonzero splits.")
    lines.append("")
    lines.append("## Accuracy")
    lines.append("")
    lines.append("| Model | FP32 balanced | FP32 full-test | INT16 PTQ balanced | INT8 PTQ balanced | INT4 PTQ balanced |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for model_name, model_result in result["models"].items():
        fp_bal = model_result["fp32_balanced"]["balanced_accuracy"]
        fp_full = model_result["fp32_full"]["accuracy"]
        ptq = model_result["ptq_full_precision"]
        lines.append(
            f"| {model_name} | {format_pct(fp_bal)} | {format_pct(fp_full)} | "
            f"{format_pct(ptq['16']['balanced_accuracy'])} | "
            f"{format_pct(ptq['8']['balanced_accuracy'])} | "
            f"{format_pct(ptq['4']['balanced_accuracy'])} |"
        )
    lines.append("")
    lines.append("## Per-Layer Sensitivity")
    for model_name, model_result in result["models"].items():
        lines.append("")
        lines.append(f"### {model_name}")
        lines.append("")
        lines.append("| Layer | INT8 one-layer balanced | INT8 drop vs INT16 | INT4 one-layer balanced | INT4 drop vs INT16 |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for layer, row in model_result["sensitivity"].items():
            lines.append(
                f"| {layer} | {format_pct(row['8bit_balanced_accuracy'])} | "
                f"{format_pct(row['8bit_drop_vs_int16'])} | "
                f"{format_pct(row['4bit_balanced_accuracy'])} | "
                f"{format_pct(row['4bit_drop_vs_int16'])} |"
            )
    lines.append("")
    lines.append("## Raw JSON")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(result, indent=2, sort_keys=True))
    lines.append("```")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=RUNS_DIR / "trained_scaled_ptq_2026_05_20")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--calib-samples-per-class", type=int, default=512)
    parser.add_argument("--train-samples-per-class", type=int, default=5000)
    parser.add_argument("--valid-samples-per-class", type=int, default=980)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)

    train_x, train_y = load_mnist_images(args.data_dir, train=True, image_size=8)
    test_x, test_y = load_mnist_images(args.data_dir, train=False, image_size=8)
    train_ds = TensorDataset(train_x, train_y)
    test_ds = TensorDataset(test_x, test_y)

    train_bal = Subset(train_ds, balanced_indices(train_y, seed=args.seed, max_per_class=args.train_samples_per_class))
    valid_bal = Subset(test_ds, balanced_indices(test_y, seed=args.seed + 1, max_per_class=args.valid_samples_per_class))
    calib_bal = Subset(train_ds, balanced_indices(train_y, seed=args.seed + 2, max_per_class=args.calib_samples_per_class))

    train_loader = DataLoader(train_bal, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_bal, batch_size=args.batch_size, shuffle=False)
    full_test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    calib_loader = DataLoader(calib_bal, batch_size=args.batch_size, shuffle=False)

    jobs = {
        "Conv wide32": Wide32ConvBinary(32),
        "MLP h256": MLPH256Binary(256),
    }
    result: dict[str, object] = {
        "seed": args.seed,
        "epochs": args.epochs,
        "train_samples": len(train_bal),
        "balanced_valid_samples": len(valid_bal),
        "full_test_samples": len(test_ds),
        "models": {},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for model_name, model in jobs.items():
        print(f"=== training {model_name} ===", flush=True)
        layer_names = tuple(model.layer_names)
        train_model(model, train_loader, valid_loader, device=device, epochs=args.epochs, lr=1.0e-3)
        model.cpu().eval()
        torch.save(model.state_dict(), args.output_dir / f"{model_name.lower().replace(' ', '_')}_fp32.pt")
        fp32_balanced = evaluate(model.to(device), valid_loader, device=device)
        fp32_full = evaluate(model.to(device), full_test_loader, device=device)
        activation_maxes = calibrate(model.to(device), calib_loader, layer_names, device=device)
        sens, full_ptq = sensitivity(
            model.to(device),
            layer_names,
            activation_maxes,
            valid_loader,
            device=device,
            baseline_bits=16,
            trial_bits=(16, 8, 4),
        )
        result["models"][model_name] = {
            "fp32_balanced": fp32_balanced,
            "fp32_full": fp32_full,
            "activation_maxes": activation_maxes,
            "ptq_full_precision": {str(bits): metrics for bits, metrics in full_ptq.items()},
            "sensitivity": sens,
        }
        print(
            f"{model_name}: fp32_bal={fp32_balanced['balanced_accuracy']:.4f} "
            f"int16_bal={full_ptq[16]['balanced_accuracy']:.4f} "
            f"int8_bal={full_ptq[8]['balanced_accuracy']:.4f} "
            f"int4_bal={full_ptq[4]['balanced_accuracy']:.4f}",
            flush=True,
        )

    json_path = args.output_dir / "results.json"
    json_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    report_path = RUNS_DIR / "TRAINED_SCALED_PTQ_ACCURACY_2026_05_20.md"
    write_report(report_path, result)
    print(f"json={json_path}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
