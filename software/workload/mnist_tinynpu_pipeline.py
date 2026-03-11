from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from software.compiler.tinynpu_jit import VerificationMode, compile_module, run_host_emulation
from software.compiler.tinynpu_quant import (
    apply_layer_quant_configs,
    build_mixed_precision_sensitivity_report,
    collect_layer_parameter_counts,
    collect_layer_quant_configs,
    QConv2d,
    QLinear,
    build_layer_config_map,
    collect_input_activation_maxes,
    collect_tensor_percentile_scale,
    convert_qat_model_for_compiler,
    copy_state_with_mapping,
    infer_chain_output_bits,
    infer_chain_output_scales,
    initialize_scale_tensors,
)


LAYER_NAMES = ["conv1", "conv2", "conv3", "fc"]


class TinyMNISTFP32(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1, bias=True)
        self.fc = nn.Linear(16, 10, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=[2, 3])
        x = self.fc(x)
        return x


class TinyMNISTQAT(nn.Module):
    def __init__(self, layer_configs=None):
        super().__init__()
        configs = build_layer_config_map(
            LAYER_NAMES,
            overrides=layer_configs,
            default_signed_activations=True,
        )
        self.conv1 = QConv2d(1, 16, 3, padding=1, config=configs["conv1"])
        self.conv2 = QConv2d(16, 16, 3, padding=1, config=configs["conv2"])
        self.conv3 = QConv2d(16, 16, 3, padding=1, config=configs["conv3"])
        self.fc = QLinear(16, 10, config=configs["fc"])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=[2, 3])
        x = self.fc(x)
        return x


def _mnist_present_locally(data_dir: str) -> bool:
    raw_dir = Path(data_dir) / "MNIST" / "raw"
    required = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    return all((raw_dir / name).exists() for name in required)


def get_mnist_loaders(data_dir: str = "./data", *, batch_size: int = 128, download: bool | None = None):
    transform = transforms.Compose([transforms.ToTensor()])
    should_download = not _mnist_present_locally(data_dir) if download is None else bool(download)
    train_ds = datasets.MNIST(data_dir, train=True, download=should_download, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=should_download, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    test_loader_single = DataLoader(test_ds, batch_size=1, shuffle=False)
    return train_loader, test_loader, test_loader_single, train_ds, test_ds


def evaluate_model(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            correct += int((model(x).argmax(1) == y).sum().item())
            total += int(y.numel())
    return correct / max(total, 1)


def evaluate_compiled_artifact(artifact, loader: DataLoader, *, max_samples: int | None = None) -> float:
    correct = 0
    total = 0
    input_name = artifact.plan.inputs[0]
    for x, y in loader:
        logits = run_host_emulation(
            artifact,
            {input_name: x.numpy()},
            VerificationMode.OFF,
            debug=False,
        ).tensors[artifact.plan.outputs[0]]
        pred = int(logits.reshape(-1).argmax())
        correct += int(pred == int(y.item()))
        total += 1
        if max_samples is not None and total >= max_samples:
            break
    return correct / max(total, 1)


def build_signed_w8a8_configs():
    return build_layer_config_map(
        LAYER_NAMES,
        default_w_bits=8,
        default_a_bits=8,
        default_signed_activations=True,
    )


def collect_relu3_boundary_scale(
    model: TinyMNISTQAT,
    calib_dataset,
    *,
    percentile: float = 99.9,
    max_samples: int = 256,
) -> float:
    def extract_relu3(module: TinyMNISTQAT, x: torch.Tensor) -> torch.Tensor:
        x1 = F.relu(module.conv1(x))
        x2 = F.relu(module.conv2(x1))
        x3 = F.relu(module.conv3(x2))
        return x3

    return collect_tensor_percentile_scale(
        model,
        calib_dataset,
        extractor=extract_relu3,
        percentile=percentile,
        max_samples=max_samples,
        qmax=127.0,
        floor=1e-8,
    )


def build_compiler_ready_mnist_model(
    qat_model: TinyMNISTQAT,
    *,
    calib_dataset,
    mixed_precision_configs=None,
    dequantize_output: bool = True,
):
    if mixed_precision_configs is not None:
        qat_model = apply_layer_quant_configs(qat_model, mixed_precision_configs, inplace=False)
    output_scales = infer_chain_output_scales(qat_model, LAYER_NAMES)
    output_bits = infer_chain_output_bits(qat_model, LAYER_NAMES)
    output_scales["conv3"] = collect_relu3_boundary_scale(qat_model, calib_dataset)
    output_bits["conv3"] = 8
    return convert_qat_model_for_compiler(
        qat_model.cpu().eval(),
        layer_order=LAYER_NAMES,
        output_scales=output_scales,
        output_bits=output_bits,
        dequantize_output=dequantize_output,
    )


def build_mnist_mixed_precision_report(
    qat_model: TinyMNISTQAT,
    *,
    validation_loader: DataLoader,
    device: str,
    candidate_bits: tuple[tuple[int, int], ...] = ((16, 16), (8, 8), (4, 4)),
    max_acceptable_drop: float = 0.01,
):
    baseline_configs = collect_layer_quant_configs(qat_model, LAYER_NAMES)
    parameter_counts = collect_layer_parameter_counts(qat_model, LAYER_NAMES)

    def evaluate_configs(configs):
        candidate_model = apply_layer_quant_configs(qat_model, configs, inplace=False).to(device).eval()
        return evaluate_model(candidate_model, validation_loader, device)

    return build_mixed_precision_sensitivity_report(
        LAYER_NAMES,
        evaluate_configs=evaluate_configs,
        baseline_configs=baseline_configs,
        candidate_bits=candidate_bits,
        max_acceptable_drop=max_acceptable_drop,
        parameter_counts=parameter_counts,
    )


def initialize_qat_from_fp32(
    fp32_model: TinyMNISTFP32,
    *,
    layer_configs,
    calib_loader: DataLoader,
    device: str,
) -> TinyMNISTQAT:
    fp32_model.eval()
    act_maxes = collect_input_activation_maxes(fp32_model, calib_loader, LAYER_NAMES, device=device)
    qat_model = TinyMNISTQAT(layer_configs).to(device)

    fp32_sd = fp32_model.state_dict()
    qat_sd = qat_model.state_dict()
    copy_state_with_mapping(
        dst_state_dict=qat_sd,
        src_state_dict=fp32_sd,
        key_mapping={
            "conv1.conv.weight": "conv1.weight",
            "conv1.conv.bias": "conv1.bias",
            "conv2.conv.weight": "conv2.weight",
            "conv2.conv.bias": "conv2.bias",
            "conv3.conv.weight": "conv3.weight",
            "conv3.conv.bias": "conv3.bias",
            "fc.linear.weight": "fc.weight",
            "fc.linear.bias": "fc.bias",
        },
    )
    initialize_scale_tensors(
        qat_state_dict=qat_sd,
        fp32_state_dict=fp32_sd,
        layer_configs=layer_configs,
        activation_maxes=act_maxes,
        fp32_weight_keys={
            "conv1": "conv1.weight",
            "conv2": "conv2.weight",
            "conv3": "conv3.weight",
            "fc": "fc.weight",
        },
    )
    qat_model.load_state_dict(qat_sd)
    return qat_model


def train_fp32(
    *,
    device: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
) -> TinyMNISTFP32:
    model = TinyMNISTFP32().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_state = None
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate_model(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"fp32 epoch {epoch + 1:02d}/{epochs} acc={acc * 100:.2f}% best={best_acc * 100:.2f}%")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def finetune_qat(
    qat_model: TinyMNISTQAT,
    *,
    device: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
) -> TinyMNISTQAT:
    params = [param for name, param in qat_model.named_parameters() if "scale" not in name]
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_state = None
    best_acc = evaluate_model(qat_model, test_loader, device)
    print(f"qat pre-finetune acc={best_acc * 100:.2f}%")
    for epoch in range(epochs):
        qat_model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(qat_model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate_model(qat_model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in qat_model.state_dict().items()}
        print(f"qat  epoch {epoch + 1:02d}/{epochs} acc={acc * 100:.2f}% best={best_acc * 100:.2f}%")

    if best_state is not None:
        qat_model.load_state_dict(best_state)
    return qat_model


def run_pipeline(
    *,
    run_dir: str,
    data_dir: str = "./data",
    fp32_epochs: int = 20,
    qat_epochs: int = 5,
    fp32_lr: float = 1e-3,
    qat_lr: float = 1e-4,
    compiled_eval_samples: int = 256,
):
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    train_loader, test_loader, test_loader_single, train_ds, test_ds = get_mnist_loaders(data_dir)
    layer_configs = build_signed_w8a8_configs()

    fp32_model = train_fp32(
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=fp32_epochs,
        lr=fp32_lr,
    )
    torch.save(fp32_model.state_dict(), run_path / "fp32.pt")
    fp32_acc = evaluate_model(fp32_model, test_loader, device)

    calib_loader = DataLoader(Subset(train_ds, range(1000)), batch_size=256, shuffle=False)
    qat_model = initialize_qat_from_fp32(
        fp32_model,
        layer_configs=layer_configs,
        calib_loader=calib_loader,
        device=device,
    )
    ptq_acc = evaluate_model(qat_model, test_loader, device)

    qat_model = finetune_qat(
        qat_model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=qat_epochs,
        lr=qat_lr,
    )
    torch.save(qat_model.state_dict(), run_path / "qat.pt")
    qat_acc = evaluate_model(qat_model, test_loader, device)

    compiler_ready = build_compiler_ready_mnist_model(
        qat_model,
        calib_dataset=test_ds,
    )
    example = test_ds[0][0].unsqueeze(0)
    artifact = compile_module(compiler_ready, (example,))

    compiled_acc = evaluate_compiled_artifact(
        artifact,
        test_loader_single,
        max_samples=compiled_eval_samples,
    )

    sample_image, sample_label = test_ds[0]
    sample_result = run_host_emulation(
        artifact,
        {artifact.plan.inputs[0]: sample_image.unsqueeze(0).numpy()},
        VerificationMode.DEBUG,
        debug=True,
    )
    sample_logits = sample_result.tensors[artifact.plan.outputs[0]].reshape(-1).tolist()
    sample_pred = int(torch.tensor(sample_logits).argmax().item())

    summary = {
        "device": device,
        "fp32_acc": fp32_acc,
        "ptq_acc": ptq_acc,
        "qat_acc": qat_acc,
        "compiled_host_acc_samples": compiled_eval_samples,
        "compiled_host_acc": compiled_acc,
        "sample_label": int(sample_label),
        "sample_pred": sample_pred,
        "sample_logits": sample_logits,
        "debug_kinds": [event["kind"] for event in sample_result.debug_trace],
    }
    with open(run_path / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary, artifact


def load_qat_model_from_run(
    run_dir: str,
    *,
    device: str = "cpu",
    layer_configs=None,
) -> TinyMNISTQAT:
    run_path = Path(run_dir)
    qat_path = run_path / "qat.pt"
    if not qat_path.exists():
        raise FileNotFoundError(f"Expected trained QAT checkpoint at {qat_path}.")
    layer_configs = layer_configs or build_signed_w8a8_configs()
    model = TinyMNISTQAT(layer_configs).to(device)
    state_dict = torch.load(qat_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.eval()


def build_compiled_artifact_from_run(
    run_dir: str,
    *,
    data_dir: str = "./data",
    sample_index: int = 0,
    dequantize_output: bool = True,
):
    _, _, _, _, test_ds = get_mnist_loaders(data_dir)
    qat_model = load_qat_model_from_run(run_dir, device="cpu")
    compiler_ready = build_compiler_ready_mnist_model(
        qat_model,
        calib_dataset=test_ds,
        dequantize_output=dequantize_output,
    )
    sample_image, sample_label = test_ds[int(sample_index)]
    example = sample_image.unsqueeze(0)
    artifact = compile_module(compiler_ready, (example,))
    return artifact, example.numpy(), int(sample_label)


def run_compiled_sample_from_run(
    run_dir: str,
    *,
    data_dir: str = "./data",
    sample_index: int = 0,
    dequantize_output: bool = True,
    verification: VerificationMode = VerificationMode.DEBUG,
    debug: bool = True,
):
    artifact, example, label = build_compiled_artifact_from_run(
        run_dir,
        data_dir=data_dir,
        sample_index=sample_index,
        dequantize_output=dequantize_output,
    )
    result = run_host_emulation(
        artifact,
        {artifact.plan.inputs[0]: example},
        verification,
        debug=debug,
    )
    logits = result.tensors[artifact.plan.outputs[0]].reshape(-1)
    return {
        "artifact": artifact,
        "result": result,
        "label": label,
        "prediction": int(torch.tensor(logits).argmax().item()),
        "logits": logits.tolist(),
    }


if __name__ == "__main__":
    summary, _ = run_pipeline(run_dir="runs/mnist_tinynpu_pipeline")
    print(json.dumps(summary, indent=2))
