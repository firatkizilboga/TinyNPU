from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPILER_ROOT = PROJECT_ROOT / "software" / "compiler"
for path in (PROJECT_ROOT, COMPILER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from software.compiler.tinynpu_jit import VerificationMode, compile_module, run_host_emulation  # noqa: E402
from software.compiler.tinynpu_jit.benchmark import (  # noqa: E402
    five_stage_in_order_model,
    ideal_issue_1_model,
    unpipelined_scalar_model,
)
from software.compiler.tinynpu_quant import (  # noqa: E402
    LayerQuantConfig,
    QConv2d,
    build_layer_config_map,
    collect_input_activation_maxes,
    convert_qat_model_for_compiler,
    copy_state_with_mapping,
    infer_chain_output_bits,
    infer_chain_output_scales,
    initialize_scale_tensors,
)
from software.workload.mnist_tinynpu_pipeline import (  # noqa: E402
    _apply_activation,
    _apply_di_sigmoid_qat,
)


DEFAULT_DATA_DIR = "/home/firatkizilboga/compiler-optimization/data"
TASK_IS_ZERO = "is_zero"
LAYER_ORDER = ["conv1", "conv2", "conv3", "conv4"]
IMAGE_SIZE = 8
HIDDEN_CH = 16
FEATURE_SIGMOID_P_OUT = 16
FEATURE_SIGMOID_OUTPUT_SCALE = 1.0 / 32767.0


def get_conv_mnist_loaders(
    data_dir: str = DEFAULT_DATA_DIR,
    *,
    image_size: int = IMAGE_SIZE,
    batch_size: int = 128,
    download: bool = False,
):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    target_transform = lambda y: int(y == 0)
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


def build_conv_feature_configs(preset: str = "int16") -> dict[str, LayerQuantConfig]:
    preset = str(preset).lower()
    if preset == "int16":
        bits = 16
    elif preset == "int8":
        bits = 8
    else:
        raise ValueError(f"Unsupported preset {preset!r}.")
    return build_layer_config_map(
        LAYER_ORDER,
        default_w_bits=bits,
        default_a_bits=bits,
        default_signed_activations=True,
    )


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


class TinyConvFeatureFP32(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, HIDDEN_CH, 3, padding=0, bias=True)
        self.conv2 = nn.Conv2d(HIDDEN_CH, HIDDEN_CH, 3, padding=0, bias=True)
        self.conv3 = nn.Conv2d(HIDDEN_CH, HIDDEN_CH, 3, padding=0, bias=True)
        self.conv4 = nn.Conv2d(HIDDEN_CH, 1, 2, padding=0, bias=True)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x))
        return x


class TinyConvFeatureQAT(nn.Module):
    def __init__(
        self,
        layer_configs: dict[str, LayerQuantConfig] | None = None,
        *,
        use_di_sigmoid_approx: bool = True,
    ):
        super().__init__()
        self.use_di_sigmoid_approx = bool(use_di_sigmoid_approx)
        configs = build_conv_feature_configs("int16") if layer_configs is None else layer_configs
        self.conv1 = QConv2d(1, HIDDEN_CH, 3, padding=0, config=configs["conv1"])
        self.conv2 = QConv2d(HIDDEN_CH, HIDDEN_CH, 3, padding=0, config=configs["conv2"])
        self.conv3 = QConv2d(HIDDEN_CH, HIDDEN_CH, 3, padding=0, config=configs["conv3"])
        self.conv4 = QConv2d(HIDDEN_CH, 1, 2, padding=0, config=configs["conv4"])

    def forward(self, x):
        x = _apply_activation(self.conv1(x), "relu")
        x = _apply_activation(self.conv2(x), "relu")
        x = _apply_activation(self.conv3(x), "relu")
        x = self.conv4(x)
        if self.use_di_sigmoid_approx:
            x = _apply_di_sigmoid_qat(
                x,
                layer=self.conv4,
                output_scale=FEATURE_SIGMOID_OUTPUT_SCALE,
                p_out=FEATURE_SIGMOID_P_OUT,
            )
        else:
            x = torch.sigmoid(x)
        return x


def conv_feature_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_float = target.to(output.dtype).view(-1)
    return F.binary_cross_entropy(output.view(-1), target_float)


def conv_feature_correct_predictions(output: torch.Tensor, target: torch.Tensor) -> int:
    preds = (output.view(-1) >= 0.5).to(target.dtype)
    return int((preds == target.view(-1)).sum().item())


def evaluate_conv_feature_model(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    *,
    progress_label: str | None = None,
) -> float:
    model.eval()
    correct = 0
    total = 0
    num_batches = len(loader) if hasattr(loader, "__len__") else None
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader, start=1):
            x = x.to(device)
            y = y.to(device)
            correct += conv_feature_correct_predictions(model(x), y)
            total += int(y.numel())
            if progress_label and (batch_idx == 1 or batch_idx % 10 == 0 or (num_batches is not None and batch_idx == num_batches)):
                running = 100.0 * correct / max(total, 1)
                print(f"{progress_label} {batch_idx}/{num_batches} running_acc={running:.2f}%", flush=True)
    return correct / max(total, 1)


def evaluate_conv_feature_compiled_artifact(
    artifact,
    loader: DataLoader,
    *,
    max_samples: int | None = None,
    progress_label: str | None = None,
) -> float:
    correct = 0
    total = 0
    input_name = artifact.plan.inputs[0]
    num_batches = min(int(max_samples), len(loader)) if max_samples is not None and hasattr(loader, "__len__") else len(loader)
    for sample_idx, (x, y) in enumerate(loader, start=1):
        result = run_host_emulation(
            artifact,
            {input_name: x.numpy()},
            VerificationMode.OFF,
            debug=False,
        ).tensors[artifact.plan.outputs[0]]
        pred = int(float(result.reshape(-1)[0]) >= 0.5)
        correct += int(pred == int(y.item()))
        total += 1
        if progress_label and (sample_idx == 1 or sample_idx % 16 == 0 or (max_samples is not None and sample_idx == max_samples)):
            running = 100.0 * correct / max(total, 1)
            print(f"{progress_label} {sample_idx}/{num_batches} running_acc={running:.2f}%", flush=True)
        if max_samples is not None and sample_idx >= int(max_samples):
            break
    return correct / max(total, 1)


def train_fp32_model(
    *,
    device: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
) -> TinyConvFeatureFP32:
    model = TinyConvFeatureFP32().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    best_state = None
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = conv_feature_loss(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate_conv_feature_model(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"fp32 epoch {epoch + 1:02d}/{epochs} acc={acc * 100:.2f}% best={best_acc * 100:.2f}%", flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def initialize_qat_from_fp32(
    fp32_model: TinyConvFeatureFP32,
    *,
    layer_configs: dict[str, LayerQuantConfig],
    calib_loader: DataLoader,
    device: str,
) -> TinyConvFeatureQAT:
    fp32_model.eval()
    act_maxes = collect_input_activation_maxes(fp32_model, calib_loader, LAYER_ORDER, device=device)
    qat_model = TinyConvFeatureQAT(layer_configs=layer_configs).to(device)

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
            "conv4.conv.weight": "conv4.weight",
            "conv4.conv.bias": "conv4.bias",
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
            "conv4": "conv4.weight",
        },
    )
    qat_model.load_state_dict(qat_sd)
    return qat_model


def finetune_qat_model(
    qat_model: TinyConvFeatureQAT,
    *,
    device: str,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    lr: float,
) -> TinyConvFeatureQAT:
    params = [param for name, param in qat_model.named_parameters() if "scale" not in name]
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    best_state = {k: v.detach().cpu().clone() for k, v in qat_model.state_dict().items()}
    best_acc = evaluate_conv_feature_model(qat_model, test_loader, device)
    print(f"qat pre-finetune acc={best_acc * 100:.2f}%", flush=True)
    for epoch in range(epochs):
        qat_model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = conv_feature_loss(qat_model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = evaluate_conv_feature_model(qat_model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu().clone() for k, v in qat_model.state_dict().items()}
        print(f"qat  epoch {epoch + 1:02d}/{epochs} acc={acc * 100:.2f}% best={best_acc * 100:.2f}%", flush=True)
    if best_state is not None:
        qat_model.load_state_dict(best_state)
    return qat_model


def build_compiler_ready_model(
    qat_model: TinyConvFeatureQAT,
    *,
    dequantize_output: bool = True,
):
    model = copy.deepcopy(qat_model).cpu().eval()
    model.use_di_sigmoid_approx = False
    output_scales = infer_chain_output_scales(model, LAYER_ORDER)
    output_bits = infer_chain_output_bits(model, LAYER_ORDER)
    output_scales["conv4"] = FEATURE_SIGMOID_OUTPUT_SCALE
    output_bits["conv4"] = FEATURE_SIGMOID_P_OUT
    return convert_qat_model_for_compiler(
        model,
        layer_order=LAYER_ORDER,
        output_scales=output_scales,
        output_bits=output_bits,
        dequantize_output=dequantize_output,
    )


def load_qat_model_from_run(run_dir: str, *, device: str = "cpu") -> tuple[TinyConvFeatureQAT, dict[str, object]]:
    run_path = Path(run_dir)
    compat_qat_path = run_path / "qat.compat.pt"
    with open(run_path / "summary.json", "r") as handle:
        summary = json.load(handle)
    layer_configs = layer_configs_from_summary(summary)
    qat_model = TinyConvFeatureQAT(layer_configs=layer_configs, use_di_sigmoid_approx=True).to(device)
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
    dequantize_output: bool = False,
):
    device = "cpu"
    qat_model, summary = load_qat_model_from_run(run_dir, device=device)
    _, _, _, _, test_ds = get_conv_mnist_loaders(data_dir)
    sample_image, sample_label = test_ds[sample_index]
    compiler_ready = build_compiler_ready_model(qat_model, dequantize_output=dequantize_output)
    artifact = compile_module(compiler_ready, (sample_image.unsqueeze(0),))
    return artifact, sample_image.unsqueeze(0).numpy(), int(sample_label), summary


def plan_stats(artifact) -> dict[str, object]:
    npu_segments = 0
    host_ops: list[str] = []
    activations: list[str] = []
    for step in artifact.plan.steps:
        if hasattr(step, "ops"):
            npu_segments += 1
            activations.extend(op.activation for op in step.ops)
        elif hasattr(step, "kind"):
            host_ops.append(step.kind)
    return {
        "npu_segments": npu_segments,
        "host_ops": host_ops,
        "host_op_count": len(host_ops),
        "activations": activations,
    }


def run_conv_feature_benchmark(
    *,
    run_dir: str,
    data_dir: str = DEFAULT_DATA_DIR,
    precision_preset: str = "int16",
    fp32_epochs: int = 5,
    qat_epochs: int = 1,
    qat_lr: float = 1e-4,
    compiled_eval_samples: int = 128,
    resume_fp32_from: str | None = None,
):
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}", flush=True)

    train_loader, test_loader, test_loader_single, train_ds, test_ds = get_conv_mnist_loaders(data_dir)
    layer_configs = build_conv_feature_configs(precision_preset)

    if resume_fp32_from:
        print(f"loading_fp32={resume_fp32_from}", flush=True)
        fp32_model = TinyConvFeatureFP32().to(device)
        fp32_model.load_state_dict(torch.load(resume_fp32_from, map_location=device))
        fp32_model.eval()
    else:
        print("stage=fp32_train", flush=True)
        fp32_model = train_fp32_model(
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=fp32_epochs,
            lr=1e-3,
        )
        torch.save(fp32_model.state_dict(), run_path / "fp32.pt")

    fp32_acc = evaluate_conv_feature_model(fp32_model, test_loader, device, progress_label="stage=fp32_eval")
    print(f"stage=fp32_eval acc={fp32_acc * 100:.2f}%", flush=True)

    calib_loader = DataLoader(Subset(train_ds, range(1000)), batch_size=256, shuffle=False)
    print("stage=qat_init", flush=True)
    qat_model = initialize_qat_from_fp32(
        fp32_model,
        layer_configs=layer_configs,
        calib_loader=calib_loader,
        device=device,
    )

    print("stage=ptq_eval", flush=True)
    ptq_acc = evaluate_conv_feature_model(qat_model, test_loader, device, progress_label="stage=ptq_eval")
    print(f"stage=ptq_eval_done acc={ptq_acc * 100:.2f}%", flush=True)

    print("stage=qat_finetune", flush=True)
    qat_model = finetune_qat_model(
        qat_model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=qat_epochs,
        lr=qat_lr,
    )
    torch.save(qat_model.state_dict(), run_path / "qat.pt")
    qat_acc = evaluate_conv_feature_model(qat_model, test_loader, device, progress_label="stage=qat_eval")
    print(f"stage=qat_eval acc={qat_acc * 100:.2f}%", flush=True)

    print("stage=compiler_ready", flush=True)
    compiler_ready = build_compiler_ready_model(qat_model)
    example = test_ds[0][0].unsqueeze(0)
    print("stage=compile_module", flush=True)
    artifact = compile_module(compiler_ready, (example,))

    stats = plan_stats(artifact)

    print("stage=compiled_eval", flush=True)
    compiled_acc = evaluate_conv_feature_compiled_artifact(
        artifact,
        test_loader_single,
        max_samples=compiled_eval_samples,
        progress_label="stage=compiled_eval",
    )
    print(f"stage=compiled_eval_done acc={compiled_acc * 100:.2f}%", flush=True)

    sample_image, sample_label = test_ds[0]
    print("stage=benchmark_sample", flush=True)
    benchmark_result = run_host_emulation(
        artifact,
        {artifact.plan.inputs[0]: sample_image.unsqueeze(0).numpy()},
        VerificationMode.OFF,
        debug=False,
        benchmark=True,
    )
    benchmark_report = benchmark_result.benchmark
    benchmark_payload = benchmark_report.to_dict() if benchmark_report is not None else None
    benchmark_comparison = (
        benchmark_report.model_comparison(
            [unpipelined_scalar_model(), ideal_issue_1_model(), five_stage_in_order_model()]
        )
        if benchmark_report is not None
        else None
    )

    summary = {
        "device": device,
        "task": TASK_IS_ZERO,
        "precision_preset": precision_preset,
        "layer_bits": {
            name: {"w_bits": int(cfg.w_bits), "a_bits": int(cfg.a_bits)}
            for name, cfg in layer_configs.items()
        },
        "fp32_acc": fp32_acc,
        "ptq_acc": ptq_acc,
        "qat_acc": qat_acc,
        "compiled_host_acc_samples": compiled_eval_samples,
        "compiled_host_acc": compiled_acc,
        **stats,
        "benchmark": benchmark_payload,
        "benchmark_comparison": benchmark_comparison,
        "sample_label": int(sample_label),
    }
    with open(run_path / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary, artifact


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a conv-heavy TinyNPU benchmark model on downsampled MNIST.")
    parser.add_argument("--run-dir", default="runs/mnist_conv_feature_benchmark", help="Output directory for checkpoints and summaries.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="MNIST data directory.")
    parser.add_argument("--precision-preset", default="int16", choices=["int8", "int16"])
    parser.add_argument("--fp32-epochs", type=int, default=5)
    parser.add_argument("--qat-epochs", type=int, default=1)
    parser.add_argument("--qat-lr", type=float, default=1e-4)
    parser.add_argument("--compiled-eval-samples", type=int, default=128)
    parser.add_argument("--resume-fp32-from", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    summary, _ = run_conv_feature_benchmark(
        run_dir=args.run_dir,
        data_dir=args.data_dir,
        precision_preset=args.precision_preset,
        fp32_epochs=args.fp32_epochs,
        qat_epochs=args.qat_epochs,
        qat_lr=args.qat_lr,
        compiled_eval_samples=args.compiled_eval_samples,
        resume_fp32_from=args.resume_fp32_from,
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
