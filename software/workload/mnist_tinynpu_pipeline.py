from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from software.compiler.tinynpu_jit import NpuSegment, VerificationMode, compile_module, run_host_emulation
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
    recalibrate_qat_scales,
    compute_fused_params,
    fake_quantize,
)
from software.compiler.tinynpu_jit.golden import di_sigmoid as di_sigmoid_int


LAYER_NAMES = ["conv1", "conv2", "conv3", "fc"]
SIGMOID_P_OUT = 8
SIGMOID_OUTPUT_SCALE = 1.0 / 127.0


class _DiSigmoidSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, input_scale: float, m_i: int, k_i: int, p_out: int):
        # Match deployed PPU semantics: sigmoid sees the requantized value after
        # the existing multiplier/shift stage, and that value is clamped to
        # signed INT16 before DI-Sigmoid is applied.
        qmax = (1 << (int(p_out) - 1)) - 1
        x_int = torch.round(x.detach() / float(input_scale)).clamp(-32768, 32767).to(torch.int32).cpu().numpy()
        y_int = np.array(
            [di_sigmoid_int(int(v), m_i=int(m_i), k_i=int(k_i), p_out=int(p_out)) for v in x_int.reshape(-1)],
            dtype=np.float32,
        ).reshape(x_int.shape)
        y = torch.from_numpy(y_int).to(device=x.device, dtype=x.dtype)
        return y / float(qmax)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


def _apply_di_sigmoid_qat(x: torch.Tensor, *, layer: QLinear, output_scale: float = SIGMOID_OUTPUT_SCALE) -> torch.Tensor:
    multiplier, shift = compute_fused_params(float(layer.w_scale.item()), float(layer.a_scale.item()), float(output_scale))
    return _DiSigmoidSTE.apply(x, float(output_scale), int(multiplier), int(shift), int(SIGMOID_P_OUT))


def _apply_qlinear_with_input_scale(layer: QLinear, x: torch.Tensor, input_scale: float) -> torch.Tensor:
    with torch.no_grad():
        w_qmax = 2 ** (layer.w_bits - 1) - 1
        layer.w_scale.data.copy_((layer.linear.weight.abs().max() / w_qmax).clamp(min=1e-8))
    x_q = fake_quantize(
        x,
        x.new_tensor(float(input_scale)),
        layer.a_bits,
        is_weight=False,
        signed_activations=layer.signed_activations,
    )
    w_q = fake_quantize(layer.linear.weight, layer.w_scale, layer.w_bits, is_weight=True)
    return F.linear(x_q, w_q, layer.linear.bias)


def _apply_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "none":
        return x
    if activation == "relu":
        return F.relu(x)
    if activation == "sigmoid":
        return torch.sigmoid(x)
    raise ValueError(f"Unsupported activation {activation!r}.")


class TinyMNISTFP32(nn.Module):
    def __init__(self, *, activation: str = "relu", output_activation: str = "none"):
        super().__init__()
        self.activation = str(activation)
        self.output_activation = str(output_activation)
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1, bias=True)
        self.fc = nn.Linear(16, 10, bias=True)

    def forward(self, x):
        x = _apply_activation(self.conv1(x), self.activation)
        x = _apply_activation(self.conv2(x), self.activation)
        x = _apply_activation(self.conv3(x), self.activation)
        x = x.mean(dim=[2, 3])
        x = self.fc(x)
        x = _apply_activation(x, self.output_activation)
        return x


class TinyMNISTQAT(nn.Module):
    def __init__(
        self,
        layer_configs=None,
        *,
        activation: str = "relu",
        output_activation: str = "none",
        use_di_sigmoid_approx: bool = False,
    ):
        super().__init__()
        self.activation = str(activation)
        self.output_activation = str(output_activation)
        self.use_di_sigmoid_approx = bool(use_di_sigmoid_approx)
        self.fc_input_scale_override: float | None = None
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
        x = _apply_activation(self.conv1(x), self.activation)
        x = _apply_activation(self.conv2(x), self.activation)
        x = _apply_activation(self.conv3(x), self.activation)
        x = x.mean(dim=[2, 3])
        if self.fc_input_scale_override is None:
            x = self.fc(x)
        else:
            x = _apply_qlinear_with_input_scale(self.fc, x, self.fc_input_scale_override)
        if self.output_activation == "sigmoid" and self.use_di_sigmoid_approx:
            x = _apply_di_sigmoid_qat(x, layer=self.fc, output_scale=SIGMOID_OUTPUT_SCALE)
        else:
            x = _apply_activation(x, self.output_activation)
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


def classification_loss(output: torch.Tensor, target: torch.Tensor, *, output_activation: str = "none") -> torch.Tensor:
    if output_activation == "sigmoid":
        one_hot = F.one_hot(target, num_classes=output.shape[1]).to(output.dtype)
        return F.binary_cross_entropy(output, one_hot)
    return F.cross_entropy(output, target)


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


def collect_activation3_boundary_scale(
    model: TinyMNISTQAT,
    calib_dataset,
    *,
    activation: str = "relu",
    percentile: float = 99.9,
    max_samples: int = 256,
) -> float:
    def extract_activation3(module: TinyMNISTQAT, x: torch.Tensor) -> torch.Tensor:
        x1 = _apply_activation(module.conv1(x), activation)
        x2 = _apply_activation(module.conv2(x1), activation)
        x3 = _apply_activation(module.conv3(x2), activation)
        return x3

    return collect_tensor_percentile_scale(
        model,
        calib_dataset,
        extractor=extract_activation3,
        percentile=percentile,
        max_samples=max_samples,
        qmax=127.0,
        floor=1e-8,
    )


def apply_compiler_boundary_overrides(
    model: TinyMNISTQAT,
    *,
    calib_dataset,
    activation: str = "relu",
) -> TinyMNISTQAT:
    model.fc_input_scale_override = collect_activation3_boundary_scale(
        model,
        calib_dataset,
        activation=activation,
    )
    return model


def build_compiler_ready_mnist_model(
    qat_model: TinyMNISTQAT,
    *,
    calib_dataset,
    mixed_precision_configs=None,
    dequantize_output: bool = True,
    activation: str = "relu",
    output_activation: str = "none",
):
    qat_model = copy.deepcopy(qat_model)
    if hasattr(qat_model, "use_di_sigmoid_approx"):
        qat_model.use_di_sigmoid_approx = False
    if mixed_precision_configs is not None:
        qat_model = apply_layer_quant_configs(qat_model, mixed_precision_configs, inplace=False)
    output_scales = infer_chain_output_scales(qat_model, LAYER_NAMES)
    output_bits = infer_chain_output_bits(qat_model, LAYER_NAMES)
    output_scales["conv3"] = collect_activation3_boundary_scale(qat_model, calib_dataset, activation=activation)
    output_bits["conv3"] = 8
    if output_activation == "sigmoid":
        output_scales["fc"] = SIGMOID_OUTPUT_SCALE
        output_bits["fc"] = SIGMOID_P_OUT
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
    calibration_loader: DataLoader,
    validation_loader: DataLoader,
    device: str,
    candidate_bits: tuple[tuple[int, int], ...] = ((16, 16), (8, 8), (4, 4)),
    max_acceptable_drop: float = 0.01,
):
    baseline_configs = collect_layer_quant_configs(qat_model, LAYER_NAMES)
    parameter_counts = collect_layer_parameter_counts(qat_model, LAYER_NAMES)

    def evaluate_configs(configs):
        candidate_model = recalibrate_qat_scales(
            apply_layer_quant_configs(qat_model, configs, inplace=False),
            layer_names=LAYER_NAMES,
            calib_loader=calibration_loader,
            device=device,
            inplace=True,
        ).to(device).eval()
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
    activation: str = "relu",
    output_activation: str = "none",
    use_di_sigmoid_approx: bool = False,
) -> TinyMNISTQAT:
    fp32_model.eval()
    act_maxes = collect_input_activation_maxes(fp32_model, calib_loader, LAYER_NAMES, device=device)
    qat_model = TinyMNISTQAT(
        layer_configs,
        activation=activation,
        output_activation=output_activation,
        use_di_sigmoid_approx=use_di_sigmoid_approx,
    ).to(device)

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
    activation: str = "relu",
    output_activation: str = "none",
) -> TinyMNISTFP32:
    model = TinyMNISTFP32(activation=activation, output_activation=output_activation).to(device)
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
            loss = classification_loss(model(x), y, output_activation=output_activation)
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
    calib_dataset,
    epochs: int,
    lr: float,
    activation: str = "relu",
    output_activation: str = "none",
) -> TinyMNISTQAT:
    params = [param for name, param in qat_model.named_parameters() if "scale" not in name]
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    apply_compiler_boundary_overrides(qat_model, calib_dataset=calib_dataset, activation=activation)
    best_state = None
    best_acc = evaluate_model(qat_model, test_loader, device)
    print(f"qat pre-finetune acc={best_acc * 100:.2f}%")
    for epoch in range(epochs):
        qat_model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = classification_loss(qat_model(x), y, output_activation=output_activation)
            loss.backward()
            optimizer.step()
        scheduler.step()
        apply_compiler_boundary_overrides(qat_model, calib_dataset=calib_dataset, activation=activation)
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
    activation: str = "relu",
    output_activation: str = "none",
    use_di_sigmoid_approx: bool = False,
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
        activation=activation,
        output_activation=output_activation,
    )
    torch.save(fp32_model.state_dict(), run_path / "fp32.pt")
    fp32_acc = evaluate_model(fp32_model, test_loader, device)

    calib_loader = DataLoader(Subset(train_ds, range(1000)), batch_size=256, shuffle=False)
    qat_model = initialize_qat_from_fp32(
        fp32_model,
        layer_configs=layer_configs,
        calib_loader=calib_loader,
        device=device,
        activation=activation,
        output_activation=output_activation,
        use_di_sigmoid_approx=use_di_sigmoid_approx,
    )
    apply_compiler_boundary_overrides(
        qat_model,
        calib_dataset=test_ds,
        activation=activation,
    )
    ptq_acc = evaluate_model(qat_model, test_loader, device)

    qat_model = finetune_qat(
        qat_model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        calib_dataset=test_ds,
        epochs=qat_epochs,
        lr=qat_lr,
        activation=activation,
        output_activation=output_activation,
    )
    torch.save(qat_model.state_dict(), run_path / "qat.pt")
    qat_acc = evaluate_model(qat_model, test_loader, device)

    compiler_ready = build_compiler_ready_mnist_model(
        qat_model,
        calib_dataset=test_ds,
        activation=activation,
        output_activation=output_activation,
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
        "activation": activation,
        "output_activation": output_activation,
        "use_di_sigmoid_approx": use_di_sigmoid_approx,
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
    activation: str = "relu",
    output_activation: str = "none",
    use_di_sigmoid_approx: bool = False,
) -> TinyMNISTQAT:
    run_path = Path(run_dir)
    qat_path = run_path / "qat.pt"
    if not qat_path.exists():
        raise FileNotFoundError(f"Expected trained QAT checkpoint at {qat_path}.")
    layer_configs = layer_configs or build_signed_w8a8_configs()
    model = TinyMNISTQAT(
        layer_configs,
        activation=activation,
        output_activation=output_activation,
        use_di_sigmoid_approx=use_di_sigmoid_approx,
    ).to(device)
    state_dict = torch.load(qat_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.eval()


def build_compiled_artifact_from_run(
    run_dir: str,
    *,
    data_dir: str = "./data",
    sample_index: int = 0,
    dequantize_output: bool = True,
    activation: str = "relu",
    output_activation: str = "none",
):
    _, _, _, _, test_ds = get_mnist_loaders(data_dir)
    qat_model = load_qat_model_from_run(
        run_dir,
        device="cpu",
        activation=activation,
        output_activation=output_activation,
    )
    compiler_ready = build_compiler_ready_mnist_model(
        qat_model,
        calib_dataset=test_ds,
        dequantize_output=dequantize_output,
        activation=activation,
        output_activation=output_activation,
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
    activation: str = "relu",
    output_activation: str = "none",
):
    artifact, example, label = build_compiled_artifact_from_run(
        run_dir,
        data_dir=data_dir,
        sample_index=sample_index,
        dequantize_output=dequantize_output,
        activation=activation,
        output_activation=output_activation,
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


def analyze_sigmoid_alignment_from_run(
    run_dir: str,
    *,
    data_dir: str = "./data",
    sample_index: int = 0,
):
    _, _, _, _, test_ds = get_mnist_loaders(data_dir)
    sample_image, sample_label = test_ds[int(sample_index)]

    qat_model = load_qat_model_from_run(
        run_dir,
        device="cpu",
        activation="relu",
        output_activation="sigmoid",
        use_di_sigmoid_approx=True,
    )
    apply_compiler_boundary_overrides(qat_model, calib_dataset=test_ds, activation="relu")
    with torch.no_grad():
        x = sample_image.unsqueeze(0)
        x1 = _apply_activation(qat_model.conv1(x), "relu")
        x2 = _apply_activation(qat_model.conv2(x1), "relu")
        x3 = _apply_activation(qat_model.conv3(x2), "relu")
        mean = x3.mean(dim=[2, 3])
        fc_pre_raw = qat_model.fc(mean)
        if qat_model.fc_input_scale_override is None:
            fc_pre = fc_pre_raw
        else:
            fc_pre = _apply_qlinear_with_input_scale(qat_model.fc, mean, qat_model.fc_input_scale_override)
        qat_sigmoid = _apply_di_sigmoid_qat(fc_pre, layer=qat_model.fc, output_scale=SIGMOID_OUTPUT_SCALE)

    artifact, example, _ = build_compiled_artifact_from_run(
        run_dir,
        data_dir=data_dir,
        sample_index=sample_index,
        activation="relu",
        output_activation="sigmoid",
    )
    result = run_host_emulation(
        artifact,
        {artifact.plan.inputs[0]: example},
        VerificationMode.DEBUG,
        debug=True,
    )

    final_segment = next(
        step
        for step in artifact.plan.steps
        if isinstance(step, NpuSegment) and any(op.activation == "sigmoid" for op in step.ops)
    )
    final_op = final_segment.ops[0]
    trace = result.trace_tensors
    lhs = trace[final_op.lhs].astype(np.int64)
    rhs = trace[final_op.rhs].astype(np.int64)
    bias = trace[final_op.bias].reshape(-1).astype(np.int64)
    acc = np.matmul(lhs, rhs) + bias
    pre_sigmoid_int = acc * np.int64(final_op.multiplier & 0xFFFF)
    if final_op.shift > 0:
        pre_sigmoid_int = (pre_sigmoid_int + (np.int64(1) << (final_op.shift - 1))) >> final_op.shift

    qat_fc_pre_raw = fc_pre_raw.detach().cpu().numpy().reshape(-1)
    qat_fc_pre = fc_pre.detach().cpu().numpy().reshape(-1)
    qat_pre_int = np.rint(qat_fc_pre / np.float32(SIGMOID_OUTPUT_SCALE)).astype(np.int32)
    qat_pre_int = np.clip(qat_pre_int, -32768, 32767)

    return {
        "label": int(sample_label),
        "compiled_prediction": int(np.array(result.tensors[artifact.plan.outputs[0]]).reshape(-1).argmax()),
        "qat_mean_preview": mean.detach().cpu().numpy().reshape(-1)[:8].tolist(),
        "compiled_mean_preview": trace["mean"].reshape(-1)[:8].tolist(),
        "qat_fc_pre_raw": qat_fc_pre_raw.tolist(),
        "qat_fc_pre": qat_fc_pre.tolist(),
        "qat_pre_sigmoid_int": qat_pre_int.reshape(-1).tolist(),
        "compiled_quant_input": trace[final_op.lhs].reshape(-1).tolist(),
        "compiled_pre_sigmoid_int": pre_sigmoid_int.reshape(-1).tolist(),
        "qat_sigmoid": qat_sigmoid.detach().cpu().numpy().reshape(-1).tolist(),
        "compiled_sigmoid_int": trace[final_op.out].reshape(-1).tolist(),
        "compiled_output": np.array(result.tensors[artifact.plan.outputs[0]]).reshape(-1).tolist(),
        "multiplier": int(final_op.multiplier),
        "shift": int(final_op.shift),
    }


if __name__ == "__main__":
    summary, _ = run_pipeline(run_dir="runs/mnist_tinynpu_pipeline")
    print(json.dumps(summary, indent=2))
