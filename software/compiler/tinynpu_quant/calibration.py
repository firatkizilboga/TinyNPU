from __future__ import annotations

from collections.abc import Iterable, Mapping

import torch

from .config import LayerQuantConfig, ensure_layer_quant_config


def collect_input_activation_maxes(
    model: torch.nn.Module,
    calib_loader: Iterable,
    layer_names: Iterable[str],
    *,
    device: str | torch.device,
) -> dict[str, float]:
    act_maxes = {name: 0.0 for name in layer_names}
    hooks = []

    named_modules = dict(model.named_modules())

    def make_hook(name: str):
        def hook_fn(module, inputs, output):
            x = inputs[0]
            act_maxes[name] = max(act_maxes[name], float(x.abs().max().item()))

        return hook_fn

    for name in layer_names:
        if name not in named_modules:
            raise KeyError(f"Model does not expose module {name!r} for calibration.")
        hooks.append(named_modules[name].register_forward_hook(make_hook(name)))

    try:
        with torch.no_grad():
            for batch in calib_loader:
                x = batch[0] if isinstance(batch, (tuple, list)) else batch
                model(x.to(device))
    finally:
        for hook in hooks:
            hook.remove()

    return act_maxes


def collect_tensor_percentile_scale(
    model: torch.nn.Module,
    dataset: Iterable,
    *,
    extractor,
    percentile: float = 99.9,
    max_samples: int = 256,
    qmax: float = 127.0,
    floor: float = 1e-8,
) -> float:
    values = []
    limit = min(int(max_samples), len(dataset))
    if limit <= 0:
        raise ValueError("Expected at least one calibration sample when collecting a percentile scale.")

    model = model.cpu().eval()
    with torch.no_grad():
        for index in range(limit):
            sample = dataset[index]
            x = sample[0] if isinstance(sample, (tuple, list)) else sample
            tensor = extractor(model, x.unsqueeze(0))
            values.append(tensor.reshape(-1).cpu())

    if not values:
        raise ValueError("Expected at least one tensor value when collecting a percentile scale.")

    flat = torch.cat(values).abs()
    percentile_q = torch.tensor(float(percentile) / 100.0)
    boundary_abs = float(torch.quantile(flat, percentile_q).item())
    return max(boundary_abs / float(qmax), float(floor))


def initialize_scale_tensors(
    *,
    qat_state_dict: dict[str, torch.Tensor],
    fp32_state_dict: Mapping[str, torch.Tensor],
    layer_configs: Mapping[str, LayerQuantConfig | Mapping[str, int]],
    activation_maxes: Mapping[str, float],
    fp32_weight_keys: Mapping[str, str],
) -> None:
    for name, cfg_like in layer_configs.items():
        cfg = ensure_layer_quant_config(cfg_like)
        a_qmax = cfg.activation_qmax()
        a_scale = activation_maxes[name] / a_qmax if a_qmax > 0 else 1e-8
        qat_state_dict[f"{name}.a_scale"] = torch.tensor(max(a_scale, 1e-8))

        fp32_weight_key = fp32_weight_keys[name]
        weight = fp32_state_dict[fp32_weight_key]
        w_qmax = cfg.weight_qmax()
        w_scale = max((weight.abs().max() / w_qmax).item(), 1e-8)
        qat_state_dict[f"{name}.w_scale"] = torch.tensor(w_scale)


def copy_state_with_mapping(
    *,
    dst_state_dict: dict[str, torch.Tensor],
    src_state_dict: Mapping[str, torch.Tensor],
    key_mapping: Mapping[str, str],
) -> None:
    for dst_key, src_key in key_mapping.items():
        dst_state_dict[dst_key] = src_state_dict[src_key]
