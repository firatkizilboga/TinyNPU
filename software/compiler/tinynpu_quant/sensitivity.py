from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from math import ceil

import torch.nn as nn

from .calibration import collect_input_activation_maxes
from .config import LayerQuantConfig, build_layer_config_map
from .conversion import (
    collect_qat_layer_names,
    convert_qat_model_for_compiler,
    infer_chain_output_bits,
)
from .qat_modules import QConv2d, QLinear


def single_layer_bit_drop_sensitivity(
    layer_names: list[str] | tuple[str, ...],
    *,
    evaluate_configs: Callable[[dict[str, LayerQuantConfig]], float],
    baseline_bits: tuple[int, int] = (8, 8),
    trial_bits: tuple[int, int] = (4, 4),
) -> tuple[float, dict[str, float]]:
    baseline_configs = build_layer_config_map(
        layer_names,
        default_w_bits=int(baseline_bits[0]),
        default_a_bits=int(baseline_bits[1]),
    )
    baseline_score = evaluate_configs(baseline_configs)

    drops: dict[str, float] = {}
    for layer_name in layer_names:
        configs = build_layer_config_map(
            layer_names,
            default_w_bits=int(baseline_bits[0]),
            default_a_bits=int(baseline_bits[1]),
            overrides={layer_name: LayerQuantConfig(w_bits=int(trial_bits[0]), a_bits=int(trial_bits[1]))},
        )
        score = evaluate_configs(configs)
        drops[layer_name] = baseline_score - score
    return baseline_score, drops


def rank_sensitivity(importances: Mapping[str, float]) -> list[tuple[str, float]]:
    return sorted(importances.items(), key=lambda item: item[1], reverse=True)


def collect_layer_quant_configs(
    model: nn.Module,
    layer_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, LayerQuantConfig]:
    layer_names = list(layer_names or collect_qat_layer_names(model))
    modules = dict(model.named_modules())
    configs: dict[str, LayerQuantConfig] = {}
    for layer_name in layer_names:
        layer = modules.get(layer_name)
        if not isinstance(layer, (QConv2d, QLinear)):
            raise TypeError(f"Layer {layer_name!r} is not a supported QAT layer.")
        configs[layer_name] = LayerQuantConfig(
            w_bits=int(layer.w_bits),
            a_bits=int(layer.a_bits),
            signed_activations=bool(layer.signed_activations),
        )
    return configs


def collect_layer_parameter_counts(
    model: nn.Module,
    layer_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, int]]:
    layer_names = list(layer_names or collect_qat_layer_names(model))
    modules = dict(model.named_modules())
    counts: dict[str, dict[str, int]] = {}
    for layer_name in layer_names:
        layer = modules.get(layer_name)
        if isinstance(layer, QConv2d):
            weight = layer.conv.weight
            bias = layer.conv.bias
        elif isinstance(layer, QLinear):
            weight = layer.linear.weight
            bias = layer.linear.bias
        else:
            raise TypeError(f"Layer {layer_name!r} is not a supported QAT layer.")
        counts[layer_name] = {
            "weight_elements": int(weight.numel()),
            "bias_elements": int(0 if bias is None else bias.numel()),
        }
    return counts


def apply_layer_quant_configs(
    model: nn.Module,
    layer_configs: Mapping[str, LayerQuantConfig | Mapping[str, int]],
    *,
    inplace: bool = False,
) -> nn.Module:
    target = model if inplace else copy.deepcopy(model)
    modules = dict(target.named_modules())
    for layer_name, raw_config in layer_configs.items():
        layer = modules.get(layer_name)
        if not isinstance(layer, (QConv2d, QLinear)):
            raise TypeError(f"Layer {layer_name!r} is not a supported QAT layer.")
        config = LayerQuantConfig(
            w_bits=int(raw_config.w_bits) if isinstance(raw_config, LayerQuantConfig) else int(raw_config["w_bits"]),
            a_bits=int(raw_config.a_bits) if isinstance(raw_config, LayerQuantConfig) else int(raw_config["a_bits"]),
            signed_activations=(
                bool(raw_config.signed_activations)
                if isinstance(raw_config, LayerQuantConfig)
                else bool(raw_config.get("signed_activations", layer.signed_activations))
            ),
        )
        layer.w_bits = int(config.w_bits)
        layer.a_bits = int(config.a_bits)
        layer.signed_activations = bool(config.signed_activations)
    return target


def recalibrate_qat_scales(
    model: nn.Module,
    *,
    layer_names: list[str] | tuple[str, ...] | None,
    calib_loader,
    device: str,
    inplace: bool = False,
) -> nn.Module:
    target = model if inplace else copy.deepcopy(model)
    resolved_layer_names = list(layer_names or collect_qat_layer_names(target))
    modules = dict(target.named_modules())
    activation_maxes = collect_input_activation_maxes(target.eval(), calib_loader, resolved_layer_names, device=device)

    for layer_name in resolved_layer_names:
        layer = modules.get(layer_name)
        if not isinstance(layer, (QConv2d, QLinear)):
            raise TypeError(f"Layer {layer_name!r} is not a supported QAT layer.")
        weight_qmax = (1 << (int(layer.w_bits) - 1)) - 1
        if weight_qmax <= 0:
            raise ValueError(f"Layer {layer_name!r} has unsupported w_bits={layer.w_bits}.")
        activation_qmax = (
            (1 << (int(layer.a_bits) - 1)) - 1 if bool(layer.signed_activations) else (1 << int(layer.a_bits)) - 1
        )
        if activation_qmax <= 0:
            raise ValueError(f"Layer {layer_name!r} has unsupported a_bits={layer.a_bits}.")

        weight = layer.conv.weight if isinstance(layer, QConv2d) else layer.linear.weight
        weight_scale = max(float(weight.detach().abs().max().item()) / float(weight_qmax), 1e-8)
        activation_scale = max(float(activation_maxes[layer_name]) / float(activation_qmax), 1e-8)

        layer.w_scale.data.copy_(layer.w_scale.data.new_tensor(weight_scale))
        layer.a_scale.data.copy_(layer.a_scale.data.new_tensor(activation_scale))

    return target


def build_mixed_precision_sensitivity_report(
    layer_names: list[str] | tuple[str, ...],
    *,
    evaluate_configs: Callable[[dict[str, LayerQuantConfig]], float],
    baseline_configs: Mapping[str, LayerQuantConfig | Mapping[str, int]] | None = None,
    candidate_bits: tuple[tuple[int, int], ...] = ((16, 16), (8, 8), (4, 4)),
    max_acceptable_drop: float = 0.01,
    parameter_counts: Mapping[str, Mapping[str, int]] | None = None,
    prepare_configs: Callable[[dict[str, LayerQuantConfig]], Any] | None = None,
) -> dict[str, object]:
    """
    Evaluate one layer at a time across candidate precisions and return a stable
    mixed-precision selection report.

    `evaluate_configs` must return a higher-is-better score for the supplied
    per-layer `LayerQuantConfig` map. The returned report contains the baseline
    score, per-layer trial results, a sensitivity ranking based on the lowest
    candidate precision, selected layer configs, and estimated static parameter
    bytes when `parameter_counts` are available.
    """
    if not layer_names:
        raise ValueError("layer_names must contain at least one layer.")
    baseline_configs = build_layer_config_map(layer_names, overrides=baseline_configs)
    normalized_candidates = _normalize_candidate_bits(candidate_bits, baseline_configs)
    baseline_score = float(evaluate_configs(dict(baseline_configs)))

    layers: list[dict[str, object]] = []
    ranking_drops: dict[str, float] = {}
    selected_layer_configs: dict[str, dict[str, int | bool]] = {}
    estimated_weight_bytes = {
        "baseline_total": 0,
        "selected_total": 0,
    }
    fine_tuning_recommended = False

    for layer_name in layer_names:
        baseline_config = baseline_configs[layer_name]
        trials: list[dict[str, object]] = []
        baseline_trial: dict[str, object] | None = None
        for w_bits, a_bits in normalized_candidates:
            configs = build_layer_config_map(layer_names, overrides=baseline_configs)
            configs[layer_name] = LayerQuantConfig(
                w_bits=int(w_bits),
                a_bits=int(a_bits),
                signed_activations=bool(baseline_config.signed_activations),
            )
            prepared_configs = prepare_configs(configs) if prepare_configs is not None else configs
            score = (
                baseline_score
                if (w_bits, a_bits) == (baseline_config.w_bits, baseline_config.a_bits) and prepare_configs is None
                else float(evaluate_configs(prepared_configs))
            )
            delta = float(baseline_score - score)
            candidate_weight_bytes = _estimate_weight_bytes(parameter_counts.get(layer_name), w_bits) if parameter_counts else None
            trial = {
                "w_bits": int(w_bits),
                "a_bits": int(a_bits),
                "score": score,
                "delta": delta,
                "accepted": bool(delta <= float(max_acceptable_drop) + 1e-12),
                "estimated_weight_bytes": candidate_weight_bytes,
            }
            if (w_bits, a_bits) == (baseline_config.w_bits, baseline_config.a_bits):
                baseline_trial = trial
            trials.append(trial)

        selected_trial = _select_preferred_trial(trials, max_acceptable_drop=max_acceptable_drop)
        if baseline_trial is None:
            raise RuntimeError(f"Baseline precision for layer {layer_name!r} was not evaluated.")
        ranking_drops[layer_name] = _lowest_precision_drop(trials)
        selected_layer_configs[layer_name] = {
            "w_bits": int(selected_trial["w_bits"]),
            "a_bits": int(selected_trial["a_bits"]),
            "signed_activations": bool(baseline_config.signed_activations),
        }
        if float(selected_trial["delta"]) > 0.0:
            fine_tuning_recommended = True

        baseline_weight_bytes = baseline_trial["estimated_weight_bytes"]
        selected_weight_bytes = selected_trial["estimated_weight_bytes"]
        if isinstance(baseline_weight_bytes, int):
            estimated_weight_bytes["baseline_total"] += baseline_weight_bytes
        if isinstance(selected_weight_bytes, int):
            estimated_weight_bytes["selected_total"] += selected_weight_bytes

        layers.append(
            {
                "layer_name": layer_name,
                "baseline": {
                    "w_bits": int(baseline_config.w_bits),
                    "a_bits": int(baseline_config.a_bits),
                    "score": baseline_score,
                    "estimated_weight_bytes": baseline_weight_bytes,
                },
                "selected": {
                    "w_bits": int(selected_trial["w_bits"]),
                    "a_bits": int(selected_trial["a_bits"]),
                    "score": float(selected_trial["score"]),
                    "delta": float(selected_trial["delta"]),
                    "estimated_weight_bytes": selected_weight_bytes,
                },
                "recommendation": _recommendation_for_trial(
                    baseline_config=baseline_config,
                    selected_trial=selected_trial,
                    max_acceptable_drop=max_acceptable_drop,
                ),
                "trials": trials,
            }
        )

    ranking = [
        {"layer_name": layer_name, "delta": float(delta)}
        for layer_name, delta in rank_sensitivity(ranking_drops)
    ]
    for rank, entry in enumerate(ranking, start=1):
        for layer_entry in layers:
            if layer_entry["layer_name"] == entry["layer_name"]:
                layer_entry["rank"] = rank
                break

    return {
        "baseline_score": baseline_score,
        "max_acceptable_drop": float(max_acceptable_drop),
        "candidate_bits": [{"w_bits": w_bits, "a_bits": a_bits} for w_bits, a_bits in normalized_candidates],
        "sensitivity_ranking": ranking,
        "selected_layer_configs": selected_layer_configs,
        "estimated_weight_bytes": estimated_weight_bytes,
        "fine_tuning_recommended": fine_tuning_recommended,
        "layers": layers,
    }


def convert_mixed_precision_qat_model_for_compiler(
    model: nn.Module,
    layer_configs_or_report: Mapping[str, object],
    *,
    layer_order: list[str] | None = None,
    input_scale: float | None = None,
    output_scales: dict[str, float] | None = None,
    output_bits: dict[str, int] | None = None,
    dequantize_output: bool = True,
) -> nn.Module:
    """
    Convert a QAT model after applying selected mixed-precision layer configs.

    `layer_configs_or_report` may be either a direct layer-config mapping or a
    full sensitivity report containing `selected_layer_configs`.
    """
    if isinstance(layer_configs_or_report, Mapping) and "selected_layer_configs" in layer_configs_or_report:
        layer_configs = _coerce_layer_configs_from_report(layer_configs_or_report)
        prepared_model = apply_layer_quant_configs(model, layer_configs, inplace=False)
    else:
        prepared_model = apply_layer_quant_configs(model, layer_configs_or_report, inplace=False)
    resolved_layer_order = layer_order or collect_qat_layer_names(prepared_model)
    resolved_output_bits = output_bits or infer_chain_output_bits(prepared_model, resolved_layer_order)
    return convert_qat_model_for_compiler(
        prepared_model,
        layer_order=resolved_layer_order,
        input_scale=input_scale,
        output_scales=output_scales,
        output_bits=resolved_output_bits,
        dequantize_output=dequantize_output,
    )


def _normalize_candidate_bits(
    candidate_bits: tuple[tuple[int, int], ...],
    baseline_configs: Mapping[str, LayerQuantConfig],
) -> list[tuple[int, int]]:
    normalized: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for bits in candidate_bits:
        pair = (int(bits[0]), int(bits[1]))
        if pair not in seen:
            normalized.append(pair)
            seen.add(pair)
    for config in baseline_configs.values():
        pair = (int(config.w_bits), int(config.a_bits))
        if pair not in seen:
            normalized.append(pair)
            seen.add(pair)
    return normalized


def _select_preferred_trial(
    trials: list[dict[str, object]],
    *,
    max_acceptable_drop: float,
) -> dict[str, object]:
    accepted = [trial for trial in trials if bool(trial["accepted"])]
    candidates = accepted or trials
    return min(
        candidates,
        key=lambda trial: (
            int(trial["estimated_weight_bytes"]) if isinstance(trial["estimated_weight_bytes"], int) else _precision_cost(int(trial["w_bits"]), int(trial["a_bits"])),
            int(trial["a_bits"]),
            int(trial["w_bits"]),
            float(trial["delta"]),
            -float(trial["score"]),
        ),
    )


def _precision_cost(w_bits: int, a_bits: int) -> int:
    return int(w_bits) + int(a_bits)


def _lowest_precision_drop(trials: list[dict[str, object]]) -> float:
    lowest_precision_trial = min(
        trials,
        key=lambda trial: (
            _precision_cost(int(trial["w_bits"]), int(trial["a_bits"])),
            int(trial["a_bits"]),
            int(trial["w_bits"]),
        ),
    )
    return float(lowest_precision_trial["delta"])


def _recommendation_for_trial(
    *,
    baseline_config: LayerQuantConfig,
    selected_trial: Mapping[str, object],
    max_acceptable_drop: float,
) -> str:
    selected_bits = (int(selected_trial["w_bits"]), int(selected_trial["a_bits"]))
    baseline_bits = (int(baseline_config.w_bits), int(baseline_config.a_bits))
    if selected_bits == baseline_bits:
        return f"keep_w{baseline_bits[0]}a{baseline_bits[1]}"
    if float(selected_trial["delta"]) <= float(max_acceptable_drop):
        return f"use_w{selected_bits[0]}a{selected_bits[1]}"
    return "qat_recommended"


def _estimate_weight_bytes(counts: Mapping[str, int] | None, w_bits: int) -> int | None:
    if counts is None:
        return None
    weight_elements = int(counts.get("weight_elements", 0))
    bias_elements = int(counts.get("bias_elements", 0))
    return int(ceil((weight_elements * int(w_bits)) / 8.0) + (bias_elements * 4))


def _coerce_layer_configs_from_report(
    layer_configs_or_report: Mapping[str, object],
) -> dict[str, LayerQuantConfig]:
    """Accept either a report dict or a raw config mapping and normalize both."""
    raw_configs = (
        layer_configs_or_report["selected_layer_configs"]
        if "selected_layer_configs" in layer_configs_or_report
        else layer_configs_or_report
    )
    if not isinstance(raw_configs, Mapping):
        raise TypeError("layer_configs_or_report must be a mapping of layer configs or a report containing selected_layer_configs.")
    return {
        str(layer_name): LayerQuantConfig(
            w_bits=int(config.w_bits) if isinstance(config, LayerQuantConfig) else int(config["w_bits"]),
            a_bits=int(config.a_bits) if isinstance(config, LayerQuantConfig) else int(config["a_bits"]),
            signed_activations=(
                bool(config.signed_activations)
                if isinstance(config, LayerQuantConfig)
                else bool(config["signed_activations"])
            ),
        )
        for layer_name, config in raw_configs.items()
    }
