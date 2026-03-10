from __future__ import annotations

from collections.abc import Callable, Mapping

from .config import LayerQuantConfig, build_layer_config_map


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
