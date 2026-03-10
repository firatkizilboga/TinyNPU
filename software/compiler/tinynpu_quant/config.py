from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class LayerQuantConfig:
    w_bits: int = 8
    a_bits: int = 8

    def weight_qmax(self) -> int:
        return (1 << (self.w_bits - 1)) - 1

    def weight_qmin(self) -> int:
        return -(1 << (self.w_bits - 1)) + 1

    def activation_qmax(self, *, signed: bool = False) -> int:
        if signed:
            return (1 << (self.a_bits - 1)) - 1
        return (1 << self.a_bits) - 1

    def activation_qmin(self, *, signed: bool = False) -> int:
        if signed:
            return -(1 << (self.a_bits - 1)) + 1
        return 0


def ensure_layer_quant_config(value: LayerQuantConfig | Mapping[str, int] | None) -> LayerQuantConfig:
    if value is None:
        return LayerQuantConfig()
    if isinstance(value, LayerQuantConfig):
        return value
    return LayerQuantConfig(w_bits=int(value["w_bits"]), a_bits=int(value["a_bits"]))


def build_layer_config_map(
    layer_names: list[str] | tuple[str, ...],
    overrides: Mapping[str, LayerQuantConfig | Mapping[str, int]] | None = None,
    *,
    default_w_bits: int = 8,
    default_a_bits: int = 8,
) -> dict[str, LayerQuantConfig]:
    overrides = overrides or {}
    configs: dict[str, LayerQuantConfig] = {}
    for name in layer_names:
        if name in overrides:
            configs[name] = ensure_layer_quant_config(overrides[name])
        else:
            configs[name] = LayerQuantConfig(w_bits=default_w_bits, a_bits=default_a_bits)
    return configs
