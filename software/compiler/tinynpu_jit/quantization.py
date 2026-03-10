from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RescaleParams:
    multiplier: int
    shift: int
    effective_scale: float
    approximated_scale: float
    absolute_error: float
    relative_error: float


def synthesize_rescale(
    effective_scale: float,
    *,
    max_multiplier: int = 0xFFFF,
    max_shift: int = 0xFF,
) -> RescaleParams:
    if effective_scale < 0:
        raise ValueError(f"effective_scale must be non-negative, got {effective_scale}.")
    if effective_scale == 0.0:
        return RescaleParams(
            multiplier=0,
            shift=0,
            effective_scale=0.0,
            approximated_scale=0.0,
            absolute_error=0.0,
            relative_error=0.0,
        )

    best: RescaleParams | None = None
    for shift in range(max_shift + 1):
        scaled = effective_scale * float(1 << shift)
        multiplier = int(round(scaled))
        if multiplier <= 0 or multiplier > max_multiplier:
            continue
        approx = multiplier / float(1 << shift)
        abs_err = abs(approx - effective_scale)
        rel_err = abs_err / effective_scale
        candidate = RescaleParams(
            multiplier=multiplier,
            shift=shift,
            effective_scale=effective_scale,
            approximated_scale=approx,
            absolute_error=abs_err,
            relative_error=rel_err,
        )
        if best is None:
            best = candidate
            continue
        if candidate.relative_error < best.relative_error - 1e-15:
            best = candidate
            continue
        if math.isclose(candidate.relative_error, best.relative_error, rel_tol=0.0, abs_tol=1e-15):
            if candidate.shift < best.shift:
                best = candidate

    if best is None:
        raise ValueError(
            f"Could not synthesize TinyNPU multiplier/shift for effective_scale={effective_scale} "
            f"within multiplier<={max_multiplier} and shift<={max_shift}."
        )
    return best
