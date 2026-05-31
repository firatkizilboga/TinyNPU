from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "report" / "figures"

P_OUT = 16
QMAX = (1 << (P_OUT - 1)) - 1

# From runs/OUTPUT_SIGMOID_CALIBRATION_2026_05_31.md.
OUTPUT_LOGIT_SCALE = 0.000499748102
OPERATING_X = -3667


def _clip_signed(value: np.ndarray | int, bits: int) -> np.ndarray | int:
    max_value = (1 << (bits - 1)) - 1
    min_value = -(1 << (bits - 1))
    return np.clip(value, min_value, max_value)


def _rtl_round_shift_positive(value: np.ndarray, shift: int) -> np.ndarray:
    if shift <= 0:
        return value.astype(np.int64)
    return ((value.astype(np.int64) + (1 << (shift - 1))) >> shift).astype(np.int64)


def _round_shift_signed(value: np.ndarray, shift: int) -> np.ndarray:
    value = value.astype(np.int64)
    if shift <= 0:
        return value
    rounder = 1 << (shift - 1)
    pos = value >= 0
    out = np.empty_like(value, dtype=np.int64)
    out[pos] = (value[pos] + rounder) >> shift
    out[~pos] = -(((-value[~pos]) + rounder) >> shift)
    return out


def ppu_hard_sigmoid(x: np.ndarray, *, shift: int) -> np.ndarray:
    bound = 8 << int(shift)
    numer = np.zeros_like(x, dtype=np.int64)
    below = x <= -bound
    above = x >= bound
    middle = ~(below | above)
    numer[above] = QMAX << (shift + 4)
    numer[middle] = (x[middle].astype(np.int64) + bound) * QMAX
    numer += 1 << (shift + 3)
    y_int = numer >> (shift + 4)
    return np.clip(y_int.astype(np.float64) / QMAX, 0.0, 1.0)


def ideal_sigmoid_from_integer(x: np.ndarray) -> np.ndarray:
    z = x.astype(np.float64) * OUTPUT_LOGIT_SCALE
    return 1.0 / (1.0 + np.exp(-z))


def ppu_relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0)


def ppu_h_gelu(x: np.ndarray, *, x_scale_shift: int = 7) -> np.ndarray:
    effective_shift = min(int(x_scale_shift), 15)
    x_ext = _clip_signed(x.astype(np.int64), 16).astype(np.int64)
    scale = 1 << effective_shift
    gate = _rtl_round_shift_positive(x_ext * 218, 7) + 3 * scale
    gate = np.clip(gate, 0, 6 * scale)
    div6 = _rtl_round_shift_positive(x_ext * gate * 10923, 16)
    return _clip_signed(_round_shift_signed(div6, effective_shift), 16).astype(np.int64)


def ideal_gelu(x: np.ndarray) -> np.ndarray:
    erf = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + erf(x / math.sqrt(2.0)))


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    x = np.arange(-12000, 12001, 8, dtype=np.int64)
    y_shift10 = ppu_hard_sigmoid(x, shift=10)
    y_ideal = ideal_sigmoid_from_integer(x)

    op_x = np.array([OPERATING_X], dtype=np.int64)
    op_ideal = float(ideal_sigmoid_from_integer(op_x)[0])
    op_shift10 = float(ppu_hard_sigmoid(op_x, shift=10)[0])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 160,
        }
    )

    fig, ax = plt.subplots(figsize=(5.9, 3.75), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf7")

    ax.set_facecolor("#fffdf8")
    ax.plot(x, y_ideal, color="#1f2937", linewidth=2.0, label="Ideal sigmoid after logit scale")
    ax.plot(x, y_shift10, color="#0f766e", linewidth=2.4, label="TinyNPU clipped sigmoid, s=10")
    ax.scatter([OPERATING_X], [op_shift10], s=42, color="#0f766e", zorder=5)
    ax.scatter([OPERATING_X], [op_ideal], s=42, color="#1f2937", marker="x", zorder=5)
    ax.axvline(0, color="#9ca3af", linewidth=1.0)
    ax.axhline(0.5, color="#9ca3af", linewidth=1.0, linestyle=":")
    ax.annotate(
        f"example x={OPERATING_X}\nideal={op_ideal:.3f}, hardware={op_shift10:.3f}",
        xy=(OPERATING_X, op_shift10),
        xytext=(-10800, 0.68),
        arrowprops={"arrowstyle": "->", "color": "#374151", "lw": 1.0},
        bbox={"boxstyle": "round,pad=0.35", "fc": "#fff7ed", "ec": "#fed7aa"},
    )
    ax.set_title("Calibrated fused sigmoid response")
    ax.set_xlabel("PPU post-rescale integer input")
    ax.set_ylabel("Probability-like output")
    ax.set_xlim(-12000, 12000)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="lower right", frameon=True, framealpha=0.95)

    stem = FIG_DIR / "iszero_sigmoid_calibrated_comparison"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)

    csv_path = stem.with_suffix(".csv")
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x_int", "ideal_sigmoid", "hard_sigmoid_s10"])
        for row in zip(x, y_ideal, y_shift10, strict=True):
            writer.writerow([int(row[0]), f"{row[1]:.9f}", f"{row[2]:.9f}"])

    relu_x = np.arange(-32768, 32768, 64, dtype=np.int64)
    relu_y = ppu_relu(relu_x)
    fig, ax = plt.subplots(figsize=(5.9, 3.4), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf7")
    ax.set_facecolor("#fffdf8")
    ax.plot(relu_x, relu_y, color="#2563eb", linewidth=2.4, label="TinyNPU ReLU")
    ax.axvline(0, color="#9ca3af", linewidth=1.0)
    ax.axhline(0, color="#9ca3af", linewidth=1.0)
    ax.set_title("ReLU response")
    ax.set_xlabel("PPU post-rescale integer input")
    ax.set_ylabel("Integer output before final saturation")
    ax.set_xlim(-32768, 32767)
    ax.set_ylim(-1500, 32767)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    relu_stem = FIG_DIR / "ppu_relu_response"
    fig.savefig(relu_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(relu_stem.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)

    gelu_shift = 7
    gelu_scale = 1 << gelu_shift
    gelu_x_int = np.arange(-4 * gelu_scale, 4 * gelu_scale + 1, 1, dtype=np.int64)
    gelu_x_real = gelu_x_int.astype(np.float64) / gelu_scale
    gelu_y_hw = ppu_h_gelu(gelu_x_int, x_scale_shift=gelu_shift).astype(np.float64) / gelu_scale
    gelu_y_ideal = ideal_gelu(gelu_x_real)
    fig, ax = plt.subplots(figsize=(5.9, 3.4), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf7")
    ax.set_facecolor("#fffdf8")
    ax.plot(gelu_x_real, gelu_y_ideal, color="#1f2937", linewidth=2.0, label="Standard GELU")
    ax.step(
        gelu_x_real,
        gelu_y_hw,
        where="mid",
        color="#7c3aed",
        linewidth=1.7,
        label="TinyNPU RTL integer hard-GELU",
    )
    marker_stride = 32
    ax.scatter(
        gelu_x_real[::marker_stride],
        gelu_y_hw[::marker_stride],
        s=14,
        color="#7c3aed",
        alpha=0.85,
        zorder=4,
    )
    ax.axvline(0, color="#9ca3af", linewidth=1.0)
    ax.axhline(0, color="#9ca3af", linewidth=1.0)
    ax.set_title("Hard-GELU integer response")
    ax.set_xlabel("Input in activation scale domain")
    ax.set_ylabel("Output in activation scale domain")
    ax.set_xlim(-4.0, 4.0)
    ax.set_ylim(-0.4, 4.1)
    ax.grid(True, color="#e5e7eb", linewidth=0.8)
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    gelu_stem = FIG_DIR / "ppu_h_gelu_response"
    fig.savefig(gelu_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(gelu_stem.with_suffix(".png"), bbox_inches="tight", dpi=220)
    plt.close(fig)

    print(stem.with_suffix(".pdf"))
    print(stem.with_suffix(".png"))
    print(csv_path)
    print(relu_stem.with_suffix(".pdf"))
    print(relu_stem.with_suffix(".png"))
    print(gelu_stem.with_suffix(".pdf"))
    print(gelu_stem.with_suffix(".png"))


if __name__ == "__main__":
    main()
