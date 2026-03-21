from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent

GROUP_COLORS = {
    "host": "#D9A441",
    "sequencing": "#6D6E71",
    "matmul": "#4A7EBB",
}


def main() -> None:
    if len(sys.argv) == 4:
        data_path = Path(sys.argv[1])
        svg_path = Path(sys.argv[2])
        pdf_path = Path(sys.argv[3])
    else:
        data_path = ROOT / "control_state_breakdown_simple_chain.json"
        svg_path = ROOT / "control_state_breakdown.svg"
        pdf_path = ROOT / "control_state_breakdown.pdf"

    payload = json.loads(data_path.read_text())
    states = payload["states"]

    labels = [item["name"].replace("CTRL_", "") for item in states]
    cycles = [item["cycles"] for item in states]
    colors = [GROUP_COLORS[item["group"]] for item in states]
    total_cycles = int(payload["total_cycles"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
        }
    )

    fig, ax_cycles = plt.subplots(
        1,
        1,
        figsize=(8.8, 5.2),
    )

    y_positions = list(range(len(labels)))
    ax_cycles.barh(y_positions, cycles, color=colors, edgecolor="#2A2A2A", linewidth=0.8)
    ax_cycles.set_yticks(y_positions, labels)
    ax_cycles.invert_yaxis()
    ax_cycles.set_xlabel("Cycles")
    ax_cycles.set_title(
        f"{payload['title']} ({payload['workload']}, total run window = {total_cycles} cycles)",
        pad=26,
    )
    ax_cycles.grid(axis="x", color="#DADADA", linewidth=0.6)
    ax_cycles.set_axisbelow(True)

    for y_pos, cycle_count in zip(y_positions, cycles):
        ax_cycles.text(cycle_count + 2, y_pos, str(cycle_count), va="center", ha="left", fontsize=9)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS["host"], ec="#2A2A2A"),
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS["sequencing"], ec="#2A2A2A"),
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS["matmul"], ec="#2A2A2A"),
    ]
    ax_cycles.legend(
        legend_handles,
        ["Host service", "Fetch / decode", "Matmul micro-sequence"],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=3,
        frameon=False,
    )

    fig.subplots_adjust(left=0.29, right=0.98, top=0.82, bottom=0.16)

    fig.text(
        0.012,
        0.008,
        "Zero-cycle states omitted: " + ", ".join(payload["omitted_zero_states"]),
        fontsize=8.5,
        color="#444444",
    )

    fig.savefig(svg_path, format="svg")
    fig.savefig(pdf_path, format="pdf")


if __name__ == "__main__":
    main()
