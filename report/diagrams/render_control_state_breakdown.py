from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "control_state_breakdown_simple_chain.json"
SVG_PATH = ROOT / "control_state_breakdown.svg"
PDF_PATH = ROOT / "control_state_breakdown.pdf"

GROUP_COLORS = {
    "host": "#D9A441",
    "sequencing": "#6D6E71",
    "matmul": "#4A7EBB",
}


def main() -> None:
    payload = json.loads(DATA_PATH.read_text())
    states = payload["states"]

    labels = [item["name"].replace("CTRL_", "") for item in states]
    cycles = [item["cycles"] for item in states]
    entries = [item["entries"] for item in states]
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

    fig, (ax_cycles, ax_entries) = plt.subplots(
        2,
        1,
        figsize=(9.2, 6.4),
        gridspec_kw={"height_ratios": [3.0, 1.6]},
        constrained_layout=True,
    )

    y_positions = list(range(len(labels)))
    ax_cycles.barh(y_positions, cycles, color=colors, edgecolor="#2A2A2A", linewidth=0.8)
    ax_cycles.set_yticks(y_positions, labels)
    ax_cycles.invert_yaxis()
    ax_cycles.set_xlabel("Cycles")
    ax_cycles.set_title(f"{payload['title']} ({payload['workload']}, total run window = {total_cycles} cycles)")
    ax_cycles.grid(axis="x", color="#DADADA", linewidth=0.6)
    ax_cycles.set_axisbelow(True)

    for y_pos, cycle_count in zip(y_positions, cycles):
        ax_cycles.text(cycle_count + 2, y_pos, str(cycle_count), va="center", ha="left", fontsize=9)

    ax_entries.barh(y_positions, entries, color=colors, edgecolor="#2A2A2A", linewidth=0.8)
    ax_entries.set_yticks(y_positions, labels)
    ax_entries.invert_yaxis()
    ax_entries.set_xlabel("Entry count")
    ax_entries.grid(axis="x", color="#DADADA", linewidth=0.6)
    ax_entries.set_axisbelow(True)

    for y_pos, entry_count in zip(y_positions, entries):
        ax_entries.text(entry_count + 0.15, y_pos, str(entry_count), va="center", ha="left", fontsize=9)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS["host"], ec="#2A2A2A"),
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS["sequencing"], ec="#2A2A2A"),
        plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS["matmul"], ec="#2A2A2A"),
    ]
    fig.legend(
        legend_handles,
        ["Host service", "Fetch / decode", "Matmul micro-sequence"],
        loc="upper right",
        frameon=False,
    )

    fig.text(
        0.012,
        0.008,
        "Zero-cycle states omitted: " + ", ".join(payload["omitted_zero_states"]),
        fontsize=8.5,
        color="#444444",
    )

    fig.savefig(SVG_PATH, format="svg")
    fig.savefig(PDF_PATH, format="pdf")


if __name__ == "__main__":
    main()
