#!/usr/bin/env python3
"""Build a kernel cycle table from Verilator runs.

Path 1 of the analytical-vs-ISS-vs-table tradeoff: instead of modeling primitive
costs, run each kernel on Verilator once and store its measured cycles. Block
predictions are then `sum(table_lookups)`, exact within Verilator.

Usage:
    # Run a hex file on Verilator and harvest per-step cycle counts.
    python3 scripts/kernel_cycle_table.py harvest \
        --hex external/cv32e40p/example_tb/core/custom/<name>.hex \
        --label qllama_A_prefill \
        --tag d_model=8,d_head=8,n_heads=1,n_kv_heads=1,ffn=8,T=8 \
        --output runs/kernel_cycles.json

    # Predict a block from the table given a step list.
    python3 scripts/kernel_cycle_table.py predict \
        --table runs/kernel_cycles.json \
        --plan tinyllama-22

The harvested log lines look like
    segment.<name>.stage cycles=1234
    segment.<name>.run cycles=5678
    segment.<name>.read cycles=910
    hostop.<name> cycles=4567
    preload.ub_image cycles=8000
which the parser collects into a per-step dictionary.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VLT_BIN = REPO_ROOT / "external" / "cv32e40p" / "example_tb" / "core" / "obj_dir" / "cv32e40p_tb_vlt_npu"

CYCLE_RE = re.compile(r"^([A-Za-z0-9_.]+)\s+cycles=(\d+)\s*$")


@dataclass
class HarvestedRun:
    label: str
    hex_path: str
    tags: dict[str, str] = field(default_factory=dict)
    steps: dict[str, int] = field(default_factory=dict)
    total_cycles: int | None = None


def parse_cycle_log(text: str) -> dict[str, int]:
    """Pull `<label> cycles=<N>` lines into a dict, last value wins."""
    found: dict[str, int] = {}
    for line in text.splitlines():
        m = CYCLE_RE.match(line.strip())
        if m:
            label, cycles = m.group(1), int(m.group(2))
            found[label] = cycles
    return found


def run_verilator(hex_path: Path, *, max_ticks: int = 3_000_000_000, max_cycles: int = 2_000_000, timeout_s: int = 600) -> str:
    if not VLT_BIN.exists():
        sys.exit(f"verilator binary not found: {VLT_BIN}")
    if not hex_path.exists():
        sys.exit(f"hex not found: {hex_path}")
    env = dict(os.environ)
    env["VERILATOR_MAX_TICKS"] = str(max_ticks)
    cmd = [str(VLT_BIN), f"+firmware={hex_path}", f"+maxcycles={max_cycles}"]
    return subprocess.run(
        cmd,
        cwd=VLT_BIN.parent.parent,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    ).stdout


def harvest(args: argparse.Namespace) -> None:
    hex_path = Path(args.hex) if args.hex else None
    if args.from_log:
        log_path = Path(args.from_log)
        print(f"reading existing log: {log_path}", flush=True)
        log = log_path.read_text()
    else:
        if hex_path is None:
            sys.exit("either --hex or --from-log is required")
        print(f"running verilator on {hex_path.name} ...", flush=True)
        log = run_verilator(hex_path, max_ticks=args.max_ticks, max_cycles=args.max_cycles, timeout_s=args.timeout_s)
        if args.save_log:
            Path(args.save_log).write_text(log)
            print(f"raw log saved to {args.save_log}", flush=True)
    steps = parse_cycle_log(log)
    if not steps:
        print("WARNING: no cycle lines found in stdout — was the program built with TNPU_RUNTIME_V2_VERBOSE_STEPS=1?", flush=True)
    tags = dict(token.split("=", 1) for token in args.tag.split(",")) if args.tag else {}
    # `segment.<n>.npu` already includes stage+run+readback+overhead, so don't double-count.
    excluded_prefixes = ("startup.", "segment.")
    total_excl_segments = sum(
        v for k, v in steps.items()
        if not any(k.startswith(p) for p in excluded_prefixes)
    )
    seg_npu_total = sum(v for k, v in steps.items() if k.startswith("segment.") and k.endswith(".npu"))
    total = total_excl_segments + seg_npu_total
    run = HarvestedRun(label=args.label, hex_path=str(hex_path) if hex_path else "", tags=tags, steps=steps, total_cycles=total)

    output = Path(args.output)
    if output.exists():
        existing = json.loads(output.read_text())
    else:
        existing = {"runs": []}
    existing["runs"].append(asdict(run))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(existing, indent=2))
    print(f"recorded {len(steps)} step(s) under label='{args.label}' in {output}", flush=True)
    for k, v in sorted(steps.items()):
        print(f"  {k:<40} {v:>10}")


def predict(args: argparse.Namespace) -> None:
    table_path = Path(args.table)
    table = json.loads(table_path.read_text())
    runs = table.get("runs", [])
    if not runs:
        sys.exit("table is empty; run `harvest` first")

    print(f"\nKernel cycle table summary ({len(runs)} run(s)):")
    print(f"{'label':<32} {'#steps':>7} {'total':>12}  tags")
    for r in runs:
        print(f"  {r['label']:<30} {len(r['steps']):>7} {r.get('total_cycles', 0):>12}  {r['tags']}")

    if args.plan == "match-by-tag":
        # Aggregate across all runs grouping by step kind for inspection.
        kinds: dict[str, list[int]] = {}
        for r in runs:
            for step, cyc in r["steps"].items():
                kinds.setdefault(_kind_of(step), []).append(cyc)
        print(f"\nPer-kind cycle ranges across runs:")
        for kind, vals in sorted(kinds.items()):
            print(f"  {kind:<28} n={len(vals):3}  min={min(vals):>10}  med={sorted(vals)[len(vals)//2]:>10}  max={max(vals):>10}")
    elif args.plan == "echo":
        for r in runs:
            print(f"\n=== {r['label']} (tags: {r['tags']}) ===")
            for k, v in sorted(r["steps"].items()):
                print(f"  {k:<40} {v:>10}")
    elif args.plan == "summary":
        # For each run, group cycles by category and report the breakdown.
        for r in runs:
            print(f"\n=== {r['label']} (tags: {r['tags']}) ===")
            host = sum(v for k, v in r["steps"].items() if k.startswith("hostop."))
            preload = sum(v for k, v in r["steps"].items() if k.startswith("preload."))
            seg_npu = sum(v for k, v in r["steps"].items() if k.startswith("segment.") and k.endswith(".npu"))
            seg_stage = sum(v for k, v in r["steps"].items() if k.startswith("segment.") and k.endswith(".stage"))
            seg_run = sum(v for k, v in r["steps"].items() if k.startswith("segment.") and k.endswith(".run"))
            seg_readback = sum(v for k, v in r["steps"].items() if k.startswith("segment.") and k.endswith(".readback"))
            seg_overhead = seg_npu - seg_stage - seg_run - seg_readback
            n_segments = sum(1 for k in r["steps"] if k.startswith("segment.") and k.endswith(".npu"))
            print(f"  segments         = {n_segments}")
            print(f"  total host       = {host}")
            print(f"  total preload    = {preload}")
            print(f"  segment.stage    = {seg_stage}    avg {seg_stage // max(1,n_segments)}/segment")
            print(f"  segment.run (NPU body) = {seg_run}")
            print(f"  segment.readback = {seg_readback}    avg {seg_readback // max(1,n_segments)}/segment")
            print(f"  segment.overhead (residual) = {seg_overhead}    avg {seg_overhead // max(1,n_segments)}/segment")
            print(f"  total cold       = {host + preload + seg_npu}")
    else:
        print(f"\n(plan='{args.plan}' not yet implemented; use 'echo', 'summary', or 'match-by-tag')")


def _kind_of(step_name: str) -> str:
    """Map a step label like `segment.seg_qkv.run` to a coarse kind `segment.run`."""
    parts = step_name.split(".")
    if parts[0] == "segment" and len(parts) >= 3:
        return f"segment.{parts[-1]}"
    if parts[0] == "preload":
        return "preload"
    if parts[0] == "hostop":
        return "hostop"
    if parts[0] == "startup":
        return "startup"
    return parts[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Verilator-backed kernel cycle table.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest", help="run a hex on Verilator and record per-step cycles")
    h.add_argument("--hex", default=None, help="path to the .hex firmware (omit if --from-log is given)")
    h.add_argument("--from-log", default=None, help="parse cycles from an existing verilator stdout log instead of running verilator")
    h.add_argument("--label", required=True, help="short label to identify this run in the table")
    h.add_argument("--tag", default="", help="comma-separated key=value tags for indexing (e.g. d_model=8,T=8)")
    h.add_argument("--output", default=str(REPO_ROOT / "runs" / "kernel_cycles.json"))
    h.add_argument("--max-ticks", type=int, default=3_000_000_000)
    h.add_argument("--max-cycles", type=int, default=2_000_000)
    h.add_argument("--timeout-s", type=int, default=600)
    h.add_argument("--save-log", default=None, help="optional path to save raw stdout")
    h.set_defaults(func=harvest)

    p = sub.add_parser("predict", help="inspect/use the table")
    p.add_argument("--table", default=str(REPO_ROOT / "runs" / "kernel_cycles.json"))
    p.add_argument("--plan", default="match-by-tag", choices=("echo", "summary", "match-by-tag", "tinyllama-22"))
    p.set_defaults(func=predict)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
