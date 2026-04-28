#!/usr/bin/env python3
"""Predict TinyLlama-22 from measured QLlama RTL points.

Strategy:
  1. From `runs/kernel_cycles.json` (the harvested table), extract per-host-op
     cyc/elem rates and per-segment overhead. These are anchored to RTL.
  2. Apply structural scaling (T, d_model, ffn, n_heads, n_kv_heads, layers) to
     produce a TinyLlama-22 estimate.

Each predicted cycle traces back to one of:
  - constant per-segment overhead (measured: ~4231 cyc/segment)
  - per-element host op cost (measured cyc/elem, by kind)
  - matmul body cycles (analytical primitive model, validated against GEMMs)
  - DMA setup (measured ~1824 cyc/segment) + per-word bandwidth
  - static preload (per-word bandwidth × weight footprint)
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TABLE = ROOT / "runs" / "kernel_cycles.json"
ARRAY_SIZE = 8

# Map host op label -> (kind, element-count formula in terms of (M, d_model, ffn, n_heads, n_kv_heads, d_head, T))
# We compute these from the QLlama plan structure observed in the runs.
# elem-count formulas only depend on shape, not on the op name suffix (per-head suffixes _h0/_h1/.. share the same per-elem rate).


def _label_kind(label: str) -> str:
    """Strip per-head/per-step suffix to get a canonical kind."""
    # hostop.dequant_q_h0 -> dequant_q
    # hostop.softmax_h0   -> softmax
    # hostop.scale_scores_h0 -> scale_scores
    # hostop.rmsnorm1     -> rmsnorm
    # hostop.residual1    -> residual
    # hostop.quant_x_norm1 -> quant_x_norm
    base = label.replace("hostop.", "")
    base = re.sub(r"_h\d+$", "", base)
    base = re.sub(r"\d+$", "", base)
    return base


# Element-count formulas keyed by canonical kind.
# `cfg` is a dict with M, d_model, ffn, n_heads, n_kv_heads, d_head, T.
def elem_count(kind: str, cfg: dict) -> int:
    M, d, f, nh, nk, dh, T = cfg["M"], cfg["d_model"], cfg["ffn"], cfg["n_heads"], cfg["n_kv_heads"], cfg["d_head"], cfg["T"]
    if kind in {"rmsnorm"}:
        return M * d
    if kind in {"quant_x_norm"}:
        return M * d
    if kind in {"quant_ffn_hidden"}:
        return M * f
    if kind in {"silu_gate", "ffn_mul"}:
        return M * f
    if kind in {"residual"}:
        return M * d
    if kind in {"dequant_q"}:
        return M * dh
    if kind in {"dequant_k"}:
        return M * dh
    if kind in {"rope_q"}:
        return M * dh
    if kind in {"rope_k"}:
        return M * dh
    if kind in {"quant_q_rope"}:
        return M * dh
    if kind in {"quant_k_rope"}:
        return M * dh
    if kind in {"k_cache_scatter_matrix"}:
        return M * dh
    if kind in {"v_cache_scatter_matrix"}:
        return M * dh
    if kind in {"dequant_scores"}:
        return M * T
    if kind in {"scale_scores"}:
        return M * T
    if kind in {"causal_mask"}:
        return M * T
    if kind in {"softmax"}:
        return M * T
    if kind in {"quant_probs"}:
        return M * T
    if kind in {"alias_attn_cat", "alias_attn_cat_a"}:
        return 1  # alias is metadata-only
    if kind in {"dequant_o"}:
        return M * d
    if kind in {"dequant_gate"}:
        return M * f
    if kind in {"dequant_up"}:
        return M * f
    if kind in {"dequant_ffn_out"}:
        return M * d
    return 0


# How many times this kind appears per layer (some are per-head).
def per_layer_count(kind: str, cfg: dict) -> int:
    nh = cfg["n_heads"]
    nk = cfg["n_kv_heads"]
    if kind in {"rmsnorm", "quant_x_norm", "residual", "dequant_o", "silu_gate", "ffn_mul",
                "dequant_gate", "dequant_up", "dequant_ffn_out", "quant_ffn_hidden",
                "alias_attn_cat", "alias_attn_cat_a"}:
        return 2 if kind == "rmsnorm" else 2 if kind == "quant_x_norm" else (
            2 if kind == "residual" else 1
        )
    if kind in {"dequant_q", "rope_q", "quant_q_rope"}:
        return nh
    if kind in {"dequant_k", "rope_k", "quant_k_rope", "k_cache_scatter_matrix", "v_cache_scatter_matrix"}:
        return nk
    if kind in {"dequant_scores", "scale_scores", "causal_mask", "softmax", "quant_probs"}:
        return nh
    return 1


@dataclass
class HostOpRate:
    kind: str
    cyc_per_elem: float
    samples: int
    src: str  # which runs contributed


def extract_host_op_rates(table: dict) -> dict[str, HostOpRate]:
    """For each host op kind, average cyc/elem across runs."""
    bucket: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for run in table["runs"]:
        cfg = _cfg_from_tags(run["tags"])
        if cfg is None:
            continue
        for label, cyc in run["steps"].items():
            if not label.startswith("hostop."):
                continue
            kind = _label_kind(label)
            elems = elem_count(kind, cfg)
            if elems <= 0:
                continue
            bucket[kind].append((cyc / elems, run["label"]))
    rates: dict[str, HostOpRate] = {}
    for kind, samples in bucket.items():
        cycs = [c for c, _ in samples]
        rates[kind] = HostOpRate(
            kind=kind,
            cyc_per_elem=sum(cycs) / len(cycs),
            samples=len(samples),
            src=",".join(sorted({src for _, src in samples})),
        )
    return rates


def _cfg_from_tags(tags: dict) -> dict | None:
    try:
        d = int(tags["d_model"])
        # Normalize "_cpu" mode suffix when extracting M (prefill behavior is
        # the same on cpu-only and hybrid for host ops; only the segment kind
        # differs).
        mode = tags["mode"]
        is_prefill = mode.startswith("prefill")
        return dict(
            d_model=d,
            d_head=int(tags["d_head"]),
            n_heads=int(tags["n_heads"]),
            n_kv_heads=int(tags["n_kv_heads"]),
            ffn=int(tags["ffn"]),
            T=int(tags["T"]),
            mode=mode,
            M=int(tags["T"]) if is_prefill else 1,
        )
    except KeyError:
        return None


@dataclass
class SegmentMeasure:
    label: str
    overhead: float  # per-segment residual after stage+run+readback
    stage: float     # per-segment avg
    run: float       # per-segment avg (NPU body)
    readback: float  # per-segment avg
    n_segments: int


def extract_segment_rates(table: dict) -> list[SegmentMeasure]:
    out = []
    for run in table["runs"]:
        steps = run["steps"]
        npu = sum(v for k, v in steps.items() if k.startswith("segment.") and k.endswith(".npu"))
        stage = sum(v for k, v in steps.items() if k.startswith("segment.") and k.endswith(".stage"))
        runc = sum(v for k, v in steps.items() if k.startswith("segment.") and k.endswith(".run"))
        read = sum(v for k, v in steps.items() if k.startswith("segment.") and k.endswith(".readback"))
        n = sum(1 for k in steps if k.startswith("segment.") and k.endswith(".npu"))
        if n == 0:
            continue
        out.append(SegmentMeasure(
            label=run["label"],
            overhead=(npu - stage - runc - read) / n,
            stage=stage / n,
            run=runc / n,
            readback=read / n,
            n_segments=n,
        ))
    return out


def matmul_body_cycles(M: int, K: int, N: int, count: int = 1) -> float:
    """Validated against the calibration GEMMs (within ~2%)."""
    mt = math.ceil(M / ARRAY_SIZE)
    kt = math.ceil(K / ARRAY_SIZE)
    nt = math.ceil(N / ARRAY_SIZE)
    output_tiles = mt * nt
    padded = output_tiles * ARRAY_SIZE * ARRAY_SIZE
    per = output_tiles * (ARRAY_SIZE + ARRAY_SIZE) + output_tiles * kt * ARRAY_SIZE + padded * 10.918
    return float(count) * per


@dataclass
class BlockPrediction:
    label: str
    cfg: dict
    n_layers: int
    # per-component cycles (n_layers blocks aggregated)
    seg_overhead_cyc: float
    seg_run_cyc: float
    seg_stage_cyc: float
    seg_readback_cyc: float
    host_cyc: float
    host_breakdown: dict[str, float]
    preload_cyc: float
    @property
    def npu_total(self): return self.seg_overhead_cyc + self.seg_run_cyc + self.seg_stage_cyc + self.seg_readback_cyc
    @property
    def hot(self): return self.npu_total + self.host_cyc
    @property
    def cold(self): return self.hot + self.preload_cyc


def predict_block(cfg: dict, host_rates: dict[str, HostOpRate], n_layers: int, *, seg_overhead: float = 4231.0) -> BlockPrediction:
    M, d, f, nh, nk, dh, T = cfg["M"], cfg["d_model"], cfg["ffn"], cfg["n_heads"], cfg["n_kv_heads"], cfg["d_head"], cfg["T"]
    out_dim = nh * dh
    kv_dim = nk * dh

    # Per-block segment.run (NPU body) — analytical from matmul shapes
    body = 0.0
    body += matmul_body_cycles(M, d, out_dim)            # q
    body += matmul_body_cycles(M, d, kv_dim)             # k
    body += matmul_body_cycles(M, d, kv_dim)             # v
    body += matmul_body_cycles(M, dh, T, count=nh)       # score qk per head
    body += matmul_body_cycles(M, T, dh, count=nh)       # value av per head
    body += matmul_body_cycles(M, out_dim, d)            # o_proj
    body += matmul_body_cycles(M, d, f)                  # gate
    body += matmul_body_cycles(M, d, f)                  # up
    body += matmul_body_cycles(M, f, d)                  # down

    # Per-block segment count (constant) and stage/readback (placeholders measured ~1824/1983)
    n_segments = 6
    seg_stage_per = 1824.0
    seg_readback_per = 1983.0

    # Host ops — apply measured cyc/elem × per-layer-count × element-count
    host_breakdown: dict[str, float] = {}
    for kind, rate in host_rates.items():
        elems = elem_count(kind, cfg)
        per_layer_n = per_layer_count(kind, cfg)
        if elems <= 0 or per_layer_n <= 0:
            continue
        host_breakdown[kind] = rate.cyc_per_elem * elems * per_layer_n
    host_total = sum(host_breakdown.values())

    # Preload: scale with weight word count and the ub_word_latency primitive
    pack = 1  # int16
    weight_words = 0
    for shape in [(d, out_dim), (d, kv_dim), (d, kv_dim), (out_dim, d), (d, f), (d, f), (f, d)]:
        K, N = shape
        kt = math.ceil((K // pack) / ARRAY_SIZE)
        nt = math.ceil(N / ARRAY_SIZE)
        weight_words += kt * nt * ARRAY_SIZE
    preload_cyc = weight_words * 16.032

    return BlockPrediction(
        label=f"d={d} {cfg['mode']} T={T}",
        cfg=cfg, n_layers=n_layers,
        seg_overhead_cyc=n_segments * seg_overhead * n_layers,
        seg_run_cyc=body * n_layers,
        seg_stage_cyc=n_segments * seg_stage_per * n_layers,
        seg_readback_cyc=n_segments * seg_readback_per * n_layers,
        host_cyc=host_total * n_layers,
        host_breakdown={k: v * n_layers for k, v in host_breakdown.items()},
        preload_cyc=preload_cyc * n_layers,
    )


def fmt(v: float) -> str:
    if abs(v) >= 1e9: return f"{v/1e9:.3f}B"
    if abs(v) >= 1e6: return f"{v/1e6:.3f}M"
    if abs(v) >= 1e3: return f"{v/1e3:.3f}K"
    return f"{v:.0f}"


def print_pred(p: BlockPrediction, *, measured_total: float | None = None) -> None:
    print(f"\n{p.label}  ({p.n_layers} layer(s)):")
    print(f"  seg.overhead   = {fmt(p.seg_overhead_cyc):>10}")
    print(f"  seg.run (body) = {fmt(p.seg_run_cyc):>10}")
    print(f"  seg.stage      = {fmt(p.seg_stage_cyc):>10}")
    print(f"  seg.readback   = {fmt(p.seg_readback_cyc):>10}")
    print(f"  preload (cold) = {fmt(p.preload_cyc):>10}")
    print(f"  host total     = {fmt(p.host_cyc):>10}")
    for k in sorted(p.host_breakdown, key=lambda x: -p.host_breakdown[x])[:8]:
        print(f"    host.{k:<20} = {fmt(p.host_breakdown[k]):>10}")
    print(f"  hot            = {fmt(p.hot):>10}")
    print(f"  cold           = {fmt(p.cold):>10}")
    if measured_total is not None:
        err = (p.cold - measured_total) / measured_total * 100
        print(f"  measured       = {fmt(measured_total):>10}    err {err:+.1f}%")


def main() -> None:
    table = json.loads(TABLE.read_text())
    host_rates = extract_host_op_rates(table)
    seg_measures = extract_segment_rates(table)

    print("Host-op cyc/elem rates (averaged across measured runs):")
    print(f"  {'kind':<28} {'cyc/elem':>10}  samples  src")
    for kind in sorted(host_rates, key=lambda k: -host_rates[k].cyc_per_elem):
        r = host_rates[kind]
        print(f"  {kind:<28} {r.cyc_per_elem:>10.1f}     {r.samples:>2}    {r.src}")

    print("\nPer-segment overhead (measured per run):")
    overheads = [m.overhead for m in seg_measures]
    print(f"  mean = {sum(overheads)/len(overheads):.0f} cyc/segment   range = {min(overheads):.0f}–{max(overheads):.0f}")

    # Round-trip predictions for the QLlama runs we measured (sanity check):
    print("\n=== Round-trip sanity (predict our own measured QLlama runs from the rates):")
    measured_totals = {}
    for run in table["runs"]:
        cfg = _cfg_from_tags(run["tags"])
        if cfg is None: continue
        steps = run["steps"]
        host = sum(v for k,v in steps.items() if k.startswith("hostop."))
        seg = sum(v for k,v in steps.items() if k.startswith("segment.") and k.endswith(".npu"))
        pre = sum(v for k,v in steps.items() if k.startswith("preload."))
        measured_totals[run["label"]] = host + seg + pre

    for run in table["runs"]:
        cfg = _cfg_from_tags(run["tags"])
        if cfg is None: continue
        pred = predict_block(cfg, host_rates, n_layers=1)
        print_pred(pred, measured_total=measured_totals[run["label"]])

    # TinyLlama-22 prediction
    print("\n\n=== TinyLlama-22 prediction (T=128 prefill, ctx=128 decode) ===")
    tl_cfg_prefill = dict(d_model=2048, d_head=64, n_heads=32, n_kv_heads=4, ffn=5632, T=128, M=128, mode="prefill")
    tl_cfg_decode  = dict(d_model=2048, d_head=64, n_heads=32, n_kv_heads=4, ffn=5632, T=128, M=1,   mode="decode")
    p_pre = predict_block(tl_cfg_prefill, host_rates, n_layers=22)
    p_dec = predict_block(tl_cfg_decode,  host_rates, n_layers=22)
    print_pred(p_pre)
    print_pred(p_dec)

    # CPU baseline. We measured cyc/MAC by running pure fp32 matmul kernels on
    # cv32e40p (M=8, N=8, K∈{64, 256, 1024}) and got a stable 8.0 cyc/MAC.
    # That's the asymptote at TinyLlama-class inner-loop sizes. The original
    # 19.43 cyc/MAC fit was on small/mid matmuls with per-call overhead.
    CPU_CYC_PER_MAC_ASYMPTOTE = 8.0
    CPU_CYC_PER_MAC_MIDSIZE   = 19.43
    print(f"\n  Note: using {CPU_CYC_PER_MAC_ASYMPTOTE} cyc/MAC (measured asymptote).")
    print(f"  At small QLlama sizes the rate is ~22-27; the asymptote applies at K>=64.")

    def macs_per_block(cfg):
        M, d, f, nh, nk, dh, T = cfg["M"], cfg["d_model"], cfg["ffn"], cfg["n_heads"], cfg["n_kv_heads"], cfg["d_head"], cfg["T"]
        out_dim = nh*dh; kv_dim = nk*dh
        return (M*d*out_dim + 2*M*d*kv_dim + nh*M*dh*T + nh*M*T*dh + M*out_dim*d + 2*M*d*f + M*f*d)
    print()
    for cfg, p, name in [(tl_cfg_prefill, p_pre, "prefill"), (tl_cfg_decode, p_dec, "decode")]:
        macs = macs_per_block(cfg) * 22
        cpu_matmul = macs * CPU_CYC_PER_MAC_ASYMPTOTE
        cpu_total = cpu_matmul + p_pre.host_cyc if name == "prefill" else cpu_matmul + p_dec.host_cyc
        speedup_hot = cpu_total / p.hot
        speedup_cold = cpu_total / p.cold
        print(f"  TinyLlama-22 {name}:  macs={fmt(macs)}, cpu={fmt(cpu_total)}, hybrid_hot={fmt(p.hot)}, hybrid_cold={fmt(p.cold)}, speedup hot/cold = {speedup_hot:.1f}x / {speedup_cold:.1f}x")


if __name__ == "__main__":
    main()
