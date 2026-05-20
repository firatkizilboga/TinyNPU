from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "software" / "workload") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "workload"))

import stories  # noqa: E402


FALLBACK_CALIBRATION_TEXT = """
once upon a time there was a tiny bot.
the tiny bot was small but kind.
user: hello
bot: hello friend.
user: who are you
bot: i am a tiny npu bot.
user: are you chatgpt
bot: no i am a tiny toy lm.
user: are you claude
bot: no i am your best friend.
the child said hello to the little model.
the model said hello and helped with the story.
"""


class ScaleCollector:
    def __init__(self) -> None:
        self.values: dict[str, list[torch.Tensor]] = {}

    def observe(self, name: str, x: torch.Tensor) -> None:
        flat = x.detach().float().abs().reshape(-1).cpu()
        if flat.numel() == 0:
            return
        self.values.setdefault(name, []).append(flat)

    def scales(self, *, qmax: float, percentile: float, floor: float = 1.0e-8) -> dict[str, float]:
        result: dict[str, float] = {}
        q = torch.tensor(float(percentile) / 100.0)
        for name in sorted(self.values):
            xs = torch.cat(self.values[name])
            boundary = float(torch.quantile(xs, q).item())
            result[name] = max(boundary / float(qmax), floor)
        return result


def fit_small_multiplier_shift(real_multiplier: float, *, bits: int = 16, max_shift: int = 255) -> dict[str, float | int]:
    if real_multiplier <= 0.0:
        raise ValueError(f"real_multiplier must be positive, got {real_multiplier}")
    best: tuple[float, int, int, float] | None = None
    max_m = (1 << bits) - 1
    for shift in range(max_shift + 1):
        m = round(real_multiplier * (1 << shift))
        if 0 < m <= max_m:
            approx = float(m) / float(1 << shift)
            err = abs(approx - real_multiplier) / real_multiplier
            cand = (err, int(m), int(shift), approx)
            if best is None or cand < best:
                best = cand
    if best is None:
        raise ValueError(f"could not fit real_multiplier={real_multiplier}")
    err, multiplier, shift, approx = best
    return {
        "multiplier": multiplier,
        "shift": shift,
        "approx": approx,
        "relative_error": err,
    }


def _cfg_from_checkpoint(ckpt: dict[str, object]) -> stories.ToyLMConfig:
    return stories.ToyLMConfig(**dict(ckpt["config"]))  # type: ignore[arg-type]


def _tokenizer_from_checkpoint(ckpt: dict[str, object]):
    tok = dict(ckpt["tokenizer"])  # type: ignore[arg-type]
    kind = str(tok.get("kind", "char"))
    if kind == "word":
        return stories.WordTokenizer(list(tok["words"]), vocab_size=len(tok["itos"]))  # type: ignore[arg-type]
    return stories.CharTokenizer(str(tok["chars"]), vocab_size=len(tok["itos"]))  # type: ignore[arg-type]


def _load_model(run_dir: Path, device: torch.device) -> tuple[stories.ToyDialogueLM, object, dict[str, object]]:
    ckpt = torch.load(run_dir / "tinystories_char_lm.pt", map_location=device)
    cfg = _cfg_from_checkpoint(ckpt)
    tokenizer = _tokenizer_from_checkpoint(ckpt)
    model = stories.ToyDialogueLM(cfg).to(device)
    model.load_state_dict(ckpt["model"])  # type: ignore[arg-type]
    model.eval()
    meta = json.loads((run_dir / "tinystories_char_lm_config.json").read_text())
    return model, tokenizer, meta


def _load_calibration_text(args: argparse.Namespace) -> str:
    if args.data_file is not None:
        return stories.normalize_text(args.data_file.read_text(encoding="utf-8", errors="replace"))
    cache = args.run_dir / "calibration_text.txt"
    if cache.exists():
        return stories.normalize_text(cache.read_text(encoding="utf-8", errors="replace"))
    repo_cache = REPO_ROOT / "data" / "tinystories_head.txt"
    if repo_cache.exists():
        return stories.normalize_text(repo_cache.read_text(encoding="utf-8", errors="replace"))
    return stories.normalize_text(FALLBACK_CALIBRATION_TEXT)


def _make_chunks(
    text: str,
    tokenizer,
    *,
    max_seq_len: int,
    stride: int,
    max_chunks: int,
    device: torch.device,
) -> torch.Tensor:
    ids = torch.tensor(tokenizer.encode(text, add_bos=True, add_eos=True), dtype=torch.long)
    if ids.numel() < max_seq_len + 1:
        reps = (max_seq_len + 1 + ids.numel() - 1) // max(ids.numel(), 1)
        ids = ids.repeat(reps + 1)
    stride = max(1, int(stride))
    n = 1 + max(0, (ids.numel() - (max_seq_len + 1)) // stride)
    n = min(n, int(max_chunks))
    windows = torch.as_strided(ids, size=(n, max_seq_len + 1), stride=(stride, 1)).contiguous()
    return windows[:, :-1].to(device)


def _manual_forward(model: stories.ToyDialogueLM, input_ids: torch.Tensor, collector: ScaleCollector) -> torch.Tensor:
    cfg = model.cfg
    x = model.tok_embeddings(input_ids)
    collector.observe("embedding_out", x)
    for layer_idx, layer in enumerate(model.layers):
        prefix = f"layer{layer_idx}"
        residual = x
        x_norm = layer.rms1(x)
        collector.observe(f"{prefix}.rms1_out", x_norm)

        attn = layer.attn
        bsz, seq_len, _ = x_norm.shape
        q = attn.q_proj(x_norm).view(bsz, seq_len, cfg.n_heads, cfg.d_head)
        k = attn.k_proj(x_norm).view(bsz, seq_len, cfg.n_kv_heads, cfg.d_head)
        v = attn.v_proj(x_norm).view(bsz, seq_len, cfg.n_kv_heads, cfg.d_head)
        collector.observe(f"{prefix}.q_out", q)
        collector.observe(f"{prefix}.k_out", k)
        collector.observe(f"{prefix}.v_out", v)

        cos = attn.rope_cos[:seq_len]
        sin = attn.rope_sin[:seq_len]
        q_rope = stories.apply_rope(q, cos, sin).transpose(1, 2)
        k_rope = stories.apply_rope(k, cos, sin).transpose(1, 2)
        v_heads = v.transpose(1, 2)
        collector.observe(f"{prefix}.rope_q", q_rope)
        collector.observe(f"{prefix}.rope_k", k_rope)

        repeat = cfg.n_heads // cfg.n_kv_heads
        if repeat != 1:
            k_read = k_rope.repeat_interleave(repeat, dim=1)
            v_read = v_heads.repeat_interleave(repeat, dim=1)
        else:
            k_read = k_rope
            v_read = v_heads

        scores = torch.matmul(q_rope, k_read.transpose(-2, -1)) / math.sqrt(cfg.d_head)
        collector.observe(f"{prefix}.score_logits", scores)
        mask = torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask[None, None, :, :], float("-inf"))
        probs = F.softmax(scores, dim=-1)
        collector.observe(f"{prefix}.attn_probs", probs)
        attn_cat = torch.matmul(probs, v_read).transpose(1, 2).contiguous().view(bsz, seq_len, cfg.n_heads * cfg.d_head)
        collector.observe(f"{prefix}.attn_cat", attn_cat)
        o_out = attn.o_proj(attn_cat)
        collector.observe(f"{prefix}.o_out", o_out)
        x = residual + o_out
        collector.observe(f"{prefix}.resid1", x)

        residual = x
        x_norm2 = layer.rms2(x)
        collector.observe(f"{prefix}.rms2_out", x_norm2)
        gate = layer.mlp.gate_proj(x_norm2)
        up = layer.mlp.up_proj(x_norm2)
        collector.observe(f"{prefix}.gate_out", gate)
        collector.observe(f"{prefix}.up_out", up)
        silu_gate = F.silu(gate)
        collector.observe(f"{prefix}.silu_gate", silu_gate)
        ffn_hidden = silu_gate * up
        collector.observe(f"{prefix}.ffn_hidden", ffn_hidden)
        down = layer.mlp.down_proj(ffn_hidden)
        collector.observe(f"{prefix}.down_out", down)
        x = residual + down
        collector.observe(f"{prefix}.resid2", x)

    final_norm = model.norm(x)
    collector.observe("final_norm", final_norm)
    logits = model.lm_head(final_norm)
    collector.observe("lm_head_logits", logits)
    return logits


def _weight_scale(meta: dict[str, object], name: str, model: stories.ToyDialogueLM, *, qmax: float) -> float:
    scales = dict(meta.get("weight_scales", {}))
    if name in scales:
        return float(scales[name])
    tensor = dict(model.state_dict())[name]
    return max(float(tensor.detach().abs().max().item()) / float(qmax), 1.0e-8)


def _matmul_entry(
    *,
    input_scale: float,
    rhs_scale: float,
    output_scale: float,
    bits: int,
    in_bits: int = 16,
    rhs_bits: int = 16,
    out_bits: int = 16,
) -> dict[str, object]:
    real = float(input_scale) * float(rhs_scale) / float(output_scale)
    fitted = fit_small_multiplier_shift(real, bits=bits)
    return {
        "input_scale": input_scale,
        "rhs_scale": rhs_scale,
        "output_scale": output_scale,
        "in_bits": int(in_bits),
        "rhs_bits": int(rhs_bits),
        "out_bits": int(out_bits),
        "real_multiplier": real,
        **fitted,
    }


def _precision_policy(name: str, *, allow_experimental_int8_score_export: bool = False) -> dict[str, int]:
    if name == "int16_baseline":
        return {
            "activation_bits": 16,
            "weight_bits": 16,
            "score_out_bits": 16,
            "prob_bits": 16,
            "value_cache_bits": 16,
        }
    if name == "int8_scores":
        if not allow_experimental_int8_score_export:
            raise NotImplementedError(
                "int8_scores is implemented as an experimental QLlama lowering, "
                "but the RTL smoke is not clean yet. Re-run calibration with "
                "--allow-experimental-int8-score-export if you intentionally want "
                "to debug this path."
            )
        return {
            "activation_bits": 16,
            "weight_bits": 16,
            "score_out_bits": 8,
            "prob_bits": 16,
            "value_cache_bits": 16,
        }
    if name == "int8_value":
        raise NotImplementedError(
            "int8_value is available as a QAT stress policy, but it is not a "
            "compiler-safe export policy yet. V-cache INT8 needs an explicit "
            "writeback/read layout instead of pretending the current INT16 "
            "cache path can consume INT8 probabilities and values."
        )
    raise ValueError(f"unknown precision policy: {name}")


def build_quant_config(
    model: stories.ToyDialogueLM,
    meta: dict[str, object],
    act_scales: dict[str, float],
    *,
    qmax_weight: float,
    multiplier_bits: int,
    attn_scale: float,
    precision_policy: str,
    allow_experimental_int8_score_export: bool = False,
) -> dict[str, object]:
    cfg = model.cfg
    policy = _precision_policy(
        precision_policy,
        allow_experimental_int8_score_export=allow_experimental_int8_score_export,
    )
    score_qmax = float((1 << (int(policy["score_out_bits"]) - 1)) - 1)
    weight_scales = {name: _weight_scale(meta, name, model, qmax=qmax_weight) for name, tensor in model.state_dict().items() if tensor.ndim >= 2}
    layers: dict[str, object] = {}
    for layer_idx in range(cfg.n_layers):
        prefix = f"layer{layer_idx}"
        wprefix = f"layers.{layer_idx}"
        layer_scales = {name.removeprefix(prefix + "."): value for name, value in act_scales.items() if name.startswith(prefix + ".")}
        score_output_scale = act_scales[f"{prefix}.score_logits"] / score_qmax
        matmuls = {
            "q_proj": _matmul_entry(input_scale=act_scales[f"{prefix}.rms1_out"], rhs_scale=weight_scales[f"{wprefix}.attn.q_proj.weight"], output_scale=act_scales[f"{prefix}.q_out"], bits=multiplier_bits),
            "k_proj": _matmul_entry(input_scale=act_scales[f"{prefix}.rms1_out"], rhs_scale=weight_scales[f"{wprefix}.attn.k_proj.weight"], output_scale=act_scales[f"{prefix}.k_out"], bits=multiplier_bits),
            "v_proj": _matmul_entry(input_scale=act_scales[f"{prefix}.rms1_out"], rhs_scale=weight_scales[f"{wprefix}.attn.v_proj.weight"], output_scale=act_scales[f"{prefix}.v_out"], bits=multiplier_bits),
            "score": _matmul_entry(
                input_scale=act_scales[f"{prefix}.rope_q"],
                rhs_scale=act_scales[f"{prefix}.rope_k"] / math.sqrt(float(cfg.d_head)),
                output_scale=score_output_scale,
                bits=multiplier_bits,
                out_bits=int(policy["score_out_bits"]),
            ),
            "value": _matmul_entry(input_scale=attn_scale, rhs_scale=act_scales[f"{prefix}.v_out"], output_scale=act_scales[f"{prefix}.attn_cat"], bits=multiplier_bits),
            "o_proj": _matmul_entry(input_scale=act_scales[f"{prefix}.attn_cat"], rhs_scale=weight_scales[f"{wprefix}.attn.o_proj.weight"], output_scale=act_scales[f"{prefix}.o_out"], bits=multiplier_bits),
            "gate_proj": _matmul_entry(input_scale=act_scales[f"{prefix}.rms2_out"], rhs_scale=weight_scales[f"{wprefix}.mlp.gate_proj.weight"], output_scale=act_scales[f"{prefix}.gate_out"], bits=multiplier_bits),
            "up_proj": _matmul_entry(input_scale=act_scales[f"{prefix}.rms2_out"], rhs_scale=weight_scales[f"{wprefix}.mlp.up_proj.weight"], output_scale=act_scales[f"{prefix}.up_out"], bits=multiplier_bits),
            "down_proj": _matmul_entry(input_scale=act_scales[f"{prefix}.ffn_hidden"], rhs_scale=weight_scales[f"{wprefix}.mlp.down_proj.weight"], output_scale=act_scales[f"{prefix}.down_out"], bits=multiplier_bits),
        }
        layers[str(layer_idx)] = {
            "activation_scales": layer_scales,
            "score_scale": act_scales[f"{prefix}.rope_q"] * act_scales[f"{prefix}.rope_k"] / math.sqrt(float(cfg.d_head)),
            "matmuls": matmuls,
        }

    return {
        "schema": "tinynpu.qllama_quant_config.v1",
        "model": {
            "vocab_size": cfg.vocab_size,
            "max_seq_len": cfg.max_seq_len,
            "n_layers": cfg.n_layers,
            "d_model": cfg.d_model,
            "d_head": cfg.d_head,
            "n_heads": cfg.n_heads,
            "n_kv_heads": cfg.n_kv_heads,
            "ffn_hidden_dim": cfg.ffn_hidden_dim,
        },
        "quantization": {
            "activation_qmax": 32767,
            "weight_qmax": int(qmax_weight),
            "attn_scale": attn_scale,
            "multiplier_bits": multiplier_bits,
            "precision_policy": precision_policy,
            **policy,
        },
        "global_activation_scales": {
            "embedding_out": act_scales["embedding_out"],
            "final_norm": act_scales["final_norm"],
            "lm_head_logits": act_scales["lm_head_logits"],
        },
        "weight_scales": weight_scales,
        "lm_head": _matmul_entry(input_scale=act_scales["final_norm"], rhs_scale=weight_scales["lm_head.weight"], output_scale=act_scales["lm_head_logits"], bits=multiplier_bits),
        "layers": layers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/tinystories_char_lm"))
    parser.add_argument("--data-file", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-chunks", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--calib-seq-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--percentile", type=float, default=99.9)
    parser.add_argument("--activation-qmax", type=float, default=32767.0)
    parser.add_argument("--weight-qmax", type=float, default=32767.0)
    parser.add_argument("--multiplier-bits", type=int, default=16)
    parser.add_argument("--attn-scale", type=float, default=1.0 / 256.0)
    parser.add_argument(
        "--precision-policy",
        choices=("int16_baseline", "int8_scores", "int8_value"),
        default="int16_baseline",
    )
    parser.add_argument(
        "--allow-experimental-int8-score-export",
        action="store_true",
        help="Permit INT8 QK-score output export. This path is wired, but current RTL smoke is not clean.",
    )
    args = parser.parse_args()
    if args.precision_policy == "int8_value" and abs(float(args.attn_scale) - (1.0 / 256.0)) < 1.0e-12:
        args.attn_scale = 1.0 / 127.0

    device = torch.device(args.device)
    model, tokenizer, meta = _load_model(args.run_dir, device)
    text = _load_calibration_text(args)
    chunks = _make_chunks(
        text,
        tokenizer,
        max_seq_len=min(int(args.calib_seq_len), int(model.cfg.max_seq_len)),
        stride=args.stride,
        max_chunks=args.max_chunks,
        device=device,
    )
    collector = ScaleCollector()
    with torch.inference_mode():
        for start in range(0, chunks.size(0), args.batch_size):
            _manual_forward(model, chunks[start : start + args.batch_size], collector)
    act_scales = collector.scales(qmax=args.activation_qmax, percentile=args.percentile)
    config = build_quant_config(
        model,
        meta,
        act_scales,
        qmax_weight=args.weight_qmax,
        multiplier_bits=args.multiplier_bits,
        attn_scale=args.attn_scale,
        precision_policy=args.precision_policy,
        allow_experimental_int8_score_export=args.allow_experimental_int8_score_export,
    )
    config["calibration"] = {
        "chunks": int(chunks.size(0)),
        "seq_len": int(chunks.size(1)),
        "stride": int(args.stride),
        "percentile": float(args.percentile),
        "data_file": str(args.data_file) if args.data_file is not None else None,
        "precision_policy": args.precision_policy,
    }
    out = args.out if args.out is not None else args.run_dir / "qllama_quant_config.json"
    out.write_text(json.dumps(config, indent=2))
    print(f"wrote={out}")
    print(f"chunks={chunks.size(0)} seq_len={chunks.size(1)} boundaries={len(act_scales)}")
    print(f"embedding_scale={config['global_activation_scales']['embedding_out']:.8g}")
    for layer_idx in range(model.cfg.n_layers):
        layer = config["layers"][str(layer_idx)]
        print(
            f"layer{layer_idx}: rms1={layer['activation_scales']['rms1_out']:.8g} "
            f"q={layer['activation_scales']['q_out']:.8g} "
            f"k={layer['activation_scales']['k_out']:.8g} "
            f"score={layer['activation_scales']['score_logits']:.8g} "
            f"v={layer['activation_scales']['v_out']:.8g} "
            f"ffn_hidden={layer['activation_scales']['ffn_hidden']:.8g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
