from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))

from tinynpu_jit.blocks.llama_block import (  # noqa: E402
    QLlamaBlock,
    QLlamaBlockConfig,
    build_decode_artifact,
    build_prefill_artifact,
)
from tinynpu_jit.memory_planner import plan_program_memory  # noqa: E402


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def _linear_weight(fp32: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    # PyTorch Linear stores (out_features, in_features); QLlamaBlock stores
    # matmul RHS layout (in_features, out_features).
    return np.asarray(fp32[name], dtype=np.float32).T.copy()


def _linear_weight_i16(qweights: np.lib.npyio.NpzFile, name: str) -> np.ndarray:
    return np.asarray(qweights[name], dtype=np.int16).T.copy()


def _calibrate_act_scale(fp32: np.lib.npyio.NpzFile) -> float:
    # First-edition conservative scale: embeddings are the direct model input to
    # layer 0, and trained RMSNorm weights are near one. Later we should replace
    # this with trace-based activation calibration over real prompts.
    emb = np.asarray(fp32["tok_embeddings.weight"], dtype=np.float32)
    max_abs = float(np.max(np.abs(emb))) if emb.size else 1.0
    return max(max_abs / 30000.0, 1.0 / 32768.0)


def load_qllama_layer(run_dir: Path, *, layer: int, act_scale: float | None = None) -> tuple[QLlamaBlock, dict[str, object]]:
    config_json = _load_json(run_dir / "tinystories_char_lm_config.json")
    model_cfg = dict(config_json["config"])  # type: ignore[arg-type]
    fp32 = np.load(run_dir / "tinystories_char_lm_fp32.npz")
    qweights = np.load(run_dir / "tinystories_char_lm_int16_weights.npz")
    quant_config_path = run_dir / "qllama_quant_config.json"
    quant_layer = None
    quant_global = None
    if quant_config_path.exists():
        quant_config = _load_json(quant_config_path)
        layers = dict(quant_config.get("layers", {}))
        quant_layer = dict(layers[str(layer)])
        quant_global = dict(quant_config.get("quantization", {}))
    resolved_act_scale = (
        float(act_scale)
        if act_scale is not None
        else float(dict(quant_layer.get("activation_scales", {})).get("rms1_out", _calibrate_act_scale(fp32)))
        if quant_layer is not None
        else _calibrate_act_scale(fp32)
    )
    resolved_attn_scale = float(quant_global.get("attn_scale", 1.0 / 256.0)) if quant_global is not None else 1.0 / 256.0
    cfg = QLlamaBlockConfig(
        d_model=int(model_cfg["d_model"]),
        d_head=int(model_cfg["d_head"]),
        n_heads=int(model_cfg["n_heads"]),
        n_kv_heads=int(model_cfg["n_kv_heads"]),
        ffn_hidden_dim=int(model_cfg["ffn_hidden_dim"]),
        act_scale=resolved_act_scale,
        attn_scale=resolved_attn_scale,
        rms_norm_eps=float(model_cfg.get("rms_norm_eps", 1.0e-5)),
        rope_theta=float(model_cfg.get("rope_theta", 500000.0)),
    )
    prefix = f"layers.{layer}"
    block = QLlamaBlock(
        config=cfg,
        input_layernorm_w=np.asarray(fp32[f"{prefix}.rms1.weight"], dtype=np.float32),
        self_attn_q_proj_w=_linear_weight_i16(qweights, f"{prefix}.attn.q_proj.weight"),
        self_attn_k_proj_w=_linear_weight_i16(qweights, f"{prefix}.attn.k_proj.weight"),
        self_attn_v_proj_w=_linear_weight_i16(qweights, f"{prefix}.attn.v_proj.weight"),
        self_attn_o_proj_w=_linear_weight_i16(qweights, f"{prefix}.attn.o_proj.weight"),
        post_attention_layernorm_w=np.asarray(fp32[f"{prefix}.rms2.weight"], dtype=np.float32),
        mlp_gate_proj_w=_linear_weight_i16(qweights, f"{prefix}.mlp.gate_proj.weight"),
        mlp_up_proj_w=_linear_weight_i16(qweights, f"{prefix}.mlp.up_proj.weight"),
        mlp_down_proj_w=_linear_weight_i16(qweights, f"{prefix}.mlp.down_proj.weight"),
        quant_config=quant_layer,
    )
    return block, config_json


def _encode_prompt(metadata: dict[str, object], prompt: str | None) -> np.ndarray | None:
    if prompt is None:
        return None
    tok = dict(metadata["tokenizer"])  # type: ignore[arg-type]
    stoi = {str(token): idx for idx, token in enumerate(tok["itos"])}  # type: ignore[index]
    ids = [int(stoi["<bos>"])]
    if str(tok.get("kind", "char")) == "word":
        import re

        pieces = re.findall(r"[a-z0-9]+|[.,!?;:'\"()/-]", prompt.lower())
        ids.extend(int(stoi.get(piece, stoi["<unk>"])) for piece in pieces)
    else:
        ids.extend(int(stoi.get(ch, stoi["<unk>"])) for ch in prompt.lower())
    return np.asarray(ids, dtype=np.int64)


def _state_for_block(
    block: QLlamaBlock,
    *,
    run_dir: Path,
    metadata: dict[str, object],
    prompt_len: int,
    seed: int,
    prompt: str | None = None,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    token_embeddings = np.load(run_dir / "tinystories_char_lm_fp32.npz")["tok_embeddings.weight"]
    encoded = _encode_prompt(metadata, prompt)
    if encoded is None:
        token_ids = np.arange(prompt_len, dtype=np.int64) % int(token_embeddings.shape[0])
    else:
        if encoded.size > prompt_len:
            token_ids = encoded[:prompt_len]
        else:
            pad_id = 0
            token_ids = np.pad(encoded, (0, prompt_len - encoded.size), constant_values=pad_id)
    x_prompt = np.asarray(token_embeddings[token_ids], dtype=np.float32)
    x_decode = np.asarray(token_embeddings[[int(token_ids[-1])]], dtype=np.float32)
    if x_prompt.shape[1] != block.config.d_model:
        x_prompt = rng.uniform(-0.05, 0.05, size=(prompt_len, block.config.d_model)).astype(np.float32)
        x_decode = rng.uniform(-0.05, 0.05, size=(1, block.config.d_model)).astype(np.float32)
    return {
        "config": block.config,
        "block": block,
        "x_prompt_in": x_prompt,
        "x_decode_in": x_decode,
        "token_ids": token_ids,
    }


def _summarize_artifact(name: str, artifact) -> None:
    report = plan_program_memory(artifact.plan, ub_capacity=0xF000)
    max_im = max(len(segment.binary["im"]) for segment in artifact.segment_artifacts.values())
    print(
        f"{name}: segments={len(artifact.segment_artifacts)} "
        f"ub_peak={report.total_ub_peak}/61440 static={report.static_zone_end} max_im={max_im}/1024"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/tinystories_char_lm"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--act-scale", type=float, default=None)
    args = parser.parse_args()

    block, metadata = load_qllama_layer(args.run_dir, layer=args.layer, act_scale=args.act_scale)
    cfg = block.config
    state = _state_for_block(
        block,
        run_dir=args.run_dir,
        metadata=metadata,
        prompt_len=args.prompt_len,
        seed=args.seed,
        prompt=args.prompt,
    )
    print(
        f"loaded layer={args.layer} d_model={cfg.d_model} d_head={cfg.d_head} "
        f"n_heads={cfg.n_heads} n_kv_heads={cfg.n_kv_heads} ffn={cfg.ffn_hidden_dim} "
        f"act_scale={cfg.act_scale:.8g}"
    )
    print(f"tokenizer_vocab={len(metadata['tokenizer']['itos'])}")  # type: ignore[index]
    if args.prompt is not None:
        print(f"prompt={args.prompt!r} token_ids={state['token_ids'].tolist()}")

    prefill, _, prefill_ref = build_prefill_artifact(
        d_model=cfg.d_model,
        d_head=cfg.d_head,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        ffn_hidden_dim=cfg.ffn_hidden_dim,
        prompt_len=args.prompt_len,
        act_scale=cfg.act_scale,
        attn_scale=cfg.attn_scale,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        state=state,
    )
    decode, _, _, decode_ref = build_decode_artifact(
        d_model=cfg.d_model,
        d_head=cfg.d_head,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        ffn_hidden_dim=cfg.ffn_hidden_dim,
        prompt_len=args.prompt_len,
        act_scale=cfg.act_scale,
        attn_scale=cfg.attn_scale,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        state=state,
    )
    _summarize_artifact("prefill", prefill)
    _summarize_artifact("decode", decode)
    print(f"prefill_out_checksum={float(np.asarray(prefill_ref['out'], dtype=np.float32).sum()):.6f}")
    print(f"decode_out_checksum={float(np.asarray(decode_ref['out'], dtype=np.float32).sum()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
