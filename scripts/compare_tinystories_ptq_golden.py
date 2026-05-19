from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))
if str(REPO_ROOT / "software" / "workload") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "workload"))

import stories  # noqa: E402
from import_tinystories_to_qllama import load_qllama_layer  # noqa: E402
from tinynpu_jit.blocks.llama_block import reference_decode, reference_prefill  # noqa: E402


def _load_model(run_dir: Path) -> tuple[stories.ToyDialogueLM, stories.WordTokenizer]:
    ckpt = torch.load(run_dir / "tinystories_char_lm.pt", map_location="cpu")
    cfg = stories.ToyLMConfig(**ckpt["config"])
    tok_json = ckpt["tokenizer"]
    if tok_json.get("kind") != "word":
        raise ValueError("this comparison runner expects the word tokenizer checkpoint")
    tokenizer = stories.WordTokenizer(list(tok_json["words"]), vocab_size=len(tok_json["itos"]))
    model = stories.ToyDialogueLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer


def _rmsnorm_np(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    xf = np.asarray(x, dtype=np.float32)
    wf = np.asarray(weight, dtype=np.float32)
    inv = 1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + np.float32(eps))
    return (xf * inv * wf).astype(np.float32)


def _topk(probs: torch.Tensor, tokenizer: stories.WordTokenizer, *, k: int) -> list[tuple[int, str, float]]:
    top = torch.topk(probs, k=k)
    return [(int(idx), tokenizer.itos[int(idx)], float(prob)) for prob, idx in zip(top.values.tolist(), top.indices.tolist())]


def _print_top(label: str, rows: list[tuple[int, str, float]]) -> None:
    print(f"{label}:")
    for idx, token, prob in rows:
        print(f"  {idx:3d} {token!r:>12} p={prob:.4f}")


def _mask_specials(probs: torch.Tensor, tokenizer: stories.WordTokenizer) -> torch.Tensor:
    masked = probs.clone()
    for idx in (tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.unk_id):
        masked[int(idx)] = 0.0
    denom = masked.sum()
    return masked / denom if float(denom) > 0.0 else probs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/tinystories_word_lm_d32_t17"))
    parser.add_argument("--prompt", default="there was a little")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    model, tokenizer = _load_model(args.run_dir)
    token_ids = tokenizer.encode(stories.normalize_text(args.prompt).rstrip("\n"), add_bos=True, add_eos=False)
    if len(token_ids) < 2:
        raise ValueError("prompt must contain at least one token after BOS")
    cache_ids = token_ids[:-1]
    decode_id = token_ids[-1]

    with torch.inference_mode():
        fp_logits, _ = model(torch.tensor([token_ids], dtype=torch.long))
        fp_probs = torch.softmax(fp_logits[0, -1], dim=-1)

    fp32 = np.load(args.run_dir / "tinystories_char_lm_fp32.npz")
    embeddings = np.asarray(fp32["tok_embeddings.weight"], dtype=np.float32)
    final_norm_w = np.asarray(fp32["norm.weight"], dtype=np.float32)
    block0, _ = load_qllama_layer(args.run_dir, layer=0)
    block1, _ = load_qllama_layer(args.run_dir, layer=1)

    state0 = {
        "config": block0.config,
        "block": block0,
        "x_prompt_in": embeddings[np.asarray(cache_ids, dtype=np.int64)],
        "x_decode_in": embeddings[np.asarray([decode_id], dtype=np.int64)],
    }
    prefill0 = reference_prefill(
        state0,
        d_head=block0.config.d_head,
        n_heads=block0.config.n_heads,
        n_kv_heads=block0.config.n_kv_heads,
        act_scale=block0.config.act_scale,
        attn_scale=block0.config.attn_scale,
        rope_theta=block0.config.rope_theta,
    )
    decode0 = reference_decode(
        state0,
        prefill0,
        d_head=block0.config.d_head,
        n_heads=block0.config.n_heads,
        n_kv_heads=block0.config.n_kv_heads,
        act_scale=block0.config.act_scale,
        attn_scale=block0.config.attn_scale,
        rope_theta=block0.config.rope_theta,
    )
    state1 = {
        "config": block1.config,
        "block": block1,
        "x_prompt_in": np.asarray(prefill0["out"], dtype=np.float32),
        "x_decode_in": np.asarray(decode0["out"], dtype=np.float32),
    }
    prefill1 = reference_prefill(
        state1,
        d_head=block1.config.d_head,
        n_heads=block1.config.n_heads,
        n_kv_heads=block1.config.n_kv_heads,
        act_scale=block1.config.act_scale,
        attn_scale=block1.config.attn_scale,
        rope_theta=block1.config.rope_theta,
    )
    decode1 = reference_decode(
        state1,
        prefill1,
        d_head=block1.config.d_head,
        n_heads=block1.config.n_heads,
        n_kv_heads=block1.config.n_kv_heads,
        act_scale=block1.config.act_scale,
        attn_scale=block1.config.attn_scale,
        rope_theta=block1.config.rope_theta,
    )
    q_hidden = _rmsnorm_np(np.asarray(decode1["out"], dtype=np.float32), final_norm_w, model.cfg.rms_norm_eps)
    q_logits = q_hidden @ embeddings.T
    q_probs = torch.softmax(torch.from_numpy(q_logits[0]), dim=-1)

    print(f"prompt={args.prompt!r}")
    print(f"token_ids={token_ids} cache_ids={cache_ids} decode_token={tokenizer.itos[decode_id]!r}")
    _print_top("FP PyTorch raw", _topk(fp_probs, tokenizer, k=args.top_k))
    _print_top("PTQ QLlama golden raw", _topk(q_probs, tokenizer, k=args.top_k))
    _print_top("FP PyTorch masked-special", _topk(_mask_specials(fp_probs, tokenizer), tokenizer, k=args.top_k))
    _print_top("PTQ QLlama golden masked-special", _topk(_mask_specials(q_probs, tokenizer), tokenizer, k=args.top_k))
    print("checksums:")
    print(f"  ptq.layer0.decode_out={float(np.asarray(decode0['out'], dtype=np.float32).sum()):.6f}")
    print(f"  ptq.layer1.decode_out={float(np.asarray(decode1['out'], dtype=np.float32).sum()):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
