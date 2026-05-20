from __future__ import annotations

import argparse
import os
import re
import subprocess
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
from run_tinystories_qllama_rtl import _build_c_elf_and_hex  # noqa: E402
from tinynpu_jit.baremetal_emit import emit_cv32e40p_c  # noqa: E402
from tinynpu_jit.blocks.llama_block import build_decode_artifact, reference_prefill  # noqa: E402
from tinynpu_jit.rtl_runner import run_vlt_npu  # noqa: E402


_ROW_RE = re.compile(r"^  row 0:(?P<values>(?:\s+-?\d+\.\d+)+)\s*$", re.MULTILINE)


def _load_checkpoint(run_dir: Path) -> tuple[stories.ToyDialogueLM, stories.WordTokenizer]:
    ckpt = torch.load(run_dir / "tinystories_char_lm.pt", map_location="cpu")
    cfg = stories.ToyLMConfig(**ckpt["config"])
    tok_json = ckpt["tokenizer"]
    if tok_json.get("kind") != "word":
        raise ValueError("this runner expects the word tokenizer checkpoint")
    tokenizer = stories.WordTokenizer(list(tok_json["words"]), vocab_size=len(tok_json["itos"]))
    model = stories.ToyDialogueLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer


def _encode(tokenizer: stories.WordTokenizer, prompt: str) -> list[int]:
    return tokenizer.encode(stories.normalize_text(prompt).rstrip("\n"), add_bos=True, add_eos=False)


def _parse_out(stdout: str, *, d_model: int) -> np.ndarray:
    match = _ROW_RE.search(stdout)
    if match is None:
        raise RuntimeError("could not parse final output row from RTL stdout")
    values = [float(v) for v in match.group("values").split()]
    if len(values) != d_model:
        raise RuntimeError(f"parsed {len(values)} output values, expected {d_model}")
    return np.asarray(values, dtype=np.float32).reshape(1, d_model)


def _run_decode_rtl(
    *,
    block,
    state: dict[str, object],
    layer: int,
    prompt_len: int,
    label: str,
    maxcycles: int,
    verilator_max_ticks: int,
    timeout_s: int,
) -> tuple[np.ndarray, str]:
    cfg = block.config
    artifact, _, _, ref = build_decode_artifact(
        d_model=cfg.d_model,
        d_head=cfg.d_head,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        ffn_hidden_dim=cfg.ffn_hidden_dim,
        prompt_len=prompt_len,
        act_scale=cfg.act_scale,
        attn_scale=cfg.attn_scale,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        state=state,
    )
    program_name = f"cv32e40p_tinystories_nexttok_l{layer}_{label}_ctx{prompt_len}"
    source = emit_cv32e40p_c(artifact, {}, program_name=program_name, repeat_count=1, cpu_only_baseline=False)
    hex_path = _build_c_elf_and_hex(program_name, source)
    proc = run_vlt_npu(
        hex_path,
        maxcycles=maxcycles,
        verilator_max_ticks=verilator_max_ticks,
        timeout_s=timeout_s,
        noassert=True,
    )
    if "EXIT SUCCESS" not in proc.stdout:
        raise RuntimeError(f"layer {layer} RTL run did not report EXIT SUCCESS")
    actual = _parse_out(proc.stdout, d_model=cfg.d_model)
    ref_out = np.asarray(ref["out"], dtype=np.float32)
    max_err = float(np.max(np.abs(actual - ref_out)))
    print(
        f"layer{layer}.rtl checksum={float(actual.sum()):.6f} "
        f"reference_checksum={float(ref_out.sum()):.6f} parsed_print_max_err={max_err:.6f}",
        flush=True,
    )
    return actual, proc.stdout


def _rmsnorm_np(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    xf = np.asarray(x, dtype=np.float32)
    wf = np.asarray(weight, dtype=np.float32)
    inv = 1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + np.float32(eps))
    return (xf * inv * wf).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/tinystories_word_lm_d32_t17"))
    parser.add_argument("--prompt", default="there was a little")
    parser.add_argument("--maxcycles", type=int, default=1_500_000)
    parser.add_argument("--verilator-max-ticks", type=int, default=6_000_000_000)
    parser.add_argument("--timeout-s", type=int, default=600)
    args = parser.parse_args()

    model, tokenizer = _load_checkpoint(args.run_dir)
    token_ids = _encode(tokenizer, args.prompt)
    if len(token_ids) < 2:
        raise ValueError("prompt must contain at least one token after BOS")
    cache_ids = token_ids[:-1]
    decode_id = token_ids[-1]

    fp32 = np.load(args.run_dir / "tinystories_char_lm_fp32.npz")
    embeddings = np.asarray(fp32["tok_embeddings.weight"], dtype=np.float32)
    final_norm_w = np.asarray(fp32["norm.weight"], dtype=np.float32)
    lm_head_w = embeddings

    block0, _ = load_qllama_layer(args.run_dir, layer=0)
    block1, _ = load_qllama_layer(args.run_dir, layer=1)

    x_cache0 = embeddings[np.asarray(cache_ids, dtype=np.int64)]
    x_decode0 = embeddings[np.asarray([decode_id], dtype=np.int64)]
    state0 = {
        "config": block0.config,
        "block": block0,
        "x_prompt_in": np.asarray(x_cache0, dtype=np.float32),
        "x_decode_in": np.asarray(x_decode0, dtype=np.float32),
    }

    layer0_prefill = reference_prefill(
        state0,
        d_head=block0.config.d_head,
        n_heads=block0.config.n_heads,
        n_kv_heads=block0.config.n_kv_heads,
        act_scale=block0.config.act_scale,
        attn_scale=block0.config.attn_scale,
        rope_theta=block0.config.rope_theta,
    )
    print(f"prompt={args.prompt!r}", flush=True)
    print(f"token_ids={token_ids} cache_ids={cache_ids} decode_id={decode_id} decode_token={tokenizer.itos[decode_id]!r}", flush=True)
    print("running layer0 decode on RTL", flush=True)
    layer0_out, _ = _run_decode_rtl(
        block=block0,
        state=state0,
        layer=0,
        prompt_len=len(cache_ids),
        label="prompt_last",
        maxcycles=args.maxcycles,
        verilator_max_ticks=args.verilator_max_ticks,
        timeout_s=args.timeout_s,
    )

    state1 = {
        "config": block1.config,
        "block": block1,
        "x_prompt_in": np.asarray(layer0_prefill["out"], dtype=np.float32),
        "x_decode_in": layer0_out,
    }
    print("running layer1 decode on RTL", flush=True)
    layer1_out, _ = _run_decode_rtl(
        block=block1,
        state=state1,
        layer=1,
        prompt_len=len(cache_ids),
        label="prompt_last",
        maxcycles=args.maxcycles,
        verilator_max_ticks=args.verilator_max_ticks,
        timeout_s=args.timeout_s,
    )

    final_hidden = _rmsnorm_np(layer1_out, final_norm_w, model.cfg.rms_norm_eps)
    logits = final_hidden @ lm_head_w.T
    probs = torch.softmax(torch.from_numpy(logits[0]), dim=-1)
    top = torch.topk(probs, k=10)
    next_id = int(top.indices[0].item())
    print(f"rtl_backed_next_id={next_id}", flush=True)
    print(f"rtl_backed_next_token={tokenizer.itos[next_id]!r}", flush=True)
    print("top10:", flush=True)
    for prob, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {int(idx):3d} {tokenizer.itos[int(idx)]!r:>12} p={prob:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
