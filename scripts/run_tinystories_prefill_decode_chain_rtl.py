from __future__ import annotations

import argparse
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
from tinynpu_jit.blocks.llama_block import build_decode_artifact, build_prefill_artifact  # noqa: E402
from tinynpu_jit.ir import NpuSegment, TensorKind, VerificationMode  # noqa: E402
from tinynpu_jit.rtl_runner import run_vlt_npu  # noqa: E402


_HEADER_RE = re.compile(r"^(?P<name>[A-Za-z0-9_]+) shape=\((?P<shape>[^)]*)\)\s*$")
_ROW_RE = re.compile(r"^  row (?P<row>\d+):(?P<values>.*)$")


def _load_checkpoint(run_dir: Path) -> tuple[stories.ToyDialogueLM, object]:
    ckpt = torch.load(run_dir / "tinystories_char_lm.pt", map_location="cpu")
    cfg = stories.ToyLMConfig(**ckpt["config"])
    tokenizer = stories.tokenizer_from_json(ckpt["tokenizer"])
    model = stories.ToyDialogueLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, tokenizer


def _encode(tokenizer: object, text: str) -> list[int]:
    return tokenizer.encode(stories.normalize_text(text).rstrip("\n"), add_bos=True, add_eos=False)


def _parse_shape(text: str) -> tuple[int, ...]:
    parts = [part.strip() for part in text.split(",") if part.strip()]
    return tuple(int(part) for part in parts)


def _parse_final_tensors(stdout: str) -> dict[str, np.ndarray]:
    tensors: dict[str, np.ndarray] = {}
    lines = stdout.splitlines()
    idx = 0
    while idx < len(lines):
        header = _HEADER_RE.match(lines[idx])
        if header is None:
            idx += 1
            continue
        name = header.group("name")
        shape = _parse_shape(header.group("shape"))
        idx += 1
        if len(shape) == 2:
            rows: list[list[float]] = []
            while idx < len(lines):
                row = _ROW_RE.match(lines[idx])
                if row is None:
                    break
                rows.append([float(value) for value in row.group("values").split()])
                idx += 1
            tensors[name] = np.asarray(rows, dtype=np.float32).reshape(shape)
            continue
        if idx < len(lines) and lines[idx].startswith("  values:"):
            values = [float(value) for value in lines[idx].split(":", 1)[1].split()]
            tensors[name] = np.asarray(values, dtype=np.float32).reshape(shape)
            idx += 1
            continue
    return tensors


def _rmsnorm_np(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    xf = np.asarray(x, dtype=np.float32)
    wf = np.asarray(weight, dtype=np.float32)
    inv = 1.0 / np.sqrt(np.mean(xf * xf, axis=-1, keepdims=True) + np.float32(eps))
    return (xf * inv * wf).astype(np.float32)


def _externalize_decode_cache_inputs(artifact, *, n_kv_heads: int) -> None:
    for kv_head in range(n_kv_heads):
        for prefix in ("k", "v"):
            name = f"{prefix}_cache_h{kv_head}"
            spec = artifact.plan.tensors[name]
            spec.kind = TensorKind.INPUT
            spec.data = None
            if name not in artifact.plan.inputs:
                artifact.plan.inputs.append(name)
    for step in artifact.plan.steps:
        if not isinstance(step, NpuSegment):
            continue
        if step.name == "seg_score":
            for kv_head in range(n_kv_heads):
                name = f"k_cache_h{kv_head}"
                if name not in step.inputs:
                    step.inputs.append(name)
        elif step.name == "seg_value":
            for kv_head in range(n_kv_heads):
                name = f"v_cache_h{kv_head}"
                if name not in step.inputs:
                    step.inputs.append(name)


def _decode_cache_inputs_from_prefill(
    prefill_tensors: dict[str, np.ndarray],
    *,
    prompt_len: int,
    d_head: int,
    n_kv_heads: int,
) -> dict[str, np.ndarray]:
    inputs: dict[str, np.ndarray] = {}
    for kv_head in range(n_kv_heads):
        k_prefill = np.asarray(prefill_tensors[f"prefill_k_cache_h{kv_head}"], dtype=np.int16)
        v_prefill = np.asarray(prefill_tensors[f"prefill_v_cache_h{kv_head}"], dtype=np.int16)
        k_cache = np.zeros((d_head, prompt_len + 1), dtype=np.int16)
        v_cache = np.zeros((prompt_len + 1, d_head), dtype=np.int16)
        k_cache[:, :prompt_len] = k_prefill
        v_cache[:prompt_len, :] = v_prefill
        inputs[f"k_cache_h{kv_head}"] = k_cache
        inputs[f"v_cache_h{kv_head}"] = v_cache
    return inputs


def _run_artifact_rtl(
    artifact,
    *,
    inputs: dict[str, np.ndarray],
    program_name: str,
    maxcycles: int,
    verilator_max_ticks: int,
    timeout_s: int,
) -> tuple[dict[str, np.ndarray], str]:
    source = emit_cv32e40p_c(artifact, inputs, program_name=program_name, repeat_count=1, cpu_only_baseline=False)
    hex_path = _build_c_elf_and_hex(program_name, source)
    try:
        proc = run_vlt_npu(
            hex_path,
            maxcycles=maxcycles,
            verilator_max_ticks=verilator_max_ticks,
            timeout_s=timeout_s,
            noassert=True,
        )
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout)
        if exc.stderr:
            print(exc.stderr, file=sys.stderr)
        raise
    except subprocess.TimeoutExpired as exc:
        if exc.stdout:
            stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout
            print(stdout)
        if exc.stderr:
            stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr
            print(stderr, file=sys.stderr)
        raise
    if "EXIT SUCCESS" not in proc.stdout:
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
        raise RuntimeError(f"{program_name} did not report EXIT SUCCESS")
    return _parse_final_tensors(proc.stdout), proc.stdout


def _run_artifact_host(artifact, *, inputs: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], str]:
    result = artifact.run_host_emulation(inputs, verification=VerificationMode.DEBUG)
    return {name: np.asarray(value, copy=True) for name, value in result.tensors.items()}, "host"


def _run_artifact(
    artifact,
    *,
    backend: str,
    inputs: dict[str, np.ndarray],
    program_name: str,
    maxcycles: int,
    verilator_max_ticks: int,
    timeout_s: int,
) -> tuple[dict[str, np.ndarray], str]:
    if backend == "host":
        return _run_artifact_host(artifact, inputs=inputs)
    return _run_artifact_rtl(
        artifact,
        inputs=inputs,
        program_name=program_name,
        maxcycles=maxcycles,
        verilator_max_ticks=verilator_max_ticks,
        timeout_s=timeout_s,
    )


def _build_prefill(block, state: dict[str, object], *, prompt_len: int):
    cfg = block.config
    artifact, _, ref = build_prefill_artifact(
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
        expose_kv_cache_outputs=True,
        state=state,
    )
    return artifact, ref


def _build_decode(block, state: dict[str, object], *, prompt_len: int):
    cfg = block.config
    artifact, _, prefill_ref, decode_ref = build_decode_artifact(
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
    _externalize_decode_cache_inputs(artifact, n_kv_heads=cfg.n_kv_heads)
    return artifact, prefill_ref, decode_ref


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/tinystories_word_lm_d32_t17_qat_int16_long"))
    parser.add_argument("--prompt", default="there was a little girl named lily .")
    parser.add_argument("--backend", choices=("rtl", "host"), default="rtl")
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--maxcycles-prefill", type=int, default=10_000_000)
    parser.add_argument("--maxcycles-decode", type=int, default=3_000_000)
    parser.add_argument("--verilator-max-ticks", type=int, default=20_000_000_000)
    parser.add_argument("--timeout-s", type=int, default=600)
    args = parser.parse_args()

    model, tokenizer = _load_checkpoint(args.run_dir)
    token_ids = _encode(tokenizer, args.prompt)
    if len(token_ids) < args.prompt_len + 1:
        raise ValueError(
            f"prompt must encode to at least prompt_len + 1 tokens; got {len(token_ids)} tokens for prompt_len={args.prompt_len}"
        )
    cache_ids = token_ids[: args.prompt_len]
    decode_id = token_ids[args.prompt_len]

    fp32 = np.load(args.run_dir / "tinystories_char_lm_fp32.npz")
    embeddings = np.asarray(fp32["tok_embeddings.weight"], dtype=np.float32)
    final_norm_w = np.asarray(fp32["norm.weight"], dtype=np.float32)
    lm_head_w = embeddings

    block0, _ = load_qllama_layer(args.run_dir, layer=0)
    block1, _ = load_qllama_layer(args.run_dir, layer=1)
    cfg = block0.config
    print(
        f"run_dir={args.run_dir} d={cfg.d_model} h={cfg.d_head} nh={cfg.n_heads} "
        f"nkv={cfg.n_kv_heads} ffn={cfg.ffn_hidden_dim} prompt_len={args.prompt_len}",
        flush=True,
    )
    print(f"prompt={args.prompt!r}", flush=True)
    print(
        f"cache_ids={cache_ids} decode_id={decode_id} decode_token={tokenizer.itos[decode_id]!r}",
        flush=True,
    )

    state0 = {
        "config": block0.config,
        "block": block0,
        "x_prompt_in": embeddings[np.asarray(cache_ids, dtype=np.int64)],
        "x_decode_in": embeddings[np.asarray([decode_id], dtype=np.int64)],
    }
    prefill0_artifact, prefill0_ref = _build_prefill(block0, state0, prompt_len=args.prompt_len)
    print(f"running layer0 prefill on {args.backend}", flush=True)
    prefill0_tensors, _ = _run_artifact(
        prefill0_artifact,
        backend=args.backend,
        inputs={},
        program_name="cv32e40p_tinystories_chain_l0_prefill",
        maxcycles=args.maxcycles_prefill,
        verilator_max_ticks=args.verilator_max_ticks,
        timeout_s=args.timeout_s,
    )
    print(
        f"layer0.prefill.out checksum={float(prefill0_tensors['out'].sum()):.6f} "
        f"reference={float(np.asarray(prefill0_ref['out'], dtype=np.float32).sum()):.6f}",
        flush=True,
    )

    decode0_artifact, _, decode0_ref = _build_decode(block0, state0, prompt_len=args.prompt_len)
    cache0_inputs = _decode_cache_inputs_from_prefill(
        prefill0_tensors,
        prompt_len=args.prompt_len,
        d_head=block0.config.d_head,
        n_kv_heads=block0.config.n_kv_heads,
    )
    print(f"running layer0 decode with {args.backend}-produced prefill cache", flush=True)
    decode0_tensors, _ = _run_artifact(
        decode0_artifact,
        backend=args.backend,
        inputs=cache0_inputs,
        program_name="cv32e40p_tinystories_chain_l0_decode",
        maxcycles=args.maxcycles_decode,
        verilator_max_ticks=args.verilator_max_ticks,
        timeout_s=args.timeout_s,
    )
    print(
        f"layer0.decode.out checksum={float(decode0_tensors['out'].sum()):.6f} "
        f"reference={float(np.asarray(decode0_ref['out'], dtype=np.float32).sum()):.6f}",
        flush=True,
    )

    state1 = {
        "config": block1.config,
        "block": block1,
        "x_prompt_in": np.asarray(prefill0_tensors["out"], dtype=np.float32),
        "x_decode_in": np.asarray(decode0_tensors["out"], dtype=np.float32),
    }
    prefill1_artifact, prefill1_ref = _build_prefill(block1, state1, prompt_len=args.prompt_len)
    print(f"running layer1 prefill on {args.backend}", flush=True)
    prefill1_tensors, _ = _run_artifact(
        prefill1_artifact,
        backend=args.backend,
        inputs={},
        program_name="cv32e40p_tinystories_chain_l1_prefill",
        maxcycles=args.maxcycles_prefill,
        verilator_max_ticks=args.verilator_max_ticks,
        timeout_s=args.timeout_s,
    )
    print(
        f"layer1.prefill.out checksum={float(prefill1_tensors['out'].sum()):.6f} "
        f"reference={float(np.asarray(prefill1_ref['out'], dtype=np.float32).sum()):.6f}",
        flush=True,
    )

    decode1_artifact, _, decode1_ref = _build_decode(block1, state1, prompt_len=args.prompt_len)
    cache1_inputs = _decode_cache_inputs_from_prefill(
        prefill1_tensors,
        prompt_len=args.prompt_len,
        d_head=block1.config.d_head,
        n_kv_heads=block1.config.n_kv_heads,
    )
    print(f"running layer1 decode with {args.backend}-produced prefill cache", flush=True)
    decode1_tensors, _ = _run_artifact(
        decode1_artifact,
        backend=args.backend,
        inputs=cache1_inputs,
        program_name="cv32e40p_tinystories_chain_l1_decode",
        maxcycles=args.maxcycles_decode,
        verilator_max_ticks=args.verilator_max_ticks,
        timeout_s=args.timeout_s,
    )
    print(
        f"layer1.decode.out checksum={float(decode1_tensors['out'].sum()):.6f} "
        f"reference={float(np.asarray(decode1_ref['out'], dtype=np.float32).sum()):.6f}",
        flush=True,
    )

    final_hidden = _rmsnorm_np(np.asarray(decode1_tensors["out"], dtype=np.float32), final_norm_w, model.cfg.rms_norm_eps)
    logits = final_hidden @ lm_head_w.T
    probs = torch.softmax(torch.from_numpy(logits[0]), dim=-1)
    top = torch.topk(probs, k=10)
    next_id = int(top.indices[0].item())
    print(f"{args.backend}_chain_next_id={next_id}", flush=True)
    print(f"{args.backend}_chain_next_token={tokenizer.itos[next_id]!r}", flush=True)
    print("top10:", flush=True)
    for prob, idx in zip(top.values.tolist(), top.indices.tolist()):
        print(f"  {int(idx):3d} {tokenizer.itos[int(idx)]!r:>12} p={prob:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
