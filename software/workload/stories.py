from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


TINYSTORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"


@dataclass(frozen=True)
class ToyLMConfig:
    vocab_size: int = 96
    max_seq_len: int = 128
    n_layers: int = 2
    d_model: int = 128
    d_head: int = 16
    n_heads: int = 8
    n_kv_heads: int = 2
    ffn_hidden_dim: int = 384
    rope_theta: float = 500000.0
    rms_norm_eps: float = 1.0e-5


class CharTokenizer:
    def __init__(self, chars: str, *, vocab_size: int) -> None:
        if len(set(chars)) != len(chars):
            raise ValueError("chars must not contain duplicates")
        specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
        if len(specials) + len(chars) > vocab_size:
            raise ValueError(
                f"character set needs {
                    len(specials) + len(chars)} tokens, vocab_size={vocab_size}"
            )
        self.specials = specials
        self.chars = chars
        self.itos = (
            specials
            + list(chars)
            + [f"<unused_{i}>" for i in range(vocab_size - len(specials) - len(chars))]
        )
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    def encode(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(ch, self.unk_id) for ch in text.lower())
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        for idx in ids:
            idx = int(idx)
            if idx == self.eos_id:
                break
            token = self.itos[idx]
            if token in self.specials or token.startswith("<unused_"):
                continue
            out.append(token)
        return "".join(out)

    def to_json(self) -> dict[str, object]:
        return {"kind": "char", "itos": self.itos, "chars": self.chars}


class WordTokenizer:
    def __init__(self, words: list[str], *, vocab_size: int) -> None:
        if len(set(words)) != len(words):
            raise ValueError("words must not contain duplicates")
        specials = ["<pad>", "<bos>", "<eos>", "<unk>"]
        if len(specials) + len(words) > vocab_size:
            raise ValueError(
                f"word list needs {len(specials) + len(words)} tokens, vocab_size={vocab_size}"
            )
        self.specials = specials
        self.words = words
        self.itos = (
            specials
            + words
            + [f"<unused_{i}>" for i in range(vocab_size - len(specials) - len(words))]
        )
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}
        self.pad_id = self.stoi["<pad>"]
        self.bos_id = self.stoi["<bos>"]
        self.eos_id = self.stoi["<eos>"]
        self.unk_id = self.stoi["<unk>"]

    def encode(
        self, text: str, *, add_bos: bool = False, add_eos: bool = False
    ) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_id)
        ids.extend(self.stoi.get(tok, self.unk_id) for tok in word_tokens(text))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        no_space_before = {".", ",", "!", "?", ";", ":", "'", ")"}
        no_space_after = {"("}
        for idx in ids:
            idx = int(idx)
            if idx == self.eos_id:
                break
            token = self.itos[idx]
            if token in self.specials or token.startswith("<unused_"):
                continue
            if not out or token in no_space_before or out[-1] in no_space_after:
                out.append(token)
            else:
                out.append(" " + token)
        return "".join(out)

    def to_json(self) -> dict[str, object]:
        return {"kind": "word", "itos": self.itos, "words": self.words}


def tokenizer_from_json(data: dict[str, object]) -> CharTokenizer | WordTokenizer:
    kind = str(data.get("kind", "char"))
    itos_obj = data.get("itos")
    if not isinstance(itos_obj, list):
        raise ValueError("checkpoint tokenizer is missing itos")
    vocab_size = len(itos_obj)
    if kind == "char":
        chars = str(data.get("chars", ""))
        return CharTokenizer(chars, vocab_size=vocab_size)
    if kind == "word":
        words_obj = data.get("words")
        if not isinstance(words_obj, list):
            raise ValueError("word tokenizer checkpoint is missing words")
        return WordTokenizer([str(w) for w in words_obj], vocab_size=vocab_size)
    raise ValueError(f"unknown tokenizer kind in checkpoint: {kind}")


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


def _rope_frequencies(
    seq_len: int,
    d_head: int,
    theta: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    half = d_head // 2
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    dim = torch.arange(half, device=device, dtype=torch.float32)
    inv_freq = theta ** (-dim / float(half))
    angles = pos[:, None] * inv_freq[None, :]
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x0 = x[..., :half]
    x1 = x[..., half:]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return torch.cat((x0 * c - x1 * s, x1 * c + x0 * s), dim=-1)


@dataclass(frozen=True)
class TinyQATConfig:
    policy: str = "none"
    enabled: bool = False
    weight_bits: int = 16
    activation_bits: int = 16
    score_bits: int = 16
    prob_bits: int = 16
    value_bits: int = 16


def qat_config_for_policy(policy: str) -> TinyQATConfig:
    if policy == "none":
        return TinyQATConfig(policy=policy, enabled=False)
    if policy == "int16_baseline":
        return TinyQATConfig(policy=policy, enabled=True)
    if policy == "int8_scores":
        return TinyQATConfig(
            policy=policy,
            enabled=True,
            score_bits=8,
            prob_bits=8,
        )
    if policy == "int8_value":
        return TinyQATConfig(
            policy=policy,
            enabled=True,
            score_bits=8,
            prob_bits=8,
            value_bits=8,
        )
    raise ValueError(f"unknown QAT policy: {policy}")


def _signed_qmax(bits: int) -> int:
    if bits <= 1:
        raise ValueError("quantized tensors need at least two bits")
    return (1 << (bits - 1)) - 1


def _fake_quant_signed(
    x: torch.Tensor,
    bits: int,
    *,
    scale: float | torch.Tensor | None = None,
) -> torch.Tensor:
    qmax = _signed_qmax(bits)
    if scale is None:
        scale_t = x.detach().abs().amax().clamp_min(1.0e-8) / float(qmax)
    elif isinstance(scale, torch.Tensor):
        scale_t = scale.detach().to(device=x.device, dtype=x.dtype).clamp_min(1.0e-8)
    else:
        scale_t = torch.tensor(max(float(scale), 1.0e-8), device=x.device, dtype=x.dtype)
    q = torch.clamp(torch.round(x / scale_t), -qmax, qmax)
    deq = q * scale_t
    return x + (deq - x).detach()


def _fake_quant_prob(x: torch.Tensor, bits: int) -> torch.Tensor:
    # Hardware probabilities are carried in signed integer lanes even though
    # softmax itself is non-negative. Keep the same qmax convention.
    qmax = _signed_qmax(bits)
    return _fake_quant_signed(x, bits, scale=1.0 / float(qmax))


def _qat_linear(
    x: torch.Tensor,
    linear: nn.Linear,
    qat: TinyQATConfig,
    *,
    out_bits: int | None = None,
) -> torch.Tensor:
    if not qat.enabled:
        return linear(x)
    x_q = _fake_quant_signed(x, qat.activation_bits)
    w_q = _fake_quant_signed(linear.weight, qat.weight_bits)
    y = F.linear(x_q, w_q, linear.bias)
    return _fake_quant_signed(y, out_bits or qat.activation_bits)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ToyLMConfig) -> None:
        super().__init__()
        if cfg.d_model != cfg.n_heads * cfg.d_head:
            raise ValueError("d_model must equal n_heads * d_head")
        if cfg.n_heads % cfg.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if cfg.d_head % 2 != 0:
            raise ValueError("d_head must be even for RoPE")

        self.cfg = cfg
        self.q_proj = nn.Linear(
            cfg.d_model, cfg.n_heads * cfg.d_head, bias=False)
        self.k_proj = nn.Linear(
            cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.v_proj = nn.Linear(
            cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=False)
        self.o_proj = nn.Linear(cfg.n_heads * cfg.d_head,
                                cfg.d_model, bias=False)

        cos, sin = _rope_frequencies(
            cfg.max_seq_len,
            cfg.d_head,
            cfg.rope_theta,
            torch.device("cpu"),
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.qat = qat_config_for_policy("none")

    def set_qat_policy(self, policy: str) -> None:
        self.qat = qat_config_for_policy(policy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        cfg = self.cfg
        qat = self.qat

        q = _qat_linear(x, self.q_proj, qat).view(
            bsz, seq_len, cfg.n_heads, cfg.d_head
        )
        k = _qat_linear(x, self.k_proj, qat).view(
            bsz, seq_len, cfg.n_kv_heads, cfg.d_head
        )
        v = _qat_linear(x, self.v_proj, qat, out_bits=qat.value_bits).view(
            bsz, seq_len, cfg.n_kv_heads, cfg.d_head
        )

        cos = self.rope_cos[:seq_len]
        sin = self.rope_sin[:seq_len]

        q = apply_rope(q, cos, sin).transpose(1, 2)
        k = apply_rope(k, cos, sin).transpose(1, 2)
        v = v.transpose(1, 2)
        if qat.enabled:
            q = _fake_quant_signed(q, qat.activation_bits)
            k = _fake_quant_signed(k, qat.activation_bits)
            v = _fake_quant_signed(v, qat.value_bits)

        repeat = cfg.n_heads // cfg.n_kv_heads
        if repeat != 1:
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)

        if qat.enabled:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(cfg.d_head)
            scores = _fake_quant_signed(scores, qat.score_bits)
            mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
            scores = scores.masked_fill(~mask, float("-inf"))
            probs = F.softmax(scores, dim=-1)
            probs = _fake_quant_prob(probs, qat.prob_bits)
            out = torch.matmul(probs, v)
            out = _fake_quant_signed(out, qat.value_bits)
        else:
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(bsz, seq_len, cfg.n_heads * cfg.d_head)
        )
        return _qat_linear(out, self.o_proj, qat)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ToyLMConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden_dim, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.ffn_hidden_dim, bias=False)
        self.down_proj = nn.Linear(cfg.ffn_hidden_dim, cfg.d_model, bias=False)
        self.qat = qat_config_for_policy("none")

    def set_qat_policy(self, policy: str) -> None:
        self.qat = qat_config_for_policy(policy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.qat.enabled:
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        gate = _qat_linear(x, self.gate_proj, self.qat)
        up = _qat_linear(x, self.up_proj, self.qat)
        hidden = F.silu(gate) * up
        hidden = _fake_quant_signed(hidden, self.qat.activation_bits)
        return _qat_linear(hidden, self.down_proj, self.qat)


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ToyLMConfig) -> None:
        super().__init__()
        self.rms1 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.rms2 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def set_qat_policy(self, policy: str) -> None:
        self.attn.set_qat_policy(policy)
        self.mlp.set_qat_policy(policy)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.rms1(x))
        x = x + self.mlp(self.rms2(x))
        return x


class ToyDialogueLM(nn.Module):
    def __init__(self, cfg: ToyLMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_embeddings.weight
        self.qat = qat_config_for_policy("none")

    def set_qat_policy(self, policy: str) -> None:
        self.qat = qat_config_for_policy(policy)
        for layer in self.layers:
            layer.set_qat_policy(policy)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if input_ids.shape[1] > self.cfg.max_seq_len:
            raise ValueError(
                f"sequence length {input_ids.shape[1]} exceeds max_seq_len={
                    self.cfg.max_seq_len}"
            )

        x = self.tok_embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
        normed = self.norm(x)
        if self.qat.enabled:
            normed = _fake_quant_signed(normed, self.qat.activation_bits)
            w_q = _fake_quant_signed(self.lm_head.weight, self.qat.weight_bits)
            logits = F.linear(normed, w_q, self.lm_head.bias)
        else:
            logits = self.lm_head(normed)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_id: int,
        temperature: float = 0.8,
        top_k: int = 0,
    ) -> torch.Tensor:
        self.eval()
        out = input_ids
        for _ in range(max_new_tokens):
            ctx = out[:, -self.cfg.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :]

            if temperature <= 0.0:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    values, _ = torch.topk(
                        logits, k=min(top_k, logits.size(-1)), dim=-1
                    )
                    cutoff = values[:, [-1]]
                    logits = logits.masked_fill(logits < cutoff, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            out = torch.cat((out, next_id), dim=1)
            if int(next_id[0, 0].item()) == eos_id:
                break
        return out


def default_chars() -> str:
    # Keep this ASCII and hardware-friendly. Text normalization maps fancy quotes/dashes into this set.
    return "abcdefghijklmnopqrstuvwxyz0123456789 \n.,!?;:'\"-()/"


def word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[.,!?;:'\"()/-]", text.lower())


def build_word_tokenizer(text: str, *, vocab_size: int, top_words: int) -> WordTokenizer:
    special_count = 4
    if top_words <= 0:
        top_words = vocab_size - special_count
    top_words = min(top_words, vocab_size - special_count)
    if top_words <= 0:
        raise ValueError("vocab_size must leave room for at least one word token")
    counts = Counter(tok for tok in word_tokens(text) if re.match(r"[a-z0-9]+$", tok))
    # Keep punctuation even if it is not frequent; it makes decoded samples readable.
    punctuation = [".", ",", "!", "?", ";", ":", "'", "\"", "-", "(", ")", "/"]
    words: list[str] = []
    for tok in punctuation:
        if tok not in words and len(words) < top_words:
            words.append(tok)
    for tok, _ in counts.most_common():
        if tok not in words:
            words.append(tok)
        if len(words) >= top_words:
            break
    return WordTokenizer(words, vocab_size=vocab_size)


def normalize_text(text: str) -> str:
    replacements = {
        "\r": "\n",
        "\t": " ",
        "“": '"',
        "”": '"',
        "„": '"',
        "‘": "'",
        "’": "'",
        "`": "'",
        "—": "-",
        "–": "-",
        "…": "...",
        "\u00a0": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    text = text.lower()
    allowed = set(default_chars())
    out: list[str] = []
    prev_space = False

    for ch in text:
        if ch in allowed:
            if ch == " ":
                if not prev_space:
                    out.append(ch)
                prev_space = True
            elif ch == "\n":
                # Keep newlines, but avoid massive blank runs.
                if len(out) < 2 or not (out[-1] == "\n" and out[-2] == "\n"):
                    out.append(ch)
                prev_space = False
            else:
                out.append(ch)
                prev_space = False
        elif ch.isascii() and ch.isprintable():
            # Unknown ASCII punctuation becomes a space instead of <unk> spam.
            if not prev_space:
                out.append(" ")
                prev_space = True
        else:
            # Drop non-ASCII after normalization. This keeps the exported model simple.
            if not prev_space:
                out.append(" ")
                prev_space = True

    return "".join(out).strip() + "\n"


def download_first_mb(url: str, *, max_mb: int, cache_file: Path | None) -> str:
    max_bytes = max_mb * 1024 * 1024

    if (
        cache_file is not None
        and cache_file.exists()
        and cache_file.stat().st_size >= max_bytes
    ):
        with cache_file.open("r", encoding="utf-8", errors="replace") as f:
            return f.read(max_bytes)

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "tiny-npu-tinystories-trainer/1.0",
            "Range": f"bytes=0-{max_bytes - 1}",
        },
    )

    chunks: list[bytes] = []
    total = 0
    print(f"downloading up to {max_mb} MB from TinyStories...", flush=True)
    with urllib.request.urlopen(req, timeout=60) as resp:
        while total < max_bytes:
            chunk = resp.read(min(1024 * 1024, max_bytes - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            print(f"  downloaded {total / 1024 / 1024:.1f} MB", flush=True)

    raw = b"".join(chunks)
    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_bytes(raw)
    return raw.decode("utf-8", errors="replace")


def load_tinystories_text(args: argparse.Namespace) -> str:
    if args.data_file is not None:
        raw = args.data_file.read_text(encoding="utf-8", errors="replace")
    else:
        raw = download_first_mb(
            TINYSTORIES_URL, max_mb=args.max_data_mb, cache_file=args.cache_file
        )

    raw = raw.replace("<|endoftext|>", "\n\n")
    text = normalize_text(raw)

    if args.dialogue_wrap:
        # Cheap instruction-ish mode for later prompting with: user: tell me a story\nbot:
        blocks = [b.strip() for b in text.split("\n\n") if len(b.strip()) > 64]
        wrapped: list[str] = []
        prompts = [
            "tell me a story",
            "write a tiny story",
            "story please",
            "can you tell me a story",
        ]
        for i, block in enumerate(blocks):
            prompt = prompts[i % len(prompts)]
            # Keep each answer bounded so char-level training has repeated structure.
            story = block.replace("\n", " ")[
                : args.dialogue_story_chars].strip()
            if len(story) > 64:
                wrapped.append(f"user: {prompt}\nbot: {story}\n\n")
        text = normalize_text("".join(wrapped))

    return text


def make_lm_tensors(
    text: str,
    tokenizer: CharTokenizer,
    *,
    max_seq_len: int,
    stride: int,
    prompt_tokens: int,
    total_tokens: int | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids_list = tokenizer.encode(text, add_bos=True, add_eos=True)
    target_total_tokens = total_tokens if total_tokens is not None else max_seq_len + 1
    if target_total_tokens < 2:
        raise ValueError("total_tokens must leave at least one input and one target token")
    if target_total_tokens > max_seq_len + 1:
        raise ValueError("total_tokens cannot exceed max_seq_len + 1 for shifted LM training")
    if prompt_tokens < 0:
        raise ValueError("prompt_tokens must be non-negative")
    if len(ids_list) < target_total_tokens + 1:
        raise ValueError("not enough text for even one training chunk")

    ids = torch.tensor(ids_list, dtype=torch.long)
    n = 1 + (ids.numel() - target_total_tokens) // stride
    if n <= 0:
        raise ValueError(
            "not enough tokenized text for the requested max_seq_len")

    windows = torch.as_strided(
        ids,
        size=(n, target_total_tokens),
        stride=(stride, 1),
    ).contiguous()
    if target_total_tokens < max_seq_len + 1:
        pad = torch.full(
            (n, max_seq_len + 1 - target_total_tokens),
            tokenizer.pad_id,
            dtype=torch.long,
        )
        windows = torch.cat((windows, pad), dim=1)

    x = windows[:, :-1].contiguous().to(device)
    y = windows[:, 1:].contiguous().to(device)
    y[y == tokenizer.pad_id] = -100
    if prompt_tokens > 0:
        # y[i] predicts token i + 1. For a prompt of N observed tokens, train
        # only from y[N - 1] onward: first completion token and later.
        y[:, : max(prompt_tokens - 1, 0)] = -100
    return x, y


def split_train_valid(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    valid_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    n = x.size(0)
    n_valid = int(round(n * valid_fraction))
    n_valid = min(max(n_valid, 1), max(n - 1, 1)) if n > 1 else 0
    if n_valid == 0:
        return x, y, x, y
    return x[:-n_valid], y[:-n_valid], x[-n_valid:], y[-n_valid:]


def quantize_int16(array: np.ndarray) -> tuple[np.ndarray, float]:
    max_abs = float(np.max(np.abs(array))) if array.size else 0.0
    scale = max(max_abs / 32767.0, 1.0e-8)
    q = np.clip(np.rint(array / scale), -32768, 32767).astype(np.int16)
    return q, scale


def save_exports(
    model: ToyDialogueLM,
    tokenizer: CharTokenizer,
    cfg: ToyLMConfig,
    run_dir: Path,
    *,
    save_npz: bool,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(cfg),
            "tokenizer": tokenizer.to_json(),
        },
        run_dir / "tinystories_char_lm.pt",
    )

    meta: dict[str, object] = {
        "config": asdict(cfg),
        "tokenizer": tokenizer.to_json(),
    }

    if save_npz:
        fp32: dict[str, np.ndarray] = {}
        qpack: dict[str, np.ndarray] = {}
        scales: dict[str, float] = {}
        for name, tensor in model.state_dict().items():
            arr = tensor.detach().cpu().numpy().astype(np.float32)
            fp32[name] = arr
            if arr.ndim >= 2:
                q, scale = quantize_int16(arr)
                qpack[name] = q
                scales[name] = scale
        np.savez(run_dir / "tinystories_char_lm_fp32.npz", **fp32)
        np.savez(run_dir / "tinystories_char_lm_int16_weights.npz", **qpack)
        meta["weight_scales"] = scales

    (run_dir / "tinystories_char_lm_config.json").write_text(json.dumps(meta, indent=2))


def format_progress(step: int, total: int, *, loss: float, start_time: float) -> str:
    width = 28
    frac = min(max(float(step) / max(float(total), 1.0), 0.0), 1.0)
    filled = int(round(frac * width))
    bar = "#" * filled + "-" * (width - filled)
    elapsed = max(time.time() - start_time, 1.0e-6)
    steps_per_s = float(step) / elapsed if step > 0 else 0.0
    toks_per_s = steps_per_s * 1.0  # caller appends real value
    remaining = (float(total - step) /
                 steps_per_s) if steps_per_s > 0.0 else 0.0
    return (
        f"[{bar}] {step}/{total} {100.0 * frac:5.1f}% "
        f"loss={loss:.4f} step/s={steps_per_s:.2f} eta={remaining:.0f}s"
    )


@torch.inference_mode()
def estimate_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    eval_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(eval_batches):
        idx = torch.randint(x.size(0), (batch_size,), device=x.device)
        xb = x.index_select(0, idx)
        yb = y.index_select(0, idx)
        _, loss = model(xb, yb)
        assert loss is not None
        losses.append(float(loss.item()))
    return sum(losses) / max(len(losses), 1)


@torch.inference_mode()
def sample_response(
    model: ToyDialogueLM,
    tokenizer: CharTokenizer,
    prompt: str,
    *,
    device: torch.device,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> str:
    ids = tokenizer.encode(
        normalize_text(prompt).rstrip("\n"), add_bos=True, add_eos=False
    )
    x = torch.tensor([ids], dtype=torch.long, device=device)
    out = (
        model.generate(
            x,
            max_new_tokens=max_new_tokens,
            eos_id=tokenizer.eos_id,
            temperature=temperature,
            top_k=top_k,
        )[0]
        .detach()
        .cpu()
        .tolist()
    )
    return tokenizer.decode(out)


def train(args: argparse.Namespace) -> None:
    if args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)

    if args.seed >= 0:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    device = torch.device(args.device if args.device else "cpu")
    if device.type != "cpu":
        print(
            f"warning: requested device={
                device}; on Asahi M1, PyTorch GPU training is usually unavailable",
            flush=True,
        )

    init_ckpt: dict[str, object] | None = None
    if args.init_from is not None:
        init_ckpt = torch.load(args.init_from, map_location=device)
        if not isinstance(init_ckpt, dict) or "config" not in init_ckpt:
            raise ValueError(f"invalid checkpoint: {args.init_from}")

    if init_ckpt is not None:
        cfg = ToyLMConfig(**init_ckpt["config"])  # type: ignore[arg-type]
    else:
        cfg = ToyLMConfig(
            vocab_size=args.vocab_size,
            max_seq_len=args.max_seq_len,
            n_layers=args.n_layers,
            d_model=args.d_model,
            d_head=args.d_head,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            ffn_hidden_dim=args.ffn_hidden_dim,
        )

    text = load_tinystories_text(args)
    if init_ckpt is not None:
        tokenizer = tokenizer_from_json(init_ckpt["tokenizer"])  # type: ignore[arg-type]
        args.tokenizer = str(init_ckpt["tokenizer"].get("kind", args.tokenizer))  # type: ignore[union-attr]
    elif args.tokenizer == "word":
        tokenizer = build_word_tokenizer(text, vocab_size=cfg.vocab_size, top_words=args.top_words)
    else:
        tokenizer = CharTokenizer(default_chars(), vocab_size=cfg.vocab_size)
    stride = args.stride if args.stride > 0 else args.max_seq_len
    all_x, all_y = make_lm_tensors(
        text,
        tokenizer,
        max_seq_len=cfg.max_seq_len,
        stride=stride,
        prompt_tokens=args.prompt_tokens,
        total_tokens=args.total_tokens,
        device=device,
    )
    train_x, train_y, valid_x, valid_y = split_train_valid(
        all_x,
        all_y,
        valid_fraction=args.valid_fraction,
    )

    raw_model = ToyDialogueLM(cfg).to(device)
    if init_ckpt is not None:
        raw_model.load_state_dict(init_ckpt["model"])  # type: ignore[arg-type]
        print(f"loaded checkpoint={args.init_from}", flush=True)
    raw_model.set_qat_policy(args.qat_policy)
    model: nn.Module = raw_model
    if args.compile:
        print("compiling model with torch.compile...", flush=True)
        model = torch.compile(raw_model, mode="default")

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        foreach=True,
    )

    n_params = sum(p.numel() for p in raw_model.parameters())
    n_train_tokens = train_x.numel()
    print(
        f"device={device} threads={torch.get_num_threads()} compile={
            args.compile} "
        f"tokenizer={args.tokenizer} qat_policy={args.qat_policy} params={n_params:,}",
        flush=True,
    )
    print(
        f"chars={len(text):,} chunks={all_x.size(
            0):,} train_chunks={train_x.size(0):,} "
        f"valid_chunks={valid_x.size(0):,} seq={
            cfg.max_seq_len} stride={stride} "
        f"prompt_tokens={args.prompt_tokens} total_tokens={args.total_tokens or (cfg.max_seq_len + 1)} "
        f"train_tokens={n_train_tokens:,}",
        flush=True,
    )
    print(
        f"steps={args.steps} batch={args.batch_size} lr={args.lr} "
        f"log_every={args.log_every} sample_every={
            args.sample_every} eval_every={args.eval_every}",
        flush=True,
    )

    # One optional warmup forward/backward. This keeps first logged speed less misleading.
    if args.warmup:
        idx = torch.randint(train_x.size(0), (args.batch_size,), device=device)
        xb = train_x.index_select(0, idx)
        yb = train_y.index_select(0, idx)
        _, loss = model(xb, yb)
        assert loss is not None
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    start_time = time.time()
    last_time = start_time
    last_step = 0

    for step in range(1, args.steps + 1):
        idx = torch.randint(train_x.size(0), (args.batch_size,), device=device)
        xb = train_x.index_select(0, idx)
        yb = train_y.index_select(0, idx)

        model.train()
        _, loss = model(xb, yb)
        assert loss is not None

        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(
                raw_model.parameters(), args.grad_clip)

        optimizer.step()

        should_log = step == 1 or step % args.log_every == 0 or step == args.steps
        if should_log:
            now = time.time()
            interval_steps = step - last_step
            interval_time = max(now - last_time, 1.0e-9)
            interval_steps_per_s = interval_steps / interval_time
            interval_tokens_per_s = (
                interval_steps_per_s * args.batch_size * cfg.max_seq_len
            )
            last_time = now
            last_step = step

            print(
                format_progress(
                    step, args.steps, loss=float(loss.item()), start_time=start_time
                ),
                flush=True,
            )
            print(
                f"  recent_tokens/s={interval_tokens_per_s:,.0f}", flush=True)

        if args.eval_every > 0 and (step % args.eval_every == 0 or step == args.steps):
            val_loss = estimate_loss(
                model,
                valid_x,
                valid_y,
                batch_size=min(args.batch_size, valid_x.size(0)),
                eval_batches=args.eval_batches,
            )
            print(f"  valid_loss={val_loss:.4f}", flush=True)

        if args.sample_every > 0 and (
            step % args.sample_every == 0 or step == args.steps
        ):
            sample = sample_response(
                raw_model,
                tokenizer,
                args.prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            print("--- sample ---", flush=True)
            print(sample, flush=True)
            print("--------------", flush=True)

    save_exports(
        raw_model,
        tokenizer,
        cfg,
        args.out_dir,
        save_npz=not args.no_npz,
    )
    print(f"saved={args.out_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fast CPU char-level TinyStories trainer for small Llama-style toy LMs."
    )

    # Data.
    parser.add_argument(
        "--data-file",
        type=Path,
        default=None,
        help="Optional local TinyStories text file.",
    )
    parser.add_argument(
        "--cache-file", type=Path, default=Path("data/tinystories_head.txt")
    )
    parser.add_argument(
        "--max-data-mb",
        type=int,
        default=4,
        help="How many MB to stream when --data-file is absent.",
    )
    parser.add_argument(
        "--dialogue-wrap",
        action="store_true",
        help="Wrap stories as user:/bot: examples.",
    )
    parser.add_argument("--dialogue-story-chars", type=int, default=300)
    parser.add_argument(
        "--tokenizer",
        choices=("char", "word"),
        default="char",
        help="Use character tokens or a fixed top-word vocabulary.",
    )
    parser.add_argument(
        "--top-words",
        type=int,
        default=124,
        help="Number of non-special word tokens to keep when --tokenizer=word. With vocab_size=128, use 124.",
    )

    # Runtime/perf.
    parser.add_argument(
        "--out-dir", type=Path, default=Path("runs/tinystories_char_lm")
    )
    parser.add_argument(
        "--init-from",
        type=Path,
        default=None,
        help="Optional tinystories_char_lm.pt checkpoint to fine-tune.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Call torch.set_num_threads(N). Try 1, 2, 4 on M1.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Try torch.compile. Benchmark; it may or may not help on CPU.",
    )
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument(
        "--no-npz",
        action="store_true",
        help="Skip fp32/int16 NPZ export for faster save.",
    )

    # Training.
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=0.0)
    parser.add_argument("--valid-fraction", type=float, default=0.02)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--sample-every", type=int, default=500)
    parser.add_argument(
        "--qat-policy",
        choices=("none", "int16_baseline", "int8_scores", "int8_value"),
        default="none",
        help="Enable fake-quant training for the NPU-facing QLlama boundaries.",
    )

    # Sampling.
    parser.add_argument("--prompt", default="once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)

    # Model.
    parser.add_argument("--vocab-size", type=int, default=96)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="Ignore loss before this many observed tokens; 5 trains completions after a 5-token prompt.",
    )
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=None,
        help="Total tokenized sequence length including prompt and completion. Use 17 for 5 prompt + 12 completion tokens.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=0,
        help="0 means non-overlapping chunks of max_seq_len.",
    )
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-head", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--ffn-hidden-dim", type=int, default=384)

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
