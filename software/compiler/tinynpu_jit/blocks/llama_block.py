from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tinynpu_jit import (
    DType,
    IRBuilder,
    TensorKind,
    TensorSpec,
    compile_plan,
    make_native_int16_kv_cache_specs,
)
from tinynpu_jit.golden import GoldenModel
from tinynpu_jit.runtime_approx import (
    quantize_fp16_to_i16_xform,
    rmsnorm_approx,
    silu_approx,
    softmax_f16_approx,
)


def _rand_i16(rng: np.random.Generator, shape: tuple[int, ...], low: int = -2, high: int = 3) -> np.ndarray:
    return rng.integers(low, high, size=shape, endpoint=False, dtype=np.int16)


def _rand_f32(rng: np.random.Generator, shape: tuple[int, ...], low: float = -0.25, high: float = 0.25) -> np.ndarray:
    return rng.uniform(low, high, size=shape).astype(np.float32)


def _quantize_weight_fp32(weight: np.ndarray) -> np.ndarray:
    weight_fp = np.asarray(weight, dtype=np.float32)
    return np.clip(np.rint(weight_fp), -32768, 32767).astype(np.int16)


def _rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return rmsnorm_approx(x, weight, eps)


def _silu(x: np.ndarray) -> np.ndarray:
    return silu_approx(x)


def _quantize_fp16_boundary(source: np.ndarray, *, scale: float) -> np.ndarray:
    return quantize_fp16_to_i16_xform(source, scale=scale)


def _rope_ref(x: np.ndarray, *, positions: np.ndarray, head_dim: int, theta: float) -> np.ndarray:
    """Apply HF-style split-halves RoPE.

    Raw Meta Llama checkpoints use interleaved Q/K rotary pairs, so callers
    importing those weights need to permute them into this layout first.
    """
    x_f = np.asarray(x, dtype=np.float32)
    if x_f.shape[-1] != head_dim:
        raise ValueError(f"rope_ref expects last dimension {head_dim}, got {x_f.shape[-1]}")
    if head_dim % 2 != 0:
        raise ValueError(f"rope_ref expects even head_dim, got {head_dim}")
    pos = np.asarray(positions, dtype=np.float32).reshape(-1, 1)
    half = head_dim // 2
    inv_freq = np.float32(1.0) / (np.float32(theta) ** (np.arange(0, half, dtype=np.float32) / np.float32(half)))
    angles = pos * inv_freq.reshape(1, -1)
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    out = np.array(x_f, dtype=np.float32, copy=True)
    first = x_f[..., :half]
    second = x_f[..., half:head_dim]
    out[..., :half] = first * cos - second * sin
    out[..., half:head_dim] = second * cos + first * sin
    return out


def _score_scale(d_head: int, shape: tuple[int, int], *, act_scale: float) -> np.ndarray:
    score_scale = np.float32((float(act_scale) * float(act_scale)) / np.sqrt(float(d_head)))
    return np.full(shape, score_scale, dtype=np.float32)


def _kv_head_for_q_head(q_head: int, *, n_heads: int, n_kv_heads: int) -> int:
    if n_heads % n_kv_heads != 0:
        raise ValueError(f"n_heads={n_heads} must be divisible by n_kv_heads={n_kv_heads}")
    return q_head // (n_heads // n_kv_heads)


@dataclass(frozen=True)
class QLlamaBlockConfig:
    """Quantized LLaMA block shape and scaling."""

    d_model: int
    d_head: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden_dim: int
    act_scale: float = 1.0 / 32.0
    attn_scale: float = 1.0 / 256.0
    rms_norm_eps: float = 1.0e-5
    rope_theta: float = 500000.0
    rope_scaling: dict[str, object] | None = None

    @property
    def attn_dim(self) -> int:
        return self.n_heads * self.d_head

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.d_head


@dataclass(frozen=True)
class QLlamaBlock:
    """Quantized LLaMA block weights with HF-style naming and RoPE layout.

    This block expects the Hugging Face split-halves rotary convention. Raw
    Meta Llama checkpoints need their rotary Q/K weights permuted on import.
    """

    config: QLlamaBlockConfig
    input_layernorm_w: np.ndarray
    self_attn_q_proj_w: np.ndarray
    self_attn_k_proj_w: np.ndarray
    self_attn_v_proj_w: np.ndarray
    self_attn_o_proj_w: np.ndarray
    post_attention_layernorm_w: np.ndarray
    mlp_gate_proj_w: np.ndarray
    mlp_up_proj_w: np.ndarray
    mlp_down_proj_w: np.ndarray

    @classmethod
    def random(cls, rng: np.random.Generator, config: QLlamaBlockConfig) -> QLlamaBlock:
        """Generate synthetic low-magnitude weights for plumbing tests only."""
        return cls(
            config=config,
            input_layernorm_w=rng.uniform(0.5, 1.5, size=(config.d_model,)).astype(np.float32),
            self_attn_q_proj_w=_rand_i16(rng, (config.d_model, config.attn_dim)),
            self_attn_k_proj_w=_rand_i16(rng, (config.d_model, config.kv_dim)),
            self_attn_v_proj_w=_rand_i16(rng, (config.d_model, config.kv_dim)),
            self_attn_o_proj_w=_rand_i16(rng, (config.attn_dim, config.d_model)),
            post_attention_layernorm_w=rng.uniform(0.5, 1.5, size=(config.d_model,)).astype(np.float32),
            mlp_gate_proj_w=_rand_i16(rng, (config.d_model, config.ffn_hidden_dim)),
            mlp_up_proj_w=_rand_i16(rng, (config.d_model, config.ffn_hidden_dim)),
            mlp_down_proj_w=_rand_i16(rng, (config.ffn_hidden_dim, config.d_model)),
        )

    @classmethod
    def from_fp32(
        cls,
        *,
        config: QLlamaBlockConfig,
        input_layernorm_w: np.ndarray,
        self_attn_q_proj_w: np.ndarray,
        self_attn_k_proj_w: np.ndarray,
        self_attn_v_proj_w: np.ndarray,
        self_attn_o_proj_w: np.ndarray,
        post_attention_layernorm_w: np.ndarray,
        mlp_gate_proj_w: np.ndarray,
        mlp_up_proj_w: np.ndarray,
        mlp_down_proj_w: np.ndarray,
    ) -> QLlamaBlock:
        """Build an HF-shaped block by naively rounding float weights to INT16."""
        return cls(
            config=config,
            input_layernorm_w=np.asarray(input_layernorm_w, dtype=np.float32),
            self_attn_q_proj_w=_quantize_weight_fp32(self_attn_q_proj_w),
            self_attn_k_proj_w=_quantize_weight_fp32(self_attn_k_proj_w),
            self_attn_v_proj_w=_quantize_weight_fp32(self_attn_v_proj_w),
            self_attn_o_proj_w=_quantize_weight_fp32(self_attn_o_proj_w),
            post_attention_layernorm_w=np.asarray(post_attention_layernorm_w, dtype=np.float32),
            mlp_gate_proj_w=_quantize_weight_fp32(mlp_gate_proj_w),
            mlp_up_proj_w=_quantize_weight_fp32(mlp_up_proj_w),
            mlp_down_proj_w=_quantize_weight_fp32(mlp_down_proj_w),
        )

    def split_q_proj_weights(self) -> list[np.ndarray]:
        w = np.asarray(self.self_attn_q_proj_w, dtype=np.int16)
        d_head = self.config.d_head
        return [np.array(w[:, i * d_head : (i + 1) * d_head], copy=True) for i in range(self.config.n_heads)]

    def split_k_proj_weights(self) -> list[np.ndarray]:
        w = np.asarray(self.self_attn_k_proj_w, dtype=np.int16)
        d_head = self.config.d_head
        return [np.array(w[:, i * d_head : (i + 1) * d_head], copy=True) for i in range(self.config.n_kv_heads)]

    def split_v_proj_weights(self) -> list[np.ndarray]:
        w = np.asarray(self.self_attn_v_proj_w, dtype=np.int16)
        d_head = self.config.d_head
        return [np.array(w[:, i * d_head : (i + 1) * d_head], copy=True) for i in range(self.config.n_kv_heads)]


def _require_supported_rope_scaling(rope_scaling: dict[str, object] | None) -> None:
    if rope_scaling is not None:
        raise NotImplementedError("QLlamaBlock rope_scaling is not implemented yet.")


def _resolve_rope_theta(block: QLlamaBlock, rope_theta: float | None) -> float:
    return float(block.config.rope_theta if rope_theta is None else rope_theta)


def _resolve_rope_scaling(block: QLlamaBlock, rope_scaling: dict[str, object] | None) -> dict[str, object] | None:
    resolved = block.config.rope_scaling if rope_scaling is None else rope_scaling
    _require_supported_rope_scaling(resolved)
    return resolved


def build_shared_state(
    *,
    d_model: int,
    d_head: int,
    n_heads: int,
    n_kv_heads: int,
    ffn_hidden_dim: int,
    prompt_len: int,
    seed: int,
    act_scale: float = 1.0 / 32.0,
    attn_scale: float = 1.0 / 256.0,
    rms_norm_eps: float = 1.0e-5,
    rope_theta: float = 500000.0,
    rope_scaling: dict[str, object] | None = None,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    config = QLlamaBlockConfig(
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        ffn_hidden_dim=ffn_hidden_dim,
        act_scale=act_scale,
        attn_scale=attn_scale,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
    )
    block = QLlamaBlock.random(rng, config)
    return {
        "config": config,
        "block": block,
        "x_prompt_in": _rand_f32(rng, (prompt_len, d_model)),
        "x_decode_in": _rand_f32(rng, (1, d_model)),
    }


def reference_prefill(
    state: dict[str, object],
    *,
    d_head: int,
    n_heads: int,
    n_kv_heads: int,
    act_scale: float,
    attn_scale: float,
    rope_theta: float | None = None,
    rope_scaling: dict[str, object] | None = None,
    x_in: np.ndarray | None = None,
) -> dict[str, object]:
    """Hardware-faithful quantized prefill reference for the compiled block."""
    golden = GoldenModel()
    block: QLlamaBlock = state["block"]
    resolved_rope_theta = _resolve_rope_theta(block, rope_theta)
    _resolve_rope_scaling(block, rope_scaling)
    if x_in is None:
        x_in = np.array(state["x_prompt_in"], dtype=np.float32, copy=True)
    else:
        x_in = np.array(x_in, dtype=np.float32, copy=True)

    prompt_len = x_in.shape[0]
    q_weights = block.split_q_proj_weights()
    k_weights = block.split_k_proj_weights()
    v_weights = block.split_v_proj_weights()
    positions = np.arange(prompt_len, dtype=np.int32)

    x_norm1 = _rmsnorm(x_in, block.input_layernorm_w, block.config.rms_norm_eps)
    x_norm1_q = golden.quantize(x_norm1, scale=act_scale, zero_point=0, out_dtype=DType.INT16)

    q_rope_heads: list[np.ndarray] = []
    k_rope_heads: list[np.ndarray] = []
    v_heads: list[np.ndarray] = []
    attn_heads: list[np.ndarray] = []
    score_scale = np.float32((float(act_scale) * float(act_scale)) / np.sqrt(float(d_head)))
    for q_head in range(n_heads):
        q_int = golden.matmul(x_norm1_q, q_weights[q_head], out_dtype=DType.INT16)
        q_f = golden.dequantize(q_int, scale=act_scale, zero_point=0)
        q_rope_f = _rope_ref(q_f, positions=positions, head_dim=d_head, theta=resolved_rope_theta)
        q_rope_q = golden.quantize(q_rope_f, scale=act_scale, zero_point=0, out_dtype=DType.INT16)
        q_rope_heads.append(q_rope_q)

    for kv_head in range(n_kv_heads):
        k_int = golden.matmul(x_norm1_q, k_weights[kv_head], out_dtype=DType.INT16)
        v_int = golden.matmul(x_norm1_q, v_weights[kv_head], out_dtype=DType.INT16)
        k_f = golden.dequantize(k_int, scale=act_scale, zero_point=0)
        k_rope_f = _rope_ref(k_f, positions=positions, head_dim=d_head, theta=resolved_rope_theta)
        k_rope_q = golden.quantize(k_rope_f, scale=act_scale, zero_point=0, out_dtype=DType.INT16)
        k_rope_heads.append(k_rope_q)
        # V stays in projected INT16 space because it is not rotary-transformed.
        v_heads.append(v_int)

    for q_head in range(n_heads):
        kv_head = _kv_head_for_q_head(q_head, n_heads=n_heads, n_kv_heads=n_kv_heads)
        scores = golden.matmul(
            q_rope_heads[q_head],
            np.array(k_rope_heads[kv_head].T, dtype=np.int16, copy=True),
            out_dtype=DType.INT16,
        ).astype(np.float32)
        scores = (scores * score_scale).astype(np.float32)
        masked = np.array(scores, copy=True)
        for row in range(prompt_len):
            if row + 1 < prompt_len:
                masked[row, row + 1 :] = np.float32(-1.0e10)
        probs = softmax_f16_approx(masked, axis=-1)
        probs_q = _quantize_fp16_boundary(probs, scale=attn_scale)
        attn = golden.matmul(probs_q, v_heads[kv_head], shift=8, out_dtype=DType.INT16)
        attn_heads.append(attn)

    attn_cat = np.concatenate(attn_heads, axis=-1).astype(np.int16) if n_heads > 1 else np.array(attn_heads[0], copy=True)
    o_int = golden.matmul(attn_cat, np.asarray(block.self_attn_o_proj_w, dtype=np.int16), out_dtype=DType.INT16)
    o_f = golden.dequantize(o_int, scale=act_scale, zero_point=0)
    resid1 = (x_in + o_f).astype(np.float32)
    x_norm2 = _rmsnorm(resid1, block.post_attention_layernorm_w, block.config.rms_norm_eps)
    x_norm2_q = golden.quantize(x_norm2, scale=act_scale, zero_point=0, out_dtype=DType.INT16)
    gate_int = golden.matmul(x_norm2_q, np.asarray(block.mlp_gate_proj_w, dtype=np.int16), out_dtype=DType.INT16)
    up_int = golden.matmul(x_norm2_q, np.asarray(block.mlp_up_proj_w, dtype=np.int16), out_dtype=DType.INT16)
    gate_f = golden.dequantize(gate_int, scale=act_scale, zero_point=0)
    up_f = golden.dequantize(up_int, scale=act_scale, zero_point=0)
    gate_act = _silu(gate_f)
    ffn_hidden = (gate_act * up_f).astype(np.float32)
    ffn_hidden_q = golden.quantize(ffn_hidden, scale=act_scale, zero_point=0, out_dtype=DType.INT16)
    ffn_out_int = golden.matmul(ffn_hidden_q, np.asarray(block.mlp_down_proj_w, dtype=np.int16), out_dtype=DType.INT16)
    ffn_out_f = golden.dequantize(ffn_out_int, scale=act_scale, zero_point=0)
    out = (resid1 + ffn_out_f).astype(np.float32)
    return {
        "x_norm1": x_norm1,
        "x_norm1_q": x_norm1_q,
        "q_rope_heads": q_rope_heads,
        "k_heads": k_rope_heads,
        "v_heads": v_heads,
        "attn_cat": attn_cat,
        "o_int": o_int,
        "o_f": o_f,
        "resid1": resid1,
        "x_norm2": x_norm2,
        "x_norm2_q": x_norm2_q,
        "gate_int": gate_int,
        "up_int": up_int,
        "ffn_hidden_q": ffn_hidden_q,
        "ffn_out_int": ffn_out_int,
        "ffn_out_f": ffn_out_f,
        "out": out,
    }


def reference_decode(
    state: dict[str, object],
    prefill_ref: dict[str, object],
    *,
    d_head: int,
    n_heads: int,
    n_kv_heads: int,
    act_scale: float,
    attn_scale: float,
    rope_theta: float | None = None,
    rope_scaling: dict[str, object] | None = None,
    x_in: np.ndarray | None = None,
) -> dict[str, object]:
    """Hardware-faithful quantized decode reference for the compiled block."""
    golden = GoldenModel()
    block: QLlamaBlock = state["block"]
    resolved_rope_theta = _resolve_rope_theta(block, rope_theta)
    _resolve_rope_scaling(block, rope_scaling)
    if x_in is None:
        x_in = np.array(state["x_decode_in"], dtype=np.float32, copy=True)
    else:
        x_in = np.array(x_in, dtype=np.float32, copy=True)

    prompt_len = np.asarray(prefill_ref["k_heads"][0]).shape[0]
    decode_pos = np.array([prompt_len], dtype=np.int32)
    q_weights = block.split_q_proj_weights()
    k_weights = block.split_k_proj_weights()
    v_weights = block.split_v_proj_weights()

    x_norm1 = _rmsnorm(x_in, block.input_layernorm_w, block.config.rms_norm_eps)
    x_norm1_q = golden.quantize(x_norm1, scale=act_scale, zero_point=0, out_dtype=DType.INT16)

    q_rope_heads: list[np.ndarray] = []
    k_cur_heads: list[np.ndarray] = []
    v_cur_heads: list[np.ndarray] = []
    attn_heads: list[np.ndarray] = []
    score_scale = np.float32((float(act_scale) * float(act_scale)) / np.sqrt(float(d_head)))
    for q_head in range(n_heads):
        q_int = golden.matmul(x_norm1_q, q_weights[q_head], out_dtype=DType.INT16)
        q_f = golden.dequantize(q_int, scale=act_scale, zero_point=0)
        q_rope_f = _rope_ref(q_f, positions=decode_pos, head_dim=d_head, theta=resolved_rope_theta)
        q_rope_q = golden.quantize(q_rope_f, scale=act_scale, zero_point=0, out_dtype=DType.INT16)
        q_rope_heads.append(q_rope_q)

    for kv_head in range(n_kv_heads):
        k_int = golden.matmul(x_norm1_q, k_weights[kv_head], out_dtype=DType.INT16)
        v_int = golden.matmul(x_norm1_q, v_weights[kv_head], out_dtype=DType.INT16)
        k_f = golden.dequantize(k_int, scale=act_scale, zero_point=0)
        k_rope_f = _rope_ref(k_f, positions=decode_pos, head_dim=d_head, theta=resolved_rope_theta)
        k_rope_q = golden.quantize(k_rope_f, scale=act_scale, zero_point=0, out_dtype=DType.INT16)
        k_cur_heads.append(k_rope_q)
        # V stays in projected INT16 space because it is not rotary-transformed.
        v_cur_heads.append(v_int)

    for q_head in range(n_heads):
        kv_head = _kv_head_for_q_head(q_head, n_heads=n_heads, n_kv_heads=n_kv_heads)
        k_full = np.concatenate(
            [
                np.asarray(prefill_ref["k_heads"][kv_head], dtype=np.int16),
                np.asarray(k_cur_heads[kv_head], dtype=np.int16),
            ],
            axis=0,
        ).astype(np.int16)
        v_full = np.concatenate(
            [
                np.asarray(prefill_ref["v_heads"][kv_head], dtype=np.int16),
                np.asarray(v_cur_heads[kv_head], dtype=np.int16),
            ],
            axis=0,
        ).astype(np.int16)
        scores = golden.matmul(
            q_rope_heads[q_head],
            np.array(k_full.T, dtype=np.int16, copy=True),
            out_dtype=DType.INT16,
        ).astype(np.float32)
        scores = (scores * score_scale).astype(np.float32)
        probs = softmax_f16_approx(scores, axis=-1)
        probs_q = _quantize_fp16_boundary(probs, scale=attn_scale)
        attn = golden.matmul(probs_q, v_full, shift=8, out_dtype=DType.INT16)
        attn_heads.append(attn)

    attn_cat = np.concatenate(attn_heads, axis=-1).astype(np.int16) if n_heads > 1 else np.array(attn_heads[0], copy=True)
    o_int = golden.matmul(attn_cat, np.asarray(block.self_attn_o_proj_w, dtype=np.int16), out_dtype=DType.INT16)
    o_f = golden.dequantize(o_int, scale=act_scale, zero_point=0)
    resid1 = (x_in + o_f).astype(np.float32)
    x_norm2 = _rmsnorm(resid1, block.post_attention_layernorm_w, block.config.rms_norm_eps)
    x_norm2_q = golden.quantize(x_norm2, scale=act_scale, zero_point=0, out_dtype=DType.INT16)
    gate_int = golden.matmul(x_norm2_q, np.asarray(block.mlp_gate_proj_w, dtype=np.int16), out_dtype=DType.INT16)
    up_int = golden.matmul(x_norm2_q, np.asarray(block.mlp_up_proj_w, dtype=np.int16), out_dtype=DType.INT16)
    gate_f = golden.dequantize(gate_int, scale=act_scale, zero_point=0)
    up_f = golden.dequantize(up_int, scale=act_scale, zero_point=0)
    gate_act = _silu(gate_f)
    ffn_hidden = (gate_act * up_f).astype(np.float32)
    ffn_hidden_q = golden.quantize(ffn_hidden, scale=act_scale, zero_point=0, out_dtype=DType.INT16)
    ffn_out_int = golden.matmul(ffn_hidden_q, np.asarray(block.mlp_down_proj_w, dtype=np.int16), out_dtype=DType.INT16)
    ffn_out_f = golden.dequantize(ffn_out_int, scale=act_scale, zero_point=0)
    out = (resid1 + ffn_out_f).astype(np.float32)
    return {
        "x_norm1": x_norm1,
        "x_norm1_q": x_norm1_q,
        "q_rope_heads": q_rope_heads,
        "k_cur_heads": k_cur_heads,
        "v_cur_heads": v_cur_heads,
        "attn_cat": attn_cat,
        "o_int": o_int,
        "o_f": o_f,
        "resid1": resid1,
        "x_norm2": x_norm2,
        "x_norm2_q": x_norm2_q,
        "gate_int": gate_int,
        "up_int": up_int,
        "ffn_hidden_q": ffn_hidden_q,
        "ffn_out_int": ffn_out_int,
        "ffn_out_f": ffn_out_f,
        "out": out,
    }


def extend_kv_cache(
    prefill_ref: dict[str, object],
    decode_ref: dict[str, object],
) -> dict[str, object]:
    return {
        "k_heads": [
            np.concatenate(
                [
                    np.asarray(prefill_ref["k_heads"][head_idx], dtype=np.int16),
                    np.asarray(decode_ref["k_cur_heads"][head_idx], dtype=np.int16),
                ],
                axis=0,
            ).astype(np.int16)
            for head_idx in range(len(prefill_ref["k_heads"]))
        ],
        "v_heads": [
            np.concatenate(
                [
                    np.asarray(prefill_ref["v_heads"][head_idx], dtype=np.int16),
                    np.asarray(decode_ref["v_cur_heads"][head_idx], dtype=np.int16),
                ],
                axis=0,
            ).astype(np.int16)
            for head_idx in range(len(prefill_ref["v_heads"]))
        ],
    }


def _common_io_tensors(
    *,
    block: QLlamaBlock,
    x_in: np.ndarray,
    d_model: int,
    ffn_hidden_dim: int,
    attn_dim: int,
    seq_len: int,
    act_scale: float,
    out_name: str = "out",
) -> dict[str, TensorSpec]:
    return {
        "x_in": TensorSpec("x_in", (seq_len, d_model), DType.FLOAT32, TensorKind.CONSTANT, data=np.array(x_in, dtype=np.float32, copy=True)),
        "rms1_w": TensorSpec("rms1_w", (d_model,), DType.FLOAT32, TensorKind.CONSTANT, data=np.array(block.input_layernorm_w, dtype=np.float32, copy=True)),
        "rms2_w": TensorSpec("rms2_w", (d_model,), DType.FLOAT32, TensorKind.CONSTANT, data=np.array(block.post_attention_layernorm_w, dtype=np.float32, copy=True)),
        "w_o": TensorSpec("w_o", (attn_dim, d_model), DType.INT16, TensorKind.CONSTANT, data=np.array(block.self_attn_o_proj_w, dtype=np.int16, copy=True)),
        "w_gate": TensorSpec("w_gate", (d_model, ffn_hidden_dim), DType.INT16, TensorKind.CONSTANT, data=np.array(block.mlp_gate_proj_w, dtype=np.int16, copy=True)),
        "w_up": TensorSpec("w_up", (d_model, ffn_hidden_dim), DType.INT16, TensorKind.CONSTANT, data=np.array(block.mlp_up_proj_w, dtype=np.int16, copy=True)),
        "w_down": TensorSpec("w_down", (ffn_hidden_dim, d_model), DType.INT16, TensorKind.CONSTANT, data=np.array(block.mlp_down_proj_w, dtype=np.int16, copy=True)),
        "x_norm1": TensorSpec("x_norm1", (seq_len, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "x_norm1_q": TensorSpec("x_norm1_q", (seq_len, d_model), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}),
        "attn_cat": TensorSpec("attn_cat", (seq_len, attn_dim), DType.INT16, TensorKind.INTERMEDIATE),
        "attn_cat_a": TensorSpec("attn_cat_a", (seq_len, attn_dim), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}),
        "o_int": TensorSpec("o_int", (seq_len, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        "o_f": TensorSpec("o_f", (seq_len, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "resid1": TensorSpec("resid1", (seq_len, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "x_norm2": TensorSpec("x_norm2", (seq_len, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "x_norm2_q": TensorSpec("x_norm2_q", (seq_len, d_model), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}),
        "gate_int": TensorSpec("gate_int", (seq_len, ffn_hidden_dim), DType.INT16, TensorKind.INTERMEDIATE),
        "up_int": TensorSpec("up_int", (seq_len, ffn_hidden_dim), DType.INT16, TensorKind.INTERMEDIATE),
        "gate_f": TensorSpec("gate_f", (seq_len, ffn_hidden_dim), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "up_f": TensorSpec("up_f", (seq_len, ffn_hidden_dim), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "gate_act": TensorSpec("gate_act", (seq_len, ffn_hidden_dim), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "ffn_hidden": TensorSpec("ffn_hidden", (seq_len, ffn_hidden_dim), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "ffn_hidden_q": TensorSpec("ffn_hidden_q", (seq_len, ffn_hidden_dim), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}),
        "ffn_out_int": TensorSpec("ffn_out_int", (seq_len, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        "ffn_out_f": TensorSpec("ffn_out_f", (seq_len, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        out_name: TensorSpec(out_name, (seq_len, d_model), DType.FLOAT32, TensorKind.OUTPUT, is_final_output=True),
    }


def _add_tensor_specs(builder: IRBuilder, specs: dict[str, TensorSpec]) -> None:
    for spec in specs.values():
        builder.add_tensor(spec)


def _add_projection_tensors(
    builder: IRBuilder,
    *,
    block: QLlamaBlock,
    d_model: int,
    d_head: int,
    n_heads: int,
    n_kv_heads: int,
) -> None:
    for q_head, w_q in enumerate(block.split_q_proj_weights()):
        builder.add_tensor(TensorSpec(f"w_q_h{q_head}", (d_model, d_head), DType.INT16, TensorKind.CONSTANT, data=np.array(w_q, dtype=np.int16, copy=True)))
    for kv_head, w_k in enumerate(block.split_k_proj_weights()):
        builder.add_tensor(TensorSpec(f"w_k_h{kv_head}", (d_model, d_head), DType.INT16, TensorKind.CONSTANT, data=np.array(w_k, dtype=np.int16, copy=True)))
    for kv_head, w_v in enumerate(block.split_v_proj_weights()):
        builder.add_tensor(TensorSpec(f"w_v_h{kv_head}", (d_model, d_head), DType.INT16, TensorKind.CONSTANT, data=np.array(w_v, dtype=np.int16, copy=True)))


def _add_prefill_head_runtime_tensors(
    builder: IRBuilder,
    *,
    d_head: int,
    n_heads: int,
    n_kv_heads: int,
    prompt_len: int,
    act_scale: float,
) -> None:
    token_names = [f"t{i}" for i in range(prompt_len)]
    for q_head in range(n_heads):
        builder.add_tensor(TensorSpec(f"q_int_h{q_head}", (prompt_len, d_head), DType.INT16, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"q_f_h{q_head}", (prompt_len, d_head), DType.FLOAT32, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"q_rope_f_h{q_head}", (prompt_len, d_head), DType.FLOAT32, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"q_rope_q_h{q_head}", (prompt_len, d_head), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}))
        kv_head = _kv_head_for_q_head(q_head, n_heads=n_heads, n_kv_heads=n_kv_heads)
        builder.add_tensor(TensorSpec(f"scores_h{q_head}", (prompt_len, prompt_len), DType.INT16, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"scores_f_h{q_head}", (prompt_len, prompt_len), DType.FLOAT32, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"score_scale_h{q_head}", (prompt_len, prompt_len), DType.FLOAT32, TensorKind.CONSTANT, data=_score_scale(d_head, (prompt_len, prompt_len), act_scale=act_scale)))
        builder.add_tensor(TensorSpec(f"scores_scaled_h{q_head}", (prompt_len, prompt_len), DType.FLOAT32, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"masked_scores_h{q_head}", (prompt_len, prompt_len), DType.FLOAT32, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"probs_h{q_head}", (prompt_len, prompt_len), DType.INT16, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"probs_q_h{q_head}", (prompt_len, prompt_len), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}))
        builder.add_tensor(TensorSpec(f"attn_h{q_head}", (prompt_len, d_head), DType.INT16, TensorKind.INTERMEDIATE))
        builder.tensors[f"scores_h{q_head}"].metadata["kv_head"] = kv_head
    for kv_head in range(n_kv_heads):
        _add_tensor_specs(
            builder,
            make_native_int16_kv_cache_specs(
                k_base_name=f"prefill_k_cache_h{kv_head}",
                v_base_name=f"prefill_v_cache_h{kv_head}",
                d_head=d_head,
                token_capacity=prompt_len,
                token_names=token_names,
                token_indices=list(range(prompt_len)),
                kind=TensorKind.INTERMEDIATE,
            ),
        )
        builder.add_tensor(TensorSpec(f"k_seq_h{kv_head}", (prompt_len, d_head), DType.INT16, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"v_seq_h{kv_head}", (prompt_len, d_head), DType.INT16, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"k_f_h{kv_head}", (prompt_len, d_head), DType.FLOAT32, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"k_rope_f_h{kv_head}", (prompt_len, d_head), DType.FLOAT32, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"k_rope_q_h{kv_head}", (prompt_len, d_head), DType.INT16, TensorKind.INTERMEDIATE))


def _add_decode_head_runtime_tensors(
    builder: IRBuilder,
    *,
    prefill_ref: dict[str, object],
    d_head: int,
    n_heads: int,
    n_kv_heads: int,
    prompt_len: int,
    cache_len: int,
    act_scale: float,
    cache_token_names: list[str],
) -> None:
    """Seed decode caches.

    K cache uses `(d_head, cache_len)` storage, while V cache uses
    `(cache_len, d_head)`. The K transpose is owned by the scatter helpers and
    by this decode seeding path.
    """
    for q_head in range(n_heads):
        builder.add_tensor(TensorSpec(f"q_int_h{q_head}", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"q_f_h{q_head}", (1, d_head), DType.FLOAT32, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"q_rope_f_h{q_head}", (1, d_head), DType.FLOAT32, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"q_rope_q_h{q_head}", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}))
        builder.add_tensor(TensorSpec(f"scores_h{q_head}", (1, cache_len), DType.INT16, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"scores_f_h{q_head}", (1, cache_len), DType.FLOAT32, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"score_scale_h{q_head}", (1, cache_len), DType.FLOAT32, TensorKind.CONSTANT, data=_score_scale(d_head, (1, cache_len), act_scale=act_scale)))
        builder.add_tensor(TensorSpec(f"scores_scaled_h{q_head}", (1, cache_len), DType.FLOAT32, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"probs_h{q_head}", (1, cache_len), DType.INT16, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"probs_q_h{q_head}", (1, cache_len), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}))
        builder.add_tensor(TensorSpec(f"attn_h{q_head}", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE))
    for kv_head in range(n_kv_heads):
        specs = make_native_int16_kv_cache_specs(
            k_base_name=f"k_cache_h{kv_head}",
            v_base_name=f"v_cache_h{kv_head}",
            d_head=d_head,
            token_capacity=cache_len,
            token_names=cache_token_names,
            token_indices=list(range(cache_len)),
            kind=TensorKind.INTERMEDIATE,
        )
        _add_tensor_specs(builder, specs)
        k_base = np.zeros((d_head, cache_len), dtype=np.int16)
        v_base = np.zeros((cache_len, d_head), dtype=np.int16)
        k_prefill = np.array(prefill_ref["k_heads"][kv_head], dtype=np.int16, copy=True)
        v_prefill = np.array(prefill_ref["v_heads"][kv_head], dtype=np.int16, copy=True)
        k_base[:, :prompt_len] = k_prefill.T
        v_base[:prompt_len, :] = v_prefill
        builder.tensors[f"k_cache_h{kv_head}"].kind = TensorKind.CONSTANT
        builder.tensors[f"k_cache_h{kv_head}"].data = k_base
        builder.tensors[f"v_cache_h{kv_head}"].kind = TensorKind.CONSTANT
        builder.tensors[f"v_cache_h{kv_head}"].data = v_base
        builder.add_tensor(TensorSpec(f"k_cur_h{kv_head}", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"v_cur_h{kv_head}", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"k_f_h{kv_head}", (1, d_head), DType.FLOAT32, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"k_rope_f_h{kv_head}", (1, d_head), DType.FLOAT32, TensorKind.INTERMEDIATE))
        builder.add_tensor(TensorSpec(f"k_rope_q_h{kv_head}", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE))


def _add_attention_concat_tensors(builder: IRBuilder, *, seq_len: int, d_head: int, n_heads: int) -> None:
    for q_head in range(1, n_heads):
        builder.add_tensor(TensorSpec(f"attn_cat_{q_head}", (seq_len, (q_head + 1) * d_head), DType.INT16, TensorKind.INTERMEDIATE))


def _append_attention_post_score_steps(
    builder: IRBuilder,
    *,
    n_heads: int,
    attn_scale: float,
    include_causal_mask: bool,
    past_kv_len: int = 0,
) -> None:
    for q_head in range(n_heads):
        builder.host(
            f"dequant_scores_h{q_head}",
            "dequantize",
            inputs=[f"scores_h{q_head}"],
            outputs=[f"scores_f_h{q_head}"],
            attrs={"scale": 1.0, "zero_point": 0},
        )
        builder.host(
            f"scale_scores_h{q_head}",
            "mul",
            inputs=[f"scores_f_h{q_head}", f"score_scale_h{q_head}"],
            outputs=[f"scores_scaled_h{q_head}"],
        )
        if include_causal_mask:
            builder.host(
                f"causal_mask_h{q_head}",
                "causal_mask",
                inputs=[f"scores_scaled_h{q_head}"],
                outputs=[f"masked_scores_h{q_head}"],
                attrs={"past_kv_len": past_kv_len, "fill_value": -1.0e10},
            )
            softmax_in = f"masked_scores_h{q_head}"
        else:
            softmax_in = f"scores_scaled_h{q_head}"
        builder.host(
            f"softmax_h{q_head}",
            "softmax_f16",
            inputs=[softmax_in],
            outputs=[f"probs_h{q_head}"],
            attrs={"axis": -1},
        )
        builder.host(
            f"quant_probs_h{q_head}",
            "quantize",
            inputs=[f"probs_h{q_head}"],
            outputs=[f"probs_q_h{q_head}"],
            attrs={"scale": attn_scale, "zero_point": 0, "dtype": DType.INT16, "input_encoding": "fp16_bits"},
        )


def _append_concat_alias_steps(builder: IRBuilder, *, n_heads: int) -> None:
    prev_attn_name = "attn_h0"
    for q_head in range(1, n_heads):
        cat_name = f"attn_cat_{q_head}"
        builder.host(
            f"concat_attn_{q_head}",
            "concat_lastdim2",
            inputs=[prev_attn_name, f"attn_h{q_head}"],
            outputs=[cat_name],
        )
        prev_attn_name = cat_name
    builder.host("alias_attn_cat", "alias", inputs=[prev_attn_name], outputs=["attn_cat"])
    builder.host("alias_attn_cat_a", "alias", inputs=["attn_cat"], outputs=["attn_cat_a"])


def _append_transformer_tail(builder: IRBuilder, *, act_scale: float, rms_norm_eps: float) -> None:
    builder.segment(
        "seg_o_proj",
        ops=[builder.matmul("op_o_proj", "attn_cat_a", "w_o", "o_int", in_dtype=DType.INT16, out_dtype=DType.INT16)],
        inputs=["attn_cat_a"],
        outputs=["o_int"],
    )
    builder.host("dequant_o", "dequantize", inputs=["o_int"], outputs=["o_f"], attrs={"scale": act_scale, "zero_point": 0})
    builder.host("residual1", "add", inputs=["x_in", "o_f"], outputs=["resid1"])
    builder.host("rmsnorm2", "rmsnorm", inputs=["resid1", "rms2_w"], outputs=["x_norm2"], attrs={"eps": rms_norm_eps})
    builder.host(
        "quant_x_norm2",
        "quantize",
        inputs=["x_norm2"],
        outputs=["x_norm2_q"],
        attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16},
    )
    builder.segment(
        "seg_ffn_up",
        ops=[
            builder.matmul("op_gate_proj", "x_norm2_q", "w_gate", "gate_int", in_dtype=DType.INT16, out_dtype=DType.INT16),
            builder.matmul("op_up_proj", "x_norm2_q", "w_up", "up_int", in_dtype=DType.INT16, out_dtype=DType.INT16),
        ],
        inputs=["x_norm2_q"],
        outputs=["gate_int", "up_int"],
    )
    builder.host("dequant_gate", "dequantize", inputs=["gate_int"], outputs=["gate_f"], attrs={"scale": act_scale, "zero_point": 0})
    builder.host("dequant_up", "dequantize", inputs=["up_int"], outputs=["up_f"], attrs={"scale": act_scale, "zero_point": 0})
    builder.host("silu_gate", "silu", inputs=["gate_f"], outputs=["gate_act"])
    builder.host("ffn_mul", "mul", inputs=["gate_act", "up_f"], outputs=["ffn_hidden"])
    builder.host(
        "quant_ffn_hidden",
        "quantize",
        inputs=["ffn_hidden"],
        outputs=["ffn_hidden_q"],
        attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16},
    )
    builder.segment(
        "seg_ffn_down",
        ops=[builder.matmul("op_down_proj", "ffn_hidden_q", "w_down", "ffn_out_int", in_dtype=DType.INT16, out_dtype=DType.INT16)],
        inputs=["ffn_hidden_q"],
        outputs=["ffn_out_int"],
    )
    builder.host("dequant_ffn_out", "dequantize", inputs=["ffn_out_int"], outputs=["ffn_out_f"], attrs={"scale": act_scale, "zero_point": 0})
    builder.host("residual2", "add", inputs=["resid1", "ffn_out_f"], outputs=["out"])


def build_prefill_artifact(
    *,
    d_model: int = 32,
    d_head: int = 8,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    ffn_hidden_dim: int = 128,
    prompt_len: int = 8,
    seed: int = 0,
    act_scale: float = 1.0 / 32.0,
    attn_scale: float = 1.0 / 256.0,
    rms_norm_eps: float = 1.0e-5,
    rope_theta: float = 500000.0,
    rope_scaling: dict[str, object] | None = None,
):
    if d_model <= 0 or d_model % 8 != 0:
        raise ValueError("d_model must be a positive multiple of 8")
    if d_head <= 0 or d_head % 8 != 0:
        raise ValueError("d_head must be a positive multiple of 8")
    if n_heads <= 0 or n_kv_heads <= 0:
        raise ValueError("n_heads and n_kv_heads must be positive")
    if n_heads % n_kv_heads != 0:
        raise ValueError("n_heads must be divisible by n_kv_heads")
    if d_model != n_heads * d_head:
        raise ValueError("d_model must equal n_heads * d_head")
    if ffn_hidden_dim <= 0 or ffn_hidden_dim % 8 != 0:
        raise ValueError("ffn_hidden_dim must be a positive multiple of 8")
    if prompt_len <= 0 or prompt_len % 8 != 0:
        raise ValueError("prompt_len must be a positive multiple of 8")
    if rms_norm_eps <= 0.0:
        raise ValueError("rms_norm_eps must be positive")
    if rope_theta <= 0.0:
        raise ValueError("rope_theta must be positive")
    _require_supported_rope_scaling(rope_scaling)

    state = build_shared_state(
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        ffn_hidden_dim=ffn_hidden_dim,
        prompt_len=prompt_len,
        seed=seed,
        act_scale=act_scale,
        attn_scale=attn_scale,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
    )
    block: QLlamaBlock = state["block"]
    act_scale = block.config.act_scale
    attn_scale = block.config.attn_scale
    rms_norm_eps = block.config.rms_norm_eps
    rope_theta = block.config.rope_theta
    ref = reference_prefill(
        state,
        d_head=d_head,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        act_scale=act_scale,
        attn_scale=attn_scale,
        rope_theta=rope_theta,
        rope_scaling=block.config.rope_scaling,
    )
    attn_dim = n_heads * d_head

    b = IRBuilder()
    _add_tensor_specs(
        b,
        _common_io_tensors(
            block=block,
            x_in=np.array(state["x_prompt_in"], dtype=np.float32, copy=True),
            d_model=d_model,
            ffn_hidden_dim=ffn_hidden_dim,
            attn_dim=attn_dim,
            seq_len=prompt_len,
            act_scale=act_scale,
        ),
    )
    _add_projection_tensors(b, block=block, d_model=d_model, d_head=d_head, n_heads=n_heads, n_kv_heads=n_kv_heads)
    _add_prefill_head_runtime_tensors(b, d_head=d_head, n_heads=n_heads, n_kv_heads=n_kv_heads, prompt_len=prompt_len, act_scale=act_scale)
    _add_attention_concat_tensors(b, seq_len=prompt_len, d_head=d_head, n_heads=n_heads)

    b.host("rmsnorm1", "rmsnorm", inputs=["x_in", "rms1_w"], outputs=["x_norm1"], attrs={"eps": rms_norm_eps})
    b.host(
        "quant_x_norm1",
        "quantize",
        inputs=["x_norm1"],
        outputs=["x_norm1_q"],
        attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16},
    )
    qkv_ops = [builder_op for q_head in range(n_heads) for builder_op in [b.matmul(f"op_q_h{q_head}", "x_norm1_q", f"w_q_h{q_head}", f"q_int_h{q_head}", in_dtype=DType.INT16, out_dtype=DType.INT16)]]
    for kv_head in range(n_kv_heads):
        qkv_ops.append(b.matmul(f"op_k_h{kv_head}", "x_norm1_q", f"w_k_h{kv_head}", f"k_seq_h{kv_head}", in_dtype=DType.INT16, out_dtype=DType.INT16))
        qkv_ops.append(b.matmul(f"op_v_h{kv_head}", "x_norm1_q", f"w_v_h{kv_head}", f"v_seq_h{kv_head}", in_dtype=DType.INT16, out_dtype=DType.INT16))
    b.segment(
        "seg_qkv",
        ops=qkv_ops,
        inputs=["x_norm1_q"],
        outputs=[*(f"q_int_h{i}" for i in range(n_heads)), *(f"k_seq_h{i}" for i in range(n_kv_heads)), *(f"v_seq_h{i}" for i in range(n_kv_heads))],
    )
    for q_head in range(n_heads):
        b.host(f"dequant_q_h{q_head}", "dequantize", inputs=[f"q_int_h{q_head}"], outputs=[f"q_f_h{q_head}"], attrs={"scale": act_scale, "zero_point": 0})
        b.host(
            f"rope_q_h{q_head}",
            "rope",
            inputs=[f"q_f_h{q_head}"],
            outputs=[f"q_rope_f_h{q_head}"],
            attrs={"head_dim": d_head, "position": 0, "theta": rope_theta},
        )
        b.host(
            f"quant_q_rope_h{q_head}",
            "quantize",
            inputs=[f"q_rope_f_h{q_head}"],
            outputs=[f"q_rope_q_h{q_head}"],
            attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16},
        )
    for kv_head in range(n_kv_heads):
        b.host(f"dequant_k_h{kv_head}", "dequantize", inputs=[f"k_seq_h{kv_head}"], outputs=[f"k_f_h{kv_head}"], attrs={"scale": act_scale, "zero_point": 0})
        b.host(
            f"rope_k_h{kv_head}",
            "rope",
            inputs=[f"k_f_h{kv_head}"],
            outputs=[f"k_rope_f_h{kv_head}"],
            attrs={"head_dim": d_head, "position": 0, "theta": rope_theta},
        )
        b.host(
            f"quant_k_rope_h{kv_head}",
            "quantize",
            inputs=[f"k_rope_f_h{kv_head}"],
            outputs=[f"k_rope_q_h{kv_head}"],
            attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16},
        )
        b.host(f"k_cache_scatter_matrix_h{kv_head}", "k_cache_scatter_matrix", inputs=[f"k_rope_q_h{kv_head}"], outputs=[f"prefill_k_cache_h{kv_head}"])
        b.host(f"v_cache_scatter_matrix_h{kv_head}", "v_cache_scatter_matrix", inputs=[f"v_seq_h{kv_head}"], outputs=[f"prefill_v_cache_h{kv_head}"])
    b.segment(
        "seg_score",
        ops=[
            b.matmul(
                f"op_qk_h{q_head}",
                f"q_rope_q_h{q_head}",
                f"prefill_k_cache_h{_kv_head_for_q_head(q_head, n_heads=n_heads, n_kv_heads=n_kv_heads)}",
                f"scores_h{q_head}",
                in_dtype=DType.INT16,
                out_dtype=DType.INT16,
            )
            for q_head in range(n_heads)
        ],
        inputs=[
            *(f"q_rope_q_h{i}" for i in range(n_heads)),
            *(f"prefill_k_cache_h{i}" for i in range(n_kv_heads)),
        ],
        outputs=[f"scores_h{i}" for i in range(n_heads)],
    )
    _append_attention_post_score_steps(b, n_heads=n_heads, attn_scale=attn_scale, include_causal_mask=True, past_kv_len=0)
    b.segment(
        "seg_value",
        ops=[
            b.matmul(
                f"op_av_h{q_head}",
                f"probs_q_h{q_head}",
                f"prefill_v_cache_h{_kv_head_for_q_head(q_head, n_heads=n_heads, n_kv_heads=n_kv_heads)}",
                f"attn_h{q_head}",
                shift=8,
                in_dtype=DType.INT16,
                out_dtype=DType.INT16,
            )
            for q_head in range(n_heads)
        ],
        inputs=[
            *(f"probs_q_h{i}" for i in range(n_heads)),
            *(f"prefill_v_cache_h{i}" for i in range(n_kv_heads)),
        ],
        outputs=[f"attn_h{i}" for i in range(n_heads)],
    )
    _append_concat_alias_steps(b, n_heads=n_heads)
    _append_transformer_tail(b, act_scale=act_scale, rms_norm_eps=rms_norm_eps)

    plan = b.finalize(inputs=[], outputs=["out"])
    plan.add_verification_step("out", "llama_prefill_out")
    artifact = compile_plan(plan, {"out": np.array(ref["out"], dtype=np.float32, copy=True)})
    return artifact, state, ref


def build_decode_artifact(
    *,
    d_model: int = 32,
    d_head: int = 8,
    n_heads: int = 4,
    n_kv_heads: int = 2,
    ffn_hidden_dim: int = 128,
    prompt_len: int = 8,
    seed: int = 0,
    act_scale: float = 1.0 / 32.0,
    attn_scale: float = 1.0 / 256.0,
    rms_norm_eps: float = 1.0e-5,
    rope_theta: float = 500000.0,
    rope_scaling: dict[str, object] | None = None,
):
    if rms_norm_eps <= 0.0:
        raise ValueError("rms_norm_eps must be positive")
    if rope_theta <= 0.0:
        raise ValueError("rope_theta must be positive")
    _require_supported_rope_scaling(rope_scaling)

    state = build_shared_state(
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        ffn_hidden_dim=ffn_hidden_dim,
        prompt_len=prompt_len,
        seed=seed,
        act_scale=act_scale,
        attn_scale=attn_scale,
        rms_norm_eps=rms_norm_eps,
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
    )
    block: QLlamaBlock = state["block"]
    act_scale = block.config.act_scale
    attn_scale = block.config.attn_scale
    rms_norm_eps = block.config.rms_norm_eps
    rope_theta = block.config.rope_theta
    prefill_ref = reference_prefill(
        state,
        d_head=d_head,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        act_scale=act_scale,
        attn_scale=attn_scale,
        rope_theta=rope_theta,
        rope_scaling=block.config.rope_scaling,
    )
    decode_ref = reference_decode(
        state,
        prefill_ref,
        d_head=d_head,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        act_scale=act_scale,
        attn_scale=attn_scale,
        rope_theta=rope_theta,
        rope_scaling=block.config.rope_scaling,
    )
    cache_len = prompt_len + 1
    attn_dim = n_heads * d_head
    decode_token_name = "td"
    cache_token_names = [f"t{i}" for i in range(prompt_len)] + [decode_token_name]

    b = IRBuilder()
    _add_tensor_specs(
        b,
        _common_io_tensors(
            block=block,
            x_in=np.array(state["x_decode_in"], dtype=np.float32, copy=True),
            d_model=d_model,
            ffn_hidden_dim=ffn_hidden_dim,
            attn_dim=attn_dim,
            seq_len=1,
            act_scale=act_scale,
        ),
    )
    _add_projection_tensors(b, block=block, d_model=d_model, d_head=d_head, n_heads=n_heads, n_kv_heads=n_kv_heads)
    _add_decode_head_runtime_tensors(
        b,
        prefill_ref=prefill_ref,
        d_head=d_head,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        prompt_len=prompt_len,
        cache_len=cache_len,
        act_scale=act_scale,
        cache_token_names=cache_token_names,
    )
    _add_attention_concat_tensors(b, seq_len=1, d_head=d_head, n_heads=n_heads)

    b.host("rmsnorm1", "rmsnorm", inputs=["x_in", "rms1_w"], outputs=["x_norm1"], attrs={"eps": rms_norm_eps})
    b.host(
        "quant_x_norm1",
        "quantize",
        inputs=["x_norm1"],
        outputs=["x_norm1_q"],
        attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16},
    )
    qkv_ops = [builder_op for q_head in range(n_heads) for builder_op in [b.matmul(f"op_q_h{q_head}", "x_norm1_q", f"w_q_h{q_head}", f"q_int_h{q_head}", in_dtype=DType.INT16, out_dtype=DType.INT16)]]
    for kv_head in range(n_kv_heads):
        qkv_ops.append(b.matmul(f"op_k_h{kv_head}", "x_norm1_q", f"w_k_h{kv_head}", f"k_cur_h{kv_head}", in_dtype=DType.INT16, out_dtype=DType.INT16))
        qkv_ops.append(b.matmul(f"op_v_h{kv_head}", "x_norm1_q", f"w_v_h{kv_head}", f"v_cur_h{kv_head}", in_dtype=DType.INT16, out_dtype=DType.INT16))
    b.segment(
        "seg_qkv",
        ops=qkv_ops,
        inputs=["x_norm1_q"],
        outputs=[*(f"q_int_h{i}" for i in range(n_heads)), *(f"k_cur_h{i}" for i in range(n_kv_heads)), *(f"v_cur_h{i}" for i in range(n_kv_heads))],
    )
    for q_head in range(n_heads):
        b.host(f"dequant_q_h{q_head}", "dequantize", inputs=[f"q_int_h{q_head}"], outputs=[f"q_f_h{q_head}"], attrs={"scale": act_scale, "zero_point": 0})
        b.host(
            f"rope_q_h{q_head}",
            "rope",
            inputs=[f"q_f_h{q_head}"],
            outputs=[f"q_rope_f_h{q_head}"],
            attrs={"head_dim": d_head, "position": prompt_len, "theta": rope_theta},
        )
        b.host(
            f"quant_q_rope_h{q_head}",
            "quantize",
            inputs=[f"q_rope_f_h{q_head}"],
            outputs=[f"q_rope_q_h{q_head}"],
            attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16},
        )
    for kv_head in range(n_kv_heads):
        b.host(f"dequant_k_h{kv_head}", "dequantize", inputs=[f"k_cur_h{kv_head}"], outputs=[f"k_f_h{kv_head}"], attrs={"scale": act_scale, "zero_point": 0})
        b.host(
            f"rope_k_h{kv_head}",
            "rope",
            inputs=[f"k_f_h{kv_head}"],
            outputs=[f"k_rope_f_h{kv_head}"],
            attrs={"head_dim": d_head, "position": prompt_len, "theta": rope_theta},
        )
        b.host(
            f"quant_k_rope_h{kv_head}",
            "quantize",
            inputs=[f"k_rope_f_h{kv_head}"],
            outputs=[f"k_rope_q_h{kv_head}"],
            attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16},
        )
        b.host(
            f"k_append_h{kv_head}",
            "k_cache_scatter_write",
            inputs=[f"k_rope_q_h{kv_head}", f"k_cache_h{kv_head}"],
            outputs=[f"k_cache_h{kv_head}_{decode_token_name}"],
            attrs={"token_index": prompt_len, "k_cache_base": f"k_cache_h{kv_head}"},
        )
        b.host(
            f"v_append_h{kv_head}",
            "v_cache_scatter_write",
            inputs=[f"v_cur_h{kv_head}", f"v_cache_h{kv_head}"],
            outputs=[f"v_cache_h{kv_head}_{decode_token_name}"],
            attrs={"token_index": prompt_len, "v_cache_base": f"v_cache_h{kv_head}"},
        )
    b.segment(
        "seg_score",
        ops=[
            b.matmul(
                f"op_qk_h{q_head}",
                f"q_rope_q_h{q_head}",
                f"k_cache_h{_kv_head_for_q_head(q_head, n_heads=n_heads, n_kv_heads=n_kv_heads)}",
                f"scores_h{q_head}",
                in_dtype=DType.INT16,
                out_dtype=DType.INT16,
            )
            for q_head in range(n_heads)
        ],
        inputs=[f"q_rope_q_h{i}" for i in range(n_heads)],
        outputs=[f"scores_h{i}" for i in range(n_heads)],
    )
    _append_attention_post_score_steps(b, n_heads=n_heads, attn_scale=attn_scale, include_causal_mask=False)
    b.segment(
        "seg_value",
        ops=[
            b.matmul(
                f"op_av_h{q_head}",
                f"probs_q_h{q_head}",
                f"v_cache_h{_kv_head_for_q_head(q_head, n_heads=n_heads, n_kv_heads=n_kv_heads)}",
                f"attn_h{q_head}",
                shift=8,
                in_dtype=DType.INT16,
                out_dtype=DType.INT16,
            )
            for q_head in range(n_heads)
        ],
        inputs=[f"probs_q_h{i}" for i in range(n_heads)],
        outputs=[f"attn_h{i}" for i in range(n_heads)],
    )
    _append_concat_alias_steps(b, n_heads=n_heads)
    _append_transformer_tail(b, act_scale=act_scale, rms_norm_eps=rms_norm_eps)

    plan = b.finalize(inputs=[], outputs=["out"])
    plan.add_verification_step("out", "llama_decode_out")
    artifact = compile_plan(plan, {"out": np.array(decode_ref["out"], dtype=np.float32, copy=True)})
    return artifact, state, prefill_ref, decode_ref
