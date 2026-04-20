from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tinynpu_jit import (
    DType,
    ExecutionPlan,
    HostOp,
    MatMulOp,
    NpuSegment,
    TensorKind,
    TensorSpec,
    compile_plan,
    make_native_int16_kv_cache_specs,
)
from tinynpu_jit.golden import GoldenModel


def _rand_i16(rng: np.random.Generator, shape: tuple[int, ...], low: int = -2, high: int = 3) -> np.ndarray:
    return rng.integers(low, high, size=shape, endpoint=False, dtype=np.int16)


def _rand_f32(rng: np.random.Generator, shape: tuple[int, ...], low: float = -0.25, high: float = 0.25) -> np.ndarray:
    return rng.uniform(low, high, size=shape).astype(np.float32)


def _quantize_bias_fp32(bias: np.ndarray, *, out_scale: float) -> np.ndarray:
    if out_scale <= 0.0:
        raise ValueError(f"Bias quantization scale must be positive, got {out_scale}.")
    bias_fp = np.asarray(bias, dtype=np.float32)
    return np.rint(bias_fp / np.float32(out_scale)).astype(np.int32)


def _quantize_weight_fp32(weight: np.ndarray) -> np.ndarray:
    weight_fp = np.asarray(weight, dtype=np.float32)
    return np.clip(np.rint(weight_fp), -32768, 32767).astype(np.int16)


def _score_scale(d_head: int, shape: tuple[int, int], *, act_scale: float) -> np.ndarray:
    score_scale = np.float32((float(act_scale) * float(act_scale)) / np.sqrt(float(d_head)))
    return np.full(shape, score_scale, dtype=np.float32)


def _materialize_inputs(token_emb: np.ndarray, pos_emb: np.ndarray, positions: slice) -> np.ndarray:
    return (np.asarray(token_emb, dtype=np.float32) + np.asarray(pos_emb[positions], dtype=np.float32)).astype(np.float32)


def _fp16_roundtrip(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).astype(np.float16).astype(np.float32)


def _layernorm_runtime_approx(x: np.ndarray, wb: np.ndarray, eps: float) -> np.ndarray:
    x_f = np.asarray(x, dtype=np.float32)
    wb_f = np.asarray(wb, dtype=np.float32)
    gamma = wb_f[0]
    beta = wb_f[1]
    mean = np.mean(x_f, axis=-1, keepdims=True, dtype=np.float32)
    centered = x_f - mean
    var = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    inv_std = np.float32(1.0) / np.sqrt(var + np.float32(eps))
    return centered * inv_std * gamma.reshape(1, -1) + beta.reshape(1, -1)


def _layernorm_exact(x: np.ndarray, wb: np.ndarray, eps: float) -> np.ndarray:
    x_f = np.asarray(x, dtype=np.float32)
    wb_f = np.asarray(wb, dtype=np.float32)
    gamma = wb_f[0]
    beta = wb_f[1]
    mean = np.mean(x_f, axis=-1, keepdims=True, dtype=np.float32)
    centered = x_f - mean
    var = np.mean(centered * centered, axis=-1, keepdims=True, dtype=np.float32)
    norm = centered / np.sqrt(var + np.float32(eps), dtype=np.float32)
    return (norm * gamma.reshape(1, -1) + beta.reshape(1, -1)).astype(np.float32)


def _gpt2_gelu_tanh(x: np.ndarray) -> np.ndarray:
    x_f = np.asarray(x, dtype=np.float32)
    coeff = np.float32(np.sqrt(2.0 / np.pi))
    inner = coeff * (x_f + np.float32(0.044715) * x_f * x_f * x_f)
    return (np.float32(0.5) * x_f * (np.float32(1.0) + np.tanh(inner).astype(np.float32))).astype(np.float32)


@dataclass(frozen=True)
class QGPT2BlockConfig:
    """Quantized GPT-2 block shape and scaling.

    This mirrors the Hugging Face GPT-2 block structure semantically:
    - ln_1
    - attn.c_attn
    - attn.c_proj
    - ln_2
    - mlp.c_fc
    - mlp.c_proj
    """

    d_model: int
    d_head: int
    n_heads: int
    ffn_dim: int
    act_scale: float = 1.0 / 32.0
    attn_scale: float = 1.0 / 256.0

    @property
    def attn_dim(self) -> int:
        return self.n_heads * self.d_head


@dataclass(frozen=True)
class QGPT2Block:
    """Quantized GPT-2 block weights with Hugging Face-style component naming."""

    config: QGPT2BlockConfig
    ln_1_wb: np.ndarray
    attn_c_attn_w: np.ndarray
    attn_c_attn_b: np.ndarray
    attn_c_proj_w: np.ndarray
    attn_c_proj_b: np.ndarray
    ln_2_wb: np.ndarray
    mlp_c_fc_w: np.ndarray
    mlp_c_fc_b: np.ndarray
    mlp_c_proj_w: np.ndarray
    mlp_c_proj_b: np.ndarray

    @classmethod
    def random(cls, rng: np.random.Generator, config: QGPT2BlockConfig) -> QGPT2Block:
        attn_dim = config.attn_dim
        return cls(
            config=config,
            ln_1_wb=np.stack(
                [
                    rng.uniform(0.5, 1.5, size=(config.d_model,)).astype(np.float32),
                    _rand_f32(rng, (config.d_model,), low=-0.1, high=0.1),
                ],
                axis=0,
            ).astype(np.float32),
            attn_c_attn_w=_rand_i16(rng, (config.d_model, 3 * attn_dim)),
            attn_c_attn_b=_rand_f32(rng, (1, 3 * attn_dim), low=-0.05, high=0.05),
            attn_c_proj_w=_rand_i16(rng, (attn_dim, config.d_model)),
            attn_c_proj_b=_rand_f32(rng, (1, config.d_model), low=-0.05, high=0.05),
            ln_2_wb=np.stack(
                [
                    rng.uniform(0.5, 1.5, size=(config.d_model,)).astype(np.float32),
                    _rand_f32(rng, (config.d_model,), low=-0.1, high=0.1),
                ],
                axis=0,
            ).astype(np.float32),
            mlp_c_fc_w=_rand_i16(rng, (config.d_model, config.ffn_dim)),
            mlp_c_fc_b=_rand_f32(rng, (1, config.ffn_dim), low=-0.05, high=0.05),
            mlp_c_proj_w=_rand_i16(rng, (config.ffn_dim, config.d_model)),
            mlp_c_proj_b=_rand_f32(rng, (1, config.d_model), low=-0.05, high=0.05),
        )

    @classmethod
    def from_fp32(
        cls,
        *,
        config: QGPT2BlockConfig,
        ln_1_wb: np.ndarray,
        attn_c_attn_w: np.ndarray,
        attn_c_attn_b: np.ndarray,
        attn_c_proj_w: np.ndarray,
        attn_c_proj_b: np.ndarray,
        ln_2_wb: np.ndarray,
        mlp_c_fc_w: np.ndarray,
        mlp_c_fc_b: np.ndarray,
        mlp_c_proj_w: np.ndarray,
        mlp_c_proj_b: np.ndarray,
    ) -> QGPT2Block:
        """Construct a quantized GPT-2 block from fused GPT2Block-style FP32 tensors.

        This matches the Hugging Face block field structure semantically:
        - ln_1.{weight,bias} packed as `ln_1_wb`
        - attn.c_attn.{weight,bias}
        - attn.c_proj.{weight,bias}
        - ln_2.{weight,bias} packed as `ln_2_wb`
        - mlp.c_fc.{weight,bias}
        - mlp.c_proj.{weight,bias}
        """

        return cls(
            config=config,
            ln_1_wb=np.asarray(ln_1_wb, dtype=np.float32),
            attn_c_attn_w=_quantize_weight_fp32(attn_c_attn_w),
            attn_c_attn_b=np.asarray(attn_c_attn_b, dtype=np.float32),
            attn_c_proj_w=_quantize_weight_fp32(attn_c_proj_w),
            attn_c_proj_b=np.asarray(attn_c_proj_b, dtype=np.float32),
            ln_2_wb=np.asarray(ln_2_wb, dtype=np.float32),
            mlp_c_fc_w=_quantize_weight_fp32(mlp_c_fc_w),
            mlp_c_fc_b=np.asarray(mlp_c_fc_b, dtype=np.float32),
            mlp_c_proj_w=_quantize_weight_fp32(mlp_c_proj_w),
            mlp_c_proj_b=np.asarray(mlp_c_proj_b, dtype=np.float32),
        )

    def split_c_attn_weights(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        qkv = np.asarray(self.attn_c_attn_w, dtype=np.int16)
        attn_dim = self.config.attn_dim
        q_all = qkv[:, :attn_dim]
        k_all = qkv[:, attn_dim : 2 * attn_dim]
        v_all = qkv[:, 2 * attn_dim :]
        d_head = self.config.d_head
        return (
            [np.array(q_all[:, i * d_head : (i + 1) * d_head], copy=True) for i in range(self.config.n_heads)],
            [np.array(k_all[:, i * d_head : (i + 1) * d_head], copy=True) for i in range(self.config.n_heads)],
            [np.array(v_all[:, i * d_head : (i + 1) * d_head], copy=True) for i in range(self.config.n_heads)],
        )

    def split_c_attn_biases_fp32(self) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        qkv = np.asarray(self.attn_c_attn_b, dtype=np.float32)
        attn_dim = self.config.attn_dim
        q_all = qkv[:, :attn_dim]
        k_all = qkv[:, attn_dim : 2 * attn_dim]
        v_all = qkv[:, 2 * attn_dim :]
        d_head = self.config.d_head
        return (
            [np.array(q_all[:, i * d_head : (i + 1) * d_head], copy=True) for i in range(self.config.n_heads)],
            [np.array(k_all[:, i * d_head : (i + 1) * d_head], copy=True) for i in range(self.config.n_heads)],
            [np.array(v_all[:, i * d_head : (i + 1) * d_head], copy=True) for i in range(self.config.n_heads)],
        )


def build_shared_state(
    *,
    d_model: int,
    d_head: int,
    n_heads: int,
    ffn_dim: int,
    prompt_len: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    pos_capacity = prompt_len + 1
    config = QGPT2BlockConfig(
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
    )
    block = QGPT2Block.random(rng, config)
    state: dict[str, object] = {
        "config": config,
        "block": block,
        "tok_prompt": _rand_f32(rng, (prompt_len, d_model)),
        "tok_decode": _rand_f32(rng, (1, d_model)),
        "pos_emb": _rand_f32(rng, (pos_capacity, d_model)),
        "ln1_wb": np.array(block.ln_1_wb, dtype=np.float32, copy=True),
        "ln2_wb": np.array(block.ln_2_wb, dtype=np.float32, copy=True),
        "attn_c_attn_w": np.array(block.attn_c_attn_w, dtype=np.int16, copy=True),
        "attn_c_attn_b": np.array(block.attn_c_attn_b, dtype=np.float32, copy=True),
        "attn_c_proj_w": np.array(block.attn_c_proj_w, dtype=np.int16, copy=True),
        "attn_c_proj_b": np.array(block.attn_c_proj_b, dtype=np.float32, copy=True),
        "mlp_c_fc_w": np.array(block.mlp_c_fc_w, dtype=np.int16, copy=True),
        "mlp_c_fc_b": np.array(block.mlp_c_fc_b, dtype=np.float32, copy=True),
        "mlp_c_proj_w": np.array(block.mlp_c_proj_w, dtype=np.int16, copy=True),
        "mlp_c_proj_b": np.array(block.mlp_c_proj_b, dtype=np.float32, copy=True),
    }
    w_q, w_k, w_v = block.split_c_attn_weights()
    b_q, b_k, b_v = block.split_c_attn_biases_fp32()
    state.update(
        {
            "w_q": w_q,
            "w_k": w_k,
            "w_v": w_v,
            "b_q": b_q,
            "b_k": b_k,
            "b_v": b_v,
            "w_o": np.array(block.attn_c_proj_w, dtype=np.int16, copy=True),
            "b_o": np.array(block.attn_c_proj_b, dtype=np.float32, copy=True),
            "w_fc": np.array(block.mlp_c_fc_w, dtype=np.int16, copy=True),
            "b_fc": np.array(block.mlp_c_fc_b, dtype=np.float32, copy=True),
            "w_proj": np.array(block.mlp_c_proj_w, dtype=np.int16, copy=True),
            "b_proj": np.array(block.mlp_c_proj_b, dtype=np.float32, copy=True),
        }
    )
    state["x_prompt_in"] = _materialize_inputs(state["tok_prompt"], state["pos_emb"], slice(0, prompt_len))
    state["x_decode_in"] = _materialize_inputs(state["tok_decode"], state["pos_emb"], slice(prompt_len, prompt_len + 1))
    return state


def reference_prefill(
    state: dict[str, object],
    *,
    d_head: int,
    n_heads: int,
    act_scale: float,
    attn_scale: float,
) -> dict[str, object]:
    golden = GoldenModel()
    block: QGPT2Block = state["block"]
    x_in = np.array(state["x_prompt_in"], dtype=np.float32, copy=True)
    prompt_len = x_in.shape[0]
    ln1_wb = np.array(block.ln_1_wb, dtype=np.float32, copy=True)
    ln2_wb = np.array(block.ln_2_wb, dtype=np.float32, copy=True)
    w_q, w_k, w_v = block.split_c_attn_weights()
    b_q_f, b_k_f, b_v_f = block.split_c_attn_biases_fp32()
    b_q = [_quantize_bias_fp32(np.array(b, dtype=np.float32, copy=True), out_scale=act_scale) for b in b_q_f]
    b_k = [_quantize_bias_fp32(np.array(b, dtype=np.float32, copy=True), out_scale=act_scale) for b in b_k_f]
    b_v = [_quantize_bias_fp32(np.array(b, dtype=np.float32, copy=True), out_scale=act_scale) for b in b_v_f]
    w_o = np.array(block.attn_c_proj_w, dtype=np.int16, copy=True)
    b_o = _quantize_bias_fp32(np.array(block.attn_c_proj_b, dtype=np.float32, copy=True), out_scale=act_scale)
    w_fc = np.array(block.mlp_c_fc_w, dtype=np.int16, copy=True)
    b_fc = _quantize_bias_fp32(np.array(block.mlp_c_fc_b, dtype=np.float32, copy=True), out_scale=act_scale)
    w_proj = np.array(block.mlp_c_proj_w, dtype=np.int16, copy=True)
    b_proj = _quantize_bias_fp32(np.array(block.mlp_c_proj_b, dtype=np.float32, copy=True), out_scale=act_scale)

    x_norm1 = _fp16_roundtrip(_layernorm_runtime_approx(x_in, ln1_wb, 1.0e-5))
    x_norm1_q = golden.quantize(x_norm1, scale=act_scale, zero_point=0, out_dtype=DType.INT16)

    attn_heads: list[np.ndarray] = []
    k_heads: list[np.ndarray] = []
    v_heads: list[np.ndarray] = []
    score_scale = np.float32((float(act_scale) * float(act_scale)) / np.sqrt(float(d_head)))
    for head_idx in range(n_heads):
        q_int = golden.matmul(x_norm1_q, w_q[head_idx], bias=b_q[head_idx], out_dtype=DType.INT16)
        k_int = golden.matmul(x_norm1_q, w_k[head_idx], bias=b_k[head_idx], out_dtype=DType.INT16)
        v_int = golden.matmul(x_norm1_q, w_v[head_idx], bias=b_v[head_idx], out_dtype=DType.INT16)
        scores = golden.matmul(q_int, np.array(k_int.T, dtype=np.int16, copy=True), out_dtype=DType.INT16).astype(np.float32)
        scores = (scores * score_scale).astype(np.float32)
        masked = np.array(scores, copy=True)
        for row in range(prompt_len):
            if row + 1 < prompt_len:
                masked[row, row + 1 :] = np.float32(-1.0e10)
        probs = _fp16_roundtrip(golden.softmax(masked, axis=-1).astype(np.float32))
        probs_q = golden.quantize(probs, scale=attn_scale, zero_point=0, out_dtype=DType.INT16)
        attn = golden.matmul(probs_q, v_int, shift=8, out_dtype=DType.INT16)
        attn_heads.append(attn)
        k_heads.append(k_int)
        v_heads.append(v_int)

    attn_cat = np.concatenate(attn_heads, axis=-1).astype(np.int16) if n_heads > 1 else np.array(attn_heads[0], copy=True)
    o_int = golden.matmul(attn_cat, w_o, bias=b_o, out_dtype=DType.INT16)
    o_f = golden.dequantize(o_int, scale=act_scale, zero_point=0)
    resid1 = (x_in + o_f).astype(np.float32)
    x_norm2 = _fp16_roundtrip(_layernorm_runtime_approx(resid1, ln2_wb, 1.0e-5))
    x_norm2_q = golden.quantize(x_norm2, scale=act_scale, zero_point=0, out_dtype=DType.INT16)
    ffn_fc_int = golden.matmul(x_norm2_q, w_fc, bias=b_fc, activation="h_gelu", out_dtype=DType.INT16)
    ffn_out_int = golden.matmul(ffn_fc_int, w_proj, bias=b_proj, out_dtype=DType.INT16)
    ffn_out_f = golden.dequantize(ffn_out_int, scale=act_scale, zero_point=0)
    out = (resid1 + ffn_out_f).astype(np.float32)
    return {
        "x_norm1": x_norm1,
        "x_norm1_q": x_norm1_q,
        "k_heads": k_heads,
        "v_heads": v_heads,
        "attn_cat": attn_cat,
        "o_int": o_int,
        "o_f": o_f,
        "resid1": resid1,
        "x_norm2": x_norm2,
        "x_norm2_q": x_norm2_q,
        "ffn_fc_int": ffn_fc_int,
        "ffn_out_int": ffn_out_int,
        "ffn_out_f": ffn_out_f,
        "out": out,
    }


def reference_prefill_float(
    state: dict[str, object],
    *,
    d_head: int,
    n_heads: int,
) -> dict[str, object]:
    block: QGPT2Block = state["block"]
    x_in = np.array(state["x_prompt_in"], dtype=np.float32, copy=True)
    prompt_len = x_in.shape[0]
    ln1_wb = np.array(block.ln_1_wb, dtype=np.float32, copy=True)
    ln2_wb = np.array(block.ln_2_wb, dtype=np.float32, copy=True)
    w_q, w_k, w_v = block.split_c_attn_weights()
    b_q, b_k, b_v = block.split_c_attn_biases_fp32()
    w_o = np.array(block.attn_c_proj_w, dtype=np.float32, copy=True)
    b_o = np.array(block.attn_c_proj_b, dtype=np.float32, copy=True)
    w_fc = np.array(block.mlp_c_fc_w, dtype=np.float32, copy=True)
    b_fc = np.array(block.mlp_c_fc_b, dtype=np.float32, copy=True)
    w_proj = np.array(block.mlp_c_proj_w, dtype=np.float32, copy=True)
    b_proj = np.array(block.mlp_c_proj_b, dtype=np.float32, copy=True)

    x_norm1 = _layernorm_exact(x_in, ln1_wb, 1.0e-5)
    attn_heads: list[np.ndarray] = []
    q_heads: list[np.ndarray] = []
    k_heads: list[np.ndarray] = []
    v_heads: list[np.ndarray] = []
    probs_heads: list[np.ndarray] = []
    scores_heads: list[np.ndarray] = []
    scale = np.float32(1.0 / np.sqrt(float(d_head)))
    for head_idx in range(n_heads):
        q = (x_norm1 @ np.asarray(w_q[head_idx], dtype=np.float32) + np.asarray(b_q[head_idx], dtype=np.float32)).astype(np.float32)
        k = (x_norm1 @ np.asarray(w_k[head_idx], dtype=np.float32) + np.asarray(b_k[head_idx], dtype=np.float32)).astype(np.float32)
        v = (x_norm1 @ np.asarray(w_v[head_idx], dtype=np.float32) + np.asarray(b_v[head_idx], dtype=np.float32)).astype(np.float32)
        scores = ((q @ k.T) * scale).astype(np.float32)
        masked = np.array(scores, copy=True)
        for row in range(prompt_len):
            if row + 1 < prompt_len:
                masked[row, row + 1 :] = np.float32(-1.0e10)
        shifted = masked - np.max(masked, axis=-1, keepdims=True)
        probs = np.exp(shifted).astype(np.float32)
        probs /= np.sum(probs, axis=-1, keepdims=True, dtype=np.float32)
        attn = (probs @ v).astype(np.float32)
        q_heads.append(q)
        k_heads.append(k)
        v_heads.append(v)
        scores_heads.append(masked)
        probs_heads.append(probs)
        attn_heads.append(attn)

    attn_cat = np.concatenate(attn_heads, axis=-1).astype(np.float32) if n_heads > 1 else np.array(attn_heads[0], copy=True)
    o_f = (attn_cat @ w_o + b_o).astype(np.float32)
    resid1 = (x_in + o_f).astype(np.float32)
    x_norm2 = _layernorm_exact(resid1, ln2_wb, 1.0e-5)
    ffn_fc = (x_norm2 @ w_fc + b_fc).astype(np.float32)
    ffn_gelu = _gpt2_gelu_tanh(ffn_fc)
    ffn_out_f = (ffn_gelu @ w_proj + b_proj).astype(np.float32)
    out = (resid1 + ffn_out_f).astype(np.float32)
    return {
        "x_norm1": x_norm1,
        "q_heads": q_heads,
        "k_heads": k_heads,
        "v_heads": v_heads,
        "scores_heads": scores_heads,
        "probs_heads": probs_heads,
        "attn_cat": attn_cat,
        "o_f": o_f,
        "resid1": resid1,
        "x_norm2": x_norm2,
        "ffn_fc": ffn_fc,
        "ffn_gelu": ffn_gelu,
        "ffn_out_f": ffn_out_f,
        "out": out,
    }


def reference_decode(
    state: dict[str, object],
    prefill_ref: dict[str, object],
    *,
    d_head: int,
    n_heads: int,
    act_scale: float,
    attn_scale: float,
) -> dict[str, object]:
    golden = GoldenModel()
    block: QGPT2Block = state["block"]
    x_in = np.array(state["x_decode_in"], dtype=np.float32, copy=True)
    ln1_wb = np.array(block.ln_1_wb, dtype=np.float32, copy=True)
    ln2_wb = np.array(block.ln_2_wb, dtype=np.float32, copy=True)
    w_q, w_k, w_v = block.split_c_attn_weights()
    b_q_f, b_k_f, b_v_f = block.split_c_attn_biases_fp32()
    b_q = [_quantize_bias_fp32(np.array(b, dtype=np.float32, copy=True), out_scale=act_scale) for b in b_q_f]
    b_k = [_quantize_bias_fp32(np.array(b, dtype=np.float32, copy=True), out_scale=act_scale) for b in b_k_f]
    b_v = [_quantize_bias_fp32(np.array(b, dtype=np.float32, copy=True), out_scale=act_scale) for b in b_v_f]
    w_o = np.array(block.attn_c_proj_w, dtype=np.int16, copy=True)
    b_o = _quantize_bias_fp32(np.array(block.attn_c_proj_b, dtype=np.float32, copy=True), out_scale=act_scale)
    w_fc = np.array(block.mlp_c_fc_w, dtype=np.int16, copy=True)
    b_fc = _quantize_bias_fp32(np.array(block.mlp_c_fc_b, dtype=np.float32, copy=True), out_scale=act_scale)
    w_proj = np.array(block.mlp_c_proj_w, dtype=np.int16, copy=True)
    b_proj = _quantize_bias_fp32(np.array(block.mlp_c_proj_b, dtype=np.float32, copy=True), out_scale=act_scale)

    x_norm1 = _fp16_roundtrip(_layernorm_runtime_approx(x_in, ln1_wb, 1.0e-5))
    x_norm1_q = golden.quantize(x_norm1, scale=act_scale, zero_point=0, out_dtype=DType.INT16)

    attn_heads: list[np.ndarray] = []
    k_cur_heads: list[np.ndarray] = []
    v_cur_heads: list[np.ndarray] = []
    score_scale = np.float32((float(act_scale) * float(act_scale)) / np.sqrt(float(d_head)))
    for head_idx in range(n_heads):
        q_int = golden.matmul(x_norm1_q, w_q[head_idx], bias=b_q[head_idx], out_dtype=DType.INT16)
        k_cur = golden.matmul(x_norm1_q, w_k[head_idx], bias=b_k[head_idx], out_dtype=DType.INT16)
        v_cur = golden.matmul(x_norm1_q, w_v[head_idx], bias=b_v[head_idx], out_dtype=DType.INT16)
        k_prefill = np.array(prefill_ref["k_heads"][head_idx], dtype=np.int16, copy=True)
        v_prefill = np.array(prefill_ref["v_heads"][head_idx], dtype=np.int16, copy=True)
        k_full = np.concatenate([k_prefill, k_cur], axis=0).astype(np.int16)
        v_full = np.concatenate([v_prefill, v_cur], axis=0).astype(np.int16)
        scores = golden.matmul(q_int, np.array(k_full.T, dtype=np.int16, copy=True), out_dtype=DType.INT16).astype(np.float32)
        scores = (scores * score_scale).astype(np.float32)
        probs = _fp16_roundtrip(golden.softmax(scores, axis=-1).astype(np.float32))
        probs_q = golden.quantize(probs, scale=attn_scale, zero_point=0, out_dtype=DType.INT16)
        attn = golden.matmul(probs_q, v_full, shift=8, out_dtype=DType.INT16)
        attn_heads.append(attn)
        k_cur_heads.append(k_cur)
        v_cur_heads.append(v_cur)

    attn_cat = np.concatenate(attn_heads, axis=-1).astype(np.int16) if n_heads > 1 else np.array(attn_heads[0], copy=True)
    o_int = golden.matmul(attn_cat, w_o, bias=b_o, out_dtype=DType.INT16)
    o_f = golden.dequantize(o_int, scale=act_scale, zero_point=0)
    resid1 = (x_in + o_f).astype(np.float32)
    x_norm2 = _fp16_roundtrip(_layernorm_runtime_approx(resid1, ln2_wb, 1.0e-5))
    x_norm2_q = golden.quantize(x_norm2, scale=act_scale, zero_point=0, out_dtype=DType.INT16)
    ffn_fc_int = golden.matmul(x_norm2_q, w_fc, bias=b_fc, activation="h_gelu", out_dtype=DType.INT16)
    ffn_out_int = golden.matmul(ffn_fc_int, w_proj, bias=b_proj, out_dtype=DType.INT16)
    ffn_out_f = golden.dequantize(ffn_out_int, scale=act_scale, zero_point=0)
    out = (resid1 + ffn_out_f).astype(np.float32)
    return {
        "x_norm1": x_norm1,
        "x_norm1_q": x_norm1_q,
        "k_cur_heads": k_cur_heads,
        "v_cur_heads": v_cur_heads,
        "attn_cat": attn_cat,
        "o_int": o_int,
        "o_f": o_f,
        "resid1": resid1,
        "x_norm2": x_norm2,
        "x_norm2_q": x_norm2_q,
        "ffn_fc_int": ffn_fc_int,
        "ffn_out_int": ffn_out_int,
        "ffn_out_f": ffn_out_f,
        "out": out,
    }


def reference_decode_float(
    state: dict[str, object],
    prefill_ref: dict[str, object],
    *,
    d_head: int,
    n_heads: int,
) -> dict[str, object]:
    block: QGPT2Block = state["block"]
    x_in = np.array(state["x_decode_in"], dtype=np.float32, copy=True)
    ln1_wb = np.array(block.ln_1_wb, dtype=np.float32, copy=True)
    ln2_wb = np.array(block.ln_2_wb, dtype=np.float32, copy=True)
    w_q, w_k, w_v = block.split_c_attn_weights()
    b_q, b_k, b_v = block.split_c_attn_biases_fp32()
    w_o = np.array(block.attn_c_proj_w, dtype=np.float32, copy=True)
    b_o = np.array(block.attn_c_proj_b, dtype=np.float32, copy=True)
    w_fc = np.array(block.mlp_c_fc_w, dtype=np.float32, copy=True)
    b_fc = np.array(block.mlp_c_fc_b, dtype=np.float32, copy=True)
    w_proj = np.array(block.mlp_c_proj_w, dtype=np.float32, copy=True)
    b_proj = np.array(block.mlp_c_proj_b, dtype=np.float32, copy=True)

    x_norm1 = _layernorm_exact(x_in, ln1_wb, 1.0e-5)
    attn_heads: list[np.ndarray] = []
    q_heads: list[np.ndarray] = []
    k_cur_heads: list[np.ndarray] = []
    v_cur_heads: list[np.ndarray] = []
    scores_heads: list[np.ndarray] = []
    probs_heads: list[np.ndarray] = []
    scale = np.float32(1.0 / np.sqrt(float(d_head)))
    for head_idx in range(n_heads):
        q = (x_norm1 @ np.asarray(w_q[head_idx], dtype=np.float32) + np.asarray(b_q[head_idx], dtype=np.float32)).astype(np.float32)
        k_cur = (x_norm1 @ np.asarray(w_k[head_idx], dtype=np.float32) + np.asarray(b_k[head_idx], dtype=np.float32)).astype(np.float32)
        v_cur = (x_norm1 @ np.asarray(w_v[head_idx], dtype=np.float32) + np.asarray(b_v[head_idx], dtype=np.float32)).astype(np.float32)
        k_full = np.concatenate([np.asarray(prefill_ref["k_heads"][head_idx], dtype=np.float32), k_cur], axis=0).astype(np.float32)
        v_full = np.concatenate([np.asarray(prefill_ref["v_heads"][head_idx], dtype=np.float32), v_cur], axis=0).astype(np.float32)
        scores = ((q @ k_full.T) * scale).astype(np.float32)
        shifted = scores - np.max(scores, axis=-1, keepdims=True)
        probs = np.exp(shifted).astype(np.float32)
        probs /= np.sum(probs, axis=-1, keepdims=True, dtype=np.float32)
        attn = (probs @ v_full).astype(np.float32)
        q_heads.append(q)
        k_cur_heads.append(k_cur)
        v_cur_heads.append(v_cur)
        scores_heads.append(scores)
        probs_heads.append(probs)
        attn_heads.append(attn)

    attn_cat = np.concatenate(attn_heads, axis=-1).astype(np.float32) if n_heads > 1 else np.array(attn_heads[0], copy=True)
    o_f = (attn_cat @ w_o + b_o).astype(np.float32)
    resid1 = (x_in + o_f).astype(np.float32)
    x_norm2 = _layernorm_exact(resid1, ln2_wb, 1.0e-5)
    ffn_fc = (x_norm2 @ w_fc + b_fc).astype(np.float32)
    ffn_gelu = _gpt2_gelu_tanh(ffn_fc)
    ffn_out_f = (ffn_gelu @ w_proj + b_proj).astype(np.float32)
    out = (resid1 + ffn_out_f).astype(np.float32)
    return {
        "x_norm1": x_norm1,
        "q_heads": q_heads,
        "k_cur_heads": k_cur_heads,
        "v_cur_heads": v_cur_heads,
        "scores_heads": scores_heads,
        "probs_heads": probs_heads,
        "attn_cat": attn_cat,
        "o_f": o_f,
        "resid1": resid1,
        "x_norm2": x_norm2,
        "ffn_fc": ffn_fc,
        "ffn_gelu": ffn_gelu,
        "ffn_out_f": ffn_out_f,
        "out": out,
    }


def _common_io_tensors(
    *,
    block: QGPT2Block,
    x_in: np.ndarray,
    d_model: int,
    ffn_dim: int,
    attn_dim: int,
    seq_len: int,
    act_scale: float,
    out_name: str = "out",
) -> dict[str, TensorSpec]:
    return {
        "x_in": TensorSpec("x_in", (seq_len, d_model), DType.FLOAT32, TensorKind.CONSTANT, data=np.array(x_in, dtype=np.float32, copy=True)),
        "ln1_wb": TensorSpec("ln1_wb", (2, d_model), DType.FLOAT32, TensorKind.CONSTANT, data=np.array(block.ln_1_wb, dtype=np.float32, copy=True)),
        "ln2_wb": TensorSpec("ln2_wb", (2, d_model), DType.FLOAT32, TensorKind.CONSTANT, data=np.array(block.ln_2_wb, dtype=np.float32, copy=True)),
        "w_o": TensorSpec("w_o", (attn_dim, d_model), DType.INT16, TensorKind.CONSTANT, data=np.array(block.attn_c_proj_w, dtype=np.int16, copy=True)),
        "b_o": TensorSpec("b_o", (1, d_model), DType.INT32, TensorKind.CONSTANT, data=_quantize_bias_fp32(np.array(block.attn_c_proj_b, dtype=np.float32, copy=True), out_scale=act_scale)),
        "w_fc": TensorSpec("w_fc", (d_model, ffn_dim), DType.INT16, TensorKind.CONSTANT, data=np.array(block.mlp_c_fc_w, dtype=np.int16, copy=True)),
        "b_fc": TensorSpec("b_fc", (1, ffn_dim), DType.INT32, TensorKind.CONSTANT, data=_quantize_bias_fp32(np.array(block.mlp_c_fc_b, dtype=np.float32, copy=True), out_scale=act_scale)),
        "w_proj": TensorSpec("w_proj", (ffn_dim, d_model), DType.INT16, TensorKind.CONSTANT, data=np.array(block.mlp_c_proj_w, dtype=np.int16, copy=True)),
        "b_proj": TensorSpec("b_proj", (1, d_model), DType.INT32, TensorKind.CONSTANT, data=_quantize_bias_fp32(np.array(block.mlp_c_proj_b, dtype=np.float32, copy=True), out_scale=act_scale)),
        "x_norm1": TensorSpec("x_norm1", (seq_len, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        "x_norm1_q": TensorSpec("x_norm1_q", (seq_len, d_model), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}),
        "attn_cat": TensorSpec("attn_cat", (seq_len, attn_dim), DType.INT16, TensorKind.INTERMEDIATE),
        "attn_cat_a": TensorSpec("attn_cat_a", (seq_len, attn_dim), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}),
        "o_int": TensorSpec("o_int", (seq_len, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        "o_f": TensorSpec("o_f", (seq_len, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "resid1": TensorSpec("resid1", (seq_len, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "x_norm2": TensorSpec("x_norm2", (seq_len, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        "x_norm2_q": TensorSpec("x_norm2_q", (seq_len, d_model), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}),
        "ffn_fc_int": TensorSpec("ffn_fc_int", (seq_len, ffn_dim), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}),
        "ffn_out_int": TensorSpec("ffn_out_int", (seq_len, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        "ffn_out_f": TensorSpec("ffn_out_f", (seq_len, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        out_name: TensorSpec(out_name, (seq_len, d_model), DType.FLOAT32, TensorKind.OUTPUT, is_final_output=True),
    }


def build_prefill_artifact(
    *,
    d_model: int = 32,
    d_head: int = 8,
    n_heads: int = 4,
    ffn_dim: int = 128,
    prompt_len: int = 8,
    seed: int = 0,
    act_scale: float = 1.0 / 32.0,
    attn_scale: float = 1.0 / 256.0,
):
    if d_model <= 0 or d_model % 8 != 0:
        raise ValueError("d_model must be a positive multiple of 8")
    if d_head <= 0 or d_head % 8 != 0:
        raise ValueError("d_head must be a positive multiple of 8")
    if n_heads <= 0:
        raise ValueError("n_heads must be positive")
    if ffn_dim <= 0 or ffn_dim % 8 != 0:
        raise ValueError("ffn_dim must be a positive multiple of 8")
    if prompt_len <= 0 or prompt_len % 8 != 0:
        raise ValueError("prompt_len must be a positive multiple of 8")

    state = build_shared_state(
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        prompt_len=prompt_len,
        seed=seed,
    )
    ref = reference_prefill(
        state,
        d_head=d_head,
        n_heads=n_heads,
        act_scale=act_scale,
        attn_scale=attn_scale,
    )
    block = state["block"]

    attn_dim = n_heads * d_head
    tensors = _common_io_tensors(
        block=block,
        x_in=np.array(state["x_prompt_in"], dtype=np.float32, copy=True),
        d_model=d_model,
        ffn_dim=ffn_dim,
        attn_dim=attn_dim,
        seq_len=prompt_len,
        act_scale=act_scale,
    )

    q_ops: list[MatMulOp] = []
    kv_ops: list[MatMulOp] = []
    score_ops: list[MatMulOp] = []
    value_ops: list[MatMulOp] = []
    concat_steps: list[HostOp] = []
    prev_attn_name: str | None = None
    score_scale = _score_scale(d_head, (prompt_len, prompt_len), act_scale=act_scale)
    token_names = [f"t{i}" for i in range(prompt_len)]
    w_q, w_k, w_v = block.split_c_attn_weights()
    b_q_f, b_k_f, b_v_f = block.split_c_attn_biases_fp32()

    for head_idx in range(n_heads):
        tensors[f"w_q_h{head_idx}"] = TensorSpec(f"w_q_h{head_idx}", (d_model, d_head), DType.INT16, TensorKind.CONSTANT, data=np.array(w_q[head_idx], dtype=np.int16, copy=True))
        tensors[f"w_k_h{head_idx}"] = TensorSpec(f"w_k_h{head_idx}", (d_model, d_head), DType.INT16, TensorKind.CONSTANT, data=np.array(w_k[head_idx], dtype=np.int16, copy=True))
        tensors[f"w_v_h{head_idx}"] = TensorSpec(f"w_v_h{head_idx}", (d_model, d_head), DType.INT16, TensorKind.CONSTANT, data=np.array(w_v[head_idx], dtype=np.int16, copy=True))
        tensors[f"b_q_h{head_idx}"] = TensorSpec(f"b_q_h{head_idx}", (1, d_head), DType.INT32, TensorKind.CONSTANT, data=_quantize_bias_fp32(np.array(b_q_f[head_idx], dtype=np.float32, copy=True), out_scale=act_scale))
        tensors[f"b_k_h{head_idx}"] = TensorSpec(f"b_k_h{head_idx}", (1, d_head), DType.INT32, TensorKind.CONSTANT, data=_quantize_bias_fp32(np.array(b_k_f[head_idx], dtype=np.float32, copy=True), out_scale=act_scale))
        tensors[f"b_v_h{head_idx}"] = TensorSpec(f"b_v_h{head_idx}", (1, d_head), DType.INT32, TensorKind.CONSTANT, data=_quantize_bias_fp32(np.array(b_v_f[head_idx], dtype=np.float32, copy=True), out_scale=act_scale))
        tensors.update(
            make_native_int16_kv_cache_specs(
                k_base_name=f"prefill_k_cache_h{head_idx}",
                v_base_name=f"prefill_v_cache_h{head_idx}",
                d_head=d_head,
                token_capacity=prompt_len,
                token_names=token_names,
                token_indices=list(range(prompt_len)),
                kind=TensorKind.INTERMEDIATE,
            )
        )
        tensors[f"q_int_h{head_idx}"] = TensorSpec(f"q_int_h{head_idx}", (prompt_len, d_head), DType.INT16, TensorKind.INTERMEDIATE)
        tensors[f"k_seq_h{head_idx}"] = TensorSpec(f"k_seq_h{head_idx}", (prompt_len, d_head), DType.INT16, TensorKind.INTERMEDIATE)
        tensors[f"v_seq_h{head_idx}"] = TensorSpec(f"v_seq_h{head_idx}", (prompt_len, d_head), DType.INT16, TensorKind.INTERMEDIATE)
        tensors[f"q_a_h{head_idx}"] = TensorSpec(f"q_a_h{head_idx}", (prompt_len, d_head), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"})
        tensors[f"scores_h{head_idx}"] = TensorSpec(f"scores_h{head_idx}", (prompt_len, prompt_len), DType.INT16, TensorKind.INTERMEDIATE)
        tensors[f"score_scale_h{head_idx}"] = TensorSpec(f"score_scale_h{head_idx}", (prompt_len, prompt_len), DType.FLOAT32, TensorKind.CONSTANT, data=score_scale)
        tensors[f"scores_scaled_h{head_idx}"] = TensorSpec(f"scores_scaled_h{head_idx}", (prompt_len, prompt_len), DType.FLOAT32, TensorKind.INTERMEDIATE)
        tensors[f"masked_scores_h{head_idx}"] = TensorSpec(f"masked_scores_h{head_idx}", (prompt_len, prompt_len), DType.FLOAT32, TensorKind.INTERMEDIATE)
        tensors[f"probs_h{head_idx}"] = TensorSpec(f"probs_h{head_idx}", (prompt_len, prompt_len), DType.INT16, TensorKind.INTERMEDIATE)
        tensors[f"probs_q_h{head_idx}"] = TensorSpec(f"probs_q_h{head_idx}", (prompt_len, prompt_len), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"})
        tensors[f"attn_h{head_idx}"] = TensorSpec(f"attn_h{head_idx}", (prompt_len, d_head), DType.INT16, TensorKind.INTERMEDIATE)

        q_ops.append(MatMulOp(f"op_q_h{head_idx}", "x_norm1_q", f"w_q_h{head_idx}", f"q_int_h{head_idx}", bias=f"b_q_h{head_idx}", in_dtype=DType.INT16, out_dtype=DType.INT16))
        kv_ops.append(MatMulOp(f"op_k_h{head_idx}", "x_norm1_q", f"w_k_h{head_idx}", f"k_seq_h{head_idx}", bias=f"b_k_h{head_idx}", in_dtype=DType.INT16, out_dtype=DType.INT16))
        kv_ops.append(MatMulOp(f"op_v_h{head_idx}", "x_norm1_q", f"w_v_h{head_idx}", f"v_seq_h{head_idx}", bias=f"b_v_h{head_idx}", in_dtype=DType.INT16, out_dtype=DType.INT16))
        score_ops.append(MatMulOp(f"op_qk_h{head_idx}", f"q_a_h{head_idx}", f"prefill_k_cache_h{head_idx}", f"scores_h{head_idx}", in_dtype=DType.INT16, out_dtype=DType.INT16))
        value_ops.append(MatMulOp(f"op_av_h{head_idx}", f"probs_q_h{head_idx}", f"prefill_v_cache_h{head_idx}", f"attn_h{head_idx}", shift=8, in_dtype=DType.INT16, out_dtype=DType.INT16))

        if prev_attn_name is None:
            prev_attn_name = f"attn_h{head_idx}"
        else:
            cat_name = f"attn_cat_{head_idx}"
            tensors[cat_name] = TensorSpec(cat_name, (prompt_len, (head_idx + 1) * d_head), DType.INT16, TensorKind.INTERMEDIATE)
            concat_steps.append(HostOp(f"concat_attn_{head_idx}", "concat_lastdim2", inputs=[prev_attn_name, f"attn_h{head_idx}"], outputs=[cat_name]))
            prev_attn_name = cat_name

    steps: list[HostOp | NpuSegment] = [
        HostOp("layernorm1", "layernorm", inputs=["x_in", "ln1_wb"], outputs=["x_norm1"], attrs={"eps": 1.0e-5, "output_encoding": "fp16_bits"}),
        HostOp("quant_x_norm1", "quantize", inputs=["x_norm1"], outputs=["x_norm1_q"], attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16, "input_encoding": "fp16_bits"}),
        NpuSegment("seg_q", q_ops, inputs=["x_norm1_q"], outputs=[f"q_int_h{i}" for i in range(n_heads)]),
        NpuSegment("seg_kv_cache", kv_ops, inputs=["x_norm1_q"], outputs=[name for name in tensors if name.startswith(("k_seq_h", "v_seq_h"))]),
    ]
    for head_idx in range(n_heads):
        steps.append(HostOp(f"k_cache_scatter_matrix_h{head_idx}", "k_cache_scatter_matrix", inputs=[f"k_seq_h{head_idx}"], outputs=[f"prefill_k_cache_h{head_idx}"]))
        steps.append(HostOp(f"v_cache_scatter_matrix_h{head_idx}", "v_cache_scatter_matrix", inputs=[f"v_seq_h{head_idx}"], outputs=[f"prefill_v_cache_h{head_idx}"]))
        steps.append(HostOp(f"alias_q_a_h{head_idx}", "alias", inputs=[f"q_int_h{head_idx}"], outputs=[f"q_a_h{head_idx}"]))
    steps.append(NpuSegment("seg_score", score_ops, inputs=[f"q_a_h{i}" for i in range(n_heads)], outputs=[f"scores_h{i}" for i in range(n_heads)]))
    for head_idx in range(n_heads):
        steps.extend(
            [
                HostOp(f"scale_scores_h{head_idx}", "mul", inputs=[f"scores_h{head_idx}", f"score_scale_h{head_idx}"], outputs=[f"scores_scaled_h{head_idx}"]),
                HostOp(f"causal_mask_h{head_idx}", "causal_mask", inputs=[f"scores_scaled_h{head_idx}"], outputs=[f"masked_scores_h{head_idx}"], attrs={"past_kv_len": 0, "fill_value": -1.0e10}),
                HostOp(f"softmax_h{head_idx}", "softmax_f16", inputs=[f"masked_scores_h{head_idx}"], outputs=[f"probs_h{head_idx}"], attrs={"axis": -1}),
                HostOp(f"quant_probs_h{head_idx}", "quantize", inputs=[f"probs_h{head_idx}"], outputs=[f"probs_q_h{head_idx}"], attrs={"scale": attn_scale, "zero_point": 0, "dtype": DType.INT16, "input_encoding": "fp16_bits"}),
            ]
        )
    steps.append(NpuSegment("seg_value", value_ops, inputs=[f"probs_q_h{i}" for i in range(n_heads)], outputs=[f"attn_h{i}" for i in range(n_heads)]))
    steps.extend(concat_steps)
    if n_heads == 1:
        steps.append(HostOp("alias_attn_cat", "alias", inputs=[prev_attn_name or "attn_h0"], outputs=["attn_cat"]))
    elif prev_attn_name != "attn_cat":
        steps.append(HostOp("alias_attn_cat", "alias", inputs=[prev_attn_name or "attn_h0"], outputs=["attn_cat"]))
    steps.extend(
        [
            HostOp("alias_attn_cat_a", "alias", inputs=["attn_cat"], outputs=["attn_cat_a"]),
            NpuSegment("seg_o_proj", [MatMulOp("op_o_proj", "attn_cat_a", "w_o", "o_int", bias="b_o", in_dtype=DType.INT16, out_dtype=DType.INT16)], inputs=["attn_cat_a"], outputs=["o_int"]),
            HostOp("dequant_o", "dequantize", inputs=["o_int"], outputs=["o_f"], attrs={"scale": act_scale, "zero_point": 0}),
            HostOp("residual1", "add", inputs=["x_in", "o_f"], outputs=["resid1"]),
            HostOp("layernorm2", "layernorm", inputs=["resid1", "ln2_wb"], outputs=["x_norm2"], attrs={"eps": 1.0e-5, "output_encoding": "fp16_bits"}),
            HostOp("quant_x_norm2", "quantize", inputs=["x_norm2"], outputs=["x_norm2_q"], attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16, "input_encoding": "fp16_bits"}),
            NpuSegment("seg_ffn_fc", [MatMulOp("op_ffn_fc", "x_norm2_q", "w_fc", "ffn_fc_int", bias="b_fc", activation="h_gelu", in_dtype=DType.INT16, out_dtype=DType.INT16)], inputs=["x_norm2_q"], outputs=["ffn_fc_int"]),
            NpuSegment("seg_ffn_proj", [MatMulOp("op_ffn_proj", "ffn_fc_int", "w_proj", "ffn_out_int", bias="b_proj", in_dtype=DType.INT16, out_dtype=DType.INT16)], inputs=["ffn_fc_int"], outputs=["ffn_out_int"]),
            HostOp("dequant_ffn_out", "dequantize", inputs=["ffn_out_int"], outputs=["ffn_out_f"], attrs={"scale": act_scale, "zero_point": 0}),
            HostOp("residual2", "add", inputs=["resid1", "ffn_out_f"], outputs=["out"]),
        ]
    )

    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=[], outputs=["out"])
    plan.add_verification_step("out", "gpt2_prefill_out")
    artifact = compile_plan(plan, {"gpt2_prefill_out": np.array(ref["out"], dtype=np.float32, copy=True)})
    return artifact, state, ref


def build_decode_artifact(
    *,
    d_model: int = 32,
    d_head: int = 8,
    n_heads: int = 4,
    ffn_dim: int = 128,
    prompt_len: int = 8,
    seed: int = 0,
    act_scale: float = 1.0 / 32.0,
    attn_scale: float = 1.0 / 256.0,
):
    state = build_shared_state(
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        prompt_len=prompt_len,
        seed=seed,
    )
    prefill_ref = reference_prefill(
        state,
        d_head=d_head,
        n_heads=n_heads,
        act_scale=act_scale,
        attn_scale=attn_scale,
    )
    decode_ref = reference_decode(
        state,
        prefill_ref,
        d_head=d_head,
        n_heads=n_heads,
        act_scale=act_scale,
        attn_scale=attn_scale,
    )
    block = state["block"]

    cache_len = prompt_len + 1
    attn_dim = n_heads * d_head
    decode_token_name = "td"
    cache_token_names = [f"t{i}" for i in range(prompt_len)] + [decode_token_name]

    tensors = _common_io_tensors(
        block=block,
        x_in=np.array(state["x_decode_in"], dtype=np.float32, copy=True),
        d_model=d_model,
        ffn_dim=ffn_dim,
        attn_dim=attn_dim,
        seq_len=1,
        act_scale=act_scale,
    )

    qkv_ops: list[MatMulOp] = []
    score_ops: list[MatMulOp] = []
    value_ops: list[MatMulOp] = []
    concat_steps: list[HostOp] = []
    prev_attn_name: str | None = None
    score_scale = _score_scale(d_head, (1, cache_len), act_scale=act_scale)
    w_q, w_k, w_v = block.split_c_attn_weights()
    b_q_f, b_k_f, b_v_f = block.split_c_attn_biases_fp32()

    for head_idx in range(n_heads):
        tensors[f"w_q_h{head_idx}"] = TensorSpec(f"w_q_h{head_idx}", (d_model, d_head), DType.INT16, TensorKind.CONSTANT, data=np.array(w_q[head_idx], dtype=np.int16, copy=True))
        tensors[f"w_k_h{head_idx}"] = TensorSpec(f"w_k_h{head_idx}", (d_model, d_head), DType.INT16, TensorKind.CONSTANT, data=np.array(w_k[head_idx], dtype=np.int16, copy=True))
        tensors[f"w_v_h{head_idx}"] = TensorSpec(f"w_v_h{head_idx}", (d_model, d_head), DType.INT16, TensorKind.CONSTANT, data=np.array(w_v[head_idx], dtype=np.int16, copy=True))
        tensors[f"b_q_h{head_idx}"] = TensorSpec(f"b_q_h{head_idx}", (1, d_head), DType.INT32, TensorKind.CONSTANT, data=_quantize_bias_fp32(np.array(b_q_f[head_idx], dtype=np.float32, copy=True), out_scale=act_scale))
        tensors[f"b_k_h{head_idx}"] = TensorSpec(f"b_k_h{head_idx}", (1, d_head), DType.INT32, TensorKind.CONSTANT, data=_quantize_bias_fp32(np.array(b_k_f[head_idx], dtype=np.float32, copy=True), out_scale=act_scale))
        tensors[f"b_v_h{head_idx}"] = TensorSpec(f"b_v_h{head_idx}", (1, d_head), DType.INT32, TensorKind.CONSTANT, data=_quantize_bias_fp32(np.array(b_v_f[head_idx], dtype=np.float32, copy=True), out_scale=act_scale))

        head_cache_specs = make_native_int16_kv_cache_specs(
            k_base_name=f"k_cache_h{head_idx}",
            v_base_name=f"v_cache_h{head_idx}",
            d_head=d_head,
            token_capacity=cache_len,
            token_names=cache_token_names,
            token_indices=list(range(cache_len)),
            kind=TensorKind.INTERMEDIATE,
        )
        tensors.update(head_cache_specs)
        k_base = np.zeros((d_head, cache_len), dtype=np.int16)
        v_base = np.zeros((cache_len, d_head), dtype=np.int16)
        k_prefill = np.array(prefill_ref["k_heads"][head_idx], dtype=np.int16, copy=True)
        v_prefill = np.array(prefill_ref["v_heads"][head_idx], dtype=np.int16, copy=True)
        k_base[:, :prompt_len] = k_prefill.T
        v_base[:prompt_len, :] = v_prefill
        tensors[f"k_cache_h{head_idx}"].data = k_base
        tensors[f"v_cache_h{head_idx}"].data = v_base

        tensors[f"q_int_h{head_idx}"] = TensorSpec(f"q_int_h{head_idx}", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE)
        tensors[f"k_cur_h{head_idx}"] = TensorSpec(f"k_cur_h{head_idx}", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE)
        tensors[f"v_cur_h{head_idx}"] = TensorSpec(f"v_cur_h{head_idx}", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE)
        tensors[f"q_a_h{head_idx}"] = TensorSpec(f"q_a_h{head_idx}", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"})
        tensors[f"scores_h{head_idx}"] = TensorSpec(f"scores_h{head_idx}", (1, cache_len), DType.INT16, TensorKind.INTERMEDIATE)
        tensors[f"score_scale_h{head_idx}"] = TensorSpec(f"score_scale_h{head_idx}", (1, cache_len), DType.FLOAT32, TensorKind.CONSTANT, data=score_scale)
        tensors[f"scores_scaled_h{head_idx}"] = TensorSpec(f"scores_scaled_h{head_idx}", (1, cache_len), DType.FLOAT32, TensorKind.INTERMEDIATE)
        tensors[f"probs_h{head_idx}"] = TensorSpec(f"probs_h{head_idx}", (1, cache_len), DType.INT16, TensorKind.INTERMEDIATE)
        tensors[f"probs_q_h{head_idx}"] = TensorSpec(f"probs_q_h{head_idx}", (1, cache_len), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"})
        tensors[f"attn_h{head_idx}"] = TensorSpec(f"attn_h{head_idx}", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE)

        qkv_ops.extend(
            [
                MatMulOp(f"op_q_h{head_idx}", "x_norm1_q", f"w_q_h{head_idx}", f"q_int_h{head_idx}", bias=f"b_q_h{head_idx}", in_dtype=DType.INT16, out_dtype=DType.INT16),
                MatMulOp(f"op_k_h{head_idx}", "x_norm1_q", f"w_k_h{head_idx}", f"k_cur_h{head_idx}", bias=f"b_k_h{head_idx}", in_dtype=DType.INT16, out_dtype=DType.INT16),
                MatMulOp(f"op_v_h{head_idx}", "x_norm1_q", f"w_v_h{head_idx}", f"v_cur_h{head_idx}", bias=f"b_v_h{head_idx}", in_dtype=DType.INT16, out_dtype=DType.INT16),
            ]
        )
        score_ops.append(MatMulOp(f"op_qk_h{head_idx}", f"q_a_h{head_idx}", f"k_cache_h{head_idx}", f"scores_h{head_idx}", in_dtype=DType.INT16, out_dtype=DType.INT16))
        value_ops.append(MatMulOp(f"op_av_h{head_idx}", f"probs_q_h{head_idx}", f"v_cache_h{head_idx}", f"attn_h{head_idx}", shift=8, in_dtype=DType.INT16, out_dtype=DType.INT16))

        if prev_attn_name is None:
            prev_attn_name = f"attn_h{head_idx}"
        else:
            cat_name = f"attn_cat_{head_idx}"
            tensors[cat_name] = TensorSpec(cat_name, (1, (head_idx + 1) * d_head), DType.INT16, TensorKind.INTERMEDIATE)
            concat_steps.append(HostOp(f"concat_attn_{head_idx}", "concat_lastdim2", inputs=[prev_attn_name, f"attn_h{head_idx}"], outputs=[cat_name]))
            prev_attn_name = cat_name

    steps: list[HostOp | NpuSegment] = [
        HostOp("layernorm1", "layernorm", inputs=["x_in", "ln1_wb"], outputs=["x_norm1"], attrs={"eps": 1.0e-5, "output_encoding": "fp16_bits"}),
        HostOp("quant_x_norm1", "quantize", inputs=["x_norm1"], outputs=["x_norm1_q"], attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16, "input_encoding": "fp16_bits"}),
        NpuSegment("seg_qkv", qkv_ops, inputs=["x_norm1_q"], outputs=[name for name in tensors if name.startswith(("q_int_h", "k_cur_h", "v_cur_h"))]),
    ]
    for head_idx in range(n_heads):
        steps.extend(
            [
                HostOp(f"k_append_h{head_idx}", "k_cache_scatter_write", inputs=[f"k_cur_h{head_idx}"], outputs=[f"k_cache_h{head_idx}_{decode_token_name}"], attrs={"token_index": prompt_len, "k_cache_base": f"k_cache_h{head_idx}"}),
                HostOp(f"v_append_h{head_idx}", "v_cache_scatter_write", inputs=[f"v_cur_h{head_idx}"], outputs=[f"v_cache_h{head_idx}_{decode_token_name}"], attrs={"token_index": prompt_len, "v_cache_base": f"v_cache_h{head_idx}"}),
                HostOp(f"alias_q_a_h{head_idx}", "alias", inputs=[f"q_int_h{head_idx}"], outputs=[f"q_a_h{head_idx}"]),
            ]
        )
    steps.append(NpuSegment("seg_score", score_ops, inputs=[f"q_a_h{i}" for i in range(n_heads)], outputs=[f"scores_h{i}" for i in range(n_heads)]))
    for head_idx in range(n_heads):
        steps.extend(
            [
                HostOp(f"scale_scores_h{head_idx}", "mul", inputs=[f"scores_h{head_idx}", f"score_scale_h{head_idx}"], outputs=[f"scores_scaled_h{head_idx}"]),
                HostOp(f"softmax_h{head_idx}", "softmax_f16", inputs=[f"scores_scaled_h{head_idx}"], outputs=[f"probs_h{head_idx}"], attrs={"axis": -1}),
                HostOp(f"quant_probs_h{head_idx}", "quantize", inputs=[f"probs_h{head_idx}"], outputs=[f"probs_q_h{head_idx}"], attrs={"scale": attn_scale, "zero_point": 0, "dtype": DType.INT16, "input_encoding": "fp16_bits"}),
            ]
        )
    steps.append(NpuSegment("seg_value", value_ops, inputs=[f"probs_q_h{i}" for i in range(n_heads)], outputs=[f"attn_h{i}" for i in range(n_heads)]))
    steps.extend(concat_steps)
    if n_heads == 1:
        steps.append(HostOp("alias_attn_cat", "alias", inputs=[prev_attn_name or "attn_h0"], outputs=["attn_cat"]))
    elif prev_attn_name != "attn_cat":
        steps.append(HostOp("alias_attn_cat", "alias", inputs=[prev_attn_name or "attn_h0"], outputs=["attn_cat"]))
    steps.extend(
        [
            HostOp("alias_attn_cat_a", "alias", inputs=["attn_cat"], outputs=["attn_cat_a"]),
            NpuSegment("seg_o_proj", [MatMulOp("op_o_proj", "attn_cat_a", "w_o", "o_int", bias="b_o", in_dtype=DType.INT16, out_dtype=DType.INT16)], inputs=["attn_cat_a"], outputs=["o_int"]),
            HostOp("dequant_o", "dequantize", inputs=["o_int"], outputs=["o_f"], attrs={"scale": act_scale, "zero_point": 0}),
            HostOp("residual1", "add", inputs=["x_in", "o_f"], outputs=["resid1"]),
            HostOp("layernorm2", "layernorm", inputs=["resid1", "ln2_wb"], outputs=["x_norm2"], attrs={"eps": 1.0e-5, "output_encoding": "fp16_bits"}),
            HostOp("quant_x_norm2", "quantize", inputs=["x_norm2"], outputs=["x_norm2_q"], attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16, "input_encoding": "fp16_bits"}),
            NpuSegment("seg_ffn_fc", [MatMulOp("op_ffn_fc", "x_norm2_q", "w_fc", "ffn_fc_int", bias="b_fc", activation="h_gelu", in_dtype=DType.INT16, out_dtype=DType.INT16)], inputs=["x_norm2_q"], outputs=["ffn_fc_int"]),
            NpuSegment("seg_ffn_proj", [MatMulOp("op_ffn_proj", "ffn_fc_int", "w_proj", "ffn_out_int", bias="b_proj", in_dtype=DType.INT16, out_dtype=DType.INT16)], inputs=["ffn_fc_int"], outputs=["ffn_out_int"]),
            HostOp("dequant_ffn_out", "dequantize", inputs=["ffn_out_int"], outputs=["ffn_out_f"], attrs={"scale": act_scale, "zero_point": 0}),
            HostOp("residual2", "add", inputs=["resid1", "ffn_out_f"], outputs=["out"]),
        ]
    )

    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=[], outputs=["out"])
    plan.add_verification_step("out", "gpt2_decode_out")
    artifact = compile_plan(plan, {"gpt2_decode_out": np.array(decode_ref["out"], dtype=np.float32, copy=True)})
    return artifact, state, prefill_ref, decode_ref
