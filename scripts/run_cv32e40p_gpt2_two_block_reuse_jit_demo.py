from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))

from tinynpu_jit import DType, IRBuilder, TensorKind, TensorSpec, compile_plan, emit_cv32e40p_program_v2, make_b_cache_view_spec  # noqa: E402
from tinynpu_jit.blocks.gpt2_block import (  # noqa: E402
    QGPT2Block,
    build_shared_state,
    extend_kv_cache,
    make_native_int16_kv_cache_specs,
    reference_decode,
    reference_prefill,
)
from run_cv32e40p_b_append_demo import (  # noqa: E402
    GENERATED_DIR,
    RunnerConfig,
    build_v2_elf_and_hex,
)


def _quantize_bias_fp32(bias: np.ndarray, *, out_scale: float) -> np.ndarray:
    return np.rint(np.asarray(bias, dtype=np.float32) / np.float32(out_scale)).astype(np.int32)


def _score_scale(d_head: int, shape: tuple[int, int], *, act_scale: float) -> np.ndarray:
    score_scale = np.float32((float(act_scale) * float(act_scale)) / np.sqrt(float(d_head)))
    return np.full(shape, score_scale, dtype=np.float32)


def _common_tensors(
    *,
    prefix: str,
    block: QGPT2Block,
    x_name: str,
    d_model: int,
    ffn_dim: int,
    attn_dim: int,
    seq_len: int,
    act_scale: float,
) -> dict[str, TensorSpec]:
    return {
        f"{prefix}_ln1_wb": TensorSpec(f"{prefix}_ln1_wb", (2, d_model), DType.FLOAT32, TensorKind.CONSTANT, data=np.array(block.ln_1_wb, dtype=np.float32, copy=True)),
        f"{prefix}_ln2_wb": TensorSpec(f"{prefix}_ln2_wb", (2, d_model), DType.FLOAT32, TensorKind.CONSTANT, data=np.array(block.ln_2_wb, dtype=np.float32, copy=True)),
        f"{prefix}_w_o": TensorSpec(f"{prefix}_w_o", (attn_dim, d_model), DType.INT16, TensorKind.CONSTANT, data=np.array(block.attn_c_proj_w, dtype=np.int16, copy=True)),
        f"{prefix}_b_o": TensorSpec(f"{prefix}_b_o", (1, d_model), DType.INT32, TensorKind.CONSTANT, data=_quantize_bias_fp32(np.array(block.attn_c_proj_b, dtype=np.float32, copy=True), out_scale=act_scale)),
        f"{prefix}_w_fc": TensorSpec(f"{prefix}_w_fc", (d_model, ffn_dim), DType.INT16, TensorKind.CONSTANT, data=np.array(block.mlp_c_fc_w, dtype=np.int16, copy=True)),
        f"{prefix}_b_fc": TensorSpec(f"{prefix}_b_fc", (1, ffn_dim), DType.INT32, TensorKind.CONSTANT, data=_quantize_bias_fp32(np.array(block.mlp_c_fc_b, dtype=np.float32, copy=True), out_scale=act_scale)),
        f"{prefix}_w_proj": TensorSpec(f"{prefix}_w_proj", (ffn_dim, d_model), DType.INT16, TensorKind.CONSTANT, data=np.array(block.mlp_c_proj_w, dtype=np.int16, copy=True)),
        f"{prefix}_b_proj": TensorSpec(f"{prefix}_b_proj", (1, d_model), DType.INT32, TensorKind.CONSTANT, data=_quantize_bias_fp32(np.array(block.mlp_c_proj_b, dtype=np.float32, copy=True), out_scale=act_scale)),
        f"{prefix}_x_norm1": TensorSpec(f"{prefix}_x_norm1", (seq_len, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        f"{prefix}_x_norm1_q": TensorSpec(f"{prefix}_x_norm1_q", (seq_len, d_model), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}),
        f"{prefix}_attn_cat": TensorSpec(f"{prefix}_attn_cat", (seq_len, attn_dim), DType.INT16, TensorKind.INTERMEDIATE),
        f"{prefix}_attn_cat_a": TensorSpec(f"{prefix}_attn_cat_a", (seq_len, attn_dim), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}),
        f"{prefix}_o_int": TensorSpec(f"{prefix}_o_int", (seq_len, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        f"{prefix}_o_f": TensorSpec(f"{prefix}_o_f", (seq_len, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        f"{prefix}_resid1": TensorSpec(f"{prefix}_resid1", (seq_len, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        f"{prefix}_x_norm2": TensorSpec(f"{prefix}_x_norm2", (seq_len, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        f"{prefix}_x_norm2_q": TensorSpec(f"{prefix}_x_norm2_q", (seq_len, d_model), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}),
        f"{prefix}_ffn_fc_int": TensorSpec(f"{prefix}_ffn_fc_int", (seq_len, ffn_dim), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}),
        f"{prefix}_ffn_out_int": TensorSpec(f"{prefix}_ffn_out_int", (seq_len, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        f"{prefix}_ffn_out_f": TensorSpec(f"{prefix}_ffn_out_f", (seq_len, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        f"{prefix}_out": TensorSpec(f"{prefix}_out", (seq_len, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE if prefix != "d2_b1" else TensorKind.OUTPUT, is_final_output=(prefix == "d2_b1")),
    }


def _block_projection_tensors(builder: IRBuilder, *, prefix: str, block: QGPT2Block, d_model: int, d_head: int, act_scale: float) -> None:
    w_q, w_k, w_v = block.split_c_attn_weights()
    b_q_f, b_k_f, b_v_f = block.split_c_attn_biases_fp32()
    builder.add_tensor(TensorSpec(f"{prefix}_w_q", (d_model, d_head), DType.INT16, TensorKind.CONSTANT, data=np.array(w_q[0], dtype=np.int16, copy=True)))
    builder.add_tensor(TensorSpec(f"{prefix}_w_k", (d_model, d_head), DType.INT16, TensorKind.CONSTANT, data=np.array(w_k[0], dtype=np.int16, copy=True)))
    builder.add_tensor(TensorSpec(f"{prefix}_w_v", (d_model, d_head), DType.INT16, TensorKind.CONSTANT, data=np.array(w_v[0], dtype=np.int16, copy=True)))
    builder.add_tensor(TensorSpec(f"{prefix}_b_q", (1, d_head), DType.INT32, TensorKind.CONSTANT, data=_quantize_bias_fp32(np.array(b_q_f[0], dtype=np.float32, copy=True), out_scale=act_scale)))
    builder.add_tensor(TensorSpec(f"{prefix}_b_k", (1, d_head), DType.INT32, TensorKind.CONSTANT, data=_quantize_bias_fp32(np.array(b_k_f[0], dtype=np.float32, copy=True), out_scale=act_scale)))
    builder.add_tensor(TensorSpec(f"{prefix}_b_v", (1, d_head), DType.INT32, TensorKind.CONSTANT, data=_quantize_bias_fp32(np.array(b_v_f[0], dtype=np.float32, copy=True), out_scale=act_scale)))


def _add_prefill_block(
    builder: IRBuilder,
    *,
    prefix: str,
    block: QGPT2Block,
    x_name: str,
    d_model: int,
    d_head: int,
    ffn_dim: int,
    prompt_len: int,
    act_scale: float,
    attn_scale: float,
    cache_prefix: str,
    cache_capacity: int,
    token_names: list[str],
) -> None:
    attn_dim = d_head
    for spec in _common_tensors(prefix=prefix, block=block, x_name=x_name, d_model=d_model, ffn_dim=ffn_dim, attn_dim=attn_dim, seq_len=prompt_len, act_scale=act_scale).values():
        builder.add_tensor(spec)
    _block_projection_tensors(builder, prefix=prefix, block=block, d_model=d_model, d_head=d_head, act_scale=act_scale)
    for spec in make_native_int16_kv_cache_specs(
        k_base_name=f"{cache_prefix}_k_cache",
        v_base_name=f"{cache_prefix}_v_cache",
        d_head=d_head,
        token_capacity=cache_capacity,
        token_names=token_names,
        token_indices=list(range(cache_capacity)),
        kind=TensorKind.INTERMEDIATE,
    ).values():
        if spec.name not in builder.tensors:
            builder.add_tensor(spec)
    if f"{cache_prefix}_k_cache_prompt" not in builder.tensors:
        builder.add_tensor(make_b_cache_view_spec(f"{cache_prefix}_k_cache_prompt", f"{cache_prefix}_k_cache", (d_head, prompt_len), DType.INT16, kind=TensorKind.INTERMEDIATE, word_offset=0))
    if f"{cache_prefix}_v_cache_prompt" not in builder.tensors:
        builder.add_tensor(make_b_cache_view_spec(f"{cache_prefix}_v_cache_prompt", f"{cache_prefix}_v_cache", (prompt_len, d_head), DType.INT16, kind=TensorKind.INTERMEDIATE, word_offset=0))
    score_scale = _score_scale(d_head, (prompt_len, prompt_len), act_scale=act_scale)
    builder.add_tensor(TensorSpec(f"{prefix}_q_int", (prompt_len, d_head), DType.INT16, TensorKind.INTERMEDIATE))
    builder.add_tensor(TensorSpec(f"{prefix}_k_seq", (prompt_len, d_head), DType.INT16, TensorKind.INTERMEDIATE))
    builder.add_tensor(TensorSpec(f"{prefix}_v_seq", (prompt_len, d_head), DType.INT16, TensorKind.INTERMEDIATE))
    builder.add_tensor(TensorSpec(f"{prefix}_q_a", (prompt_len, d_head), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}))
    builder.add_tensor(TensorSpec(f"{prefix}_scores", (prompt_len, prompt_len), DType.INT16, TensorKind.INTERMEDIATE))
    builder.add_tensor(TensorSpec(f"{prefix}_score_scale", (prompt_len, prompt_len), DType.FLOAT32, TensorKind.CONSTANT, data=score_scale))
    builder.add_tensor(TensorSpec(f"{prefix}_scores_scaled", (prompt_len, prompt_len), DType.FLOAT32, TensorKind.INTERMEDIATE))
    builder.add_tensor(TensorSpec(f"{prefix}_masked_scores", (prompt_len, prompt_len), DType.FLOAT32, TensorKind.INTERMEDIATE))
    builder.add_tensor(TensorSpec(f"{prefix}_probs_h", (prompt_len, prompt_len), DType.INT16, TensorKind.INTERMEDIATE))
    builder.add_tensor(TensorSpec(f"{prefix}_probs_q", (prompt_len, prompt_len), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}))
    builder.add_tensor(TensorSpec(f"{prefix}_attn", (prompt_len, d_head), DType.INT16, TensorKind.INTERMEDIATE))

    builder.host(f"{prefix}_layernorm1", "layernorm", inputs=[x_name, f"{prefix}_ln1_wb"], outputs=[f"{prefix}_x_norm1"], attrs={"eps": 1.0e-5, "output_encoding": "fp16_bits"})
    builder.host(f"{prefix}_quant_x_norm1", "quantize", inputs=[f"{prefix}_x_norm1"], outputs=[f"{prefix}_x_norm1_q"], attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16, "input_encoding": "fp16_bits"})
    builder.segment(f"{prefix}_seg_q", ops=[builder.matmul(f"{prefix}_op_q", f"{prefix}_x_norm1_q", f"{prefix}_w_q", f"{prefix}_q_int", bias=f"{prefix}_b_q", in_dtype=DType.INT16, out_dtype=DType.INT16)], inputs=[f"{prefix}_x_norm1_q"], outputs=[f"{prefix}_q_int"])
    builder.segment(
        f"{prefix}_seg_kv_cache",
        ops=[
            builder.matmul(f"{prefix}_op_k", f"{prefix}_x_norm1_q", f"{prefix}_w_k", f"{prefix}_k_seq", bias=f"{prefix}_b_k", in_dtype=DType.INT16, out_dtype=DType.INT16),
            builder.matmul(f"{prefix}_op_v", f"{prefix}_x_norm1_q", f"{prefix}_w_v", f"{prefix}_v_seq", bias=f"{prefix}_b_v", in_dtype=DType.INT16, out_dtype=DType.INT16),
        ],
        inputs=[f"{prefix}_x_norm1_q"],
        outputs=[f"{prefix}_k_seq", f"{prefix}_v_seq"],
    )
    builder.host(f"{prefix}_k_cache_scatter", "k_cache_scatter_matrix", inputs=[f"{prefix}_k_seq"], outputs=[f"{cache_prefix}_k_cache"])
    builder.host(f"{prefix}_v_cache_scatter", "v_cache_scatter_matrix", inputs=[f"{prefix}_v_seq"], outputs=[f"{cache_prefix}_v_cache"])
    builder.host(f"{prefix}_alias_q_a", "alias", inputs=[f"{prefix}_q_int"], outputs=[f"{prefix}_q_a"])
    builder.segment(
        f"{prefix}_seg_score",
        ops=[builder.matmul(f"{prefix}_op_qk", f"{prefix}_q_a", f"{cache_prefix}_k_cache_prompt", f"{prefix}_scores", in_dtype=DType.INT16, out_dtype=DType.INT16)],
        inputs=[f"{prefix}_q_a"],
        outputs=[f"{prefix}_scores"],
    )
    builder.host(f"{prefix}_scale_scores", "mul", inputs=[f"{prefix}_scores", f"{prefix}_score_scale"], outputs=[f"{prefix}_scores_scaled"])
    builder.host(f"{prefix}_causal_mask", "causal_mask", inputs=[f"{prefix}_scores_scaled"], outputs=[f"{prefix}_masked_scores"], attrs={"past_kv_len": 0, "fill_value": -1.0e10})
    builder.host(f"{prefix}_softmax", "softmax_f16", inputs=[f"{prefix}_masked_scores"], outputs=[f"{prefix}_probs_h"], attrs={"axis": -1})
    builder.host(f"{prefix}_quant_probs", "quantize", inputs=[f"{prefix}_probs_h"], outputs=[f"{prefix}_probs_q"], attrs={"scale": attn_scale, "zero_point": 0, "dtype": DType.INT16, "input_encoding": "fp16_bits"})
    builder.segment(
        f"{prefix}_seg_value",
        ops=[builder.matmul(f"{prefix}_op_av", f"{prefix}_probs_q", f"{cache_prefix}_v_cache_prompt", f"{prefix}_attn", shift=8, in_dtype=DType.INT16, out_dtype=DType.INT16)],
        inputs=[f"{prefix}_probs_q"],
        outputs=[f"{prefix}_attn"],
    )
    builder.host(f"{prefix}_alias_attn_cat", "alias", inputs=[f"{prefix}_attn"], outputs=[f"{prefix}_attn_cat"])
    builder.host(f"{prefix}_alias_attn_cat_a", "alias", inputs=[f"{prefix}_attn_cat"], outputs=[f"{prefix}_attn_cat_a"])
    _append_tail(builder, prefix=prefix, x_name=x_name, act_scale=act_scale)


def _append_tail(builder: IRBuilder, *, prefix: str, x_name: str, act_scale: float) -> None:
    builder.segment(f"{prefix}_seg_o_proj", ops=[builder.matmul(f"{prefix}_op_o_proj", f"{prefix}_attn_cat_a", f"{prefix}_w_o", f"{prefix}_o_int", bias=f"{prefix}_b_o", in_dtype=DType.INT16, out_dtype=DType.INT16)], inputs=[f"{prefix}_attn_cat_a"], outputs=[f"{prefix}_o_int"])
    builder.host(f"{prefix}_dequant_o", "dequantize", inputs=[f"{prefix}_o_int"], outputs=[f"{prefix}_o_f"], attrs={"scale": act_scale, "zero_point": 0})
    builder.host(f"{prefix}_residual1", "add", inputs=[x_name, f"{prefix}_o_f"], outputs=[f"{prefix}_resid1"])
    builder.host(f"{prefix}_layernorm2", "layernorm", inputs=[f"{prefix}_resid1", f"{prefix}_ln2_wb"], outputs=[f"{prefix}_x_norm2"], attrs={"eps": 1.0e-5, "output_encoding": "fp16_bits"})
    builder.host(f"{prefix}_quant_x_norm2", "quantize", inputs=[f"{prefix}_x_norm2"], outputs=[f"{prefix}_x_norm2_q"], attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16, "input_encoding": "fp16_bits"})
    builder.segment(f"{prefix}_seg_ffn_fc", ops=[builder.matmul(f"{prefix}_op_ffn_fc", f"{prefix}_x_norm2_q", f"{prefix}_w_fc", f"{prefix}_ffn_fc_int", bias=f"{prefix}_b_fc", activation="h_gelu", in_dtype=DType.INT16, out_dtype=DType.INT16)], inputs=[f"{prefix}_x_norm2_q"], outputs=[f"{prefix}_ffn_fc_int"])
    builder.segment(f"{prefix}_seg_ffn_proj", ops=[builder.matmul(f"{prefix}_op_ffn_proj", f"{prefix}_ffn_fc_int", f"{prefix}_w_proj", f"{prefix}_ffn_out_int", bias=f"{prefix}_b_proj", in_dtype=DType.INT16, out_dtype=DType.INT16)], inputs=[f"{prefix}_ffn_fc_int"], outputs=[f"{prefix}_ffn_out_int"])
    builder.host(f"{prefix}_dequant_ffn_out", "dequantize", inputs=[f"{prefix}_ffn_out_int"], outputs=[f"{prefix}_ffn_out_f"], attrs={"scale": act_scale, "zero_point": 0})
    builder.host(f"{prefix}_residual2", "add", inputs=[f"{prefix}_resid1", f"{prefix}_ffn_out_f"], outputs=[f"{prefix}_out"])


def _add_decode_block(
    builder: IRBuilder,
    *,
    prefix: str,
    block: QGPT2Block,
    x_name: str,
    d_model: int,
    d_head: int,
    ffn_dim: int,
    cache_len: int,
    decode_token_name: str,
    act_scale: float,
    attn_scale: float,
    cache_prefix: str,
) -> None:
    attn_dim = d_head
    for spec in _common_tensors(prefix=prefix, block=block, x_name=x_name, d_model=d_model, ffn_dim=ffn_dim, attn_dim=attn_dim, seq_len=1, act_scale=act_scale).values():
        builder.add_tensor(spec)
    _block_projection_tensors(builder, prefix=prefix, block=block, d_model=d_model, d_head=d_head, act_scale=act_scale)
    score_scale = _score_scale(d_head, (1, cache_len), act_scale=act_scale)
    if f"{cache_prefix}_k_cache_l{cache_len}" not in builder.tensors:
        builder.add_tensor(make_b_cache_view_spec(f"{cache_prefix}_k_cache_l{cache_len}", f"{cache_prefix}_k_cache", (d_head, cache_len), DType.INT16, kind=TensorKind.INTERMEDIATE, word_offset=0))
    if f"{cache_prefix}_v_cache_l{cache_len}" not in builder.tensors:
        builder.add_tensor(make_b_cache_view_spec(f"{cache_prefix}_v_cache_l{cache_len}", f"{cache_prefix}_v_cache", (cache_len, d_head), DType.INT16, kind=TensorKind.INTERMEDIATE, word_offset=0))
    builder.add_tensor(TensorSpec(f"{prefix}_q_int", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE))
    builder.add_tensor(TensorSpec(f"{prefix}_k_cur", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE))
    builder.add_tensor(TensorSpec(f"{prefix}_v_cur", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE))
    builder.add_tensor(TensorSpec(f"{prefix}_q_a", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}))
    builder.add_tensor(TensorSpec(f"{prefix}_scores", (1, cache_len), DType.INT16, TensorKind.INTERMEDIATE))
    builder.add_tensor(TensorSpec(f"{prefix}_score_scale", (1, cache_len), DType.FLOAT32, TensorKind.CONSTANT, data=score_scale))
    builder.add_tensor(TensorSpec(f"{prefix}_scores_scaled", (1, cache_len), DType.FLOAT32, TensorKind.INTERMEDIATE))
    builder.add_tensor(TensorSpec(f"{prefix}_probs_h", (1, cache_len), DType.INT16, TensorKind.INTERMEDIATE))
    builder.add_tensor(TensorSpec(f"{prefix}_probs_q", (1, cache_len), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}))
    builder.add_tensor(TensorSpec(f"{prefix}_attn", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE))

    builder.host(f"{prefix}_layernorm1", "layernorm", inputs=[x_name, f"{prefix}_ln1_wb"], outputs=[f"{prefix}_x_norm1"], attrs={"eps": 1.0e-5, "output_encoding": "fp16_bits"})
    builder.host(f"{prefix}_quant_x_norm1", "quantize", inputs=[f"{prefix}_x_norm1"], outputs=[f"{prefix}_x_norm1_q"], attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16, "input_encoding": "fp16_bits"})
    builder.segment(
        f"{prefix}_seg_qkv",
        ops=[
            builder.matmul(f"{prefix}_op_q", f"{prefix}_x_norm1_q", f"{prefix}_w_q", f"{prefix}_q_int", bias=f"{prefix}_b_q", in_dtype=DType.INT16, out_dtype=DType.INT16),
            builder.matmul(f"{prefix}_op_k", f"{prefix}_x_norm1_q", f"{prefix}_w_k", f"{prefix}_k_cur", bias=f"{prefix}_b_k", in_dtype=DType.INT16, out_dtype=DType.INT16),
            builder.matmul(f"{prefix}_op_v", f"{prefix}_x_norm1_q", f"{prefix}_w_v", f"{prefix}_v_cur", bias=f"{prefix}_b_v", in_dtype=DType.INT16, out_dtype=DType.INT16),
        ],
        inputs=[f"{prefix}_x_norm1_q"],
        outputs=[f"{prefix}_q_int", f"{prefix}_k_cur", f"{prefix}_v_cur"],
    )
    builder.host(f"{prefix}_k_append", "k_cache_scatter_write", inputs=[f"{prefix}_k_cur"], outputs=[f"{cache_prefix}_k_cache_{decode_token_name}"], attrs={"token_index": cache_len - 1, "k_cache_base": f"{cache_prefix}_k_cache"})
    builder.host(f"{prefix}_v_append", "v_cache_scatter_write", inputs=[f"{prefix}_v_cur"], outputs=[f"{cache_prefix}_v_cache_{decode_token_name}"], attrs={"token_index": cache_len - 1, "v_cache_base": f"{cache_prefix}_v_cache"})
    builder.host(f"{prefix}_alias_q_a", "alias", inputs=[f"{prefix}_q_int"], outputs=[f"{prefix}_q_a"])
    builder.segment(f"{prefix}_seg_score", ops=[builder.matmul(f"{prefix}_op_qk", f"{prefix}_q_a", f"{cache_prefix}_k_cache_l{cache_len}", f"{prefix}_scores", in_dtype=DType.INT16, out_dtype=DType.INT16)], inputs=[f"{prefix}_q_a"], outputs=[f"{prefix}_scores"])
    builder.host(f"{prefix}_scale_scores", "mul", inputs=[f"{prefix}_scores", f"{prefix}_score_scale"], outputs=[f"{prefix}_scores_scaled"])
    builder.host(f"{prefix}_softmax", "softmax_f16", inputs=[f"{prefix}_scores_scaled"], outputs=[f"{prefix}_probs_h"], attrs={"axis": -1})
    builder.host(f"{prefix}_quant_probs", "quantize", inputs=[f"{prefix}_probs_h"], outputs=[f"{prefix}_probs_q"], attrs={"scale": attn_scale, "zero_point": 0, "dtype": DType.INT16, "input_encoding": "fp16_bits"})
    builder.segment(f"{prefix}_seg_value", ops=[builder.matmul(f"{prefix}_op_av", f"{prefix}_probs_q", f"{cache_prefix}_v_cache_l{cache_len}", f"{prefix}_attn", shift=8, in_dtype=DType.INT16, out_dtype=DType.INT16)], inputs=[f"{prefix}_probs_q"], outputs=[f"{prefix}_attn"])
    builder.host(f"{prefix}_alias_attn_cat", "alias", inputs=[f"{prefix}_attn"], outputs=[f"{prefix}_attn_cat"])
    builder.host(f"{prefix}_alias_attn_cat_a", "alias", inputs=[f"{prefix}_attn_cat"], outputs=[f"{prefix}_attn_cat_a"])
    _append_tail(builder, prefix=prefix, x_name=x_name, act_scale=act_scale)


def build_artifact(
    *,
    d_model: int = 8,
    d_head: int = 8,
    n_heads: int = 1,
    ffn_dim: int = 8,
    prompt_len: int = 8,
):
    act_scale = 1.0 / 32.0
    attn_scale = 1.0 / 256.0
    cache_capacity = prompt_len + 2
    token_names = [f"t{i}" for i in range(prompt_len)] + ["td1", "td2"]

    layer0 = build_shared_state(d_model=d_model, d_head=d_head, n_heads=n_heads, ffn_dim=ffn_dim, prompt_len=prompt_len, seed=0)
    layer1 = build_shared_state(d_model=d_model, d_head=d_head, n_heads=n_heads, ffn_dim=ffn_dim, prompt_len=prompt_len, seed=1)
    prompt_x0 = np.asarray(layer0["x_prompt_in"], dtype=np.float32)
    decode1_x0 = np.asarray(layer0["x_decode_in"], dtype=np.float32)
    decode2_x0 = np.random.default_rng(123).uniform(-0.25, 0.25, size=(1, d_model)).astype(np.float32)

    p0_ref = reference_prefill(layer0, d_head=d_head, n_heads=n_heads, act_scale=act_scale, attn_scale=attn_scale, x_in=prompt_x0)
    p1_ref = reference_prefill(layer1, d_head=d_head, n_heads=n_heads, act_scale=act_scale, attn_scale=attn_scale, x_in=np.asarray(p0_ref["out"], dtype=np.float32))
    d10_ref = reference_decode(layer0, p0_ref, d_head=d_head, n_heads=n_heads, act_scale=act_scale, attn_scale=attn_scale, x_in=decode1_x0)
    d11_ref = reference_decode(layer1, p1_ref, d_head=d_head, n_heads=n_heads, act_scale=act_scale, attn_scale=attn_scale, x_in=np.asarray(d10_ref["out"], dtype=np.float32))
    p0_cache1 = extend_kv_cache(p0_ref, d10_ref)
    p1_cache1 = extend_kv_cache(p1_ref, d11_ref)
    d20_ref = reference_decode(layer0, p0_cache1, d_head=d_head, n_heads=n_heads, act_scale=act_scale, attn_scale=attn_scale, x_in=decode2_x0)
    d21_ref = reference_decode(layer1, p1_cache1, d_head=d_head, n_heads=n_heads, act_scale=act_scale, attn_scale=attn_scale, x_in=np.asarray(d20_ref["out"], dtype=np.float32))

    b = IRBuilder()
    b.add_tensor(TensorSpec("prompt_x0", prompt_x0.shape, DType.FLOAT32, TensorKind.CONSTANT, data=prompt_x0))
    b.add_tensor(TensorSpec("decode1_x0", decode1_x0.shape, DType.FLOAT32, TensorKind.CONSTANT, data=decode1_x0))
    b.add_tensor(TensorSpec("decode2_x0", decode2_x0.shape, DType.FLOAT32, TensorKind.CONSTANT, data=decode2_x0))

    _add_prefill_block(b, prefix="p0_b0", block=layer0["block"], x_name="prompt_x0", d_model=d_model, d_head=d_head, ffn_dim=ffn_dim, prompt_len=prompt_len, act_scale=act_scale, attn_scale=attn_scale, cache_prefix="b0", cache_capacity=cache_capacity, token_names=token_names)
    _add_prefill_block(b, prefix="p0_b1", block=layer1["block"], x_name="p0_b0_out", d_model=d_model, d_head=d_head, ffn_dim=ffn_dim, prompt_len=prompt_len, act_scale=act_scale, attn_scale=attn_scale, cache_prefix="b1", cache_capacity=cache_capacity, token_names=token_names)
    _add_decode_block(b, prefix="d1_b0", block=layer0["block"], x_name="decode1_x0", d_model=d_model, d_head=d_head, ffn_dim=ffn_dim, cache_len=prompt_len + 1, decode_token_name="td1", act_scale=act_scale, attn_scale=attn_scale, cache_prefix="b0")
    _add_decode_block(b, prefix="d1_b1", block=layer1["block"], x_name="d1_b0_out", d_model=d_model, d_head=d_head, ffn_dim=ffn_dim, cache_len=prompt_len + 1, decode_token_name="td1", act_scale=act_scale, attn_scale=attn_scale, cache_prefix="b1")
    _add_decode_block(b, prefix="d2_b0", block=layer0["block"], x_name="decode2_x0", d_model=d_model, d_head=d_head, ffn_dim=ffn_dim, cache_len=prompt_len + 2, decode_token_name="td2", act_scale=act_scale, attn_scale=attn_scale, cache_prefix="b0")
    _add_decode_block(b, prefix="d2_b1", block=layer1["block"], x_name="d2_b0_out", d_model=d_model, d_head=d_head, ffn_dim=ffn_dim, cache_len=prompt_len + 2, decode_token_name="td2", act_scale=act_scale, attn_scale=attn_scale, cache_prefix="b1")

    plan = b.finalize(inputs=[], outputs=["d2_b1_out"])
    plan.add_verification_step("d2_b1_out", "gpt2_two_block_reuse_out")
    artifact = compile_plan(plan, {"gpt2_two_block_reuse_out": np.asarray(d21_ref["out"], dtype=np.float32)})
    return artifact


def _emit_and_build(artifact, *, program_name: str) -> Path:
    GENERATED_DIR.mkdir(exist_ok=True)
    program_path = GENERATED_DIR / f"{program_name}_program.c"
    source = emit_cv32e40p_program_v2(artifact, {}, program_name=program_name)
    _, _, _, hex_path = build_v2_elf_and_hex(
        program_name,
        source,
        runner_config=RunnerConfig(
            repeat_count=1,
            dump_final_outputs=True,
            verbose_steps=True,
        ),
    )
    program_path.write_text(source)
    return hex_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--d-head", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--ffn-dim", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=8)
    args = parser.parse_args()

    artifact = build_artifact(
        d_model=args.d_model,
        d_head=args.d_head,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        prompt_len=args.prompt_len,
    )
    program_name = (
        f"cv32e40p_gpt2_two_block_reuse_d{args.d_model}_h{args.d_head}_nh{args.n_heads}_"
        f"f{args.ffn_dim}_t{args.prompt_len}_v2"
    )
    hex_path = _emit_and_build(artifact, program_name=program_name)
    print(hex_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
