from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))

from tinynpu_jit import (  # noqa: E402
    DType,
    ExecutionPlan,
    HostOp,
    MatMulOp,
    NpuSegment,
    TensorKind,
    TensorSpec,
    compile_plan,
    emit_cv32e40p_program_v2,
    make_native_int16_kv_cache_specs,
)
from tinynpu_jit.golden import GoldenModel  # noqa: E402

from run_cv32e40p_b_append_demo import (  # noqa: E402
    CORE_DIR,
    CUSTOM_DIR,
    GENERATED_DIR,
    RUNTIME_DIR,
    TNPU_RISCV_MABI,
    TNPU_RISCV_MARCH,
    _run,
    _runner_source,
    _sanitize,
    _toolchain_prefix,
    _toolchain_root,
)


def _rand_i16(rng: np.random.Generator, shape: tuple[int, ...], low: int = -2, high: int = 3) -> np.ndarray:
    return rng.integers(low, high, size=shape, endpoint=False, dtype=np.int16)


def _rand_f32(rng: np.random.Generator, shape: tuple[int, ...], low: float = -0.25, high: float = 0.25) -> np.ndarray:
    return rng.uniform(low, high, size=shape).astype(np.float32)


def _layernorm_ref(x: np.ndarray, weight_bias: np.ndarray, eps: float) -> np.ndarray:
    hidden = x.shape[-1]
    flat = np.asarray(weight_bias, dtype=np.float32).reshape(-1)
    weight = flat[:hidden]
    bias = flat[hidden:]
    mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
    centered = x - mean
    var = np.mean(np.square(centered, dtype=np.float32), axis=-1, keepdims=True, dtype=np.float32)
    norm = centered / np.sqrt(var + np.float32(eps)).astype(np.float32)
    shape = (1,) * (x.ndim - 1) + (hidden,)
    return (norm * weight.reshape(shape) + bias.reshape(shape)).astype(np.float32)


def _fp16_roundtrip(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


def _fp16_bits_carrier(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).astype(np.float16).view(np.uint16).astype(np.int16)


def _host_recip_approx_scalar(x: float) -> float:
    if x <= 0.0:
        raise ValueError("reciprocal input must be positive")
    exp2 = 0
    while x > 1.0:
        x *= 0.5
        exp2 += 1
    while x < 0.5:
        x *= 2.0
        exp2 -= 1
    y = 2.8235295 - 1.8823529 * x
    y = y * (2.0 - x * y)
    y = y * (2.0 - x * y)
    while exp2 > 0:
        y *= 0.5
        exp2 -= 1
    while exp2 < 0:
        y *= 2.0
        exp2 += 1
    return y


def _host_rsqrt_approx_scalar(x: float) -> float:
    if x <= 0.0:
        raise ValueError("rsqrt input must be positive")
    scale = 1.0
    while x > 2.0:
        x *= 0.25
        scale *= 0.5
    while x < 0.5:
        x *= 4.0
        scale *= 2.0
    y = 1.25 - 0.25 * x
    for _ in range(4):
        y = y * (1.5 - 0.5 * x * y * y)
    return y * scale


def _host_exp_approx_scalar(x: float) -> float:
    exp_neg_int = (
        1.0,
        0.36787945,
        0.13533528,
        0.049787067,
        0.018315639,
        0.0067379470,
        0.0024787522,
        0.00091188195,
        0.00033546263,
        0.00012340980,
        0.000045399930,
        0.000016701700,
        0.0000061442124,
        0.0000022603294,
        0.00000083152872,
        0.00000030590232,
        0.00000011253518,
    )
    if x == 0.0:
        return 1.0
    if x > 0.0:
        return _host_recip_approx_scalar(_host_exp_approx_scalar(-x))
    if x <= -16.0:
        return 0.0
    k = int(-x)
    r = x + float(k)
    poly = 1.0 + r * (1.0 + r * (0.5 + r * (0.16666667 + r * (0.04166667 + r * 0.0083333333))))
    return exp_neg_int[k] * max(poly, 0.0)


def _host_erf_approx_scalar(x: float) -> float:
    p = 0.3275911
    a1 = 0.25482959
    a2 = -0.28449672
    a3 = 1.4214138
    a4 = -1.4531521
    a5 = 1.0614054
    sign = 1.0
    if x < 0.0:
        sign = -1.0
        x = -x
    t = _host_recip_approx_scalar(1.0 + p * x)
    poly = (((((a5 * t) + a4) * t + a3) * t + a2) * t + a1) * t
    return sign * (1.0 - poly * _host_exp_approx_scalar(-(x * x)))


def _gelu_runtime_approx(x: np.ndarray) -> np.ndarray:
    source = np.asarray(x, dtype=np.float32)
    erf_term = np.vectorize(lambda v: _host_erf_approx_scalar(float(v) * 0.70710678), otypes=[np.float32])(source)
    return (0.5 * source * (1.0 + erf_term)).astype(np.float32)


def _layernorm_runtime_approx(x: np.ndarray, weight_bias: np.ndarray, eps: float) -> np.ndarray:
    hidden = x.shape[-1]
    flat = np.asarray(weight_bias, dtype=np.float32).reshape(-1)
    weight = flat[:hidden]
    bias = flat[hidden:]
    out = np.zeros_like(x, dtype=np.float32)
    x2 = np.asarray(x, dtype=np.float32).reshape(-1, hidden)
    out2 = out.reshape(-1, hidden)
    for row_idx in range(x2.shape[0]):
        row = x2[row_idx]
        mean = float(np.sum(row, dtype=np.float32)) * _host_recip_approx_scalar(float(hidden))
        centered = row - np.float32(mean)
        var = float(np.sum(centered * centered, dtype=np.float32)) * _host_recip_approx_scalar(float(hidden))
        inv_std = _host_rsqrt_approx_scalar(var + float(eps))
        out2[row_idx] = (centered * np.float32(inv_std) * weight + bias).astype(np.float32)
    return out


def build_plan(
    *,
    d_model: int = 32,
    d_head: int = 16,
    n_heads: int = 2,
    ffn_dim: int = 64,
    token_count: int = 8,
    seed: int = 0,
    act_scale: float = 1.0 / 32.0,
    attn_scale: float = 1.0 / 256.0,
    verify_tensors: tuple[str, ...] = ("out",),
    ffn_fc_activation: str = "h_gelu",
    ffn_fc_zero_bias: bool = False,
    ffn_fc_weight_mode: str = "normal",
    ffn_fc_input_name: str = "x_norm2_q",
):
    if d_model <= 0 or d_model % 8 != 0:
        raise ValueError("d_model must be a positive multiple of 8")
    if d_head <= 0 or d_head % 8 != 0:
        raise ValueError("d_head must be a positive multiple of 8")
    if n_heads <= 0:
        raise ValueError("n_heads must be positive")
    if ffn_dim <= 0 or ffn_dim % 8 != 0:
        raise ValueError("ffn_dim must be a positive multiple of 8")
    if token_count <= 0 or token_count % 8 != 0:
        raise ValueError("token_count must be a positive multiple of 8")

    attn_dim = n_heads * d_head
    rng = np.random.default_rng(seed)

    x = _rand_i16(rng, (token_count, d_model))
    x_f = x.astype(np.float32)
    pos_emb = _rand_f32(rng, (token_count, d_model))
    ln1_gamma = rng.uniform(0.5, 1.5, size=(d_model,)).astype(np.float32)
    ln1_beta = _rand_f32(rng, (d_model,), low=-0.1, high=0.1)
    ln2_gamma = rng.uniform(0.5, 1.5, size=(d_model,)).astype(np.float32)
    ln2_beta = _rand_f32(rng, (d_model,), low=-0.1, high=0.1)
    ln1_wb = np.stack([ln1_gamma, ln1_beta], axis=0).astype(np.float32)
    ln2_wb = np.stack([ln2_gamma, ln2_beta], axis=0).astype(np.float32)
    w_o = _rand_i16(rng, (attn_dim, d_model))
    if ffn_fc_weight_mode == "identity":
        if d_model != ffn_dim:
            raise ValueError("ffn_fc_weight_mode=identity requires d_model == ffn_dim")
        w_fc = np.eye(d_model, dtype=np.int16)
    else:
        w_fc = _rand_i16(rng, (d_model, ffn_dim))
    w_proj = _rand_i16(rng, (ffn_dim, d_model))
    ffn_zero_bias = np.zeros((1, ffn_dim), dtype=np.int16) if ffn_fc_zero_bias else None

    tensors: dict[str, TensorSpec] = {
        "x": TensorSpec("x", x.shape, DType.INT16, TensorKind.CONSTANT, data=x),
        "x_f": TensorSpec("x_f", x_f.shape, DType.FLOAT32, TensorKind.CONSTANT, data=x_f),
        "pos_emb": TensorSpec("pos_emb", pos_emb.shape, DType.FLOAT32, TensorKind.CONSTANT, data=pos_emb),
        "ln1_wb": TensorSpec("ln1_wb", ln1_wb.shape, DType.FLOAT32, TensorKind.CONSTANT, data=ln1_wb),
        "ln2_wb": TensorSpec("ln2_wb", ln2_wb.shape, DType.FLOAT32, TensorKind.CONSTANT, data=ln2_wb),
        "w_o": TensorSpec("w_o", w_o.shape, DType.INT16, TensorKind.CONSTANT, data=w_o),
        "w_fc": TensorSpec("w_fc", w_fc.shape, DType.INT16, TensorKind.CONSTANT, data=w_fc),
        "w_proj": TensorSpec("w_proj", w_proj.shape, DType.INT16, TensorKind.CONSTANT, data=w_proj),
        "x_pos": TensorSpec("x_pos", (token_count, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "x_norm1": TensorSpec("x_norm1", (token_count, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        "x_norm1_q": TensorSpec(
            "x_norm1_q",
            (token_count, d_model),
            DType.INT16,
            TensorKind.INTERMEDIATE,
            metadata={"storage_role": "A"},
        ),
        "o_int": TensorSpec("o_int", (token_count, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        "o_f": TensorSpec("o_f", (token_count, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "resid1": TensorSpec("resid1", (token_count, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "x_norm2": TensorSpec("x_norm2", (token_count, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        "x_norm2_q": TensorSpec(
            "x_norm2_q",
            (token_count, d_model),
            DType.INT16,
            TensorKind.INTERMEDIATE,
            metadata={"storage_role": "A"},
        ),
        "ffn_fc_int": TensorSpec("ffn_fc_int", (token_count, ffn_dim), DType.INT16, TensorKind.INTERMEDIATE),
        "ffn_fc_a": TensorSpec(
            "ffn_fc_a",
            (token_count, ffn_dim),
            DType.INT16,
            TensorKind.INTERMEDIATE,
            metadata={"storage_role": "A"},
        ),
        "ffn_out_int": TensorSpec("ffn_out_int", (token_count, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        "ffn_out_f": TensorSpec("ffn_out_f", (token_count, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "out": TensorSpec(
            "out",
            (token_count, d_model),
            DType.FLOAT32,
            TensorKind.OUTPUT,
            is_final_output=True,
            metadata={"verify_atol": 0.5},
        ),
    }
    if ffn_zero_bias is not None:
        tensors["a_ffn_zero_bias"] = TensorSpec(
            "a_ffn_zero_bias",
            ffn_zero_bias.shape,
            DType.INT16,
            TensorKind.CONSTANT,
            data=ffn_zero_bias,
        )

    q_ops: list[MatMulOp] = []
    kv_cache_ops: list[MatMulOp] = []
    concat_steps: list[HostOp] = []
    prev_concat_name: str | None = None
    token_indices = list(range(token_count))
    token_names = [f"t{idx}" for idx in token_indices]

    for head_idx in range(n_heads):
        w_q = _rand_i16(rng, (d_model, d_head))
        w_k = _rand_i16(rng, (d_model, d_head))
        w_v = _rand_i16(rng, (d_model, d_head))
        tensors[f"w_q_h{head_idx}"] = TensorSpec(f"w_q_h{head_idx}", w_q.shape, DType.INT16, TensorKind.CONSTANT, data=w_q)
        tensors[f"w_k_h{head_idx}"] = TensorSpec(f"w_k_h{head_idx}", w_k.shape, DType.INT16, TensorKind.CONSTANT, data=w_k)
        tensors[f"w_v_h{head_idx}"] = TensorSpec(f"w_v_h{head_idx}", w_v.shape, DType.INT16, TensorKind.CONSTANT, data=w_v)
        tensors[f"q_int_h{head_idx}"] = TensorSpec(f"q_int_h{head_idx}", (token_count, d_head), DType.INT16, TensorKind.INTERMEDIATE)
        tensors[f"k_seq_h{head_idx}"] = TensorSpec(f"k_seq_h{head_idx}", (token_count, d_head), DType.INT16, TensorKind.INTERMEDIATE)
        tensors[f"v_seq_h{head_idx}"] = TensorSpec(f"v_seq_h{head_idx}", (token_count, d_head), DType.INT16, TensorKind.INTERMEDIATE)
        tensors[f"q_a_h{head_idx}"] = TensorSpec(
            f"q_a_h{head_idx}",
            (token_count, d_head),
            DType.INT16,
            TensorKind.INTERMEDIATE,
            metadata={"storage_role": "A"},
        )
        tensors.update(
            make_native_int16_kv_cache_specs(
                k_base_name=f"k_cache_h{head_idx}",
                v_base_name=f"v_cache_h{head_idx}",
                d_head=d_head,
                token_capacity=token_count,
                token_names=token_names,
                token_indices=token_indices,
                kind=TensorKind.INTERMEDIATE,
            )
        )
        tensors[f"scores_h{head_idx}"] = TensorSpec(f"scores_h{head_idx}", (token_count, token_count), DType.INT16, TensorKind.INTERMEDIATE)
        tensors[f"masked_scores_h{head_idx}"] = TensorSpec(
            f"masked_scores_h{head_idx}",
            (token_count, token_count),
            DType.INT16,
            TensorKind.INTERMEDIATE,
        )
        tensors[f"probs_h{head_idx}"] = TensorSpec(
            f"probs_h{head_idx}",
            (token_count, token_count),
            DType.INT16,
            TensorKind.INTERMEDIATE,
        )
        tensors[f"probs_q_h{head_idx}"] = TensorSpec(
            f"probs_q_h{head_idx}",
            (token_count, token_count),
            DType.INT16,
            TensorKind.INTERMEDIATE,
            metadata={"storage_role": "A"},
        )
        tensors[f"attn_h{head_idx}"] = TensorSpec(f"attn_h{head_idx}", (token_count, d_head), DType.INT16, TensorKind.INTERMEDIATE)

        q_ops.append(MatMulOp(f"op_q_h{head_idx}", "x_norm1_q", f"w_q_h{head_idx}", f"q_int_h{head_idx}", in_dtype=DType.INT16, out_dtype=DType.INT16))
        kv_cache_ops.extend(
            [
                MatMulOp(
                    f"op_k_h{head_idx}",
                    "x_norm1_q",
                    f"w_k_h{head_idx}",
                    f"k_seq_h{head_idx}",
                    in_dtype=DType.INT16,
                    out_dtype=DType.INT16,
                ),
                MatMulOp(
                    f"op_v_h{head_idx}",
                    "x_norm1_q",
                    f"w_v_h{head_idx}",
                    f"v_seq_h{head_idx}",
                    in_dtype=DType.INT16,
                    out_dtype=DType.INT16,
                ),
            ]
        )
        if prev_concat_name is None:
            prev_concat_name = f"attn_h{head_idx}"
        else:
            concat_name = f"attn_cat_{head_idx}"
            prev_width = d_head * head_idx
            tensors[concat_name] = TensorSpec(concat_name, (token_count, prev_width + d_head), DType.INT16, TensorKind.INTERMEDIATE)
            concat_steps.append(
                HostOp(
                    f"concat_attn_{head_idx}",
                    "concat_lastdim2",
                    inputs=[prev_concat_name, f"attn_h{head_idx}"],
                    outputs=[concat_name],
                )
            )
            prev_concat_name = concat_name

    attn_cat_name = prev_concat_name if prev_concat_name is not None else "attn_h0"
    if n_heads == 1:
        tensors["attn_cat"] = TensorSpec("attn_cat", (token_count, d_head), DType.INT16, TensorKind.INTERMEDIATE)
        concat_steps.append(HostOp("alias_attn_cat", "alias", inputs=[attn_cat_name], outputs=["attn_cat"]))
        attn_cat_name = "attn_cat"
    elif attn_cat_name != "attn_cat":
        tensors["attn_cat"] = TensorSpec("attn_cat", (token_count, attn_dim), DType.INT16, TensorKind.INTERMEDIATE)
        concat_steps.append(HostOp("alias_attn_cat", "alias", inputs=[attn_cat_name], outputs=["attn_cat"]))
        attn_cat_name = "attn_cat"
    tensors["attn_cat_a"] = TensorSpec(
        "attn_cat_a",
        tensors[attn_cat_name].shape,
        DType.INT16,
        TensorKind.INTERMEDIATE,
        metadata={"storage_role": "A"},
    )

    steps: list[object] = [
        HostOp("add_pos", "add", inputs=["x_f", "pos_emb"], outputs=["x_pos"]),
        HostOp(
            "layernorm1",
            "layernorm",
            inputs=["x_pos", "ln1_wb"],
            outputs=["x_norm1"],
            attrs={"eps": 1.0e-5, "output_encoding": "fp16_bits"},
        ),
        HostOp(
            "quant_x_norm1",
            "quantize",
            inputs=["x_norm1"],
            outputs=["x_norm1_q"],
            attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16, "input_encoding": "fp16_bits"},
        ),
    ]

    steps.extend(
        [
        NpuSegment(
            "seg_q",
            q_ops,
            inputs=["x_norm1_q"],
            outputs=[name for name in tensors if name.startswith("q_int_h")],
        ),
        NpuSegment(
            "seg_kv_cache",
            kv_cache_ops,
            inputs=["x_norm1_q"],
            outputs=[name for name in tensors if name.startswith("k_seq_h") or name.startswith("v_seq_h")],
        ),
    ])

    for head_idx in range(n_heads):
        steps.extend(
            [
                HostOp(
                    f"k_cache_scatter_matrix_h{head_idx}",
                    "k_cache_scatter_matrix",
                    inputs=[f"k_seq_h{head_idx}"],
                    outputs=[f"k_cache_h{head_idx}"],
                ),
            ]
        )

    for head_idx in range(n_heads):
        steps.extend(
            [
                HostOp(f"alias_q_a_h{head_idx}", "alias", inputs=[f"q_int_h{head_idx}"], outputs=[f"q_a_h{head_idx}"]),
                NpuSegment(
                    f"seg_score_h{head_idx}",
                    [MatMulOp(f"op_qk_h{head_idx}", f"q_a_h{head_idx}", f"k_cache_h{head_idx}", f"scores_h{head_idx}", in_dtype=DType.INT16, out_dtype=DType.INT16)],
                    inputs=[f"q_a_h{head_idx}"],
                    outputs=[f"scores_h{head_idx}"],
                ),
                HostOp(
                    f"causal_mask_h{head_idx}",
                    "causal_mask",
                    inputs=[f"scores_h{head_idx}"],
                    outputs=[f"masked_scores_h{head_idx}"],
                    attrs={"past_kv_len": 0, "fill_value": float(np.iinfo(np.int16).min)},
                ),
                HostOp(
                    f"softmax_scores_h{head_idx}",
                    "softmax_f16",
                    inputs=[f"masked_scores_h{head_idx}"],
                    outputs=[f"probs_h{head_idx}"],
                    attrs={"axis": -1},
                ),
                HostOp(
                    f"quantize_probs_h{head_idx}",
                    "quantize",
                    inputs=[f"probs_h{head_idx}"],
                    outputs=[f"probs_q_h{head_idx}"],
                    attrs={"scale": attn_scale, "zero_point": 0, "dtype": DType.INT16, "input_encoding": "fp16_bits"},
                ),
                HostOp(
                    f"v_cache_scatter_matrix_h{head_idx}",
                    "v_cache_scatter_matrix",
                    inputs=[f"v_seq_h{head_idx}"],
                    outputs=[f"v_cache_h{head_idx}"],
                ),
                NpuSegment(
                    f"seg_av_h{head_idx}",
                    [MatMulOp(f"op_av_h{head_idx}", f"probs_q_h{head_idx}", f"v_cache_h{head_idx}", f"attn_h{head_idx}", shift=8, in_dtype=DType.INT16, out_dtype=DType.INT16)],
                    inputs=[f"probs_q_h{head_idx}"],
                    outputs=[f"attn_h{head_idx}"],
                ),
            ]
        )

    steps.extend(concat_steps)
    steps.extend(
        [
            HostOp("alias_attn_cat_a", "alias", inputs=[attn_cat_name], outputs=["attn_cat_a"]),
            NpuSegment(
                "seg_o_proj",
                [MatMulOp("op_o_proj", "attn_cat_a", "w_o", "o_int", in_dtype=DType.INT16, out_dtype=DType.INT16)],
                inputs=["attn_cat_a"],
                outputs=["o_int"],
            ),
            HostOp("dequant_o", "dequantize", inputs=["o_int"], outputs=["o_f"], attrs={"scale": act_scale, "zero_point": 0}),
            HostOp("residual1", "add", inputs=["x_pos", "o_f"], outputs=["resid1"]),
            HostOp(
                "layernorm2",
                "layernorm",
                inputs=["resid1", "ln2_wb"],
                outputs=["x_norm2"],
                attrs={"eps": 1.0e-5, "output_encoding": "fp16_bits"},
            ),
            HostOp(
                "quant_x_norm2",
                "quantize",
                inputs=["x_norm2"],
                outputs=["x_norm2_q"],
                attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16, "input_encoding": "fp16_bits"},
            ),
            NpuSegment(
                "seg_ffn_fc",
                [MatMulOp(
                    "op_ffn_fc",
                    ffn_fc_input_name,
                    "w_fc",
                    "ffn_fc_int",
                    bias="a_ffn_zero_bias" if ffn_fc_zero_bias else None,
                    activation=ffn_fc_activation,
                    in_dtype=DType.INT16,
                    out_dtype=DType.INT16,
                )],
                inputs=[ffn_fc_input_name],
                outputs=["ffn_fc_int"],
            ),
            HostOp("alias_ffn_fc_a", "alias", inputs=["ffn_fc_int"], outputs=["ffn_fc_a"]),
            NpuSegment(
                "seg_ffn_proj",
                [MatMulOp("op_ffn_proj", "ffn_fc_a", "w_proj", "ffn_out_int", in_dtype=DType.INT16, out_dtype=DType.INT16)],
                inputs=["ffn_fc_a"],
                outputs=["ffn_out_int"],
            ),
            HostOp("dequant_ffn_out", "dequantize", inputs=["ffn_out_int"], outputs=["ffn_out_f"], attrs={"scale": act_scale, "zero_point": 0}),
            HostOp("residual2", "add", inputs=["resid1", "ffn_out_f"], outputs=["out"]),
        ]
    )

    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=[], outputs=["out"])
    verify_labels = {
        "out": "prefill_transformer_block_out",
        "ffn_fc_int": "ffn_fc_int",
        "ffn_out_f": "ffn_out_f",
        "o_int": "o_int",
        "o_f": "o_f",
        "resid1": "resid1",
        "x_norm1": "x_norm1",
        "x_norm2": "x_norm2",
        "x_norm1_q": "x_norm1_q",
        "x_norm2_q": "x_norm2_q",
        "attn_cat": "attn_cat",
        "attn_h0": "attn_h0",
        "scores_h0": "scores_h0",
        "probs_h0": "probs_h0",
        "v_cache_h0": "v_cache_h0",
        "v_seq_h0": "v_seq_h0",
    }
    for tensor_name in verify_tensors:
        plan.add_verification_step(tensor_name, verify_labels.get(tensor_name, tensor_name))
    return plan


def build_artifact(**kwargs):
    verify_tensors = tuple(kwargs.get("verify_tensors", ("out",)))
    plan = build_plan(**kwargs)
    golden = GoldenModel()
    tensors = plan.tensors
    act_scale = float(kwargs.get("act_scale", 1.0 / 32.0))
    attn_scale = float(kwargs.get("attn_scale", 1.0 / 256.0))
    n_heads = int(kwargs["n_heads"])
    ffn_fc_activation = str(kwargs.get("ffn_fc_activation", "h_gelu"))
    ffn_fc_zero_bias = bool(kwargs.get("ffn_fc_zero_bias", False))
    ffn_fc_input_name = str(kwargs.get("ffn_fc_input_name", "x_norm2_q"))

    x_f = np.array(tensors["x_f"].data, dtype=np.float32, copy=True)
    pos_emb = np.array(tensors["pos_emb"].data, dtype=np.float32, copy=True)
    ln1_wb = np.array(tensors["ln1_wb"].data, dtype=np.float32, copy=True)
    ln2_wb = np.array(tensors["ln2_wb"].data, dtype=np.float32, copy=True)
    w_o = np.array(tensors["w_o"].data, dtype=np.int16, copy=True)
    w_fc = np.array(tensors["w_fc"].data, dtype=np.int16, copy=True)
    w_proj = np.array(tensors["w_proj"].data, dtype=np.int16, copy=True)

    x_pos = (x_f + pos_emb).astype(np.float32)
    x_norm1 = _fp16_roundtrip(_layernorm_runtime_approx(x_pos, ln1_wb, 1.0e-5))
    x_norm1_q = golden.quantize(x_norm1, scale=act_scale, zero_point=0, out_dtype=DType.INT16)

    attn_heads: list[np.ndarray] = []
    expected_scores: dict[str, np.ndarray] = {}
    expected_probs: dict[str, np.ndarray] = {}
    expected_v_seq: dict[str, np.ndarray] = {}
    for head_idx in range(n_heads):
        w_q = np.array(tensors[f"w_q_h{head_idx}"].data, dtype=np.int16, copy=True)
        w_k = np.array(tensors[f"w_k_h{head_idx}"].data, dtype=np.int16, copy=True)
        w_v = np.array(tensors[f"w_v_h{head_idx}"].data, dtype=np.int16, copy=True)
        q_int = golden.matmul(x_norm1_q, w_q, out_dtype=DType.INT16)
        k_seq = golden.matmul(x_norm1_q, w_k, out_dtype=DType.INT16)
        v_seq = golden.matmul(x_norm1_q, w_v, out_dtype=DType.INT16)
        expected_v_seq[f"v_seq_h{head_idx}"] = np.array(v_seq, copy=True)
        scores = golden.matmul(q_int, np.array(k_seq.T, dtype=np.int16, copy=True), out_dtype=DType.INT16)
        expected_scores[f"scores_h{head_idx}"] = np.array(scores, copy=True)
        masked_scores = np.array(scores, copy=True)
        for row in range(masked_scores.shape[0]):
            if row + 1 < masked_scores.shape[1]:
                masked_scores[row, row + 1 :] = np.iinfo(np.int16).min
        probs = _fp16_roundtrip(golden.softmax(masked_scores, axis=-1).astype(np.float32))
        expected_probs[f"probs_h{head_idx}"] = _fp16_bits_carrier(probs)
        probs_q = golden.quantize(probs, scale=attn_scale, zero_point=0, out_dtype=DType.INT16)
        attn_h = golden.matmul(probs_q, v_seq, shift=8, out_dtype=DType.INT16)
        attn_heads.append(np.array(attn_h, copy=True))

    attn_cat = attn_heads[0]
    for attn_h in attn_heads[1:]:
        attn_cat = np.concatenate([attn_cat, attn_h], axis=-1).astype(np.int16)

    o_int = golden.matmul(attn_cat, w_o, out_dtype=DType.INT16)
    o_f = golden.dequantize(o_int, scale=act_scale, zero_point=0)
    resid1 = (x_pos + o_f).astype(np.float32)

    x_norm2 = _fp16_roundtrip(_layernorm_runtime_approx(resid1, ln2_wb, 1.0e-5))
    x_norm2_q = golden.quantize(x_norm2, scale=act_scale, zero_point=0, out_dtype=DType.INT16)
    ffn_zero_bias = np.array(tensors["a_ffn_zero_bias"].data, dtype=np.int16, copy=True) if ffn_fc_zero_bias else None
    ffn_fc_input = x_norm2_q if ffn_fc_input_name == "x_norm2_q" else x_norm1_q
    ffn_fc_int = golden.matmul(
        ffn_fc_input,
        w_fc,
        bias=ffn_zero_bias,
        activation=ffn_fc_activation,
        out_dtype=DType.INT16,
    )
    ffn_out_int = golden.matmul(ffn_fc_int, w_proj, out_dtype=DType.INT16)
    ffn_out_f = golden.dequantize(ffn_out_int, scale=act_scale, zero_point=0)
    out = (resid1 + ffn_out_f).astype(np.float32)

    expected_tensors = {
        "out": np.array(out, copy=True),
        "ffn_fc_int": np.array(ffn_fc_int, copy=True),
        "ffn_out_f": np.array(ffn_out_f, copy=True),
        "o_int": np.array(o_int, copy=True),
        "o_f": np.array(o_f, copy=True),
        "resid1": np.array(resid1, copy=True),
        "x_norm1": np.array(x_norm1, copy=True),
        "x_norm2": np.array(x_norm2, copy=True),
        "x_norm1_q": np.array(x_norm1_q, copy=True),
        "x_norm2_q": np.array(x_norm2_q, copy=True),
        "attn_cat": np.array(attn_cat, copy=True),
    }
    if attn_heads:
        expected_tensors["attn_h0"] = np.array(attn_heads[0], copy=True)
    expected_tensors.update(expected_scores)
    expected_tensors.update(expected_probs)
    expected_tensors.update(expected_v_seq)
    if "v_seq_h0" in expected_v_seq:
        expected_tensors["v_cache_h0"] = np.array(expected_v_seq["v_seq_h0"], copy=True)
    artifact = compile_plan(
        plan,
        {name: expected_tensors[name] for name in verify_tensors if name in expected_tensors},
    )
    return artifact, out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-head", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify-tensor", action="append", default=None)
    parser.add_argument("--ffn-fc-activation", type=str, default="h_gelu")
    parser.add_argument("--ffn-fc-zero-bias", action="store_true")
    parser.add_argument("--ffn-fc-weight-mode", type=str, default="normal")
    parser.add_argument("--ffn-fc-input-name", type=str, default="x_norm2_q")
    parser.add_argument("--maxcycles", type=int, default=3000000)
    parser.add_argument("--verilator-max-ticks", type=int, default=30000000000)
    args = parser.parse_args()

    verify_tensors = tuple(args.verify_tensor) if args.verify_tensor else ("out",)
    artifact, expected = build_artifact(
        d_model=args.d_model,
        d_head=args.d_head,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        token_count=args.token_count,
        seed=args.seed,
        verify_tensors=verify_tensors,
        ffn_fc_activation=args.ffn_fc_activation,
        ffn_fc_zero_bias=args.ffn_fc_zero_bias,
        ffn_fc_weight_mode=args.ffn_fc_weight_mode,
        ffn_fc_input_name=args.ffn_fc_input_name,
    )

    program_name = (
        f"cv32e40p_prefill_transformer_block_d{args.d_model}"
        f"_h{args.d_head}_nh{args.n_heads}_f{args.ffn_dim}_t{args.token_count}_s{args.seed}_v2"
    )
    runner_name = f"{program_name}_runner"
    GENERATED_DIR.mkdir(exist_ok=True)
    program_path = GENERATED_DIR / f"{program_name}_program.c"
    runner_path = GENERATED_DIR / f"{runner_name}.c"

    source = emit_cv32e40p_program_v2(artifact, artifact.expected_tensors, program_name=program_name)
    program_path.write_text(source)
    runner_path.write_text(_runner_source(program_name))

    prefix = _toolchain_prefix()
    gcc = f"{prefix}gcc"
    objcopy = f"{prefix}objcopy"
    toolchain_root = _toolchain_root(prefix)
    include_dir = toolchain_root / "riscv32-unknown-elf" / "include"
    lib_dir = toolchain_root / "riscv32-unknown-elf" / "lib"
    elf_path = CUSTOM_DIR / f"{runner_name}.elf"
    hex_path = CUSTOM_DIR / f"{runner_name}.hex"

    build_env = dict(os.environ)
    build_env["CCACHE_DISABLE"] = "1"
    build_env["TMPDIR"] = "/tmp"
    _run(["make", "verilator-build-npu"], cwd=CORE_DIR, env=build_env)
    _run(
        [
            gcc,
            f"-march={TNPU_RISCV_MARCH}",
            f"-mabi={TNPU_RISCV_MABI}",
            "-o",
            str(elf_path),
            "-w",
            "-O3",
            "-g",
            "-nostdlib",
            "-T",
            "custom/link.ld",
            "-static",
            "custom/crt0.S",
            str(program_path),
            str(runner_path),
            str(RUNTIME_DIR / "tinynpu_runtime_v2.c"),
            "mem_stall/mem_stall.c",
            "custom/syscalls.c",
            "custom/vectors.S",
            "-I",
            str(include_dir),
            "-I",
            "mem_stall",
            "-I",
            str(RUNTIME_DIR),
            "-L",
            str(lib_dir),
            "-lc",
            "-lm",
            "-lgcc",
        ],
        cwd=CORE_DIR,
        env=build_env,
    )
    _run([objcopy, "-O", "verilog", str(elf_path), str(hex_path)], cwd=CORE_DIR, env=build_env)

    env = dict(os.environ)
    env["VERILATOR_MAX_TICKS"] = str(args.verilator_max_ticks)
    proc = _run(
        [
            str(CORE_DIR / "obj_dir" / "cv32e40p_tb_vlt_npu"),
            "+verilator+noassert",
            f"+firmware={hex_path}",
            f"+maxcycles={args.maxcycles}",
        ],
        cwd=CORE_DIR,
        env=env,
        capture=True,
    )
    print(f"program={program_name}")
    print(
        f"d_model={args.d_model} d_head={args.d_head} n_heads={args.n_heads} "
        f"ffn_dim={args.ffn_dim} token_count={args.token_count} seed={args.seed}"
    )
    print(f"ffn_fc_activation={args.ffn_fc_activation}")
    print(f"ffn_fc_zero_bias={args.ffn_fc_zero_bias}")
    print(f"ffn_fc_weight_mode={args.ffn_fc_weight_mode}")
    print(f"ffn_fc_input_name={args.ffn_fc_input_name}")
    print(f"verify_tensors={verify_tensors}")
    print(f"expected_checksum={float(np.array(expected, dtype=np.float32).sum()):.6f}")
    print(proc.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
