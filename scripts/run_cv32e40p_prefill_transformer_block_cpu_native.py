from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

from run_cv32e40p_b_append_demo import (  # noqa: E402
    CORE_DIR,
    CUSTOM_DIR,
    GENERATED_DIR,
    TNPU_RISCV_MABI,
    TNPU_RISCV_MARCH,
    _run,
    _toolchain_prefix,
    _toolchain_root,
)
from run_cv32e40p_prefill_transformer_block_jit_demo import build_plan  # noqa: E402


def _emit_f32_array(name: str, values: np.ndarray, *, section_data: bool) -> str:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    attr = ' __attribute__((section(".data")))' if section_data else ' __attribute__((section(".bss")))'
    if flat.size == 0:
        return f"static float {name}[1]{attr} = {{0.0f}};"
    if section_data:
        body = ", ".join(f"{float(v):.8e}f" for v in flat)
        return f"static float {name}[{flat.size}]{attr} = {{\n    {body}\n}};"
    return f"static float {name}[{flat.size}]{attr};"


def _native_float_reference(*, d_model: int, d_head: int, n_heads: int, ffn_dim: int, token_count: int, seed: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    plan = build_plan(
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        token_count=token_count,
        seed=seed,
    )
    t = plan.tensors

    x_f = np.asarray(t["x_f"].data, dtype=np.float32)
    pos_emb = np.asarray(t["pos_emb"].data, dtype=np.float32)
    ln1_wb = np.asarray(t["ln1_wb"].data, dtype=np.float32)
    ln2_wb = np.asarray(t["ln2_wb"].data, dtype=np.float32)
    w_o = np.asarray(t["w_o"].data, dtype=np.float32)
    w_fc = np.asarray(t["w_fc"].data, dtype=np.float32)
    w_proj = np.asarray(t["w_proj"].data, dtype=np.float32)

    def layernorm(x: np.ndarray, wb: np.ndarray, eps: float = 1.0e-5) -> np.ndarray:
        weight = wb[0].astype(np.float32)
        bias = wb[1].astype(np.float32)
        mean = np.mean(x, axis=-1, keepdims=True, dtype=np.float32)
        centered = x - mean
        var = np.mean(np.square(centered, dtype=np.float32), axis=-1, keepdims=True, dtype=np.float32)
        norm = centered / np.sqrt(var + np.float32(eps)).astype(np.float32)
        return (norm * weight.reshape(1, -1) + bias.reshape(1, -1)).astype(np.float32)

    x_pos = (x_f + pos_emb).astype(np.float32)
    x_norm1 = layernorm(x_pos, ln1_wb)
    ln1_mean = np.mean(x_pos, axis=-1, dtype=np.float32)
    ln1_var = np.var(x_pos, axis=-1, dtype=np.float32)
    ln1_inv_std = (1.0 / np.sqrt(ln1_var + np.float32(1.0e-5))).astype(np.float32)

    attn_heads: list[np.ndarray] = []
    tensors: dict[str, np.ndarray] = {
        "x_f": x_f,
        "pos_emb": pos_emb,
        "ln1_wb": ln1_wb,
        "ln2_wb": ln2_wb,
        "w_o": w_o,
        "w_fc": w_fc,
        "w_proj": w_proj,
        "x_pos": x_pos,
        "x_norm1": x_norm1,
        "ln1_mean_expected": ln1_mean,
        "ln1_var_expected": ln1_var,
        "ln1_inv_std_expected": ln1_inv_std,
    }
    for head_idx in range(n_heads):
        w_q = np.asarray(t[f"w_q_h{head_idx}"].data, dtype=np.float32)
        w_k = np.asarray(t[f"w_k_h{head_idx}"].data, dtype=np.float32)
        w_v = np.asarray(t[f"w_v_h{head_idx}"].data, dtype=np.float32)
        q = (x_norm1 @ w_q).astype(np.float32)
        k = (x_norm1 @ w_k).astype(np.float32)
        v = (x_norm1 @ w_v).astype(np.float32)
        scores = (q @ k.T).astype(np.float32)
        scores[np.triu_indices(token_count, 1)] = np.float32(-1.0e9)
        scores_max = np.max(scores, axis=-1, keepdims=True)
        probs = np.exp(scores - scores_max).astype(np.float32)
        probs /= np.sum(probs, axis=-1, keepdims=True, dtype=np.float32)
        attn = (probs @ v).astype(np.float32)
        attn_heads.append(attn)
        tensors[f"q_h{head_idx}_expected"] = q
        tensors[f"k_h{head_idx}_expected"] = k
        tensors[f"v_h{head_idx}_expected"] = v
        tensors[f"scores_h{head_idx}_expected"] = scores
        tensors[f"probs_h{head_idx}_expected"] = probs
        tensors[f"attn_h{head_idx}_expected"] = attn
        tensors[f"w_q_h{head_idx}"] = w_q
        tensors[f"w_k_h{head_idx}"] = w_k
        tensors[f"w_v_h{head_idx}"] = w_v

    attn_cat = np.concatenate(attn_heads, axis=-1).astype(np.float32)
    tensors["attn_cat_expected"] = attn_cat
    o = (attn_cat @ w_o).astype(np.float32)
    resid1 = (x_pos + o).astype(np.float32)
    x_norm2 = layernorm(resid1, ln2_wb)
    ffn_fc = (x_norm2 @ w_fc).astype(np.float32)
    ffn_gelu = (0.5 * ffn_fc * (1.0 + np.vectorize(math.erf, otypes=[np.float32])(ffn_fc / np.sqrt(2.0)).astype(np.float32))).astype(np.float32)
    ffn_out = (ffn_gelu @ w_proj).astype(np.float32)
    out = (resid1 + ffn_out).astype(np.float32)

    tensors["out_expected"] = out
    return tensors, out


def _emit_native_source(
    *,
    program_name: str,
    d_model: int,
    d_head: int,
    n_heads: int,
    ffn_dim: int,
    token_count: int,
    tensors: dict[str, np.ndarray],
    expected: np.ndarray,
    host_native: bool,
    stop_after_attention: bool,
    diagnose_attention: bool,
) -> str:
    decls: list[str] = []
    for name, values in tensors.items():
        if name == "out_expected":
            continue
        decls.append(_emit_f32_array(name, values, section_data=True))

    scratch_shapes = {
        "x_pos_buf": (token_count, d_model),
        "x_norm1_buf": (token_count, d_model),
        "q_buf": (token_count, d_head),
        "k_buf": (token_count, d_head),
        "v_buf": (token_count, d_head),
        "scores_buf": (token_count, token_count),
        "probs_buf": (token_count, token_count),
        "attn_buf": (token_count, d_head),
        "attn_cat_buf": (token_count, n_heads * d_head),
        "o_buf": (token_count, d_model),
        "resid1_buf": (token_count, d_model),
        "x_norm2_buf": (token_count, d_model),
        "ffn_fc_buf": (token_count, ffn_dim),
        "ffn_gelu_buf": (token_count, ffn_dim),
        "ffn_out_buf": (token_count, d_model),
        "out_buf": (token_count, d_model),
        "out_expected": expected.shape,
    }
    for name, shape in scratch_shapes.items():
        section_data = name == "out_expected"
        values = expected if name == "out_expected" else np.zeros(shape, dtype=np.float32)
        decls.append(_emit_f32_array(name, values, section_data=section_data))

    for head_idx in range(n_heads):
        decls.append(_emit_f32_array(f"attn_h{head_idx}_buf", np.zeros((token_count, d_head), dtype=np.float32), section_data=False))

    head_weight_select: list[str] = []
    head_attn_select: list[str] = []
    for head_idx in range(n_heads):
        head_weight_select.append(
            f"""        if (head == {head_idx}) {{
            w_q = w_q_h{head_idx};
            w_k = w_k_h{head_idx};
            w_v = w_v_h{head_idx};
            attn_out = attn_h{head_idx}_buf;
        }}"""
        )
        head_attn_select.append(
            f"""        if (head == {head_idx}) {{
            src = attn_h{head_idx}_buf;
        }}"""
        )

    timer_block = """static volatile uint32_t *const tb_timer_count = (volatile uint32_t *)0x15001000u;

static inline uint32_t read_mcycle32(void)
{
    return *tb_timer_count;
}
"""
    if host_native:
        timer_block = """static inline uint32_t read_mcycle32(void)
{
    return 0u;
}
"""

    source = f"""#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

volatile uint32_t runtime_cycle_start __attribute__((section(".noinit")));
volatile uint32_t runtime_cycle_post_bss __attribute__((section(".noinit")));
volatile uint32_t runtime_cycle_post_init __attribute__((section(".noinit")));
volatile uint32_t runtime_cycle_pre_main __attribute__((section(".noinit")));

{timer_block}

static void print_cycle_delta32(const char *label, uint32_t start, uint32_t end)
{{
    printf("%s cycles=%lu\\n", label, (unsigned long)(start - end));
}}

static int g_diag_attention = 0;

static void add_f32(const float *a, const float *b, float *out, int elems)
{{
    for (int i = 0; i < elems; ++i) {{
        out[i] = a[i] + b[i];
    }}
}}

static void matmul_f32(const float *a, const float *b, float *out, int m, int k, int n)
{{
    for (int i = 0; i < m; ++i) {{
        for (int j = 0; j < n; ++j) {{
            float acc = 0.0f;
            for (int p = 0; p < k; ++p) {{
                acc += a[i * k + p] * b[p * n + j];
            }}
            out[i * n + j] = acc;
        }}
    }}
}}

static void matmul_rhs_transposed_f32(const float *a, const float *b_t, float *out, int m, int k, int n)
{{
    for (int i = 0; i < m; ++i) {{
        for (int j = 0; j < n; ++j) {{
            float acc = 0.0f;
            for (int p = 0; p < k; ++p) {{
                acc += a[i * k + p] * b_t[j * k + p];
            }}
            out[i * n + j] = acc;
        }}
    }}
}}

static float exp_approx_f32(float x)
{{
    const float ln2 = 0.69314718056f;
    const float inv_ln2 = 1.44269504089f;
    int k = (int)(x * inv_ln2);
    if ((float)k * ln2 > x) {{
        k -= 1;
    }}
    float r = x - (float)k * ln2;
    float p = 1.0f + r;
    float t = r * r;
    p += 0.5f * t;
    t *= r;
    p += (1.0f / 6.0f) * t;
    t *= r;
    p += (1.0f / 24.0f) * t;
    t *= r;
    p += (1.0f / 120.0f) * t;

    float scale = 1.0f;
    if (k > 0) {{
        for (int i = 0; i < k; ++i) scale *= 2.0f;
    }} else if (k < 0) {{
        for (int i = 0; i < -k; ++i) scale *= 0.5f;
    }}
    return p * scale;
}}

static float inv_sqrt_nr_f32(float x)
{{
    if (!(x > 0.0f)) {{
        return 0.0f;
    }}
    union {{
        float f;
        uint32_t i;
    }} u;
    u.f = x;
    u.i = 0x5f3759dfu - (u.i >> 1);
    float y = u.f;
    y = y * (1.5f - 0.5f * x * y * y);
    y = y * (1.5f - 0.5f * x * y * y);
    y = y * (1.5f - 0.5f * x * y * y);
    return y;
}}

static float reciprocal_nr_f32(float x)
{{
    if (!(x > 0.0f) && !(x < 0.0f)) {{
        return 0.0f;
    }}
    union {{
        float f;
        uint32_t i;
    }} u;
    u.f = x;
    u.i = 0x7EF311C2u - u.i;
    float y = u.f;
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    return y;
}}

static void layernorm_f32(const float *x, const float *weight_bias, float *out, int rows, int cols, float eps)
{{
    const float *weight = weight_bias;
    const float *bias = weight_bias + cols;
    for (int r = 0; r < rows; ++r) {{
        float mean = 0.0f;
        for (int c = 0; c < cols; ++c) {{
            mean += x[r * cols + c];
        }}
        mean /= (float)cols;
        float var = 0.0f;
        for (int c = 0; c < cols; ++c) {{
            float centered = x[r * cols + c] - mean;
            var += centered * centered;
        }}
        var /= (float)cols;
        float inv_std = inv_sqrt_nr_f32(var + eps);
        for (int c = 0; c < cols; ++c) {{
            float norm = (x[r * cols + c] - mean) * inv_std;
            out[r * cols + c] = norm * weight[c] + bias[c];
        }}
    }}
}}

static void causal_softmax_f32(float *scores, float *probs, int rows, int cols)
{{
    // Avoid depending on expf ABI/libm behavior on bare-metal ILP32F.
    // Use double-precision exp and clamp inputs for numerical stability.
    const float kExpHi = 80.0f;
    const float kExpLo = -80.0f;
    for (int r = 0; r < rows; ++r) {{
        float maxv = -1.0e30f;
        for (int c = 0; c < cols; ++c) {{
            float v = (c > r) ? -1.0e9f : scores[r * cols + c];
            scores[r * cols + c] = v;
            if (v > maxv) maxv = v;
        }}
        float denom = 0.0f;
        for (int c = 0; c < cols; ++c) {{
            float z = scores[r * cols + c] - maxv;
            if (z > kExpHi) z = kExpHi;
            if (z < kExpLo) z = kExpLo;
            float e = exp_approx_f32(z);
            probs[r * cols + c] = e;
            denom += e;
        }}
        if (denom <= 0.0f || !isfinite(denom)) {{
            for (int c = 0; c < cols; ++c) {{
                probs[r * cols + c] = (c == r) ? 1.0f : 0.0f;
            }}
        }} else {{
            float inv_denom = reciprocal_nr_f32(denom);
            for (int c = 0; c < cols; ++c) {{
                probs[r * cols + c] = probs[r * cols + c] * inv_denom;
            }}
        }}
    }}
}}

static void causal_attention_fused_f32(
    const float *q,
    const float *k,
    const float *v,
    float *attn,
    float *scores,
    float *probs,
    int token_count,
    int d_head)
{{
    for (int r = 0; r < token_count; ++r) {{
        float *score_row = scores + r * token_count;
        float *prob_row = probs + r * token_count;
        float maxv = -1.0e30f;
        for (int c = 0; c < token_count; ++c) {{
            float s = (c > r) ? -1.0e9f : 0.0f;
            if (c <= r) {{
                float dot = 0.0f;
                for (int d = 0; d < d_head; ++d) {{
                    dot += q[r * d_head + d] * k[c * d_head + d];
                }}
                s = dot;
            }}
            score_row[c] = s;
            if (s > maxv) maxv = s;
        }}

        float denom = 0.0f;
        float dbg_e0 = 0.0f;
        for (int c = 0; c < token_count; ++c) {{
            float z = score_row[c] - maxv;
            if (z > 80.0f) z = 80.0f;
            if (z < -80.0f) z = -80.0f;
            float e = exp_approx_f32(z);
            if (r == 0 && c == 0) {{
                dbg_e0 = e;
            }}
            prob_row[c] = e;
            denom += e;
        }}
        if (g_diag_attention && r == 0) {{
            printf("cpu_native.diag fused row0 e0=%f denom=%f score00=%f score01=%f\\n",
                   (double)dbg_e0, (double)denom, (double)score_row[0], (double)score_row[1]);
        }}

        for (int d = 0; d < d_head; ++d) {{
            attn[r * d_head + d] = 0.0f;
        }}
        if (denom <= 0.0f || !isfinite(denom)) {{
            for (int d = 0; d < d_head; ++d) {{
                attn[r * d_head + d] = v[r * d_head + d];
            }}
            continue;
        }}

        float inv_denom = reciprocal_nr_f32(denom);
        if (g_diag_attention && r == 0) {{
            printf("cpu_native.diag fused row0 inv_denom=%f\\n", (double)inv_denom);
        }}
        for (int c = 0; c <= r; ++c) {{
            float w = prob_row[c] * inv_denom;
            for (int d = 0; d < d_head; ++d) {{
                attn[r * d_head + d] += w * v[c * d_head + d];
            }}
        }}
        for (int c = r + 1; c < token_count; ++c) {{
            prob_row[c] = 0.0f;
        }}
        for (int c = 0; c <= r; ++c) {{
            prob_row[c] *= inv_denom;
        }}
    }}
}}

static void gelu_f32(const float *x, float *out, int elems)
{{
    const float inv_sqrt2 = 0.70710678118f;
    for (int i = 0; i < elems; ++i) {{
        out[i] = 0.5f * x[i] * (1.0f + erff(x[i] * inv_sqrt2));
    }}
}}

static float max_abs_diff(const float *a, const float *b, int elems)
{{
    float maxv = 0.0f;
    for (int i = 0; i < elems; ++i) {{
        float d = fabsf(a[i] - b[i]);
        if (d > maxv) maxv = d;
    }}
    return maxv;
}}

{chr(10).join(decls)}

int main(void)
{{
    uint32_t cycle_t0;
    uint32_t cycle_t1;
    const int token_count = {token_count};
    const int d_model = {d_model};
    const int d_head = {d_head};
    const int n_heads = {n_heads};
    const int ffn_dim = {ffn_dim};
    const int diagnose_attention = {1 if diagnose_attention else 0};
    g_diag_attention = diagnose_attention;

    printf("Native CPU float program: {program_name}\\n");

    cycle_t0 = read_mcycle32();
    add_f32(x_f, pos_emb, x_pos_buf, token_count * d_model);
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("cpu_native.add_pos", cycle_t0, cycle_t1);
    if (diagnose_attention) {{
        printf("cpu_native.diag ptrs probs=%p attn_h0=%p v=%p scores=%p\\n",
               (void *)probs_buf, (void *)attn_h0_buf, (void *)v_buf, (void *)scores_buf);
        float d_xpos = max_abs_diff(x_pos_buf, x_pos, token_count * d_model);
        printf("cpu_native.diag d_x_pos=%f\\n", (double)d_xpos);
    }}

    cycle_t0 = read_mcycle32();
    layernorm_f32(x_pos_buf, ln1_wb, x_norm1_buf, token_count, d_model, 1.0e-5f);
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("cpu_native.layernorm1", cycle_t0, cycle_t1);
    if (diagnose_attention) {{
        float d_xnorm1 = max_abs_diff(x_norm1_buf, x_norm1, token_count * d_model);
        printf("cpu_native.diag d_x_norm1=%f\\n", (double)d_xnorm1);
        printf("cpu_native.diag x_norm1 row0=%f,%f expected=%f,%f\\n",
               (double)x_norm1_buf[0], (double)x_norm1_buf[1],
               (double)x_norm1[0], (double)x_norm1[1]);
        {{
            float mean0 = 0.0f;
            for (int c = 0; c < d_model; ++c) {{
                mean0 += x_pos_buf[c];
            }}
            mean0 /= (float)d_model;
            float var0 = 0.0f;
            for (int c = 0; c < d_model; ++c) {{
                float centered = x_pos_buf[c] - mean0;
                var0 += centered * centered;
            }}
            var0 /= (float)d_model;
            float inv_std0 = inv_sqrt_nr_f32(var0 + 1.0e-5f);
            printf("cpu_native.diag ln1_stats row0 mean=%f var=%f inv_std=%f expected=%f,%f,%f\\n",
                   (double)mean0, (double)var0, (double)inv_std0,
                   (double)ln1_mean_expected[0], (double)ln1_var_expected[0], (double)ln1_inv_std_expected[0]);
        }}
    }}

    cycle_t0 = read_mcycle32();
    for (int head = 0; head < n_heads; ++head) {{
        const float *w_q = w_q_h0;
        const float *w_k = w_k_h0;
        const float *w_v = w_v_h0;
        const float *q_expected = q_h0_expected;
        const float *k_expected = k_h0_expected;
        const float *v_expected = v_h0_expected;
        const float *scores_expected = scores_h0_expected;
        const float *probs_expected = probs_h0_expected;
        const float *attn_expected = attn_h0_expected;
        float *attn_out = attn_h0_buf;
{chr(10).join(head_weight_select)}
"""
    for head_idx in range(n_heads):
        source += f"""
        if (head == {head_idx}) {{
            q_expected = q_h{head_idx}_expected;
            k_expected = k_h{head_idx}_expected;
            v_expected = v_h{head_idx}_expected;
            scores_expected = scores_h{head_idx}_expected;
            probs_expected = probs_h{head_idx}_expected;
            attn_expected = attn_h{head_idx}_expected;
        }}"""
    source += f"""
        matmul_f32(x_norm1_buf, w_q, q_buf, token_count, d_model, d_head);
        matmul_f32(x_norm1_buf, w_k, k_buf, token_count, d_model, d_head);
        matmul_f32(x_norm1_buf, w_v, v_buf, token_count, d_model, d_head);
        causal_attention_fused_f32(q_buf, k_buf, v_buf, attn_out, scores_buf, probs_buf, token_count, d_head);
        if (diagnose_attention) {{
            float d_q = max_abs_diff(q_buf, q_expected, token_count * d_head);
            float d_k = max_abs_diff(k_buf, k_expected, token_count * d_head);
            float d_v = max_abs_diff(v_buf, v_expected, token_count * d_head);
            float d_scores = max_abs_diff(scores_buf, scores_expected, token_count * token_count);
            float d_probs = max_abs_diff(probs_buf, probs_expected, token_count * token_count);
            float d_attn = max_abs_diff(attn_out, attn_expected, token_count * d_head);
            printf("cpu_native.diag head=%d d_q=%f d_k=%f d_v=%f d_scores=%f d_probs=%f d_attn=%f\\n",
                   head, (double)d_q, (double)d_k, (double)d_v, (double)d_scores, (double)d_probs, (double)d_attn);
            printf("cpu_native.diag row0 probs=%f,%f attn=%f,%f expected_attn=%f,%f\\n",
                   (double)probs_buf[0], (double)probs_buf[1],
                   (double)attn_out[0], (double)attn_out[1],
                   (double)attn_expected[0], (double)attn_expected[1]);
        }}
    }}
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("cpu_native.attention", cycle_t0, cycle_t1);

    cycle_t0 = read_mcycle32();
"""

    if n_heads == 1:
        source += f"""    for (int i = 0; i < token_count * d_head; ++i) {{
        attn_cat_buf[i] = attn_h0_buf[i];
    }}
"""
    else:
        source += "    for (int head = 0; head < n_heads; ++head) {\n"
        source += "        const float *src = attn_h0_buf;\n"
        source += "\n".join(head_attn_select) + "\n"
        source += "        for (int r = 0; r < token_count; ++r) {\n"
        source += "            for (int c = 0; c < d_head; ++c) {\n"
        source += "                attn_cat_buf[r * (n_heads * d_head) + head * d_head + c] = src[r * d_head + c];\n"
        source += "            }\n"
        source += "        }\n"
        source += "    }\n"

    if stop_after_attention:
        source += """    {
        float diff_attn = max_abs_diff(attn_cat_buf, attn_cat_expected, token_count * n_heads * d_head);
        printf("cpu_native.attention_max_abs_diff=%f\\n", (double)diff_attn);
        if (diff_attn > 1.0e-3f) {
            printf("verification failed: native_cpu_attention\\n");
            printf("probs row0:");
            for (int i = 0; i < token_count && i < 8; ++i) {
                printf(" %f", (double)probs_buf[i]);
            }
            printf("\\nv row0:");
            for (int i = 0; i < d_head && i < 8; ++i) {
                printf(" %f", (double)v_buf[i]);
            }
            printf("\\n");
            printf("attn row0:");
            for (int i = 0; i < d_head && i < 8; ++i) {
                printf(" %f", (double)attn_h0_buf[i]);
            }
            printf("\\nattn_expected row0:");
            for (int i = 0; i < d_head && i < 8; ++i) {
                printf(" %f", (double)attn_h0_expected[i]);
            }
            printf("\\n");
            return EXIT_FAILURE;
        }
        printf("EXIT SUCCESS\\n");
        return EXIT_SUCCESS;
    }
"""

    source += f"""    matmul_f32(attn_cat_buf, w_o, o_buf, token_count, n_heads * d_head, d_model);
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("cpu_native.o_proj", cycle_t0, cycle_t1);

    cycle_t0 = read_mcycle32();
    add_f32(x_pos_buf, o_buf, resid1_buf, token_count * d_model);
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("cpu_native.residual1", cycle_t0, cycle_t1);

    cycle_t0 = read_mcycle32();
    layernorm_f32(resid1_buf, ln2_wb, x_norm2_buf, token_count, d_model, 1.0e-5f);
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("cpu_native.layernorm2", cycle_t0, cycle_t1);

    cycle_t0 = read_mcycle32();
    matmul_f32(x_norm2_buf, w_fc, ffn_fc_buf, token_count, d_model, ffn_dim);
    gelu_f32(ffn_fc_buf, ffn_gelu_buf, token_count * ffn_dim);
    matmul_f32(ffn_gelu_buf, w_proj, ffn_out_buf, token_count, ffn_dim, d_model);
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("cpu_native.ffn", cycle_t0, cycle_t1);

    cycle_t0 = read_mcycle32();
    add_f32(resid1_buf, ffn_out_buf, out_buf, token_count * d_model);
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("cpu_native.residual2", cycle_t0, cycle_t1);

    {{
        float diff = max_abs_diff(out_buf, out_expected, token_count * d_model);
        printf("cpu_native.max_abs_diff=%f\\n", (double)diff);
        if (diff > 1.0e-3f) {{
            printf("verification failed: native_cpu_out\\n");
            printf("q row0:");
            for (int i = 0; i < d_head && i < 8; ++i) {{
                printf(" %f", (double)q_buf[i]);
            }}
            printf("\\nk row0:");
            for (int i = 0; i < d_head && i < 8; ++i) {{
                printf(" %f", (double)k_buf[i]);
            }}
            printf("\\nv row0:");
            for (int i = 0; i < d_head && i < 8; ++i) {{
                printf(" %f", (double)v_buf[i]);
            }}
            printf("\\nprobs row0:");
            for (int i = 0; i < token_count && i < 8; ++i) {{
                printf(" %f", (double)probs_buf[i]);
            }}
            printf("\\nattn row0:");
            for (int i = 0; i < d_head && i < 8; ++i) {{
                printf(" %f", (double)attn_h0_buf[i]);
            }}
            printf("\\n");
            printf("o row0:");
            for (int i = 0; i < d_model && i < 8; ++i) {{
                printf(" %f", (double)o_buf[i]);
            }}
            printf("\\nffn_out row0:");
            for (int i = 0; i < d_model && i < 8; ++i) {{
                printf(" %f", (double)ffn_out_buf[i]);
            }}
            printf("\\n");
            printf("actual row0:");
            for (int i = 0; i < d_model && i < 8; ++i) {{
                printf(" %f", (double)out_buf[i]);
            }}
            printf("\\nexpected row0:");
            for (int i = 0; i < d_model && i < 8; ++i) {{
                printf(" %f", (double)out_expected[i]);
            }}
            printf("\\n");
            return EXIT_FAILURE;
        }}
    }}

    printf("EXIT SUCCESS\\n");
    return EXIT_SUCCESS;
}}
"""
    return source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--d-head", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--ffn-dim", type=int, default=8)
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", choices=("py", "host", "rtl"), default="py")
    parser.add_argument("--stop-after", choices=("attention", "full"), default="full")
    parser.add_argument("--reuse-verilator", action="store_true")
    parser.add_argument("--diagnose-attention", action="store_true")
    parser.add_argument("--maxcycles", type=int, default=3000000)
    parser.add_argument("--verilator-max-ticks", type=int, default=30000000000)
    args = parser.parse_args()

    tensors, expected = _native_float_reference(
        d_model=args.d_model,
        d_head=args.d_head,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        token_count=args.token_count,
        seed=args.seed,
    )

    program_name = (
        f"cv32e40p_prefill_transformer_block_cpu_native_d{args.d_model}"
        f"_h{args.d_head}_nh{args.n_heads}_f{args.ffn_dim}_t{args.token_count}_s{args.seed}"
    )
    GENERATED_DIR.mkdir(exist_ok=True)
    program_path = GENERATED_DIR / f"{program_name}.c"
    program_path.write_text(
        _emit_native_source(
            program_name=program_name,
            d_model=args.d_model,
            d_head=args.d_head,
            n_heads=args.n_heads,
            ffn_dim=args.ffn_dim,
            token_count=args.token_count,
            tensors=tensors,
            expected=expected,
            host_native=args.mode == "host",
            stop_after_attention=args.stop_after == "attention",
            diagnose_attention=args.diagnose_attention,
        )
    )

    if args.mode == "py":
        print(f"program={program_name}")
        print(
            f"d_model={args.d_model} d_head={args.d_head} n_heads={args.n_heads} "
            f"ffn_dim={args.ffn_dim} token_count={args.token_count} seed={args.seed}"
        )
        print(f"expected_checksum={float(np.array(expected, dtype=np.float32).sum()):.6f}")
        print(f"attention_checksum={float(np.array(tensors['attn_cat_expected'], dtype=np.float32).sum()):.6f}")
        return 0

    if args.mode == "host":
        host_bin = GENERATED_DIR / f"{program_name}_host"
        subprocess.run(
            ["gcc", "-O3", "-std=c11", str(program_path), "-lm", "-o", str(host_bin)],
            check=True,
            cwd=REPO_ROOT,
        )
        proc = subprocess.run(
            [str(host_bin)],
            check=False,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.stderr:
            print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n")
        return proc.returncode

    prefix = _toolchain_prefix()
    gcc = f"{prefix}gcc"
    objcopy = f"{prefix}objcopy"
    toolchain_root = _toolchain_root(prefix)
    include_dir = toolchain_root / "riscv32-unknown-elf" / "include"
    lib_dir = toolchain_root / "riscv32-unknown-elf" / "lib"
    elf_path = CUSTOM_DIR / f"{program_name}.elf"
    hex_path = CUSTOM_DIR / f"{program_name}.hex"

    build_env = dict(os.environ)
    build_env["CCACHE_DISABLE"] = "1"
    build_env["TMPDIR"] = "/tmp"
    if not args.reuse_verilator:
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
            "mem_stall/mem_stall.c",
            "custom/syscalls.c",
            "custom/vectors.S",
            "-I",
            str(include_dir),
            "-I",
            "mem_stall",
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
    print(proc.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
