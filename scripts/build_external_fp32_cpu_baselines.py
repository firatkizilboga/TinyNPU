from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))
if str(REPO_ROOT / "software" / "workload") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "workload"))

import stories  # noqa: E402
from tinynpu_jit.rtl_runner import (  # noqa: E402
    CORE_DIR,
    CUSTOM_DIR,
    GENERATED_DIR,
    RUNTIME_DIR,
    TNPU_RISCV_MABI,
    TNPU_RISCV_MARCH,
    run_checked,
    run_vlt_npu,
    toolchain_include_lib_dirs,
    toolchain_prefix,
)


DATA_DIR = REPO_ROOT / "data" / "MNIST" / "raw"
TINYSTORIES_RUN_DIR = REPO_ROOT / "runs" / "tinystories_word_lm_d32_t17_qat_int16_long"


def _fmt_f32(values: np.ndarray, *, per_line: int = 8) -> str:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)

    def lit(value: np.float32) -> str:
        text = f"{float(value):.9g}"
        if "e" not in text and "." not in text:
            text += ".0"
        return text + "f"

    lines: list[str] = []
    for start in range(0, flat.size, per_line):
        chunk = flat[start : start + per_line]
        lines.append("    " + ", ".join(lit(v) for v in chunk))
    return ",\n".join(lines)


def _c_array(name: str, values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    return f"static const float {name}[{flat.size}] = {{\n{_fmt_f32(flat)}\n}};"


def _f32_lit(value: float) -> str:
    text = f"{float(value):.9g}"
    if "e" not in text and "." not in text:
        text += ".0"
    return text + "f"


def _c_int_array(name: str, values: list[int]) -> str:
    body = ", ".join(str(int(v)) for v in values)
    return f"static const int {name}[{len(values)}] = {{ {body} }};"


def _c_string_array(name: str, values: list[str]) -> str:
    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"')

    body = ", ".join(f'"{esc(value)}"' for value in values)
    return f"static const char *const {name}[{len(values)}] = {{ {body} }};"


def _common_c_prefix(title: str) -> str:
    return f"""// Standalone fp32 CPU baseline generated outside the TinyNPU compiler.
// Model: {title}
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

uint32_t runtime_cycle_start;
uint32_t runtime_cycle_post_bss;
uint32_t runtime_cycle_post_init;
uint32_t runtime_cycle_pre_main;

#define TIMER_CTRL  ((volatile uint32_t *) 0x15000000u)
#define TIMER_VALUE ((volatile uint32_t *) 0x15000004u)
#define TIMER_COUNT ((volatile uint32_t *) 0x15001000u)

static inline uint32_t read_mcycle32(void) {{ return *TIMER_COUNT; }}

static void reset_timer(void)
{{
    *TIMER_CTRL = 0u;
    *TIMER_VALUE = 0xFFFFFFFFu;
    while (*TIMER_COUNT == 0u) {{ }}
}}

"""


def _load_mnist_sample_flat(sample_index: int = 0, image_size: int = 8) -> tuple[np.ndarray, int]:
    from software.workload.mnist_mlp_feature_benchmark import get_flat_mnist_loaders, TASK_IS_ZERO

    _, _, _, _, test_ds = get_flat_mnist_loaders(str(REPO_ROOT / "data"), image_size=image_size, task=TASK_IS_ZERO)
    image, label = test_ds[sample_index]
    return image.numpy().astype(np.float32), int(label)


def _load_mnist_sample_image(sample_index: int = 0, image_size: int = 8) -> tuple[np.ndarray, int]:
    flat, label = _load_mnist_sample_flat(sample_index=sample_index, image_size=image_size)
    return flat.reshape(1, image_size, image_size), label


def render_iszero_mlp(sample_index: int = 0) -> str:
    x, label = _load_mnist_sample_flat(sample_index=sample_index, image_size=8)
    return (
        _common_c_prefix("is_zero MLP, fp32 C/libm")
        + _c_array("input_x", x)
        + f"""

#define IN_DIM 64
#define HIDDEN 64
#define OUT_DIM 1

static float fc1_w[HIDDEN * IN_DIM], fc1_b[HIDDEN];
static float fc2_w[HIDDEN * HIDDEN], fc2_b[HIDDEN];
static float fc3_w[HIDDEN * HIDDEN], fc3_b[HIDDEN];
static float fc4_w[OUT_DIM * HIDDEN], fc4_b[OUT_DIM];
static float h1[HIDDEN], h2[HIDDEN], h3[HIDDEN], out[OUT_DIM];

static float init_weight(int idx, int mod, float scale)
{{
    return ((float)((idx * 37 + 11) % mod) - (float)(mod / 2)) * scale;
}}

static void init_weights(void)
{{
    for (int i = 0; i < HIDDEN * IN_DIM; ++i) fc1_w[i] = init_weight(i, 23, 0.0075f);
    for (int i = 0; i < HIDDEN * HIDDEN; ++i) fc2_w[i] = init_weight(i, 29, 0.0065f);
    for (int i = 0; i < HIDDEN * HIDDEN; ++i) fc3_w[i] = init_weight(i, 31, 0.0060f);
    for (int i = 0; i < OUT_DIM * HIDDEN; ++i) fc4_w[i] = init_weight(i, 17, 0.0100f);
    for (int i = 0; i < HIDDEN; ++i) {{
        fc1_b[i] = init_weight(i, 19, 0.002f);
        fc2_b[i] = init_weight(i + 101, 19, 0.002f);
        fc3_b[i] = init_weight(i + 211, 19, 0.002f);
    }}
    fc4_b[0] = -0.05f;
}}

static float relu(float x) {{ return x > 0.0f ? x : 0.0f; }}
static float sigmoidf_ref(float x) {{ return 1.0f / (1.0f + expf(-x)); }}
static float geluf_ref(float x)
{{
    const float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}}

static void linear(const float *x, const float *w, const float *b, float *y, int in_dim, int out_dim)
{{
    for (int o = 0; o < out_dim; ++o) {{
        float acc = b[o];
        for (int i = 0; i < in_dim; ++i) acc += w[o * in_dim + i] * x[i];
        y[o] = acc;
    }}
}}

static void run_model(void)
{{
    linear(input_x, fc1_w, fc1_b, h1, IN_DIM, HIDDEN);
    for (int i = 0; i < HIDDEN; ++i) h1[i] = relu(h1[i]);
    linear(h1, fc2_w, fc2_b, h2, HIDDEN, HIDDEN);
    for (int i = 0; i < HIDDEN; ++i) h2[i] = relu(h2[i]);
    linear(h2, fc3_w, fc3_b, h3, HIDDEN, HIDDEN);
    for (int i = 0; i < HIDDEN; ++i) h3[i] = geluf_ref(h3[i]);
    linear(h3, fc4_w, fc4_b, out, HIDDEN, OUT_DIM);
    out[0] = sigmoidf_ref(out[0]);
}}

int main(void)
{{
    init_weights();
    reset_timer();
    uint32_t t0 = read_mcycle32();
    run_model();
    uint32_t t1 = read_mcycle32();
    uint32_t cycles = t0 - t1;
    printf("external_fp32_iszero_mlp sample_index={sample_index} label={label} cycles=%lu output=%.9g pred=%d\\n",
           (unsigned long)cycles, (double)out[0], out[0] >= 0.5f);
    puts("EXIT SUCCESS");
    return 0;
}}
"""
    )


def render_conv(sample_index: int = 0) -> str:
    x, label = _load_mnist_sample_image(sample_index=sample_index, image_size=8)
    return (
        _common_c_prefix("4-layer conv is_zero, fp32 C/libm")
        + _c_array("input_x", x)
        + f"""

static float conv1_w[16 * 1 * 3 * 3], conv1_b[16];
static float conv2_w[16 * 16 * 3 * 3], conv2_b[16];
static float conv3_w[16 * 16 * 3 * 3], conv3_b[16];
static float conv4_w[1 * 16 * 2 * 2], conv4_b[1];
static float a1[16 * 6 * 6], a2[16 * 4 * 4], a3[16 * 2 * 2], out[1];

static float init_weight(int idx, int mod, float scale)
{{
    return ((float)((idx * 31 + 7) % mod) - (float)(mod / 2)) * scale;
}}

static void init_weights(void)
{{
    for (int i = 0; i < 16 * 1 * 3 * 3; ++i) conv1_w[i] = init_weight(i, 23, 0.020f);
    for (int i = 0; i < 16 * 16 * 3 * 3; ++i) conv2_w[i] = init_weight(i, 29, 0.006f);
    for (int i = 0; i < 16 * 16 * 3 * 3; ++i) conv3_w[i] = init_weight(i, 31, 0.006f);
    for (int i = 0; i < 1 * 16 * 2 * 2; ++i) conv4_w[i] = init_weight(i, 17, 0.012f);
    for (int i = 0; i < 16; ++i) {{
        conv1_b[i] = init_weight(i, 19, 0.002f);
        conv2_b[i] = init_weight(i + 101, 19, 0.002f);
        conv3_b[i] = init_weight(i + 211, 19, 0.002f);
    }}
    conv4_b[0] = -0.05f;
}}

static float relu(float x) {{ return x > 0.0f ? x : 0.0f; }}
static float sigmoidf_ref(float x) {{ return 1.0f / (1.0f + expf(-x)); }}

static void conv2d_valid(
    const float *x, const float *w, const float *b, float *y,
    int in_c, int in_h, int in_w, int out_c, int k)
{{
    int out_h = in_h - k + 1;
    int out_w = in_w - k + 1;
    for (int oc = 0; oc < out_c; ++oc) {{
        for (int oh = 0; oh < out_h; ++oh) {{
            for (int ow = 0; ow < out_w; ++ow) {{
                float acc = b[oc];
                for (int ic = 0; ic < in_c; ++ic) {{
                    for (int kh = 0; kh < k; ++kh) {{
                        for (int kw = 0; kw < k; ++kw) {{
                            int xi = (ic * in_h + (oh + kh)) * in_w + (ow + kw);
                            int wi = ((oc * in_c + ic) * k + kh) * k + kw;
                            acc += x[xi] * w[wi];
                        }}
                    }}
                }}
                y[(oc * out_h + oh) * out_w + ow] = acc;
            }}
        }}
    }}
}}

static void run_model(void)
{{
    conv2d_valid(input_x, conv1_w, conv1_b, a1, 1, 8, 8, 16, 3);
    for (int i = 0; i < 16 * 6 * 6; ++i) a1[i] = relu(a1[i]);
    conv2d_valid(a1, conv2_w, conv2_b, a2, 16, 6, 6, 16, 3);
    for (int i = 0; i < 16 * 4 * 4; ++i) a2[i] = relu(a2[i]);
    conv2d_valid(a2, conv3_w, conv3_b, a3, 16, 4, 4, 16, 3);
    for (int i = 0; i < 16 * 2 * 2; ++i) a3[i] = relu(a3[i]);
    conv2d_valid(a3, conv4_w, conv4_b, out, 16, 2, 2, 1, 2);
    out[0] = sigmoidf_ref(out[0]);
}}

int main(void)
{{
    init_weights();
    reset_timer();
    uint32_t t0 = read_mcycle32();
    run_model();
    uint32_t t1 = read_mcycle32();
    uint32_t cycles = t0 - t1;
    printf("external_fp32_conv4 sample_index={sample_index} label={label} cycles=%lu output=%.9g pred=%d\\n",
           (unsigned long)cycles, (double)out[0], out[0] >= 0.5f);
    puts("EXIT SUCCESS");
    return 0;
}}
"""
    )


def _encode_prompt(run_dir: Path, prompt: str, prompt_len: int) -> tuple[list[int], list[str]]:
    ckpt = np.load(run_dir / "tinystories_char_lm_fp32.npz")
    del ckpt
    meta = json.loads((run_dir / "tinystories_char_lm_config.json").read_text())
    tokenizer = stories.tokenizer_from_json(meta["tokenizer"])
    ids = tokenizer.encode(stories.normalize_text(prompt).rstrip("\n"), add_bos=True, add_eos=False)
    if len(ids) < prompt_len:
        raise ValueError(f"prompt encodes to {len(ids)} tokens, need at least {prompt_len}")
    return ids[:prompt_len], list(tokenizer.itos)


def render_tiny_lm(run_dir: Path = TINYSTORIES_RUN_DIR, prompt: str = "there was a little girl named lily .", prompt_len: int = 9) -> str:
    meta = json.loads((run_dir / "tinystories_char_lm_config.json").read_text())
    cfg = meta["config"]
    if (cfg["d_model"], cfg["d_head"], cfg["n_heads"], cfg["n_kv_heads"], cfg["ffn_hidden_dim"], cfg["n_layers"]) != (32, 16, 2, 1, 32, 2):
        raise NotImplementedError("standalone TinyLM C emitter currently expects d32 h16 nh2 nkv1 f32 nl2")
    ids, vocab = _encode_prompt(run_dir, prompt, prompt_len)
    fp = np.load(run_dir / "tinystories_char_lm_fp32.npz")
    arrays = [
        _c_array("tok_embeddings", fp["tok_embeddings.weight"]),
        _c_array("norm_w", fp["norm.weight"]),
        _c_array("lm_head_w", fp["lm_head.weight"]),
        _c_int_array("input_ids", ids),
        _c_string_array("vocab_tokens", vocab),
    ]
    for layer in range(2):
        prefix = f"layers.{layer}"
        cprefix = f"l{layer}"
        for suffix, cname in [
            ("rms1.weight", "rms1_w"),
            ("attn.q_proj.weight", "q_w"),
            ("attn.k_proj.weight", "k_w"),
            ("attn.v_proj.weight", "v_w"),
            ("attn.o_proj.weight", "o_w"),
            ("rms2.weight", "rms2_w"),
            ("mlp.gate_proj.weight", "gate_w"),
            ("mlp.up_proj.weight", "up_w"),
            ("mlp.down_proj.weight", "down_w"),
        ]:
            arrays.append(_c_array(f"{cprefix}_{cname}", fp[f"{prefix}.{suffix}"]))
    return (
        _common_c_prefix("TinyStories toy LM full fp32 prefill, C/libm")
        + "\n\n".join(arrays)
        + f"""

#define S {prompt_len}
#define D 32
#define DH 16
#define NH 2
#define F 32
#define V 128
#define RMS_EPS {_f32_lit(float(cfg["rms_norm_eps"]))}
#define ROPE_THETA {_f32_lit(float(cfg["rope_theta"]))}

static float x[S * D], x_norm[S * D], q[S * NH * DH], k[S * DH], v[S * DH];
static float attn[S * D], resid[S * D], gate[S * F], up[S * F], hidden[S * F];
static float logits[V];

static void matmul_seq(const float *inp, const float *w, float *out, int seq, int in_dim, int out_dim)
{{
    for (int t = 0; t < seq; ++t) {{
        for (int o = 0; o < out_dim; ++o) {{
            float acc = 0.0f;
            for (int i = 0; i < in_dim; ++i) acc += w[o * in_dim + i] * inp[t * in_dim + i];
            out[t * out_dim + o] = acc;
        }}
    }}
}}

static void rmsnorm_seq(const float *inp, const float *w, float *out, int seq)
{{
    for (int t = 0; t < seq; ++t) {{
        float ss = 0.0f;
        for (int i = 0; i < D; ++i) ss += inp[t * D + i] * inp[t * D + i];
        float scale = 1.0f / sqrtf(ss / (float)D + RMS_EPS);
        for (int i = 0; i < D; ++i) out[t * D + i] = inp[t * D + i] * scale * w[i];
    }}
}}

static void apply_rope_qk(void)
{{
    for (int t = 0; t < S; ++t) {{
        for (int h = 0; h < NH; ++h) {{
            for (int i = 0; i < DH / 2; ++i) {{
                float angle = (float)t / powf(ROPE_THETA, (float)i / (float)(DH / 2));
                float c = cosf(angle), s = sinf(angle);
                int base = (t * NH + h) * DH;
                float x0 = q[base + i];
                float x1 = q[base + i + DH / 2];
                q[base + i] = x0 * c - x1 * s;
                q[base + i + DH / 2] = x1 * c + x0 * s;
            }}
        }}
        for (int i = 0; i < DH / 2; ++i) {{
            float angle = (float)t / powf(ROPE_THETA, (float)i / (float)(DH / 2));
            float c = cosf(angle), s = sinf(angle);
            int base = t * DH;
            float x0 = k[base + i];
            float x1 = k[base + i + DH / 2];
            k[base + i] = x0 * c - x1 * s;
            k[base + i + DH / 2] = x1 * c + x0 * s;
        }}
    }}
}}

static void attention(void)
{{
    for (int i = 0; i < S * D; ++i) attn[i] = 0.0f;
    for (int h = 0; h < NH; ++h) {{
        for (int t = 0; t < S; ++t) {{
            float scores[S];
            float maxv = -3.402823466e+38f;
            for (int j = 0; j <= t; ++j) {{
                float acc = 0.0f;
                for (int d = 0; d < DH; ++d) {{
                    acc += q[(t * NH + h) * DH + d] * k[j * DH + d];
                }}
                scores[j] = acc / 4.0f;
                if (scores[j] > maxv) maxv = scores[j];
            }}
            float sum = 0.0f;
            for (int j = 0; j <= t; ++j) {{
                scores[j] = expf(scores[j] - maxv);
                sum += scores[j];
            }}
            for (int j = 0; j <= t; ++j) {{
                float p = scores[j] / sum;
                for (int d = 0; d < DH; ++d) {{
                    attn[t * D + h * DH + d] += p * v[j * DH + d];
                }}
            }}
        }}
    }}
}}

static void add_seq(const float *a, const float *b, float *out)
{{
    for (int i = 0; i < S * D; ++i) out[i] = a[i] + b[i];
}}

static void silu_mul(void)
{{
    for (int i = 0; i < S * F; ++i) {{
        float g = gate[i];
        hidden[i] = (g / (1.0f + expf(-g))) * up[i];
    }}
}}

static void layer(
    const float *rms1_w, const float *q_w, const float *k_w, const float *v_w, const float *o_w,
    const float *rms2_w, const float *gate_w, const float *up_w, const float *down_w)
{{
    rmsnorm_seq(x, rms1_w, x_norm, S);
    matmul_seq(x_norm, q_w, q, S, D, NH * DH);
    matmul_seq(x_norm, k_w, k, S, D, DH);
    matmul_seq(x_norm, v_w, v, S, D, DH);
    apply_rope_qk();
    attention();
    matmul_seq(attn, o_w, resid, S, D, D);
    add_seq(x, resid, x);
    rmsnorm_seq(x, rms2_w, x_norm, S);
    matmul_seq(x_norm, gate_w, gate, S, D, F);
    matmul_seq(x_norm, up_w, up, S, D, F);
    silu_mul();
    matmul_seq(hidden, down_w, resid, S, F, D);
    add_seq(x, resid, x);
}}

static void run_model(void)
{{
    for (int t = 0; t < S; ++t) {{
        int tok = input_ids[t];
        for (int i = 0; i < D; ++i) x[t * D + i] = tok_embeddings[tok * D + i];
    }}
    layer(l0_rms1_w, l0_q_w, l0_k_w, l0_v_w, l0_o_w, l0_rms2_w, l0_gate_w, l0_up_w, l0_down_w);
    layer(l1_rms1_w, l1_q_w, l1_k_w, l1_v_w, l1_o_w, l1_rms2_w, l1_gate_w, l1_up_w, l1_down_w);
    rmsnorm_seq(x, norm_w, x_norm, S);
    const float *last = &x_norm[(S - 1) * D];
    for (int tok = 0; tok < V; ++tok) {{
        float acc = 0.0f;
        for (int i = 0; i < D; ++i) acc += lm_head_w[tok * D + i] * last[i];
        logits[tok] = acc;
    }}
}}

int main(void)
{{
    reset_timer();
    uint32_t t0 = read_mcycle32();
    run_model();
    uint32_t t1 = read_mcycle32();
    uint32_t cycles = t0 - t1;
    int best = 0;
    for (int i = 1; i < V; ++i) if (logits[i] > logits[best]) best = i;
    printf("external_fp32_tiny_lm prompt_len={prompt_len} cycles=%lu next_id=%d next_token='%s' logit=%.9g\\n",
           (unsigned long)cycles, best, vocab_tokens[best], (double)logits[best]);
    puts("EXIT SUCCESS");
    return 0;
}}
"""
    )


def render_model(model: str, *, sample_index: int, prompt: str, prompt_len: int, run_dir: Path) -> str:
    if model == "iszero_mlp":
        return render_iszero_mlp(sample_index=sample_index)
    if model == "conv":
        return render_conv(sample_index=sample_index)
    if model == "tiny_lm":
        return render_tiny_lm(run_dir=run_dir, prompt=prompt, prompt_len=prompt_len)
    raise ValueError(f"unknown model {model!r}")


def build_elf_and_hex(program_name: str, source: str) -> tuple[Path, Path, Path]:
    GENERATED_DIR.mkdir(exist_ok=True)
    CUSTOM_DIR.mkdir(exist_ok=True)
    source_path = GENERATED_DIR / f"{program_name}.c"
    source_path.write_text(source)

    prefix = toolchain_prefix()
    gcc = f"{prefix}gcc"
    objcopy = f"{prefix}objcopy"
    include_dir, lib_dir = toolchain_include_lib_dirs(prefix)
    elf_path = CUSTOM_DIR / f"{program_name}.elf"
    hex_path = CUSTOM_DIR / f"{program_name}.hex"
    build_env = dict(os.environ)
    build_env["CCACHE_DISABLE"] = "1"
    build_env["TMPDIR"] = "/tmp"
    run_checked(["make", "verilator-build-npu"], cwd=CORE_DIR, env=build_env)
    run_checked(
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
            str(source_path),
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
    run_checked([objcopy, "-O", "verilog", str(elf_path), str(hex_path)], cwd=CORE_DIR, env=build_env)
    return source_path, elf_path, hex_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("iszero_mlp", "conv", "tiny_lm", "all"), default="all")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--run-dir", type=Path, default=TINYSTORIES_RUN_DIR)
    parser.add_argument("--prompt", default="there was a little girl named lily .")
    parser.add_argument("--prompt-len", type=int, default=9)
    parser.add_argument("--run-rtl", action="store_true")
    parser.add_argument("--maxcycles", type=int, default=80_000_000)
    parser.add_argument("--verilator-max-ticks", type=int, default=120_000_000_000)
    parser.add_argument("--timeout-s", type=int, default=1200)
    args = parser.parse_args()

    models = ["iszero_mlp", "conv", "tiny_lm"] if args.model == "all" else [args.model]
    for model in models:
        program_name = f"external_fp32_cpu_{model}"
        source = render_model(
            model,
            sample_index=args.sample_index,
            prompt=args.prompt,
            prompt_len=args.prompt_len,
            run_dir=args.run_dir,
        )
        source_path, elf_path, hex_path = build_elf_and_hex(program_name, source)
        print(f"model={model}")
        print(f"source={source_path}")
        print(f"elf={elf_path}")
        print(f"hex={hex_path}")
        if args.run_rtl:
            try:
                proc = run_vlt_npu(
                    hex_path,
                    maxcycles=args.maxcycles,
                    verilator_max_ticks=args.verilator_max_ticks,
                    timeout_s=args.timeout_s,
                    noassert=True,
                )
            except subprocess.TimeoutExpired as exc:
                if exc.stdout:
                    print(exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout)
                if exc.stderr:
                    print(exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr, file=sys.stderr)
                raise
            print(proc.stdout)
            if proc.stderr:
                print(proc.stderr, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
