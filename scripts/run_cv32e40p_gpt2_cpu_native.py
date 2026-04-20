from __future__ import annotations

import argparse
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
from tinynpu_jit.blocks.gpt2_block import (  # noqa: E402
    build_shared_state,
    reference_decode_float,
    reference_prefill_float,
)


def _emit_f32_array(name: str, values: np.ndarray, *, section_data: bool) -> str:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    attr = ' __attribute__((section(".data")))' if section_data else ' __attribute__((section(".bss")))'
    if flat.size == 0:
        return f"static float {name}[1]{attr} = {{0.0f}};"
    if section_data:
        body = ", ".join(f"{float(v):.8e}f" for v in flat)
        return f"static float {name}[{flat.size}]{attr} = {{\n    {body}\n}};"
    return f"static float {name}[{flat.size}]{attr};"


def _build_case(*, mode: str, d_model: int, d_head: int, n_heads: int, ffn_dim: int, prompt_len: int, seed: int):
    state = build_shared_state(
        d_model=d_model,
        d_head=d_head,
        n_heads=n_heads,
        ffn_dim=ffn_dim,
        prompt_len=prompt_len,
        seed=seed,
    )
    block = state["block"]
    prefill_ref = reference_prefill_float(state, d_head=d_head, n_heads=n_heads)
    if mode == "prefill":
        expected = np.asarray(prefill_ref["out"], dtype=np.float32)
    else:
        expected = np.asarray(reference_decode_float(state, prefill_ref, d_head=d_head, n_heads=n_heads)["out"], dtype=np.float32)

    tensors: dict[str, np.ndarray] = {
        "x_in": np.asarray(state["x_prompt_in"] if mode == "prefill" else state["x_decode_in"], dtype=np.float32),
        "ln1_wb": np.asarray(block.ln_1_wb, dtype=np.float32),
        "ln2_wb": np.asarray(block.ln_2_wb, dtype=np.float32),
        "attn_c_attn_w": np.asarray(block.attn_c_attn_w, dtype=np.float32),
        "attn_c_attn_b": np.asarray(block.attn_c_attn_b, dtype=np.float32),
        "attn_c_proj_w": np.asarray(block.attn_c_proj_w, dtype=np.float32),
        "attn_c_proj_b": np.asarray(block.attn_c_proj_b, dtype=np.float32),
        "mlp_c_fc_w": np.asarray(block.mlp_c_fc_w, dtype=np.float32),
        "mlp_c_fc_b": np.asarray(block.mlp_c_fc_b, dtype=np.float32),
        "mlp_c_proj_w": np.asarray(block.mlp_c_proj_w, dtype=np.float32),
        "mlp_c_proj_b": np.asarray(block.mlp_c_proj_b, dtype=np.float32),
        "out_expected": expected,
    }
    if mode == "decode":
        cache_len = prompt_len + 1
        for head_idx in range(n_heads):
            k_base = np.zeros((cache_len, d_head), dtype=np.float32)
            v_base = np.zeros((cache_len, d_head), dtype=np.float32)
            k_prefill = np.asarray(prefill_ref["k_heads"][head_idx], dtype=np.float32)
            v_prefill = np.asarray(prefill_ref["v_heads"][head_idx], dtype=np.float32)
            k_base[:prompt_len, :] = k_prefill
            v_base[:prompt_len, :] = v_prefill
            tensors[f"k_cache_init_h{head_idx}"] = k_base
            tensors[f"v_cache_init_h{head_idx}"] = v_base
    return tensors, expected


def _emit_source(
    *,
    program_name: str,
    mode: str,
    host_native: bool,
    d_model: int,
    d_head: int,
    n_heads: int,
    ffn_dim: int,
    prompt_len: int,
    tensors: dict[str, np.ndarray],
    expected: np.ndarray,
) -> str:
    seq_len = prompt_len if mode == "prefill" else 1
    attn_dim = n_heads * d_head
    cache_len = prompt_len if mode == "prefill" else prompt_len + 1

    decls: list[str] = []
    for name, values in tensors.items():
        decls.append(_emit_f32_array(name, values, section_data=True))

    scratch_shapes = {
        "x_norm1_buf": (seq_len, d_model),
        "qkv_buf": (seq_len, 3 * attn_dim),
        "q_buf": (seq_len, attn_dim),
        "k_buf": (seq_len, attn_dim),
        "v_buf": (seq_len, attn_dim),
        "scores_buf": (seq_len, cache_len),
        "probs_buf": (seq_len, cache_len),
        "attn_cat_buf": (seq_len, attn_dim),
        "o_buf": (seq_len, d_model),
        "resid1_buf": (seq_len, d_model),
        "x_norm2_buf": (seq_len, d_model),
        "ffn_fc_buf": (seq_len, ffn_dim),
        "ffn_gelu_buf": (seq_len, ffn_dim),
        "ffn_out_buf": (seq_len, d_model),
        "out_buf": (seq_len, d_model),
    }
    for head_idx in range(n_heads):
        if mode == "prefill":
            scratch_shapes[f"k_cache_h{head_idx}_buf"] = (prompt_len, d_head)
            scratch_shapes[f"v_cache_h{head_idx}_buf"] = (prompt_len, d_head)
        else:
            scratch_shapes[f"k_cache_h{head_idx}_buf"] = (prompt_len + 1, d_head)
            scratch_shapes[f"v_cache_h{head_idx}_buf"] = (prompt_len + 1, d_head)
    for name, shape in scratch_shapes.items():
        decls.append(_emit_f32_array(name, np.zeros(shape, dtype=np.float32), section_data=False))

    prefill_cache_init = []
    if mode == "prefill":
        for head_idx in range(n_heads):
            prefill_cache_init.append(
                f"""    for (int i = 0; i < prompt_len * d_head; ++i) {{
        k_cache_h{head_idx}_buf[i] = 0.0f;
        v_cache_h{head_idx}_buf[i] = 0.0f;
    }}"""
            )
    else:
        for head_idx in range(n_heads):
            prefill_cache_init.append(
                f"""    for (int i = 0; i < cache_len * d_head; ++i) {{
        k_cache_h{head_idx}_buf[i] = k_cache_init_h{head_idx}[i];
        v_cache_h{head_idx}_buf[i] = v_cache_init_h{head_idx}[i];
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

static float reciprocal_nr_f32(float x)
{{
    if (!(x > 0.0f) && !(x < 0.0f)) {{
        return 0.0f;
    }}
    return 1.0f / x;
}}

static float inv_sqrt_nr_f32(float x)
{{
    if (!(x > 0.0f)) {{
        return 0.0f;
    }}
    return 1.0f / sqrtf(x);
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

static void add_bias_f32(float *x, const float *bias, int rows, int cols)
{{
    for (int r = 0; r < rows; ++r) {{
        for (int c = 0; c < cols; ++c) {{
            x[r * cols + c] += bias[c];
        }}
    }}
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

static void split_qkv_heads(
    const float *qkv,
    float *q_out,
    float *k_out,
    float *v_out,
    int rows,
    int n_heads,
    int d_head)
{{
    const int attn_dim = n_heads * d_head;
    const int qkv_cols = 3 * attn_dim;
    for (int r = 0; r < rows; ++r) {{
        const float *src = qkv + r * qkv_cols;
        for (int i = 0; i < attn_dim; ++i) {{
            q_out[r * attn_dim + i] = src[i];
            k_out[r * attn_dim + i] = src[attn_dim + i];
            v_out[r * attn_dim + i] = src[2 * attn_dim + i];
        }}
    }}
}}

static void softmax_row_f32(float *scores, float *probs, int cols)
{{
    float maxv = -1.0e30f;
    for (int i = 0; i < cols; ++i) {{
        if (scores[i] > maxv) maxv = scores[i];
    }}
    float denom = 0.0f;
    for (int i = 0; i < cols; ++i) {{
        float z = scores[i] - maxv;
        if (z > 80.0f) z = 80.0f;
        if (z < -80.0f) z = -80.0f;
        float e = expf(z);
        probs[i] = e;
        denom += e;
    }}
    float inv_denom = reciprocal_nr_f32(denom);
    for (int i = 0; i < cols; ++i) {{
        probs[i] *= inv_denom;
    }}
}}

static void gpt2_gelu_tanh_f32(const float *x, float *out, int elems)
{{
    const float coeff = 0.79788456080f;
    for (int i = 0; i < elems; ++i) {{
        float xi = x[i];
        float inner = coeff * (xi + 0.044715f * xi * xi * xi);
        out[i] = 0.5f * xi * (1.0f + tanhf(inner));
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
    const int seq_len = {seq_len};
    const int prompt_len = {prompt_len};
    const int cache_len = {cache_len};
    const int d_model = {d_model};
    const int d_head = {d_head};
    const int n_heads = {n_heads};
    const int attn_dim = {attn_dim};
    const int ffn_dim = {ffn_dim};
    const float score_scale = {float(1.0 / np.sqrt(float(d_head))):.8e}f;
    uint32_t cycle_t0;
    uint32_t cycle_t1;

    printf("Native CPU GPT2 block: {program_name}\\n");

{chr(10).join(prefill_cache_init)}

    cycle_t0 = read_mcycle32();
    layernorm_f32(x_in, ln1_wb, x_norm1_buf, seq_len, d_model, 1.0e-5f);
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("cpu_native.layernorm1", cycle_t0, cycle_t1);

    cycle_t0 = read_mcycle32();
    matmul_f32(x_norm1_buf, attn_c_attn_w, qkv_buf, seq_len, d_model, 3 * attn_dim);
    add_bias_f32(qkv_buf, attn_c_attn_b, seq_len, 3 * attn_dim);
    split_qkv_heads(qkv_buf, q_buf, k_buf, v_buf, seq_len, n_heads, d_head);
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("cpu_native.c_attn", cycle_t0, cycle_t1);

    cycle_t0 = read_mcycle32();
    for (int head = 0; head < n_heads; ++head) {{
        float *k_cache = NULL;
        float *v_cache = NULL;
        if (head == 0) {{ k_cache = k_cache_h0_buf; v_cache = v_cache_h0_buf; }}
"""
    for head_idx in range(1, 16):
        source += f'        if (head == {head_idx}) {{ k_cache = k_cache_h{head_idx}_buf; v_cache = v_cache_h{head_idx}_buf; }}\n' if head_idx < n_heads else ""
    source += f"""        for (int d = 0; d < d_head; ++d) {{
"""
    if mode == "prefill":
        source += """            for (int r = 0; r < prompt_len; ++r) {
                k_cache[r * d_head + d] = k_buf[r * attn_dim + head * d_head + d];
                v_cache[r * d_head + d] = v_buf[r * attn_dim + head * d_head + d];
            }
"""
    else:
        source += """            k_cache[prompt_len * d_head + d] = k_buf[head * d_head + d];
            v_cache[prompt_len * d_head + d] = v_buf[head * d_head + d];
"""
    source += """        }
        for (int r = 0; r < seq_len; ++r) {
            float *scores_row = scores_buf + r * cache_len;
            float *probs_row = probs_buf + r * cache_len;
            const int visible = """
    source += "r + 1" if mode == "prefill" else "cache_len"
    source += """;
            for (int c = 0; c < visible; ++c) {
                float dot = 0.0f;
                for (int d = 0; d < d_head; ++d) {
                    dot += q_buf[r * attn_dim + head * d_head + d] * k_cache[c * d_head + d];
                }
                scores_row[c] = dot * score_scale;
            }
"""
    if mode == "prefill":
        source += """            for (int c = visible; c < cache_len; ++c) {
                scores_row[c] = -1.0e10f;
            }
"""
    source += """            softmax_row_f32(scores_row, probs_row, visible);
"""
    if mode == "prefill":
        source += """            for (int c = visible; c < cache_len; ++c) {
                probs_row[c] = 0.0f;
            }
"""
    source += """            for (int d = 0; d < d_head; ++d) {
                float acc = 0.0f;
                for (int c = 0; c < visible; ++c) {
                    acc += probs_row[c] * v_cache[c * d_head + d];
                }
                attn_cat_buf[r * attn_dim + head * d_head + d] = acc;
            }
        }
    }
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("cpu_native.attention", cycle_t0, cycle_t1);

    cycle_t0 = read_mcycle32();
    matmul_f32(attn_cat_buf, attn_c_proj_w, o_buf, seq_len, attn_dim, d_model);
    add_bias_f32(o_buf, attn_c_proj_b, seq_len, d_model);
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("cpu_native.c_proj", cycle_t0, cycle_t1);

    cycle_t0 = read_mcycle32();
    for (int i = 0; i < seq_len * d_model; ++i) {{
        resid1_buf[i] = x_in[i] + o_buf[i];
    }}
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("cpu_native.residual1", cycle_t0, cycle_t1);

    cycle_t0 = read_mcycle32();
    layernorm_f32(resid1_buf, ln2_wb, x_norm2_buf, seq_len, d_model, 1.0e-5f);
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("cpu_native.layernorm2", cycle_t0, cycle_t1);

    cycle_t0 = read_mcycle32();
    matmul_f32(x_norm2_buf, mlp_c_fc_w, ffn_fc_buf, seq_len, d_model, ffn_dim);
    add_bias_f32(ffn_fc_buf, mlp_c_fc_b, seq_len, ffn_dim);
    gpt2_gelu_tanh_f32(ffn_fc_buf, ffn_gelu_buf, seq_len * ffn_dim);
    matmul_f32(ffn_gelu_buf, mlp_c_proj_w, ffn_out_buf, seq_len, ffn_dim, d_model);
    add_bias_f32(ffn_out_buf, mlp_c_proj_b, seq_len, d_model);
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("cpu_native.mlp", cycle_t0, cycle_t1);

    cycle_t0 = read_mcycle32();
    for (int i = 0; i < seq_len * d_model; ++i) {{
        out_buf[i] = resid1_buf[i] + ffn_out_buf[i];
    }}
    cycle_t1 = read_mcycle32();
    print_cycle_delta32("cpu_native.residual2", cycle_t0, cycle_t1);

    {{
        float diff = max_abs_diff(out_buf, out_expected, seq_len * d_model);
        printf("cpu_native.max_abs_diff=%f\\n", (double)diff);
        if (diff > 1.0e-3f) {{
            printf("verification failed: cpu_native_gpt2_out\\n");
            return EXIT_FAILURE;
        }}
    }}

    printf("EXIT SUCCESS\\n");
    return EXIT_SUCCESS;
}}
"""
    return source.replace("{{", "{").replace("}}", "}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-head", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-mode", choices=("py", "host", "rtl"), default="py")
    parser.add_argument("--reuse-verilator", action="store_true")
    parser.add_argument("--maxcycles", type=int, default=3000000)
    parser.add_argument("--verilator-max-ticks", type=int, default=30000000000)
    args = parser.parse_args()

    tensors, expected = _build_case(
        mode=args.mode,
        d_model=args.d_model,
        d_head=args.d_head,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        prompt_len=args.prompt_len,
        seed=args.seed,
    )

    program_name = (
        f"cv32e40p_gpt2_{args.mode}_cpu_native_d{args.d_model}"
        f"_h{args.d_head}_nh{args.n_heads}_f{args.ffn_dim}_t{args.prompt_len}_s{args.seed}"
    )
    GENERATED_DIR.mkdir(exist_ok=True)
    program_path = GENERATED_DIR / f"{program_name}.c"
    program_path.write_text(
        _emit_source(
            program_name=program_name,
            mode=args.mode,
            d_model=args.d_model,
            d_head=args.d_head,
            n_heads=args.n_heads,
            ffn_dim=args.ffn_dim,
            prompt_len=args.prompt_len,
            host_native=args.run_mode == "host",
            tensors=tensors,
            expected=expected,
        )
    )

    if args.run_mode == "py":
        print(f"program={program_name}")
        print(f"mode={args.mode}")
        print(f"expected_checksum={float(np.asarray(expected, dtype=np.float32).sum()):.6f}")
        return 0

    if args.run_mode == "host":
        host_bin = GENERATED_DIR / f"{program_name}_host"
        subprocess.run(["gcc", "-O3", "-std=c11", str(program_path), "-lm", "-o", str(host_bin)], check=True, cwd=REPO_ROOT)
        proc = subprocess.run([str(host_bin)], check=False, cwd=REPO_ROOT, text=True, capture_output=True)
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
