from __future__ import annotations

import argparse
import os
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
from run_tinystories_prefill_decode_chain_rtl import _encode, _load_checkpoint  # noqa: E402
from tinynpu_jit.baremetal_emit_v2 import emit_cv32e40p_program_v2  # noqa: E402
from tinynpu_jit.blocks.llama_block import build_decode_artifact, build_prefill_artifact  # noqa: E402
from tinynpu_jit.ir import NpuSegment, TensorKind, VerifyTensor  # noqa: E402
from tinynpu_jit.rtl_runner import (  # noqa: E402
    CORE_DIR,
    CUSTOM_DIR,
    GENERATED_DIR,
    RUNTIME_DIR,
    RunnerConfig,
    TNPU_RISCV_MABI,
    TNPU_RISCV_MARCH,
    run_checked,
    run_vlt_npu,
    runtime_cflags,
    toolchain_include_lib_dirs,
    toolchain_prefix,
)


def _externalize_input(artifact, name: str) -> None:
    spec = artifact.plan.tensors[name]
    spec.kind = TensorKind.INPUT
    spec.data = None
    if name not in artifact.plan.inputs:
        artifact.plan.inputs.append(name)


def _externalize_decode_cache_inputs(artifact, *, n_kv_heads: int) -> None:
    for kv_head in range(n_kv_heads):
        for prefix in ("k", "v"):
            name = f"{prefix}_cache_h{kv_head}"
            _externalize_input(artifact, name)
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


def _strip_verification(artifact) -> None:
    artifact.plan.steps = [step for step in artifact.plan.steps if not isinstance(step, VerifyTensor)]
    artifact.expected_tensors = {}


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
    artifact, _, _, decode_ref = build_decode_artifact(
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
    return artifact, decode_ref


def _c_float_array(name: str, values: np.ndarray) -> str:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    body = ", ".join(f"{float(v):.9g}f" for v in flat)
    return f"static const float {name}[{flat.size}] = {{\n    {body}\n}};"


def _c_string_array(name: str, values: list[str]) -> str:
    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"')

    body = ", ".join(f'"{esc(value)}"' for value in values)
    return f"static const char *const {name}[{len(values)}] = {{\n    {body}\n}};"


def _render_runner(
    *,
    vocab_tokens: list[str],
    final_norm_w: np.ndarray,
    lm_head_w: np.ndarray,
    d_model: int,
    d_head: int,
    prompt_len: int,
    rms_eps: float,
) -> str:
    vocab_size = int(np.asarray(lm_head_w).shape[0])
    return f"""#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tinynpu_runtime_v2.h"

extern const TnpuProgram tinystories_single_l0_prefill;
extern const TnpuProgram tinystories_single_l0_decode;
extern const TnpuProgram tinystories_single_l1_prefill;
extern const TnpuProgram tinystories_single_l1_decode;

#define D_MODEL {d_model}
#define D_HEAD {d_head}
#define PROMPT_LEN {prompt_len}
#define DECODE_LEN ({prompt_len} + 1)
#define VOCAB_SIZE {vocab_size}
#define RMS_EPS {float(rms_eps):.9g}f

{_c_float_array("final_norm_w", final_norm_w)}

{_c_float_array("lm_head_w", lm_head_w)}

{_c_string_array("vocab_tokens", vocab_tokens)}

static int find_tensor_index(const TnpuProgram *program, const char *name)
{{
    for (uint32_t i = 0; i < program->tensor_count; ++i) {{
        if (strcmp(program->tensors[i].name, name) == 0) {{
            return (int)i;
        }}
    }}
    printf("missing tensor %s in %s\\n", name, program->name);
    return -1;
}}

static void fill_program_outputs(const TnpuProgram *program, TnpuTensor *outs, const TnpuTensor **op)
{{
    for (uint32_t i = 0; i < program->output_count; ++i) {{
        uint16_t t = program->output_tensor_indices[i];
        outs[i].data = program->tensors[t].data;
        outs[i].desc = &program->tensors[t];
        outs[i].elem_count = program->tensors[t].elem_count;
        op[i] = &outs[i];
    }}
}}

static int fill_program_inputs(const TnpuProgram *program, TnpuTensor *ins, const TnpuTensor **ip)
{{
    for (uint32_t i = 0; i < program->input_count; ++i) {{
        uint16_t t = program->input_tensor_indices[i];
        ins[i].data = program->tensors[t].data;
        ins[i].desc = &program->tensors[t];
        ins[i].elem_count = program->tensors[t].elem_count;
        ip[i] = &ins[i];
    }}
    return 0;
}}

static int override_input(
    const TnpuProgram *program,
    TnpuTensor *ins,
    const char *name,
    void *data,
    int elem_count)
{{
    for (uint32_t i = 0; i < program->input_count; ++i) {{
        uint16_t t = program->input_tensor_indices[i];
        if (strcmp(program->tensors[t].name, name) == 0) {{
            ins[i].data = data;
            ins[i].elem_count = elem_count;
            return 0;
        }}
    }}
    printf("input %s not found in %s\\n", name, program->name);
    return 1;
}}

static void *tensor_data(const TnpuProgram *program, const char *name)
{{
    int idx = find_tensor_index(program, name);
    if (idx < 0) {{
        return NULL;
    }}
    return program->tensors[idx].data;
}}

static int run_program(const TnpuProgram *program, TnpuTensor *ins, const TnpuTensor **ip)
{{
    TnpuTensor outs[8];
    const TnpuTensor *op[8];
    if (program->input_count > 8u || program->output_count > 8u) {{
        printf("too many inputs/outputs in %s\\n", program->name);
        return 1;
    }}
    fill_program_outputs(program, outs, op);
    puts(program->name);
    return tinynpu_run(program, ip, op, NULL, 0u);
}}

static int run_program_default_inputs(const TnpuProgram *program)
{{
    TnpuTensor ins[8];
    const TnpuTensor *ip[8];
    if (program->input_count > 8u) {{
        return 1;
    }}
    if (fill_program_inputs(program, ins, ip) != 0) {{
        return 1;
    }}
    return run_program(program, ins, ip);
}}

static void copy_k_cache(int32_t *dst, const int32_t *src)
{{
    for (int row = 0; row < D_HEAD; ++row) {{
        for (int col = 0; col < PROMPT_LEN; ++col) {{
            dst[row * DECODE_LEN + col] = src[row * PROMPT_LEN + col];
        }}
        dst[row * DECODE_LEN + PROMPT_LEN] = 0;
    }}
}}

static void copy_v_cache(int32_t *dst, const int32_t *src)
{{
    for (int row = 0; row < PROMPT_LEN; ++row) {{
        for (int col = 0; col < D_HEAD; ++col) {{
            dst[row * D_HEAD + col] = src[row * D_HEAD + col];
        }}
    }}
    for (int col = 0; col < D_HEAD; ++col) {{
        dst[PROMPT_LEN * D_HEAD + col] = 0;
    }}
}}

static float checksum_f32(const float *data, int count)
{{
    float total = 0.0f;
    for (int i = 0; i < count; ++i) {{
        total += data[i];
    }}
    return total;
}}

static int argmax_next_token(const float *hidden)
{{
    float ss = 0.0f;
    for (int i = 0; i < D_MODEL; ++i) {{
        ss += hidden[i] * hidden[i];
    }}
    float inv = 1.0f / sqrtf(ss / (float)D_MODEL + RMS_EPS);
    int best = 0;
    float best_logit = -3.402823466e+38f;
    for (int tok = 0; tok < VOCAB_SIZE; ++tok) {{
        float logit = 0.0f;
        for (int i = 0; i < D_MODEL; ++i) {{
            float x = hidden[i] * inv * final_norm_w[i];
            logit += x * lm_head_w[tok * D_MODEL + i];
        }}
        if (tok == 0 || logit > best_logit) {{
            best = tok;
            best_logit = logit;
        }}
    }}
    return best;
}}

static int32_t l0_k_cache_h0[D_HEAD * DECODE_LEN] __attribute__((section(".data")));
static int32_t l0_v_cache_h0[DECODE_LEN * D_HEAD] __attribute__((section(".data")));
static int32_t l1_k_cache_h0[D_HEAD * DECODE_LEN] __attribute__((section(".data")));
static int32_t l1_v_cache_h0[DECODE_LEN * D_HEAD] __attribute__((section(".data")));

int main(void)
{{
    TnpuTensor ins[8];
    const TnpuTensor *ip[8];
    int32_t *k_prefill;
    int32_t *v_prefill;
    float *l0_prefill_out;
    float *l0_decode_out;
    float *l1_prefill_out;
    float *l1_decode_out;
    int next_id;

    puts("tinystories_single_binary_prefill_decode");

    if (run_program_default_inputs(&tinystories_single_l0_prefill) != 0) return EXIT_FAILURE;
    k_prefill = (int32_t *)tensor_data(&tinystories_single_l0_prefill, "prefill_k_cache_h0");
    v_prefill = (int32_t *)tensor_data(&tinystories_single_l0_prefill, "prefill_v_cache_h0");
    if (k_prefill == NULL || v_prefill == NULL) return EXIT_FAILURE;
    copy_k_cache(l0_k_cache_h0, k_prefill);
    copy_v_cache(l0_v_cache_h0, v_prefill);

    fill_program_inputs(&tinystories_single_l0_decode, ins, ip);
    if (override_input(&tinystories_single_l0_decode, ins, "k_cache_h0", l0_k_cache_h0, D_HEAD * DECODE_LEN) != 0) return EXIT_FAILURE;
    if (override_input(&tinystories_single_l0_decode, ins, "v_cache_h0", l0_v_cache_h0, DECODE_LEN * D_HEAD) != 0) return EXIT_FAILURE;
    if (run_program(&tinystories_single_l0_decode, ins, ip) != 0) return EXIT_FAILURE;

    l0_prefill_out = (float *)tensor_data(&tinystories_single_l0_prefill, "out");
    l0_decode_out = (float *)tensor_data(&tinystories_single_l0_decode, "out");
    if (l0_prefill_out == NULL || l0_decode_out == NULL) return EXIT_FAILURE;

    fill_program_inputs(&tinystories_single_l1_prefill, ins, ip);
    if (override_input(&tinystories_single_l1_prefill, ins, "x_in", l0_prefill_out, PROMPT_LEN * D_MODEL) != 0) return EXIT_FAILURE;
    if (run_program(&tinystories_single_l1_prefill, ins, ip) != 0) return EXIT_FAILURE;
    k_prefill = (int32_t *)tensor_data(&tinystories_single_l1_prefill, "prefill_k_cache_h0");
    v_prefill = (int32_t *)tensor_data(&tinystories_single_l1_prefill, "prefill_v_cache_h0");
    if (k_prefill == NULL || v_prefill == NULL) return EXIT_FAILURE;
    copy_k_cache(l1_k_cache_h0, k_prefill);
    copy_v_cache(l1_v_cache_h0, v_prefill);

    fill_program_inputs(&tinystories_single_l1_decode, ins, ip);
    if (override_input(&tinystories_single_l1_decode, ins, "x_in", l0_decode_out, D_MODEL) != 0) return EXIT_FAILURE;
    if (override_input(&tinystories_single_l1_decode, ins, "k_cache_h0", l1_k_cache_h0, D_HEAD * DECODE_LEN) != 0) return EXIT_FAILURE;
    if (override_input(&tinystories_single_l1_decode, ins, "v_cache_h0", l1_v_cache_h0, DECODE_LEN * D_HEAD) != 0) return EXIT_FAILURE;
    if (run_program(&tinystories_single_l1_decode, ins, ip) != 0) return EXIT_FAILURE;

    l1_prefill_out = (float *)tensor_data(&tinystories_single_l1_prefill, "out");
    l1_decode_out = (float *)tensor_data(&tinystories_single_l1_decode, "out");
    if (l1_prefill_out == NULL || l1_decode_out == NULL) return EXIT_FAILURE;

    printf("single.layer0.prefill.checksum %.6f\\n", (double)checksum_f32(l0_prefill_out, PROMPT_LEN * D_MODEL));
    printf("single.layer0.decode.checksum %.6f\\n", (double)checksum_f32(l0_decode_out, D_MODEL));
    printf("single.layer1.prefill.checksum %.6f\\n", (double)checksum_f32(l1_prefill_out, PROMPT_LEN * D_MODEL));
    printf("single.layer1.decode.checksum %.6f\\n", (double)checksum_f32(l1_decode_out, D_MODEL));
    next_id = argmax_next_token(l1_decode_out);
    printf("single.next_id %d\\n", next_id);
    printf("single.next_token '%s'\\n", vocab_tokens[next_id]);
    puts("single_binary_done");
    return EXIT_SUCCESS;
}}
"""


def _render_multitoken_runner(
    *,
    vocab_tokens: list[str],
    final_norm_w: np.ndarray,
    lm_head_w: np.ndarray,
    d_model: int,
    d_head: int,
    prompt_len: int,
    rms_eps: float,
    initial_decode_id: int,
    generate_tokens: int,
) -> str:
    vocab_size = int(np.asarray(lm_head_w).shape[0])
    externs = "\n".join(
        [
            "extern const TnpuProgram tinystories_single_l0_prefill;",
            "extern const TnpuProgram tinystories_single_l1_prefill;",
            *[
                f"extern const TnpuProgram tinystories_single_l0_decode_s{step};"
                for step in range(generate_tokens)
            ],
            *[
                f"extern const TnpuProgram tinystories_single_l1_decode_s{step};"
                for step in range(generate_tokens)
            ],
        ]
    )
    l0_decode_ptrs = ", ".join(f"&tinystories_single_l0_decode_s{step}" for step in range(generate_tokens))
    l1_decode_ptrs = ", ".join(f"&tinystories_single_l1_decode_s{step}" for step in range(generate_tokens))
    cache_lens = ", ".join(str(prompt_len + step + 1) for step in range(generate_tokens))
    cache_decls = "\n".join(
        [
            *[
                f'static int32_t l0_k_cache_s{step}[D_HEAD * {prompt_len + step + 1}] __attribute__((section(".data")));'
                for step in range(generate_tokens)
            ],
            *[
                f'static int32_t l0_v_cache_s{step}[{prompt_len + step + 1} * D_HEAD] __attribute__((section(".data")));'
                for step in range(generate_tokens)
            ],
            *[
                f'static int32_t l1_k_cache_s{step}[D_HEAD * {prompt_len + step + 1}] __attribute__((section(".data")));'
                for step in range(generate_tokens)
            ],
            *[
                f'static int32_t l1_v_cache_s{step}[{prompt_len + step + 1} * D_HEAD] __attribute__((section(".data")));'
                for step in range(generate_tokens)
            ],
        ]
    )
    l0_k_ptrs = ", ".join(f"l0_k_cache_s{step}" for step in range(generate_tokens))
    l0_v_ptrs = ", ".join(f"l0_v_cache_s{step}" for step in range(generate_tokens))
    l1_k_ptrs = ", ".join(f"l1_k_cache_s{step}" for step in range(generate_tokens))
    l1_v_ptrs = ", ".join(f"l1_v_cache_s{step}" for step in range(generate_tokens))
    return f"""#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tinynpu_runtime_v2.h"

{externs}

#define D_MODEL {d_model}
#define D_HEAD {d_head}
#define PROMPT_LEN {prompt_len}
#define GENERATE_TOKENS {generate_tokens}
#define INITIAL_DECODE_ID {initial_decode_id}
#define VOCAB_SIZE {vocab_size}
#define RMS_EPS {float(rms_eps):.9g}f

{_c_float_array("final_norm_w", final_norm_w)}

{_c_float_array("lm_head_w", lm_head_w)}

{_c_string_array("vocab_tokens", vocab_tokens)}

{cache_decls}

static const TnpuProgram *const l0_decode_programs[GENERATE_TOKENS] = {{ {l0_decode_ptrs} }};
static const TnpuProgram *const l1_decode_programs[GENERATE_TOKENS] = {{ {l1_decode_ptrs} }};
static const int cache_lens[GENERATE_TOKENS] = {{ {cache_lens} }};
static int32_t *const l0_k_caches[GENERATE_TOKENS] = {{ {l0_k_ptrs} }};
static int32_t *const l0_v_caches[GENERATE_TOKENS] = {{ {l0_v_ptrs} }};
static int32_t *const l1_k_caches[GENERATE_TOKENS] = {{ {l1_k_ptrs} }};
static int32_t *const l1_v_caches[GENERATE_TOKENS] = {{ {l1_v_ptrs} }};

static int find_tensor_index(const TnpuProgram *program, const char *name)
{{
    for (uint32_t i = 0; i < program->tensor_count; ++i) {{
        if (strcmp(program->tensors[i].name, name) == 0) {{
            return (int)i;
        }}
    }}
    printf("missing tensor %s in %s\\n", name, program->name);
    return -1;
}}

static void fill_program_outputs(const TnpuProgram *program, TnpuTensor *outs, const TnpuTensor **op)
{{
    for (uint32_t i = 0; i < program->output_count; ++i) {{
        uint16_t t = program->output_tensor_indices[i];
        outs[i].data = program->tensors[t].data;
        outs[i].desc = &program->tensors[t];
        outs[i].elem_count = program->tensors[t].elem_count;
        op[i] = &outs[i];
    }}
}}

static int fill_program_inputs(const TnpuProgram *program, TnpuTensor *ins, const TnpuTensor **ip)
{{
    for (uint32_t i = 0; i < program->input_count; ++i) {{
        uint16_t t = program->input_tensor_indices[i];
        ins[i].data = program->tensors[t].data;
        ins[i].desc = &program->tensors[t];
        ins[i].elem_count = program->tensors[t].elem_count;
        ip[i] = &ins[i];
    }}
    return 0;
}}

static int override_input(
    const TnpuProgram *program,
    TnpuTensor *ins,
    const char *name,
    void *data,
    int elem_count)
{{
    for (uint32_t i = 0; i < program->input_count; ++i) {{
        uint16_t t = program->input_tensor_indices[i];
        if (strcmp(program->tensors[t].name, name) == 0) {{
            ins[i].data = data;
            ins[i].elem_count = elem_count;
            return 0;
        }}
    }}
    printf("input %s not found in %s\\n", name, program->name);
    return 1;
}}

static void *tensor_data(const TnpuProgram *program, const char *name)
{{
    int idx = find_tensor_index(program, name);
    if (idx < 0) {{
        return NULL;
    }}
    return program->tensors[idx].data;
}}

static int run_program(const TnpuProgram *program, TnpuTensor *ins, const TnpuTensor **ip)
{{
    TnpuTensor outs[8];
    const TnpuTensor *op[8];
    if (program->input_count > 8u || program->output_count > 8u) {{
        printf("too many inputs/outputs in %s\\n", program->name);
        return 1;
    }}
    fill_program_outputs(program, outs, op);
    puts(program->name);
    return tinynpu_run(program, ip, op, NULL, 0u);
}}

static int run_program_default_inputs(const TnpuProgram *program)
{{
    TnpuTensor ins[8];
    const TnpuTensor *ip[8];
    if (program->input_count > 8u) {{
        return 1;
    }}
    if (fill_program_inputs(program, ins, ip) != 0) {{
        return 1;
    }}
    return run_program(program, ins, ip);
}}

static void copy_k_cache(int32_t *dst, int dst_len, const int32_t *src, int src_len)
{{
    for (int row = 0; row < D_HEAD; ++row) {{
        for (int col = 0; col < dst_len; ++col) {{
            dst[row * dst_len + col] = (col < src_len) ? src[row * src_len + col] : 0;
        }}
    }}
}}

static void copy_v_cache(int32_t *dst, int dst_len, const int32_t *src, int src_len)
{{
    for (int row = 0; row < dst_len; ++row) {{
        for (int col = 0; col < D_HEAD; ++col) {{
            dst[row * D_HEAD + col] = (row < src_len) ? src[row * D_HEAD + col] : 0;
        }}
    }}
}}

static void copy_embedding(int token_id, float *dst)
{{
    const float *src = &lm_head_w[token_id * D_MODEL];
    for (int i = 0; i < D_MODEL; ++i) {{
        dst[i] = src[i];
    }}
}}

static float checksum_f32(const float *data, int count)
{{
    float total = 0.0f;
    for (int i = 0; i < count; ++i) {{
        total += data[i];
    }}
    return total;
}}

static int argmax_next_token(const float *hidden)
{{
    float ss = 0.0f;
    for (int i = 0; i < D_MODEL; ++i) {{
        ss += hidden[i] * hidden[i];
    }}
    float inv = 1.0f / sqrtf(ss / (float)D_MODEL + RMS_EPS);
    int best = 0;
    float best_logit = -3.402823466e+38f;
    for (int tok = 0; tok < VOCAB_SIZE; ++tok) {{
        float logit = 0.0f;
        for (int i = 0; i < D_MODEL; ++i) {{
            float x = hidden[i] * inv * final_norm_w[i];
            logit += x * lm_head_w[tok * D_MODEL + i];
        }}
        if (tok == 0 || logit > best_logit) {{
            best = tok;
            best_logit = logit;
        }}
    }}
    return best;
}}

int main(void)
{{
    TnpuTensor ins[8];
    const TnpuTensor *ip[8];
    int32_t *k_prefill;
    int32_t *v_prefill;
    float *l0_prefill_out;
    float *l0_decode_out;
    float *l1_prefill_out;
    float *l1_decode_out;
    float current_embed[D_MODEL];
    int current_token_id = INITIAL_DECODE_ID;
    int generated[GENERATE_TOKENS];

    puts("tinystories_single_binary_prefill_decode_multitoken");

    if (run_program_default_inputs(&tinystories_single_l0_prefill) != 0) return EXIT_FAILURE;
    k_prefill = (int32_t *)tensor_data(&tinystories_single_l0_prefill, "prefill_k_cache_h0");
    v_prefill = (int32_t *)tensor_data(&tinystories_single_l0_prefill, "prefill_v_cache_h0");
    if (k_prefill == NULL || v_prefill == NULL) return EXIT_FAILURE;
    copy_k_cache(l0_k_caches[0], cache_lens[0], k_prefill, PROMPT_LEN);
    copy_v_cache(l0_v_caches[0], cache_lens[0], v_prefill, PROMPT_LEN);

    l0_prefill_out = (float *)tensor_data(&tinystories_single_l0_prefill, "out");
    if (l0_prefill_out == NULL) return EXIT_FAILURE;
    fill_program_inputs(&tinystories_single_l1_prefill, ins, ip);
    if (override_input(&tinystories_single_l1_prefill, ins, "x_in", l0_prefill_out, PROMPT_LEN * D_MODEL) != 0) return EXIT_FAILURE;
    if (run_program(&tinystories_single_l1_prefill, ins, ip) != 0) return EXIT_FAILURE;
    k_prefill = (int32_t *)tensor_data(&tinystories_single_l1_prefill, "prefill_k_cache_h0");
    v_prefill = (int32_t *)tensor_data(&tinystories_single_l1_prefill, "prefill_v_cache_h0");
    if (k_prefill == NULL || v_prefill == NULL) return EXIT_FAILURE;
    copy_k_cache(l1_k_caches[0], cache_lens[0], k_prefill, PROMPT_LEN);
    copy_v_cache(l1_v_caches[0], cache_lens[0], v_prefill, PROMPT_LEN);

    l1_prefill_out = (float *)tensor_data(&tinystories_single_l1_prefill, "out");
    if (l1_prefill_out == NULL) return EXIT_FAILURE;
    printf("single.layer0.prefill.checksum %.6f\\n", (double)checksum_f32(l0_prefill_out, PROMPT_LEN * D_MODEL));
    printf("single.layer1.prefill.checksum %.6f\\n", (double)checksum_f32(l1_prefill_out, PROMPT_LEN * D_MODEL));

    for (int step = 0; step < GENERATE_TOKENS; ++step) {{
        const TnpuProgram *l0_decode = l0_decode_programs[step];
        const TnpuProgram *l1_decode = l1_decode_programs[step];
        int cache_len = cache_lens[step];
        copy_embedding(current_token_id, current_embed);
        printf("single.decode_input.step%d id=%d token='%s'\\n", step, current_token_id, vocab_tokens[current_token_id]);

        fill_program_inputs(l0_decode, ins, ip);
        if (override_input(l0_decode, ins, "x_in", current_embed, D_MODEL) != 0) return EXIT_FAILURE;
        if (override_input(l0_decode, ins, "k_cache_h0", l0_k_caches[step], D_HEAD * cache_len) != 0) return EXIT_FAILURE;
        if (override_input(l0_decode, ins, "v_cache_h0", l0_v_caches[step], cache_len * D_HEAD) != 0) return EXIT_FAILURE;
        if (run_program(l0_decode, ins, ip) != 0) return EXIT_FAILURE;
        l0_decode_out = (float *)tensor_data(l0_decode, "out");
        if (l0_decode_out == NULL) return EXIT_FAILURE;

        fill_program_inputs(l1_decode, ins, ip);
        if (override_input(l1_decode, ins, "x_in", l0_decode_out, D_MODEL) != 0) return EXIT_FAILURE;
        if (override_input(l1_decode, ins, "k_cache_h0", l1_k_caches[step], D_HEAD * cache_len) != 0) return EXIT_FAILURE;
        if (override_input(l1_decode, ins, "v_cache_h0", l1_v_caches[step], cache_len * D_HEAD) != 0) return EXIT_FAILURE;
        if (run_program(l1_decode, ins, ip) != 0) return EXIT_FAILURE;
        l1_decode_out = (float *)tensor_data(l1_decode, "out");
        if (l1_decode_out == NULL) return EXIT_FAILURE;

        generated[step] = argmax_next_token(l1_decode_out);
        printf("single.step%d.layer0.decode.checksum %.6f\\n", step, (double)checksum_f32(l0_decode_out, D_MODEL));
        printf("single.step%d.layer1.decode.checksum %.6f\\n", step, (double)checksum_f32(l1_decode_out, D_MODEL));
        printf("single.generated.step%d id=%d token='%s'\\n", step, generated[step], vocab_tokens[generated[step]]);
        current_token_id = generated[step];

        if (step + 1 < GENERATE_TOKENS) {{
            copy_k_cache(l0_k_caches[step + 1], cache_lens[step + 1], l0_k_caches[step], cache_lens[step]);
            copy_v_cache(l0_v_caches[step + 1], cache_lens[step + 1], l0_v_caches[step], cache_lens[step]);
            copy_k_cache(l1_k_caches[step + 1], cache_lens[step + 1], l1_k_caches[step], cache_lens[step]);
            copy_v_cache(l1_v_caches[step + 1], cache_lens[step + 1], l1_v_caches[step], cache_lens[step]);
        }}
    }}

    printf("single.generated_tokens");
    for (int step = 0; step < GENERATE_TOKENS; ++step) {{
        printf(" %s", vocab_tokens[generated[step]]);
    }}
    printf("\\n");
    puts("single_binary_done");
    return EXIT_SUCCESS;
}}
"""


def _build_multi_v2_elf_and_hex(
    program_name: str,
    program_sources: dict[str, str],
    runner_source: str,
    *,
    runner_config: RunnerConfig | None = None,
) -> tuple[list[Path], Path, Path, Path]:
    cfg = runner_config or RunnerConfig(dump_final_outputs=False, verbose_steps=False)
    GENERATED_DIR.mkdir(exist_ok=True)
    CUSTOM_DIR.mkdir(exist_ok=True)
    program_paths: list[Path] = []
    for stem, source in program_sources.items():
        path = GENERATED_DIR / f"{program_name}_{stem}.c"
        path.write_text(source)
        program_paths.append(path)
    runner_path = GENERATED_DIR / f"{program_name}_runner.c"
    runner_path.write_text(runner_source)

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
            *runtime_cflags(cfg),
            "-T",
            "custom/link.ld",
            "-static",
            "custom/crt0.S",
            str(runner_path),
            *[str(path) for path in program_paths],
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
    run_checked([objcopy, "-O", "verilog", str(elf_path), str(hex_path)], cwd=CORE_DIR, env=build_env)
    return program_paths, runner_path, elf_path, hex_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/tinystories_word_lm_d32_t17_qat_int16_long"))
    parser.add_argument("--prompt", default="there was a little girl named lily .")
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--generate-tokens", type=int, default=1)
    parser.add_argument("--program-name", default="tinystories_single_prefill_decode")
    parser.add_argument("--run-rtl", action="store_true")
    parser.add_argument("--maxcycles", type=int, default=20_000_000)
    parser.add_argument("--verilator-max-ticks", type=int, default=40_000_000_000)
    parser.add_argument("--timeout-s", type=int, default=900)
    args = parser.parse_args()
    if args.generate_tokens < 1:
        raise ValueError("--generate-tokens must be at least 1")

    model, tokenizer = _load_checkpoint(args.run_dir)
    token_ids = _encode(tokenizer, args.prompt)
    if len(token_ids) < args.prompt_len + 1:
        raise ValueError(
            f"prompt must encode to at least prompt_len + 1 tokens; got {len(token_ids)} for prompt_len={args.prompt_len}"
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
    if cfg.n_kv_heads != 1:
        raise NotImplementedError("single-binary runner currently emits one KV head cache copy path")
    if block1.config.d_model != cfg.d_model or block1.config.d_head != cfg.d_head:
        raise ValueError("layer configs must match for single-binary runner")

    state0 = {
        "config": block0.config,
        "block": block0,
        "x_prompt_in": embeddings[np.asarray(cache_ids, dtype=np.int64)],
        "x_decode_in": embeddings[np.asarray([decode_id], dtype=np.int64)],
    }
    prefill0_artifact, prefill0_ref = _build_prefill(block0, state0, prompt_len=args.prompt_len)

    state1 = {
        "config": block1.config,
        "block": block1,
        "x_prompt_in": np.asarray(prefill0_ref["out"], dtype=np.float32),
        "x_decode_in": np.zeros((1, cfg.d_model), dtype=np.float32),
    }
    prefill1_artifact, prefill1_ref = _build_prefill(block1, state1, prompt_len=args.prompt_len)
    _externalize_input(prefill1_artifact, "x_in")

    decode_artifacts: list[tuple[object, object]] = []
    for step in range(args.generate_tokens):
        ctx_len = args.prompt_len + step
        zero_prompt = np.zeros((ctx_len, cfg.d_model), dtype=np.float32)
        zero_decode = np.zeros((1, cfg.d_model), dtype=np.float32)
        state0_step = {
            "config": block0.config,
            "block": block0,
            "x_prompt_in": zero_prompt,
            "x_decode_in": embeddings[np.asarray([decode_id], dtype=np.int64)],
        }
        decode0_artifact, decode0_ref = _build_decode(block0, state0_step, prompt_len=ctx_len)
        _externalize_input(decode0_artifact, "x_in")
        state1_step = {
            "config": block1.config,
            "block": block1,
            "x_prompt_in": zero_prompt,
            "x_decode_in": zero_decode,
        }
        decode1_artifact, decode1_ref = _build_decode(block1, state1_step, prompt_len=ctx_len)
        _externalize_input(decode1_artifact, "x_in")
        decode_artifacts.append((decode0_artifact, decode1_artifact))

    for artifact in (prefill0_artifact, prefill1_artifact):
        _strip_verification(artifact)
    for decode0_artifact, decode1_artifact in decode_artifacts:
        _strip_verification(decode0_artifact)
        _strip_verification(decode1_artifact)

    zero_prefill_x = np.zeros((args.prompt_len, cfg.d_model), dtype=np.float32)

    program_sources = {
        "l0_prefill_program": emit_cv32e40p_program_v2(
            prefill0_artifact,
            {},
            program_name="tinystories_single_l0_prefill",
        ),
        "l1_prefill_program": emit_cv32e40p_program_v2(
            prefill1_artifact,
            {"x_in": zero_prefill_x},
            program_name="tinystories_single_l1_prefill",
        ),
    }
    for step, (decode0_artifact, decode1_artifact) in enumerate(decode_artifacts):
        ctx_len = args.prompt_len + step
        zero_decode_x = np.zeros((1, cfg.d_model), dtype=np.float32)
        zero_k_cache = np.zeros((cfg.d_head, ctx_len + 1), dtype=np.int16)
        zero_v_cache = np.zeros((ctx_len + 1, cfg.d_head), dtype=np.int16)
        program_sources[f"l0_decode_s{step}_program"] = emit_cv32e40p_program_v2(
            decode0_artifact,
            {"x_in": zero_decode_x, "k_cache_h0": zero_k_cache, "v_cache_h0": zero_v_cache},
            program_name=f"tinystories_single_l0_decode_s{step}",
        )
        program_sources[f"l1_decode_s{step}_program"] = emit_cv32e40p_program_v2(
            decode1_artifact,
            {"x_in": zero_decode_x, "k_cache_h0": zero_k_cache, "v_cache_h0": zero_v_cache},
            program_name=f"tinystories_single_l1_decode_s{step}",
        )

    runner_source = _render_multitoken_runner(
        vocab_tokens=list(tokenizer.itos),
        final_norm_w=final_norm_w,
        lm_head_w=lm_head_w,
        d_model=cfg.d_model,
        d_head=cfg.d_head,
        prompt_len=args.prompt_len,
        rms_eps=model.cfg.rms_norm_eps,
        initial_decode_id=decode_id,
        generate_tokens=args.generate_tokens,
    )
    program_paths, runner_path, elf_path, hex_path = _build_multi_v2_elf_and_hex(
        args.program_name,
        program_sources,
        runner_source,
        runner_config=RunnerConfig(dump_final_outputs=False, verbose_steps=False),
    )

    print(f"program={args.program_name}")
    print(f"prompt={args.prompt!r}")
    print(f"cache_ids={cache_ids}")
    print(f"decode_id={decode_id} decode_token={tokenizer.itos[decode_id]!r}")
    print(f"generate_tokens={args.generate_tokens}")
    print(f"expected.layer0.prefill.checksum={float(np.asarray(prefill0_ref['out'], dtype=np.float32).sum()):.6f}")
    print(f"expected.layer1.prefill.checksum={float(np.asarray(prefill1_ref['out'], dtype=np.float32).sum()):.6f}")
    print(f"runner={runner_path}")
    for path in program_paths:
        print(f"program_source={path}")
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
