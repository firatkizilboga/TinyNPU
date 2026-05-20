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


ONNX2C_DIR = REPO_ROOT / "external" / "third_party" / "onnx2c"
ONNX2C_BIN = ONNX2C_DIR / "build" / "onnx2c"
ONNX_PYTHON = REPO_ROOT / ".venv-onnx" / "bin" / "python"
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


def _deterministic_weight(shape: tuple[int, ...], *, mod: int, scale: float, offset: int = 0) -> np.ndarray:
    size = int(np.prod(shape))
    idx = np.arange(size, dtype=np.int32) + int(offset)
    vals = ((idx * 31 + 7) % int(mod)).astype(np.float32) - float(mod // 2)
    return (vals * float(scale)).reshape(shape).astype(np.float32)


def _deterministic_input() -> np.ndarray:
    values = np.zeros((1, 1, 8, 8), dtype=np.float32)
    for h in range(8):
        for w in range(8):
            values[0, 0, h, w] = ((h * 11 + w * 7 + 3) % 17) / 16.0
    return values


def _deterministic_mlp_input(hidden_dim: int = 64) -> np.ndarray:
    values = np.zeros((1, hidden_dim), dtype=np.float32)
    for i in range(hidden_dim):
        values[0, i] = ((i * 5 + 3) % 19) / 18.0
    return values


def _write_mlp_onnx(model_path: Path, hidden_dim: int = 64) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    h = int(hidden_dim)
    input_x = helper.make_tensor_value_info("input_x", TensorProto.FLOAT, [1, h])
    output_y = helper.make_tensor_value_info("output_y", TensorProto.FLOAT, [1, 1])

    def w(shape: tuple[int, ...], mod: int, scale: float, offset: int) -> np.ndarray:
        return _deterministic_weight(shape, mod=mod, scale=scale, offset=offset).astype(np.float32)

    initializers = [
        numpy_helper.from_array(w((h, h), 23, 0.0075, 0).T, "fc1_w_t"),
        numpy_helper.from_array(w((1, h), 19, 0.0020, 0), "fc1_b"),
        numpy_helper.from_array(w((h, h), 29, 0.0065, 101).T, "fc2_w_t"),
        numpy_helper.from_array(w((1, h), 19, 0.0020, 101), "fc2_b"),
        numpy_helper.from_array(w((h, h), 31, 0.0060, 211).T, "fc3_w_t"),
        numpy_helper.from_array(w((1, h), 19, 0.0020, 211), "fc3_b"),
        numpy_helper.from_array(w((1, h), 17, 0.0100, 307).T, "fc4_w_t"),
        numpy_helper.from_array(np.array([[-0.05]], dtype=np.float32), "fc4_b"),
    ]
    nodes = [
        helper.make_node("MatMul", ["input_x", "fc1_w_t"], ["fc1_mm"]),
        helper.make_node("Add", ["fc1_mm", "fc1_b"], ["fc1"]),
        helper.make_node("Relu", ["fc1"], ["relu1"]),
        helper.make_node("MatMul", ["relu1", "fc2_w_t"], ["fc2_mm"]),
        helper.make_node("Add", ["fc2_mm", "fc2_b"], ["fc2"]),
        helper.make_node("Relu", ["fc2"], ["relu2"]),
        helper.make_node("MatMul", ["relu2", "fc3_w_t"], ["fc3_mm"]),
        helper.make_node("Add", ["fc3_mm", "fc3_b"], ["fc3"]),
        helper.make_node("Relu", ["fc3"], ["relu3"]),
        helper.make_node("MatMul", ["relu3", "fc4_w_t"], ["fc4_mm"]),
        helper.make_node("Add", ["fc4_mm", "fc4_b"], ["fc4"]),
        helper.make_node("Sigmoid", ["fc4"], ["output_y"]),
    ]
    graph = helper.make_graph(nodes, "third_party_onnx2c_mlp", [input_x], [output_y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, model_path)


def _write_conv_onnx(model_path: Path, conv_channels: int = 16) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    ch = int(conv_channels)
    input_x = helper.make_tensor_value_info("input_x", TensorProto.FLOAT, [1, 1, 8, 8])
    output_y = helper.make_tensor_value_info("output_y", TensorProto.FLOAT, [1, 1, 1, 1])
    initializers = [
        numpy_helper.from_array(_deterministic_weight((ch, 1, 3, 3), mod=23, scale=0.020, offset=0), "conv1_w"),
        numpy_helper.from_array(_deterministic_weight((ch,), mod=19, scale=0.002, offset=0), "conv1_b"),
        numpy_helper.from_array(_deterministic_weight((ch, ch, 3, 3), mod=29, scale=0.006, offset=101), "conv2_w"),
        numpy_helper.from_array(_deterministic_weight((ch,), mod=19, scale=0.002, offset=101), "conv2_b"),
        numpy_helper.from_array(_deterministic_weight((ch, ch, 3, 3), mod=31, scale=0.006, offset=211), "conv3_w"),
        numpy_helper.from_array(_deterministic_weight((ch,), mod=19, scale=0.002, offset=211), "conv3_b"),
        numpy_helper.from_array(_deterministic_weight((1, ch, 2, 2), mod=17, scale=0.012, offset=307), "conv4_w"),
        numpy_helper.from_array(np.array([-0.05], dtype=np.float32), "conv4_b"),
    ]
    nodes = [
        helper.make_node("Conv", ["input_x", "conv1_w", "conv1_b"], ["conv1"], kernel_shape=[3, 3]),
        helper.make_node("Relu", ["conv1"], ["relu1"]),
        helper.make_node("Conv", ["relu1", "conv2_w", "conv2_b"], ["conv2"], kernel_shape=[3, 3]),
        helper.make_node("Relu", ["conv2"], ["relu2"]),
        helper.make_node("Conv", ["relu2", "conv3_w", "conv3_b"], ["conv3"], kernel_shape=[3, 3]),
        helper.make_node("Relu", ["conv3"], ["relu3"]),
        helper.make_node("Conv", ["relu3", "conv4_w", "conv4_b"], ["conv4"], kernel_shape=[2, 2]),
        helper.make_node("Sigmoid", ["conv4"], ["output_y"]),
    ]
    graph = helper.make_graph(nodes, "third_party_onnx2c_conv4", [input_x], [output_y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, model_path)


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+|[.,!?;:'\"()/-]", text.lower())


def _encode_prompt(run_dir: Path, prompt: str, prompt_len: int) -> tuple[list[int], list[str]]:
    meta = json.loads((run_dir / "tinystories_char_lm_config.json").read_text())
    tokenizer = meta["tokenizer"]
    if tokenizer.get("kind") != "word":
        raise ValueError("expected a word tokenizer checkpoint")
    itos = [str(x) for x in tokenizer["itos"]]
    stoi = {tok: idx for idx, tok in enumerate(itos)}
    ids = [stoi["<bos>"]]
    ids.extend(stoi.get(tok, stoi["<unk>"]) for tok in _word_tokens(prompt))
    if len(ids) < prompt_len:
        raise ValueError(f"prompt encodes to {len(ids)} tokens, need at least {prompt_len}")
    return ids[:prompt_len], itos


def _rope_tables(seq_len: int, d_head: int, theta: float) -> tuple[np.ndarray, np.ndarray]:
    half = d_head // 2
    pos = np.arange(seq_len, dtype=np.float32)
    dim = np.arange(half, dtype=np.float32)
    inv_freq = np.power(np.float32(theta), -dim / np.float32(half)).astype(np.float32)
    angles = pos[:, None] * inv_freq[None, :]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


def _causal_mask(seq_len: int) -> np.ndarray:
    mask = np.zeros((seq_len, seq_len), dtype=np.float32)
    for i in range(seq_len):
        for j in range(i + 1, seq_len):
            mask[i, j] = -1.0e9
    return mask


def _write_tiny_lm_onnx(model_path: Path, prompt_len: int = 9) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    run_dir = TINYSTORIES_RUN_DIR
    meta = json.loads((run_dir / "tinystories_char_lm_config.json").read_text())
    cfg = meta["config"]
    if (cfg["d_model"], cfg["d_head"], cfg["n_heads"], cfg["n_kv_heads"], cfg["ffn_hidden_dim"], cfg["n_layers"]) != (
        32,
        16,
        2,
        1,
        32,
        2,
    ):
        raise NotImplementedError("TinyLM ONNX baseline currently expects d32 h16 nh2 nkv1 f32 nl2")

    s = int(prompt_len)
    d = 32
    dh = 16
    nh = 2
    half = dh // 2
    vocab = int(cfg["vocab_size"])
    fp = np.load(run_dir / "tinystories_char_lm_fp32.npz")
    cos, sin = _rope_tables(s, dh, float(cfg["rope_theta"]))

    initializers = [
        numpy_helper.from_array(np.array([float(cfg["rms_norm_eps"])], dtype=np.float32), "rms_eps"),
        numpy_helper.from_array(np.array([2.0], dtype=np.float32), "pow_two"),
        numpy_helper.from_array(np.array([1.0 / np.sqrt(float(dh))], dtype=np.float32), "attn_temperature"),
        numpy_helper.from_array(cos, "rope_cos"),
        numpy_helper.from_array(sin, "rope_sin"),
        numpy_helper.from_array(_causal_mask(s), "causal_mask"),
        numpy_helper.from_array(fp["norm.weight"].astype(np.float32), "norm_w"),
        numpy_helper.from_array(fp["lm_head.weight"].astype(np.float32).T, "lm_head_w_t"),
    ]
    for layer in range(2):
        prefix = f"layers.{layer}"
        for suffix, name in [
            ("rms1.weight", f"l{layer}_rms1_w"),
            ("attn.q_proj.weight", f"l{layer}_q_w_t"),
            ("attn.k_proj.weight", f"l{layer}_k_w_t"),
            ("attn.v_proj.weight", f"l{layer}_v_w_t"),
            ("attn.o_proj.weight", f"l{layer}_o_w_t"),
            ("rms2.weight", f"l{layer}_rms2_w"),
            ("mlp.gate_proj.weight", f"l{layer}_gate_w_t"),
            ("mlp.up_proj.weight", f"l{layer}_up_w_t"),
            ("mlp.down_proj.weight", f"l{layer}_down_w_t"),
        ]:
            arr = fp[f"{prefix}.{suffix}"].astype(np.float32)
            if arr.ndim == 2:
                arr = arr.T
            initializers.append(numpy_helper.from_array(arr, name))

    nodes = []
    counter = 0

    def out(prefix: str) -> str:
        nonlocal counter
        counter += 1
        return f"{prefix}_{counter}"

    def node(op: str, inputs: list[str], prefix: str, **attrs: object) -> str:
        y = out(prefix)
        nodes.append(helper.make_node(op, inputs, [y], **attrs))
        return y

    def slice2(x: str, prefix: str, rows: tuple[int, int] | None = None, cols: tuple[int, int] | None = None) -> str:
        starts = [0, 0]
        ends = [s, d]
        axes = [0, 1]
        if rows is not None:
            starts[0], ends[0] = rows
        if cols is not None:
            starts[1], ends[1] = cols
        return node("Slice", [x], prefix, starts=starts, ends=ends, axes=axes)

    def rmsnorm(x: str, weight: str, prefix: str) -> str:
        sq = node("Mul", [x, x], f"{prefix}_sq")
        mean = node("ReduceMean", [sq], f"{prefix}_mean", axes=[1], keepdims=1)
        add = node("Add", [mean, "rms_eps"], f"{prefix}_eps")
        root = node("Sqrt", [add], f"{prefix}_sqrt")
        div = node("Div", [x, root], f"{prefix}_div")
        return node("Mul", [div, weight], f"{prefix}_out")

    def rope(x: str, prefix: str) -> str:
        x0 = slice2(x, f"{prefix}_x0", cols=(0, half))
        x1 = slice2(x, f"{prefix}_x1", cols=(half, dh))
        x0c = node("Mul", [x0, "rope_cos"], f"{prefix}_x0c")
        x1s = node("Mul", [x1, "rope_sin"], f"{prefix}_x1s")
        y0 = node("Sub", [x0c, x1s], f"{prefix}_y0")
        x1c = node("Mul", [x1, "rope_cos"], f"{prefix}_x1c")
        x0s = node("Mul", [x0, "rope_sin"], f"{prefix}_x0s")
        y1 = node("Add", [x1c, x0s], f"{prefix}_y1")
        return node("Concat", [y0, y1], f"{prefix}_out", axis=1)

    def layer(x: str, layer_id: int) -> str:
        n1 = rmsnorm(x, f"l{layer_id}_rms1_w", f"l{layer_id}_rms1")
        q = node("MatMul", [n1, f"l{layer_id}_q_w_t"], f"l{layer_id}_q")
        k = node("MatMul", [n1, f"l{layer_id}_k_w_t"], f"l{layer_id}_k")
        v = node("MatMul", [n1, f"l{layer_id}_v_w_t"], f"l{layer_id}_v")
        k_rope = rope(k, f"l{layer_id}_k_rope")
        k_t = node("Transpose", [k_rope], f"l{layer_id}_k_t", perm=[1, 0])
        heads = []
        for head in range(nh):
            qh = slice2(q, f"l{layer_id}_q_h{head}", cols=(head * dh, (head + 1) * dh))
            qh = rope(qh, f"l{layer_id}_q_h{head}_rope")
            scores = node("MatMul", [qh, k_t], f"l{layer_id}_score_h{head}")
            scaled = node("Mul", [scores, "attn_temperature"], f"l{layer_id}_scaled_h{head}")
            masked = node("Add", [scaled, "causal_mask"], f"l{layer_id}_masked_h{head}")
            probs = node("Softmax", [masked], f"l{layer_id}_probs_h{head}", axis=1)
            heads.append(node("MatMul", [probs, v], f"l{layer_id}_ctx_h{head}"))
        attn_cat = node("Concat", heads, f"l{layer_id}_attn_cat", axis=1)
        attn_out = node("MatMul", [attn_cat, f"l{layer_id}_o_w_t"], f"l{layer_id}_attn_out")
        x = node("Add", [x, attn_out], f"l{layer_id}_resid_attn")
        n2 = rmsnorm(x, f"l{layer_id}_rms2_w", f"l{layer_id}_rms2")
        gate = node("MatMul", [n2, f"l{layer_id}_gate_w_t"], f"l{layer_id}_gate")
        up = node("MatMul", [n2, f"l{layer_id}_up_w_t"], f"l{layer_id}_up")
        sig = node("Sigmoid", [gate], f"l{layer_id}_silu_sig")
        silu = node("Mul", [gate, sig], f"l{layer_id}_silu")
        hidden = node("Mul", [silu, up], f"l{layer_id}_hidden")
        down = node("MatMul", [hidden, f"l{layer_id}_down_w_t"], f"l{layer_id}_down")
        return node("Add", [x, down], f"l{layer_id}_resid_ffn")

    x = "input_x"
    x = layer(x, 0)
    x = layer(x, 1)
    x = rmsnorm(x, "norm_w", "final_norm")
    last = slice2(x, "last_token", rows=(s - 1, s), cols=(0, d))
    logits = node("MatMul", [last, "lm_head_w_t"], "logits")

    input_x = helper.make_tensor_value_info("input_x", TensorProto.FLOAT, [s, d])
    output_y = helper.make_tensor_value_info("output_y", TensorProto.FLOAT, [1, vocab])
    nodes.append(helper.make_node("Identity", [logits], ["output_y"]))
    graph = helper.make_graph(nodes, "third_party_onnx2c_tiny_lm", [input_x], [output_y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 9)])
    model.ir_version = 7
    onnx.checker.check_model(model)
    onnx.save(model, model_path)


def _onnx_generator_source(
    model: str,
    output_path: Path,
    *,
    prompt_len: int,
    mlp_hidden: int,
    conv_channels: int,
) -> str:
    script_path = Path(__file__).resolve()
    extra = ""
    if model == "tiny_lm":
        extra = f", prompt_len={int(prompt_len)}"
    elif model == "mlp":
        extra = f", hidden_dim={int(mlp_hidden)}"
    elif model == "conv":
        extra = f", conv_channels={int(conv_channels)}"
    return f"""import runpy
ns = runpy.run_path({str(script_path)!r})
ns['_write_{model}_onnx']({str(output_path)!r}{extra})
"""


def _generate_onnx(
    model: str,
    model_path: Path,
    *,
    prompt_len: int,
    mlp_hidden: int,
    conv_channels: int,
) -> None:
    if not ONNX_PYTHON.exists():
        raise FileNotFoundError(f"{ONNX_PYTHON} not found; create it and install onnx/numpy first")
    generator = GENERATED_DIR / f"third_party_onnx2c_{model}_make_onnx.py"
    generator.write_text(
        _onnx_generator_source(
            model,
            model_path,
            prompt_len=prompt_len,
            mlp_hidden=mlp_hidden,
            conv_channels=conv_channels,
        )
    )
    run_checked([str(ONNX_PYTHON), str(generator)], cwd=REPO_ROOT)


def _generate_c_from_onnx(model_path: Path, c_path: Path, *, func_name: str) -> None:
    if not ONNX2C_BIN.exists():
        raise FileNotFoundError(f"{ONNX2C_BIN} not found; build onnx2c first")
    proc = subprocess.run(
        [str(ONNX2C_BIN), "-l", "0", "-f", func_name, str(model_path)],
        cwd=str(REPO_ROOT),
        check=True,
        text=True,
        capture_output=True,
    )
    c_path.write_text(proc.stdout)


def _read_entry_signature(c_path: Path, func_name: str) -> str:
    text = c_path.read_text()
    match = re.search(rf"void\s+{re.escape(func_name)}\s*\(([^)]*)\)", text)
    if not match:
        raise RuntimeError(f"could not find generated {func_name} signature in {c_path}")
    return " ".join(match.group(1).split())


def _render_conv_wrapper(generated_c_name: str, signature: str) -> str:
    input_x = _deterministic_input()
    return f"""// Bare-metal baseline generated by third-party onnx2c.
// onnx2c: https://github.com/kraiskil/onnx2c
// Generated model source: {generated_c_name}
#include <stdint.h>
#include <stdio.h>

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

void onnx2c_conv4({signature});

{_c_array("input_x_flat", input_x)}
static float output_y[1][1][1][1];

int main(void)
{{
    const float (*input_x)[1][8][8] = (const float (*)[1][8][8])input_x_flat;
    reset_timer();
    uint32_t t0 = read_mcycle32();
    onnx2c_conv4(input_x, output_y);
    uint32_t t1 = read_mcycle32();
    uint32_t cycles = t0 - t1;
    float y = output_y[0][0][0][0];
    printf("third_party_onnx2c_conv4 cycles=%lu output=%.9g pred=%d\\n",
           (unsigned long)cycles, (double)y, y >= 0.5f);
    puts("EXIT SUCCESS");
    return 0;
}}
"""


def _render_mlp_wrapper(generated_c_name: str, signature: str, *, hidden_dim: int) -> str:
    input_x = _deterministic_mlp_input(hidden_dim)
    return f"""// Bare-metal MLP baseline generated by third-party onnx2c.
// onnx2c: https://github.com/kraiskil/onnx2c
// Generated model source: {generated_c_name}
#include <stdint.h>
#include <stdio.h>

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

void onnx2c_mlp({signature});

{_c_array("input_x_flat", input_x)}
static float output_y[1][1];

int main(void)
{{
    const float (*input_x)[{int(hidden_dim)}] = (const float (*)[{int(hidden_dim)}])input_x_flat;
    reset_timer();
    uint32_t t0 = read_mcycle32();
    onnx2c_mlp(input_x, output_y);
    uint32_t t1 = read_mcycle32();
    uint32_t cycles = t0 - t1;
    float y = output_y[0][0];
    printf("third_party_onnx2c_mlp cycles=%lu output=%.9g pred=%d\\n",
           (unsigned long)cycles, (double)y, y >= 0.5f);
    puts("EXIT SUCCESS");
    return 0;
}}
"""


def _render_tiny_lm_wrapper(generated_c_name: str, signature: str, *, prompt: str, prompt_len: int) -> str:
    run_dir = TINYSTORIES_RUN_DIR
    ids, vocab = _encode_prompt(run_dir, prompt, prompt_len)
    fp = np.load(run_dir / "tinystories_char_lm_fp32.npz")
    embeddings = fp["tok_embeddings.weight"].astype(np.float32)
    input_x = embeddings[np.array(ids, dtype=np.int64)]

    def esc(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"')

    vocab_body = ", ".join(f'"{esc(tok)}"' for tok in vocab)
    return f"""// Bare-metal TinyLM baseline generated by third-party onnx2c.
// onnx2c: https://github.com/kraiskil/onnx2c
// Generated model source: {generated_c_name}
#include <stdint.h>
#include <stdio.h>

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

void onnx2c_tiny_lm({signature});

{_c_array("input_x_flat", input_x)}
static const int input_ids[{len(ids)}] = {{ {", ".join(str(i) for i in ids)} }};
static const char *const vocab_tokens[{len(vocab)}] = {{ {vocab_body} }};
static float output_y[1][{len(vocab)}];

int main(void)
{{
    const float (*input_x)[32] = (const float (*)[32])input_x_flat;
    reset_timer();
    uint32_t t0 = read_mcycle32();
    onnx2c_tiny_lm(input_x, output_y);
    uint32_t t1 = read_mcycle32();
    uint32_t cycles = t0 - t1;
    int best = 0;
    for (int i = 1; i < {len(vocab)}; ++i) {{
        if (output_y[0][i] > output_y[0][best]) best = i;
    }}
    printf("third_party_onnx2c_tiny_lm prompt_len={prompt_len} cycles=%lu next_id=%d next_token='%s' logit=%.9g first_id=%d last_id=%d\\n",
           (unsigned long)cycles, best, vocab_tokens[best], (double)output_y[0][best],
           input_ids[0], input_ids[{len(ids) - 1}]);
    puts("EXIT SUCCESS");
    return 0;
}}
"""


def build_elf_and_hex(program_name: str, wrapper_source: str, generated_model_c: Path) -> tuple[Path, Path, Path]:
    GENERATED_DIR.mkdir(exist_ok=True)
    CUSTOM_DIR.mkdir(exist_ok=True)
    wrapper_path = GENERATED_DIR / f"{program_name}.c"
    wrapper_path.write_text(wrapper_source)

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
            "-DNDEBUG",
            "-nostdlib",
            "-T",
            "custom/link.ld",
            "-static",
            "custom/crt0.S",
            str(wrapper_path),
            str(generated_model_c),
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
    return wrapper_path, elf_path, hex_path


def build_model(
    model: str,
    program_name: str,
    *,
    prompt: str,
    prompt_len: int,
    mlp_hidden: int,
    conv_channels: int,
) -> tuple[Path, Path, Path, Path, Path]:
    GENERATED_DIR.mkdir(exist_ok=True)
    onnx_path = GENERATED_DIR / f"{program_name}.onnx"
    generated_c = GENERATED_DIR / f"{program_name}_onnx2c_model.c"
    if model == "mlp":
        func_name = "onnx2c_mlp"
    elif model == "conv":
        func_name = "onnx2c_conv4"
    else:
        func_name = "onnx2c_tiny_lm"
    _generate_onnx(
        model,
        onnx_path,
        prompt_len=prompt_len,
        mlp_hidden=mlp_hidden,
        conv_channels=conv_channels,
    )
    _generate_c_from_onnx(onnx_path, generated_c, func_name=func_name)
    signature = _read_entry_signature(generated_c, func_name)
    if model == "mlp":
        wrapper = _render_mlp_wrapper(generated_c.name, signature, hidden_dim=mlp_hidden)
    elif model == "conv":
        wrapper = _render_conv_wrapper(generated_c.name, signature)
    elif model == "tiny_lm":
        wrapper = _render_tiny_lm_wrapper(generated_c.name, signature, prompt=prompt, prompt_len=prompt_len)
    else:
        raise ValueError(f"unsupported model: {model}")
    wrapper_path, elf_path, hex_path = build_elf_and_hex(program_name, wrapper, generated_c)
    return onnx_path, generated_c, wrapper_path, elf_path, hex_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["mlp", "conv", "tiny_lm"], default="conv")
    parser.add_argument("--program-name", default=None)
    parser.add_argument("--prompt", default="there was a little girl named lily .")
    parser.add_argument("--prompt-len", type=int, default=9)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--conv-channels", type=int, default=16)
    parser.add_argument("--run-rtl", action="store_true")
    parser.add_argument("--maxcycles", type=int, default=40_000_000)
    parser.add_argument("--verilator-max-ticks", type=int, default=40_000_000_000)
    parser.add_argument("--timeout-s", type=int, default=420)
    args = parser.parse_args()

    default_programs = {
        "mlp": "third_party_onnx2c_mlp",
        "conv": "third_party_onnx2c_conv4",
        "tiny_lm": "third_party_onnx2c_tiny_lm",
    }
    program_name = args.program_name or default_programs[args.model]
    onnx_path, generated_c, wrapper_path, elf_path, hex_path = build_model(
        args.model,
        program_name,
        prompt=args.prompt,
        prompt_len=args.prompt_len,
        mlp_hidden=args.mlp_hidden,
        conv_channels=args.conv_channels,
    )
    print(f"onnx={onnx_path}")
    print(f"generated_c={generated_c}")
    print(f"wrapper={wrapper_path}")
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
