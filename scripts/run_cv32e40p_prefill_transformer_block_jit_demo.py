from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
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


def _causal_mask_int16(x: np.ndarray, fill_value: int = np.iinfo(np.int16).min) -> np.ndarray:
    out = np.array(x, copy=True)
    q_len = out.shape[-2]
    k_len = out.shape[-1]
    for row in range(q_len):
        if row + 1 < k_len:
            out[..., row, row + 1 :] = fill_value
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
    rms1_w = rng.uniform(0.5, 1.5, size=(d_model,)).astype(np.float32)
    rms2_w = rng.uniform(0.5, 1.5, size=(d_model,)).astype(np.float32)
    w_o = _rand_i16(rng, (attn_dim, d_model))
    w_gate = _rand_i16(rng, (d_model, ffn_dim))
    w_up = _rand_i16(rng, (d_model, ffn_dim))
    w_down = _rand_i16(rng, (ffn_dim, d_model))

    tensors: dict[str, TensorSpec] = {
        "x": TensorSpec("x", x.shape, DType.INT16, TensorKind.CONSTANT, data=x),
        "x_f": TensorSpec("x_f", x_f.shape, DType.FLOAT32, TensorKind.CONSTANT, data=x_f),
        "rms1_w": TensorSpec("rms1_w", rms1_w.shape, DType.FLOAT32, TensorKind.CONSTANT, data=rms1_w),
        "rms2_w": TensorSpec("rms2_w", rms2_w.shape, DType.FLOAT32, TensorKind.CONSTANT, data=rms2_w),
        "w_o": TensorSpec("w_o", w_o.shape, DType.INT16, TensorKind.CONSTANT, data=w_o),
        "w_gate": TensorSpec("w_gate", w_gate.shape, DType.INT16, TensorKind.CONSTANT, data=w_gate),
        "w_up": TensorSpec("w_up", w_up.shape, DType.INT16, TensorKind.CONSTANT, data=w_up),
        "w_down": TensorSpec("w_down", w_down.shape, DType.INT16, TensorKind.CONSTANT, data=w_down),
        "x_norm1": TensorSpec("x_norm1", (token_count, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
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
        "x_norm2": TensorSpec("x_norm2", (token_count, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "x_norm2_q": TensorSpec(
            "x_norm2_q",
            (token_count, d_model),
            DType.INT16,
            TensorKind.INTERMEDIATE,
            metadata={"storage_role": "A"},
        ),
        "gate_int": TensorSpec("gate_int", (token_count, ffn_dim), DType.INT16, TensorKind.INTERMEDIATE),
        "up_int": TensorSpec("up_int", (token_count, ffn_dim), DType.INT16, TensorKind.INTERMEDIATE),
        "gate_f": TensorSpec("gate_f", (token_count, ffn_dim), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "up_f": TensorSpec("up_f", (token_count, ffn_dim), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "gate_act": TensorSpec("gate_act", (token_count, ffn_dim), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "ffn_hidden": TensorSpec("ffn_hidden", (token_count, ffn_dim), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "ffn_hidden_q": TensorSpec(
            "ffn_hidden_q",
            (token_count, ffn_dim),
            DType.INT16,
            TensorKind.INTERMEDIATE,
            metadata={"storage_role": "A"},
        ),
        "ffn_out_int": TensorSpec("ffn_out_int", (token_count, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        "ffn_out_f": TensorSpec("ffn_out_f", (token_count, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "out": TensorSpec("out", (token_count, d_model), DType.FLOAT32, TensorKind.OUTPUT, is_final_output=True),
    }

    qkv_ops: list[MatMulOp] = []
    head_attn_tensors: list[str] = []
    concat_steps: list[HostOp] = []
    prev_concat_name: str | None = None

    for head_idx in range(n_heads):
        w_q = _rand_i16(rng, (d_model, d_head))
        w_k = _rand_i16(rng, (d_model, d_head))
        w_v = _rand_i16(rng, (d_model, d_head))
        tensors[f"w_q_h{head_idx}"] = TensorSpec(f"w_q_h{head_idx}", w_q.shape, DType.INT16, TensorKind.CONSTANT, data=w_q)
        tensors[f"w_k_h{head_idx}"] = TensorSpec(f"w_k_h{head_idx}", w_k.shape, DType.INT16, TensorKind.CONSTANT, data=w_k)
        tensors[f"w_v_h{head_idx}"] = TensorSpec(f"w_v_h{head_idx}", w_v.shape, DType.INT16, TensorKind.CONSTANT, data=w_v)
        tensors[f"q_int_h{head_idx}"] = TensorSpec(f"q_int_h{head_idx}", (token_count, d_head), DType.INT16, TensorKind.INTERMEDIATE)
        tensors[f"q_a_h{head_idx}"] = TensorSpec(
            f"q_a_h{head_idx}",
            (token_count, d_head),
            DType.INT16,
            TensorKind.INTERMEDIATE,
            metadata={"storage_role": "A"},
        )
        tensors[f"k_seq_h{head_idx}"] = TensorSpec(f"k_seq_h{head_idx}", (token_count, d_head), DType.INT16, TensorKind.INTERMEDIATE)
        tensors[f"k_t_h{head_idx}"] = TensorSpec(f"k_t_h{head_idx}", (d_head, token_count), DType.INT16, TensorKind.INTERMEDIATE)
        tensors[f"v_seq_h{head_idx}"] = TensorSpec(f"v_seq_h{head_idx}", (token_count, d_head), DType.INT16, TensorKind.INTERMEDIATE)
        tensors[f"v_b_h{head_idx}"] = TensorSpec(f"v_b_h{head_idx}", (token_count, d_head), DType.INT16, TensorKind.INTERMEDIATE)
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
            DType.FLOAT32,
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

        qkv_ops.extend(
            [
                MatMulOp(f"op_q_h{head_idx}", "x_norm1_q", f"w_q_h{head_idx}", f"q_int_h{head_idx}", in_dtype=DType.INT16, out_dtype=DType.INT16),
                MatMulOp(f"op_k_h{head_idx}", "x_norm1_q", f"w_k_h{head_idx}", f"k_seq_h{head_idx}", in_dtype=DType.INT16, out_dtype=DType.INT16),
                MatMulOp(f"op_v_h{head_idx}", "x_norm1_q", f"w_v_h{head_idx}", f"v_seq_h{head_idx}", in_dtype=DType.INT16, out_dtype=DType.INT16),
            ]
        )
        head_attn_tensors.append(f"attn_h{head_idx}")
        if prev_concat_name is None:
            prev_concat_name = f"attn_h{head_idx}"
            continue
        concat_name = f"attn_cat_{head_idx}"
        left_shape = prev_concat_name
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
        HostOp("rmsnorm1", "rmsnorm", inputs=["x_f", "rms1_w"], outputs=["x_norm1"], attrs={"eps": 1.0e-5}),
        HostOp(
            "quant_x_norm1",
            "quantize",
            inputs=["x_norm1"],
            outputs=["x_norm1_q"],
            attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16},
        ),
        NpuSegment(
            "seg_qkv",
            qkv_ops,
            inputs=["x_norm1_q"],
            outputs=[name for name in tensors if name.startswith(("q_int_h", "k_seq_h", "v_seq_h"))],
        ),
    ]

    for head_idx in range(n_heads):
        steps.extend(
            [
                HostOp(f"alias_q_a_h{head_idx}", "alias", inputs=[f"q_int_h{head_idx}"], outputs=[f"q_a_h{head_idx}"]),
                HostOp(
                    f"transpose_k_h{head_idx}",
                    "transpose",
                    inputs=[f"k_seq_h{head_idx}"],
                    outputs=[f"k_t_h{head_idx}"],
                    attrs={"axes": (1, 0)},
                ),
                HostOp(f"alias_v_b_h{head_idx}", "alias", inputs=[f"v_seq_h{head_idx}"], outputs=[f"v_b_h{head_idx}"]),
                NpuSegment(
                    f"seg_score_h{head_idx}",
                    [MatMulOp(f"op_qk_h{head_idx}", f"q_a_h{head_idx}", f"k_t_h{head_idx}", f"scores_h{head_idx}", in_dtype=DType.INT16, out_dtype=DType.INT16)],
                    inputs=[f"q_a_h{head_idx}", f"k_t_h{head_idx}"],
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
                    "softmax",
                    inputs=[f"masked_scores_h{head_idx}"],
                    outputs=[f"probs_h{head_idx}"],
                    attrs={"axis": -1},
                ),
                HostOp(
                    f"quantize_probs_h{head_idx}",
                    "quantize",
                    inputs=[f"probs_h{head_idx}"],
                    outputs=[f"probs_q_h{head_idx}"],
                    attrs={"scale": attn_scale, "zero_point": 0, "dtype": DType.INT16},
                ),
                NpuSegment(
                    f"seg_av_h{head_idx}",
                    [MatMulOp(f"op_av_h{head_idx}", f"probs_q_h{head_idx}", f"v_b_h{head_idx}", f"attn_h{head_idx}", shift=8, in_dtype=DType.INT16, out_dtype=DType.INT16)],
                    inputs=[f"probs_q_h{head_idx}", f"v_b_h{head_idx}"],
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
            HostOp("dequant_o", "dequantize", inputs=["o_int"], outputs=["o_f"], attrs={"scale": 1.0, "zero_point": 0}),
            HostOp("residual1", "add", inputs=["x_f", "o_f"], outputs=["resid1"]),
            HostOp("rmsnorm2", "rmsnorm", inputs=["resid1", "rms2_w"], outputs=["x_norm2"], attrs={"eps": 1.0e-5}),
            HostOp(
                "quant_x_norm2",
                "quantize",
                inputs=["x_norm2"],
                outputs=["x_norm2_q"],
                attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16},
            ),
            NpuSegment(
                "seg_ffn_up",
                [
                    MatMulOp("op_gate_proj", "x_norm2_q", "w_gate", "gate_int", in_dtype=DType.INT16, out_dtype=DType.INT16),
                    MatMulOp("op_up_proj", "x_norm2_q", "w_up", "up_int", in_dtype=DType.INT16, out_dtype=DType.INT16),
                ],
                inputs=["x_norm2_q"],
                outputs=["gate_int", "up_int"],
            ),
            HostOp("dequant_gate", "dequantize", inputs=["gate_int"], outputs=["gate_f"], attrs={"scale": act_scale, "zero_point": 0}),
            HostOp("dequant_up", "dequantize", inputs=["up_int"], outputs=["up_f"], attrs={"scale": act_scale, "zero_point": 0}),
            HostOp("silu_gate", "silu", inputs=["gate_f"], outputs=["gate_act"]),
            HostOp("ffn_mul", "mul", inputs=["gate_act", "up_f"], outputs=["ffn_hidden"]),
            HostOp(
                "quant_ffn_hidden",
                "quantize",
                inputs=["ffn_hidden"],
                outputs=["ffn_hidden_q"],
                attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16},
            ),
            NpuSegment(
                "seg_ffn_down",
                [MatMulOp("op_down_proj", "ffn_hidden_q", "w_down", "ffn_out_int", in_dtype=DType.INT16, out_dtype=DType.INT16)],
                inputs=["ffn_hidden_q"],
                outputs=["ffn_out_int"],
            ),
            HostOp("dequant_ffn_out", "dequantize", inputs=["ffn_out_int"], outputs=["ffn_out_f"], attrs={"scale": act_scale, "zero_point": 0}),
            HostOp("residual2", "add", inputs=["resid1", "ffn_out_f"], outputs=["out"]),
        ]
    )

    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=[], outputs=["out"])
    plan.add_verification_step("out", "prefill_transformer_block_out")
    return plan


def build_artifact(**kwargs):
    plan = build_plan(**kwargs)
    golden = GoldenModel()
    tensors = plan.tensors
    act_scale = float(kwargs.get("act_scale", 1.0 / 32.0))
    attn_scale = float(kwargs.get("attn_scale", 1.0 / 256.0))
    n_heads = int(kwargs["n_heads"])
    d_head = int(kwargs["d_head"])

    x = np.array(tensors["x"].data, dtype=np.int16, copy=True)
    x_f = np.array(tensors["x_f"].data, dtype=np.float32, copy=True)
    rms1_w = np.array(tensors["rms1_w"].data, dtype=np.float32, copy=True)
    rms2_w = np.array(tensors["rms2_w"].data, dtype=np.float32, copy=True)
    w_o = np.array(tensors["w_o"].data, dtype=np.int16, copy=True)
    w_gate = np.array(tensors["w_gate"].data, dtype=np.int16, copy=True)
    w_up = np.array(tensors["w_up"].data, dtype=np.int16, copy=True)
    w_down = np.array(tensors["w_down"].data, dtype=np.int16, copy=True)

    eps = np.float32(1.0e-5)
    rms1 = np.sqrt(np.mean(np.square(x_f, dtype=np.float32), axis=-1, keepdims=True, dtype=np.float32) + eps).astype(np.float32)
    x_norm1 = ((x_f / rms1) * rms1_w.reshape(1, -1)).astype(np.float32)
    x_norm1_q = golden.quantize(x_norm1, scale=act_scale, zero_point=0, out_dtype=DType.INT16)

    attn_heads: list[np.ndarray] = []
    for head_idx in range(n_heads):
        w_q = np.array(tensors[f"w_q_h{head_idx}"].data, dtype=np.int16, copy=True)
        w_k = np.array(tensors[f"w_k_h{head_idx}"].data, dtype=np.int16, copy=True)
        w_v = np.array(tensors[f"w_v_h{head_idx}"].data, dtype=np.int16, copy=True)
        q_int = golden.matmul(x_norm1_q, w_q, out_dtype=DType.INT16)
        k_seq = golden.matmul(x_norm1_q, w_k, out_dtype=DType.INT16)
        v_seq = golden.matmul(x_norm1_q, w_v, out_dtype=DType.INT16)
        scores = golden.matmul(q_int, np.array(k_seq.T, dtype=np.int16, copy=True), out_dtype=DType.INT16)
        masked_scores = _causal_mask_int16(scores)
        probs = golden.softmax(masked_scores, axis=-1).astype(np.float32)
        probs_q = golden.quantize(probs, scale=attn_scale, zero_point=0, out_dtype=DType.INT16)
        attn_h = golden.matmul(probs_q, v_seq, shift=8, out_dtype=DType.INT16)
        attn_heads.append(np.array(attn_h, copy=True))

    attn_cat = attn_heads[0]
    for attn_h in attn_heads[1:]:
        attn_cat = np.concatenate([attn_cat, attn_h], axis=-1).astype(np.int16)

    o_int = golden.matmul(attn_cat, w_o, out_dtype=DType.INT16)
    o_f = golden.dequantize(o_int, scale=1.0, zero_point=0)
    resid1 = (x_f + o_f).astype(np.float32)

    rms2 = np.sqrt(np.mean(np.square(resid1, dtype=np.float32), axis=-1, keepdims=True, dtype=np.float32) + eps).astype(np.float32)
    x_norm2 = ((resid1 / rms2) * rms2_w.reshape(1, -1)).astype(np.float32)
    x_norm2_q = golden.quantize(x_norm2, scale=act_scale, zero_point=0, out_dtype=DType.INT16)
    gate_int = golden.matmul(x_norm2_q, w_gate, out_dtype=DType.INT16)
    up_int = golden.matmul(x_norm2_q, w_up, out_dtype=DType.INT16)
    gate_f = golden.dequantize(gate_int, scale=act_scale, zero_point=0)
    up_f = golden.dequantize(up_int, scale=act_scale, zero_point=0)
    gate_act = (gate_f / (np.float32(1.0) + np.exp(-gate_f))).astype(np.float32)
    ffn_hidden = (gate_act * up_f).astype(np.float32)
    ffn_hidden_q = golden.quantize(ffn_hidden, scale=act_scale, zero_point=0, out_dtype=DType.INT16)
    ffn_out_int = golden.matmul(ffn_hidden_q, w_down, out_dtype=DType.INT16)
    ffn_out_f = golden.dequantize(ffn_out_int, scale=act_scale, zero_point=0)
    out = (resid1 + ffn_out_f).astype(np.float32)

    artifact = compile_plan(plan, {"out": np.array(out, copy=True)})
    return artifact, out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-head", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=64)
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    artifact, expected = build_artifact(
        d_model=args.d_model,
        d_head=args.d_head,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        token_count=args.token_count,
        seed=args.seed,
    )

    program_name = (
        f"cv32e40p_prefill_transformer_block_d{args.d_model}_h{args.d_head}"
        f"_nh{args.n_heads}_f{args.ffn_dim}_t{args.token_count}_s{args.seed}_v2"
    )
    program_symbol = _sanitize(program_name)
    program_path = GENERATED_DIR / f"{program_name}_program.c"
    runner_path = GENERATED_DIR / f"{program_name}_runner.c"

    source = emit_cv32e40p_program_v2(artifact, {}, program_name=program_name)
    GENERATED_DIR.mkdir(exist_ok=True)
    program_path.write_text(source)
    runner_path.write_text(_runner_source(program_symbol))

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
            "-DTINYNPU_USE_SHARED_SRAM=1",
            "-DTNPU_RUNTIME_V2_DUMP_FINAL_OUTPUTS=1",
            "-T",
            "custom/link.ld",
            "-static",
            "custom/crt0.S",
            str(runner_path),
            str(program_path),
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
    env["VERILATOR_MAX_TICKS"] = "30000000000"
    proc = _run(
        [
            str(CORE_DIR / "obj_dir" / "cv32e40p_tb_vlt_npu"),
            "+verilator+noassert",
            f"+firmware={hex_path}",
            "+maxcycles=3000000",
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
    print(f"expected_checksum={float(np.array(expected, dtype=np.float32).sum()):.6f}")
    print(proc.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
