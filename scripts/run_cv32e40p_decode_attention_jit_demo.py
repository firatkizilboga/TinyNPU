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
    make_native_int16_kv_cache_specs,
    make_rope_cos_sin_table_q14,
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


def _parse_token_indices(raw: str | None, token_capacity: int) -> list[int]:
    if raw is None or raw.strip() == "":
        return [idx for idx in (1, 9) if idx < token_capacity]
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("token index list must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("token indices must be unique")
    for idx in values:
        if idx < 0 or idx >= token_capacity:
            raise ValueError(f"token index {idx} is outside token capacity {token_capacity}")
    return values


def _rand_i16(rng: np.random.Generator, shape: tuple[int, ...], low: int = -2, high: int = 3) -> np.ndarray:
    return rng.integers(low, high, size=shape, endpoint=False, dtype=np.int16)


def _make_b_view(
    name: str,
    base_name: str,
    shape: tuple[int, int],
    *,
    cache_kind: str | None = None,
) -> TensorSpec:
    metadata = {
        "storage_view_of": base_name,
        "storage_role": "B",
        "storage_word_offset": 0,
    }
    if cache_kind is not None:
        metadata["cache_kind"] = cache_kind
    return TensorSpec(name, shape, DType.INT16, TensorKind.INTERMEDIATE, metadata=metadata)


def _rope_ref(x: np.ndarray, *, position: int, head_dim: int, theta: float) -> np.ndarray:
    x_f = np.asarray(x, dtype=np.float32)
    if x_f.shape[-1] != head_dim:
        raise ValueError(f"rope_ref expects last dimension {head_dim}, got {x_f.shape[-1]}")
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / np.float32(half)))
    angles = np.float32(position) * inv_freq
    cos = np.cos(angles).astype(np.float32).reshape(1, half)
    sin = np.sin(angles).astype(np.float32).reshape(1, half)
    out = np.array(x_f, copy=True)
    first = x_f[..., :half]
    second = x_f[..., half:head_dim]
    out[..., :half] = first * cos - second * sin
    out[..., half:head_dim] = second * cos + first * sin
    return out.astype(np.float32)


def _rope_q14_int16_ref(x_q: np.ndarray, *, position: int, head_dim: int, theta: float) -> np.ndarray:
    x_i16 = np.asarray(x_q, dtype=np.int16)
    if x_i16.shape[-1] != head_dim:
        raise ValueError(f"rope_q14_int16_ref expects last dimension {head_dim}, got {x_i16.shape[-1]}")
    if head_dim % 2 != 0:
        raise ValueError(f"rope_q14_int16_ref expects even head_dim, got {head_dim}")
    half = head_dim // 2
    cs = make_rope_cos_sin_table_q14(head_dim, position, theta).astype(np.int32)
    cos = cs[:half].reshape(1, half)
    sin = cs[half:].reshape(1, half)
    first = x_i16[..., :half].astype(np.int32)
    second = x_i16[..., half:head_dim].astype(np.int32)
    lo = np.clip((first * cos - second * sin) >> 14, -32768, 32767).astype(np.int16)
    hi = np.clip((second * cos + first * sin) >> 14, -32768, 32767).astype(np.int16)
    return np.concatenate([lo, hi], axis=-1).astype(np.int16)


def build_artifact(
    *,
    d_model: int | None = None,
    n_heads: int = 1,
    n_kv_heads: int = 1,
    d_head: int = 8,
    token_capacity: int = 16,
    token_indices: list[int] | None = None,
    seed: int = 0,
    attn_scale: float = 1.0 / 256.0,
    rope_theta: float = 10000.0,
):
    resolved_d_model = d_model if d_model is not None else (n_heads * d_head)
    if resolved_d_model <= 0 or resolved_d_model % 8 != 0:
        raise ValueError("d_model must be a positive multiple of 8")
    if n_heads <= 0:
        raise ValueError("n_heads must be positive")
    if n_kv_heads <= 0:
        raise ValueError("n_kv_heads must be positive")
    if n_kv_heads > n_heads:
        raise ValueError("n_kv_heads must be <= n_heads")
    if n_heads % n_kv_heads != 0:
        raise ValueError("n_heads must be divisible by n_kv_heads for grouped-query mapping")
    if d_head <= 0 or d_head % 8 != 0:
        raise ValueError("d_head must be a positive multiple of 8")
    if resolved_d_model != n_heads * d_head:
        raise ValueError("d_model must equal n_heads * d_head")
    if token_capacity <= 0 or token_capacity % 8 != 0:
        raise ValueError("token_capacity must be a positive multiple of 8")

    indices = token_indices or [idx for idx in (1, 9) if idx < token_capacity]
    if not indices:
        raise ValueError("at least one token index is required")
    if len(indices) > token_capacity:
        raise ValueError("number of decode steps must be <= token_capacity")
    if rope_theta <= 0.0:
        raise ValueError("rope_theta must be positive")
    for prev, cur in zip(indices, indices[1:]):
        if cur <= prev:
            raise ValueError("token_indices must be strictly increasing decode positions")

    rng = np.random.default_rng(seed)
    golden = GoldenModel()
    use_hw_rope = (d_head % 16) == 0

    tensors: dict[str, TensorSpec] = {}
    q_to_kv_head = [q_head * n_kv_heads // n_heads for q_head in range(n_heads)]
    decode_steps = len(indices)
    decode_step_names = [f"s{step_idx}" for step_idx in range(decode_steps)]
    cache_slot_indices = list(range(decode_steps))

    k_cache_by_head = [np.zeros((d_head, token_capacity), dtype=np.int16) for _ in range(n_kv_heads)]
    v_cache_by_head = [np.zeros((token_capacity, d_head), dtype=np.int16) for _ in range(n_kv_heads)]

    for kv_head in range(n_kv_heads):
        tensors.update(
            make_native_int16_kv_cache_specs(
                k_base_name=f"k_cache_h{kv_head}",
                v_base_name=f"v_cache_h{kv_head}",
                d_head=d_head,
                token_capacity=token_capacity,
                token_names=decode_step_names,
                token_indices=cache_slot_indices,
            )
        )

    w_q_by_head: dict[int, np.ndarray] = {}
    w_o_by_head: dict[int, np.ndarray] = {}
    for q_head in range(n_heads):
        w_q = _rand_i16(rng, (resolved_d_model, d_head))
        w_o = _rand_i16(rng, (d_head, resolved_d_model))
        w_q_by_head[q_head] = w_q
        w_o_by_head[q_head] = w_o
        w_q_name = f"w_q_h{q_head}"
        w_o_name = f"w_o_h{q_head}"
        tensors[w_q_name] = TensorSpec(w_q_name, w_q.shape, DType.INT16, TensorKind.CONSTANT, data=w_q)
        tensors[w_o_name] = TensorSpec(w_o_name, w_o.shape, DType.INT16, TensorKind.CONSTANT, data=w_o)
    w_k_by_head: dict[int, np.ndarray] = {}
    w_v_by_head: dict[int, np.ndarray] = {}
    for kv_head in range(n_kv_heads):
        w_k = _rand_i16(rng, (resolved_d_model, d_head))
        w_v = _rand_i16(rng, (resolved_d_model, d_head))
        w_k_by_head[kv_head] = w_k
        w_v_by_head[kv_head] = w_v
        w_k_name = f"w_k_h{kv_head}"
        w_v_name = f"w_v_h{kv_head}"
        tensors[w_k_name] = TensorSpec(w_k_name, w_k.shape, DType.INT16, TensorKind.CONSTANT, data=w_k)
        tensors[w_v_name] = TensorSpec(w_v_name, w_v.shape, DType.INT16, TensorKind.CONSTANT, data=w_v)
    if n_heads == 1:
        tensors["reduce_zero"] = TensorSpec(
            "reduce_zero",
            (1, resolved_d_model),
            DType.INT16,
            TensorKind.CONSTANT,
            data=np.zeros((1, resolved_d_model), dtype=np.int16),
        )
    tensors["rope_identity"] = TensorSpec(
        "rope_identity",
        (d_head, d_head),
        DType.INT16,
        TensorKind.CONSTANT,
        data=np.eye(d_head, dtype=np.int16),
    )
    proj_scale = 1.0

    steps: list[NpuSegment | HostOp] = []
    scores_verify_name = "scores_s0_h0"
    scores_verify_expected: np.ndarray | None = None
    k_rope_verify_name: str | None = None
    k_rope_verify_expected: np.ndarray | None = None
    final_out_name = f"out_s{decode_steps - 1}"
    final_out_expected = np.zeros((1, resolved_d_model), dtype=np.float32)

    for step_idx, token_pos in enumerate(indices):
        step_name = decode_step_names[step_idx]
        cache_slot = cache_slot_indices[step_idx]
        valid_tokens = step_idx + 1

        seg_proj_ops: list[MatMulOp] = []
        seg_proj_outputs: list[str] = []
        seg_cache_score_ops: list[MatMulOp] = []
        seg_cache_score_outputs: list[str] = []
        seg_value_ops: list[MatMulOp] = []
        seg_value_outputs: list[str] = []
        step_out_part_names: list[str] = []
        step_out_expected = np.zeros((1, resolved_d_model), dtype=np.float32)
        x_t = _rand_i16(rng, (1, resolved_d_model))
        x_name = f"x_s{step_idx}"
        tensors[x_name] = TensorSpec(x_name, x_t.shape, DType.INT16, TensorKind.CONSTANT, data=x_t)

        q_rope_q_by_head: dict[int, np.ndarray] = {}
        k_rope_q_by_head: dict[int, np.ndarray] = {}
        for kv_head in range(n_kv_heads):
            k_proj_name = f"k_proj_s{step_idx}_h{kv_head}"
            k_proj_f_name = f"k_proj_f_s{step_idx}_h{kv_head}"
            k_rope_f_name = f"k_rope_f_s{step_idx}_h{kv_head}"
            k_rope_q_name = f"k_rope_q_s{step_idx}_h{kv_head}"
            v_slot_name = f"v_cache_h{kv_head}_{step_name}"

            k_val = golden.matmul(x_t, w_k_by_head[kv_head], out_dtype=DType.INT16)
            v_val = golden.matmul(x_t, w_v_by_head[kv_head], out_dtype=DType.INT16)
            v_cache_by_head[kv_head][cache_slot, :] = v_val[0]
            k_val_f = golden.dequantize(k_val, scale=proj_scale, zero_point=0)
            if use_hw_rope:
                k_rope_q = _rope_q14_int16_ref(k_val, position=token_pos, head_dim=d_head, theta=rope_theta)
            else:
                k_rope_f = _rope_ref(k_val_f, position=token_pos, head_dim=d_head, theta=rope_theta)
                k_rope_q = golden.quantize(k_rope_f, scale=proj_scale, zero_point=0, out_dtype=DType.INT16)
            k_cache_by_head[kv_head][:, cache_slot] = k_rope_q[0]
            k_rope_q_by_head[kv_head] = k_rope_q
            if step_idx == 0 and kv_head == 0:
                k_rope_verify_name = k_rope_q_name
                k_rope_verify_expected = k_rope_q.copy()

            tensors[k_rope_q_name] = TensorSpec(k_rope_q_name, k_val.shape, DType.INT16, TensorKind.INTERMEDIATE)
            tensors[k_proj_name] = TensorSpec(k_proj_name, k_val.shape, DType.INT16, TensorKind.INTERMEDIATE)
            tensors[k_proj_f_name] = TensorSpec(k_proj_f_name, k_val.shape, DType.FLOAT32, TensorKind.INTERMEDIATE)
            tensors[k_rope_f_name] = TensorSpec(k_rope_f_name, k_val.shape, DType.FLOAT32, TensorKind.INTERMEDIATE)
            seg_proj_ops.append(MatMulOp(f"op_k_proj_h{kv_head}_s{step_idx}", x_name, f"w_k_h{kv_head}", k_proj_name))
            seg_proj_outputs.append(k_proj_name)
            seg_proj_ops.append(MatMulOp(f"op_v_proj_h{kv_head}_s{step_idx}", x_name, f"w_v_h{kv_head}", v_slot_name))

        for q_head in range(n_heads):
            q_proj_name = f"q_proj_s{step_idx}_h{q_head}"
            q_proj_f_name = f"q_proj_f_s{step_idx}_h{q_head}"
            q_rope_f_name = f"q_rope_f_s{step_idx}_h{q_head}"
            q_rope_q_name = f"q_rope_q_s{step_idx}_h{q_head}"
            q_val = golden.matmul(x_t, w_q_by_head[q_head], out_dtype=DType.INT16)
            if use_hw_rope:
                q_rope_q = _rope_q14_int16_ref(q_val, position=token_pos, head_dim=d_head, theta=rope_theta)
            else:
                q_val_f = golden.dequantize(q_val, scale=proj_scale, zero_point=0)
                q_rope_f = _rope_ref(q_val_f, position=token_pos, head_dim=d_head, theta=rope_theta)
                q_rope_q = golden.quantize(q_rope_f, scale=proj_scale, zero_point=0, out_dtype=DType.INT16)
            q_rope_q_by_head[q_head] = q_rope_q
            tensors[q_proj_name] = TensorSpec(q_proj_name, q_val.shape, DType.INT16, TensorKind.INTERMEDIATE)
            tensors[q_proj_f_name] = TensorSpec(q_proj_f_name, q_val.shape, DType.FLOAT32, TensorKind.INTERMEDIATE)
            tensors[q_rope_f_name] = TensorSpec(q_rope_f_name, q_val.shape, DType.FLOAT32, TensorKind.INTERMEDIATE)
            tensors[q_rope_q_name] = TensorSpec(
                q_rope_q_name,
                q_val.shape,
                DType.INT16,
                TensorKind.INTERMEDIATE,
                metadata={"storage_role": "A"},
            )
            seg_proj_ops.append(MatMulOp(f"op_q_proj_h{q_head}_s{step_idx}", x_name, f"w_q_h{q_head}", q_proj_name))
            seg_proj_outputs.append(q_proj_name)

        steps.append(NpuSegment(f"seg_proj_s{step_idx}", seg_proj_ops, inputs=[], outputs=seg_proj_outputs))

        for kv_head in range(n_kv_heads):
            steps.append(
                HostOp(
                    f"dequant_k_proj_s{step_idx}_h{kv_head}",
                    "dequantize",
                    inputs=[f"k_proj_s{step_idx}_h{kv_head}"],
                    outputs=[f"k_proj_f_s{step_idx}_h{kv_head}"],
                    attrs={"scale": proj_scale, "zero_point": 0},
                )
            )
            steps.append(
                HostOp(
                    f"rope_k_proj_s{step_idx}_h{kv_head}",
                    "rope",
                    inputs=[f"k_proj_f_s{step_idx}_h{kv_head}"],
                    outputs=[f"k_rope_f_s{step_idx}_h{kv_head}"],
                    attrs={"head_dim": d_head, "position": token_pos, "theta": rope_theta},
                )
            )
            steps.append(
                HostOp(
                    f"quant_k_rope_s{step_idx}_h{kv_head}",
                    "quantize",
                    inputs=[f"k_rope_f_s{step_idx}_h{kv_head}"],
                    outputs=[f"k_rope_q_s{step_idx}_h{kv_head}"],
                    attrs={"scale": proj_scale, "zero_point": 0, "dtype": DType.INT16},
                )
            )
        for kv_head in range(n_kv_heads):
            steps.append(
                HostOp(
                    f"k_scatter_s{step_idx}_h{kv_head}",
                    "k_cache_scatter_write",
                    inputs=[f"k_rope_q_s{step_idx}_h{kv_head}"],
                    outputs=[f"k_cache_h{kv_head}_{step_name}"],
                    attrs={"token_index": cache_slot, "k_cache_base": f"k_cache_h{kv_head}"},
                )
            )

        for q_head in range(n_heads):
            steps.append(
                HostOp(
                    f"dequant_q_proj_s{step_idx}_h{q_head}",
                    "dequantize",
                    inputs=[f"q_proj_s{step_idx}_h{q_head}"],
                    outputs=[f"q_proj_f_s{step_idx}_h{q_head}"],
                    attrs={"scale": proj_scale, "zero_point": 0},
                )
            )
            steps.append(
                HostOp(
                    f"rope_q_proj_s{step_idx}_h{q_head}",
                    "rope",
                    inputs=[f"q_proj_f_s{step_idx}_h{q_head}"],
                    outputs=[f"q_rope_f_s{step_idx}_h{q_head}"],
                    attrs={"head_dim": d_head, "position": token_pos, "theta": rope_theta},
                )
            )
            steps.append(
                HostOp(
                    f"quant_q_rope_s{step_idx}_h{q_head}",
                    "quantize",
                    inputs=[f"q_rope_f_s{step_idx}_h{q_head}"],
                    outputs=[f"q_rope_q_s{step_idx}_h{q_head}"],
                    attrs={"scale": proj_scale, "zero_point": 0, "dtype": DType.INT16},
                )
            )

        cache_score_inputs: list[str] = []
        attn_q_inputs: list[str] = []
        for q_head in range(n_heads):
            kv_head = q_to_kv_head[q_head]
            query_name = f"q_rope_q_s{step_idx}_h{q_head}"
            cache_score_inputs.append(query_name)
            scores_name = f"scores_s{step_idx}_h{q_head}"
            probs_f16_name = f"probs_f16_s{step_idx}_h{q_head}"
            attn_q_name = f"attn_q_s{step_idx}_h{q_head}"
            attn_name = f"attn_s{step_idx}_h{q_head}"
            out_part_name = f"out_part_s{step_idx}_h{q_head}"
            k_view_name = f"k_cache_h{kv_head}_valid_s{step_idx}"
            v_view_name = f"v_cache_h{kv_head}_valid_s{step_idx}"

            tensors[scores_name] = TensorSpec(scores_name, (1, valid_tokens), DType.INT16, TensorKind.INTERMEDIATE)
            tensors[probs_f16_name] = TensorSpec(probs_f16_name, (1, valid_tokens), DType.INT16, TensorKind.INTERMEDIATE)
            tensors[attn_q_name] = TensorSpec(
                attn_q_name,
                (1, valid_tokens),
                DType.INT16,
                TensorKind.INTERMEDIATE,
                metadata={"storage_role": "A"},
            )
            tensors[attn_name] = TensorSpec(attn_name, (1, d_head), DType.INT16, TensorKind.INTERMEDIATE)
            tensors[out_part_name] = TensorSpec(out_part_name, (1, resolved_d_model), DType.INT16, TensorKind.INTERMEDIATE)
            if k_view_name not in tensors:
                tensors[k_view_name] = _make_b_view(k_view_name, f"k_cache_h{kv_head}", (d_head, valid_tokens), cache_kind="K")
            if v_view_name not in tensors:
                tensors[v_view_name] = _make_b_view(v_view_name, f"v_cache_h{kv_head}", (valid_tokens, d_head), cache_kind="V")

            k_view = k_cache_by_head[kv_head][:, :valid_tokens]
            v_view = v_cache_by_head[kv_head][:valid_tokens, :]
            scores = golden.matmul(q_rope_q_by_head[q_head], k_view, out_dtype=DType.INT16)
            probs = golden.softmax(scores, axis=-1).astype(np.float32)
            probs_f16 = probs.astype(np.float16).astype(np.float32)
            attn_q = golden.quantize(probs_f16, scale=attn_scale, out_dtype=DType.INT16)
            attn = golden.matmul(attn_q, v_view, shift=8, out_dtype=DType.INT16)
            out_part = golden.matmul(attn, w_o_by_head[q_head], out_dtype=DType.INT16)
            step_out_expected = step_out_expected + out_part.astype(np.float32)

            if step_idx == 0 and q_head == 0:
                scores_verify_expected = scores

            seg_cache_score_ops.append(MatMulOp(f"op_qk_s{step_idx}_h{q_head}", query_name, k_view_name, scores_name))
            seg_cache_score_outputs.append(scores_name)
            seg_value_ops.append(MatMulOp(f"op_av_s{step_idx}_h{q_head}", attn_q_name, v_view_name, attn_name, shift=8))
            seg_value_ops.append(MatMulOp(f"op_o_s{step_idx}_h{q_head}", attn_name, f"w_o_h{q_head}", out_part_name))
            seg_value_outputs.append(out_part_name)
            step_out_part_names.append(out_part_name)
            attn_q_inputs.append(attn_q_name)

        steps.append(
            NpuSegment(
                f"seg_cache_score_s{step_idx}",
                seg_cache_score_ops,
                inputs=cache_score_inputs,
                outputs=seg_cache_score_outputs,
            )
        )
        for q_head in range(n_heads):
            scores_name = f"scores_s{step_idx}_h{q_head}"
            probs_f16_name = f"probs_f16_s{step_idx}_h{q_head}"
            attn_q_name = f"attn_q_s{step_idx}_h{q_head}"
            steps.append(
                HostOp(
                    f"softmax_scores_f16_s{step_idx}_h{q_head}",
                    "softmax_f16",
                    inputs=[scores_name],
                    outputs=[probs_f16_name],
                    attrs={"axis": -1},
                )
            )
            steps.append(
                HostOp(
                    f"quantize_probs_s{step_idx}_h{q_head}",
                    "quantize",
                    inputs=[probs_f16_name],
                    outputs=[attn_q_name],
                    attrs={"scale": attn_scale, "zero_point": 0, "dtype": DType.INT16},
                )
            )

        steps.append(
            NpuSegment(
                f"seg_value_s{step_idx}",
                seg_value_ops,
                inputs=attn_q_inputs,
                outputs=seg_value_outputs,
            )
        )

        out_step_name = f"out_s{step_idx}"
        out_step_kind = TensorKind.OUTPUT if step_idx == (decode_steps - 1) else TensorKind.INTERMEDIATE
        tensors[out_step_name] = TensorSpec(
            out_step_name,
            (1, resolved_d_model),
            DType.FLOAT32,
            out_step_kind,
            is_final_output=step_idx == (decode_steps - 1),
        )

        if n_heads == 1:
            steps.append(
                HostOp(
                    f"reduce_heads_s{step_idx}_add0",
                    "add",
                    inputs=[step_out_part_names[0], "reduce_zero"],
                    outputs=[out_step_name],
                )
            )
        else:
            first_add_to_out = n_heads == 2
            acc_name = out_step_name if first_add_to_out else f"head_sum_s{step_idx}_1"
            if not first_add_to_out:
                tensors[acc_name] = TensorSpec(acc_name, (1, resolved_d_model), DType.FLOAT32, TensorKind.INTERMEDIATE)
            steps.append(
                HostOp(
                    f"reduce_heads_s{step_idx}_1",
                    "add",
                    inputs=[step_out_part_names[0], step_out_part_names[1]],
                    outputs=[acc_name],
                )
            )
            for q_head in range(2, n_heads):
                is_last = q_head == (n_heads - 1)
                dst_name = out_step_name if is_last else f"head_sum_s{step_idx}_{q_head}"
                if not is_last:
                    tensors[dst_name] = TensorSpec(dst_name, (1, resolved_d_model), DType.FLOAT32, TensorKind.INTERMEDIATE)
                steps.append(
                    HostOp(
                        f"reduce_heads_s{step_idx}_{q_head}",
                        "add",
                        inputs=[acc_name, step_out_part_names[q_head]],
                        outputs=[dst_name],
                    )
                )
                acc_name = dst_name

        if step_idx == (decode_steps - 1):
            final_out_expected = step_out_expected

    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=[], outputs=[final_out_name])
    if scores_verify_expected is None:
        raise RuntimeError("internal error: missing score verification tensor")
    if k_rope_verify_name is None or k_rope_verify_expected is None:
        raise RuntimeError("internal error: missing K-RoPE verification tensor")
    plan.add_verification_step(k_rope_verify_name, k_rope_verify_name)
    plan.add_verification_step(scores_verify_name, scores_verify_name)
    plan.add_verification_step(final_out_name, "decode_attention")
    artifact = compile_plan(
        plan,
        {
            k_rope_verify_name: k_rope_verify_expected,
            scores_verify_name: scores_verify_expected,
            final_out_name: final_out_expected.astype(np.float32),
        },
    )
    return artifact, final_out_expected.astype(np.float32), resolved_d_model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--n-kv-heads", type=int, default=1)
    parser.add_argument("--d-head", type=int, default=8)
    parser.add_argument("--token-capacity", type=int, default=16)
    parser.add_argument("--token-indices", type=str, default="1,9")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    token_indices = _parse_token_indices(args.token_indices, args.token_capacity)
    artifact, expected, resolved_d_model = build_artifact(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_head=args.d_head,
        token_capacity=args.token_capacity,
        token_indices=token_indices,
        seed=args.seed,
    )

    program_name = (
        f"cv32e40p_decode_attention_dm{resolved_d_model}_nh{args.n_heads}_nkv{args.n_kv_heads}"
        f"_dh{args.d_head}_t{args.token_capacity}_n{len(token_indices)}_s{args.seed}_v2"
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
    )
    _run([objcopy, "-O", "verilog", str(elf_path), str(hex_path)], cwd=CORE_DIR)

    env = dict(os.environ)
    env["VERILATOR_MAX_TICKS"] = "3000000000"
    proc = _run(
        [
            str(CORE_DIR / "obj_dir" / "cv32e40p_tb_vlt_npu"),
            "+verilator+noassert",
            f"+firmware={hex_path}",
            "+maxcycles=1000000",
        ],
        cwd=CORE_DIR,
        env=env,
        capture=True,
    )
    print(f"program={program_name}")
    print(
        f"d_model={resolved_d_model} n_heads={args.n_heads} n_kv_heads={args.n_kv_heads} "
        f"d_head={args.d_head} token_capacity={args.token_capacity} token_indices={token_indices} seed={args.seed}"
    )
    print(f"expected_checksum={float(expected.astype(np.float32).sum()):.6f}")
    print(proc.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
