from __future__ import annotations

import numpy as np

from tinynpu_jit import (
    DType,
    ExecutionPlan,
    HostOp,
    IRBuilder,
    MatMulOp,
    NpuSegment,
    TensorKind,
    TensorSpec,
    VerifyTensor,
    compile_plan,
    make_native_int16_kv_cache_specs,
    make_rope_cos_sin_table_q14,
)
from tinynpu_jit.golden import GoldenModel


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


class _LegacyPlanCollector:
    def __init__(self):
        self.tensors: dict[str, TensorSpec] = {}
        self.steps: list[NpuSegment | HostOp | VerifyTensor] = []
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.metadata: dict[str, object] = {}

    def tensor(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: DType,
        kind: TensorKind,
        *,
        data: np.ndarray | None = None,
        is_final_output: bool = False,
        verify_label: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TensorSpec:
        spec = TensorSpec(
            name=name,
            shape=shape,
            dtype=dtype,
            kind=kind,
            data=data,
            is_final_output=is_final_output,
            verify_label=verify_label,
            metadata=dict(metadata or {}),
        )
        self.tensors[name] = spec
        return spec

    def constant(self, name: str, data: np.ndarray, dtype: DType, *, metadata: dict[str, object] | None = None) -> TensorSpec:
        return self.tensor(name, tuple(int(dim) for dim in data.shape), dtype, TensorKind.CONSTANT, data=data, metadata=metadata)

    def intermediate(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: DType,
        *,
        metadata: dict[str, object] | None = None,
    ) -> TensorSpec:
        return self.tensor(name, shape, dtype, TensorKind.INTERMEDIATE, metadata=metadata)

    def output(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: DType,
        *,
        is_final_output: bool = False,
        verify_label: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> TensorSpec:
        return self.tensor(
            name,
            shape,
            dtype,
            TensorKind.OUTPUT if is_final_output else TensorKind.INTERMEDIATE,
            is_final_output=is_final_output,
            verify_label=verify_label,
            metadata=metadata,
        )

    def host(
        self,
        name: str,
        kind: str,
        *,
        inputs: list[str],
        outputs: list[str],
        attrs: dict[str, object] | None = None,
    ) -> HostOp:
        step = HostOp(name=name, kind=kind, inputs=inputs, outputs=outputs, attrs=dict(attrs or {}))
        self.steps.append(step)
        return step

    def matmul(self, name: str, lhs: str, rhs: str, out: str, **kwargs: object) -> MatMulOp:
        return MatMulOp(name=name, lhs=lhs, rhs=rhs, out=out, **kwargs)

    def segment(self, name: str, *, ops: list[MatMulOp], inputs: list[str], outputs: list[str]) -> NpuSegment:
        step = NpuSegment(name=name, ops=ops, inputs=inputs, outputs=outputs)
        self.steps.append(step)
        return step

    def add_verification(self, tensor_name: str, label: str | None = None) -> None:
        tensor = self.tensors[tensor_name]
        self.steps.append(
            VerifyTensor(
                tensor_name=tensor_name,
                label=label or tensor.verify_label or tensor_name,
                is_final_output=tensor.is_final_output,
                float_atol=float(tensor.metadata.get("verify_atol", 1.0e-3)),
            )
        )

    def verify(self, tensor_name: str, label: str | None = None) -> None:
        self.add_verification(tensor_name, label)

    def finalize(
        self,
        *,
        inputs: list[str],
        outputs: list[str],
        metadata: dict[str, object] | None = None,
    ) -> ExecutionPlan:
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        merged = dict(self.metadata)
        if metadata:
            merged.update(metadata)
        return ExecutionPlan(
            tensors=dict(self.tensors),
            steps=list(self.steps),
            inputs=list(self.inputs),
            outputs=list(self.outputs),
            metadata=merged,
        )


def _build_decode_artifact(
    collector: IRBuilder | _LegacyPlanCollector,
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
    tensors = collector.tensors
    steps = collector.steps
    b = collector

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
        b.constant(w_q_name, w_q, DType.INT16)
        b.constant(w_o_name, w_o, DType.INT16)
    w_k_by_head: dict[int, np.ndarray] = {}
    w_v_by_head: dict[int, np.ndarray] = {}
    for kv_head in range(n_kv_heads):
        w_k = _rand_i16(rng, (resolved_d_model, d_head))
        w_v = _rand_i16(rng, (resolved_d_model, d_head))
        w_k_by_head[kv_head] = w_k
        w_v_by_head[kv_head] = w_v
        w_k_name = f"w_k_h{kv_head}"
        w_v_name = f"w_v_h{kv_head}"
        b.constant(w_k_name, w_k, DType.INT16)
        b.constant(w_v_name, w_v, DType.INT16)
    if n_heads == 1:
        b.constant("reduce_zero", np.zeros((1, resolved_d_model), dtype=np.int16), DType.INT16)
    b.constant("rope_identity", np.eye(d_head, dtype=np.int16), DType.INT16)
    proj_scale = 1.0

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
        b.constant(x_name, x_t, DType.INT16)

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

            b.intermediate(k_rope_q_name, k_val.shape, DType.INT16)
            b.intermediate(k_proj_name, k_val.shape, DType.INT16)
            b.intermediate(k_proj_f_name, k_val.shape, DType.FLOAT32)
            b.intermediate(k_rope_f_name, k_val.shape, DType.FLOAT32)
            seg_proj_ops.append(b.matmul(f"op_k_proj_h{kv_head}_s{step_idx}", x_name, f"w_k_h{kv_head}", k_proj_name))
            seg_proj_outputs.append(k_proj_name)
            seg_proj_ops.append(b.matmul(f"op_v_proj_h{kv_head}_s{step_idx}", x_name, f"w_v_h{kv_head}", v_slot_name))

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
            b.intermediate(q_proj_name, q_val.shape, DType.INT16)
            b.intermediate(q_proj_f_name, q_val.shape, DType.FLOAT32)
            b.intermediate(q_rope_f_name, q_val.shape, DType.FLOAT32)
            b.intermediate(q_rope_q_name, q_val.shape, DType.INT16, metadata={"storage_role": "A"})
            seg_proj_ops.append(b.matmul(f"op_q_proj_h{q_head}_s{step_idx}", x_name, f"w_q_h{q_head}", q_proj_name))
            seg_proj_outputs.append(q_proj_name)

        b.segment(f"seg_proj_s{step_idx}", ops=seg_proj_ops, inputs=[], outputs=seg_proj_outputs)

        for kv_head in range(n_kv_heads):
            b.host(
                f"dequant_k_proj_s{step_idx}_h{kv_head}",
                "dequantize",
                inputs=[f"k_proj_s{step_idx}_h{kv_head}"],
                outputs=[f"k_proj_f_s{step_idx}_h{kv_head}"],
                attrs={"scale": proj_scale, "zero_point": 0},
            )
            b.host(
                f"rope_k_proj_s{step_idx}_h{kv_head}",
                "rope",
                inputs=[f"k_proj_f_s{step_idx}_h{kv_head}"],
                outputs=[f"k_rope_f_s{step_idx}_h{kv_head}"],
                attrs={"head_dim": d_head, "position": token_pos, "theta": rope_theta},
            )
            b.host(
                f"quant_k_rope_s{step_idx}_h{kv_head}",
                "quantize",
                inputs=[f"k_rope_f_s{step_idx}_h{kv_head}"],
                outputs=[f"k_rope_q_s{step_idx}_h{kv_head}"],
                attrs={"scale": proj_scale, "zero_point": 0, "dtype": DType.INT16},
            )
        for kv_head in range(n_kv_heads):
            b.host(
                f"k_scatter_s{step_idx}_h{kv_head}",
                "k_cache_scatter_write",
                inputs=[f"k_rope_q_s{step_idx}_h{kv_head}"],
                outputs=[f"k_cache_h{kv_head}_{step_name}"],
                attrs={"token_index": cache_slot, "k_cache_base": f"k_cache_h{kv_head}"},
            )

        for q_head in range(n_heads):
            b.host(
                f"dequant_q_proj_s{step_idx}_h{q_head}",
                "dequantize",
                inputs=[f"q_proj_s{step_idx}_h{q_head}"],
                outputs=[f"q_proj_f_s{step_idx}_h{q_head}"],
                attrs={"scale": proj_scale, "zero_point": 0},
            )
            b.host(
                f"rope_q_proj_s{step_idx}_h{q_head}",
                "rope",
                inputs=[f"q_proj_f_s{step_idx}_h{q_head}"],
                outputs=[f"q_rope_f_s{step_idx}_h{q_head}"],
                attrs={"head_dim": d_head, "position": token_pos, "theta": rope_theta},
            )
            b.host(
                f"quant_q_rope_s{step_idx}_h{q_head}",
                "quantize",
                inputs=[f"q_rope_f_s{step_idx}_h{q_head}"],
                outputs=[f"q_rope_q_s{step_idx}_h{q_head}"],
                attrs={"scale": proj_scale, "zero_point": 0, "dtype": DType.INT16},
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

            b.intermediate(scores_name, (1, valid_tokens), DType.INT16)
            b.intermediate(probs_f16_name, (1, valid_tokens), DType.INT16)
            b.intermediate(attn_q_name, (1, valid_tokens), DType.INT16, metadata={"storage_role": "A"})
            b.intermediate(attn_name, (1, d_head), DType.INT16)
            b.intermediate(out_part_name, (1, resolved_d_model), DType.INT16)
            if k_view_name not in tensors:
                b.tensor(
                    k_view_name,
                    (d_head, valid_tokens),
                    DType.INT16,
                    TensorKind.INTERMEDIATE,
                    metadata=_make_b_view(k_view_name, f"k_cache_h{kv_head}", (d_head, valid_tokens), cache_kind="K").metadata,
                )
            if v_view_name not in tensors:
                b.tensor(
                    v_view_name,
                    (valid_tokens, d_head),
                    DType.INT16,
                    TensorKind.INTERMEDIATE,
                    metadata=_make_b_view(v_view_name, f"v_cache_h{kv_head}", (valid_tokens, d_head), cache_kind="V").metadata,
                )

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

            seg_cache_score_ops.append(b.matmul(f"op_qk_s{step_idx}_h{q_head}", query_name, k_view_name, scores_name))
            seg_cache_score_outputs.append(scores_name)
            seg_value_ops.append(b.matmul(f"op_av_s{step_idx}_h{q_head}", attn_q_name, v_view_name, attn_name, shift=8))
            seg_value_ops.append(b.matmul(f"op_o_s{step_idx}_h{q_head}", attn_name, f"w_o_h{q_head}", out_part_name))
            seg_value_outputs.append(out_part_name)
            step_out_part_names.append(out_part_name)
            attn_q_inputs.append(attn_q_name)

        b.segment(
            f"seg_cache_score_s{step_idx}",
            ops=seg_cache_score_ops,
            inputs=cache_score_inputs,
            outputs=seg_cache_score_outputs,
        )
        for q_head in range(n_heads):
            scores_name = f"scores_s{step_idx}_h{q_head}"
            probs_f16_name = f"probs_f16_s{step_idx}_h{q_head}"
            attn_q_name = f"attn_q_s{step_idx}_h{q_head}"
            b.host(
                f"softmax_scores_f16_s{step_idx}_h{q_head}",
                "softmax_f16",
                inputs=[scores_name],
                outputs=[probs_f16_name],
                attrs={"axis": -1},
            )
            b.host(
                f"quantize_probs_s{step_idx}_h{q_head}",
                "quantize",
                inputs=[probs_f16_name],
                outputs=[attn_q_name],
                attrs={"scale": attn_scale, "zero_point": 0, "dtype": DType.INT16},
            )

        b.segment(
            f"seg_value_s{step_idx}",
            ops=seg_value_ops,
            inputs=attn_q_inputs,
            outputs=seg_value_outputs,
        )

        out_step_name = f"out_s{step_idx}"
        out_step_kind = TensorKind.OUTPUT if step_idx == (decode_steps - 1) else TensorKind.INTERMEDIATE
        b.output(
            out_step_name,
            (1, resolved_d_model),
            DType.FLOAT32,
            is_final_output=step_idx == (decode_steps - 1),
        )

        if n_heads == 1:
            b.host(
                f"reduce_heads_s{step_idx}_add0",
                "add",
                inputs=[step_out_part_names[0], "reduce_zero"],
                outputs=[out_step_name],
            )
        else:
            first_add_to_out = n_heads == 2
            acc_name = out_step_name if first_add_to_out else f"head_sum_s{step_idx}_1"
            if not first_add_to_out:
                b.intermediate(acc_name, (1, resolved_d_model), DType.FLOAT32)
            b.host(
                f"reduce_heads_s{step_idx}_1",
                "add",
                inputs=[step_out_part_names[0], step_out_part_names[1]],
                outputs=[acc_name],
            )
            for q_head in range(2, n_heads):
                is_last = q_head == (n_heads - 1)
                dst_name = out_step_name if is_last else f"head_sum_s{step_idx}_{q_head}"
                if not is_last:
                    b.intermediate(dst_name, (1, resolved_d_model), DType.FLOAT32)
                b.host(
                    f"reduce_heads_s{step_idx}_{q_head}",
                    "add",
                    inputs=[acc_name, step_out_part_names[q_head]],
                    outputs=[dst_name],
                )
                acc_name = dst_name

        if step_idx == (decode_steps - 1):
            final_out_expected = step_out_expected

    if scores_verify_expected is None:
        raise RuntimeError("internal error: missing score verification tensor")
    if k_rope_verify_name is None or k_rope_verify_expected is None:
        raise RuntimeError("internal error: missing K-RoPE verification tensor")
    collector.verify(k_rope_verify_name, k_rope_verify_name)
    collector.verify(scores_verify_name, scores_verify_name)
    collector.verify(final_out_name, "decode_attention")
    plan = collector.finalize(
        inputs=[],
        outputs=[final_out_name],
        metadata={"frontend": "hand.decode_attention"},
    )
    artifact = compile_plan(
        plan,
        {
            k_rope_verify_name: k_rope_verify_expected,
            scores_verify_name: scores_verify_expected,
            final_out_name: final_out_expected.astype(np.float32),
        },
    )
    return artifact, final_out_expected.astype(np.float32), resolved_d_model


def build_artifact_legacy(
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
    return _build_decode_artifact(
        _LegacyPlanCollector(),
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        token_capacity=token_capacity,
        token_indices=token_indices,
        seed=seed,
        attn_scale=attn_scale,
        rope_theta=rope_theta,
    )


def build_artifact_via_builder(
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
    return _build_decode_artifact(
        IRBuilder(metadata={"construction": "builder"}),
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        token_capacity=token_capacity,
        token_indices=token_indices,
        seed=seed,
        attn_scale=attn_scale,
        rope_theta=rope_theta,
    )


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
    return build_artifact_via_builder(
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_head=d_head,
        token_capacity=token_capacity,
        token_indices=token_indices,
        seed=seed,
        attn_scale=attn_scale,
        rope_theta=rope_theta,
    )
