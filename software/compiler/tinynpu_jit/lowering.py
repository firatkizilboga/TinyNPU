from __future__ import annotations

from collections import defaultdict

import numpy as np

from tinynpu import TinyNPUProgram

from .artifact import CompiledArtifact, SegmentArtifact
from .ir import (
    DType,
    ExecutionPlan,
    HostOp,
    MatMulOp,
    NpuSegment,
    TensorKind,
    VerifyTensor,
    make_rope_cs_tensor_spec,
    to_precision_mode,
)
from .memory_planner import (
    MemoryPlanEntry,
    SegmentMemoryPlan,
    infer_roles,
    plan_program_memory,
)


def _tensor_use_counts(plan: ExecutionPlan) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for step in plan.steps:
        if isinstance(step, HostOp):
            for name in step.inputs:
                counts[name] += 1
        elif isinstance(step, NpuSegment):
            for op in step.ops:
                counts[op.lhs] += 1
                counts[op.rhs] += 1
                if op.bias:
                    counts[op.bias] += 1
        elif isinstance(step, VerifyTensor):
            counts[step.tensor_name] += 1
    return counts


def _replace_segment_output(segment: NpuSegment, old_name: str, new_name: str) -> None:
    segment.outputs = [new_name if name == old_name else name for name in segment.outputs]


def _can_lower_rope_pattern(
    plan: ExecutionPlan,
    producer_op: MatMulOp,
    source_name: str,
    dequant: HostOp,
    rope: HostOp,
    quant: HostOp,
    use_counts: dict[str, int],
) -> bool:
    if producer_op.rope_cs_name is not None:
        return False
    if producer_op.in_dtype != DType.INT16 or producer_op.out_dtype != DType.INT16:
        return False
    if plan.tensors[source_name].dtype != DType.INT16:
        return False
    if len(dequant.inputs) != 1 or len(dequant.outputs) != 1:
        return False
    if len(rope.inputs) != 1 or len(rope.outputs) != 1:
        return False
    if len(quant.inputs) != 1 or len(quant.outputs) != 1:
        return False
    if dequant.outputs[0] != rope.inputs[0]:
        return False
    if rope.outputs[0] != quant.inputs[0]:
        return False
    if use_counts.get(source_name, 0) != 1:
        return False
    if use_counts.get(dequant.outputs[0], 0) != 1:
        return False
    if use_counts.get(rope.outputs[0], 0) != 1:
        return False
    source_spec = plan.tensors[source_name]
    target_spec = plan.tensors[quant.outputs[0]]
    if source_spec.shape != target_spec.shape or target_spec.dtype != DType.INT16:
        return False

    deq_scale = float(dequant.attrs.get("scale", 1.0))
    quant_scale = float(quant.attrs.get("scale", 1.0))
    deq_zero = int(dequant.attrs.get("zero_point", 0))
    quant_zero = int(quant.attrs.get("zero_point", 0))
    quant_dtype = quant.attrs.get("dtype", DType.INT16)
    if quant_dtype != DType.INT16:
        return False
    if deq_zero != quant_zero:
        return False
    if abs(deq_scale - quant_scale) > 1.0e-12:
        return False

    head_dim = int(rope.attrs.get("head_dim", 0))
    position = rope.attrs.get("position")
    theta = rope.attrs.get("theta")
    if head_dim <= 0 or head_dim % 16 != 0:
        return False
    if position is None or theta is None:
        return False
    if source_spec.shape[-1] != head_dim:
        return False
    return True


def rewrite_host_rope_patterns(plan: ExecutionPlan) -> None:
    producer_by_tensor: dict[str, tuple[NpuSegment, MatMulOp]] = {}
    for step in plan.steps:
        if isinstance(step, NpuSegment):
            for op in step.ops:
                producer_by_tensor[op.out] = (step, op)

    use_counts = _tensor_use_counts(plan)
    new_steps: list[HostOp | NpuSegment | VerifyTensor] = []
    index = 0
    while index < len(plan.steps):
        step = plan.steps[index]
        if (
            index + 2 < len(plan.steps)
            and isinstance(step, HostOp)
            and step.kind == "dequantize"
            and isinstance(plan.steps[index + 1], HostOp)
            and plan.steps[index + 1].kind == "rope"
            and isinstance(plan.steps[index + 2], HostOp)
            and plan.steps[index + 2].kind == "quantize"
        ):
            dequant = step
            rope = plan.steps[index + 1]
            quant = plan.steps[index + 2]
            source_name = dequant.inputs[0]
            producer = producer_by_tensor.get(source_name)
            if producer is not None:
                segment, producer_op = producer
                if _can_lower_rope_pattern(plan, producer_op, source_name, dequant, rope, quant, use_counts):
                    target_name = quant.outputs[0]
                    rope_cs_name = f"{target_name}__rope_cs"
                    if rope_cs_name not in plan.tensors:
                        plan.tensors[rope_cs_name] = make_rope_cs_tensor_spec(
                            rope_cs_name,
                            int(rope.attrs["head_dim"]),
                            int(rope.attrs["position"]),
                            float(rope.attrs["theta"]),
                            kind=TensorKind.CONSTANT,
                        )
                    producer_op.out = target_name
                    producer_op.rope_cs_name = rope_cs_name
                    _replace_segment_output(segment, source_name, target_name)
                    producer_by_tensor.pop(source_name, None)
                    producer_by_tensor[target_name] = (segment, producer_op)
                    index += 3
                    continue
        new_steps.append(step)
        index += 1
    plan.steps = new_steps


def prune_unused_tensors(plan: ExecutionPlan) -> None:
    referenced: set[str] = set(plan.inputs + plan.outputs)
    for step in plan.steps:
        if isinstance(step, HostOp):
            referenced.update(step.inputs)
            referenced.update(step.outputs)
        elif isinstance(step, NpuSegment):
            referenced.update(step.inputs)
            referenced.update(step.outputs)
            for op in step.ops:
                referenced.add(op.lhs)
                referenced.add(op.rhs)
                referenced.add(op.out)
                if op.bias:
                    referenced.add(op.bias)
                if op.rope_cs_name:
                    referenced.add(op.rope_cs_name)
        elif isinstance(step, VerifyTensor):
            referenced.add(step.tensor_name)

    changed = True
    while changed:
        changed = False
        for name in list(referenced):
            spec = plan.tensors.get(name)
            if spec is None:
                continue
            base_name = spec.metadata.get("storage_view_of")
            if base_name and str(base_name) not in referenced:
                referenced.add(str(base_name))
                changed = True

    plan.tensors = {name: spec for name, spec in plan.tensors.items() if name in referenced}


class SegmentCompiler:
    def __init__(self, defines_path: str | None = None):
        self.defines_path = defines_path

    def compile(self, plan: ExecutionPlan, expected_tensors: dict[str, np.ndarray]) -> CompiledArtifact:
        _tmp = TinyNPUProgram(defines_path=self.defines_path)
        array_size = int(_tmp.hw.params.get("ARRAY_SIZE", 8))
        rewrite_host_rope_patterns(plan)
        prune_unused_tensors(plan)
        self._fuse_layout_restore_im2col(plan)
        self._annotate_output_layouts(plan)
        # Read UB capacity from hardware config
        ub_capacity = int(_tmp.hw.params.get("BUFFER_DEPTH", 0))
        if ub_capacity <= 0:
            ub_capacity = int(_tmp.hw.params.get("IM_BASE_ADDR", 0x8000))

        # Global memory plan: static weights get unique addresses, dynamic tensors
        # share a zone with within-segment liveness reuse.
        memory_report = plan_program_memory(plan, ub_capacity)

        # Build per-segment lookup: tensor_name -> MemoryPlanEntry
        addr_maps: dict[str, dict[str, MemoryPlanEntry]] = {}
        seg_plan_map: dict[str, SegmentMemoryPlan] = {}
        for sp in memory_report.segments:
            addr_maps[sp.segment_name] = {e.name: e for e in sp.entries}
            seg_plan_map[sp.segment_name] = sp

        artifacts: dict[str, SegmentArtifact] = {}
        for step in plan.steps:
            if isinstance(step, NpuSegment):
                addr_map = addr_maps.get(step.name, {})
                seg_plan = seg_plan_map.get(step.name)
                artifacts[step.name] = self._compile_npu_segment(plan, step, addr_map, seg_plan)

        return CompiledArtifact(
            plan=plan,
            expected_tensors=expected_tensors,
            segment_artifacts=artifacts,
            metadata={"compiler": "tinynpu_jit", "segment_count": len(artifacts)},
            memory_report=memory_report,
            static_ub_image=memory_report.static_ub_image,
        )

    def _fuse_layout_restore_im2col(self, plan: ExecutionPlan) -> None:
        return

    @staticmethod
    def _cache_kind_for_tensor(plan: ExecutionPlan, tensor_name: str) -> str | None:
        spec = plan.tensors[tensor_name]
        cache_kind = spec.metadata.get("cache_kind")
        if cache_kind is not None:
            return str(cache_kind)
        base_name = spec.metadata.get("storage_view_of")
        if base_name:
            return LoweringPass._cache_kind_for_tensor(plan, str(base_name))
        return None

    def _annotate_output_layouts(self, plan: ExecutionPlan) -> None:
        for step in plan.steps:
            if not isinstance(step, NpuSegment):
                continue
            for index, op in enumerate(step.ops):
                out_spec = plan.tensors[op.out]
                out_cache_kind = self._cache_kind_for_tensor(plan, op.out)
                rhs_cache_kind = self._cache_kind_for_tensor(plan, op.rhs)
                if op.writeback_mode == "normal":
                    if out_spec.metadata.get("storage_view_of") and out_spec.metadata.get("storage_role", "B") == "B":
                        if out_cache_kind == "K" and op.in_dtype == DType.INT16 and op.out_dtype == DType.INT16:
                            if not op.rope_cs_name:
                                # With rope_cs_name the XFORM needs contiguous C-layout output;
                                # skip K_CACHE_APPEND and let the caller scatter separately.
                                op.writeback_mode = "k_cache_append_int16"
                        elif out_cache_kind == "V" and op.in_dtype == DType.INT16 and op.out_dtype == DType.INT16:
                            op.writeback_mode = "v_cache_append_int16"
                        op.output_layout = "b"
                    else:
                        if op.out in step.outputs:
                            op.output_layout = "c"
                        else:
                            later_uses: set[str] = set()
                            for later in step.ops[index + 1 :]:
                                if later.lhs == op.out:
                                    later_uses.add("lhs")
                                if later.rhs == op.out:
                                    later_uses.add("rhs")
                            if later_uses == {"lhs"}:
                                op.output_layout = "a"
                            elif later_uses == {"rhs"}:
                                op.output_layout = "b"
                            else:
                                op.output_layout = "c"
                if op.b_read_mode == "normal" and rhs_cache_kind == "K" and op.in_dtype == DType.INT16:
                    op.b_read_mode = "k_cache_int16"

    @staticmethod
    def _fallback_role(plan: ExecutionPlan, tensor_name: str) -> str:
        spec = plan.tensors[tensor_name]
        if spec.metadata.get("storage_role"):
            return str(spec.metadata["storage_role"])
        for candidate in plan.tensors.values():
            if candidate.metadata.get("storage_view_of") == tensor_name:
                return str(candidate.metadata.get("storage_role", "B"))
        return "C"

    def _compile_npu_segment(
        self,
        plan: ExecutionPlan,
        segment: NpuSegment,
        addr_map: dict[str, MemoryPlanEntry],
        seg_plan: SegmentMemoryPlan | None,
    ) -> SegmentArtifact:
        program = TinyNPUProgram(defines_path=self.defines_path)
        roles = infer_roles(segment)

        referenced = set(segment.inputs + segment.outputs)
        for op in segment.ops:
            referenced.add(op.lhs)
            referenced.add(op.rhs)
            referenced.add(op.out)
            if op.bias:
                referenced.add(op.bias)
            if op.rope_cs_name:
                referenced.add(op.rope_cs_name)
        for name in list(referenced):
            spec = plan.tensors[name]
            base_name = spec.metadata.get("storage_view_of")
            if base_name:
                referenced.add(str(base_name))

        for name in sorted(referenced):
            spec = plan.tensors[name]
            if name in program.symbols:
                continue
            base_name = spec.metadata.get("storage_view_of")
            if base_name:
                base_spec = plan.tensors[str(base_name)]
                base_precision = to_precision_mode(
                    base_spec.dtype if base_spec.dtype in (DType.INT4, DType.INT8, DType.INT16) else DType.INT16
                )
                if str(base_name) not in program.symbols:
                    if base_spec.kind == TensorKind.OUTPUT and base_spec.data is None:
                        base_data = np.zeros(base_spec.shape, dtype=np.int16)
                    elif base_spec.kind == TensorKind.INTERMEDIATE and base_spec.data is None:
                        base_data = np.zeros(base_spec.shape, dtype=np.int16)
                    elif base_spec.kind == TensorKind.INPUT and base_spec.data is None:
                        base_data = np.zeros(base_spec.shape, dtype=np.int16)
                    else:
                        base_data = np.array(base_spec.data if base_spec.data is not None else np.zeros(base_spec.shape), copy=True)
                    program.declare_data(str(base_name), base_data, precision=base_precision, role="B")
                precision = to_precision_mode(spec.dtype if spec.dtype in (DType.INT4, DType.INT8, DType.INT16) else DType.INT16)
                program.declare_b_view(
                    name,
                    str(base_name),
                    spec.shape,
                    precision=precision,
                    word_offset=int(spec.metadata.get("storage_word_offset", 0)),
                )
                continue
            if spec.kind == TensorKind.OUTPUT and spec.data is None:
                data = np.zeros(spec.shape, dtype=np.int16)
            elif spec.kind == TensorKind.INTERMEDIATE and spec.data is None:
                data = np.zeros(spec.shape, dtype=np.int16)
            elif spec.kind == TensorKind.INPUT and spec.data is None:
                data = np.zeros(spec.shape, dtype=np.int16)
            else:
                data = np.array(spec.data if spec.data is not None else np.zeros(spec.shape), copy=True)

            role = roles.get(name, self._fallback_role(plan, name))
            precision = to_precision_mode(spec.dtype if spec.dtype in (DType.INT4, DType.INT8, DType.INT16) else DType.INT16)
            program.declare_data(name, data, precision=precision, role=role)

        for op in segment.ops:
            activation = 0
            if op.activation == "relu":
                activation = 1
            elif op.activation == "sigmoid":
                activation = 2
            elif op.activation == "h_gelu":
                activation = 3
            output_layout = 0
            if op.output_layout == "a":
                output_layout = 1
            elif op.output_layout == "b":
                output_layout = 2
            writeback_mode = 0
            if op.writeback_mode == "v_cache_append_int16":
                writeback_mode = 1
            elif op.writeback_mode == "k_cache_append_int16":
                writeback_mode = 2
            b_read_mode = 0
            if op.b_read_mode == "k_cache_int16":
                b_read_mode = 1
            program.matmul(
                op.lhs,
                op.rhs,
                op.out,
                bias_name=op.bias,
                shift=op.shift,
                multiplier=op.multiplier,
                activation=activation,
                in_precision=to_precision_mode(op.in_dtype),
                out_precision=to_precision_mode(op.out_dtype),
                write_offset=0,
                h_gelu_x_scale_shift=int(op.h_gelu_x_scale_shift),
                output_layout=output_layout,
                writeback_mode=writeback_mode,
                output_word_offset=int(op.output_word_offset),
                b_word_offset=int(op.b_word_offset),
                b_read_mode=b_read_mode,
            )
            if op.rope_cs_name:
                program.xform_rope_k16(op.out, op.rope_cs_name)

        # Pre-assign globally planned addresses before compile() runs so that
        # program.compile() respects the planner layout instead of bump-allocating.
        for name, sym in program.symbols.items():
            if name in addr_map:
                sym.addr = addr_map[name].address

        program.halt()
        binary = program.compile()
        ub_capacity = int(program.hw.params.get("BUFFER_DEPTH", 0))
        if ub_capacity and len(binary["ub"]) > ub_capacity:
            raise MemoryError(
                f"Segment '{segment.name}' requires {len(binary['ub'])} UB words but hardware exposes {ub_capacity}."
            )

        symbol_table = {}
        for name, symbol in program.symbols.items():
            symbol_table[name] = {
                "addr": symbol.addr,
                "shape": tuple(symbol.shape),
                "role": symbol.storage_role,
                "precision": int(symbol.precision),
                "word_count": symbol.word_count,
                "base_name": symbol.base_name,
                "word_offset": int(symbol.word_offset),
            }

        return SegmentArtifact(
            name=segment.name,
            binary=binary,
            symbol_table=symbol_table,
            ub_words=len(binary["ub"]),
            im_words=len(binary["im"]),
            memory_plan=seg_plan,
        )
