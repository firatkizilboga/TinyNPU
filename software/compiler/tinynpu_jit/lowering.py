from __future__ import annotations

import numpy as np

from tinynpu import TinyNPUProgram
from tinynpu.isa import ActivationMode

from .artifact import CompiledArtifact, SegmentArtifact
from .ir import DType, ExecutionPlan, NpuSegment, TensorKind, to_precision_mode
from .memory_planner import (
    MemoryPlanEntry,
    SegmentMemoryPlan,
    infer_roles,
    plan_program_memory,
)


class SegmentCompiler:
    def __init__(self, defines_path: str | None = None):
        self.defines_path = defines_path

    def compile(self, plan: ExecutionPlan, expected_tensors: dict[str, np.ndarray]) -> CompiledArtifact:
        # Read UB capacity from hardware config
        _tmp = TinyNPUProgram(defines_path=self.defines_path)
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

        for name in sorted(referenced):
            spec = plan.tensors[name]
            if spec.kind == TensorKind.OUTPUT and spec.data is None:
                data = np.zeros(spec.shape, dtype=np.int16)
            elif spec.kind == TensorKind.INTERMEDIATE and spec.data is None:
                data = np.zeros(spec.shape, dtype=np.int16)
            elif spec.kind == TensorKind.INPUT and spec.data is None:
                data = np.zeros(spec.shape, dtype=np.int16)
            else:
                data = np.array(spec.data if spec.data is not None else np.zeros(spec.shape), copy=True)

            role = roles[name]
            precision = to_precision_mode(spec.dtype if spec.dtype in (DType.INT4, DType.INT8, DType.INT16) else DType.INT16)
            program.declare_data(name, data, precision=precision, role=role)

        for op in segment.ops:
            activation = ActivationMode.NONE
            if op.activation == "relu":
                activation = ActivationMode.RELU
            elif op.activation == "sigmoid":
                activation = ActivationMode.SIGMOID
            elif op.activation == "h_gelu":
                activation = ActivationMode.H_GELU
            program.matmul(
                op.lhs,
                op.rhs,
                op.out,
                bias_name=op.bias,
                shift=op.shift,
                multiplier=op.multiplier,
                activation=int(activation),
                in_precision=to_precision_mode(op.in_dtype),
                out_precision=to_precision_mode(op.out_dtype),
                h_gelu_x_scale_shift=int(op.h_gelu_x_scale_shift),
            )

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
            }

        return SegmentArtifact(
            name=segment.name,
            binary=binary,
            symbol_table=symbol_table,
            ub_words=len(binary["ub"]),
            im_words=len(binary["im"]),
            memory_plan=seg_plan,
        )
