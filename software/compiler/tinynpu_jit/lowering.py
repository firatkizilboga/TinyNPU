from __future__ import annotations

from collections import defaultdict

import numpy as np

from tinynpu import TinyNPUProgram
from tinynpu.isa import PrecisionMode

from .artifact import CompiledArtifact, SegmentArtifact
from .ir import DType, ExecutionPlan, MatMulOp, NpuSegment, TensorKind, TensorSpec, to_precision_mode


class SegmentCompiler:
    def __init__(self, defines_path: str | None = None):
        self.defines_path = defines_path

    def compile(self, plan: ExecutionPlan, expected_tensors: dict[str, np.ndarray]) -> CompiledArtifact:
        artifacts: dict[str, SegmentArtifact] = {}
        for step in plan.steps:
            if isinstance(step, NpuSegment):
                artifacts[step.name] = self._compile_npu_segment(plan, step)
        return CompiledArtifact(
            plan=plan,
            expected_tensors=expected_tensors,
            segment_artifacts=artifacts,
            metadata={"compiler": "tinynpu_jit", "segment_count": len(artifacts)},
        )

    def _compile_npu_segment(self, plan: ExecutionPlan, segment: NpuSegment) -> SegmentArtifact:
        program = TinyNPUProgram(defines_path=self.defines_path)
        roles = self._infer_roles(segment)

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
            program.matmul(
                op.lhs,
                op.rhs,
                op.out,
                bias_name=op.bias,
                shift=op.shift,
                multiplier=op.multiplier,
                activation=1 if op.activation == "relu" else 0,
                in_precision=to_precision_mode(op.in_dtype),
                out_precision=to_precision_mode(op.out_dtype),
            )

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
        )

    def _infer_roles(self, segment: NpuSegment) -> dict[str, str]:
        uses: dict[str, set[str]] = defaultdict(set)
        for op in segment.ops:
            uses[op.lhs].add("lhs")
            uses[op.rhs].add("rhs")
            if op.bias:
                uses[op.bias].add("bias")
            uses[op.out].add("out")

        roles = {}
        for name, kinds in uses.items():
            if "bias" in kinds:
                roles[name] = "BIAS"
            elif kinds == {"rhs"}:
                roles[name] = "B"
            elif kinds == {"lhs"}:
                roles[name] = "A"
            elif "out" in kinds:
                roles[name] = "C"
            else:
                roles[name] = "C"
        return roles
