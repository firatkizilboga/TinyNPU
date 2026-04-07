from __future__ import annotations

from collections import Counter
import os

import numpy as np

from tinynpu import TinyNPUProgram

from .artifact import CompiledArtifact, SegmentArtifact
from .ir import DType, ExecutionPlan, HostOp, NpuSegment, TensorKind, VerifyTensor, to_precision_mode
from .memory_planner import (
    MemoryPlanEntry,
    SegmentMemoryPlan,
    infer_roles,
    plan_program_memory,
)


class SegmentCompiler:
    def __init__(self, defines_path: str | None = None, enable_conv_stream: bool | None = None):
        self.defines_path = defines_path
        if enable_conv_stream is None:
            env_value = os.getenv("TINYNPU_ENABLE_CONV_STREAM", "1").strip().lower()
            self.enable_conv_stream = env_value in {"1", "true", "yes", "on"}
        else:
            self.enable_conv_stream = bool(enable_conv_stream)

    def compile(self, plan: ExecutionPlan, expected_tensors: dict[str, np.ndarray]) -> CompiledArtifact:
        _tmp = TinyNPUProgram(defines_path=self.defines_path)
        array_size = int(_tmp.hw.params.get("ARRAY_SIZE", 8))
        self._fuse_layout_restore_im2col(plan)
        if self.enable_conv_stream:
            self._materialize_conv_stream_im2col(plan, array_size=array_size)
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

    @staticmethod
    def _is_matrix_hwc_source_shape(shape: tuple[int, ...], out_h: int, out_w: int, out_c: int) -> bool:
        if len(shape) != 2:
            return False
        return int(shape[0]) == (out_h * out_w) and int(shape[1]) == out_c

    def _fuse_layout_restore_im2col(self, plan: ExecutionPlan) -> None:
        tensor_uses: Counter[str] = Counter()
        for step in plan.steps:
            if isinstance(step, HostOp):
                for name in step.inputs:
                    tensor_uses[name] += 1
            elif isinstance(step, NpuSegment):
                for op in step.ops:
                    tensor_uses[op.lhs] += 1
                    tensor_uses[op.rhs] += 1
                    if op.bias:
                        tensor_uses[op.bias] += 1
            elif isinstance(step, VerifyTensor):
                tensor_uses[step.tensor_name] += 1
        for out_name in plan.outputs:
            tensor_uses[out_name] += 1

        rewritten_steps = []
        i = 0
        while i < len(plan.steps):
            step = plan.steps[i]
            if (
                isinstance(step, HostOp)
                and step.kind == "layout_restore"
                and i + 1 < len(plan.steps)
                and isinstance(plan.steps[i + 1], HostOp)
                and plan.steps[i + 1].kind == "im2col"
            ):
                next_step = plan.steps[i + 1]
                restored_name = step.outputs[0]
                if next_step.inputs and next_step.inputs[0] == restored_name and tensor_uses[restored_name] == 1:
                    out_h = int(step.attrs["out_h"])
                    out_w = int(step.attrs["out_w"])
                    out_channels = int(step.attrs["out_channels"])
                    src_name = step.inputs[0]
                    src_spec = plan.tensors.get(src_name)
                    if src_spec and self._is_matrix_hwc_source_shape(src_spec.shape, out_h, out_w, out_channels):
                        fused_attrs = dict(next_step.attrs)
                        fused_attrs["input_layout"] = "matrix_hwc"
                        fused_attrs["matrix_h"] = out_h
                        fused_attrs["matrix_w"] = out_w
                        fused_attrs["matrix_c"] = out_channels
                        next_step.inputs = [src_name]
                        next_step.attrs = fused_attrs
                        rewritten_steps.append(next_step)
                        i += 2
                        continue
            rewritten_steps.append(step)
            i += 1

        plan.steps = rewritten_steps

    def _materialize_conv_stream_im2col(self, plan: ExecutionPlan, *, array_size: int) -> None:
        def physical_per_word(dtype: DType) -> int:
            if dtype == DType.INT4:
                return 4
            if dtype == DType.INT8:
                return 2
            return 1

        tensor_uses: Counter[str] = Counter()
        for step in plan.steps:
            if isinstance(step, HostOp):
                for name in step.inputs:
                    tensor_uses[name] += 1
            elif isinstance(step, NpuSegment):
                for op in step.ops:
                    tensor_uses[op.lhs] += 1
                    tensor_uses[op.rhs] += 1
                    if op.bias:
                        tensor_uses[op.bias] += 1

        rewritten_steps = []
        i = 0
        while i < len(plan.steps):
            step = plan.steps[i]
            if (
                isinstance(step, HostOp)
                and step.kind == "im2col"
                and str(step.attrs.get("input_layout")) == "matrix_hwc"
                and i + 1 < len(plan.steps)
                and isinstance(plan.steps[i + 1], NpuSegment)
            ):
                seg = plan.steps[i + 1]
                if seg.ops:
                    op = seg.ops[0]
                    cols_name = step.outputs[0]
                    src_name = step.inputs[0]
                    if (
                        op.lhs == cols_name
                        and tensor_uses[cols_name] == 1
                        and src_name in plan.tensors
                    ):
                        kernel = int(step.attrs.get("kernel_size", 0))
                        stride = int(step.attrs.get("stride", 1))
                        padding = int(step.attrs.get("padding", 0))
                        in_h = int(step.attrs.get("matrix_h", 0))
                        in_w = int(step.attrs.get("matrix_w", 0))
                        in_c = int(step.attrs.get("matrix_c", 0))
                        src_shape = tuple(int(dim) for dim in plan.tensors[src_name].shape)
                        source_is_matrix = len(src_shape) == 2 and src_shape[0] == (in_h * in_w) and src_shape[1] == in_c
                        if (
                            source_is_matrix
                            and kernel > 0
                            and stride > 0
                        ):
                            p_in = physical_per_word(op.in_dtype)
                            c_phys = (in_c + p_in - 1) // p_in
                            if c_phys > 0:
                                op.lhs = src_name
                                op.conv_stream = {
                                    "input_h": in_h,
                                    "input_w": in_w,
                                    "input_c": in_c,
                                    "kernel_size": kernel,
                                    "stride": stride,
                                    "padding": padding,
                                }
                                seg.inputs = sorted({name for name in seg.inputs if name != cols_name} | {src_name})
                                rewritten_steps.append(seg)
                                i += 2
                                continue

            rewritten_steps.append(step)
            i += 1

        plan.steps = rewritten_steps

    def _annotate_output_layouts(self, plan: ExecutionPlan) -> None:
        for step in plan.steps:
            if not isinstance(step, NpuSegment):
                continue
            for index, op in enumerate(step.ops):
                if op.out in step.outputs:
                    op.output_layout = "c"
                    continue
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
                conv_stream=op.conv_stream,
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
