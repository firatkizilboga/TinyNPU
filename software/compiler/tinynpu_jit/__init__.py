from .api import compile_module, compile_plan, run_host_emulation
from .artifact import CompiledArtifact, ExecutionResult, SegmentArtifact
from .baremetal_emit import emit_cv32e40p_c, write_cv32e40p_c
from .baremetal_emit_v2 import emit_cv32e40p_program_v2, write_cv32e40p_program_v2
from .benchmark import (
    BenchmarkEntry,
    BenchmarkReport,
    CostModel,
    PrimitiveCounts,
    five_stage_in_order_model,
    ideal_issue_1_model,
    unpipelined_scalar_model,
)
from .host_ops import HostOpSpec, get_host_op_spec, register_host_op, registered_host_op_kinds
from .inspect import inspect_artifact
from .ir import (
    DType,
    ExecutionPlan,
    HostOp,
    MatMulOp,
    NpuSegment,
    TensorKind,
    TensorSpec,
    VerificationMode,
    VerifyTensor,
    b_slot_word_stride,
    make_b_cache_specs,
    make_b_cache_view_spec,
    make_kv_cache_specs,
)
from .markers import im2col_for_npu, mark_for_verify, npu_matmul, quantize_for_npu
from .memory_planner import GlobalMemoryReport, SegmentMemoryPlan, plan_program_memory, plan_segment_memory
from .runtime import run
from .simulator import SimulatorExecutor, run_sim

__all__ = [
    "CompiledArtifact",
    "BenchmarkEntry",
    "b_slot_word_stride",
    "BenchmarkReport",
    "CostModel",
    "DType",
    "ExecutionPlan",
    "ExecutionResult",
    "GlobalMemoryReport",
    "HostOp",
    "HostOpSpec",
    "im2col_for_npu",
    "MatMulOp",
    "make_b_cache_specs",
    "make_b_cache_view_spec",
    "make_kv_cache_specs",
    "NpuSegment",
    "PrimitiveCounts",
    "five_stage_in_order_model",
    "ideal_issue_1_model",
    "unpipelined_scalar_model",
    "SegmentArtifact",
    "SegmentMemoryPlan",
    "SimulatorExecutor",
    "inspect_artifact",
    "TensorKind",
    "TensorSpec",
    "VerificationMode",
    "VerifyTensor",
    "compile_module",
    "compile_plan",
    "emit_cv32e40p_c",
    "emit_cv32e40p_program_v2",
    "get_host_op_spec",
    "mark_for_verify",
    "npu_matmul",
    "plan_program_memory",
    "plan_segment_memory",
    "quantize_for_npu",
    "register_host_op",
    "registered_host_op_kinds",
    "run",
    "run_sim",
    "run_host_emulation",
    "write_cv32e40p_c",
    "write_cv32e40p_program_v2",
]
