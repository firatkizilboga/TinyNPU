from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .benchmark import BenchmarkReport, CostModel
from .ir import ExecutionPlan, VerificationMode


@dataclass
class SegmentArtifact:
    name: str
    binary: dict[str, Any]
    symbol_table: dict[str, dict[str, Any]]
    ub_words: int
    im_words: int


@dataclass
class ExecutionResult:
    tensors: dict[str, np.ndarray]
    verified: list[str] = field(default_factory=list)
    trace_tensors: dict[str, np.ndarray] = field(default_factory=dict)
    vector_captures: dict[str, dict[str, Any]] = field(default_factory=dict)
    debug_trace: list[dict[str, Any]] = field(default_factory=list)
    benchmark: BenchmarkReport | None = None


@dataclass
class CompiledArtifact:
    plan: ExecutionPlan
    expected_tensors: dict[str, np.ndarray]
    segment_artifacts: dict[str, SegmentArtifact]
    metadata: dict[str, Any] = field(default_factory=dict)

    def run_host_emulation(
        self,
        inputs: dict[str, np.ndarray],
        verification: VerificationMode = VerificationMode.OFF,
        *,
        debug: bool = False,
        benchmark: bool = False,
        cost_model: CostModel | None = None,
    ):
        from .executor import HostEmulationExecutor

        return HostEmulationExecutor().run(self, inputs, verification, debug=debug, benchmark=benchmark, cost_model=cost_model)

    def run(self, inputs: dict[str, np.ndarray], verification: VerificationMode = VerificationMode.OFF, **kwargs):
        from .runtime import run

        return run(self, inputs, verification=verification, **kwargs)

    def inspect(self, inputs: dict[str, np.ndarray], **kwargs) -> str:
        from .inspect import inspect_artifact

        return inspect_artifact(self, inputs, **kwargs)

    def format_debug_trace(self, execution_result: "ExecutionResult") -> str:
        from .inspect import format_debug_trace

        return format_debug_trace(execution_result)

    def format_benchmark_report(self, execution_result: "ExecutionResult") -> str:
        from .inspect import format_benchmark_report

        return format_benchmark_report(execution_result)

    def format_benchmark_comparison(self, execution_result: "ExecutionResult", cost_models: list[object]) -> str:
        from .inspect import format_benchmark_comparison

        return format_benchmark_comparison(execution_result, cost_models)
