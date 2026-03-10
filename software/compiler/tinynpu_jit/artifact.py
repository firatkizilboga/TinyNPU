from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .ir import ExecutionPlan, VerificationMode

if TYPE_CHECKING:
    from .memory_planner import GlobalMemoryReport, SegmentMemoryPlan


@dataclass
class SegmentArtifact:
    name: str
    binary: dict[str, Any]
    symbol_table: dict[str, dict[str, Any]]
    ub_words: int
    im_words: int
    memory_plan: "SegmentMemoryPlan | None" = None


@dataclass
class ExecutionResult:
    tensors: dict[str, np.ndarray]
    verified: list[str] = field(default_factory=list)
    trace_tensors: dict[str, np.ndarray] = field(default_factory=dict)
    vector_captures: dict[str, dict[str, Any]] = field(default_factory=dict)
    debug_trace: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CompiledArtifact:
    plan: ExecutionPlan
    expected_tensors: dict[str, np.ndarray]
    segment_artifacts: dict[str, SegmentArtifact]
    metadata: dict[str, Any] = field(default_factory=dict)
    memory_report: "GlobalMemoryReport | None" = None
    static_ub_image: list[int] | None = None

    def run_host_emulation(
        self,
        inputs: dict[str, np.ndarray],
        verification: VerificationMode = VerificationMode.OFF,
        *,
        debug: bool = False,
    ):
        from .executor import HostEmulationExecutor

        return HostEmulationExecutor().run(self, inputs, verification, debug=debug)

    def run(self, inputs: dict[str, np.ndarray], verification: VerificationMode = VerificationMode.OFF, **kwargs):
        from .runtime import run

        return run(self, inputs, verification=verification, **kwargs)

    def inspect(self, inputs: dict[str, np.ndarray], **kwargs) -> str:
        from .inspect import inspect_artifact

        return inspect_artifact(self, inputs, **kwargs)

    def format_debug_trace(self, execution_result: "ExecutionResult") -> str:
        from .inspect import format_debug_trace

        return format_debug_trace(execution_result)

    def print_memory_report(self) -> str:
        """Return a human-readable memory layout report."""
        if self.memory_report is None:
            return "No memory report available."

        r = self.memory_report
        lines: list[str] = []
        lines.append("=== TinyNPU Memory Report ===")
        lines.append(
            f"Static zone : [0, {r.static_zone_end})  = {r.static_zone_end} words"
            f"  (load once, {len(r.static_ub_image)} words preloaded)"
        )
        lines.append(f"UB peak     : {r.total_ub_peak} words")
        lines.append(f"Theoretical : {r.theoretical_minimum_ub} words minimum")
        if r.cross_segment_tensors:
            lines.append(f"Cross-seg   : {', '.join(r.cross_segment_tensors)}")
        lines.append("")

        for sp in r.segments:
            status = "OK" if sp.is_feasible else "OOM!"
            savings = f", {sp.reused_words} words reused" if sp.reused_words else ""
            lines.append(
                f"  [{status}] Segment '{sp.segment_name}':  "
                f"{sp.total_words} / {sp.ub_capacity} words used{savings}"
            )
            for e in sorted(sp.entries, key=lambda e: e.address):
                reuse = f"  ← reuses '{e.reuses_from}'" if e.reuses_from else ""
                lines.append(
                    f"    [{e.address:5d} .. {e.address + e.word_count - 1:5d}]"
                    f"  {e.name}  ({e.word_count} words){reuse}"
                )

        return "\n".join(lines)
