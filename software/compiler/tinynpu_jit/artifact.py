from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

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


@dataclass
class CompiledArtifact:
    plan: ExecutionPlan
    expected_tensors: dict[str, np.ndarray]
    segment_artifacts: dict[str, SegmentArtifact]
    metadata: dict[str, Any] = field(default_factory=dict)

    def run_host_emulation(self, inputs: dict[str, np.ndarray], verification: VerificationMode = VerificationMode.OFF):
        from .executor import HostEmulationExecutor

        return HostEmulationExecutor().run(self, inputs, verification)

    def run(self, inputs: dict[str, np.ndarray], verification: VerificationMode = VerificationMode.OFF, **kwargs):
        from .runtime import run

        return run(self, inputs, verification=verification, **kwargs)

    def inspect(self, inputs: dict[str, np.ndarray], **kwargs) -> str:
        from .inspect import inspect_artifact

        return inspect_artifact(self, inputs, **kwargs)
