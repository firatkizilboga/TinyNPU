from __future__ import annotations

from typing import Any

import numpy as np

from .artifact import CompiledArtifact
from .benchmark import CostModel
from .executor import HostEmulationExecutor
from .ir import VerificationMode
from .simulator import run_sim


def run(
    artifact: CompiledArtifact,
    inputs: dict[str, np.ndarray],
    *,
    backend: str = "host-emulation",
    verification: VerificationMode = VerificationMode.OFF,
    debug: bool = False,
    benchmark: bool = False,
    cost_model: CostModel | None = None,
    **backend_kwargs: Any,
):
    if backend == "host-emulation":
        return HostEmulationExecutor().run(
            artifact,
            inputs,
            verification,
            debug=debug,
            benchmark=benchmark,
            cost_model=cost_model,
        )
    if backend == "sim":
        return run_sim(
            artifact,
            inputs,
            verification=verification,
            debug=debug,
            benchmark=benchmark,
            cost_model=cost_model,
            **backend_kwargs,
        )
    raise ValueError(f"Unknown backend '{backend}'.")
