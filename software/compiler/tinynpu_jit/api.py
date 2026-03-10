from __future__ import annotations

from typing import Any

import numpy as np

from .artifact import CompiledArtifact
from .executor import HostEmulationExecutor
from .ir import ExecutionPlan, VerificationMode
from .lowering import SegmentCompiler


def compile_plan(plan: ExecutionPlan, expected_tensors: dict[str, np.ndarray], defines_path: str | None = None) -> CompiledArtifact:
    return SegmentCompiler(defines_path=defines_path).compile(plan, expected_tensors)


def compile_module(module: Any, example_inputs: tuple[Any, ...], **kwargs) -> CompiledArtifact:
    try:
        import torch.fx as fx  # type: ignore
    except Exception as exc:
        raise ImportError(
            "torch is required for compile_module(). Install torch to enable the PyTorch frontend."
        ) from exc

    from .partitioner import partition_fx_graph
    from .markers import im2col_for_npu, mark_for_verify, npu_matmul, quantize_for_npu

    class TinyNPUTracer(fx.Tracer):
        def __init__(self):
            super().__init__(autowrap_functions=(mark_for_verify, quantize_for_npu, npu_matmul, im2col_for_npu))

    tracer = TinyNPUTracer()
    graph = tracer.trace(module)
    graph_module = fx.GraphModule(tracer.root, graph)
    try:
        from torch.fx.passes.shape_prop import ShapeProp  # type: ignore

        ShapeProp(graph_module).propagate(*example_inputs)
    except Exception:
        pass

    plan, expected = partition_fx_graph(graph_module, example_inputs, **kwargs)
    return compile_plan(plan, expected)


def run_host_emulation(artifact: CompiledArtifact, inputs: dict[str, np.ndarray], verification: VerificationMode = VerificationMode.OFF):
    return HostEmulationExecutor().run(artifact, inputs, verification)
