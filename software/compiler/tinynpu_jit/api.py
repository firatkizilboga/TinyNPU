from __future__ import annotations

from typing import Any

import numpy as np

from .artifact import CompiledArtifact
from .benchmark import CostModel
from .executor import HostEmulationExecutor
from .ir import ExecutionPlan, VerificationMode
from .lowering import SegmentCompiler


def compile_plan(plan: ExecutionPlan, expected_tensors: dict[str, np.ndarray], defines_path: str | None = None) -> CompiledArtifact:
    return SegmentCompiler(defines_path=defines_path).compile(plan, expected_tensors)


def compile_module(module: Any, example_inputs: tuple[Any, ...], **kwargs) -> CompiledArtifact:
    try:
        import torch.fx as fx  # type: ignore
        try:
            from torch.ao.nn.quantized import DeQuantize as QDeQuantize, Quantize as QQuantize  # type: ignore
        except Exception:
            QQuantize = QDeQuantize = ()
        try:
            from torch.ao.nn.quantized import Conv2d as QConv2d, Linear as QLinear  # type: ignore
        except Exception:
            QLinear = QConv2d = ()
        try:
            from torch.ao.quantization import DeQuantStub, QuantStub  # type: ignore
        except Exception:
            try:
                from torch.quantization import DeQuantStub, QuantStub  # type: ignore
            except Exception:
                QuantStub = DeQuantStub = ()
        try:
            from software.compiler.tinynpu_quant import (  # type: ignore
                CompilerDequantize,
                CompilerQuantize,
                CompilerReadyConv2d,
                CompilerReadyLinear,
            )
        except Exception:
            CompilerQuantize = CompilerDequantize = CompilerReadyLinear = CompilerReadyConv2d = ()
    except Exception as exc:
        raise ImportError(
            "torch is required for compile_module(). Install torch to enable the PyTorch frontend."
        ) from exc

    from .partitioner import partition_fx_graph
    from .markers import im2col_for_npu, mark_for_verify, npu_matmul, quantize_for_npu

    class TinyNPUTracer(fx.Tracer):
        def __init__(self):
            super().__init__(autowrap_functions=(mark_for_verify, quantize_for_npu, npu_matmul, im2col_for_npu))

        def is_leaf_module(self, m: Any, module_qualified_name: str) -> bool:
            quant_leaf_types = tuple(
                t
                for t in (
                    QQuantize,
                    QDeQuantize,
                    QuantStub,
                    DeQuantStub,
                    QLinear,
                    QConv2d,
                    CompilerQuantize,
                    CompilerDequantize,
                    CompilerReadyLinear,
                    CompilerReadyConv2d,
                )
                if t
            )
            if quant_leaf_types and isinstance(m, quant_leaf_types):
                return True
            return super().is_leaf_module(m, module_qualified_name)

    tracer = TinyNPUTracer()
    graph = tracer.trace(module)
    graph_module = fx.GraphModule(tracer.root, graph)
    try:
        from torch.fx.passes.shape_prop import ShapeProp  # type: ignore

        quant_leaf_types = tuple(
            t
            for t in (
                QQuantize,
                QDeQuantize,
                QuantStub,
                DeQuantStub,
                QLinear,
                QConv2d,
                CompilerQuantize,
                CompilerDequantize,
                CompilerReadyLinear,
                CompilerReadyConv2d,
            )
            if t
        )
        has_quant_boundaries = any(
            node.op == "call_module"
            and quant_leaf_types
            and isinstance(dict(graph_module.named_modules())[node.target], quant_leaf_types)
            for node in graph_module.graph.nodes
        )
        if not has_quant_boundaries:
            ShapeProp(graph_module).propagate(*example_inputs)
    except Exception:
        pass

    plan, expected = partition_fx_graph(graph_module, example_inputs, **kwargs)
    return compile_plan(plan, expected)


def run_host_emulation(
    artifact: CompiledArtifact,
    inputs: dict[str, np.ndarray],
    verification: VerificationMode = VerificationMode.OFF,
    *,
    debug: bool = False,
    benchmark: bool = False,
    cost_model: CostModel | None = None,
):
    return HostEmulationExecutor().run(
        artifact,
        inputs,
        verification,
        debug=debug,
        benchmark=benchmark,
        cost_model=cost_model,
    )
