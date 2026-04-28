from __future__ import annotations

from dataclasses import dataclass, field

from .ir import DType
from .semantic_ir import (
    AdaptiveAvgPool2dOp,
    ActivationOp,
    AvgPool2dOp,
    BinaryOp,
    Conv2dOp,
    CompilerReadyConv2dOp,
    CompilerReadyLinearOp,
    DequantizeOp,
    LinearOp,
    MaxPool2dOp,
    MeanOp,
    QuantizeOp,
    ReshapeOp,
    SemanticGraph,
    VerifyOp,
)


@dataclass(frozen=True)
class CapabilityIssue:
    op_name: str
    op_kind: str
    reason: str


@dataclass
class SemanticCapabilityReport:
    assignments: dict[str, str] = field(default_factory=dict)
    issues: list[CapabilityIssue] = field(default_factory=list)

    @property
    def is_supported(self) -> bool:
        return not self.issues


_NPU_DTYPES = {DType.INT4, DType.INT8, DType.INT16}
_NPU_FUSED_ACTIVATIONS = {"none", "relu", "sigmoid", "gelu"}


def analyze_semantic_capabilities(graph: SemanticGraph) -> SemanticCapabilityReport:
    report = SemanticCapabilityReport()
    producer_by_output: dict[str, object] = {}

    for op in graph.ops:
        if isinstance(op, VerifyOp):
            continue
        for output_name in op.outputs:
            producer_by_output[output_name] = op

    for op in graph.ops:
        if isinstance(op, QuantizeOp):
            report.assignments[op.name] = "host"
            continue
        if isinstance(op, DequantizeOp):
            report.assignments[op.name] = "host"
            continue
        if isinstance(op, ReshapeOp):
            report.assignments[op.name] = "host"
            continue
        if isinstance(op, MeanOp):
            report.assignments[op.name] = "host"
            continue
        if isinstance(op, (MaxPool2dOp, AvgPool2dOp, AdaptiveAvgPool2dOp)):
            report.assignments[op.name] = "host"
            continue
        if isinstance(op, BinaryOp):
            input_dtypes = [graph.values[name].dtype for name in op.inputs]
            if any(dtype != DType.FLOAT32 for dtype in input_dtypes):
                report.issues.append(
                    CapabilityIssue(
                        op.name,
                        "BinaryOp",
                        f"host binary op {op.kind!r} currently requires FLOAT32 inputs, got {[dtype.value for dtype in input_dtypes]}",
                    )
                )
                continue
            report.assignments[op.name] = "host"
            continue
        if isinstance(op, VerifyOp):
            report.assignments[op.name] = "host"
            continue
        if isinstance(op, LinearOp):
            report.assignments[op.name] = "host"
            continue
        if isinstance(op, Conv2dOp):
            report.assignments[op.name] = "host"
            continue
        if isinstance(op, CompilerReadyLinearOp):
            if op.in_dtype not in _NPU_DTYPES or op.out_dtype not in _NPU_DTYPES:
                report.issues.append(
                    CapabilityIssue(op.name, "CompilerReadyLinear", f"unsupported NPU dtypes {op.in_dtype.value}->{op.out_dtype.value}")
                )
                continue
            if op.activation not in _NPU_FUSED_ACTIVATIONS:
                report.issues.append(
                    CapabilityIssue(op.name, "CompilerReadyLinear", f"unsupported fused activation {op.activation!r}")
                )
                continue
            report.assignments[op.name] = "npu"
            continue
        if isinstance(op, CompilerReadyConv2dOp):
            if op.in_dtype not in _NPU_DTYPES or op.out_dtype not in _NPU_DTYPES:
                report.issues.append(
                    CapabilityIssue(op.name, "CompilerReadyConv2d", f"unsupported NPU dtypes {op.in_dtype.value}->{op.out_dtype.value}")
                )
                continue
            if op.activation not in _NPU_FUSED_ACTIVATIONS:
                report.issues.append(
                    CapabilityIssue(op.name, "CompilerReadyConv2d", f"unsupported fused activation {op.activation!r}")
                )
                continue
            report.assignments[op.name] = "npu"
            continue
        if isinstance(op, ActivationOp):
            if op.kind not in {"relu", "sigmoid", "gelu"}:
                report.issues.append(CapabilityIssue(op.name, "Activation", f"unsupported activation {op.kind!r}"))
                continue
            producer = producer_by_output.get(op.inputs[0])
            if not isinstance(producer, (CompilerReadyLinearOp, CompilerReadyConv2dOp)):
                input_dtype = graph.values[op.inputs[0]].dtype
                if input_dtype != DType.FLOAT32:
                    report.issues.append(
                        CapabilityIssue(
                            op.name,
                            "Activation",
                            f"non-fuseable activation {op.kind!r} requires FLOAT32 input, got {input_dtype.value}",
                        )
                    )
                    continue
                report.assignments[op.name] = "host"
                continue
            report.assignments[op.name] = "npu_fused_post_op"
            continue

        report.issues.append(CapabilityIssue(op.name, type(op).__name__, "unsupported semantic op"))

    return report
