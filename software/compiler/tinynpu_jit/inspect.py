from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tinynpu import TinyNPUProgram
from tinynpu.isa import PrecisionMode

from .artifact import CompiledArtifact
from .executor import HostEmulationExecutor
from .ir import HostOp, NpuSegment, TensorKind, VerifyTensor


@dataclass
class TensorVectorView:
    tensor_name: str
    segment_name: str
    role: str
    precision: str
    addr: int
    logical_preview: np.ndarray
    expected_vector_rows: list[list[str]]
    actual_vector_rows: list[list[str]] | None = None
    rows_match: bool | None = None


def inspect_artifact(
    artifact: CompiledArtifact,
    inputs: dict[str, np.ndarray],
    *,
    execution_result: Any | None = None,
    preview_rows: int = 4,
    preview_cols: int = 4,
    vector_rows: int = 8,
) -> str:
    values = (
        execution_result.trace_tensors
        if execution_result is not None and execution_result.trace_tensors
        else _materialize_values(artifact, inputs)
    )
    inspector = _ArtifactInspector()

    lines: list[str] = []
    lines.append("Execution Plan")
    lines.append(f"  Inputs: {artifact.plan.inputs}")
    lines.append(f"  Outputs: {artifact.plan.outputs}")
    lines.append(f"  Steps: {len(artifact.plan.steps)}")
    lines.append("")

    for index, step in enumerate(artifact.plan.steps):
        if isinstance(step, NpuSegment):
            lines.append(f"[{index}] NpuSegment {step.name}")
            lines.append(f"  Inputs: {step.inputs}")
            lines.append(f"  Outputs: {step.outputs}")
            for output_name in step.outputs:
                symbol = artifact.segment_artifacts[step.name].symbol_table[output_name]
                view = inspector.tensor_vector_view(
                    segment_name=step.name,
                    tensor_name=output_name,
                    value=values[output_name],
                    symbol=symbol,
                    actual_capture=(execution_result.vector_captures.get(output_name) if execution_result is not None else None),
                    preview_rows=preview_rows,
                    preview_cols=preview_cols,
                    vector_rows=vector_rows,
                )
                lines.extend(_format_vector_view(view))
            lines.append("")
            continue
        if isinstance(step, HostOp):
            lines.append(f"[{index}] HostOp {step.kind} {step.name}")
            lines.append(f"  Inputs: {step.inputs}")
            lines.append(f"  Outputs: {step.outputs}")
            for output_name in step.outputs:
                preview = _preview_array(values[output_name], preview_rows, preview_cols)
                lines.append(f"  {output_name} logical preview:")
                lines.extend(_indent(_format_array(preview), "    "))
            lines.append("")
            continue
        if isinstance(step, VerifyTensor):
            lines.append(f"[{index}] VerifyTensor {step.label} ({step.tensor_name})")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _materialize_values(artifact: CompiledArtifact, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    executor = HostEmulationExecutor()
    values: dict[str, np.ndarray] = {}
    for name, spec in artifact.plan.tensors.items():
        if spec.kind == TensorKind.CONSTANT and spec.data is not None:
            values[name] = np.array(spec.data, copy=True)

    for name in artifact.plan.inputs:
        if name not in inputs:
            raise KeyError(f"Missing runtime input '{name}'.")
        values[name] = np.array(inputs[name], copy=True)

    for step in artifact.plan.steps:
        if isinstance(step, NpuSegment):
            executor._run_npu_segment(step, values)
        elif isinstance(step, HostOp):
            executor._run_host_op(step, values)

    return values


class _ArtifactInspector:
    def __init__(self):
        program = TinyNPUProgram()
        self.array_size = program.array_size
        self.packer = program.packer

    def tensor_vector_view(
        self,
        *,
        segment_name: str,
        tensor_name: str,
        value: np.ndarray,
        symbol: dict[str, Any],
        actual_capture: dict[str, Any] | None,
        preview_rows: int,
        preview_cols: int,
        vector_rows: int,
    ) -> TensorVectorView:
        role = symbol["role"]
        precision = PrecisionMode(symbol["precision"])
        packed = self._pack_tensor(np.array(value), role, precision)
        expected_rows = self._first_vector_rows(packed, vector_rows)
        actual_rows = None
        rows_match = None
        if actual_capture is not None:
            actual_rows = self._format_actual_rows(actual_capture["rows"], vector_rows)
            rows_match = expected_rows == actual_rows
        return TensorVectorView(
            tensor_name=tensor_name,
            segment_name=segment_name,
            role=role,
            precision=precision.name,
            addr=int(symbol["addr"]),
            logical_preview=_preview_array(np.array(value), preview_rows, preview_cols),
            expected_vector_rows=expected_rows,
            actual_vector_rows=actual_rows,
            rows_match=rows_match,
        )

    def _pack_tensor(self, value: np.ndarray, role: str, precision: PrecisionMode) -> list[int]:
        m_tiles, k_tiles, n_tiles = self._tile_counts(tuple(value.shape), role, precision)
        packed = self.packer.pack(value, role, precision, m_tiles, k_tiles, n_tiles)
        return [int(word) for word in packed]

    def _tile_counts(self, shape: tuple[int, int], role: str, precision: PrecisionMode) -> tuple[int, int, int]:
        p = 1 << (2 - precision)
        sz = self.array_size
        m = (shape[0] + sz - 1) // sz
        if role == "A":
            k = (shape[1] // p + sz - 1) // sz
            return m, k, 1
        if role == "B":
            k = (shape[0] // p + sz - 1) // sz
            n = (shape[1] + sz - 1) // sz
            return 1, k, n
        if role == "BIAS":
            return 1, 1, (shape[1] + sz - 1) // sz
        return m, 1, (shape[1] + sz - 1) // sz

    def _first_vector_rows(self, packed_words: list[int], row_count: int) -> list[list[str]]:
        rows: list[list[str]] = []
        for word in packed_words[:row_count]:
            row = []
            for lane in range(self.array_size):
                lane_word = (word >> (lane * 16)) & 0xFFFF
                row.append(f"0x{lane_word:04x}")
            rows.append(row)
        return rows

    def _format_actual_rows(self, rows: list[list[int]], row_count: int) -> list[list[str]]:
        return [[f"0x{lane & 0xFFFF:04x}" for lane in row] for row in rows[:row_count]]


def _preview_array(value: np.ndarray, rows: int, cols: int) -> np.ndarray:
    if value.ndim == 1:
        return value[:rows]
    return value[:rows, :cols]


def _format_vector_view(view: TensorVectorView) -> list[str]:
    lines = [
        f"  Tensor {view.tensor_name}",
        f"    addr={view.addr} role={view.role} precision={view.precision}",
        "    logical preview:",
    ]
    lines.extend(_indent(_format_array(view.logical_preview), "      "))
    lines.append("    expected packed vectors:")
    for idx, row in enumerate(view.expected_vector_rows):
        lines.append(f"      row {idx:02d}: {row}")
    if view.actual_vector_rows is not None:
        lines.append(f"    actual packed vectors: match={view.rows_match}")
        for idx, row in enumerate(view.actual_vector_rows):
            lines.append(f"      row {idx:02d}: {row}")
    return lines


def _format_array(value: np.ndarray) -> list[str]:
    return np.array2string(value, separator=", ").splitlines()


def _indent(lines: list[str], prefix: str) -> list[str]:
    return [prefix + line for line in lines]
