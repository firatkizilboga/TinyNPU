from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from .host_ops import benchmark_host_op
from .ir import DType, HostOp, MatMulOp, NpuSegment, TensorSpec


@dataclass
class PrimitiveCounts:
    reads: int = 0
    writes: int = 0
    adds: int = 0
    muls: int = 0
    branches: int = 0
    shifts: int = 0
    clamps: int = 0
    bitops: int = 0
    divs: int = 0
    nonlinear: int = 0
    if_writes: int = 0
    if_reads: int = 0

    def __iadd__(self, other: "PrimitiveCounts") -> "PrimitiveCounts":
        for field_name in self.__dataclass_fields__:
            setattr(self, field_name, getattr(self, field_name) + getattr(other, field_name))
        return self

    def __add__(self, other: "PrimitiveCounts") -> "PrimitiveCounts":
        result = PrimitiveCounts()
        result += self
        result += other
        return result

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class CostModel:
    # CostModel is an instruction-class CPI model over the counted reference
    # machine events. The benchmark first counts "what work happened", then
    # applies a chosen CPU comparison model to those counts. Different models
    # therefore reuse the exact same trace and only change CPI assumptions.
    name: str = "ideal_issue_1"
    read_cost: int = 1
    write_cost: int = 1
    add_cost: int = 1
    mul_cost: int = 1
    branch_cost: int = 1
    shift_cost: int = 1
    clamp_cost: int = 1
    bitop_cost: int = 1
    div_cost: int = 1
    nonlinear_cost: int = 1
    interface_write_cost: int = 1
    interface_read_cost: int = 1

    def estimate_cycles(self, counts: PrimitiveCounts) -> int:
        return (
            counts.reads * self.read_cost
            + counts.writes * self.write_cost
            + counts.adds * self.add_cost
            + counts.muls * self.mul_cost
            + counts.branches * self.branch_cost
            + counts.shifts * self.shift_cost
            + counts.clamps * self.clamp_cost
            + counts.bitops * self.bitop_cost
            + counts.divs * self.div_cost
            + counts.nonlinear * self.nonlinear_cost
            + counts.if_writes * self.interface_write_cost
            + counts.if_reads * self.interface_read_cost
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def unpipelined_scalar_model() -> CostModel:
    # Unpipelined scalar reference:
    # each instruction occupies the machine for its whole work duration.
    # This is the slowest CPU baseline and is useful as a simple "no overlap"
    # reference against the accelerator.
    return CostModel(
        name="unpipelined_scalar",
        read_cost=2,
        write_cost=2,
        add_cost=1,
        mul_cost=4,
        branch_cost=1,
        shift_cost=1,
        clamp_cost=1,
        bitop_cost=1,
        div_cost=16,
        nonlinear_cost=8,
        interface_write_cost=1,
        interface_read_cost=1,
    )


def ideal_issue_1_model() -> CostModel:
    # Ideal issue-1 upper bound:
    # every counted instruction-like event retires at CPI=1 with no hazards.
    # This is not an "unpipelined scalar" model; it is a best-case issue-1 CPU.
    return CostModel(name="ideal_issue_1")


def five_stage_in_order_model() -> CostModel:
    # Practical 5-stage RISC-V style in-order model:
    # - loads/stores/ALU/shift/clamp/bitops stay at CPI=1
    # - branches pay a small control-hazard penalty on average
    # - integer multiply is multi-cycle
    # - divide / nonlinear helpers remain expensive host work
    return CostModel(
        name="five_stage_in_order",
        read_cost=1,
        write_cost=1,
        add_cost=1,
        mul_cost=3,
        branch_cost=2,
        shift_cost=1,
        clamp_cost=1,
        bitop_cost=1,
        div_cost=16,
        nonlinear_cost=8,
        interface_write_cost=1,
        interface_read_cost=1,
    )


@dataclass
class BenchmarkEntry:
    step: str
    bucket: str
    counts: PrimitiveCounts = field(default_factory=PrimitiveCounts)
    cycle_override: int | None = None
    attrs: dict[str, Any] = field(default_factory=dict)

    def resolved_cycles(self, cost_model: CostModel) -> int:
        if self.cycle_override is not None:
            return int(self.cycle_override)
        return cost_model.estimate_cycles(self.counts)

    def to_dict(self, cost_model: CostModel) -> dict[str, Any]:
        return {
            "step": self.step,
            "bucket": self.bucket,
            "counts": self.counts.to_dict(),
            "cycles": self.resolved_cycles(cost_model),
            "attrs": dict(self.attrs),
        }


@dataclass
class BenchmarkReport:
    cost_model: CostModel = field(default_factory=ideal_issue_1_model)
    entries: list[BenchmarkEntry] = field(default_factory=list)

    def add_entry(
        self,
        *,
        step: str,
        bucket: str,
        counts: PrimitiveCounts | None = None,
        cycles: int | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        actual_counts = counts or PrimitiveCounts()
        self.entries.append(
            BenchmarkEntry(
                step=step,
                bucket=bucket,
                counts=actual_counts,
                cycle_override=None if cycles is None else int(cycles),
                attrs=dict(attrs or {}),
            )
        )

    def total_counts(self, bucket: str) -> PrimitiveCounts:
        total = PrimitiveCounts()
        for entry in self.entries:
            if entry.bucket == bucket:
                total += entry.counts
        return total

    def total_cycles(self, bucket: str, cost_model: CostModel | None = None) -> int:
        model = cost_model or self.cost_model
        return sum(entry.resolved_cycles(model) for entry in self.entries if entry.bucket == bucket)

    @property
    def cpu_replaced_cycles(self) -> int:
        return self.total_cycles("cpu_replaced")

    @property
    def npu_compute_cycles(self) -> int:
        return self.total_cycles("npu_compute")

    @property
    def npu_overhead_cycles(self) -> int:
        return self.total_cycles("npu_overhead")

    @property
    def host_intrinsic_cycles(self) -> int:
        return self.total_cycles("host_intrinsic")

    @property
    def pure_acceleration_speedup(self) -> float | None:
        if self.npu_compute_cycles <= 0:
            return None
        return self.cpu_replaced_cycles / self.npu_compute_cycles

    @property
    def integration_adjusted_speedup(self) -> float | None:
        denominator = self.npu_compute_cycles + self.npu_overhead_cycles
        if denominator <= 0:
            return None
        return self.cpu_replaced_cycles / denominator

    def to_dict(self) -> dict[str, Any]:
        return self.to_dict_for_model(self.cost_model)

    def to_dict_for_model(self, cost_model: CostModel) -> dict[str, Any]:
        return {
            "cost_model": cost_model.to_dict(),
            "totals": {
                "cpu_replaced_counts": self.total_counts("cpu_replaced").to_dict(),
                "cpu_replaced_cycles": self.total_cycles("cpu_replaced", cost_model),
                "npu_compute_cycles": self.total_cycles("npu_compute", cost_model),
                "npu_overhead_counts": self.total_counts("npu_overhead").to_dict(),
                "npu_overhead_cycles": self.total_cycles("npu_overhead", cost_model),
                "host_intrinsic_counts": self.total_counts("host_intrinsic").to_dict(),
                "host_intrinsic_cycles": self.total_cycles("host_intrinsic", cost_model),
                "pure_acceleration_speedup": self._pure_acceleration_speedup(cost_model),
                "integration_adjusted_speedup": self._integration_adjusted_speedup(cost_model),
            },
            "entries": [entry.to_dict(cost_model) for entry in self.entries],
        }

    def model_comparison(self, cost_models: list[CostModel]) -> list[dict[str, Any]]:
        comparisons: list[dict[str, Any]] = []
        for model in cost_models:
            payload = self.to_dict_for_model(model)
            comparisons.append(
                {
                    "name": model.name,
                    **payload["totals"],
                }
            )
        return comparisons

    def _pure_acceleration_speedup(self, cost_model: CostModel) -> float | None:
        npu_compute_cycles = self.total_cycles("npu_compute", cost_model)
        if npu_compute_cycles <= 0:
            return None
        return self.total_cycles("cpu_replaced", cost_model) / npu_compute_cycles

    def _integration_adjusted_speedup(self, cost_model: CostModel) -> float | None:
        denominator = self.total_cycles("npu_compute", cost_model) + self.total_cycles("npu_overhead", cost_model)
        if denominator <= 0:
            return None
        return self.total_cycles("cpu_replaced", cost_model) / denominator


def estimate_npu_segment_cpu_counts(segment: NpuSegment, tensors: dict[str, TensorSpec]) -> PrimitiveCounts:
    # Reference CPU model:
    # each NPU segment is compared against a scalar in-order CPU kernel that
    # performs the same logical compute. We sum the per-op CPU counts inside
    # the segment so the benchmark represents "what the CPU would have done
    # instead of this NPU segment", not the current Python runtime behavior.
    total = PrimitiveCounts()
    for op in segment.ops:
        total += estimate_matmul_cpu_counts(op, tensors)
    return total


def estimate_matmul_cpu_counts(op: MatMulOp, tensors: dict[str, TensorSpec]) -> PrimitiveCounts:
    # Reference scalar CPU kernel model for one MxK * KxN matmul:
    #
    # For each output element and each inner-loop iteration k:
    # - 2 reads    : lhs[i, k], rhs[k, j]
    # - 1 mul      : lhs * rhs
    # - 1 add      : accumulate into acc
    # - 2 adds     : pointer/index maintenance for lhs/rhs streams
    # - 1 branch   : loop-control/check for the k loop body
    #
    # Per output element after the inner loop:
    # - optional bias read + add
    # - optional requant multiplier + shift
    # - optional clamp / relu
    # - 1 write for the output element
    # - 2 adds + 1 branch for the outer-loop/output-store progression
    lhs_shape = tuple(int(dim) for dim in tensors[op.lhs].shape)
    rhs_shape = tuple(int(dim) for dim in tensors[op.rhs].shape)
    out_shape = tuple(int(dim) for dim in tensors[op.out].shape)
    if len(lhs_shape) != 2 or len(rhs_shape) != 2 or len(out_shape) != 2:
        raise ValueError(f"Expected rank-2 tensors for matmul cost model, got {lhs_shape}, {rhs_shape}, {out_shape}.")

    m, k = lhs_shape

    k_rhs, n = rhs_shape
    if k_rhs != k:
        raise ValueError(f"Incompatible matmul shapes for cost model: lhs_effective=({m}, {k}) x rhs={rhs_shape}.")

    out_elements = m * n
    counts = PrimitiveCounts()
    counts.reads += out_elements * (2 * k)
    counts.muls += out_elements * k
    counts.adds += out_elements * k
    counts.adds += out_elements * (2 * k)
    counts.branches += out_elements * k
    if op.bias is not None:
        counts.reads += out_elements
        counts.adds += out_elements
    if op.multiplier != 1 or op.shift != 0:
        counts.muls += out_elements
        counts.shifts += out_elements
    if op.activation == "relu":
        counts.clamps += out_elements
    elif op.activation in {"sigmoid", "h_gelu"}:
        counts.nonlinear += out_elements
    if op.out_dtype in (DType.INT4, DType.INT8, DType.INT16):
        counts.clamps += out_elements
    counts.adds += out_elements * 2
    counts.branches += out_elements
    counts.writes += out_elements
    return counts


def estimate_host_op_counts(step: HostOp, values: dict[str, np.ndarray]) -> tuple[str, PrimitiveCounts]:
    # Host-op accounting is routed through the registry so evaluation semantics,
    # validation, and benchmark categorization stay in one place.
    bucket, counts = benchmark_host_op(step, values)
    return bucket, counts


def estimate_pack_counts(value: np.ndarray, packed_words: int) -> PrimitiveCounts:
    # Packing model:
    # each logical element is read, shifted/masked into a partial word, and
    # advances loop/index state. Each completed packed word is written once.
    elements = int(np.array(value, copy=False).size)
    return PrimitiveCounts(
        reads=elements,
        writes=packed_words,
        bitops=elements * 2,
        adds=elements,
        branches=elements,
    )


def estimate_unpack_counts(shape: tuple[int, ...], word_count: int) -> PrimitiveCounts:
    # Unpacking model:
    # each packed word is read, element lanes are shifted/masked out, loop/index
    # state advances, and the expanded logical elements are written back out.
    elements = int(np.prod(shape))
    return PrimitiveCounts(
        reads=word_count,
        writes=elements,
        bitops=elements * 2,
        adds=elements,
        branches=elements,
    )


def estimate_interface_write_counts(byte_writes: int) -> PrimitiveCounts:
    # Interface writes are kept separate from generic stores because they are
    # accelerator-induced protocol traffic, not normal CPU memory writes.
    return PrimitiveCounts(if_writes=int(byte_writes))


def estimate_interface_read_counts(byte_reads: int) -> PrimitiveCounts:
    # Interface reads are likewise tracked as protocol traffic so the report can
    # separate transfer overhead from replaced compute.
    return PrimitiveCounts(if_reads=int(byte_reads))
