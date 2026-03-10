"""
Global tensor lifetime analysis and memory planner for TinyNPU.

Two public entry points:
  plan_segment_memory(segment, plan, ub_capacity) -> SegmentMemoryPlan
  plan_program_memory(plan, ub_capacity) -> GlobalMemoryReport

The global planner splits UB into two zones:
  [0, static_zone_end)         Static zone: weights + biases (CONSTANT tensors).
                                Globally unique addresses shared across all segments.
                                Load once at model init.
  [static_zone_end, ub_cap)    Dynamic zone: activations + outputs.
                                Allocated per-segment with within-segment liveness
                                reuse (linear-scan). All segments share this zone
                                because only one segment executes at a time.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from tinynpu.isa import PrecisionMode
from tinynpu.packer import Packer

from .ir import DType, ExecutionPlan, NpuSegment, TensorKind, TensorSpec


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class TensorLiveness:
    name: str
    birth_op: int   # -1  = segment input (alive before op 0)
    death_op: int   # ≥ n = segment output (alive past the last op)


@dataclass
class MemoryPlanEntry:
    name: str
    address: int
    word_count: int
    reuses_from: str | None = None   # tensor whose UB slot was reused


@dataclass
class SegmentMemoryPlan:
    segment_name: str
    entries: list[MemoryPlanEntry]
    total_words: int
    reused_words: int
    ub_capacity: int
    is_feasible: bool


@dataclass
class GlobalMemoryReport:
    segments: list[SegmentMemoryPlan]
    cross_segment_tensors: list[str]
    total_ub_peak: int            # max(total_words) across all segments
    theoretical_minimum_ub: int   # static_zone_end + max per-segment dynamic footprint
    static_zone_end: int          # first address after all static (weight/bias) data
    static_ub_image: list[int]    # packed UB words for [0, static_zone_end) — load once


# ---------------------------------------------------------------------------
# Role inference  (also imported by lowering.py to avoid duplication)
# ---------------------------------------------------------------------------

def infer_roles(segment: NpuSegment) -> dict[str, str]:
    """Return per-tensor hardware role (A / B / BIAS / C) for a segment."""
    uses: dict[str, set[str]] = defaultdict(set)
    for op in segment.ops:
        uses[op.lhs].add("lhs")
        uses[op.rhs].add("rhs")
        if op.bias:
            uses[op.bias].add("bias")
        uses[op.out].add("out")

    roles: dict[str, str] = {}
    for name, kinds in uses.items():
        if "bias" in kinds:
            roles[name] = "BIAS"
        elif kinds == {"rhs"}:
            roles[name] = "B"
        elif kinds == {"lhs"}:
            roles[name] = "A"
        elif "out" in kinds:
            roles[name] = "C"
        else:
            roles[name] = "C"
    return roles


# ---------------------------------------------------------------------------
# Liveness analysis
# ---------------------------------------------------------------------------

def compute_liveness(segment: NpuSegment) -> dict[str, TensorLiveness]:
    """
    Compute per-tensor live intervals within a segment.

    birth_op = -1   tensor is a segment input (alive before op 0)
    birth_op = i    tensor is first produced by op i
    death_op = n    tensor is a segment output (must survive past the last op)
    death_op = i    tensor last consumed by op i
    """
    ops = segment.ops
    n = len(ops)

    all_names: set[str] = set(segment.inputs + segment.outputs)
    for op in ops:
        all_names.update([op.lhs, op.rhs, op.out])
        if op.bias:
            all_names.add(op.bias)

    # Default: born before segment, outlives segment
    birth: dict[str, int] = {name: -1 for name in all_names}
    death: dict[str, int] = {name: n for name in all_names}

    # Tensors produced by an op (and not declared as a segment input) are born at that op
    for i, op in enumerate(ops):
        if op.out not in segment.inputs and birth[op.out] == -1:
            birth[op.out] = i

    # death = last op index that reads the tensor; segment outputs override to n
    last_use: dict[str, int] = {}
    for i, op in enumerate(ops):
        for name in [op.lhs, op.rhs] + ([op.bias] if op.bias else []):
            last_use[name] = i

    for name in all_names:
        if name in segment.outputs:
            death[name] = n          # must outlive segment
        elif name in last_use:
            death[name] = last_use[name]
        else:
            death[name] = birth[name]  # never consumed → dead at birth

    return {
        name: TensorLiveness(name=name, birth_op=birth[name], death_op=death[name])
        for name in all_names
    }


# ---------------------------------------------------------------------------
# Word-count and packing helpers
# ---------------------------------------------------------------------------

def _effective_precision(dtype: DType) -> PrecisionMode:
    _map = {
        DType.INT4:  PrecisionMode.INT4,
        DType.INT8:  PrecisionMode.INT8,
        DType.INT16: PrecisionMode.INT16,
    }
    return _map.get(dtype, PrecisionMode.INT16)


def _shape2d(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 1:
        return (1, shape[0])
    return (shape[0], shape[1])


def _compute_word_count(spec: TensorSpec, role: str, packer: Packer, array_size: int) -> int:
    precision = _effective_precision(spec.dtype)
    p = 1 << (2 - precision)
    sz = array_size
    rows, cols = _shape2d(spec.shape)

    if role == "A":
        m = (rows + sz - 1) // sz
        k = (cols // p + sz - 1) // sz
        n = 1
    elif role == "B":
        k = (rows // p + sz - 1) // sz
        n = (cols + sz - 1) // sz
        m = 1
    elif role == "BIAS":
        m, k, n = 1, 1, (cols + sz - 1) // sz
    else:  # C
        m = (rows + sz - 1) // sz
        n = (cols + sz - 1) // sz
        k = 1

    return packer.get_physical_word_count(role, precision, m, k, n)


def _pack_data(spec: TensorSpec, role: str, packer: Packer, array_size: int) -> list[int]:
    precision = _effective_precision(spec.dtype)
    p = 1 << (2 - precision)
    sz = array_size
    rows, cols = _shape2d(spec.shape)

    if spec.data is None:
        return [0] * _compute_word_count(spec, role, packer, array_size)

    if role == "BIAS":
        data = np.array(spec.data, dtype=np.int32)
    else:
        data = np.array(spec.data, dtype=np.int16)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if role == "A":
        m = (rows + sz - 1) // sz
        k = (cols // p + sz - 1) // sz
        n = 1
    elif role == "B":
        k = (rows // p + sz - 1) // sz
        n = (cols + sz - 1) // sz
        m = 1
    elif role == "BIAS":
        m, k, n = 1, 1, (cols + sz - 1) // sz
    else:
        m = (rows + sz - 1) // sz
        n = (cols + sz - 1) // sz
        k = 1

    return list(packer.pack(data, role, precision, m, k, n))


# ---------------------------------------------------------------------------
# Linear-scan free-list allocator
# ---------------------------------------------------------------------------

def _first_fit(free_list: list[list[int]], size: int) -> int | None:
    for slot in free_list:
        if slot[1] >= size:
            addr = slot[0]
            slot[0] += size
            slot[1] -= size
            if slot[1] == 0:
                free_list.remove(slot)
            return addr
    return None


def _free_slot(free_list: list[list[int]], addr: int, size: int) -> None:
    free_list.append([addr, size])
    free_list.sort(key=lambda s: s[0])
    i = 0
    while i < len(free_list) - 1:
        if free_list[i][0] + free_list[i][1] == free_list[i + 1][0]:
            free_list[i][1] += free_list[i + 1][1]
            free_list.pop(i + 1)
        else:
            i += 1


def _linear_scan_dynamic(
    segment: NpuSegment,
    plan: ExecutionPlan,
    roles: dict[str, str],
    liveness: dict[str, TensorLiveness],
    packer: Packer,
    array_size: int,
    dynamic_start: int,
    ub_capacity: int,
) -> dict[str, MemoryPlanEntry]:
    """
    Allocate dynamic (non-CONSTANT) tensors in [dynamic_start, ub_capacity)
    using a linear-scan allocator that reuses slots after tensors die.
    """
    n = len(segment.ops)

    # Build sorted event list: (step, priority, name, kind)
    # priority 0 = free (process before alloc at same step to enable reuse)
    events: list[tuple[int, int, str, str]] = []
    for name, live in liveness.items():
        spec = plan.tensors.get(name)
        if spec is None or spec.kind == TensorKind.CONSTANT:
            continue
        alloc_at = max(0, live.birth_op)
        events.append((alloc_at, 1, name, "alloc"))
        if live.death_op < n:   # segment outputs (death_op == n) are never freed
            events.append((live.death_op + 1, 0, name, "free"))

    events.sort()

    free_list: list[list[int]] = [[dynamic_start, ub_capacity - dynamic_start]]
    entries: dict[str, MemoryPlanEntry] = {}
    live_slots: dict[str, tuple[int, int]] = {}
    last_tenant: dict[int, str] = {}   # addr -> most recently freed tensor name

    for _, _, name, kind in events:
        if kind == "free":
            if name in live_slots:
                addr, wc = live_slots.pop(name)
                last_tenant[addr] = name
                _free_slot(free_list, addr, wc)
        else:  # alloc
            role = roles.get(name, "C")
            spec = plan.tensors[name]
            wc = _compute_word_count(spec, role, packer, array_size)
            addr = _first_fit(free_list, wc)
            if addr is None:
                raise MemoryError(
                    f"OOM: cannot allocate {wc} words for '{name}' "
                    f"in segment '{segment.name}' (dynamic zone "
                    f"[{dynamic_start}, {ub_capacity}))"
                )
            reuses_from = last_tenant.get(addr)
            entries[name] = MemoryPlanEntry(
                name=name, address=addr, word_count=wc, reuses_from=reuses_from
            )
            live_slots[name] = (addr, wc)

    return entries


# ---------------------------------------------------------------------------
# Hardware config helper
# ---------------------------------------------------------------------------

def _get_packer() -> tuple[Packer, int]:
    try:
        from tinynpu.program import HardwareConfig
        hw = HardwareConfig()
        sz = int(hw.params.get("ARRAY_SIZE", 8))
        return Packer(sz), sz
    except Exception:
        return Packer(8), 8


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plan_segment_memory(
    segment: NpuSegment,
    plan: ExecutionPlan,
    ub_capacity: int,
) -> SegmentMemoryPlan:
    """
    Plan memory for a single segment in isolation (no cross-segment static sharing).
    Useful for unit tests and single-segment models.
    """
    if ub_capacity <= 0:
        ub_capacity = 0x8000

    packer, array_size = _get_packer()
    roles = infer_roles(segment)
    liveness = compute_liveness(segment)

    # Bump-allocate static tensors from address 0
    static_addr = 0
    static_entries: dict[str, MemoryPlanEntry] = {}
    for name in sorted(roles):
        spec = plan.tensors.get(name)
        if spec is None or spec.kind != TensorKind.CONSTANT:
            continue
        wc = _compute_word_count(spec, roles[name], packer, array_size)
        static_entries[name] = MemoryPlanEntry(name=name, address=static_addr, word_count=wc)
        static_addr += wc

    dynamic_entries = _linear_scan_dynamic(
        segment, plan, roles, liveness, packer, array_size, static_addr, ub_capacity
    )

    all_entries = list(static_entries.values()) + list(dynamic_entries.values())
    total_words = max((e.address + e.word_count for e in all_entries), default=0)
    reused_words = sum(e.word_count for e in all_entries if e.reuses_from is not None)

    return SegmentMemoryPlan(
        segment_name=segment.name,
        entries=all_entries,
        total_words=total_words,
        reused_words=reused_words,
        ub_capacity=ub_capacity,
        is_feasible=total_words <= ub_capacity,
    )


def plan_program_memory(
    plan: ExecutionPlan,
    ub_capacity: int = 0,
) -> GlobalMemoryReport:
    """
    Plan memory globally across all NPU segments in the execution plan.

    Static tensors (CONSTANT kind) receive globally unique UB addresses and are
    consolidated into a single static_ub_image that the simulator loads once at
    model-init time rather than reloading on every invocation.

    Dynamic tensors (activations, outputs) share a dynamic zone that starts
    immediately after the static zone.  Within each segment, a linear-scan
    allocator reuses freed dynamic slots.  Across segments the dynamic zone is
    shared because only one segment executes at a time.
    """
    if ub_capacity <= 0:
        ub_capacity = 0x8000

    packer, array_size = _get_packer()
    segments = [step for step in plan.steps if isinstance(step, NpuSegment)]

    # ------------------------------------------------------------------
    # Phase 1: globally unique addresses for all static tensors
    # ------------------------------------------------------------------
    # A static tensor that appears in multiple segments always has the same
    # global address, so it's loaded once and accessed from any segment.

    roles_by_seg: dict[str, dict[str, str]] = {}
    first_role: dict[str, tuple[str, str]] = {}   # tensor_name -> (seg_name, role)

    for seg in segments:
        roles = infer_roles(seg)
        roles_by_seg[seg.name] = roles
        for name, role in roles.items():
            spec = plan.tensors.get(name)
            if spec is not None and spec.kind == TensorKind.CONSTANT and name not in first_role:
                first_role[name] = (seg.name, role)

    static_entries: dict[str, MemoryPlanEntry] = {}
    static_addr = 0
    for name in sorted(first_role):
        _, role = first_role[name]
        spec = plan.tensors[name]
        wc = _compute_word_count(spec, role, packer, array_size)
        static_entries[name] = MemoryPlanEntry(name=name, address=static_addr, word_count=wc)
        static_addr += wc

    static_zone_end = static_addr

    # Build the static UB image (covers [0, static_zone_end))
    static_ub_image: list[int] = [0] * static_zone_end
    for name, entry in static_entries.items():
        _, role = first_role[name]
        spec = plan.tensors[name]
        packed = _pack_data(spec, role, packer, array_size)
        for i, word in enumerate(packed):
            static_ub_image[entry.address + i] = word

    # ------------------------------------------------------------------
    # Phase 2: per-segment dynamic allocation (shared dynamic zone)
    # ------------------------------------------------------------------
    segment_plans: list[SegmentMemoryPlan] = []

    for seg in segments:
        roles = roles_by_seg[seg.name]
        liveness = compute_liveness(seg)

        dynamic_entries = _linear_scan_dynamic(
            seg, plan, roles, liveness, packer, array_size, static_zone_end, ub_capacity
        )

        # Collect static entries that are referenced by this segment
        seg_static = [static_entries[name] for name in roles if name in static_entries]
        all_entries = seg_static + list(dynamic_entries.values())

        total_words = max((e.address + e.word_count for e in all_entries), default=0)
        reused_words = sum(e.word_count for e in all_entries if e.reuses_from is not None)

        segment_plans.append(SegmentMemoryPlan(
            segment_name=seg.name,
            entries=all_entries,
            total_words=total_words,
            reused_words=reused_words,
            ub_capacity=ub_capacity,
            is_feasible=total_words <= ub_capacity,
        ))

    # ------------------------------------------------------------------
    # Phase 3: cross-segment analysis
    # ------------------------------------------------------------------
    tensor_seg_usage: dict[str, set[str]] = defaultdict(set)
    for seg in segments:
        for name in seg.inputs + seg.outputs:
            tensor_seg_usage[name].add(seg.name)
    cross_segment = sorted(name for name, segs in tensor_seg_usage.items() if len(segs) > 1)

    total_ub_peak = max((sp.total_words for sp in segment_plans), default=0)
    max_dynamic = max(
        (sp.total_words - static_zone_end for sp in segment_plans), default=0
    )
    theoretical_min = static_zone_end + max_dynamic

    return GlobalMemoryReport(
        segments=segment_plans,
        cross_segment_tensors=cross_segment,
        total_ub_peak=total_ub_peak,
        theoretical_minimum_ub=theoretical_min,
        static_zone_end=static_zone_end,
        static_ub_image=static_ub_image,
    )
