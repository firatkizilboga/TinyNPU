from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from tinynpu.isa import PrecisionMode
from tinynpu.packer import Packer


class DType(str, Enum):
    INT4 = "int4"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    FLOAT32 = "float32"


class TensorKind(str, Enum):
    INPUT = "input"
    CONSTANT = "constant"
    INTERMEDIATE = "intermediate"
    OUTPUT = "output"


class VerificationMode(str, Enum):
    OFF = "off"
    FINAL = "final"
    DEBUG = "debug"


@dataclass
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: DType
    kind: TensorKind
    data: np.ndarray | None = None
    is_final_output: bool = False
    verify_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def clone_without_data(self, name: str | None = None) -> "TensorSpec":
        return TensorSpec(
            name=name or self.name,
            shape=self.shape,
            dtype=self.dtype,
            kind=self.kind,
            data=None,
            is_final_output=self.is_final_output,
            verify_label=self.verify_label,
            metadata=dict(self.metadata),
        )


@dataclass
class MatMulOp:
    name: str
    lhs: str
    rhs: str
    out: str
    bias: str | None = None
    multiplier: int = 1
    shift: int = 0
    activation: str = "none"
    h_gelu_x_scale_shift: int = 7
    in_dtype: DType = DType.INT16
    out_dtype: DType = DType.INT16
    output_layout: str = "c"
    writeback_mode: str = "normal"
    output_word_offset: int = 0
    b_word_offset: int = 0


@dataclass
class NpuSegment:
    name: str
    ops: list[MatMulOp]
    inputs: list[str]
    outputs: list[str]


@dataclass
class HostOp:
    name: str
    kind: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerifyTensor:
    tensor_name: str
    label: str
    is_final_output: bool = False


PlanStep = NpuSegment | HostOp | VerifyTensor


@dataclass
class ExecutionPlan:
    tensors: dict[str, TensorSpec]
    steps: list[PlanStep]
    inputs: list[str]
    outputs: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_verification_step(self, tensor_name: str, label: str | None = None) -> None:
        tensor = self.tensors[tensor_name]
        self.steps.append(
            VerifyTensor(
                tensor_name=tensor_name,
                label=label or tensor.verify_label or tensor_name,
                is_final_output=tensor.is_final_output,
            )
        )


def normalize_shape(shape: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in shape)


_DTYPE_TO_PRECISION = {
    DType.INT4: PrecisionMode.INT4,
    DType.INT8: PrecisionMode.INT8,
    DType.INT16: PrecisionMode.INT16,
}


def to_precision_mode(dtype: DType) -> PrecisionMode:
    if dtype not in _DTYPE_TO_PRECISION:
        raise ValueError(f"TinyNPU precision does not support dtype {dtype}.")
    return _DTYPE_TO_PRECISION[dtype]


def numpy_dtype_for(dtype: DType) -> np.dtype:
    if dtype == DType.FLOAT32:
        return np.float32
    if dtype == DType.INT32:
        return np.int32
    return np.int16


def b_slot_word_stride(shape: tuple[int, int], dtype: DType, array_size: int = 8) -> int:
    precision = to_precision_mode(dtype)
    pack_ratio = 1 << (2 - precision)
    rows, cols = shape
    k = (rows // pack_ratio + array_size - 1) // array_size
    n = (cols + array_size - 1) // array_size
    return Packer(array_size).get_physical_word_count("B", precision, 1, k, n)


def make_b_cache_view_spec(
    name: str,
    base_name: str,
    shape: tuple[int, int],
    dtype: DType,
    *,
    kind: TensorKind = TensorKind.INTERMEDIATE,
    word_offset: int,
    cache_kind: str | None = None,
    slot_index: int | None = None,
) -> TensorSpec:
    metadata: dict[str, Any] = {
        "storage_view_of": base_name,
        "storage_role": "B",
        "storage_word_offset": int(word_offset),
    }
    if cache_kind is not None:
        metadata["cache_kind"] = cache_kind
    if slot_index is not None:
        metadata["cache_slot_index"] = int(slot_index)
    return TensorSpec(name, shape, dtype, kind, metadata=metadata)


def make_b_cache_specs(
    base_name: str,
    slot_shape: tuple[int, int],
    dtype: DType,
    *,
    slot_names: list[str],
    kind: TensorKind = TensorKind.INTERMEDIATE,
    cache_kind: str | None = None,
    array_size: int = 8,
) -> dict[str, TensorSpec]:
    slot_stride_words = b_slot_word_stride(slot_shape, dtype, array_size=array_size)
    base_shape = (slot_shape[0] * len(slot_names), slot_shape[1])
    specs: dict[str, TensorSpec] = {
        base_name: TensorSpec(base_name, base_shape, dtype, kind),
    }
    for slot_index, slot_name in enumerate(slot_names):
        specs[slot_name] = make_b_cache_view_spec(
            slot_name,
            base_name,
            slot_shape,
            dtype,
            kind=kind,
            word_offset=slot_index * slot_stride_words,
            cache_kind=cache_kind,
            slot_index=slot_index,
        )
        specs[slot_name].metadata["cache_slot_stride_words"] = slot_stride_words
    return specs


def make_kv_cache_specs(
    *,
    k_base_name: str,
    v_base_name: str,
    k_slot_shape: tuple[int, int],
    v_slot_shape: tuple[int, int],
    dtype: DType,
    slot_suffixes: list[str],
    kind: TensorKind = TensorKind.INTERMEDIATE,
    array_size: int = 8,
) -> dict[str, TensorSpec]:
    specs: dict[str, TensorSpec] = {}
    specs.update(
        make_b_cache_specs(
            k_base_name,
            k_slot_shape,
            dtype,
            slot_names=[f"{k_base_name}_{suffix}" for suffix in slot_suffixes],
            kind=kind,
            cache_kind="K",
            array_size=array_size,
        )
    )
    specs.update(
        make_b_cache_specs(
            v_base_name,
            v_slot_shape,
            dtype,
            slot_names=[f"{v_base_name}_{suffix}" for suffix in slot_suffixes],
            kind=kind,
            cache_kind="V",
            array_size=array_size,
        )
    )
    return specs


@dataclass(frozen=True)
class Int16KCacheAppendContract:
    cache_shape: tuple[int, int]
    token_index: int
    token_block: int
    token_lane: int
    k_tiles: int
    block_word_base: int
    block_word_count: int
    scatter_word_addrs: tuple[int, ...]
    lane_partial_write: bool = True


@dataclass(frozen=True)
class Int16VCacheAppendContract:
    cache_shape: tuple[int, int]
    token_index: int
    token_block: int
    row_in_block: int
    n_tiles: int
    block_word_base: int
    block_word_count: int
    scatter_word_addrs: tuple[int, ...]
    lane_partial_write: bool = False


def describe_int16_k_cache_append(d_head: int, token_capacity: int, token_index: int, array_size: int = 8) -> Int16KCacheAppendContract:
    if d_head <= 0 or token_capacity <= 0:
        raise ValueError("d_head and token_capacity must be positive.")
    if token_index < 0 or token_index >= token_capacity:
        raise ValueError("token_index out of range for K cache.")
    k_tiles = (d_head + array_size - 1) // array_size
    token_block = token_index // array_size
    token_lane = token_index % array_size
    block_word_count = k_tiles * array_size
    block_word_base = token_block * block_word_count
    scatter = tuple(
        block_word_base + (k_tile * array_size) + row_idx
        for k_tile in range(k_tiles)
        for row_idx in range(array_size)
    )
    return Int16KCacheAppendContract(
        cache_shape=(d_head, token_capacity),
        token_index=token_index,
        token_block=token_block,
        token_lane=token_lane,
        k_tiles=k_tiles,
        block_word_base=block_word_base,
        block_word_count=block_word_count,
        scatter_word_addrs=scatter,
    )


def describe_int16_v_cache_append(d_head: int, token_capacity: int, token_index: int, array_size: int = 8) -> Int16VCacheAppendContract:
    if d_head <= 0 or token_capacity <= 0:
        raise ValueError("d_head and token_capacity must be positive.")
    if token_index < 0 or token_index >= token_capacity:
        raise ValueError("token_index out of range for V cache.")
    n_tiles = (d_head + array_size - 1) // array_size
    token_block = token_index // array_size
    row_in_block = token_index % array_size
    block_word_count = n_tiles * array_size
    block_word_base = token_block * block_word_count
    scatter = tuple(
        block_word_base + (n_tile * array_size) + row_in_block
        for n_tile in range(n_tiles)
    )
    return Int16VCacheAppendContract(
        cache_shape=(token_capacity, d_head),
        token_index=token_index,
        token_block=token_block,
        row_in_block=row_in_block,
        n_tiles=n_tiles,
        block_word_base=block_word_base,
        block_word_count=block_word_count,
        scatter_word_addrs=scatter,
    )
