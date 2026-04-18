from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from tinynpu import TinyNPUProgram

from .artifact import CompiledArtifact
from .ir import DType, HostOp, NpuSegment, TensorKind, TensorSpec, VerifyTensor


_DTYPE_TO_ENUM = {
    DType.INT4: "TNPU_DTYPE_INT4",
    DType.INT8: "TNPU_DTYPE_INT8",
    DType.INT16: "TNPU_DTYPE_INT16",
    DType.INT32: "TNPU_DTYPE_INT32",
    DType.FLOAT32: "TNPU_DTYPE_FLOAT32",
}

_HOST_KIND_ENUM = {
    "alias": "TNPU_HOST_ALIAS",
    "relu": "TNPU_HOST_RELU",
    "sigmoid": "TNPU_HOST_SIGMOID",
    "gelu": "TNPU_HOST_GELU",
    "quantize": "TNPU_HOST_QUANTIZE",
    "dequantize": "TNPU_HOST_DEQUANTIZE",
    "requantize": "TNPU_HOST_REQUANTIZE",
    "reshape": "TNPU_HOST_RESHAPE",
    "slice_row": "TNPU_HOST_SLICE_ROW",
    "transpose": "TNPU_HOST_TRANSPOSE",
    "softmax": "TNPU_HOST_SOFTMAX",
    "softmax_f16": "TNPU_HOST_SOFTMAX_F16",
    "mean": "TNPU_HOST_MEAN",
    "im2col": "TNPU_HOST_IM2COL",
    "layout_restore": "TNPU_HOST_LAYOUT_RESTORE",
    "rmsnorm": "TNPU_HOST_RMSNORM",
    "layernorm": "TNPU_HOST_LAYERNORM",
    "rope": "TNPU_HOST_ROPE",
    "silu": "TNPU_HOST_SILU",
    "mul": "TNPU_HOST_MUL",
    "add": "TNPU_HOST_ADD",
    "k_cache_scatter_write": "TNPU_HOST_K_CACHE_SCATTER_WRITE",
    "v_cache_scatter_write": "TNPU_HOST_V_CACHE_SCATTER_WRITE",
    "k_cache_scatter_matrix": "TNPU_HOST_K_CACHE_SCATTER_MATRIX",
    "v_cache_scatter_matrix": "TNPU_HOST_V_CACHE_SCATTER_MATRIX",
    "causal_mask": "TNPU_HOST_CAUSAL_MASK",
    "concat_lastdim2": "TNPU_HOST_CONCAT_LASTDIM2",
}


def _sanitize(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not cleaned:
        cleaned = "value"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned


def _flatten_tensor(data: np.ndarray | Any) -> np.ndarray:
    return np.array(data, copy=True).reshape(-1)


def _shape4(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(shape) > 4:
        raise NotImplementedError(f"Bare-metal emitter v2 supports rank <= 4, got shape {shape}.")
    padded = [int(dim) for dim in shape]
    while len(padded) < 4:
        padded.append(1)
    return tuple(padded)  # type: ignore[return-value]


def _as_int32_array(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=np.int32)


def _as_float32_array(data: np.ndarray) -> np.ndarray:
    return np.asarray(data, dtype=np.float32)


def _format_scalar(value: Any, *, dtype: DType) -> str:
    if dtype == DType.FLOAT32:
        number = float(value)
        if np.isnan(number):
            return "NAN"
        if np.isposinf(number):
            return "INFINITY"
        if np.isneginf(number):
            return "-INFINITY"
        text = repr(number)
        if "e" not in text and "." not in text:
            text += ".0"
        return f"{text}f"
    return str(int(value))


def _emit_array_decl(
    c_type: str,
    name: str,
    values: list[str] | None,
    length: int,
    *,
    section_data: bool,
) -> str:
    attr = ' __attribute__((section(".data")))' if section_data else ' __attribute__((section(".noinit")))'
    if values is None:
        return f"static {c_type} {name}[{length}]{attr};"
    if not values:
        return f"static {c_type} {name}[1]{attr} = {{0}};"
    return f"static {c_type} {name}[{length}]{attr} = {{\n    " + ", ".join(values) + "\n};"


def _emit_u32x4_image(name: str, words: list[int]) -> str:
    rows: list[str] = []
    for word in words:
        chunks = [(word >> (32 * idx)) & 0xFFFFFFFF for idx in range(4)]
        rows.append("    {" + ", ".join(f"0x{chunk:08x}u" for chunk in chunks) + "}")
    if not rows:
        rows = ["    {0u, 0u, 0u, 0u}"]
    return (
        f'static uint32_t {name}[{len(rows)}][TNPU_MMVR_WORDS_32] __attribute__((section(".data"))) = {{\n'
        + ",\n".join(rows)
        + "\n};"
    )


def _is_step_instance(step: Any, cls: type[Any]) -> bool:
    if isinstance(step, cls):
        return True
    return step.__class__.__name__ == cls.__name__


def _build_consumers(plan_steps: list[Any]) -> dict[str, list[Any]]:
    consumers: dict[str, list[Any]] = {}
    for step in plan_steps:
        if _is_step_instance(step, HostOp):
            for name in step.inputs:
                consumers.setdefault(name, []).append(step)
        elif _is_step_instance(step, NpuSegment):
            for name in step.inputs:
                consumers.setdefault(name, []).append(step)
        elif _is_step_instance(step, VerifyTensor):
            consumers.setdefault(step.tensor_name, []).append(step)
    return consumers


def emit_cv32e40p_program_v2(
    artifact: CompiledArtifact,
    inputs: dict[str, np.ndarray],
    *,
    program_name: str = "tinynpu_program_v2",
    defines_path: str | None = None,
) -> str:
    for name in artifact.plan.inputs:
        if name not in inputs:
            raise KeyError(f"Missing runtime input '{name}' for bare-metal v2 emission.")

    hw = TinyNPUProgram(defines_path=defines_path).hw.params
    array_size = int(hw.get("ARRAY_SIZE", 8))
    buffer_width = int(hw.get("BUFFER_WIDTH", 128))
    im_base_addr = int(hw.get("IM_BASE_ADDR", 0x8000))
    if array_size != 8 or buffer_width != 128:
        raise NotImplementedError(
            f"Bare-metal runtime v2 currently assumes ARRAY_SIZE=8 and BUFFER_WIDTH=128 (got {array_size}, {buffer_width})."
        )
    im_chunks_per_inst = 2

    symbol = _sanitize(program_name)
    decls: list[str] = []

    tensor_entries: list[str] = []
    tensor_index: dict[str, int] = {}

    def add_tensor(spec: TensorSpec, *, suffix: str = "", initial_override: np.ndarray | None = None) -> int:
        key = f"{spec.name}{suffix}"
        if key in tensor_index:
            return tensor_index[key]

        tensor_sym = f"{_sanitize(spec.name)}{suffix}"
        storage_name = f"{tensor_sym}_data"
        initial: np.ndarray | None = initial_override
        if initial is None:
            if suffix == "" and spec.name in inputs:
                initial = np.array(inputs[spec.name], copy=True)
            elif spec.data is not None:
                initial = np.array(spec.data, copy=True)
        elem_count = int(np.prod(spec.shape, dtype=np.int64)) if spec.shape else 1
        shape4 = _shape4(spec.shape if spec.shape else (1,))
        dtype_enum = _DTYPE_TO_ENUM[spec.dtype]

        if spec.dtype == DType.FLOAT32:
            if initial is None:
                storage_decl = _emit_array_decl("float", storage_name, None, elem_count, section_data=False)
            else:
                values = [_format_scalar(v, dtype=spec.dtype) for v in _as_float32_array(_flatten_tensor(initial))]
                storage_decl = _emit_array_decl("float", storage_name, values, elem_count, section_data=True)
        else:
            if initial is None:
                storage_decl = _emit_array_decl("int32_t", storage_name, None, elem_count, section_data=False)
            else:
                values = [_format_scalar(v, dtype=spec.dtype) for v in _as_int32_array(_flatten_tensor(initial))]
                storage_decl = _emit_array_decl("int32_t", storage_name, values, elem_count, section_data=True)
        decls.append(storage_decl)

        tensor_entries.append(
            "    {"
            f'.name = "{key}", .data = {storage_name}, .dtype = {dtype_enum}, .rank = {len(spec.shape) if spec.shape else 1}, '
            f".shape = {{{shape4[0]}, {shape4[1]}, {shape4[2]}, {shape4[3]}}}, .elem_count = {elem_count}"
            "}"
        )
        idx = len(tensor_entries) - 1
        tensor_index[key] = idx
        return idx

    for spec in artifact.plan.tensors.values():
        add_tensor(spec)

    expected_tensor_names: set[str] = set()
    for output_name in artifact.plan.outputs:
        if output_name in artifact.expected_tensors:
            expected_tensor_names.add(output_name)
    for step in artifact.plan.steps:
        if _is_step_instance(step, VerifyTensor) and step.tensor_name in artifact.expected_tensors:
            expected_tensor_names.add(step.tensor_name)

    def resolve_tensor_addr(tensor_name: str) -> int:
        for seg_artifact in artifact.segment_artifacts.values():
            symbol = seg_artifact.symbol_table.get(tensor_name)
            if symbol is not None:
                return int(symbol["addr"])
        raise KeyError(f"Unable to resolve planned address for tensor '{tensor_name}'.")
    for name in sorted(expected_tensor_names):
        expected_spec = artifact.plan.tensors[name].clone_without_data()
        expected_spec.data = np.array(artifact.expected_tensors[name], copy=True)
        add_tensor(expected_spec, suffix="_expected", initial_override=expected_spec.data)

    input_indices = [tensor_index[name] for name in artifact.plan.inputs]
    output_indices = [tensor_index[name] for name in artifact.plan.outputs]

    if input_indices:
        decls.append(
            f"static const uint16_t {symbol}_input_indices[{len(input_indices)}] = "
            + "{"
            + ", ".join(str(i) for i in input_indices)
            + "};"
        )
    if output_indices:
        decls.append(
            f"static const uint16_t {symbol}_output_indices[{len(output_indices)}] = "
            + "{"
            + ", ".join(str(i) for i in output_indices)
            + "};"
        )

    ub_preloads: list[str] = []
    if artifact.static_ub_image:
        image_name = f"{symbol}_ub_image"
        decls.append(_emit_u32x4_image(image_name, [int(word) for word in artifact.static_ub_image]))
        ub_preloads.append(
            "    {"
            '.label = "preload.ub_image", '
            f".base_addr = 0u, .image = {image_name}, .word_count = {len(artifact.static_ub_image)}"
            "}"
        )
    if ub_preloads:
        decls.append(
            f"static const TnpuImageLoad {symbol}_ub_preloads[{len(ub_preloads)}] = {{\n"
            + ",\n".join(ub_preloads)
            + "\n};"
        )

    segment_im_start: dict[str, int] = {}
    im_preloads: list[str] = []
    next_im_addr = im_base_addr
    for step in artifact.plan.steps:
        if not _is_step_instance(step, NpuSegment):
            continue
        segment = artifact.segment_artifacts[step.name]
        flattened_chunks: list[int] = []
        for inst in segment.binary["im"]:
            inst_int = int(inst)
            for chunk_idx in range(im_chunks_per_inst):
                chunk = (inst_int >> (chunk_idx * buffer_width)) & ((1 << buffer_width) - 1)
                flattened_chunks.append(int(chunk))
        image_name = f"{symbol}_im_{_sanitize(step.name)}"
        decls.append(_emit_u32x4_image(image_name, flattened_chunks))
        segment_im_start[step.name] = next_im_addr
        im_preloads.append(
            "    {"
            f'.label = "preload.im_{_sanitize(step.name)}", '
            f".base_addr = 0x{next_im_addr:04x}u, .image = {image_name}, .word_count = {len(flattened_chunks)}"
            "}"
        )
        next_im_addr += len(flattened_chunks)
    if im_preloads:
        decls.append(
            f"static const TnpuImageLoad {symbol}_im_preloads[{len(im_preloads)}] = {{\n"
            + ",\n".join(im_preloads)
            + "\n};"
        )

    tensor_consumers = _build_consumers(artifact.plan.steps)
    producer_by_output: dict[str, HostOp] = {}
    for step in artifact.plan.steps:
        if _is_step_instance(step, HostOp):
            for output_name in step.outputs:
                producer_by_output[output_name] = step

    absorbed_quantize_outputs: set[str] = set()
    for output_name, step in producer_by_output.items():
        if step.kind != "quantize":
            continue
        uses = tensor_consumers.get(output_name, [])
        non_verify_uses = [use for use in uses if not _is_step_instance(use, VerifyTensor)]
        if not non_verify_uses:
            continue
        if not all(_is_step_instance(use, NpuSegment) for use in non_verify_uses):
            continue
        absorbed_quantize_outputs.add(output_name)

    absorbed_dequantize_by_input: dict[str, HostOp] = {}
    for step in artifact.plan.steps:
        if not _is_step_instance(step, HostOp):
            continue
        if step.kind != "dequantize":
            continue
        source_name = step.inputs[0]
        source_spec = artifact.plan.tensors[source_name]
        output_spec = artifact.plan.tensors[step.outputs[0]]
        uses = tensor_consumers.get(source_name, [])
        non_verify_uses = [use for use in uses if not _is_step_instance(use, VerifyTensor)]
        if len(non_verify_uses) != 1 or non_verify_uses[0] is not step:
            continue
        if source_name in expected_tensor_names:
            continue
        if source_spec.dtype != DType.INT16:
            continue
        if output_spec.dtype != DType.FLOAT32:
            continue
        absorbed_dequantize_by_input[source_name] = step

    absorbed_dequantize_step_names: set[str] = set()

    segment_entries: list[str] = []
    segment_index: dict[str, int] = {}
    for step in artifact.plan.steps:
        if not _is_step_instance(step, NpuSegment):
            continue
        segment = artifact.segment_artifacts[step.name]
        produced_inside = {op.out for op in step.ops}
        writes: list[str] = []
        for tensor_name in step.inputs:
            spec = artifact.plan.tensors[tensor_name]
            if spec.kind == TensorKind.CONSTANT:
                continue
            if tensor_name in produced_inside:
                continue
            sym = segment.symbol_table[tensor_name]
            write_tensor_name = tensor_name
            transform = "TNPU_WRITE_TRANSFORM_NONE"
            attrs_i32 = [0, 0]
            attrs_f32 = [0.0]
            if tensor_name in absorbed_quantize_outputs:
                producer = producer_by_output[tensor_name]
                source_name = producer.inputs[0]
                source_spec = artifact.plan.tensors[source_name]
                if (
                    source_spec.dtype == DType.FLOAT32
                    and int(sym["precision"]) == 2
                    and str(sym["role"]) == "A"
                ):
                    write_tensor_name = source_name
                    transform = "TNPU_WRITE_QUANTIZE_F32_TO_INT16"
                    attrs_i32[0] = int(producer.attrs.get("zero_point", 0))
                    attrs_f32[0] = 1.0 / float(producer.attrs["scale"])
                elif (
                    source_spec.dtype == DType.INT16
                    and int(sym["precision"]) == 2
                    and str(sym["role"]) == "A"
                    and int(producer.attrs.get("zero_point", 0)) == 0
                ):
                    source_producer = producer_by_output.get(source_name)
                    if (
                        str(producer.attrs.get("input_encoding", "")) == "fp16_bits"
                        or (source_producer is not None and source_producer.kind == "softmax_f16")
                        or (source_producer is not None and str(source_producer.attrs.get("output_encoding", "")) == "fp16_bits")
                    ):
                        write_tensor_name = source_name
                        transform = "TNPU_WRITE_XFORM_Q_F16_I16"
                        attrs_f32[0] = 1.0 / float(producer.attrs["scale"])
            writes.append(
                "    {"
                f".tensor_idx = {tensor_index[write_tensor_name]}, .addr = 0x{int(sym['addr']):04x}u, "
                f".word_count = {int(sym['word_count'])}u, .precision = {int(sym['precision'])}, "
                f".transform = {transform}, "
                f".attrs_i32 = {{{attrs_i32[0]}, {attrs_i32[1]}}}, "
                f".attrs_f32 = {{{_format_scalar(attrs_f32[0], dtype=DType.FLOAT32)}}}, "
                f'.role = "{sym["role"]}"'
                "}"
            )
        for symbol_name, sym in sorted(segment.symbol_table.items()):
            if sym["role"] != "BIAS":
                continue
            spec = artifact.plan.tensors.get(symbol_name)
            if spec is not None and spec.kind == TensorKind.CONSTANT:
                continue
            if symbol_name not in tensor_index:
                continue
            writes.append(
                "    {"
                f".tensor_idx = {tensor_index[symbol_name]}, .addr = 0x{int(sym['addr']):04x}u, "
                f".word_count = {int(sym['word_count'])}u, .precision = {int(sym['precision'])}, "
                ".transform = TNPU_WRITE_TRANSFORM_NONE, .attrs_i32 = {0, 0}, .attrs_f32 = {0.0f}, "
                f'.role = "{sym["role"]}"'
                "}"
            )
        reads: list[str] = []
        for output_name in step.outputs:
            sym = segment.symbol_table[output_name]
            read_tensor_name = output_name
            read_transform = "TNPU_READ_TRANSFORM_NONE"
            read_attrs_i32 = [0, 0]
            read_attrs_f32 = [0.0]
            dequant_step = absorbed_dequantize_by_input.get(output_name)
            if dequant_step is not None:
                read_tensor_name = dequant_step.outputs[0]
                read_transform = "TNPU_READ_DEQUANTIZE_INT16_TO_FLOAT32"
                read_attrs_i32[0] = int(dequant_step.attrs.get("zero_point", 0))
                read_attrs_f32[0] = float(dequant_step.attrs["scale"])
                absorbed_dequantize_step_names.add(dequant_step.name)
            reads.append(
                "    {"
                f".tensor_idx = {tensor_index[read_tensor_name]}, .addr = 0x{int(sym['addr']):04x}u, "
                f".precision = {int(sym['precision'])}, .transform = {read_transform}, "
                f".attrs_i32 = {{{read_attrs_i32[0]}, {read_attrs_i32[1]}}}, "
                f".attrs_f32 = {{{_format_scalar(read_attrs_f32[0], dtype=DType.FLOAT32)}}}, "
                f".role = \"{sym['role']}\""
                "}"
            )
        writes_name = f"{symbol}_seg_{_sanitize(step.name)}_writes"
        reads_name = f"{symbol}_seg_{_sanitize(step.name)}_reads"
        if writes:
            decls.append(
                f"static const TnpuTensorWrite {writes_name}[{len(writes)}] = {{\n"
                + ",\n".join(writes)
                + "\n};"
            )
        if reads:
            decls.append(
                f"static const TnpuTensorRead {reads_name}[{len(reads)}] = {{\n"
                + ",\n".join(reads)
                + "\n};"
            )
        segment_entries.append(
            "    {"
            f'.name = "{step.name}", .im_start_addr = 0x{segment_im_start[step.name]:04x}u, '
            f".writes = {writes_name if writes else 'NULL'}, .write_count = {len(writes)}u, "
            f".reads = {reads_name if reads else 'NULL'}, .read_count = {len(reads)}u"
            "}"
        )
        segment_index[step.name] = len(segment_entries) - 1
    if segment_entries:
        decls.append(
            f"static const TnpuSegment {symbol}_segments[{len(segment_entries)}] = {{\n"
            + ",\n".join(segment_entries)
            + "\n};"
        )

    host_entries: list[str] = []
    host_index: dict[str, int] = {}
    verify_entries: list[str] = []
    verify_index: dict[str, int] = {}
    op_entries: list[str] = []

    if ub_preloads:
        op_entries.append(f"    {{.kind = TNPU_OP_PRELOAD_UB, .index = 0u}}")
    for idx in range(len(im_preloads)):
        op_entries.append(f"    {{.kind = TNPU_OP_PRELOAD_IM, .index = {idx}u}}")

    for step in artifact.plan.steps:
        if _is_step_instance(step, HostOp):
            if step.kind == "quantize" and step.outputs and step.outputs[0] in absorbed_quantize_outputs:
                continue
            if step.kind == "dequantize" and step.name in absorbed_dequantize_step_names:
                continue
            if step.kind not in _HOST_KIND_ENUM:
                raise NotImplementedError(f"Bare-metal runtime v2 emitter does not support host op '{step.kind}'.")
            attrs_i32 = [0] * 8
            attrs_f32 = [0.0, 0.0]
            arr0_name = "NULL"
            arr0_len = 0

            if step.kind in {"dequantize", "requantize"}:
                attrs_f32[0] = float(step.attrs["scale"])
                attrs_i32[0] = int(step.attrs.get("zero_point", 0))
            elif step.kind == "quantize":
                attrs_f32[0] = 1.0 / float(step.attrs["scale"])
                attrs_i32[0] = int(step.attrs.get("zero_point", 0))
                attrs_i32[1] = 1 if str(step.attrs.get("input_encoding", "")) == "fp16_bits" else 0
            elif step.kind == "slice_row":
                attrs_i32[0] = int(step.attrs.get("row_index", 0))
            elif step.kind == "transpose":
                axes = tuple(int(axis) for axis in step.attrs.get("axes", []))
                if axes:
                    arr0_name = f"{symbol}_host_{_sanitize(step.name)}_axes"
                    arr0_len = len(axes)
                    decls.append(
                        f"static const int {arr0_name}[{arr0_len}] = "
                        + "{"
                        + ", ".join(str(v) for v in axes)
                        + "};"
                    )
            elif step.kind in {"softmax", "softmax_f16"}:
                attrs_i32[0] = int(step.attrs.get("axis", -1))
            elif step.kind == "mean":
                dims = step.attrs.get("dim")
                if dims is not None:
                    dims_tuple = tuple(int(dim) for dim in dims)
                    if dims_tuple:
                        arr0_name = f"{symbol}_host_{_sanitize(step.name)}_dims"
                        arr0_len = len(dims_tuple)
                        decls.append(
                            f"static const int {arr0_name}[{arr0_len}] = "
                            + "{"
                            + ", ".join(str(v) for v in dims_tuple)
                            + "};"
                        )
                attrs_i32[0] = 1 if bool(step.attrs.get("keepdim", False)) else 0
                input_quant = step.attrs.get("input_quantization")
                if input_quant is None:
                    attrs_i32[1] = 0
                    attrs_f32[0] = 1.0
                    attrs_i32[2] = 0
                else:
                    attrs_i32[1] = 1
                    attrs_f32[0] = float(input_quant["scale"])
                    attrs_i32[2] = int(input_quant.get("zero_point", 0))
            elif step.kind == "im2col":
                attrs_i32[0] = int(step.attrs["kernel_size"])
                attrs_i32[1] = int(step.attrs.get("stride", 1))
                attrs_i32[2] = int(step.attrs.get("padding", 0))
                layout = str(step.attrs.get("input_layout", "hwc"))
                if layout == "matrix_hwc":
                    attrs_i32[3] = 2
                    attrs_i32[4] = int(step.attrs["matrix_h"])
                    attrs_i32[5] = int(step.attrs["matrix_w"])
                    attrs_i32[6] = int(step.attrs["matrix_c"])
                else:
                    attrs_i32[3] = 1 if layout == "chw" else 0
            elif step.kind == "layout_restore":
                attrs_i32[0] = 1 if str(step.attrs["layout"]) == "chw" else 0
                attrs_i32[1] = len(tuple(step.attrs["original_shape"]))
                attrs_i32[2] = int(step.attrs["out_h"])
                attrs_i32[3] = int(step.attrs["out_w"])
                attrs_i32[4] = int(step.attrs["out_channels"])
            elif step.kind in {"rmsnorm", "layernorm"}:
                attrs_f32[0] = float(step.attrs.get("eps", 1.0e-6))
            elif step.kind == "rope":
                head_dim = int(step.attrs.get("head_dim", 0))
                attrs_i32[0] = head_dim
                attrs_i32[1] = int(step.attrs.get("position", 0))
                theta = float(step.attrs.get("theta", 10000.0))
                attrs_f32[0] = theta
                half = head_dim // 2
                if half > 0:
                    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / np.float32(half)))
                    inv_freq_bits = np.asarray(inv_freq, dtype=np.float32).view(np.int32)
                    arr0_name = f"{symbol}_host_{_sanitize(step.name)}_inv_freq_bits"
                    arr0_len = int(inv_freq_bits.size)
                    decls.append(
                        f"static const int {arr0_name}[{arr0_len}] = "
                        + "{"
                        + ", ".join(str(int(v)) for v in inv_freq_bits.tolist())
                        + "};"
                    )
            elif step.kind == "k_cache_scatter_write":
                out_spec = artifact.plan.tensors[step.outputs[0]]
                base_name = str(out_spec.metadata.get("storage_view_of", step.outputs[0]))
                base_addr = resolve_tensor_addr(base_name)
                scatter_addrs = tuple(base_addr + int(offset) for offset in out_spec.metadata["cache_scatter_word_addrs"])
                token_lane = int(out_spec.metadata["cache_token_index"]) % 8
                attrs_i32[0] = token_lane
                arr0_name = f"{symbol}_host_{_sanitize(step.name)}_scatter_addrs"
                arr0_len = len(scatter_addrs)
                decls.append(
                    f"static const int {arr0_name}[{arr0_len}] = "
                    + "{"
                    + ", ".join(str(v) for v in scatter_addrs)
                    + "};"
                )
            elif step.kind == "v_cache_scatter_write":
                out_spec = artifact.plan.tensors[step.outputs[0]]
                base_name = str(out_spec.metadata.get("storage_view_of", step.outputs[0]))
                base_addr = resolve_tensor_addr(base_name)
                scatter_addrs = tuple(base_addr + int(offset) for offset in out_spec.metadata["cache_scatter_word_addrs"])
                arr0_name = f"{symbol}_host_{_sanitize(step.name)}_scatter_addrs"
                arr0_len = len(scatter_addrs)
                decls.append(
                    f"static const int {arr0_name}[{arr0_len}] = "
                    + "{"
                    + ", ".join(str(v) for v in scatter_addrs)
                    + "};"
                )
            elif step.kind in {"k_cache_scatter_matrix", "v_cache_scatter_matrix"}:
                attrs_i32[0] = resolve_tensor_addr(step.outputs[0])
            elif step.kind == "causal_mask":
                attrs_i32[0] = int(step.attrs.get("past_kv_len", 0))
                attrs_f32[0] = float(step.attrs.get("fill_value", -32768.0))

            host_entries.append(
                "    {"
                f'.name = "{step.name}", .kind = {_HOST_KIND_ENUM[step.kind]}, '
                f".input_idx = {tensor_index[step.inputs[0]]}, "
                f".input1_idx = {tensor_index[step.inputs[1]] if len(step.inputs) > 1 else 0xFFFF}, "
                f".output_idx = {tensor_index[step.outputs[0]]}, "
                ".attrs_i32 = {"
                + ", ".join(str(v) for v in attrs_i32)
                + "}, .attrs_f32 = {"
                + ", ".join(_format_scalar(v, dtype=DType.FLOAT32) for v in attrs_f32)
                + "}, "
                f".arr0 = {arr0_name}, .arr0_len = {arr0_len}u"
                "}"
            )
            host_index[step.name] = len(host_entries) - 1
            op_entries.append(f"    {{.kind = TNPU_OP_HOST, .index = {host_index[step.name]}u}}")
            continue

        if _is_step_instance(step, NpuSegment):
            op_entries.append(f"    {{.kind = TNPU_OP_SEGMENT, .index = {segment_index[step.name]}u}}")
            continue

        if _is_step_instance(step, VerifyTensor):
            if step.tensor_name in absorbed_quantize_outputs:
                continue
            if step.tensor_name not in expected_tensor_names:
                continue
            verify_entries.append(
                "    {"
                f'.label = "{step.label}", .actual_tensor_idx = {tensor_index[step.tensor_name]}, '
                f'.expected_tensor_idx = {tensor_index[step.tensor_name + "_expected"]}, '
                f".is_final_output = {1 if step.is_final_output else 0}, "
                f".float_atol = {_format_scalar(step.float_atol, dtype=DType.FLOAT32)}"
                "}"
            )
            verify_index[step.tensor_name] = len(verify_entries) - 1
            op_entries.append(
                f"    {{.kind = TNPU_OP_VERIFY, .index = {verify_index[step.tensor_name]}u}}"
            )

    if tensor_entries:
        decls.append(
            f"static const TnpuTensorDesc {symbol}_tensors[{len(tensor_entries)}] = {{\n"
            + ",\n".join(tensor_entries)
            + "\n};"
        )
    if host_entries:
        decls.append(
            f"static const TnpuHostOp {symbol}_host_ops[{len(host_entries)}] = {{\n"
            + ",\n".join(host_entries)
            + "\n};"
        )
    if verify_entries:
        decls.append(
            f"static const TnpuVerifyOp {symbol}_verify_ops[{len(verify_entries)}] = {{\n"
            + ",\n".join(verify_entries)
            + "\n};"
        )
    if op_entries:
        decls.append(
            f"static const TnpuOp {symbol}_ops[{len(op_entries)}] = {{\n"
            + ",\n".join(op_entries)
            + "\n};"
        )

    input_idx_ptr = f"{symbol}_input_indices" if input_indices else "NULL"
    output_idx_ptr = f"{symbol}_output_indices" if output_indices else "NULL"
    ub_preloads_ptr = f"{symbol}_ub_preloads" if ub_preloads else "NULL"
    im_preloads_ptr = f"{symbol}_im_preloads" if im_preloads else "NULL"
    segments_ptr = f"{symbol}_segments" if segment_entries else "NULL"
    host_ops_ptr = f"{symbol}_host_ops" if host_entries else "NULL"
    verify_ops_ptr = f"{symbol}_verify_ops" if verify_entries else "NULL"
    ops_ptr = f"{symbol}_ops" if op_entries else "NULL"

    program_decl = (
        f"const TnpuProgram {symbol} = {{\n"
        f'    .name = "{program_name}",\n'
        f"    .tensors = {symbol}_tensors,\n"
        f"    .tensor_count = {len(tensor_entries)}u,\n"
        f"    .input_tensor_indices = {input_idx_ptr},\n"
        f"    .input_count = {len(input_indices)}u,\n"
        f"    .output_tensor_indices = {output_idx_ptr},\n"
        f"    .output_count = {len(output_indices)}u,\n"
        f"    .ub_preloads = {ub_preloads_ptr},\n"
        f"    .ub_preload_count = {len(ub_preloads)}u,\n"
        f"    .im_preloads = {im_preloads_ptr},\n"
        f"    .im_preload_count = {len(im_preloads)}u,\n"
        f"    .segments = {segments_ptr},\n"
        f"    .segment_count = {len(segment_entries)}u,\n"
        f"    .host_ops = {host_ops_ptr},\n"
        f"    .host_op_count = {len(host_entries)}u,\n"
        f"    .verify_ops = {verify_ops_ptr},\n"
        f"    .verify_op_count = {len(verify_entries)}u,\n"
        f"    .ops = {ops_ptr},\n"
        f"    .op_count = {len(op_entries)}u,\n"
        "};"
    )

    source = (
        '#include <stddef.h>\n#include <stdint.h>\n#include "tinynpu_runtime_v2.h"\n\n'
        + "\n\n".join(decls)
        + "\n\n"
        + program_decl
        + "\n"
    )
    return source


def write_cv32e40p_program_v2(
    artifact: CompiledArtifact,
    inputs: dict[str, np.ndarray],
    output_path: str | Path,
    *,
    program_name: str = "tinynpu_program_v2",
    defines_path: str | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        emit_cv32e40p_program_v2(
            artifact,
            inputs,
            program_name=program_name,
            defines_path=defines_path,
        )
    )
    return output
