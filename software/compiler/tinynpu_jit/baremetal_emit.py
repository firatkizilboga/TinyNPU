from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from tinynpu import TinyNPUProgram

from .artifact import CompiledArtifact
from .ir import DType, HostOp, NpuSegment, TensorKind, TensorSpec, VerifyTensor


_DTYPE_TO_ENUM = {
    DType.INT4: "TINY_DTYPE_INT4",
    DType.INT8: "TINY_DTYPE_INT8",
    DType.INT16: "TINY_DTYPE_INT16",
    DType.INT32: "TINY_DTYPE_INT32",
    DType.FLOAT32: "TINY_DTYPE_FLOAT32",
}


def _sanitize(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    if not cleaned:
        cleaned = "tensor"
    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    return cleaned


def _flatten_tensor(data: np.ndarray | Any) -> np.ndarray:
    return np.array(data, copy=True).reshape(-1)


def _shape4(shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    if len(shape) > 4:
        raise NotImplementedError(f"Bare-metal emitter supports rank <= 4, got shape {shape}.")
    padded = list(int(dim) for dim in shape)
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
    if section_data:
        attr = ' __attribute__((section(".data")))'
    else:
        attr = ' __attribute__((section(".noinit")))'
    if values is None:
        return f"static {c_type} {name}[{length}]{attr};"
    if not values:
        return f"static {c_type} {name}[1]{attr} = {{0}};"
    return f"static {c_type} {name}[{length}]{attr} = {{\n    " + ", ".join(values) + "\n};"


def _emit_u32x4_image(name: str, words: list[int]) -> str:
    rows: list[str] = []
    for word in words:
        chunks = [(word >> (32 * idx)) & 0xFFFFFFFF for idx in range(4)]
        rows.append(
            "    {" + ", ".join(f"0x{chunk:08x}u" for chunk in chunks) + "}"
        )
    if not rows:
        rows = ["    {0u, 0u, 0u, 0u}"]
    return (
        f'static uint32_t {name}[{len(rows)}][TINY_BUFFER_WORDS_32] __attribute__((section(".data"), aligned(16))) = {{\n'
        + ",\n".join(rows)
        + "\n};"
    )


def _emit_tensor_storage(
    spec: TensorSpec,
    *,
    runtime_inputs: dict[str, np.ndarray],
    suffix: str = "",
) -> tuple[str, str]:
    storage_name = f"{_sanitize(spec.name)}{suffix}_data"
    tensor_name = f"{_sanitize(spec.name)}{suffix}"
    initial: np.ndarray | None = None
    if spec.name in runtime_inputs:
        initial = np.array(runtime_inputs[spec.name], copy=True)
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

    tensor_decl = (
        f'static TinyTensor {tensor_name} = {{'
        f'"{spec.name}", {storage_name}, {dtype_enum}, {len(spec.shape) if spec.shape else 1}, '
        f'{{{shape4[0]}, {shape4[1]}, {shape4[2]}, {shape4[3]}}}, {elem_count}'
        f"}};"
    )
    return storage_decl, tensor_decl


def _emit_tensor_reference(name: str, suffix: str = "") -> str:
    return f"&{_sanitize(name)}{suffix}"


def _activation_code(kind: str) -> int:
    mapping = {
        "none": 0,
        "relu": 1,
        "sigmoid": 2,
        "h_gelu": 3,
    }
    if kind not in mapping:
        raise NotImplementedError(f"Bare-metal CPU baseline does not support activation '{kind}'.")
    return mapping[kind]


def _emit_host_step_attrs(step: HostOp) -> tuple[list[str], list[str]]:
    decls: list[str] = []
    lines: list[str] = []
    out_ref = _emit_tensor_reference(step.outputs[0])
    in_ref = _emit_tensor_reference(step.inputs[0])
    prefix = _sanitize(step.name)

    if step.kind == "alias":
        lines.append(f"    host_alias({out_ref}, {in_ref});")
    elif step.kind == "relu":
        lines.append(f"    host_relu({out_ref}, {in_ref});")
    elif step.kind == "sigmoid":
        lines.append(f"    host_sigmoid({out_ref}, {in_ref});")
    elif step.kind == "gelu":
        lines.append(f"    host_gelu({out_ref}, {in_ref});")
    elif step.kind == "quantize":
        scale = float(step.attrs["scale"])
        inv_scale = 1.0 / scale
        zero_point = int(step.attrs.get("zero_point", 0))
        lines.append(f"    host_quantize({out_ref}, {in_ref}, {_format_scalar(inv_scale, dtype=DType.FLOAT32)}, {zero_point});")
    elif step.kind == "dequantize":
        scale = float(step.attrs["scale"])
        zero_point = int(step.attrs.get("zero_point", 0))
        lines.append(f"    host_dequantize({out_ref}, {in_ref}, {_format_scalar(scale, dtype=DType.FLOAT32)}, {zero_point});")
    elif step.kind == "requantize":
        scale = float(step.attrs["scale"])
        zero_point = int(step.attrs.get("zero_point", 0))
        lines.append(f"    host_requantize({out_ref}, {in_ref}, {_format_scalar(scale, dtype=DType.FLOAT32)}, {zero_point});")
    elif step.kind == "reshape":
        lines.append(f"    host_reshape({out_ref}, {in_ref});")
    elif step.kind == "transpose":
        axes = tuple(int(axis) for axis in step.attrs.get("axes", []))
        if axes:
            axes_name = f"{prefix}_axes"
            decls.append(f"static int {axes_name}[{len(axes)}] = " + "{" + ", ".join(str(axis) for axis in axes) + "};")
            lines.append(f"    host_transpose({out_ref}, {in_ref}, {axes_name}, {len(axes)});")
        else:
            lines.append(f"    host_transpose({out_ref}, {in_ref}, NULL, 0);")
    elif step.kind == "softmax":
        axis = int(step.attrs.get("axis", -1))
        lines.append(f"    host_softmax({out_ref}, {in_ref}, {axis});")
    elif step.kind == "mean":
        dims = step.attrs.get("dim")
        if dims is None:
            dims_name = "NULL"
            dim_count = 0
        else:
            dims_tuple = tuple(int(dim) for dim in dims)
            dims_name = f"{prefix}_dims"
            dim_count = len(dims_tuple)
            decls.append(f"static int {dims_name}[{dim_count}] = " + "{" + ", ".join(str(dim) for dim in dims_tuple) + "};")
        keepdim = 1 if bool(step.attrs.get("keepdim", False)) else 0
        input_quant = step.attrs.get("input_quantization")
        if input_quant is None:
            has_quant = 0
            scale = 1.0
            zero_point = 0
        else:
            has_quant = 1
            scale = float(input_quant["scale"])
            zero_point = int(input_quant.get("zero_point", 0))
        lines.append(
            "    host_mean("
            f"{out_ref}, {in_ref}, {dims_name}, {dim_count}, {keepdim}, {has_quant}, "
            f"{_format_scalar(scale, dtype=DType.FLOAT32)}, {zero_point});"
        )
    elif step.kind == "im2col":
        kernel_size = int(step.attrs["kernel_size"])
        stride = int(step.attrs.get("stride", 1))
        padding = int(step.attrs.get("padding", 0))
        input_layout = str(step.attrs.get("input_layout", "hwc"))
        if input_layout == "matrix_hwc":
            matrix_h = int(step.attrs["matrix_h"])
            matrix_w = int(step.attrs["matrix_w"])
            matrix_c = int(step.attrs["matrix_c"])
            lines.append(
                f"    host_im2col_matrix({out_ref}, {in_ref}, {matrix_h}, {matrix_w}, {matrix_c}, {kernel_size}, {stride}, {padding});"
            )
        else:
            layout_is_chw = 1 if input_layout == "chw" else 0
            lines.append(
                f"    host_im2col({out_ref}, {in_ref}, {kernel_size}, {stride}, {padding}, {layout_is_chw});"
            )
    elif step.kind == "layout_restore":
        layout_is_chw = 1 if str(step.attrs["layout"]) == "chw" else 0
        original_rank = len(tuple(step.attrs["original_shape"]))
        out_h = int(step.attrs["out_h"])
        out_w = int(step.attrs["out_w"])
        out_channels = int(step.attrs["out_channels"])
        lines.append(
            f"    host_layout_restore({out_ref}, {in_ref}, {layout_is_chw}, {original_rank}, {out_h}, {out_w}, {out_channels});"
        )
    else:
        raise NotImplementedError(f"Bare-metal runtime emitter does not support host op '{step.kind}'.")
    return decls, lines


def _load_runtime_template() -> str:
    template_path = Path(__file__).resolve().parent / "templates" / "cv32e40p_runtime_template.c"
    return template_path.read_text()


def emit_cv32e40p_c(
    artifact: CompiledArtifact,
    inputs: dict[str, np.ndarray],
    *,
    program_name: str = "tinynpu_generated",
    defines_path: str | None = None,
    emit_cpu_baseline: bool = False,
    verify_cpu_baseline: bool = False,
    repeat_count: int = 1,
    cpu_only_baseline: bool = False,
) -> str:
    if verify_cpu_baseline and not emit_cpu_baseline:
        raise ValueError("verify_cpu_baseline requires emit_cpu_baseline=True.")
    if cpu_only_baseline and emit_cpu_baseline:
        raise ValueError("cpu_only_baseline replaces emit_cpu_baseline; do not enable both.")
    if repeat_count < 1:
        raise ValueError("repeat_count must be >= 1.")

    for name in artifact.plan.inputs:
        if name not in inputs:
            raise KeyError(f"Missing runtime input '{name}' for bare-metal emission.")

    hw = TinyNPUProgram(defines_path=defines_path).hw.params
    array_size = int(hw.get("ARRAY_SIZE", 8))
    buffer_width = int(hw.get("BUFFER_WIDTH", 128))
    im_base_addr = int(hw.get("IM_BASE_ADDR", 0x8000))
    if buffer_width != 128:
        raise NotImplementedError(f"Bare-metal emitter currently assumes 128-bit MMIO words, got {buffer_width}.")
    buffer_words_32 = buffer_width // 32
    inst_width = 256
    im_chunks_per_inst = max(1, inst_width // buffer_width)

    decls: list[str] = []
    main_lines: list[str] = [
        "    uint32_t cycle_t0 = 0;",
        "    uint32_t cycle_t1 = 0;",
        "    uint32_t cycle_segment_t0 = 0;",
        f'    printf("TinyNPU bare-metal program: {program_name}\\n");',
        "    tb_timer_reset_counter();",
    ]
    if repeat_count > 1:
        main_lines.extend(
            [
                "    uint32_t delta = 0;",
                "    uint32_t preload_total = 0;",
                "    uint32_t host_total = 0;",
                "    uint32_t segment_npu_total = 0;",
                "    uint32_t segment_cpu_total = 0;",
            ]
        )
    verify_lines: list[str] = []

    tensor_names_emitted: set[str] = set()
    for spec in artifact.plan.tensors.values():
        storage_decl, tensor_decl = _emit_tensor_storage(spec, runtime_inputs=inputs)
        decls.append(storage_decl)
        decls.append(tensor_decl)
        tensor_names_emitted.add(spec.name)

    expected_tensor_names: set[str] = set()
    for output_name in artifact.plan.outputs:
        if output_name in artifact.expected_tensors:
            expected_tensor_names.add(output_name)
    for step in artifact.plan.steps:
        if isinstance(step, VerifyTensor) and step.tensor_name in artifact.expected_tensors:
            expected_tensor_names.add(step.tensor_name)

    for tensor_name in sorted(expected_tensor_names):
        expected_spec = artifact.plan.tensors[tensor_name].clone_without_data()
        expected_spec.data = np.array(artifact.expected_tensors[tensor_name], copy=True)
        storage_decl, tensor_decl = _emit_tensor_storage(expected_spec, runtime_inputs={}, suffix="_expected")
        decls.append(storage_decl)
        decls.append(tensor_decl)

    if emit_cpu_baseline:
        for step in artifact.plan.steps:
            if not isinstance(step, NpuSegment):
                continue
            cpu_suffix = f"__cpu_{_sanitize(step.name)}"
            for tensor_name in sorted({op.out for op in step.ops}):
                cpu_spec = artifact.plan.tensors[tensor_name].clone_without_data()
                storage_decl, tensor_decl = _emit_tensor_storage(cpu_spec, runtime_inputs={}, suffix=cpu_suffix)
                decls.append(storage_decl)
                decls.append(tensor_decl)

    if artifact.static_ub_image and not cpu_only_baseline:
        decls.append(_emit_u32x4_image("tinynpu_static_ub_image", [int(word) for word in artifact.static_ub_image]))
        main_lines.append("    cycle_t0 = read_mcycle32();")
        main_lines.append(
            f"    load_ub_image(0u, tinynpu_static_ub_image, {len(artifact.static_ub_image)});"
        )
        main_lines.append("    cycle_t1 = read_mcycle32();")
        if repeat_count > 1:
            main_lines.append("    delta = cycle_t0 - cycle_t1;")
            main_lines.append("    preload_total += delta;")
        main_lines.append('    print_cycle_delta32("preload.ub_image", cycle_t0, cycle_t1);')

    im_start_addrs: dict[str, int] = {}
    next_im_addr = im_base_addr
    if not cpu_only_baseline:
        for step in artifact.plan.steps:
            if not isinstance(step, NpuSegment):
                continue
            segment = artifact.segment_artifacts[step.name]
            flattened_chunks: list[int] = []
            for inst in segment.binary["im"]:
                inst_int = int(inst)
                for chunk_idx in range(im_chunks_per_inst):
                    chunk = (inst_int >> (chunk_idx * buffer_width)) & ((1 << buffer_width) - 1)
                    flattened_chunks.append(int(chunk))
            image_name = f"im_{_sanitize(step.name)}"
            im_start_addrs[step.name] = next_im_addr
            decls.append(_emit_u32x4_image(image_name, flattened_chunks))
            main_lines.append("    cycle_t0 = read_mcycle32();")
            main_lines.append(
                f"    load_im_image(0x{next_im_addr:04x}u, {image_name}, {len(flattened_chunks)});"
            )
            main_lines.append("    cycle_t1 = read_mcycle32();")
            if repeat_count > 1:
                main_lines.append("    delta = cycle_t0 - cycle_t1;")
                main_lines.append("    preload_total += delta;")
            main_lines.append(f'    print_cycle_delta32("preload.{image_name}", cycle_t0, cycle_t1);')
            next_im_addr += len(flattened_chunks)

    body_lines: list[str] = []

    for step in artifact.plan.steps:
        if isinstance(step, HostOp):
            attr_decls, lines = _emit_host_step_attrs(step)
            decls.extend(attr_decls)
            if repeat_count > 1:
                body_lines.append(f'    if (repeat_iter == 0) printf("HostOp {step.kind}: {step.name}\\n");')
                body_lines.append("    cycle_t0 = read_mcycle32();")
                body_lines.extend(lines)
                body_lines.append("    cycle_t1 = read_mcycle32();")
                body_lines.append("    delta = cycle_t0 - cycle_t1;")
                body_lines.append("    host_total += delta;")
                body_lines.append(
                    f'    if (repeat_iter == 0) print_cycle_delta32("hostop.{_sanitize(step.name)}", cycle_t0, cycle_t1);'
                )
            else:
                body_lines.append(f'    printf("HostOp {step.kind}: {step.name}\\n");')
                body_lines.append("    cycle_t0 = read_mcycle32();")
                body_lines.extend(lines)
                body_lines.append("    cycle_t1 = read_mcycle32();")
                body_lines.append(f'    print_cycle_delta32("hostop.{_sanitize(step.name)}", cycle_t0, cycle_t1);')
            continue

        if isinstance(step, NpuSegment):
            produced_inside = {op.out for op in step.ops}
            cpu_suffix = f"__cpu_{_sanitize(step.name)}"
            cpu_outputs = {op.out for op in step.ops}

            def _segment_ref(name: str | None) -> str:
                if name is None:
                    return "NULL"
                if emit_cpu_baseline and name in cpu_outputs:
                    return _emit_tensor_reference(name, cpu_suffix)
                return _emit_tensor_reference(name)

            if cpu_only_baseline:
                if repeat_count > 1:
                    body_lines.append(f'    if (repeat_iter == 0) printf("CpuSegment: {step.name}\\n");')
                else:
                    body_lines.append(f'    printf("CpuSegment: {step.name}\\n");')
                body_lines.append("    cycle_t0 = read_mcycle32();")
                for op in step.ops:
                    body_lines.append(
                        "    host_matmul("
                        f"{_segment_ref(op.out)}, {_segment_ref(op.lhs)}, {_segment_ref(op.rhs)}, {_segment_ref(op.bias)}, "
                        f"{int(op.multiplier)}, {int(op.shift)}, {_activation_code(op.activation)}, {int(op.h_gelu_x_scale_shift)});"
                    )
                body_lines.append("    cycle_t1 = read_mcycle32();")
                if repeat_count > 1:
                    body_lines.append("    delta = cycle_t0 - cycle_t1;")
                    body_lines.append("    segment_cpu_total += delta;")
                    body_lines.append(
                        f'    if (repeat_iter == 0) print_cycle_delta32("segment.{_sanitize(step.name)}.cpu", cycle_t0, cycle_t1);'
                    )
                else:
                    body_lines.append(f'    print_cycle_delta32("segment.{_sanitize(step.name)}.cpu", cycle_t0, cycle_t1);')
                continue

            segment = artifact.segment_artifacts[step.name]

            if repeat_count > 1:
                body_lines.append(f'    if (repeat_iter == 0) printf("NpuSegment: {step.name}\\n");')
            else:
                body_lines.append(f'    printf("NpuSegment: {step.name}\\n");')
            body_lines.append("    cycle_segment_t0 = read_mcycle32();")
            body_lines.append("    cycle_t0 = read_mcycle32();")
            for tensor_name in step.inputs:
                spec = artifact.plan.tensors[tensor_name]
                if spec.kind == TensorKind.CONSTANT:
                    continue
                if tensor_name in produced_inside:
                    continue
                symbol = segment.symbol_table[tensor_name]
                body_lines.append(
                    "    write_tensor_to_npu("
                    f"{_emit_tensor_reference(tensor_name)}, 0x{int(symbol['addr']):04x}u, "
                    f"\"{symbol['role']}\", {int(symbol['precision'])}, {int(symbol['word_count'])});"
                )
            for symbol_name, symbol in sorted(segment.symbol_table.items()):
                if symbol["role"] != "BIAS":
                    continue
                spec = artifact.plan.tensors.get(symbol_name)
                if spec is not None and spec.kind == TensorKind.CONSTANT:
                    continue
                body_lines.append(
                    "    write_tensor_to_npu("
                    f"{_emit_tensor_reference(symbol_name)}, 0x{int(symbol['addr']):04x}u, "
                    f"\"{symbol['role']}\", {int(symbol['precision'])}, {int(symbol['word_count'])});"
                )
            body_lines.append("    cycle_t1 = read_mcycle32();")
            if repeat_count > 1:
                body_lines.append(
                    f'    if (repeat_iter == 0) print_cycle_delta32("segment.{_sanitize(step.name)}.stage", cycle_t0, cycle_t1);'
                )
            else:
                body_lines.append(f'    print_cycle_delta32("segment.{_sanitize(step.name)}.stage", cycle_t0, cycle_t1);')
            body_lines.append("    cycle_t0 = read_mcycle32();")
            body_lines.append(f"    if (npu_run(0x{im_start_addrs[step.name]:04x}u) != 0) return EXIT_FAILURE;")
            body_lines.append("    cycle_t1 = read_mcycle32();")
            if repeat_count > 1:
                body_lines.append(
                    f'    if (repeat_iter == 0) print_cycle_delta32("segment.{_sanitize(step.name)}.run", cycle_t0, cycle_t1);'
                )
            else:
                body_lines.append(f'    print_cycle_delta32("segment.{_sanitize(step.name)}.run", cycle_t0, cycle_t1);')
            body_lines.append("    cycle_t0 = read_mcycle32();")
            for output_name in step.outputs:
                symbol = segment.symbol_table[output_name]
                body_lines.append(
                    "    read_tensor_from_npu("
                    f"{_emit_tensor_reference(output_name)}, 0x{int(symbol['addr']):04x}u, "
                    f"\"{symbol['role']}\", {int(symbol['precision'])});"
                )
            body_lines.append("    cycle_t1 = read_mcycle32();")
            if repeat_count > 1:
                body_lines.append("    delta = cycle_segment_t0 - cycle_t1;")
                body_lines.append("    segment_npu_total += delta;")
                body_lines.append(
                    f'    if (repeat_iter == 0) print_cycle_delta32("segment.{_sanitize(step.name)}.readback", cycle_t0, cycle_t1);'
                )
                body_lines.append(
                    f'    if (repeat_iter == 0) print_cycle_delta32("segment.{_sanitize(step.name)}.npu", cycle_segment_t0, cycle_t1);'
                )
            else:
                body_lines.append(f'    print_cycle_delta32("segment.{_sanitize(step.name)}.readback", cycle_t0, cycle_t1);')
                body_lines.append(f'    print_cycle_delta32("segment.{_sanitize(step.name)}.npu", cycle_segment_t0, cycle_t1);')

            if emit_cpu_baseline:
                body_lines.append("    cycle_t0 = read_mcycle32();")
                for op in step.ops:
                    body_lines.append(
                        "    host_matmul("
                        f"{_segment_ref(op.out)}, {_segment_ref(op.lhs)}, {_segment_ref(op.rhs)}, {_segment_ref(op.bias)}, "
                        f"{int(op.multiplier)}, {int(op.shift)}, {_activation_code(op.activation)}, {int(op.h_gelu_x_scale_shift)});"
                    )
                body_lines.append("    cycle_t1 = read_mcycle32();")
                if repeat_count > 1:
                    body_lines.append("    delta = cycle_t0 - cycle_t1;")
                    body_lines.append("    segment_cpu_total += delta;")
                    body_lines.append(
                        f'    if (repeat_iter == 0) print_cycle_delta32("segment.{_sanitize(step.name)}.cpu", cycle_t0, cycle_t1);'
                    )
                else:
                    body_lines.append(f'    print_cycle_delta32("segment.{_sanitize(step.name)}.cpu", cycle_t0, cycle_t1);')

                if verify_cpu_baseline:
                    for output_name in step.outputs:
                        cpu_output_ref = _emit_tensor_reference(output_name, cpu_suffix)
                        npu_output_ref = _emit_tensor_reference(output_name)
                        body_lines.append(f"    if (!tensor_matches_expected({npu_output_ref}, {cpu_output_ref})) {{")
                        body_lines.append(
                            f'        printf("cpu baseline mismatch: segment {step.name} output {output_name}\\n");'
                        )
                        body_lines.append(f"        print_tensor({npu_output_ref});")
                        body_lines.append(f"        print_tensor({cpu_output_ref});")
                        body_lines.append("        return EXIT_FAILURE;")
                        body_lines.append("    }")
            continue

        if isinstance(step, VerifyTensor):
            if step.tensor_name in artifact.expected_tensors:
                verify_lines.append(
                    f"    if (!tensor_matches_expected({_emit_tensor_reference(step.tensor_name)}, {_emit_tensor_reference(step.tensor_name, '_expected')})) {{"
                )
                verify_lines.append(
                    f'        printf("verification failed: {step.label} ({step.tensor_name})\\n");'
                )
                verify_lines.append(
                    f'        printf("meta actual dtype=%d elems=%d expected dtype=%d elems=%d\\n", '
                    f'{_sanitize(step.tensor_name)}.dtype, {_sanitize(step.tensor_name)}.elem_count, '
                    f'{_sanitize(step.tensor_name)}_expected.dtype, {_sanitize(step.tensor_name)}_expected.elem_count);'
                )
                verify_lines.append(f"        print_tensor({_emit_tensor_reference(step.tensor_name)});")
                verify_lines.append(f"        print_tensor({_emit_tensor_reference(step.tensor_name, '_expected')});")
                verify_lines.append("        return EXIT_FAILURE;")
                verify_lines.append("    }")

    if repeat_count > 1:
        main_lines.append(f"    for (int repeat_iter = 0; repeat_iter < {repeat_count}; ++repeat_iter) {{")
        main_lines.append('        printf("repeat.iter=%d\\n", repeat_iter + 1);')
        main_lines.append("        fflush(stdout);")
        main_lines.extend([f"    {line}" for line in body_lines])
        main_lines.append("    }")
        main_lines.append(f'    printf("repeat.count=%d\\n", {repeat_count});')
        main_lines.append('    printf("repeat.preload.total cycles=%lu\\n", (unsigned long)preload_total);')
        main_lines.append('    printf("repeat.host.shared.total cycles=%lu\\n", (unsigned long)host_total);')
        if cpu_only_baseline:
            main_lines.append('    printf("repeat.segment.cpu.total cycles=%lu\\n", (unsigned long)segment_cpu_total);')
            main_lines.append(
                '    printf("repeat.program.cpu.hot.total cycles=%lu\\n", (unsigned long)(host_total + segment_cpu_total));'
            )
            main_lines.append(
                '    printf("repeat.program.cpu.cold.total cycles=%lu\\n", (unsigned long)(preload_total + host_total + segment_cpu_total));'
            )
            main_lines.append(
                '    printf("repeat.program.cpu.hot.avg cycles=%lu\\n", (unsigned long)((host_total + segment_cpu_total) / (uint32_t)'
                f"{repeat_count}"
                '));'
            )
        else:
            main_lines.append('    printf("repeat.segment.npu.total cycles=%lu\\n", (unsigned long)segment_npu_total);')
            main_lines.append(
                '    printf("repeat.program.npu.hot.total cycles=%lu\\n", (unsigned long)(host_total + segment_npu_total));'
            )
            main_lines.append(
                '    printf("repeat.program.npu.cold.total cycles=%lu\\n", (unsigned long)(preload_total + host_total + segment_npu_total));'
            )
            main_lines.append(
                '    printf("repeat.program.npu.hot.avg cycles=%lu\\n", (unsigned long)((host_total + segment_npu_total) / (uint32_t)'
                f"{repeat_count}"
                '));'
            )
            if emit_cpu_baseline:
                main_lines.append('    printf("repeat.segment.cpu.total cycles=%lu\\n", (unsigned long)segment_cpu_total);')
                main_lines.append(
                    '    printf("repeat.program.cpu.hot.total cycles=%lu\\n", (unsigned long)(host_total + segment_cpu_total));'
                )
                main_lines.append(
                    '    printf("repeat.program.cpu.hot.avg cycles=%lu\\n", (unsigned long)((host_total + segment_cpu_total) / (uint32_t)'
                    f"{repeat_count}"
                    '));'
                )
    else:
        main_lines.extend(body_lines)

    main_lines.append('    printf("Final outputs:\\n");')
    for output_name in artifact.plan.outputs:
        main_lines.append(f"    print_tensor({_emit_tensor_reference(output_name)});")
    main_lines.extend(verify_lines)
    if artifact.plan.outputs:
        main_lines.append('    printf("All outputs matched expected tensors\\n");')
    main_lines.append("    return EXIT_SUCCESS;")

    source = _load_runtime_template()
    source = source.replace("__TINY_ARRAY_SIZE__", str(array_size))
    source = source.replace("__TINY_BUFFER_WORDS_32__", str(buffer_words_32))
    source = source.replace("__GENERATED_DECLS__", "\n\n".join(decls))
    source = source.replace("__GENERATED_MAIN__", "\n".join(main_lines))
    return source


def write_cv32e40p_c(
    artifact: CompiledArtifact,
    inputs: dict[str, np.ndarray],
    output_path: str | Path,
    *,
    program_name: str = "tinynpu_generated",
    defines_path: str | None = None,
    emit_cpu_baseline: bool = False,
    verify_cpu_baseline: bool = False,
    repeat_count: int = 1,
    cpu_only_baseline: bool = False,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        emit_cv32e40p_c(
            artifact,
            inputs,
            program_name=program_name,
            defines_path=defines_path,
            emit_cpu_baseline=emit_cpu_baseline,
            verify_cpu_baseline=verify_cpu_baseline,
            repeat_count=repeat_count,
            cpu_only_baseline=cpu_only_baseline,
        )
    )
    return output
