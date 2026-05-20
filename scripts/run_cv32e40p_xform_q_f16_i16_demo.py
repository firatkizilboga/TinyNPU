from __future__ import annotations

from pathlib import Path

import numpy as np

from run_cv32e40p_b_append_demo import (  # noqa: E402
    GENERATED_DIR,
    RunnerConfig,
    build_v2_elf_and_hex,
    _emit_i32_array,
    _emit_u32x4_image,
    _runner_source,
    _sanitize,
    run_vlt_npu,
)

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))

from tinynpu import TinyNPUProgram  # noqa: E402
from tinynpu.isa import PrecisionMode  # noqa: E402


def build_program() -> tuple[TinyNPUProgram, np.ndarray]:
    src_f32 = np.array(
        [
            [1.0, -2.0, 3.5, -4.25, 0.5, -0.5, 2.5, -1.5],
            [0.0, 1.0, -1.0, 2.0, 3.0, -3.0, 4.0, -4.0],
            [2.0, 2.0, 2.0, 2.0, -2.0, -2.0, -2.0, -2.0],
            [1.5, -1.5, 0.0, 0.0, 1.0, -1.0, 2.0, -2.0],
            [3.0, 0.0, -3.0, 1.0, -1.0, 2.0, -2.0, 4.0],
            [0.25, -0.25, 0.75, -0.75, 1.25, -1.25, 1.75, -1.75],
            [4.0, 3.0, 2.0, 1.0, -1.0, -2.0, -3.0, -4.0],
            [1.0, 0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )

    multiplier = 16
    shift = 0
    src_f16 = src_f32.astype(np.float16)
    expected_i16 = np.clip(np.rint(src_f16.astype(np.float32) * float(multiplier)), -32768, 32767).astype(np.int16)
    src_f16_bits = src_f16.view(np.uint16).view(np.int16)

    program = TinyNPUProgram()
    # Source stores raw FP16 bit patterns in 16-bit lanes.
    program.declare_data("src_f16_bits", src_f16_bits, precision=PrecisionMode.INT16, role="A")
    program.declare_data("dst_i16", np.zeros((8, 8), dtype=np.int16), precision=PrecisionMode.INT16, role="A")
    program.xform_q_f16_i16("src_f16_bits", "dst_i16", multiplier=multiplier, shift=shift)
    program.halt()
    return program, expected_i16


def _emit_program_source(program_name: str, program: TinyNPUProgram, expected: np.ndarray) -> str:
    symbol = _sanitize(program_name)
    binary = program.compile()
    dst = program.symbols["dst_i16"]

    decls: list[str] = []
    decls.append(_emit_i32_array(f"{symbol}_dst_data", np.zeros(expected.shape, dtype=np.int32), section_data=False))
    decls.append(_emit_i32_array(f"{symbol}_dst_expected_data", expected.astype(np.int32), section_data=True))
    decls.append(f"static const uint16_t {symbol}_output_indices[1] = {{0}};")
    decls.append(_emit_u32x4_image(f"{symbol}_ub_image", [int(word) for word in binary["ub"]]))

    im_words: list[int] = []
    for inst in binary["im"]:
        inst_int = int(inst)
        for chunk_idx in range(2):
            im_words.append((inst_int >> (chunk_idx * 128)) & ((1 << 128) - 1))
    decls.append(_emit_u32x4_image(f"{symbol}_im_image", im_words))

    decls.append(
        "static const TnpuImageLoad "
        f"{symbol}_ub_preloads[1] = {{\n"
        f'    {{.label = "preload.ub_image", .base_addr = 0u, .image = {symbol}_ub_image, .word_count = {len(binary["ub"])}}}\n'
        "};"
    )
    decls.append(
        "static const TnpuImageLoad "
        f"{symbol}_im_preloads[1] = {{\n"
        f'    {{.label = "preload.im_segment_000", .base_addr = 0x9000u, .image = {symbol}_im_image, .word_count = {len(im_words)}}}\n'
        "};"
    )
    decls.append(
        "static const TnpuTensorRead "
        f"{symbol}_seg_reads[1] = {{\n"
        f'    {{.tensor_idx = 0, .addr = 0x{dst.addr:04x}u, .precision = 2, .role = "A"}}\n'
        "};"
    )
    decls.append(
        "static const TnpuSegment "
        f"{symbol}_segments[1] = {{\n"
        f'    {{.name = "segment_000", .im_start_addr = 0x9000u, .writes = NULL, .write_count = 0u, .reads = {symbol}_seg_reads, .read_count = 1u}}\n'
        "};"
    )
    decls.append(
        f"static const TnpuTensorDesc {symbol}_tensors[2] = {{\n"
        f'    {{.name = "dst_i16", .data = {symbol}_dst_data, .dtype = TNPU_DTYPE_INT16, .rank = 2, .shape = {{8, 8, 1, 1}}, .elem_count = 64}},\n'
        f'    {{.name = "dst_i16_expected", .data = {symbol}_dst_expected_data, .dtype = TNPU_DTYPE_INT16, .rank = 2, .shape = {{8, 8, 1, 1}}, .elem_count = 64}}\n'
        "};"
    )
    decls.append(
        "static const TnpuVerifyOp "
        f"{symbol}_verify_ops[1] = {{\n"
        '    {.label = "xform_q_f16_i16", .actual_tensor_idx = 0, .expected_tensor_idx = 1, .is_final_output = 1}\n'
        "};"
    )
    decls.append(
        "static const TnpuOp "
        f"{symbol}_ops[4] = {{\n"
        "    {.kind = TNPU_OP_PRELOAD_UB, .index = 0u},\n"
        "    {.kind = TNPU_OP_PRELOAD_IM, .index = 0u},\n"
        "    {.kind = TNPU_OP_SEGMENT, .index = 0u},\n"
        "    {.kind = TNPU_OP_VERIFY, .index = 0u}\n"
        "};"
    )
    decls.append(
        f"const TnpuProgram {symbol} = {{\n"
        f'    .name = "{program_name}",\n'
        f"    .tensors = {symbol}_tensors,\n"
        "    .tensor_count = 2u,\n"
        "    .input_tensor_indices = NULL,\n"
        "    .input_count = 0u,\n"
        f"    .output_tensor_indices = {symbol}_output_indices,\n"
        "    .output_count = 1u,\n"
        f"    .ub_preloads = {symbol}_ub_preloads,\n"
        "    .ub_preload_count = 1u,\n"
        f"    .im_preloads = {symbol}_im_preloads,\n"
        "    .im_preload_count = 1u,\n"
        f"    .segments = {symbol}_segments,\n"
        "    .segment_count = 1u,\n"
        "    .host_ops = NULL,\n"
        "    .host_op_count = 0u,\n"
        f"    .verify_ops = {symbol}_verify_ops,\n"
        "    .verify_op_count = 1u,\n"
        f"    .ops = {symbol}_ops,\n"
        "    .op_count = 4u,\n"
        "};"
    )

    return (
        '#include "tinynpu_runtime_v2.h"\n'
        "#include <stddef.h>\n"
        "#include <stdint.h>\n\n"
        + "\n\n".join(decls)
        + "\n"
    )


def main() -> int:
    program_name = "cv32e40p_xform_q_f16_i16_demo_v2"
    program_symbol = _sanitize(program_name)
    program_path = GENERATED_DIR / f"{program_name}_program.c"
    runner_path = GENERATED_DIR / f"{program_name}_runner.c"

    program, expected = build_program()
    GENERATED_DIR.mkdir(exist_ok=True)
    program_path.write_text(_emit_program_source(program_name, program, expected))
    runner_path.write_text(_runner_source(program_symbol))

    _, _, _, hex_path = build_v2_elf_and_hex(
        program_name,
        program_path.read_text(),
        runner_config=RunnerConfig(dump_final_outputs=True, verbose_steps=True),
        extra_cflags=["-ffast-math", "-fno-builtin-printf"],
    )
    sim = run_vlt_npu(hex_path, maxcycles=2_000_000, noassert=True, timeout_s=240)
    print(sim.stdout, end="")
    if sim.stderr:
        print(sim.stderr, file=sys.stderr, end="")
    if "EXIT SUCCESS" not in sim.stdout:
        raise RuntimeError("simulation did not report EXIT SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
