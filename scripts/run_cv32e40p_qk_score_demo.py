from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from run_cv32e40p_b_append_demo import (  # noqa: E402
    CORE_DIR,
    CUSTOM_DIR,
    GENERATED_DIR,
    RUNTIME_DIR,
    REPO_ROOT,
    TNPU_RISCV_MABI,
    TNPU_RISCV_MARCH,
    _emit_i32_array,
    _emit_u32x4_image,
    _run,
    _runner_source,
    _sanitize,
    _toolchain_prefix,
    _toolchain_root,
)

import sys

if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))

from tinynpu import TinyNPUProgram  # noqa: E402
from tinynpu.isa import BReadMode, OutputLayout, PrecisionMode, WritebackMode  # noqa: E402


def build_program() -> tuple[TinyNPUProgram, np.ndarray]:
    lhs0 = np.array([[1, 2, -1, 0, 3, -2, 1, 4]], dtype=np.int16)
    rhs0 = np.array(
        [
            [1, 0, 1, 0, -1, 2, 0, 1, 2, -1, 0, 1, 1, 0, -1, 2],
            [0, 1, 0, 1, 2, -1, 1, 0, 1, 0, 2, -1, 0, 1, 2, 1],
            [1, -1, 1, 0, 0, 1, 2, 1, -1, 1, 0, 2, 1, 2, 0, -1],
            [2, 0, -1, 1, 1, 0, -1, 2, 0, 1, -1, 1, 2, 0, 1, 1],
            [0, 2, 1, -1, 1, 1, 0, 0, 2, 1, 1, 0, -1, 1, 0, 2],
            [1, 1, 0, 2, -1, 0, 1, -1, 1, 0, 1, 2, 0, -1, 1, 0],
            [0, -1, 2, 1, 0, 1, 1, 2, -1, 2, 1, 0, 1, 1, 2, 0],
            [1, 0, 1, 1, 2, 0, -1, 1, 0, 1, 1, -1, 1, 2, 0, 1],
        ],
        dtype=np.int16,
    )
    lhs1 = np.array([[2, 1, 0, -1, 1, 2, 0, 1]], dtype=np.int16)
    rhs1 = np.array(
        [
            [0, 1, 1, 0, 2, 1, 0, -1, 1, 0, 2, 1, -1, 0, 1, 2],
            [1, 0, 2, 1, -1, 0, 1, 2, 0, 2, 1, -1, 1, 1, 0, 1],
            [2, 1, 0, -1, 1, 2, 1, 0, 1, -1, 1, 2, 0, 1, 2, 1],
            [1, -1, 1, 2, 0, 1, 2, 1, -1, 1, 2, 0, 1, 2, 1, 0],
            [0, 2, 1, 0, 1, -1, 1, 2, 2, 1, 0, 1, -1, 1, 2, 0],
            [1, 1, -1, 1, 2, 0, 0, 1, 1, -1, 1, 0, 2, 1, 0, 1],
            [2, 0, 1, 1, 0, 2, -1, 1, 0, 1, 1, 2, 1, 0, 2, -1],
            [1, 2, 0, 1, -1, 1, 2, 0, 2, 0, 1, -1, 1, 2, 0, 1],
        ],
        dtype=np.int16,
    )
    query = np.array([[1, -1, 2, 0, 1, 3, -2, 1, 0, 2, 1, -1, 2, 0, 1, 1]], dtype=np.int16)

    token0 = np.clip(lhs0.astype(np.int32) @ rhs0.astype(np.int32), -32768, 32767).astype(np.int16)
    token1 = np.clip(lhs1.astype(np.int32) @ rhs1.astype(np.int32), -32768, 32767).astype(np.int16)
    expected_k = np.zeros((16, 16), dtype=np.int16)
    expected_k[:, 1] = token0[0]
    expected_k[:, 9] = token1[0]
    expected_scores = np.clip(query.astype(np.int32) @ expected_k.astype(np.int32), -32768, 32767).astype(np.int16)

    program = TinyNPUProgram()
    program.declare_data("lhs0", lhs0, precision=PrecisionMode.INT16, role="A")
    program.declare_data("rhs0", rhs0, precision=PrecisionMode.INT16, role="B")
    program.declare_data("lhs1", lhs1, precision=PrecisionMode.INT16, role="A")
    program.declare_data("rhs1", rhs1, precision=PrecisionMode.INT16, role="B")
    program.declare_data("query", query, precision=PrecisionMode.INT16, role="A")
    program.declare_data("k_cache", np.zeros((16, 16), dtype=np.int16), precision=PrecisionMode.INT16, role="B")
    program.declare_data("scores", np.zeros((1, 16), dtype=np.int16), precision=PrecisionMode.INT16, role="C")

    program.matmul(
        "lhs0",
        "rhs0",
        "k_cache",
        in_precision=PrecisionMode.INT16,
        out_precision=PrecisionMode.INT16,
        output_layout=OutputLayout.B,
        writeback_mode=WritebackMode.K_CACHE_APPEND_INT16,
        output_word_offset=1,
    )
    program.matmul(
        "lhs1",
        "rhs1",
        "k_cache",
        in_precision=PrecisionMode.INT16,
        out_precision=PrecisionMode.INT16,
        output_layout=OutputLayout.B,
        writeback_mode=WritebackMode.K_CACHE_APPEND_INT16,
        output_word_offset=17,
    )
    program.matmul(
        "query",
        "k_cache",
        "scores",
        in_precision=PrecisionMode.INT16,
        out_precision=PrecisionMode.INT16,
        output_layout=OutputLayout.C,
        b_read_mode=BReadMode.K_CACHE_INT16,
    )
    program.halt()
    return program, expected_scores


def _emit_program_source(program_name: str, program: TinyNPUProgram, expected_scores: np.ndarray) -> str:
    symbol = _sanitize(program_name)
    binary = program.compile()
    scores = program.symbols["scores"]

    decls: list[str] = []
    decls.append(_emit_i32_array(f"{symbol}_scores_data", np.zeros(expected_scores.shape, dtype=np.int32), section_data=False))
    decls.append(_emit_i32_array(f"{symbol}_scores_expected_data", expected_scores.astype(np.int32), section_data=True))
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
        f'    {{.label = "preload.im_segment_000", .base_addr = 0x8000u, .image = {symbol}_im_image, .word_count = {len(im_words)}}}\n'
        "};"
    )
    decls.append(
        "static const TnpuTensorRead "
        f"{symbol}_seg_reads[1] = {{\n"
        f'    {{.tensor_idx = 0, .addr = 0x{scores.addr:04x}u, .precision = 2, .role = "C"}}\n'
        "};"
    )
    decls.append(
        "static const TnpuSegment "
        f"{symbol}_segments[1] = {{\n"
        f'    {{.name = "segment_000", .im_start_addr = 0x8000u, .writes = NULL, .write_count = 0u, .reads = {symbol}_seg_reads, .read_count = 1u}}\n'
        "};"
    )
    decls.append(
        f"static const TnpuTensorDesc {symbol}_tensors[2] = {{\n"
        f'    {{.name = "scores", .data = {symbol}_scores_data, .dtype = TNPU_DTYPE_INT16, .rank = 2, .shape = {{1, 16, 1, 1}}, .elem_count = 16}},\n'
        f'    {{.name = "scores_expected", .data = {symbol}_scores_expected_data, .dtype = TNPU_DTYPE_INT16, .rank = 2, .shape = {{1, 16, 1, 1}}, .elem_count = 16}}\n'
        "};"
    )
    decls.append(
        "static const TnpuVerifyOp "
        f"{symbol}_verify_ops[1] = {{\n"
        '    {.label = "qk_score", .actual_tensor_idx = 0, .expected_tensor_idx = 1, .is_final_output = 1}\n'
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
        "#include <stddef.h>\n"
        "#include <stdint.h>\n"
        '#include "tinynpu_runtime_v2.h"\n\n'
        + "\n\n".join(decls)
        + "\n"
    )


def main() -> int:
    program_name = "cv32e40p_qk_score_demo_v2"
    program_symbol = _sanitize(program_name)
    program_path = GENERATED_DIR / f"{program_name}_program.c"
    runner_path = GENERATED_DIR / f"{program_name}_runner.c"

    program, expected_scores = build_program()
    source = _emit_program_source(program_name, program, expected_scores)
    GENERATED_DIR.mkdir(exist_ok=True)
    program_path.write_text(source)
    runner_path.write_text(_runner_source(program_symbol))

    prefix = _toolchain_prefix()
    gcc = f"{prefix}gcc"
    objcopy = f"{prefix}objcopy"
    toolchain_root = _toolchain_root(prefix)
    include_dir = toolchain_root / "riscv32-unknown-elf" / "include"
    lib_dir = toolchain_root / "riscv32-unknown-elf" / "lib"
    elf_path = CUSTOM_DIR / f"{program_name}.elf"
    hex_path = CUSTOM_DIR / f"{program_name}.hex"

    build_env = dict(os.environ)
    build_env["CCACHE_DISABLE"] = "1"
    build_env["TMPDIR"] = "/tmp"
    _run(["make", "verilator-build-npu"], cwd=CORE_DIR, env=build_env)
    _run(
        [
            gcc,
            f"-march={TNPU_RISCV_MARCH}",
            f"-mabi={TNPU_RISCV_MABI}",
            "-o",
            str(elf_path),
            "-w",
            "-O3",
            "-g",
            "-nostdlib",
            "-DTINYNPU_USE_SHARED_SRAM=1",
            "-DTNPU_RUNTIME_V2_DUMP_FINAL_OUTPUTS=1",
            "-T",
            "custom/link.ld",
            "-static",
            "custom/crt0.S",
            str(runner_path),
            str(program_path),
            str(RUNTIME_DIR / "tinynpu_runtime_v2.c"),
            "mem_stall/mem_stall.c",
            "custom/syscalls.c",
            "custom/vectors.S",
            "-I",
            str(include_dir),
            "-I",
            "mem_stall",
            "-I",
            str(RUNTIME_DIR),
            "-L",
            str(lib_dir),
            "-lc",
            "-lm",
            "-lgcc",
        ],
        cwd=CORE_DIR,
    )
    _run([objcopy, "-O", "verilog", str(elf_path), str(hex_path)], cwd=CORE_DIR)

    env = dict(os.environ)
    env["VERILATOR_MAX_TICKS"] = "3000000000"
    proc = _run(
        [
            str(CORE_DIR / "obj_dir" / "cv32e40p_tb_vlt_npu"),
            "+verilator+noassert",
            f"+firmware={hex_path}",
            "+maxcycles=500000",
        ],
        cwd=CORE_DIR,
        env=env,
        capture=True,
    )
    print(f"program={program_name}")
    print(f"expected_checksum={int(expected_scores.astype(np.int64).sum())}")
    print(proc.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
