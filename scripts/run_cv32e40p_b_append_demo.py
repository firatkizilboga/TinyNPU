from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))

from tinynpu import TinyNPUProgram  # noqa: E402
from tinynpu.isa import OutputLayout, PrecisionMode  # noqa: E402


CORE_DIR = REPO_ROOT / "external" / "cv32e40p" / "example_tb" / "core"
CUSTOM_DIR = CORE_DIR / "custom"
GENERATED_DIR = REPO_ROOT / "generated"
RUNTIME_DIR = REPO_ROOT / "software" / "compiler" / "tinynpu_jit"
TNPU_RISCV_MARCH = os.environ.get("TINYNPU_RISCV_MARCH", "rv32imfc")
TNPU_RISCV_MABI = os.environ.get("TINYNPU_RISCV_MABI", "ilp32f")


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def _runner_source(program_symbol: str) -> str:
    return f"""#include <stdlib.h>
#include <stdint.h>
#include "tinynpu_runtime_v2.h"

extern const TnpuProgram {program_symbol};

int main(void)
{{
    const TnpuProgram *program = &{program_symbol};
    TnpuTensor ins[8];
    const TnpuTensor *ip[8];
    TnpuTensor outs[8];
    const TnpuTensor *op[8];
    if (program->input_count > 8u) return EXIT_FAILURE;
    if (program->output_count > 8u) return EXIT_FAILURE;
    for (uint32_t i = 0; i < program->input_count; ++i) {{
        uint16_t t = program->input_tensor_indices[i];
        ins[i].data = program->tensors[t].data;
        ins[i].desc = &program->tensors[t];
        ins[i].elem_count = program->tensors[t].elem_count;
        ip[i] = &ins[i];
    }}
    for (uint32_t i = 0; i < program->output_count; ++i) {{
        uint16_t t = program->output_tensor_indices[i];
        outs[i].data = program->tensors[t].data;
        outs[i].desc = &program->tensors[t];
        outs[i].elem_count = program->tensors[t].elem_count;
        op[i] = &outs[i];
    }}
    return tinynpu_run(program, ip, op, NULL, 0u);
}}
"""


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


def _emit_i32_array(name: str, values: np.ndarray, *, section_data: bool) -> str:
    flat = values.reshape(-1)
    attr = ' __attribute__((section(".data")))' if section_data else ' __attribute__((section(".noinit")))'
    if flat.size == 0:
        return f"static int32_t {name}[1]{attr} = {{0}};"
    if section_data:
        body = ", ".join(str(int(v)) for v in flat)
        return f"static int32_t {name}[{flat.size}]{attr} = {{\n    {body}\n}};"
    return f"static int32_t {name}[{flat.size}]{attr};"


def _toolchain_prefix() -> Path:
    prefix_override = os.environ.get("TINYNPU_RISCV_PREFIX")
    if prefix_override:
        prefix = Path(prefix_override)
        gcc = prefix.parent / (prefix.name + "gcc")
        if not gcc.exists():
            raise FileNotFoundError(f"{gcc} not found for TINYNPU_RISCV_PREFIX")
        return prefix
    preferred = Path("/opt/riscv-ilp32f/bin/riscv32-unknown-elf-gcc")
    if preferred.exists():
        return preferred.parent / "riscv32-unknown-elf-"
    gcc = shutil.which("riscv32-unknown-elf-gcc")
    if gcc is None:
        raise FileNotFoundError("riscv32-unknown-elf-gcc not found in PATH")
    return Path(gcc).resolve().parent / "riscv32-unknown-elf-"


def _toolchain_root(prefix: Path) -> Path:
    return prefix.parent.parent


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        check=True,
        text=True,
        capture_output=capture,
    )


def build_program() -> tuple[TinyNPUProgram, np.ndarray]:
    lhs0 = np.array(
        [
            [1, 2, -1, 0, 3, -2, 1, 4],
            [0, -1, 2, 3, -2, 1, 0, 2],
            [3, 1, 0, -1, 2, 2, -3, 1],
            [2, 0, 1, 1, -1, 3, 2, -2],
            [-1, 2, 3, 0, 1, -1, 2, 1],
            [4, -2, 1, 2, 0, 1, -1, 3],
            [1, 1, -2, 4, 2, 0, 3, -1],
            [0, 3, 2, -1, 1, 2, 1, 0],
        ],
        dtype=np.int16,
    )
    rhs0 = np.array(
        [
            [1, 0, 1, 0, -1, 2, 0, 1],
            [0, 1, 0, 1, 2, -1, 1, 0],
            [1, -1, 1, 0, 0, 1, 2, 1],
            [2, 0, -1, 1, 1, 0, -1, 2],
            [0, 2, 1, -1, 1, 1, 0, 0],
            [1, 1, 0, 2, -1, 0, 1, -1],
            [0, -1, 2, 1, 0, 1, 1, 2],
            [1, 0, 1, 1, 2, 0, -1, 1],
        ],
        dtype=np.int16,
    )
    lhs1 = np.array(
        [
            [2, 1, 0, -1, 1, 2, 0, 1],
            [1, 0, 2, 1, -1, 0, 1, 2],
            [0, 1, 1, 2, 0, -1, 2, 1],
            [1, 2, -1, 0, 2, 1, 1, 0],
            [2, -1, 1, 1, 0, 2, -1, 1],
            [0, 2, 1, -1, 1, 0, 2, 1],
            [1, 1, 0, 2, 1, -1, 0, 2],
            [2, 0, 1, 1, -1, 1, 2, 0],
        ],
        dtype=np.int16,
    )
    rhs1 = np.array(
        [
            [0, 1, 1, 0, 2, 1, 0, -1],
            [1, 0, 2, 1, -1, 0, 1, 2],
            [2, 1, 0, -1, 1, 2, 1, 0],
            [1, -1, 1, 2, 0, 1, 2, 1],
            [0, 2, 1, 0, 1, -1, 1, 2],
            [1, 1, -1, 1, 2, 0, 0, 1],
            [2, 0, 1, 1, 0, 2, -1, 1],
            [1, 2, 0, 1, -1, 1, 2, 0],
        ],
        dtype=np.int16,
    )

    token0 = np.clip(lhs0.astype(np.int32) @ rhs0.astype(np.int32), -32768, 32767).astype(np.int16)
    token1 = np.clip(lhs1.astype(np.int32) @ rhs1.astype(np.int32), -32768, 32767).astype(np.int16)
    expected_cache = np.vstack([token0, token1]).astype(np.int16)

    program = TinyNPUProgram()
    program.declare_data("lhs0", lhs0, precision=PrecisionMode.INT16, role="A")
    program.declare_data("rhs0", rhs0, precision=PrecisionMode.INT16, role="B")
    program.declare_data("lhs1", lhs1, precision=PrecisionMode.INT16, role="A")
    program.declare_data("rhs1", rhs1, precision=PrecisionMode.INT16, role="B")
    program.declare_data("cache", np.zeros((16, 8), dtype=np.int16), precision=PrecisionMode.INT16, role="B")

    stride_words = 8
    program.matmul(
        "lhs0",
        "rhs0",
        "cache",
        in_precision=PrecisionMode.INT16,
        out_precision=PrecisionMode.INT16,
        output_layout=OutputLayout.B,
        output_word_offset=0,
    )
    program.matmul(
        "lhs1",
        "rhs1",
        "cache",
        in_precision=PrecisionMode.INT16,
        out_precision=PrecisionMode.INT16,
        output_layout=OutputLayout.B,
        output_word_offset=stride_words,
    )
    program.halt()
    return program, expected_cache


def _emit_program_source(program_name: str, program: TinyNPUProgram, expected_cache: np.ndarray) -> str:
    symbol = _sanitize(program_name)
    binary = program.compile()
    cache = program.symbols["cache"]

    decls: list[str] = []
    decls.append(_emit_i32_array(f"{symbol}_cache_data", np.zeros(expected_cache.shape, dtype=np.int32), section_data=False))
    decls.append(_emit_i32_array(f"{symbol}_cache_expected_data", expected_cache.astype(np.int32), section_data=True))
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
        f'    {{.tensor_idx = 0, .addr = 0x{cache.addr:04x}u, .precision = 2, .role = "B"}}\n'
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
        f'    {{.name = "cache", .data = {symbol}_cache_data, .dtype = TNPU_DTYPE_INT16, .rank = 2, .shape = {{16, 8, 1, 1}}, .elem_count = 128}},\n'
        f'    {{.name = "cache_expected", .data = {symbol}_cache_expected_data, .dtype = TNPU_DTYPE_INT16, .rank = 2, .shape = {{16, 8, 1, 1}}, .elem_count = 128}}\n'
        "};"
    )
    decls.append(
        "static const TnpuVerifyOp "
        f"{symbol}_verify_ops[1] = {{\n"
        '    {.label = "cache_append", .actual_tensor_idx = 0, .expected_tensor_idx = 1, .is_final_output = 1}\n'
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
    program_name = "cv32e40p_b_append_demo_v2"
    program_symbol = _sanitize(program_name)
    program_path = GENERATED_DIR / f"{program_name}_program.c"
    runner_path = GENERATED_DIR / f"{program_name}_runner.c"

    program, expected_cache = build_program()
    GENERATED_DIR.mkdir(exist_ok=True)
    program_path.write_text(_emit_program_source(program_name, program, expected_cache))
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
            "+maxcycles=250000",
        ],
        cwd=CORE_DIR,
        env=env,
        capture=True,
    )
    print(f"program={program_name}")
    print(f"cache_addr=0x{program.symbols['cache'].addr:04x}")
    print(f"expected_checksum={int(expected_cache.astype(np.int64).sum())}")
    print(proc.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
