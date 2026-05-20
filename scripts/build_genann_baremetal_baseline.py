from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))

from tinynpu_jit.rtl_runner import (  # noqa: E402
    CORE_DIR,
    CUSTOM_DIR,
    GENERATED_DIR,
    RUNTIME_DIR,
    TNPU_RISCV_MABI,
    TNPU_RISCV_MARCH,
    run_checked,
    run_vlt_npu,
    toolchain_include_lib_dirs,
    toolchain_prefix,
)


GENANN_DIR = REPO_ROOT / "external" / "third_party" / "genann"


def _fmt_f64(values: np.ndarray, *, per_line: int = 6) -> str:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)

    def lit(value: np.float64) -> str:
        text = f"{float(value):.17g}"
        if "e" not in text and "." not in text:
            text += ".0"
        return text

    lines: list[str] = []
    for start in range(0, flat.size, per_line):
        chunk = flat[start : start + per_line]
        lines.append("    " + ", ".join(lit(v) for v in chunk))
    return ",\n".join(lines)


def _load_mnist_sample_flat(sample_index: int = 0, image_size: int = 8) -> tuple[np.ndarray, int]:
    from software.workload.mnist_mlp_feature_benchmark import TASK_IS_ZERO, get_flat_mnist_loaders

    _, _, _, _, test_ds = get_flat_mnist_loaders(str(REPO_ROOT / "data"), image_size=image_size, task=TASK_IS_ZERO)
    image, label = test_ds[sample_index]
    return image.numpy().astype(np.float64), int(label)


def render_genann_runner(*, sample_index: int = 0) -> str:
    x, label = _load_mnist_sample_flat(sample_index=sample_index, image_size=8)
    input_decl = f"static const double input_x[64] = {{\n{_fmt_f64(x)}\n}};"
    return f"""// Third-party bare-metal neural-network baseline using genann.
// genann: https://github.com/codeplea/genann
// This program intentionally bypasses the TinyNPU compiler and runtime.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "genann.h"

uint32_t runtime_cycle_start;
uint32_t runtime_cycle_post_bss;
uint32_t runtime_cycle_post_init;
uint32_t runtime_cycle_pre_main;

#define TIMER_CTRL  ((volatile uint32_t *) 0x15000000u)
#define TIMER_VALUE ((volatile uint32_t *) 0x15000004u)
#define TIMER_COUNT ((volatile uint32_t *) 0x15001000u)

static inline uint32_t read_mcycle32(void) {{ return *TIMER_COUNT; }}

static void reset_timer(void)
{{
    *TIMER_CTRL = 0u;
    *TIMER_VALUE = 0xFFFFFFFFu;
    while (*TIMER_COUNT == 0u) {{ }}
}}

{input_decl}

static double deterministic_weight(int idx)
{{
    return ((double)((idx * 37 + 11) % 29) - 14.0) * 0.015;
}}

static double ann_weights[(64 + 1) * 16 + (16 + 1) * 1];
static double ann_outputs[64 + 16 + 1];
static double ann_deltas[16 + 1];
static genann ann;

static void init_ann(void)
{{
    ann.inputs = 64;
    ann.hidden_layers = 1;
    ann.hidden = 16;
    ann.outputs = 1;
    ann.total_weights = (64 + 1) * 16 + (16 + 1) * 1;
    ann.total_neurons = 64 + 16 + 1;
    ann.weight = ann_weights;
    ann.output = ann_outputs;
    ann.delta = ann_deltas;
    ann.activation_hidden = genann_act_sigmoid;
    ann.activation_output = genann_act_sigmoid;
    for (int i = 0; i < ann.total_weights; ++i) {{
        ann.weight[i] = deterministic_weight(i);
    }}
}}

int main(void)
{{
    init_ann();

    reset_timer();
    uint32_t t0 = read_mcycle32();
    const double *out = genann_run(&ann, input_x);
    uint32_t t1 = read_mcycle32();
    uint32_t cycles = t0 - t1;

    printf("third_party_genann_mlp sample_index={sample_index} label={label} cycles=%lu output=%.12g pred=%d weights=%d\\n",
           (unsigned long)cycles, out[0], out[0] >= 0.5, ann.total_weights);
    puts("EXIT SUCCESS");
    return 0;
}}
"""


def build_elf_and_hex(program_name: str, source: str) -> tuple[Path, Path, Path]:
    if not (GENANN_DIR / "genann.c").exists() or not (GENANN_DIR / "genann.h").exists():
        raise FileNotFoundError(f"genann source not found under {GENANN_DIR}")
    GENERATED_DIR.mkdir(exist_ok=True)
    CUSTOM_DIR.mkdir(exist_ok=True)
    source_path = GENERATED_DIR / f"{program_name}.c"
    source_path.write_text(source)

    prefix = toolchain_prefix()
    gcc = f"{prefix}gcc"
    objcopy = f"{prefix}objcopy"
    include_dir, lib_dir = toolchain_include_lib_dirs(prefix)
    elf_path = CUSTOM_DIR / f"{program_name}.elf"
    hex_path = CUSTOM_DIR / f"{program_name}.hex"
    build_env = dict(os.environ)
    build_env["CCACHE_DISABLE"] = "1"
    build_env["TMPDIR"] = "/tmp"
    run_checked(["make", "verilator-build-npu"], cwd=CORE_DIR, env=build_env)
    run_checked(
        [
            gcc,
            f"-march={TNPU_RISCV_MARCH}",
            f"-mabi={TNPU_RISCV_MABI}",
            "-o",
            str(elf_path),
            "-w",
            "-O3",
            "-g",
            "-DNDEBUG",
            "-nostdlib",
            "-T",
            "custom/link.ld",
            "-static",
            "custom/crt0.S",
            str(source_path),
            str(GENANN_DIR / "genann.c"),
            "mem_stall/mem_stall.c",
            "custom/syscalls.c",
            "custom/vectors.S",
            "-I",
            str(include_dir),
            "-I",
            "mem_stall",
            "-I",
            str(RUNTIME_DIR),
            "-I",
            str(GENANN_DIR),
            "-L",
            str(lib_dir),
            "-lc",
            "-lm",
            "-lgcc",
        ],
        cwd=CORE_DIR,
        env=build_env,
    )
    run_checked([objcopy, "-O", "verilog", str(elf_path), str(hex_path)], cwd=CORE_DIR, env=build_env)
    return source_path, elf_path, hex_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--program-name", default="third_party_genann_mlp")
    parser.add_argument("--run-rtl", action="store_true")
    parser.add_argument("--maxcycles", type=int, default=20_000_000)
    parser.add_argument("--verilator-max-ticks", type=int, default=20_000_000_000)
    parser.add_argument("--timeout-s", type=int, default=300)
    args = parser.parse_args()

    source = render_genann_runner(sample_index=args.sample_index)
    source_path, elf_path, hex_path = build_elf_and_hex(args.program_name, source)
    print(f"source={source_path}")
    print(f"elf={elf_path}")
    print(f"hex={hex_path}")
    if args.run_rtl:
        try:
            proc = run_vlt_npu(
                hex_path,
                maxcycles=args.maxcycles,
                verilator_max_ticks=args.verilator_max_ticks,
                timeout_s=args.timeout_s,
                noassert=True,
            )
        except subprocess.TimeoutExpired as exc:
            if exc.stdout:
                print(exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout)
            if exc.stderr:
                print(exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr, file=sys.stderr)
            raise
        print(proc.stdout)
        if proc.stderr:
            print(proc.stderr, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
