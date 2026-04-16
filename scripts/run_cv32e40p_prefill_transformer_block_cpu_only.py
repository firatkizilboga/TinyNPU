from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))

from tinynpu_jit import write_cv32e40p_c  # noqa: E402

from run_cv32e40p_b_append_demo import (  # noqa: E402
    CORE_DIR,
    CUSTOM_DIR,
    GENERATED_DIR,
    TNPU_RISCV_MABI,
    TNPU_RISCV_MARCH,
    _run,
    _toolchain_prefix,
    _toolchain_root,
)
from run_cv32e40p_prefill_transformer_block_jit_demo import build_artifact  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--d-head", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=2)
    parser.add_argument("--ffn-dim", type=int, default=8)
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    artifact, expected = build_artifact(
        d_model=args.d_model,
        d_head=args.d_head,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        token_count=args.token_count,
        seed=args.seed,
    )

    program_name = (
        f"cv32e40p_prefill_transformer_block_cpu_only_d{args.d_model}"
        f"_h{args.d_head}_nh{args.n_heads}_f{args.ffn_dim}_t{args.token_count}_s{args.seed}"
    )
    program_path = GENERATED_DIR / f"{program_name}.c"
    GENERATED_DIR.mkdir(exist_ok=True)
    write_cv32e40p_c(
        artifact,
        {},
        program_path,
        program_name=program_name,
        cpu_only_baseline=True,
    )

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
            "-T",
            "custom/link.ld",
            "-static",
            "custom/crt0.S",
            str(program_path),
            "mem_stall/mem_stall.c",
            "custom/syscalls.c",
            "custom/vectors.S",
            "-I",
            str(include_dir),
            "-I",
            "mem_stall",
            "-L",
            str(lib_dir),
            "-lc",
            "-lm",
            "-lgcc",
        ],
        cwd=CORE_DIR,
        env=build_env,
    )
    _run([objcopy, "-O", "verilog", str(elf_path), str(hex_path)], cwd=CORE_DIR, env=build_env)

    env = dict(os.environ)
    env["VERILATOR_MAX_TICKS"] = "30000000000"
    proc = _run(
        [
            str(CORE_DIR / "obj_dir" / "cv32e40p_tb_vlt_npu"),
            "+verilator+noassert",
            f"+firmware={hex_path}",
            "+maxcycles=3000000",
        ],
        cwd=CORE_DIR,
        env=env,
        capture=True,
    )
    print(f"program={program_name}")
    print(
        f"d_model={args.d_model} d_head={args.d_head} n_heads={args.n_heads} "
        f"ffn_dim={args.ffn_dim} token_count={args.token_count} seed={args.seed}"
    )
    print(f"expected_checksum={float(np.array(expected, dtype=np.float32).sum()):.6f}")
    print(proc.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
