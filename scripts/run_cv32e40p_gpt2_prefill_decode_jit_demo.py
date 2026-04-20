from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))

from tinynpu_jit import emit_cv32e40p_program_v2  # noqa: E402
from tinynpu_jit.blocks.gpt2_block import (  # noqa: E402
    build_decode_artifact,
    build_prefill_artifact,
)

from run_cv32e40p_b_append_demo import (  # noqa: E402
    CORE_DIR,
    CUSTOM_DIR,
    GENERATED_DIR,
    RUNTIME_DIR,
    TNPU_RISCV_MABI,
    TNPU_RISCV_MARCH,
    _run,
    _runner_source,
    _toolchain_prefix,
    _toolchain_root,
)


def _emit_and_build(artifact, *, program_name: str) -> Path:
    GENERATED_DIR.mkdir(exist_ok=True)
    program_path = GENERATED_DIR / f"{program_name}_program.c"
    runner_path = GENERATED_DIR / f"{program_name}_runner.c"
    source = emit_cv32e40p_program_v2(artifact, artifact.expected_tensors, program_name=program_name)
    program_path.write_text(source)
    runner_path.write_text(_runner_source(program_name))

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
        env=build_env,
    )
    _run([objcopy, "-O", "verilog", str(elf_path), str(hex_path)], cwd=CORE_DIR, env=build_env)
    return hex_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-head", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--emit", action="store_true")
    args = parser.parse_args()

    if args.mode == "prefill":
        artifact, _, ref = build_prefill_artifact(
            d_model=args.d_model,
            d_head=args.d_head,
            n_heads=args.n_heads,
            ffn_dim=args.ffn_dim,
            prompt_len=args.prompt_len,
            seed=args.seed,
        )
        print(f"mode=prefill segments={sorted(artifact.segment_artifacts.keys())}")
        print(f"prefill_checksum={float(np.asarray(ref['out'], dtype=np.float32).sum()):.6f}")
        if args.emit:
            program_name = f"cv32e40p_gpt2_prefill_d{args.d_model}_h{args.d_head}_nh{args.n_heads}_f{args.ffn_dim}_t{args.prompt_len}_s{args.seed}_v2"
            hex_path = _emit_and_build(artifact, program_name=program_name)
            print(f"hex={hex_path}")
    else:
        artifact, _, _, ref = build_decode_artifact(
            d_model=args.d_model,
            d_head=args.d_head,
            n_heads=args.n_heads,
            ffn_dim=args.ffn_dim,
            prompt_len=args.prompt_len,
            seed=args.seed,
        )
        print(f"mode=decode segments={sorted(artifact.segment_artifacts.keys())}")
        print(f"decode_checksum={float(np.asarray(ref['out'], dtype=np.float32).sum()):.6f}")
        if args.emit:
            program_name = f"cv32e40p_gpt2_decode_d{args.d_model}_h{args.d_head}_nh{args.n_heads}_f{args.ffn_dim}_t{args.prompt_len}_s{args.seed}_v2"
            hex_path = _emit_and_build(artifact, program_name=program_name)
            print(f"hex={hex_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
