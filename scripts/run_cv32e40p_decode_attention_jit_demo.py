from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))

from tinynpu_jit import (  # noqa: E402
    emit_cv32e40p_program_v2,
)
from tinynpu_jit.blocks.decode_attention import (  # noqa: E402
    build_artifact,
    build_artifact_legacy,
    build_artifact_via_builder,
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
    _sanitize,
    _toolchain_prefix,
    _toolchain_root,
)


def _parse_token_indices(raw: str | None, token_capacity: int) -> list[int]:
    if raw is None or raw.strip() == "":
        return [idx for idx in (1, 9) if idx < token_capacity]
    values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("token index list must not be empty")
    if len(set(values)) != len(values):
        raise ValueError("token indices must be unique")
    for idx in values:
        if idx < 0 or idx >= token_capacity:
            raise ValueError(f"token index {idx} is outside token capacity {token_capacity}")
    return values



def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--n-kv-heads", type=int, default=1)
    parser.add_argument("--d-head", type=int, default=8)
    parser.add_argument("--token-capacity", type=int, default=16)
    parser.add_argument("--token-indices", type=str, default="1,9")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    token_indices = _parse_token_indices(args.token_indices, args.token_capacity)
    artifact, expected, resolved_d_model = build_artifact(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_head=args.d_head,
        token_capacity=args.token_capacity,
        token_indices=token_indices,
        seed=args.seed,
    )

    program_name = (
        f"cv32e40p_decode_attention_dm{resolved_d_model}_nh{args.n_heads}_nkv{args.n_kv_heads}"
        f"_dh{args.d_head}_t{args.token_capacity}_n{len(token_indices)}_s{args.seed}_v2"
    )
    program_symbol = _sanitize(program_name)
    program_path = GENERATED_DIR / f"{program_name}_program.c"
    runner_path = GENERATED_DIR / f"{program_name}_runner.c"

    source = emit_cv32e40p_program_v2(artifact, {}, program_name=program_name)
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
            "+maxcycles=1000000",
        ],
        cwd=CORE_DIR,
        env=env,
        capture=True,
    )
    print(f"program={program_name}")
    print(
        f"d_model={resolved_d_model} n_heads={args.n_heads} n_kv_heads={args.n_kv_heads} "
        f"d_head={args.d_head} token_capacity={args.token_capacity} token_indices={token_indices} seed={args.seed}"
    )
    print(f"expected_checksum={float(expected.astype(np.float32).sum()):.6f}")
    print(proc.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
