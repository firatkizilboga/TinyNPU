from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))

from import_tinystories_to_qllama import _state_for_block, load_qllama_layer  # noqa: E402
from tinynpu_jit.baremetal_emit import emit_cv32e40p_c  # noqa: E402
from tinynpu_jit.blocks.llama_block import build_decode_artifact, build_prefill_artifact  # noqa: E402
from tinynpu_jit.rtl_runner import (  # noqa: E402
    CORE_DIR,
    CUSTOM_DIR,
    GENERATED_DIR,
    TNPU_RISCV_MABI,
    TNPU_RISCV_MARCH,
    run_checked,
    run_vlt_npu,
    toolchain_include_lib_dirs,
    toolchain_prefix,
)


_METRIC_RE = re.compile(r"^([A-Za-z0-9_.]+) cycles=(\d+)$", re.MULTILINE)


def _build_c_elf_and_hex(program_name: str, source: str) -> Path:
    GENERATED_DIR.mkdir(exist_ok=True)
    program_path = GENERATED_DIR / f"{program_name}.c"
    program_path.write_text(source)

    prefix = toolchain_prefix()
    gcc = f"{prefix}gcc"
    objcopy = f"{prefix}objcopy"
    include_dir, lib_dir = toolchain_include_lib_dirs(prefix)
    elf_path = CUSTOM_DIR / f"{program_name}.elf"
    hex_path = CUSTOM_DIR / f"{program_name}.hex"

    env = dict(os.environ)
    env["CCACHE_DISABLE"] = "1"
    env["TMPDIR"] = "/tmp"
    run_checked(["make", "verilator-build-npu"], cwd=CORE_DIR, env=env)
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
        env=env,
    )
    run_checked([objcopy, "-O", "verilog", str(elf_path), str(hex_path)], cwd=CORE_DIR, env=env)
    return hex_path


def _artifact_for_mode(args: argparse.Namespace):
    block, metadata = load_qllama_layer(args.run_dir, layer=args.layer, act_scale=args.act_scale)
    cfg = block.config
    state = _state_for_block(
        block,
        run_dir=args.run_dir,
        metadata=metadata,
        prompt_len=args.prompt_len,
        seed=args.seed,
        prompt=args.prompt,
    )
    kwargs = dict(
        d_model=cfg.d_model,
        d_head=cfg.d_head,
        n_heads=cfg.n_heads,
        n_kv_heads=cfg.n_kv_heads,
        ffn_hidden_dim=cfg.ffn_hidden_dim,
        prompt_len=args.prompt_len,
        act_scale=cfg.act_scale,
        attn_scale=cfg.attn_scale,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        state=state,
    )
    if args.mode == "prefill":
        artifact, _, ref = build_prefill_artifact(**kwargs)
    else:
        artifact, _, _, ref = build_decode_artifact(**kwargs)
    return artifact, ref, cfg, state, metadata


def _print_metrics(stdout: str, *, mode: str) -> None:
    metrics = {name: int(value) for name, value in _METRIC_RE.findall(stdout)}
    preload = sum(value for name, value in metrics.items() if name.startswith("preload."))
    host = sum(value for name, value in metrics.items() if name.startswith("hostop."))
    segment_npu = sum(value for name, value in metrics.items() if name.startswith("segment.") and name.endswith(".npu"))
    print(f"[{mode}]")
    print(f"preload.total={preload}")
    print(f"host.total={host}")
    print(f"segment.npu.total={segment_npu}")
    print(f"program.npu.hot={host + segment_npu}")
    print(f"program.npu.cold={preload + host + segment_npu}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/tinystories_word_lm_d32_t17"))
    parser.add_argument("--mode", choices=("prefill", "decode"), default="prefill")
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--prompt", default="there was a little")
    parser.add_argument("--prompt-len", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--act-scale", type=float, default=None)
    parser.add_argument("--maxcycles", type=int, default=1_500_000)
    parser.add_argument("--verilator-max-ticks", type=int, default=3_000_000_000)
    parser.add_argument("--timeout-s", type=int, default=420)
    parser.add_argument("--dump-stdout", action="store_true")
    args = parser.parse_args()

    artifact, ref, cfg, state, metadata = _artifact_for_mode(args)
    print(
        f"run_dir={args.run_dir} layer={args.layer} mode={args.mode} "
        f"d={cfg.d_model} h={cfg.d_head} nh={cfg.n_heads} nkv={cfg.n_kv_heads} "
        f"ffn={cfg.ffn_hidden_dim}"
    )
    print(f"prompt={args.prompt!r} token_ids={state['token_ids'].tolist()}")
    print(f"tokenizer_kind={metadata['tokenizer'].get('kind', 'char')} vocab={len(metadata['tokenizer']['itos'])}")  # type: ignore[index]
    print(f"reference_checksum={float(np.asarray(ref['out'], dtype=np.float32).sum()):.6f}")

    safe_prompt = re.sub(r"[^a-zA-Z0-9]+", "_", args.prompt).strip("_")[:40]
    program_name = f"cv32e40p_tinystories_{args.mode}_l{args.layer}_{safe_prompt}_t{args.prompt_len}"
    source = emit_cv32e40p_c(artifact, {}, program_name=program_name, repeat_count=1, cpu_only_baseline=False)
    hex_path = _build_c_elf_and_hex(program_name, source)
    try:
        proc = run_vlt_npu(
            hex_path,
            maxcycles=args.maxcycles,
            verilator_max_ticks=args.verilator_max_ticks,
            timeout_s=args.timeout_s,
            noassert=True,
        )
    except subprocess.CalledProcessError as exc:
        print(exc.stdout or "")
        print(exc.stderr or "", file=sys.stderr)
        raise
    if args.dump_stdout:
        print(proc.stdout)
    if "EXIT SUCCESS" not in proc.stdout:
        raise RuntimeError("RTL run did not report EXIT SUCCESS")
    _print_metrics(proc.stdout, mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
