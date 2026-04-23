from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))

from tinynpu_jit.baremetal_emit import emit_cv32e40p_c  # noqa: E402
from tinynpu_jit.blocks import build_llama_decode_artifact, build_llama_prefill_artifact  # noqa: E402
from tinynpu_jit.rtl_runner import (  # noqa: E402
    CORE_DIR,
    CUSTOM_DIR,
    GENERATED_DIR,
    TNPU_RISCV_MABI,
    TNPU_RISCV_MARCH,
    run_checked,
    run_vlt_npu,
    toolchain_prefix,
    toolchain_root,
)


_METRIC_RE = re.compile(r"^([A-Za-z0-9_.]+) cycles=(\d+)$", re.MULTILINE)


def _build_c_elf_and_hex(program_name: str, source: str) -> tuple[Path, Path, Path]:
    GENERATED_DIR.mkdir(exist_ok=True)
    program_path = GENERATED_DIR / f"{program_name}.c"
    program_path.write_text(source)

    prefix = toolchain_prefix()
    gcc = f"{prefix}gcc"
    objcopy = f"{prefix}objcopy"
    root = toolchain_root(prefix)
    include_dir = root / "riscv32-unknown-elf" / "include"
    lib_dir = root / "riscv32-unknown-elf" / "lib"
    elf_path = CUSTOM_DIR / f"{program_name}.elf"
    hex_path = CUSTOM_DIR / f"{program_name}.hex"

    build_env = dict(os.environ)
    build_env["CCACHE_DISABLE"] = "1"
    build_env["TMPDIR"] = "/tmp"
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
        env=build_env,
    )
    run_checked([objcopy, "-O", "verilog", str(elf_path), str(hex_path)], cwd=CORE_DIR, env=build_env)
    return program_path, elf_path, hex_path


def _parse_metrics(stdout: str) -> dict[str, int]:
    metrics = {name: int(value) for name, value in _METRIC_RE.findall(stdout)}
    if any(name.startswith("repeat.") for name in metrics):
        return metrics

    preload_total = sum(value for name, value in metrics.items() if name.startswith("preload."))
    host_total = sum(value for name, value in metrics.items() if name.startswith("hostop."))
    segment_cpu_total = sum(
        value for name, value in metrics.items() if name.startswith("segment.") and name.endswith(".cpu")
    )
    segment_npu_total = sum(
        value for name, value in metrics.items() if name.startswith("segment.") and name.endswith(".npu")
    )
    if segment_cpu_total > 0 and segment_npu_total == 0:
        metrics["repeat.preload.total"] = preload_total
        metrics["repeat.host.shared.total"] = host_total
        metrics["repeat.segment.cpu.total"] = segment_cpu_total
        metrics["repeat.program.cpu.hot.avg"] = host_total + segment_cpu_total
        metrics["repeat.program.cpu.cold.total"] = preload_total + host_total + segment_cpu_total
    elif segment_npu_total > 0:
        metrics["repeat.preload.total"] = preload_total
        metrics["repeat.host.shared.total"] = host_total
        metrics["repeat.segment.npu.total"] = segment_npu_total
        metrics["repeat.program.npu.hot.avg"] = host_total + segment_npu_total
        metrics["repeat.program.npu.cold.total"] = preload_total + host_total + segment_npu_total
    return metrics


def _artifact_for_mode(
    mode: str,
    *,
    d_model: int,
    d_head: int,
    n_heads: int,
    n_kv_heads: int,
    ffn_hidden_dim: int,
    prompt_len: int,
    seed: int,
):
    if mode == "prefill":
        artifact, _, _ = build_llama_prefill_artifact(
            d_model=d_model,
            d_head=d_head,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            prompt_len=prompt_len,
            seed=seed,
        )
        return artifact
    if mode == "decode":
        artifact, _, _, _ = build_llama_decode_artifact(
            d_model=d_model,
            d_head=d_head,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            prompt_len=prompt_len,
            seed=seed,
        )
        return artifact
    raise ValueError(f"unsupported mode {mode}")


def _run_variant(
    *,
    artifact,
    program_name: str,
    repeat_count: int,
    cpu_only: bool,
    maxcycles: int,
    verilator_max_ticks: int,
    timeout_s: int,
    dump_stdout: bool,
) -> tuple[dict[str, int], str]:
    source = emit_cv32e40p_c(
        artifact,
        {},
        program_name=program_name,
        repeat_count=repeat_count,
        cpu_only_baseline=cpu_only,
    )
    _, _, hex_path = _build_c_elf_and_hex(program_name, source)
    proc = run_vlt_npu(
        hex_path,
        maxcycles=maxcycles,
        verilator_max_ticks=verilator_max_ticks,
        timeout_s=timeout_s,
        noassert=True,
    )
    stdout = proc.stdout
    if dump_stdout:
        print(stdout)
    return _parse_metrics(stdout), stdout


def _print_summary(mode: str, cpu_metrics: dict[str, int], npu_metrics: dict[str, int]) -> None:
    cpu_cold = cpu_metrics.get("repeat.program.cpu.cold.total")
    cpu_hot = cpu_metrics.get("repeat.program.cpu.hot.avg")
    npu_cold = npu_metrics.get("repeat.program.npu.cold.total")
    npu_hot = npu_metrics.get("repeat.program.npu.hot.avg")

    print(f"[{mode}]")
    if cpu_cold is not None:
        print(f"cpu.cold.total={cpu_cold}")
    if cpu_hot is not None:
        print(f"cpu.hot.avg={cpu_hot}")
    if npu_cold is not None:
        print(f"npu.cold.total={npu_cold}")
    if npu_hot is not None:
        print(f"npu.hot.avg={npu_hot}")
    if cpu_cold is not None and npu_cold is not None and npu_cold > 0:
        print(f"speedup.cold={cpu_cold / npu_cold:.2f}x")
    if cpu_hot is not None and npu_hot is not None and npu_hot > 0:
        print(f"speedup.hot={cpu_hot / npu_hot:.2f}x")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prefill", "decode", "both"], default="both")
    parser.add_argument("--variant", choices=["both", "npu", "cpu"], default="both")
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--d-head", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--n-kv-heads", type=int, default=1)
    parser.add_argument("--ffn-hidden-dim", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--prefill-maxcycles", type=int, default=2_000_000)
    parser.add_argument("--decode-maxcycles", type=int, default=1_000_000)
    parser.add_argument("--verilator-max-ticks", type=int, default=30_000_000_000)
    parser.add_argument("--timeout-s", type=int, default=300)
    parser.add_argument("--dump-stdout", action="store_true")
    args = parser.parse_args()

    build_env = dict(os.environ)
    build_env["CCACHE_DISABLE"] = "1"
    build_env["TMPDIR"] = "/tmp"
    run_checked(["make", "verilator-build-npu"], cwd=CORE_DIR, env=build_env)

    modes = ["prefill", "decode"] if args.mode == "both" else [args.mode]
    for mode in modes:
        artifact = _artifact_for_mode(
            mode,
            d_model=args.d_model,
            d_head=args.d_head,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            ffn_hidden_dim=args.ffn_hidden_dim,
            prompt_len=args.prompt_len,
            seed=args.seed,
        )
        maxcycles = args.prefill_maxcycles if mode == "prefill" else args.decode_maxcycles

        npu_metrics: dict[str, int] = {}
        cpu_metrics: dict[str, int] = {}

        if args.variant in {"both", "npu"}:
            npu_metrics, npu_stdout = _run_variant(
                artifact=artifact,
                program_name=f"cv32e40p_qllama_{mode}_npu_d{args.d_model}_h{args.d_head}_nh{args.n_heads}_nkv{args.n_kv_heads}_f{args.ffn_hidden_dim}_t{args.prompt_len}_s{args.seed}_r{args.repeat_count}",
                repeat_count=args.repeat_count,
                cpu_only=False,
                maxcycles=maxcycles,
                verilator_max_ticks=args.verilator_max_ticks,
                timeout_s=args.timeout_s,
                dump_stdout=args.dump_stdout,
            )
            if "EXIT SUCCESS" not in npu_stdout:
                raise RuntimeError(f"{mode} NPU run did not report EXIT SUCCESS")

        if args.variant in {"both", "cpu"}:
            cpu_metrics, cpu_stdout = _run_variant(
                artifact=artifact,
                program_name=f"cv32e40p_qllama_{mode}_cpu_d{args.d_model}_h{args.d_head}_nh{args.n_heads}_nkv{args.n_kv_heads}_f{args.ffn_hidden_dim}_t{args.prompt_len}_s{args.seed}_r{args.repeat_count}",
                repeat_count=args.repeat_count,
                cpu_only=True,
                maxcycles=maxcycles,
                verilator_max_ticks=args.verilator_max_ticks,
                timeout_s=args.timeout_s,
                dump_stdout=args.dump_stdout,
            )
            if "EXIT SUCCESS" not in cpu_stdout:
                raise RuntimeError(f"{mode} CPU-only run did not report EXIT SUCCESS")

        if args.variant == "both":
            _print_summary(mode, cpu_metrics, npu_metrics)
        elif args.variant == "npu":
            print(f"[{mode}]")
            for key in ("repeat.preload.total", "repeat.program.npu.hot.avg", "repeat.program.npu.cold.total"):
                if key in npu_metrics:
                    print(f"{key}={npu_metrics[key]}")
        else:
            print(f"[{mode}]")
            for key in ("repeat.preload.total", "repeat.program.cpu.hot.avg", "repeat.program.cpu.cold.total"):
                if key in cpu_metrics:
                    print(f"{key}={cpu_metrics[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
