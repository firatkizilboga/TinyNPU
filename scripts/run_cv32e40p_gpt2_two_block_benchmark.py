from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))

from run_cv32e40p_gpt2_two_block_reuse_jit_demo import build_artifact  # noqa: E402
from run_cv32e40p_qllama_block_benchmark import _build_c_elf_and_hex, _parse_metrics  # noqa: E402
from tinynpu_jit import emit_cv32e40p_program_v2  # noqa: E402
from tinynpu_jit.baremetal_emit import emit_cv32e40p_c  # noqa: E402
from tinynpu_jit.rtl_runner import CORE_DIR, RunnerConfig, build_v2_elf_and_hex, run_checked, run_vlt_npu  # noqa: E402


_V2_METRIC_RE = re.compile(r"^([A-Za-z0-9_.]+) cycles=(\d+)$", re.MULTILINE)


def _parse_v2_metrics(stdout: str) -> dict[str, int]:
    metrics = {name: int(value) for name, value in _V2_METRIC_RE.findall(stdout)}
    if "cold.e2e" in metrics:
        metrics["repeat.program.npu.cold.total"] = metrics["cold.e2e"]
    if "cold.body" in metrics:
        metrics["repeat.program.npu.hot.avg"] = metrics["cold.body"]
    if "warm.avg.body" in metrics:
        metrics["repeat.program.npu.hot.avg"] = metrics["warm.avg.body"]
    if "preload.total" in metrics:
        metrics["repeat.preload.total"] = metrics["preload.total"]
    return metrics


def _run_npu_v2(
    *,
    artifact,
    program_name: str,
    repeat_count: int,
    maxcycles: int,
    verilator_max_ticks: int,
    timeout_s: int,
    dump_stdout: bool,
) -> tuple[dict[str, int], str]:
    source = emit_cv32e40p_program_v2(artifact, {}, program_name=program_name)
    _, _, _, hex_path = build_v2_elf_and_hex(
        program_name,
        source,
        runner_config=RunnerConfig(
            repeat_count=repeat_count,
            dump_final_outputs=True,
            verbose_steps=False,
            timed=True,
        ),
    )
    proc = run_vlt_npu(
        hex_path,
        maxcycles=maxcycles,
        verilator_max_ticks=verilator_max_ticks,
        timeout_s=timeout_s,
        noassert=True,
    )
    if dump_stdout:
        print(proc.stdout)
    if "EXIT SUCCESS" not in proc.stdout:
        raise RuntimeError("GPT2 NPU run did not report EXIT SUCCESS")
    return _parse_v2_metrics(proc.stdout), proc.stdout


def _run_cpu_v1(
    *,
    artifact,
    program_name: str,
    repeat_count: int,
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
        cpu_only_baseline=True,
    )
    _, _, hex_path = _build_c_elf_and_hex(program_name, source)
    proc = run_vlt_npu(
        hex_path,
        maxcycles=maxcycles,
        verilator_max_ticks=verilator_max_ticks,
        timeout_s=timeout_s,
        noassert=True,
    )
    if dump_stdout:
        print(proc.stdout)
    if "EXIT SUCCESS" not in proc.stdout:
        raise RuntimeError("GPT2 CPU baseline run did not report EXIT SUCCESS")
    return _parse_metrics(proc.stdout), proc.stdout


def _print_summary(cpu_metrics: dict[str, int], npu_metrics: dict[str, int]) -> None:
    cpu_cold = cpu_metrics.get("repeat.program.cpu.cold.total")
    cpu_hot = cpu_metrics.get("repeat.program.cpu.hot.avg")
    npu_cold = npu_metrics.get("repeat.program.npu.cold.total")
    npu_hot = npu_metrics.get("repeat.program.npu.hot.avg")

    print("[gpt2_two_block_reuse]")
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
    parser.add_argument("--variant", choices=["both", "npu", "cpu"], default="both")
    parser.add_argument("--d-model", type=int, default=8)
    parser.add_argument("--d-head", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=1)
    parser.add_argument("--ffn-dim", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=4)
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--maxcycles", type=int, default=2_000_000)
    parser.add_argument("--verilator-max-ticks", type=int, default=30_000_000_000)
    parser.add_argument("--timeout-s", type=int, default=600)
    parser.add_argument("--dump-stdout", action="store_true")
    args = parser.parse_args()

    build_env = dict(os.environ)
    build_env["CCACHE_DISABLE"] = "1"
    build_env["TMPDIR"] = "/tmp"
    run_checked(["make", "verilator-build-npu"], cwd=CORE_DIR, env=build_env)

    artifact = build_artifact(
        d_model=args.d_model,
        d_head=args.d_head,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        prompt_len=args.prompt_len,
    )
    stem = (
        f"cv32e40p_gpt2_two_block_reuse_d{args.d_model}_h{args.d_head}_"
        f"nh{args.n_heads}_f{args.ffn_dim}_t{args.prompt_len}_r{args.repeat_count}"
    )

    npu_metrics: dict[str, int] = {}
    cpu_metrics: dict[str, int] = {}
    if args.variant in {"both", "npu"}:
        npu_metrics, _ = _run_npu_v2(
            artifact=artifact,
            program_name=f"{stem}_npu_v2",
            repeat_count=args.repeat_count,
            maxcycles=args.maxcycles,
            verilator_max_ticks=args.verilator_max_ticks,
            timeout_s=args.timeout_s,
            dump_stdout=args.dump_stdout,
        )
    if args.variant in {"both", "cpu"}:
        cpu_metrics, _ = _run_cpu_v1(
            artifact=artifact,
            program_name=f"{stem}_cpu_v1",
            repeat_count=args.repeat_count,
            maxcycles=args.maxcycles,
            verilator_max_ticks=args.verilator_max_ticks,
            timeout_s=args.timeout_s,
            dump_stdout=args.dump_stdout,
        )
    _print_summary(cpu_metrics, npu_metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
