#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Check:
    name: str
    cmd: list[str]
    cwd: Path = ROOT
    timeout_s: int = 300


def _run(check: Check) -> tuple[bool, float, str]:
    start = time.monotonic()
    try:
        proc = subprocess.run(
            check.cmd,
            cwd=check.cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=check.timeout_s,
            check=False,
        )
        elapsed = time.monotonic() - start
        output = proc.stdout
        failed_test = " TESTS=" in output and " FAIL=0 " not in output
        return proc.returncode == 0 and not failed_test, elapsed, output
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - start
        output = exc.stdout or ""
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        return False, elapsed, output + f"\nTIMEOUT after {check.timeout_s}s\n"


def _static_rtl_guards() -> tuple[bool, str]:
    rtl_files = sorted((ROOT / "rtl").glob("*.sv"))
    text_by_file = {path: path.read_text() for path in rtl_files}
    failures: list[str] = []

    for path, text in text_by_file.items():
        if "negedge clk" in text:
            failures.append(f"{path.relative_to(ROOT)} uses negedge clk")

    for path, text in text_by_file.items():
        if "results_flat" in text:
            failures.append(f"{path.relative_to(ROOT)} still exposes results_flat")

    if "XFORM_MODE_ROPE_K16" not in (ROOT / "rtl" / "defines.sv").read_text():
        failures.append("defines.sv no longer declares XFORM_MODE_ROPE_K16; update acceptance policy")

    if failures:
        return False, "\n".join(failures)
    return True, "static RTL guards passed"


def _checks(*, include_integration: bool, include_qllama: bool) -> list[Check]:
    cocotb = ROOT / "verification" / "cocotb"
    checks = [
        Check(
            "python_syntax",
            [sys.executable, "-m", "py_compile", "scripts/synth_tinynpu_yosys.py", "scripts/synth_asic_yosys.py"],
            timeout_s=30,
        ),
        Check(
            "npu_coarse_asic_synth",
            [sys.executable, "scripts/synth_asic_yosys.py", "npu", "--coarse-only", "--skip-abc"],
            timeout_s=120,
        ),
        Check(
            "ppu_unit",
            [
                "make",
                "-f",
                "Makefile.npu",
                "SIM_BUILD=sim_build_accept_ppu",
                "TOPLEVEL=ppu",
                "MODULE=test_ppu_unit",
                "CCACHE_DISABLE=1",
            ],
            cwd=cocotb,
            timeout_s=300,
        ),
        Check(
            "rtl_multitile_matmul",
            [
                "make",
                "-f",
                "Makefile.npu",
                "SIM_BUILD=sim_build_accept_multitile",
                "TOPLEVEL=tinynpu_top",
                "MODULE=test_jit_multitile_matmul",
                "CCACHE_DISABLE=1",
            ],
            cwd=cocotb,
            timeout_s=300,
        ),
        Check(
            "xform_q_f16_i16_shared",
            [
                "make",
                "-f",
                "Makefile.npu",
                "SIM_BUILD=sim_build_accept_xform",
                "MODULE=test_xform_shared",
                "CCACHE_DISABLE=1",
            ],
            cwd=cocotb,
            timeout_s=300,
        ),
    ]
    if include_integration:
        checks.extend(
            [
                Check(
                    "rtl_is_zero_mlp",
                    [
                        "make",
                        "-f",
                        "Makefile.npu",
                        "SIM_BUILD=sim_build_accept_iszero",
                        "MODULE=test_jit_iszero_mlp_runtime",
                        "CCACHE_DISABLE=1",
                    ],
                    cwd=cocotb,
                    timeout_s=300,
                ),
                Check(
                    "rtl_shared_sram_move",
                    [
                        "make",
                        "-f",
                        "Makefile.npu",
                        "SIM_BUILD=sim_build_accept_shared",
                        "MODULE=test_shared_sram_move",
                        "CCACHE_DISABLE=1",
                    ],
                    cwd=cocotb,
                    timeout_s=300,
                ),
            ]
        )
    if include_qllama:
        checks.append(
        Check(
            "rtl_qllama_decode_tiny",
            [
                sys.executable,
                "scripts/run_cv32e40p_qllama_block_benchmark.py",
                "--mode",
                "decode",
                "--variant",
                "both",
                "--d-model",
                "8",
                "--d-head",
                "8",
                "--n-heads",
                "1",
                "--n-kv-heads",
                "1",
                "--ffn-hidden-dim",
                "8",
                "--prompt-len",
                "1",
                "--repeat-count",
                "1",
                "--timeout-s",
                "300",
                "--decode-maxcycles",
                "500000",
            ],
            timeout_s=420,
        )
    )
    return checks


def main() -> int:
    ap = argparse.ArgumentParser(description="Run synthability acceptance checks for TinyNPU RTL changes.")
    ap.add_argument("--include-integration", action="store_true", help="Also run is_zero and shared-SRAM integration RTL checks.")
    ap.add_argument("--include-qllama", action="store_true", help="Also run the slow tiny QLlama CV32E40P RTL smoke.")
    ap.add_argument("--stop-on-fail", action="store_true", help="Stop after the first failed check.")
    args = ap.parse_args()

    print("== static_rtl_guards ==")
    ok, output = _static_rtl_guards()
    print(output)
    failures = 0 if ok else 1
    if not ok and args.stop_on_fail:
        return 1

    for check in _checks(include_integration=args.include_integration, include_qllama=args.include_qllama):
        print(f"\n== {check.name} ==")
        print("$ " + " ".join(check.cmd))
        ok, elapsed, output = _run(check)
        status = "PASS" if ok else "FAIL"
        print(f"{status} {elapsed:.1f}s")
        if output.strip():
            tail = "\n".join(output.rstrip().splitlines()[-80:])
            print(tail)
        if not ok:
            failures += 1
            if args.stop_on_fail:
                break

    if failures:
        print(f"\nACCEPTANCE FAIL: {failures} check(s) failed")
        return 1
    print("\nACCEPTANCE PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
