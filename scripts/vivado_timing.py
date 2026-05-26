#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIVADO = Path("/opt/Xilinx/2025.2/Vivado/bin/vivado")
DEFAULT_PART = "xc7a200tsbg484-1"


def rtl_files() -> list[Path]:
    excluded = {"ub_skewer_wrapper.sv", "cv32e40p_tinynpu_synth_top.sv"}
    files = [p for p in sorted((ROOT / "rtl").glob("*.sv")) if p.name not in excluded]
    defines = ROOT / "rtl" / "defines.sv"
    return [defines] + [p for p in files if p != defines]


def cv32e40p_files() -> list[Path]:
    rtl = ROOT / "external" / "cv32e40p" / "rtl"
    bhv = ROOT / "external" / "cv32e40p" / "bhv"
    names = [
        "include/cv32e40p_apu_core_pkg.sv",
        "include/cv32e40p_fpu_pkg.sv",
        "include/cv32e40p_pkg.sv",
        "cv32e40p_if_stage.sv",
        "cv32e40p_cs_registers.sv",
        "cv32e40p_register_file_ff.sv",
        "cv32e40p_load_store_unit.sv",
        "cv32e40p_id_stage.sv",
        "cv32e40p_aligner.sv",
        "cv32e40p_decoder.sv",
        "cv32e40p_compressed_decoder.sv",
        "cv32e40p_fifo.sv",
        "cv32e40p_prefetch_buffer.sv",
        "cv32e40p_hwloop_regs.sv",
        "cv32e40p_mult.sv",
        "cv32e40p_int_controller.sv",
        "cv32e40p_ex_stage.sv",
        "cv32e40p_alu_div.sv",
        "cv32e40p_alu.sv",
        "cv32e40p_ff_one.sv",
        "cv32e40p_popcnt.sv",
        "cv32e40p_apu_disp.sv",
        "cv32e40p_controller.sv",
        "cv32e40p_obi_interface.sv",
        "cv32e40p_prefetch_controller.sv",
        "cv32e40p_sleep_unit.sv",
        "cv32e40p_core.sv",
        "cv32e40p_top.sv",
    ]
    return [rtl / name for name in names] + [bhv / "cv32e40p_sim_clock_gate.sv"]


def write_tcl(
    workdir: Path,
    *,
    target: str,
    part: str,
    clock_ns: float,
    extra_defines: list[str],
) -> Path:
    if target == "npu":
        top = "tinynpu_top"
        clock_port = "clk"
        files = rtl_files()
    elif target == "cpu":
        top = "cv32e40p_top"
        clock_port = "clk_i"
        files = cv32e40p_files()
    else:
        top = "cv32e40p_tinynpu_synth_top"
        clock_port = "clk_i"
        files = cv32e40p_files() + rtl_files() + [ROOT / "rtl" / "cv32e40p_tinynpu_synth_top.sv"]

    define_flags = " ".join(
        ["-define TINYNPU_FPGA_BRAM", "-define TINYNPU_VIVADO_BRAM"]
        + [f"-define {name}" for name in extra_defines]
    )
    read_flags = f"-sv {define_flags}"
    file_lines = "\n".join(f"read_verilog {read_flags} {{{path}}}" for path in files)
    report_dir = workdir / "reports"
    checkpoint_dir = workdir / "checkpoints"
    script = f"""
set_msg_config -id {{Common 17-55}} -new_severity {{WARNING}}
set_param general.maxThreads 30
set_property is_compile_unit_mode true [current_fileset]
file mkdir {{{report_dir}}}
file mkdir {{{checkpoint_dir}}}

{file_lines}

synth_design -top {top} -part {part} -mode out_of_context -flatten rebuilt
create_clock -period {clock_ns:.3f} [get_ports {clock_port}]
write_checkpoint -force {{{checkpoint_dir / "post_synth.dcp"}}}
report_utilization -file {{{report_dir / "post_synth_util.rpt"}}}

opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force {{{checkpoint_dir / "post_route.dcp"}}}
report_utilization -file {{{report_dir / "post_route_util.rpt"}}}
report_timing_summary -file {{{report_dir / "post_route_timing.rpt"}}}
report_timing -max_paths 20 -sort_by group -file {{{report_dir / "post_route_critical_paths.rpt"}}}
"""
    path = workdir / f"{target}_timing.tcl"
    path.write_text(script)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description="Run Vivado OOC implementation timing for TinyNPU targets.")
    ap.add_argument("target", choices=["npu", "cpu", "cpu-npu"])
    ap.add_argument("--part", default=DEFAULT_PART)
    ap.add_argument("--clock-ns", type=float, default=20.0)
    ap.add_argument("--workdir", default="")
    ap.add_argument(
        "--pipelined-pe-mac",
        action="store_true",
        help="Enable the optional PE product register and matching control wait.",
    )
    args = ap.parse_args()

    if not VIVADO.exists():
        raise SystemExit(f"Vivado not found: {VIVADO}")

    workdir = Path(args.workdir or (ROOT / "runs" / f"vivado_{args.target}")).resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    extra_defines = ["TINYNPU_PIPELINED_PE_MAC"] if args.pipelined_pe_mac else []
    tcl = write_tcl(
        workdir,
        target=args.target,
        part=args.part,
        clock_ns=args.clock_ns,
        extra_defines=extra_defines,
    )
    log = workdir / "vivado.log"
    cmd = [str(VIVADO), "-mode", "batch", "-source", str(tcl), "-log", str(log), "-journal", str(workdir / "vivado.jou")]
    print(" ".join(cmd))
    return subprocess.run(cmd, cwd=ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
