#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RTL_DIR = ROOT / "rtl"

PPU_DEBUG_RE = re.compile(
    r"\n\s*always_ff @\(posedge clk\) begin\n\s*if \(rst_n && capture_en\) begin\n.*?\n\s*end\n\s*end\n",
    re.S,
)
TRACE_INIT_RE = re.compile(
    r"\n\s*initial begin\n\s*if \(\$test\$plusargs\(\"trace\"\)\) begin\n.*?\n\s*end\n\s*end\n",
    re.S,
)

UNIFIED_BUFFER_BLACKBOX = '''`include "defines.sv"
(* blackbox *) module unified_buffer #(parameter INIT_FILE = "") (
    input logic clk,
    input logic rst_n,
    input logic                     wr_en,
    input logic [`BUFFER_WIDTH-1:0] wr_mask,
    input logic [  `ADDR_WIDTH-1:0] wr_addr,
    input logic [`BUFFER_WIDTH-1:0] wr_data,
    input  logic                     input_first_in,
    input  logic                     input_last_in,
    input  logic [  `ADDR_WIDTH-1:0] input_addr,
    output logic                     input_first_out,
    output logic                     input_last_out,
    output logic [`BUFFER_WIDTH-1:0] input_data,
    output logic [`BUFFER_WIDTH-1:0] input_data_comb,
    input  logic                     weight_first_in,
    input  logic                     weight_last_in,
    input  logic [  `ADDR_WIDTH-1:0] weight_addr,
    output logic                     weight_first_out,
    output logic                     weight_last_out,
    output logic [`BUFFER_WIDTH-1:0] weight_data
);
endmodule
'''

INSTRUCTION_MEMORY_BLACKBOX = '''`include "defines.sv"
(* blackbox *) module instruction_memory #(parameter INIT_FILE = "") (
    input logic clk,
    input logic rst_n,
    input  logic                     wr_en,
    input  logic [  `ADDR_WIDTH-1:0] wr_addr,
    input  logic [`BUFFER_WIDTH-1:0] wr_data,
    input  logic [  `ADDR_WIDTH-1:0] rd_addr,
    output logic [  `INST_WIDTH-1:0] rd_data
);
endmodule
'''


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def stage_rtl(dst: Path, abstract_memory: bool) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(RTL_DIR, dst)

    ppu = (dst / "ppu.sv").read_text()
    ppu = PPU_DEBUG_RE.sub("\n", ppu)
    (dst / "ppu.sv").write_text(ppu)

    top = (dst / "tinynpu_top.sv").read_text()
    top = TRACE_INIT_RE.sub("\n", top)
    (dst / "tinynpu_top.sv").write_text(top)

    if abstract_memory:
        (dst / "unified_buffer.sv").write_text(UNIFIED_BUFFER_BLACKBOX)
        (dst / "instruction_memory.sv").write_text(INSTRUCTION_MEMORY_BLACKBOX)


def build_script(staged_rtl: Path, script_path: Path) -> None:
    files = sorted(str(p) for p in staged_rtl.glob("*.sv"))
    script = (
        "read_slang -I {inc} -DSYNTHESIS {files}; hierarchy -top tinynpu_top; "
        "synth_xilinx -family xc7; stat\n"
    ).format(inc=staged_rtl, files=" ".join(files))
    script_path.write_text(script)


def parse_stats(log: str) -> dict[str, str]:
    stats: dict[str, str] = {}
    lc = re.search(r"Estimated number of LCs:\s+(\d+)", log)
    if lc:
        stats["lcs"] = lc.group(1)

    top_block = re.search(r"=== tinynpu_top ===.*?End of script\.", log, re.S)
    if not top_block:
        return stats
    block = top_block.group(0)
    for cell in [
        "DSP48E1",
        "FDCE",
        "CARRY4",
        "MUXF7",
        "MUXF8",
        "RAM128X1D",
        "RAM256X1S",
        "LUT1",
        "LUT2",
        "LUT3",
        "LUT4",
        "LUT5",
        "LUT6",
    ]:
        m = re.search(rf"\n\s*(\d+)\s+{cell}\b", block)
        if m:
            stats[cell] = m.group(1)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproducible TinyNPU Yosys/slang synthesis flow")
    ap.add_argument("--mode", choices=["abstract-ram", "full-ram"], default="abstract-ram")
    ap.add_argument("--yosys-bin", default=str(Path("/tmp/yosys058-install/bin/yosys")))
    ap.add_argument("--slang-plugin", default=str(Path("/tmp/yosys-slang-rec/build/slang.so")))
    ap.add_argument("--workdir", default=str(Path("/tmp/tinynpu_synth")))
    ap.add_argument("--save-log", default="")
    args = ap.parse_args()

    workdir = Path(args.workdir).resolve()
    staged = workdir / ("rtl_abstract_ram" if args.mode == "abstract-ram" else "rtl_full_ram")
    script = workdir / ("synth_abstract_ram.ys" if args.mode == "abstract-ram" else "synth_full_ram.ys")
    workdir.mkdir(parents=True, exist_ok=True)

    stage_rtl(staged, abstract_memory=args.mode == "abstract-ram")
    build_script(staged, script)

    cmd = [args.yosys_bin, "-Q", "-m", args.slang_plugin, str(script)]
    proc = run(cmd, cwd=ROOT)
    if args.save_log:
        Path(args.save_log).write_text(proc.stdout + proc.stderr)

    text = proc.stdout + proc.stderr
    sys.stdout.write(text)
    if proc.returncode != 0:
        return proc.returncode

    stats = parse_stats(text)
    print("\nSummary:")
    print(f"  mode: {args.mode}")
    for key in ["lcs", "DSP48E1", "FDCE", "CARRY4", "MUXF7", "MUXF8", "RAM128X1D", "RAM256X1S", "LUT1", "LUT2", "LUT3", "LUT4", "LUT5", "LUT6"]:
        if key in stats:
            print(f"  {key}: {stats[key]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
