#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOSYS = ROOT / "tools" / "install" / "yosys" / "bin" / "yosys"
SLANG = ROOT / "tools" / "src" / "yosys-slang" / "build" / "slang.so"
SG13G2_GZ = ROOT / "tools" / "src" / "yosys" / "tests" / "liberty" / "foundry_data" / "sg13g2_stdcell_typ_1p20V_25C.lib.filtered.gz"
SG13G2_LIB = Path("/tmp/sg13g2_stdcell_typ_1p20V_25C.lib")


def ensure_default_liberty() -> Path:
    if SG13G2_LIB.exists():
        return SG13G2_LIB
    if not SG13G2_GZ.exists():
        raise FileNotFoundError(f"missing default Liberty: {SG13G2_GZ}")
    with gzip.open(SG13G2_GZ, "rb") as src:
        SG13G2_LIB.write_bytes(src.read())
    return SG13G2_LIB


def run(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, capture_output=True, check=False)


def npu_files(workdir: Path) -> tuple[list[str], list[Path], str]:
    sys.path.insert(0, str(ROOT / "scripts"))
    from synth_tinynpu_yosys import stage_rtl

    staged = workdir / "rtl_tinynpu_abstract_ram"
    stage_rtl(staged, memory_mode="abstract-ram")
    excluded = {"ub_skewer_wrapper.sv"}
    files = [str(p) for p in sorted(staged.glob("*.sv")) if p.name not in excluded]
    return files, [staged], "tinynpu_top"


def parse_manifest(path: Path, *, design_rtl_dir: Path, include_fpu: bool) -> tuple[list[str], list[Path], str]:
    incdirs: list[Path] = []
    files: list[str] = []
    top = "cv32e40p_top"
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        line = line.replace("${DESIGN_RTL_DIR}", str(design_rtl_dir))
        if line.startswith("+incdir+"):
            incdirs.append(Path(line.removeprefix("+incdir+")))
            continue
        p = Path(line)
        if p.name in {"cv32e40p_tb_wrapper.sv", "cv32e40p_tracer_pkg.sv"}:
            continue
        if not include_fpu and "vendor/pulp_platform_fpnew" in str(p):
            continue
        files.append(str(p))
    return files, incdirs, top


def cpu_files(*, include_fpu: bool) -> tuple[list[str], list[Path], str]:
    rtl = ROOT / "external" / "cv32e40p" / "rtl"
    manifest = ROOT / "external" / "cv32e40p" / ("cv32e40p_fpu_manifest.flist" if include_fpu else "cv32e40p_manifest.flist")
    return parse_manifest(manifest, design_rtl_dir=rtl, include_fpu=include_fpu)


def write_yosys_script(
    path: Path,
    *,
    files: list[str],
    incdirs: list[Path],
    top: str,
    liberty: Path,
    delay_ps: int,
    no_alumacc: bool,
    skip_abc: bool,
    coarse_only: bool,
) -> None:
    inc = " ".join(f"-I {p}" for p in incdirs)
    file_list = " ".join(files)
    synth_opts = f"-top {top} -flatten -noshare -noabc"
    if no_alumacc:
        synth_opts += " -noalumacc"
    if coarse_only:
        synth_opts += " -run begin:coarse"
    lines = [
        f"read_slang {inc} -DSYNTHESIS {file_list}",
        f"hierarchy -check -top {top}",
        f"synth {synth_opts}",
        f"dfflibmap -liberty {liberty}",
    ]
    if not skip_abc and not coarse_only:
        lines.append(f"abc -liberty {liberty} -D {delay_ps}")
    lines.extend(
        [
            "clean",
            f"stat -liberty {liberty}",
        ]
    )
    if not skip_abc and not coarse_only:
        lines.append("sta")
    lines.append("")
    path.write_text("\n".join(lines))


def summarize(log: str) -> list[str]:
    out: list[str] = []
    for pat, label in [
        (r"Chip area for top module '\\?([^']+)':\s+([0-9.]+)", "area"),
        (r"Longest topological path in ([^:]+):\s+([0-9.]+) ns", "path"),
        (r"Delay:\s+([0-9.]+)", "abc_delay"),
    ]:
        matches = re.findall(pat, log)
        if matches:
            out.append(f"{label}: {matches[-1]}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Rough ASIC-style Yosys synthesis for TinyNPU/CV32E40P.")
    ap.add_argument("target", choices=["npu", "cpu", "cpu-fpu"])
    ap.add_argument("--workdir", default=str(ROOT / "runs" / "asic_synth"))
    ap.add_argument("--liberty", default="")
    ap.add_argument("--delay-ps", type=int, default=5000)
    ap.add_argument("--no-alumacc", action="store_true", help="Skip Yosys arithmetic-combining pass.")
    ap.add_argument("--skip-abc", action="store_true", help="Stop after generic synthesis and DFF mapping.")
    ap.add_argument("--coarse-only", action="store_true", help="Stop Yosys synth before fine arithmetic techmapping.")
    ap.add_argument("--emit-only", action="store_true", help="Write the Yosys script but do not run it.")
    ap.add_argument("--yosys-bin", default=str(YOSYS))
    ap.add_argument("--slang-plugin", default=str(SLANG))
    args = ap.parse_args()

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    liberty = Path(args.liberty) if args.liberty else ensure_default_liberty()

    if args.target == "npu":
        files, incdirs, top = npu_files(workdir)
    elif args.target == "cpu-fpu":
        files, incdirs, top = cpu_files(include_fpu=True)
    else:
        files, incdirs, top = cpu_files(include_fpu=False)

    script = workdir / f"{args.target}.ys"
    log_path = workdir / f"{args.target}.log"
    write_yosys_script(
        script,
        files=files,
        incdirs=incdirs,
        top=top,
        liberty=liberty,
        delay_ps=args.delay_ps,
        no_alumacc=args.no_alumacc,
        skip_abc=args.skip_abc,
        coarse_only=args.coarse_only,
    )
    if args.emit_only:
        print(f"target={args.target}")
        print(f"script={script}")
        print(f"top={top}")
        print(f"files={len(files)}")
        return 0

    proc = run([args.yosys_bin, "-Q", "-m", args.slang_plugin, str(script)], cwd=ROOT)
    text = proc.stdout + proc.stderr
    log_path.write_text(text)

    print(f"target={args.target}")
    print(f"script={script}")
    print(f"log={log_path}")
    print(f"returncode={proc.returncode}")
    for line in summarize(text):
        print(line)
    if proc.returncode != 0:
        print(text[-4000:])
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
