# TinyNPU Synthesis Characterization

This repository now includes a reproducible Yosys/slang synthesis flow for `tinynpu_top`.

Script:
- `scripts/synth_tinynpu_yosys.py`

## Scope

The script supports two modes:

1. `abstract-ram`
- keeps the real datapath and control logic
- replaces only `unified_buffer` and `instruction_memory` with blackbox RAM stubs
- useful for estimating logic overhead without charging the current UB/IM RTL implementation

2. `full-ram`
- keeps the real datapath, control logic, and current UB/IM RTL
- useful for seeing the whole current RTL design cost, including the existing UB/IM implementation
- in the measured flow, Yosys maps the memory RTL into RAM primitives rather than leaving it as generic logic

In both modes the script stages a synthesis-only copy of the RTL and strips simulation-only constructs that break synthesis frontends:
- top-level trace `$test$plusargs(...)` block in `tinynpu_top.sv`
- PPU debug `$display/$fflush` block in `ppu.sv`

## Tool Assumptions

The flow expects:
- a Yosys binary
- a `yosys-slang` plugin (`slang.so`)

Defaults are set to the local paths used during bring-up:
- `--yosys-bin /tmp/yosys058-install/bin/yosys`
- `--slang-plugin /tmp/yosys-slang-rec/build/slang.so`

These can be overridden on the command line.

## Usage

Abstract memory only:

```sh
python scripts/synth_tinynpu_yosys.py --mode abstract-ram
```

Include current UB/IM RTL:

```sh
python scripts/synth_tinynpu_yosys.py --mode full-ram
```

Save the raw Yosys log:

```sh
python scripts/synth_tinynpu_yosys.py --mode full-ram --save-log /tmp/tinynpu_full_ram.log
```

## Measured Results

Using the local Yosys `v0.58` + `yosys-slang` flow on March 11, 2026:

### `abstract-ram`
- `Estimated number of LCs: 25117`
- `DSP48E1: 227`

### `full-ram`
- `Estimated number of LCs: 33644`
- `DSP48E1: 227`
- inferred memory-related primitives:
  - `RAM128X1D: 8192`
  - `RAM256X1S: 944`

## Interpretation

- The datapath remains the dominant compute cost; DSP usage is unchanged across both modes.
- Including the current UB/IM RTL adds about `8527` LUT-equivalent cells, roughly `34%` over the memory-abstracted run.
- Because Yosys inferred RAM primitives in `full-ram` mode, that number is a valid whole-design FPGA-style proxy for the current RTL, not just a pessimistic LUT-only fallback.
- These are still FPGA-style synthesis proxies from `synth_xilinx`, not ASIC area or final silicon numbers.
