# NPU Recovery and Timing Status - 2026-05-18

## Success Criteria

- NPU RTL must pass a model-shaped run, not only unit tests.
- PPU, XFORM, multitile matmul, shared SRAM MOVE, and is_zero MLP acceptance tests must pass.
- CPU vs CPU+NPU cycle numbers must be collected on the current compiler/runtime state.
- CPU-only FPGA timing must be measured separately from NPU and CPU+NPU timing.

## Functional RTL Results

Command:

```sh
python3 scripts/run_synthability_acceptance.py --include-integration --stop-on-fail
```

Result: `ACCEPTANCE PASS`

Covered checks:

- `static_rtl_guards`: pass
- `python_syntax`: pass
- `npu_coarse_asic_synth`: pass
- `ppu_unit`: pass, including ReLU, sigmoid, hard-GELU, int8/int4 packing, and done timing
- `rtl_multitile_matmul`: pass
- `xform_q_f16_i16_shared`: pass for quantize and dequantize one-word paths
- `rtl_is_zero_mlp`: pass, synthetic 3-layer MLP, host/RTL `dq_out = -0.671875`
- `rtl_shared_sram_move`: pass, CPU shared write -> NPU MOVE -> CPU shared read

Current correctness workaround:

- Runtime readback is forced through MMIO by default with `TINYNPU_SHARED_READBACK_MMIO=1`.
- Shared SRAM writes/preloads are still used.
- This avoids the CPU shared-window readback lane hazard seen on back-to-back 32-bit reads from 128-bit UB words.
- This is functionally reliable but slower than the intended shared readback path.

## QLlama Decode CPU vs NPU

All runs used Verilator with `-O3` through the cv32e40p testbench and matched expected final tensors.

| Config | CPU cold cycles | NPU cold cycles | NPU hot cycles | Cold speedup | Hot speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `d_model=8, d_head=8, n_heads=1, n_kv_heads=1, ffn=8, T=8` | 41,382 | 73,768 | 71,896 | 0.56x | 0.58x |
| `d_model=16, d_head=8, n_heads=2, n_kv_heads=1, ffn=16, T=8` | 86,032 | 110,081 | 105,951 | 0.78x | 0.81x |
| `d_model=32, d_head=8, n_heads=4, n_kv_heads=2, ffn=32, T=8` | 224,719 | 187,030 | 172,789 | 1.20x | 1.30x |

Interpretation:

- The NPU path is alive and numerically correct.
- Tiny shapes are still worse than CPU because fixed segment launch, staging, and MMIO readback dominate.
- The trend crosses parity at the `d32` decode shape even with the conservative MMIO readback workaround.

## QLlama Prefill CPU vs NPU

Command:

```sh
python3 scripts/run_cv32e40p_qllama_block_benchmark.py --mode prefill --variant both --d-model 8 --d-head 8 --n-heads 1 --n-kv-heads 1 --ffn-hidden-dim 8 --prompt-len 8 --repeat-count 1 --prefill-maxcycles 2000000 --timeout-s 300
```

Result:

| Config | CPU cold cycles | NPU cold cycles | NPU hot cycles | Cold speedup | Hot speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| `d_model=8, d_head=8, n_heads=1, n_kv_heads=1, ffn=8, T=8` | 261,341 | 214,731 | 213,366 | 1.22x | 1.22x |

Interpretation:

- Prefill is already favorable for the NPU at the small `d8/T8` shape because sequence-level matmul work amortizes fixed segment overhead better than single-token decode.

## FPGA Timing

Target part: `xc7a200tsbg484-1`

Script:

```sh
python3 scripts/vivado_timing.py <target> --clock-ns <period> --workdir <dir>
```

CPU-only timing was added as a new script target:

```sh
python3 scripts/vivado_timing.py cpu --clock-ns 20.0 --workdir runs/vivado_cpu_only_a200_50mhz
python3 scripts/vivado_timing.py cpu --clock-ns 19.0 --workdir runs/vivado_cpu_only_a200_52p6mhz
python3 scripts/vivado_timing.py cpu --clock-ns 18.5 --workdir runs/vivado_cpu_only_a200_54p1mhz
python3 scripts/vivado_timing.py cpu --clock-ns 17.5 --workdir runs/vivado_cpu_only_a200_57p1mhz
```

Results:

| Target | Workdir | Period | Frequency | WNS | Status |
| --- | --- | ---: | ---: | ---: | --- |
| CPU only | `runs/vivado_cpu_only_a200_50mhz` | 20.0 ns | 50.0 MHz | +0.325 ns | met |
| CPU only | `runs/vivado_cpu_only_a200_52p6mhz` | 19.0 ns | 52.6 MHz | +0.011 ns | met |
| CPU only | `runs/vivado_cpu_only_a200_54p1mhz` | 18.5 ns | 54.1 MHz | +0.631 ns | met |
| CPU only | `runs/vivado_cpu_only_a200_57p1mhz` | 17.5 ns | 57.1 MHz | +0.000 ns | met |
| NPU only, real BRAM | `runs/vivado_npu_a200_bram_50mhz` | 20.0 ns | 50.0 MHz | -4.853 ns | failed |
| CPU+NPU, real BRAM | `runs/vivado_cpu_npu_a200_bram_real_50mhz` | 20.0 ns | 50.0 MHz | -5.527 ns | failed |
| CPU+NPU, abstract memories | `runs/vivado_cpu_npu_a200_bram_50mhz` | 20.0 ns | 50.0 MHz | +7.359 ns | met |

CPU-only utilization at 50 MHz:

- Slice LUTs: 4,480
- Slice registers: 2,132
- DSPs: 5

Timing interpretation:

- CPU-only no-FPU cv32e40p routes at 57.1 MHz on the selected Artix-7 part with zero positive setup margin (`WNS=+0.000 ns`).
- Treat 57.1 MHz as the measured CPU-only OOC FMax point for this run, not a board-level signoff number, because Vivado warns that `HD.CLK_SRC` is not set in out-of-context mode.
- Real-memory NPU and CPU+NPU do not meet 50 MHz yet; based only on 20 ns WNS, the current routed real-BRAM design is around the 39-40 MHz regime.
- The abstract-memory CPU+NPU result is not a real implementation result; it is useful only to prove the non-memory logic can route easily.

## Current Blockers

- Shared SRAM model readback is not trusted for model outputs; MMIO readback is the safe default until the 32-bit lane/read latency issue is fixed.
- Real BRAM NPU timing misses 50 MHz. The next timing work should target the current NPU real-BRAM critical path rather than CPU-only timing.
- Small transformer blocks are dominated by host ops and transfer/readback overhead; speedup appears only once matmul work is large enough to amortize fixed costs.

## Completion Audit

Objective: create the missing experiments for new CPU vs CPU+NPU numbers and obtain CPU-only FMax.

Checklist:

- CPU vs CPU+NPU decode numbers: covered by QLlama decode `d8`, `d16`, and `d32` RTL runs above.
- CPU vs CPU+NPU prefill number: covered by QLlama prefill `d8/T8` RTL run above.
- Current NPU functionality gate: covered by `run_synthability_acceptance.py --include-integration --stop-on-fail`, which passed.
- CPU-only FMax: covered by routed Vivado CPU-only runs at 50.0 MHz, 52.6 MHz, 54.1 MHz, and 57.1 MHz; 57.1 MHz met timing with `WNS=+0.000 ns`.
- Real CPU+NPU timing status: covered by existing routed real-BRAM CPU+NPU report; it fails 50 MHz with `WNS=-5.527 ns`.
- Report artifact: this file contains commands, cycle counts, timing report locations, and caveats.

Remaining weak points:

- CPU-only FMax was not pushed past a failing point; the measured OOC point is 57.1 MHz with zero setup slack.
- GPT-like two-block reuse is functionally fixed from prior runs, but this report does not include a fresh CPU-only GPT2 baseline because the immediate deliverable was CPU vs CPU+NPU experiment numbers plus CPU-only FMax, and QLlama now covers both decode and prefill.
