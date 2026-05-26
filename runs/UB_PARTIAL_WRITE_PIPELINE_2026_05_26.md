# UB partial-write pipeline timing note

Date: 2026-05-26

## Change

The FPGA-BRAM unified buffer path now registers write requests before the
INT4/partial-byte packer.  The byte-RAM write port uses the registered request
address, data, and mask, and the read path includes a same-address bypass for
the registered write commit.

This breaks the previous long path from control-unit matrix address arithmetic
directly into `partial_data` / `partial_mask` update logic.

## Validation

Dedicated FPGA-BRAM INT4 partial-write test:

```sh
make -f Makefile.npu SIM_BUILD=sim_build_ub_fpga_int4 TOPLEVEL=unified_buffer \
  MODULE=test_unified_buffer_fpga_int4 \
  USER_EXTRA_ARGS='-DTINYNPU_FPGA_BRAM -DTINYNPU_VIVADO_BRAM /root/compiler-optimization/rtl/tinynpu_byte_ram.sv' \
  CCACHE_DISABLE=1
```

Result: `TESTS=1 PASS=1 FAIL=0`.

Synthability acceptance:

```sh
python3 scripts/run_synthability_acceptance.py --include-integration --stop-on-fail
```

Result: `ACCEPTANCE PASS`.

## Vivado post-route timing

Target: `xc7a200t`, `20.0 ns` clock constraint.

| Build | Before WNS | Before approx Fmax | After WNS | After approx Fmax |
| --- | ---: | ---: | ---: | ---: |
| NPU-only real BRAM | `-9.851 ns` | `33.50 MHz` | `+0.106 ns` | `50.27 MHz` |
| CPU+NPU real BRAM | `-9.486 ns` | `33.91 MHz` | `-1.786 ns` | `45.90 MHz` |

The NPU-only real-BRAM design now closes at 50 MHz.

The integrated CPU+NPU design no longer fails on the UB partial-write path.  Its
current top failing path is inside a systolic PE accumulator:

```text
Source:      u_tinynpu/u_muscle/u_array/gen_rows[1].gen_cols[5].pe_inst/weight_latch_reg[0]/C
Destination: u_tinynpu/u_muscle/u_array/gen_rows[1].gen_cols[5].pe_inst/accumulator_reg[63]/D
Slack:       -1.786 ns
Data delay:  21.734 ns
Logic:       26 levels (CARRY4=18 LUT2=3 LUT5=1 LUT6=4)
```

Relevant reports:

- `runs/vivado_npu_a200_ub_partial_pipe_50mhz/reports/post_route_timing.rpt`
- `runs/vivado_cpu_npu_a200_ub_partial_pipe_50mhz/reports/post_route_timing.rpt`
- `runs/vivado_cpu_npu_a200_ub_partial_pipe_50mhz/reports/post_route_critical_paths.rpt`
