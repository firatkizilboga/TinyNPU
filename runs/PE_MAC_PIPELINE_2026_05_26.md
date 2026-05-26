# PE MAC pipeline timing note

Date: 2026-05-26

## Change

An optional `TINYNPU_PIPELINED_PE_MAC` build mode registers each PE product
before the stationary accumulator add.  The control unit waits one extra cycle
between `all_done` and drain in that build mode so the final registered product
is committed before accumulator drain begins.

This targets the previous CPU+NPU critical path:

```text
Source:      u_tinynpu/u_muscle/u_array/.../weight_latch_reg[0]/C
Destination: u_tinynpu/u_muscle/u_array/.../accumulator_reg[63]/D
Slack:       -1.786 ns
```

The default non-pipelined PE path is left available for behavioral comparison.

## Validation

Python syntax:

```sh
python3 -m py_compile scripts/vivado_timing.py
```

Result: pass.

Pipelined PE RTL tests:

```sh
make -f Makefile.npu SIM_BUILD=sim_build_pe_pipe_multitile TOPLEVEL=tinynpu_top \
  MODULE=test_jit_multitile_matmul USER_EXTRA_ARGS='-DTINYNPU_PIPELINED_PE_MAC' \
  CCACHE_DISABLE=1
```

Result: `TESTS=1 PASS=1 FAIL=0`.

```sh
make -f Makefile.npu SIM_BUILD=sim_build_pe_pipe_iszero TOPLEVEL=tinynpu_top \
  MODULE=test_jit_iszero_mlp_runtime USER_EXTRA_ARGS='-DTINYNPU_PIPELINED_PE_MAC' \
  CCACHE_DISABLE=1
```

Result: `TESTS=1 PASS=1 FAIL=0`.

## Vivado post-route timing

Command:

```sh
python3 scripts/vivado_timing.py cpu-npu --clock-ns 20.0 --pipelined-pe-mac \
  --workdir runs/vivado_cpu_npu_a200_pe_pipe_50mhz
```

Target: `xc7a200t`, `20.0 ns` clock constraint.

| Build | WNS | Hold slack | Approx Fmax |
| --- | ---: | ---: | ---: |
| CPU+NPU real BRAM, UB partial-write pipelined | `-1.786 ns` | `+0.049 ns` | `45.90 MHz` |
| CPU+NPU real BRAM, UB + PE MAC pipelined | `+0.069 ns` | `+0.058 ns` | `50.17 MHz` |

The integrated CPU+NPU real-BRAM design now closes the 50 MHz constraint.

Current top setup path after the PE fix:

```text
Source:      u_tinynpu/u_brain/u_cu/mm_n_total_reg[3]/C
Destination: data_rdata_q_reg[12]/D
Slack:       +0.069 ns
Data delay:  19.924 ns
Logic:       24 levels (CARRY4=10 DSP48E1=1 LUT3=1 LUT4=1 LUT5=4 LUT6=7)
```

Relevant reports:

- `runs/vivado_cpu_npu_a200_pe_pipe_50mhz/reports/post_route_timing.rpt`
- `runs/vivado_cpu_npu_a200_pe_pipe_50mhz/reports/post_route_critical_paths.rpt`
