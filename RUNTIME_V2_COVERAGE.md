# Runtime V2 Coverage (Updated)

Date: 2026-04-09
Branch baseline: `6ecf1e3` (`rtl: fix packed conv-stream gather for multi-element lanes`)

This report supersedes the earlier failing baseline. It reflects the post-fix behavior with current methodology and verified cycle counts.

## Methodology

- Toolchain/runtime flags:
  - `-march=rv32imfc -mabi=ilp32 -O3`
  - `-DTNPU_RUNTIME_V2_DUMP_FINAL_OUTPUTS=0`
  - shared-SRAM path enabled by default (`TINYNPU_USE_SHARED_SRAM=1`)
- Runtime verification mode:
  - `autoverify` against `<output>_expected` when explicit verify ops are absent.
- Simulator run command:
  - `VERILATOR_MAX_TICKS=3000000000 ./obj_dir/cv32e40p_tb_vlt_npu +firmware=custom/<program>.hex +maxcycles=250000`
- INT4 validity rule used for fair verification:
  - INT4 tensors are interpreted as signed 4-bit values (`[-8, 7]`).
  - Fixtures with source values outside `[-8, 7]` are treated as invalid INT4 vectors for pass/fail claims.

## Build Template

```bash
/opt/riscv/bin/riscv32-unknown-elf-gcc \
  -DTNPU_RUNTIME_V2_DUMP_FINAL_OUTPUTS=0 \
  -march=rv32imfc -mabi=ilp32 -w -O3 -g -nostdlib -static \
  -T custom/link.ld custom/crt0.S \
  generated/<program>_program.c \
  generated/<program>_runner.c \
  software/compiler/tinynpu_jit/tinynpu_runtime_v2.c \
  mem_stall/mem_stall.c custom/syscalls.c custom/vectors.S \
  -I software/compiler/tinynpu_jit -I /opt/riscv/riscv32-unknown-elf/include -I mem_stall \
  -L /opt/riscv/riscv32-unknown-elf/lib -lc -lm -lgcc \
  -o custom/<program>.elf
/opt/riscv/bin/riscv32-unknown-elf-objcopy -O verilog custom/<program>.elf custom/<program>.hex
```

## Current Results

| Program | Path | Host Im2Col | Stage | Run | Readback | Segment.NPU | Autoverify |
|---|---|---:|---:|---:|---:|---:|---|
| `cv32e40p_int16_convstream_same_shape_v2` | conv-stream | 0 | 3810 | 530 | 3387 | 13707 | OK |
| `cv32e40p_int8_packed_convstream_default_v2` | direct packed conv-stream | 0 | 7309 | 98 | 2931 | 16267 | OK |
| `cv32e40p_int4_convstream_same_shape_v2` | host `im2col` + NPU | 52988 | 40004 | 290 | 9453 | 55821 | OK |
| `cv32e40p_int4_packed_convstream_default_v2_inrange` | direct packed conv-stream | 0 | 9176 | 82 | 3199 | 18386 | OK |
| `cv32e40p_int4_packed_convstream_default_v2` | direct packed conv-stream | 0 | 9176 | 82 | 3195 | 18382 | FAIL (`@0 actual=-8 expected=7`) |

## Interpretation

- The packed direct conv-stream RTL path is now validated on INT8 and INT4 with valid INT4-range inputs.
- The remaining INT4 failure is fixture-quality, not path-quality:
  - `cv32e40p_int4_packed_convstream_default_v2_program.c` uses `xmat_data` in `[-14, 14]` (out-of-range for INT4).
  - `cv32e40p_int4_packed_convstream_default_v2_inrange_program.c` clips to `[-8, 7]` and passes.

## Compiler Test Coverage

`software/compiler/tests/test_baremetal_emit.py` (runtime-v2 emission checks):

```bash
pytest -q software/compiler/tests/test_baremetal_emit.py
# 12 passed in 0.60s
```
