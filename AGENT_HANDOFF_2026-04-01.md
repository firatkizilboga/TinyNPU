# Agent Handoff: CV32E40P Bare-Metal Bring-Up

## Goal

The current goal is to evaluate a minimal RISC-V software context for TinyNPU without expanding scope into RocketChip/F2 or LLM inference.

Target outcome:

- Run bare-metal code on `cv32e40p`
- Use the core's instruction/data interfaces with a minimal memory shell
- Later memory-map TinyNPU behind the data interface
- Use polling only; no interrupts, no debug, no Linux, no DMA for the first milestone

This is meant to answer:

- Can TinyNPU live behind a simple RISC-V MMIO runtime?
- Can we run a bare-metal C/C++ driver that launches the NPU and polls for completion?

## Current Understanding

The `cv32e40p` repo is a core repo, not a full SoC, but its example testbench already provides most of the minimal shell we need:

- `external/cv32e40p/example_tb/core/cv32e40p_tb_subsystem.sv`
- `external/cv32e40p/example_tb/core/mm_ram.sv`
- `external/cv32e40p/example_tb/core/tb_top.sv`

This means the first integration path is not "build a SoC from scratch". It is:

1. get the stock example running under Verilator
2. confirm bare-metal firmware executes
3. later add a TinyNPU MMIO region into `mm_ram.sv`

## Why We Are Not Chasing LLMs Right Now

The current conclusion is that LLM inference is too large a scope increase for TinyNPU right now because it requires:

- a coherent quantized attention contract
- awkward `Q/K/V` layout handling
- decode-time KV-cache strategy
- memory-system changes that are bigger than a thesis-scale extension

So the near-term path is a minimal RISC-V runtime experiment instead.

## Future Note: Float Path

The current bare-metal runtime work is intentionally focused on the existing integer TinyNPU flow.

However, the next major direction the user wants to keep on the table is a float-capable path:

- a float-capable TinyNPU configuration or companion "float chip" path
- fast host-side float implementations of the host ops
- a runtime contract that is not forced through integer-only quantized staging when the experiment is explicitly about float execution

This is a future branch, not the current bring-up target. The immediate priority remains getting the integer bare-metal runtime emitter and host-op execution path correct end to end.

## Additional Experiment: Core-Side Float Bring-Up

After the integer runtime milestone, a separate experiment checked whether the CV32E40P shell can execute hard-float code at all.

What was confirmed:

- the installed compiler can emit real `rv32imfc` instructions if built with:
  - `-march=rv32imfc -mabi=ilp32`
- the existing toolchain does **not** ship hard-float newlib libraries, so:
  - `-mabi=ilp32f` fails to link
  - `-mabi=ilp32` must be used for the smoke test
- the disassembly of the smoke test contains:
  - `flw`
  - `fadd.s`
  - `fmul.s`
  - `fsw`
  - `fmv.x.w`
- a hard-float Verilator build now works and a float smoke test exits successfully

Important runtime detail:

- the first float attempt trapped repeatedly with the illegal-instruction handler
- the fix was to set `mstatus.FS` in software before the first FP instruction

Confirmed output:

```text
float bits: 0x40f00000
float smoke passed
EXIT SUCCESS
```

Current local experimental files:

- `external/cv32e40p/example_tb/core/custom_fp/float_smoke.c`
- `external/cv32e40p/example_tb/core/div_sqrt_top_mvp_stub.sv`
- `external/cv32e40p/example_tb/core/fpnew_divsqrt_multi_stub.sv`

Current local experimental RTL tweak:

- `external/cv32e40p/rtl/cv32e40p_fp_wrapper.sv`
  - `PulpDivsqrt` changed from `1'b0` to `1'b1`

Why the stubs exist:

- the vendored `fpnew` tree in this checkout is incomplete for Verilator hard-float bring-up
- `div_sqrt_top_mvp` is missing from the tree
- Verilator also rejects the shipped div/sqrt pipeline modules because they mix blocking and nonblocking assignments
- this is acceptable only for the current smoke because the program uses `fadd.s` and `fmul.s`, not `fdiv.s` or `fsqrt.s`

So the current state is:

- hard-float add/mul smoke: working
- full production-quality div/sqrt path: **not** solved
- current float setup is good enough for core-side feasibility checks, not yet for a real float runtime stack

## What Was Done

### 1. Cloned `cv32e40p`

Local path:

- `external/cv32e40p`

### 2. Confirmed the Repo Has Verilator-Aware RTL

There is no turnkey Verilator run target, but the repo is clearly Verilator-compatible:

- `src_files.yml` includes `cv32e40p_regfile_verilator`
- several files have `` `ifdef VERILATOR ``
- the example README mentions Verilator

### 3. Built the Stock Bare-Metal Firmware

The example firmware build works.

Produced files:

- `external/cv32e40p/example_tb/core/custom/hello_world.elf`
- `external/cv32e40p/example_tb/core/custom/hello_world.hex`

Important detail:

- the linker script places `.init` at `0x180`
- the example testbench boot address matches this
- `_exit` writes to `0x20000004`, which `mm_ram.sv` recognizes

### 4. Added a Local Verilator Harness

Added:

- `external/cv32e40p/example_tb/core/verilator_main.cpp`

Purpose:

- pass `+firmware=...`
- run the testbench until finish or timeout

### 5. Added a Stub for the FP Wrapper

Added:

- `external/cv32e40p/example_tb/core/cv32e40p_fp_wrapper_stub.sv`

Reason:

- avoid dragging the full floating-point block into a hello-world bring-up

### 6. Patched Parameter Drift in the Example Testbench

Modified:

- `external/cv32e40p/example_tb/core/cv32e40p_tb_subsystem.sv`

Change:

- `.PULP_XPULP(...)` -> `.COREV_PULP(...)`
- `.PULP_CLUSTER(...)` -> `.COREV_CLUSTER(...)`

Reason:

- newer `cv32e40p_top.sv` expects `COREV_*` parameter names

### 7. Patched a Verilator Incompatibility in HPM CSR Logic

Modified:

- `external/cv32e40p/rtl/cv32e40p_cs_registers.sv`

Reason:

- Verilator 5.008 complained about mixed blocking/nonblocking assignments in the HPM generate logic

Temporary workaround:

- under `` `ifdef VERILATOR `` / `` `else ``, HPM-related arrays are tied to zero

This is a bring-up hack, not a final upstream-quality fix.

### 8. Integrated a First TinyNPU MMIO Window Into the CV32E40P Example Shell

Modified:

- `external/cv32e40p/example_tb/core/mm_ram.sv`

What this does:

- maps TinyNPU at `0x3000_0000`
- instantiates `tinynpu_top` inside the testbench RAM/MMIO shell
- routes CPU data-bus accesses in that region to TinyNPU's existing byte-oriented host MMIO port

Important limitation:

- this first adapter is intentionally byte-oriented
- it is suitable for bare-metal byte reads/writes and register bring-up
- it is not yet a polished 32-bit convenience interface

### 9. Added a Bare-Metal NPU MMIO Smoke Test

Added:

- `external/cv32e40p/example_tb/core/custom/npu_mmio_smoke.c`

What it does:

- reads TinyNPU `STATUS`
- writes TinyNPU `CMD`
- reads `CMD` back
- exits successfully if the echoed byte matches

### 10. Added a Bare-Metal TinyNPU MatMul Smoke Test

Added:

- `external/cv32e40p/example_tb/core/custom/npu_matmul_smoke.c`

What it does:

- mirrors the current TinyNPU MMIO protocol in bare-metal C
- writes UB words through `CMD_WRITE_MEM`
- writes IM chunks through `CMD_WRITE_MEM`
- launches execution with `CMD_RUN`
- polls `STATUS`
- reads result vectors back with `CMD_READ_MEM`
- checks the resulting 4x4 output matrix

## Latest Status

The current checkpoint is stronger than the original MMIO bring-up:

- the generated bare-metal two-segment TinyNPU runtime works end to end on the CV32E40P shell
- startup for larger generated programs was improved by moving uninitialized scratch tensors into a `.noinit` section
- the generated MNIST `conv1` bare-metal demo now gets past reset and startup, enters `main()`, and prints:
  - `TinyNPU bare-metal program: cv32e40p_mnist_conv1_demo`
  - `HostOp im2col: im2col_for_npu`
- that `conv1` demo is currently stuck in the host-side `im2col` step, not in TinyNPU launch and not in boot

So the current blocker is no longer MMIO correctness. It is host-side conv preprocessing cost inside the CV32E40P simulation path.

Source of the payload:

- the UB and IM payloads were generated from the current TinyNPU compiler classes, not hand-guessed
- this used `software/compiler/tinynpu/program.py` and `software/compiler/tinynpu/isa.py`

## Current Status

### What Works

- bare-metal firmware builds
- Verilator elaboration and compilation now succeed
- the simulation binary is produced

Built binary:

- `external/cv32e40p/example_tb/core/obj_dir/cv32e40p_tb_vlt`

### What Works Now

The stock CV32E40P example now runs successfully under Verilator.

Confirmed output:

- `hello world!`
- `EXIT SUCCESS`

Key finding:

- the earlier "hang" was not a broken system
- the simulation was simply being stopped far too early in simulator ticks
- the core was already fetching and executing correctly after reset

Additional local bring-up aid:

- `tb_top.sv` now has optional `+trace_instr` / `+trace_data` plusargs for lightweight bus tracing

### What Else Works Now

The first TinyNPU MMIO integration also works.

Confirmed output:

```text
npu status: 0x00
npu cmd echo: 0x5a
EXIT SUCCESS
```

That means:

- CV32E40P bare-metal code can reach the TinyNPU MMIO window
- writes and readbacks across the data interface are functioning
- the integration path is viable

### What Else Works Now

The full bare-metal matmul path now also works.

Confirmed output:

```text
Loading UB...
Loading IM...
Running matmul...
row 0: 3 6 9 12
row 1: 15 18 21 24
row 2: 3 3 3 3
row 3: 6 6 6 6
npu matmul smoke passed
EXIT SUCCESS
```

That means:

- the full `WRITE_MEM -> RUN -> HALT -> READ_MEM` path works from bare-metal C
- TinyNPU is no longer just MMIO-visible; it is executable from the RISC-V shell
- the minimal CV32E40P + TinyNPU bring-up goal is achieved

### Root Cause That Was Fixed

The later CV32E40P runtime failure was not a TinyNPU arithmetic bug. It was an MMIO/MMVR ownership bug in:

- `rtl/mmio_interface.sv`

Problem:

- after `CMD_READ_MEM`, the control unit sits in `CTRL_READ_WAIT`
- in that state, `mmvr_wr_en` stays high so readback data remains visible in `MMVR`
- CV32E40P software writes the next `MMVR` payload byte by byte, with gaps between stores
- between those byte stores, stale readback data could overwrite `MMVR` again

Concrete failure mode:

- raw UB write/readback at nonzero addresses such as `0x0008` came back corrupted
- the generated two-segment bare-metal runtime failed on the second segment because its prepared input tile was being clobbered while software was loading the next command

Fix:

- add a small `mmvr_host_override` latch in `mmio_interface.sv`
- once the host starts writing during `READ_WAIT`, `MMVR` stays under host control until the control unit leaves that state

This preserves the old direct path and fixes the CV32E40P bare-metal handoff.

### Current Verified State

The following now work together:

- classic direct TinyNPU path:
  - `test_mmio_readwrite_handoff`
  - `test_jit_runtime_two_segments`
  - `test_jit_relu_chain_runtime`
  - `test_jit_mnist_fc`
  - `test_jit_mnist_conv1`
  - `test_jit_mnist_conv2`
  - `test_jit_mnist_conv3`
- CV32E40P bare-metal path:
  - `npu_mmio_smoke`
  - `npu_matmul_smoke`
  - `npu_ub_roundtrip`
  - generated `cv32e40p_relu_chain_demo`

The generated relu-chain program now prints:

```text
TinyNPU bare-metal program: cv32e40p_relu_chain_demo
NpuSegment: segment_000
HostOp relu: relu_h
NpuSegment: segment_001
Final outputs:
y shape=(4, 4)
  row 0: 4 9 1 4
  row 1: 12 25 1 12
  row 2: 3 3 0 6
  row 3: 6 4 1 9
All outputs matched expected tensors
EXIT SUCCESS
```

### Current Limitation

A generated bare-metal MNIST `conv1` demo was also emitted:

- `generated/cv32e40p_mnist_conv1_demo.c`

It compiles and fits in the example RAM, but it does not complete yet under the CV32E40P Verilator wrapper. With a much larger harness budget it still times out, so this looks like a separate bare-metal runtime/performance issue around the host-heavy conv path rather than the earlier MMVR corruption bug.

## Likely Next Step

The immediate bring-up goal is complete. The next sensible steps are:

1. keep this Verilator flow as the base platform
2. keep `npu_mmio_smoke.c` and `npu_matmul_smoke.c` as regression bring-up tests
3. add a cleaner local build/run target so the Verilator command line is not hand-carried
4. decide what the next runtime contract should look like:
   - pure MMIO staging
   - or shared RAM buffers plus control/status MMIO
5. debug why the generated bare-metal `conv1` path does not complete
6. remove or trim the temporary `+trace_instr` / `+trace_data` / `+trace_npu` debug prints in `tb_top.sv` once no longer needed

## Exact Commands That Were Run

### Basic repo inspection

```bash
cd /home/firatkizilboga/compiler-optimization
rg -n "verilator|VERILATOR|verilate" external/cv32e40p -g '!**/.git/**'
```

### Build the example firmware

```bash
cd /home/firatkizilboga/compiler-optimization/external/cv32e40p/example_tb/core
RISCV=/opt/riscv make custom/hello_world.hex
```

### Check ELF layout

```bash
/opt/riscv/bin/riscv32-unknown-elf-objdump -h custom/hello_world.elf
/opt/riscv/bin/riscv32-unknown-elf-objdump -d custom/hello_world.elf | sed -n '1,140p'
head -n 40 custom/hello_world.hex
```

### Build the Verilator simulation

Run from:

```bash
cd /home/firatkizilboga/compiler-optimization/external/cv32e40p/example_tb/core
```

Build command:

```bash
CCACHE_DISABLE=1 verilator -Wall -Wno-fatal -Wno-DECLFILENAME --timing -GNUM_MHPMCOUNTERS=0 \
  -cc -exe --build --top-module tb_top \
  -CFLAGS "-std=c++17" \
  -DVERILATOR \
  -I../../include \
  -I../../rtl/include \
  tb_top.sv \
  cv32e40p_tb_wrapper.sv \
  cv32e40p_tb_subsystem.sv \
  mm_ram.sv \
  dp_ram.sv \
  riscv_gnt_stall.sv \
  riscv_rvalid_stall.sv \
  cv32e40p_random_interrupt_generator.sv \
  cv32e40p_fp_wrapper_stub.sv \
  ../../bhv/include/perturbation_defines.sv \
  ../../rtl/include/cv32e40p_apu_core_pkg.sv \
  ../../rtl/include/cv32e40p_pkg.sv \
  ../../rtl/cv32e40p_core.sv \
  ../../rtl/cv32e40p_alu.sv \
  ../../rtl/cv32e40p_aligner.sv \
  ../../rtl/cv32e40p_apu_disp.sv \
  ../../rtl/cv32e40p_compressed_decoder.sv \
  ../../rtl/cv32e40p_controller.sv \
  ../../rtl/cv32e40p_cs_registers.sv \
  ../../rtl/cv32e40p_decoder.sv \
  ../../rtl/cv32e40p_ex_stage.sv \
  ../../rtl/cv32e40p_fetch_fifo.sv \
  ../../rtl/cv32e40p_hwloop_regs.sv \
  ../../rtl/cv32e40p_id_stage.sv \
  ../../rtl/cv32e40p_if_stage.sv \
  ../../rtl/cv32e40p_int_controller.sv \
  ../../rtl/cv32e40p_load_store_unit.sv \
  ../../rtl/cv32e40p_mult.sv \
  ../../rtl/cv32e40p_obi_interface.sv \
  ../../rtl/cv32e40p_popcnt.sv \
  ../../rtl/cv32e40p_prefetch_buffer.sv \
  ../../rtl/cv32e40p_prefetch_controller.sv \
  ../../rtl/cv32e40p_register_file_ff.sv \
  ../../rtl/cv32e40p_sleep_unit.sv \
  ../../rtl/cv32e40p_top.sv \
  ../../rtl/cv32e40p_wrapper.sv \
  verilator_main.cpp
```

### Run the built simulation

```bash
./obj_dir/cv32e40p_tb_vlt +firmware=custom/hello_world.hex
```

### Known-good run that reaches completion

```bash
VERILATOR_MAX_TICKS=100000000 ./obj_dir/cv32e40p_tb_vlt \
  +firmware=custom/hello_world.hex \
  +maxcycles=50000
```

Expected output:

```text
hello world!
EXIT SUCCESS
```

### Build the TinyNPU MMIO smoke firmware

```bash
cd /home/firatkizilboga/compiler-optimization/external/cv32e40p/example_tb/core
/opt/riscv/bin/riscv32-unknown-elf-gcc -march=rv32imc -o custom/npu_mmio_smoke.elf -w -Os -g -nostdlib \
  -T custom/link.ld -static \
  custom/crt0.S custom/npu_mmio_smoke.c mem_stall/mem_stall.c custom/syscalls.c custom/vectors.S \
  -I /opt/riscv/riscv32-unknown-elf/include -I mem_stall \
  -L /opt/riscv/riscv32-unknown-elf/lib -lc -lm -lgcc
/opt/riscv/bin/riscv32-unknown-elf-objcopy -O verilog custom/npu_mmio_smoke.elf custom/npu_mmio_smoke.hex
```

### Build the CV32E40P + TinyNPU Verilator platform

Run from:

```bash
cd /home/firatkizilboga/compiler-optimization/external/cv32e40p/example_tb/core
```

Build command:

```bash
CCACHE_DISABLE=1 verilator -Wall -Wno-fatal -Wno-DECLFILENAME --timing -GNUM_MHPMCOUNTERS=0 \
  -cc -exe --build --top-module tb_top -CFLAGS "-std=c++17" -o cv32e40p_tb_vlt_npu \
  -I../../../../rtl \
  verilator_main.cpp include/perturbation_pkg.sv \
  ../../rtl/include/cv32e40p_apu_core_pkg.sv ../../rtl/include/cv32e40p_fpu_pkg.sv ../../rtl/include/cv32e40p_pkg.sv \
  ../../bhv/include/cv32e40p_tracer_pkg.sv ../../bhv/cv32e40p_sim_clock_gate.sv \
  cv32e40p_fp_wrapper_stub.sv \
  ../../../../rtl/unified_buffer.sv ../../../../rtl/skewer.sv ../../../../rtl/pe.sv ../../../../rtl/systolic_array.sv \
  ../../../../rtl/ppu.sv ../../../../rtl/ubss.sv ../../../../rtl/mmio_interface.sv ../../../../rtl/instruction_memory.sv \
  ../../../../rtl/control_unit.sv ../../../../rtl/control_top.sv ../../../../rtl/tinynpu_top.sv \
  ../../rtl/cv32e40p_if_stage.sv ../../rtl/cv32e40p_cs_registers.sv ../../rtl/cv32e40p_register_file_ff.sv \
  ../../rtl/cv32e40p_load_store_unit.sv ../../rtl/cv32e40p_id_stage.sv ../../rtl/cv32e40p_aligner.sv \
  ../../rtl/cv32e40p_decoder.sv ../../rtl/cv32e40p_compressed_decoder.sv ../../rtl/cv32e40p_fifo.sv \
  ../../rtl/cv32e40p_prefetch_buffer.sv ../../rtl/cv32e40p_hwloop_regs.sv ../../rtl/cv32e40p_mult.sv \
  ../../rtl/cv32e40p_int_controller.sv ../../rtl/cv32e40p_ex_stage.sv ../../rtl/cv32e40p_alu_div.sv \
  ../../rtl/cv32e40p_alu.sv ../../rtl/cv32e40p_ff_one.sv ../../rtl/cv32e40p_popcnt.sv ../../rtl/cv32e40p_apu_disp.sv \
  ../../rtl/cv32e40p_controller.sv ../../rtl/cv32e40p_obi_interface.sv ../../rtl/cv32e40p_prefetch_controller.sv \
  ../../rtl/cv32e40p_sleep_unit.sv ../../rtl/cv32e40p_core.sv ../../rtl/cv32e40p_top.sv ../../bhv/cv32e40p_tb_wrapper.sv \
  amo_shim.sv riscv_gnt_stall.sv riscv_rvalid_stall.sv dp_ram.sv mm_ram.sv cv32e40p_tb_subsystem.sv tb_top.sv
```

### Run the TinyNPU MMIO smoke test

```bash
VERILATOR_MAX_TICKS=100000000 ./obj_dir/cv32e40p_tb_vlt_npu \
  +firmware=custom/npu_mmio_smoke.hex \
  +maxcycles=50000
```

Expected output:

```text
npu status: 0x00
npu cmd echo: 0x5a
EXIT SUCCESS
```

### Build the TinyNPU matmul smoke firmware

```bash
cd /home/firatkizilboga/compiler-optimization/external/cv32e40p/example_tb/core
/opt/riscv/bin/riscv32-unknown-elf-gcc -march=rv32imc -o custom/npu_matmul_smoke.elf -w -Os -g -nostdlib \
  -T custom/link.ld -static \
  custom/crt0.S custom/npu_matmul_smoke.c mem_stall/mem_stall.c custom/syscalls.c custom/vectors.S \
  -I /opt/riscv/riscv32-unknown-elf/include -I mem_stall \
  -L /opt/riscv/riscv32-unknown-elf/lib -lc -lm -lgcc
/opt/riscv/bin/riscv32-unknown-elf-objcopy -O verilog custom/npu_matmul_smoke.elf custom/npu_matmul_smoke.hex
```

### Run the TinyNPU matmul smoke test

```bash
VERILATOR_MAX_TICKS=100000000 ./obj_dir/cv32e40p_tb_vlt_npu \
  +firmware=custom/npu_matmul_smoke.hex \
  +maxcycles=50000
```

Expected output:

```text
Loading UB...
Loading IM...
Running matmul...
row 0: 3 6 9 12
row 1: 15 18 21 24
row 2: 3 3 3 3
row 3: 6 6 6 6
npu matmul smoke passed
EXIT SUCCESS
```

### Run the raw UB round-trip diagnostic

```bash
VERILATOR_MAX_TICKS=100000000 ./obj_dir/cv32e40p_tb_vlt_npu \
  +firmware=custom/npu_ub_roundtrip.hex \
  +maxcycles=50000
```

Expected output:

```text
write addr 0x0000
readback addr 0x0000: 00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff
write addr 0x0008
readback addr 0x0008: 01 23 45 67 89 ab cd ef fe dc ba 98 76 54 32 10
npu ub roundtrip passed
EXIT SUCCESS
```

### Run the generated bare-metal relu-chain demo

```bash
cd /home/firatkizilboga/compiler-optimization/external/cv32e40p/example_tb/core
/opt/riscv/bin/riscv32-unknown-elf-gcc -march=rv32imc -o custom/cv32e40p_relu_chain_demo.elf -w -Os -g -nostdlib \
  -T custom/link.ld -static \
  custom/crt0.S /home/firatkizilboga/compiler-optimization/generated/cv32e40p_relu_chain_demo.c \
  mem_stall/mem_stall.c custom/syscalls.c custom/vectors.S \
  -I /opt/riscv/riscv32-unknown-elf/include -I mem_stall \
  -L /opt/riscv/riscv32-unknown-elf/lib -lc -lm -lgcc
/opt/riscv/bin/riscv32-unknown-elf-objcopy -O verilog custom/cv32e40p_relu_chain_demo.elf custom/cv32e40p_relu_chain_demo.hex
VERILATOR_MAX_TICKS=2000000000 ./obj_dir/cv32e40p_tb_vlt_npu \
  +firmware=custom/cv32e40p_relu_chain_demo.hex \
  +maxcycles=1000000
```

### Current conv bare-metal limitation

The generated `conv1` demo can be built similarly from:

- `generated/cv32e40p_mnist_conv1_demo.c`

but currently times out under the CV32E40P wrapper even with a much larger `VERILATOR_MAX_TICKS` budget.

### Short timeout run for quick sanity check

```bash
VERILATOR_MAX_TICKS=20000 ./obj_dir/cv32e40p_tb_vlt +firmware=custom/hello_world.hex +verbose
```

### Optional bus tracing

```bash
VERILATOR_MAX_TICKS=1000000 ./obj_dir/cv32e40p_tb_vlt \
  +firmware=custom/hello_world.hex \
  +verbose +trace_instr +trace_data +maxcycles=10000
```

## Important Caveats

- `vsim` is not installed here
- this effort is intentionally using Verilator only
- the current HPM workaround in `cv32e40p_cs_registers.sv` is a local bring-up hack
- the stock example does complete successfully, but it needs a large enough `VERILATOR_MAX_TICKS` budget

## Recommended Resume Plan

1. keep the current Verilator NPU smoke build as the known-good baseline
2. add a C helper layer that mirrors `verification/cocotb/npu_driver.py`
3. prove `CMD_WRITE_MEM` and `CMD_READ_MEM` through bare-metal software
4. debug the current generated bare-metal `conv1` path
5. compare its progress against:
   - the passing classic `test_jit_mnist_conv1` path
   - the passing generated bare-metal relu-chain path
6. compare bare-metal transaction ordering and MMVR doorbell timing against:
   - `verification/cocotb/npu_driver.py`
   - `verification/cocotb/test_npu.py`
7. once conv bare-metal is understood, refactor the helper layer into reusable C functions

## FPU Experiment Note

The next requested experiment is enabling floating point on the CV32E40P side and compiling firmware with the `F` extension.

Relevant facts already confirmed:

- the repo has a real `cv32e40p_fp_wrapper.sv`
- the example testbench currently uses `cv32e40p_fp_wrapper_stub.sv`
- `tb_top.sv` and `cv32e40p_tb_subsystem.sv` already expose `FPU` and `ZFINX` parameters

So the likely first experiment is:

1. swap the stub out for the real FP wrapper in the Verilator build
2. pass `-GFPU=1` into the testbench
3. compile a small firmware with `-march=rv32imfc -mabi=ilp32f`
4. see whether the minimal floating-point program executes before trying anything larger

## Minimal Vision After Resume

If the stock base system runs, the next integration should stay minimal:

- keep instruction memory as-is
- keep data memory as-is
- add one TinyNPU MMIO region in `mm_ram.sv`
- expose only a few control/status registers
- use a polling-based bare-metal C runtime

That is the smallest viable RISC-V + TinyNPU experiment.
