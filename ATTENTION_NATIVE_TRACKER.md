# Decode Attention Native Support Tracker

This file tracks the minimum required changes to support decoder-style attention cleanly on the current shared-unified-buffer TinyNPU stack.

## Scope

Target workload:

- single-token decode attention
- shared unified buffer model
- host still allowed for `softmax`, `RMSNorm`, `RoPE`, and orchestration initially
- focus on making `QK^T` and `A @ V` native to the NPU memory/streaming contract

Non-goals for the first pass:

- fused `QKV`
- batching
- replacing host `softmax`
- speculative full LLM end-to-end support

## Current Status

- Shared-SRAM V2 runtime path is working for the validated MLP and explicit-conv benchmarks.
- Gather/`conv_stream` has been removed from the live source path.
- Current writeback modes are effectively `C`, plus `A` for selected chained outputs.
- There is no clean native path for persistent `K/V` cache tensors in a right-fed layout.

## Minimum Required Features

### 1. B-layout writeback

Status: implemented

Need:

- allow NPU outputs to be written back in `B` layout, not only `C`/`A`
- use this for decode-cache residency, especially `K` and `V`

Primary files:

- `rtl/ppu.sv`
- `rtl/control_unit.sv`
- `rtl/control_top.sv`
- `software/compiler/tinynpu/isa.py`
- `software/compiler/tinynpu/program.py`
- `software/compiler/tinynpu_jit/lowering.py`

### 2. KV-cache append/write-offset support

Status: implemented for explicit slot offsets

Need:

- append one new token's `K` and `V` without rewriting the full cache
- explicit per-op write offset / cache index contract

Primary files:

- `rtl/control_unit.sv`
- `rtl/ppu.sv`
- `software/compiler/tinynpu/isa.py`
- `software/compiler/tinynpu/program.py`
- `software/compiler/tinynpu_jit/ir.py`
- `software/compiler/tinynpu_jit/lowering.py`

### 3. Decode-native cache layouts

Status: partial software contract

Need two separate layouts:

- `K` cache layout optimized for `QK^T`
- `V` cache layout optimized for `A @ V`

Notes:

- do not force one universal cache format
- both can be stored as packed/tiled words in the same unified buffer
- this is mainly a storage contract problem, not a separate memory-structure problem
- current JIT layer now has helper APIs for:
  - B-cache slot stride computation
  - named B-cache slot views
  - paired `K`/`V` cache declarations with per-slot metadata
- current hardware still treats these as ordinary B-packed cache regions; decode execution mode is still missing

Primary files:

- `software/compiler/tinynpu/program.py`
- `software/compiler/tinynpu/packer.py`
- `software/compiler/tinynpu_jit/lowering.py`
- `software/compiler/tinynpu_jit/memory_planner.py`

### 4. Decode execution mode on existing array

Status: not implemented

Need:

- reuse current array/MAC datapath
- support vector-by-matrix style streaming for:
  - `QK^T`
  - `A @ V`

This should be treated as a second dataflow mode on the same array, not a separate accelerator.

Primary files:

- `rtl/control_unit.sv`
- `rtl/control_top.sv`
- `rtl/skewer.sv`
- `rtl/systolic_array.sv`
- `rtl/tinynpu_top.sv`
- `software/compiler/tinynpu/isa.py`

### 5. Cheap A-side vector staging

Status: partially available through normal staging

Need:

- efficient staging for short vectors such as `Q` or attention weights
- no heavyweight matrix-style boundary assumptions for these inputs

This is not the main blocker, but it should be kept simple.

Primary files:

- `software/compiler/tinynpu_jit/tinynpu_runtime_v2.c`
- `software/compiler/tinynpu_jit/tinynpu_runtime_v2.h`
- `software/compiler/tinynpu_jit/templates/cv32e40p_runtime_template.c`

### 6. Score/output precision policy for attention

Status: not implemented

Need:

- wide accumulation for `QK^T`
- likely keep score boundary at `INT16` or float before host `softmax`

Primary files:

- `rtl/ppu.sv`
- `rtl/di_sigmoid.sv` (only if revisited for later on-chip nonlinear support)
- `software/compiler/tinynpu_jit/lowering.py`
- `software/compiler/tinynpu_jit/host_ops.py`

## Recommended Implementation Order

1. Add `B`-layout writeback.
2. Add cache append/write-offset support.
3. Define concrete `K` and `V` packed cache layouts.
4. Add decode execution mode for `QK^T` and `A @ V`.
5. Add a small benchmark for one attention head with host `softmax`.
6. Revisit fused `QKV` only after the separate `Q/K/V` path works.

## Questions To Resolve

- Exact `K` cache packing order for `QK^T`
- Exact `V` cache packing order for `A @ V`
- Whether current `B` packing can be reused directly or needs a variant
- How write offsets should be encoded in ISA/runtime
- Whether decode mode should expose one opcode or two (`QK^T` and `A @ V`)
