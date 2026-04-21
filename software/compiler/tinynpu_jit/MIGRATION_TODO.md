# TinyNPU JIT Migration TODO

This compiler path works, but the architecture is inverted.

Today the system behaves roughly like this:

- `partitioner.py` decides semantics, expected tensors, and low-level layout
- `ir.py` mostly exposes one real NPU op (`MatMulOp`) plus stringly-typed
  `HostOp(kind="...")`
- `lowering.py` and `memory_planner.py` reconstruct storage/layout policy from
  metadata and special cases
- emitters and runtime know too much about high-level meaning because the IR
  did not make it explicit

The problem is not just code size. The real problem is that the semantic
boundaries are wrong:

- the IR does not describe the program clearly enough
- the frontend does too much at once
- storage policy is smuggled through metadata and role inference
- emitters have to compensate for information the IR never made explicit

The goal of this migration is not a flag day rewrite. The shippable path is:

1. introduce a typed, hand-authorable IR layer alongside the current one
2. lower that typed IR into the existing `HostOp`/`NpuSegment` plan
3. move frontend and storage policy over incrementally
4. only then simplify emitters and delete legacy paths

## Immediate target

Add a typed high-level IR and builder that can describe a decoder block without
depending on the FX partitioner:

- typed ops such as `Linear`, `LayerNorm`, `Softmax`, `RoPE`, `KVAppend`,
  `Attention`, `Add`, `Mul`, `Concat`
- a small `IRBuilder`/fluent API to hand-author block plans
- a lowering pass from the typed IR into the existing
  `HostOp`/`MatMulOp`/`NpuSegment` plan

This gives us a clean path for hand-tuned emitters like `GPT2DecoderBlock`
without needing to solve the whole frontend first.

## What should be modularized

The compiler should be split by responsibility, not by historical file growth.

### 1. Typed semantic IR

This layer should describe the model/program, not the runtime tricks.

Examples:

- `Input`
- `Constant`
- `Linear`
- `LayerNorm`
- `RMSNorm`
- `Softmax`
- `Add`
- `Mul`
- `Concat`
- `RoPE`
- `KVAppend`
- `AttentionScore`
- `AttentionValue`
- `OutputProjection`

This layer should not care about:

- UB addresses
- `A/B/C` storage roles
- `b_read_mode`
- cache scatter helper names
- whether a softmax is emitted as `softmax` or `softmax_f16`

### 2. Builder layer

The builder should be the human authoring surface for hand-tuned blocks.

Examples:

- `builder.linear(...)`
- `builder.layernorm(...)`
- `builder.attention_prefill(...)`
- `builder.attention_decode(...)`
- `builder.kv_append(...)`
- `builder.output(...)`

This is where a `GPT2Block`, `LlamaBlock`, or `TinyLlamaDecodeBlock` should be
authored, not inside scripts as manual `HostOp`/`NpuSegment` lists.

### 3. Lowering passes

This is where semantic ops turn into the existing low-level execution plan.

Examples:

- `Linear` -> `MatMulOp`
- `LayerNorm` -> `HostOp("layernorm")`
- `RoPE` -> host rope or `rope_cs_name`-driven XFORM
- `KVAppend` -> cache append writeback or scatter host op
- `AttentionScore` -> score segment + scale + mask + softmax

This is also where storage policy should be decided:

- output layout
- cache append vs materialize
- persistent storage class
- absorbed quant/dequant boundaries

### 4. Frontend

FX handling should become a translation layer into the typed IR, not directly
into low-level plan steps.

The frontend should be:

- registry-based
- per-op-family
- easy to override for custom modules

Examples:

- `frontend/linear.py`
- `frontend/norm.py`
- `frontend/attention.py`
- `frontend/quant.py`
- `frontend/custom_blocks.py`

### 5. Golden / expected-tensor pass

Expected tensor tracking should be a separate pass over IR, not entangled with
frontend graph partitioning.

This pass should:

- shadow-execute typed IR or low-level plan
- materialize expected tensors
- attach verification labels

It should not be fused into graph matching logic.

### 6. Storage planner

The planner should consume explicit storage classes, not infer too much from
usage accidents.

The biggest current smell is the cross-segment `{"A", "C"}` relaxation in
`memory_planner.py`. That is compensating for missing concepts.

The planner should eventually reason about:

- canonical persistent storage
- segment-local staging roles
- cache storage vs cache view
- constants vs persistent activations vs local temporaries

instead of guessing from per-segment usage and metadata patches.

### 7. Emitters

Emitters should be low-level codegen only.

They should not be responsible for reconstructing high-level semantics from:

- host-op string kinds
- layout metadata
- cache naming conventions

After typed IR lowering exists, emitters should only see:

- concrete low-level ops
- concrete storage/layout decisions
- concrete verification requirements

## Staged migration plan

1. Introduce typed IR ops alongside the existing IR.
   - Keep `HostOp` as a real escape hatch for one-off runtime helpers.
   - Keep `MatMulOp`/`NpuSegment` as the low-level target for now.
   - Add lowering from typed ops to the current low-level plan.

2. Add an `IRBuilder`.
   - Builder should be the main way to author block-level plans by hand.
   - The first proof point should be a single transformer block or attention
     core that compiles end-to-end without going through the FX partitioner.

3. Split the FX frontend into a registry.
   - Replace the monolithic `partition_fx_graph()` control flow with a dispatch
     table keyed by module type / function target.
   - Move op-family handling into separate frontend modules.
   - Make it possible to override a high-level module such as
     `GPT2DecoderBlock` with a custom emitter instead of recursively tracing
     into internals.

4. Extract golden/expected-tensor tracking into its own pass.
   - The partitioner currently does graph interpretation, expected tensor
     tracking, and IR construction at the same time.
   - Split shadow execution from IR construction to make both simpler.

5. Consolidate emitters.
   - Prefer `baremetal_emit_v2.py` as the surviving path.
   - After typed IR lowering exists, split the surviving emitter by category:
     matmul emission, xform emission, host emission.
   - Delete the old emitter once the remaining call sites are migrated.

## Concrete TODOs by file

### `ir.py`

- Add typed op dataclasses next to the current low-level ones.
- Add a common base/protocol for plan steps if needed.
- Distinguish clearly between:
  - semantic IR
  - low-level execution IR
- Stop overloading `metadata` for semantic meaning where a typed field would be
  clearer.

### `partitioner.py`

- Split FX pattern matching into a registry.
- Move op-family handlers into separate modules.
- Stop constructing `HostOp`/`NpuSegment` directly for everything.
- Stop computing expected tensors inline.

### `lowering.py`

- Add lowering entry points from typed IR to low-level plan.
- Move current rewrites into named passes.
- Replace ad hoc rewrites with a pass pipeline:
  - semantic simplification
  - storage annotation
  - quant boundary absorption
  - backend lowering

### `memory_planner.py`

- Replace role-conflict relaxation with explicit persistent-storage semantics.
- Separate storage classes from consumer roles.
- Make cache base/view handling first-class instead of metadata-driven.

### `host_ops.py`

- Keep only true host intrinsics/runtime helpers here.
- Remove semantic ownership of ops that should become typed IR nodes.
- Split registry/eval/benchmark logic by op family if the file keeps growing.

### `baremetal_emit.py`

- Treat this as deprecated after v2 parity is achieved.
- Do not add new architecture here unless required for compatibility.

### `baremetal_emit_v2.py`

- Split into:
  - tensor/image declarations
  - host step emission
  - segment/program emission
  - verification emission
- Keep it focused on low-level codegen only.

## Highest-value first slice

If time is limited, do only this:

1. typed semantic IR
2. `IRBuilder`
3. lowering from typed IR to current low-level plan
4. one hand-authored transformer block proof point

That alone gives:

- a clean authoring path for decoder blocks
- less dependence on `partitioner.py`
- a place to put model-family logic without polluting emitters

Everything else is valuable, but secondary.

## Proposed target directory shape

One reasonable target structure is:

- `ir_semantic.py`
- `ir_lowlevel.py`
- `builder.py`
- `passes/`
  - `rewrite.py`
  - `storage.py`
  - `quant.py`
  - `verify.py`
- `frontend/`
  - `registry.py`
  - `linear.py`
  - `norm.py`
  - `attention.py`
  - `custom_blocks.py`
- `backend/`
  - `lower_to_lowlevel.py`
  - `emit_v2/`
    - `program.py`
    - `host.py`
    - `segment.py`
    - `verify.py`

The exact names are flexible. The important part is the dependency direction:

- frontend -> semantic IR
- semantic IR -> lowering passes
- lowering passes -> low-level plan
- low-level plan -> emitter/runtime

not the other way around.

## Files to start with

- `software/compiler/tinynpu_jit/ir.py`
- `software/compiler/tinynpu_jit/partitioner.py`
- `software/compiler/tinynpu_jit/lowering.py`
- `software/compiler/tinynpu_jit/baremetal_emit_v2.py`
- `software/compiler/tinynpu_jit/host_ops.py`

## Current invariants to preserve during migration

- Existing manually-authored `ExecutionPlan`s must keep compiling.
- Existing scripted demos should not need a full rewrite during step 1.
- Host emulation, memory planning, and bare-metal emission should continue to
  consume the current low-level `ExecutionPlan` shape until typed IR lowering is
  ready.
- The migration should be incremental and bisectable.

## Things not to do

- Do not rewrite the whole compiler in one branch.
- Do not make emitters depend directly on the new semantic IR.
- Do not add more semantic meaning to `HostOp(kind=str)` if a typed op is the
  better abstraction.
- Do not keep growing `partitioner.py` while claiming the migration is underway.
