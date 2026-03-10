# TinyNPU Compiler Redesign Requirements (Draft v1)

## 1. Problem Statement
The current compiler API and test workload flow expose hardware packing details directly (`role`, precision packing, implicit remaps), which creates fragile behavior:
- logical tensor intent and physical UB layout are mixed
- layer chaining can silently depend on undocumented packed-layout semantics
- expected-value generation can diverge from hardware behavior
- running a program end-to-end requires too many manual steps

This document defines requirements for a redesign that keeps hardware capability while making program authoring and execution reliable and easy.

## 2. Scope
In scope:
- compiler front-end/API semantics
- internal IR/layout model
- compile/run/verify UX
- diagnostics and debug visibility
- compatibility and migration strategy

Out of scope:
- RTL architectural changes (unless explicitly approved later)
- new op types beyond current `MATMUL`, `MOVE`, and post-processing controls

## 3. Design Goals
1. Logical-first programming model: users describe math, not UB packing.
2. Deterministic behavior: same inputs produce same binary and verification output.
3. Explicit semantics: no hidden role mutation or implicit precision reinterpretation.
4. Fast iteration: one command to compile + run + verify.
5. Debuggability: mismatches should be explainable at op/tile/lane level.

## 4. Primary Users
- Accelerator developer writing synthetic workloads.
- Model engineer compiling layer sequences for TinyNPU.
- Verification engineer running cocotb regression and triaging mismatches.

## 5. Functional Requirements

### R1. Typed Logical Tensor Model
The compiler shall represent every symbol as a typed logical tensor with:
- shape
- logical dtype
- storage layout tag
- producer/consumer edges

`role` shall not be the primary user-facing semantic type.

### R2. No Implicit Role Mutation
Compiler passes shall never silently mutate a symbol's logical role/layout in-place.
If a different physical layout is required, the compiler shall create an explicit derived buffer node.

### R3. Explicit Layout Transform Nodes
IR shall include explicit transforms for layout/packing reinterpretation (e.g. `pack_c_for_b`, `unpack_int4`, `repack_int8`), so chaining semantics are visible and testable.

### R4. Precision Contract
Each matmul-like op shall have a single precision contract:
- input element precision
- accumulator precision (fixed by hardware)
- output precision + quantization config (`shift`, `multiplier`, activation)

The compiler shall validate that producer output precision/layout is compatible with consumer input expectations.

### R5. Golden Semantics as Shared Library
Expected-result computation shall use a shared, single-source implementation in compiler core (not duplicated ad-hoc in generators).
All workload generators shall call this API.

### R6. Deterministic Compilation
Compilation shall be deterministic given:
- program source
- seed (if random generation exists)
- hardware config

Generated `.npu` ordering and addresses shall be stable.

### R7. Hardware Config Isolation
Hardware constants loaded from RTL (`defines.sv`) shall be versioned into a compiler config snapshot in each artifact to avoid ambiguity.

### R8. End-to-End CLI
A first-class CLI shall support:
- `compile`
- `run` (sim)
- `verify`
- `compile-run-verify` (single command)

Example target UX:
```bash
tinynpu run software/workload/complex_chain_gen.py --sim verilator --verify
```

### R9. Failure Diagnostics
On mismatch, tool output shall include:
- first mismatch location
- expected vs actual
- op index/layer name
- precision/layout context

Optional verbose mode should dump tile/lane level traces.

### R10. Artifact Contract
Compiled artifact schema shall be versioned and documented.
It shall contain enough metadata to reproduce execution and verification without generator code.

### R11. Backward Compatibility Layer
Legacy Python API shall remain available for one transition period with deprecation warnings.

### R12. Verification Modes
Verification shall support:
- final-output-only (fast functional checks)
- full intermediate checks (debug mode)

This must be configurable per workload.

### R13. Compiler Memory Planning
The compiler shall own UB/IM memory planning instead of relying on declaration order side effects.
It shall plan allocations from tensor lifetimes in IR.

Minimum required baseline algorithm:
- linear-scan lifetime allocation over topologically ordered ops
- reuse freed buffers when size/layout/precision constraints allow

This gives a simple, deterministic starting point and can be upgraded later.

### R14. Memory Planning Policies
The compiler shall expose memory policy modes:
- `deterministic`: stable addresses for reproducibility/debug
- `compact`: prioritize UB footprint reduction

Policy choice shall be recorded in artifact metadata.

### R15. Allocation Explainability
Compiler output shall optionally emit a memory plan report including:
- per-symbol lifetime interval
- assigned address/size
- reuse decisions and conflicts

This report is required for debugging OOM or unexpected overlap behavior.

### R16. PyTorch Integration Compatibility
The redesign shall keep an explicit integration path for PyTorch-based flows.
At minimum:
- import tensors/weights/biases from PyTorch checkpoints or exported arrays
- map graph segments (initially GEMM/conv-im2col style) into TinyNPU ops
- preserve quantization metadata (scale/shift/multiplier equivalents) needed by backend

Initial integration can be exporter-first (offline compile), not runtime JIT.

### R17. Stable Python API Boundary
Provide a clean Python API layer intended for framework bridges (including PyTorch) that is:
- independent from low-level packing internals
- versioned
- tested with at least one reference PyTorch pipeline

### R18. Hermetic Runtime/Import Contract
Compiler and test entrypoints shall not depend on caller working directory side effects.
All runtime modules must resolve imports through package paths or explicit, centralized path bootstrap logic.

Acceptance expectation:
- running from any repository subdirectory through supported CLI/test commands must not fail with missing-module import errors.

### R19. One-Command Executability
A compiled workload shall be runnable through a single supported command without manual environment surgery (`PYTHONPATH` hacks, ad-hoc module swaps, or file moves).
If prerequisites are missing, the tool shall emit a clear actionable error list up front.

### R20. No Silent UB Overflow
The compiler shall guarantee that every generated program is UB-safe by construction.
For each op/layer, it must either:
- produce a legal memory/tiling/output-precision plan that fits UB, or
- fail compilation with an explicit OOM/placement diagnostic.

Silent address overflow, wraparound, or runtime reliance on accidental buffer corruption is forbidden.

## 6. Non-Functional Requirements
- Performance: compile time should not regress by >20% on existing workloads.
- Reliability: no silent fallback on semantic incompatibility; fail fast with actionable errors.
- Maintainability: new ops/layouts must be addable without modifying workload scripts broadly.
- Traceability: every compiled instruction should map back to a high-level op identifier.

## 7. Acceptance Criteria
1. All existing shipped workloads compile and run through the new CLI.
2. Existing regression suite passes under default mode.
3. At least one chained mixed-precision workload is validated in both:
- final-output-only mode
- intermediate-check mode
4. Compiler rejects an intentionally incompatible precision/layout chain with a clear message.
5. Removing all manual `role` manipulation from workload scripts remains possible for core use cases.
6. Memory planner reuses buffers on at least one multi-layer workload and reports the plan.
7. A reference PyTorch-to-TinyNPU export flow compiles and runs at least one model segment.
8. A stress workload that previously exceeded UB capacity is either:
- automatically remapped into a legal plan, or
- rejected at compile time with a clear memory report (required capacity vs available).

## 8. Migration Plan (High Level)
Phase 1:
- Introduce new IR + shared golden library.
- Keep legacy API active with adapter layer.

Phase 2:
- Add CLI orchestration and artifact schema versioning.
- Migrate workload generators to new API.

Phase 3:
- Deprecate direct low-level role-based APIs.
- Keep a low-level "expert mode" namespace for advanced users.

## 9. Open Decisions For Us
1. Should we keep `role` visible in expert mode, or remove it completely from public API?
2. Should intermediate tensor checking be opt-in by default, or enabled automatically only on failures?
3. Do we want one unified artifact (`.npu`) or split logical program + packed binary outputs?
4. Should CLI own cocotb invocation directly, or call existing makefiles as a backend?
5. How strict should compatibility be with old generator scripts (best-effort vs guaranteed)?
6. Memory planning default: prioritize deterministic addresses or compact footprint?
7. PyTorch integration depth for v1: exporter-only or include graph capture helpers?
8. Should we make a strict "supported invocation matrix" and fail closed outside it, or attempt best-effort execution?
