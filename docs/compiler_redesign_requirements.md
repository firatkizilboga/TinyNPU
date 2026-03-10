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

## 10. Decision Log (2026-03-10)

### 10.1 Initial Execution Model: Segmented Compile/Run

We will start with a segmented execution model because it is materially easier to implement and validate than a full whole-program compiler, while still creating the right long-term architecture.

Concretely, the compiler will:
- trace a PyTorch module or subgraph
- identify which operations are supported by TinyNPU
- group consecutive supported operations into explicit TinyNPU execution segments
- represent unsupported operations as explicit host-executed steps in the runtime plan

Example:
- PyTorch graph: `linear1 -> relu -> softmax -> linear2`
- Compiled runtime plan: `NpuSegment([linear1, relu]) -> HostOp(softmax) -> NpuSegment([linear2])`

This is what was previously referred to in planning discussion as `Option A`.

Why we are starting here:
- avoids solving full-program allocation/scheduling first
- allows a usable PyTorch frontend sooner
- enables explicit host/NPU boundaries for unsupported ops
- keeps the path open to whole-program compilation later

This does not mean the architecture will be throwaway. The plan/IR should still be designed so that multiple segments can later be merged into a larger whole-program flow.

### 10.2 Frontend Direction: PyTorch-First, Not a Device Backend

PyTorch is the authoring and tracing frontend. The first implementation is not a `torch.device("tinynpu")` backend.

Instead, the compiler/runtime flow will be:
- `compile_module(model, example_inputs)`
- produce a compiled execution plan
- runtime executes that plan on host-emulation or simulator backends

So PyTorch defines the program, but execution still goes through TinyNPU software/runtime artifacts, not through direct PyTorch device dispatch.

### 10.3 Initial NPU Lowering Scope

The first TinyNPU lowering pass will support only:
- `linear`
- `matmul`

Other operations, including `conv2d`, will be deferred until the segmented compiler/runtime path is stable end-to-end.

### 10.4 Unsupported Ops Become Explicit Runtime Plan Nodes

Unsupported ops such as `softmax` and `sigmoid` will not be handled implicitly.

Instead, the partitioner will generate explicit runtime plan nodes such as:
- `HostOp(kind="softmax")`
- `HostOp(kind="sigmoid")`

Runtime behavior for such a boundary will be:
- run preceding TinyNPU segment
- read the segment output tensor back into logical form
- execute the host op on CPU
- quantize/re-encode the result for the next TinyNPU segment
- continue execution

Fallback must always be explicit in the plan, never silent.

### 10.5 Memory Planning Policy

For v1, memory planning will be deterministic only.

Compiler behavior must be:
- produce a legal deterministic plan for the target UB/IM limits, or
- fail compilation explicitly

Forbidden behavior:
- silent overflow
- wraparound
- accidental address aliasing
- best-effort remapping that changes semantics without surfacing it

### 10.6 Verification Architecture

Verification is part of the runtime/execution-plan layer, not the RTL ISA layer.

There are two instruction layers:
- hardware ISA instructions executed by TinyNPU (`MATMUL`, `MOVE`, `HALT`)
- runtime-plan instructions executed by the host/runtime (`RunNpuSegment`, `RunHostOp`, `VerifyTensor`, etc.)

Compiler responsibilities:
- compute expected tensors using a shared golden-model API
- store those expected tensors in the compiled artifact
- emit verify points in the runtime plan

Runtime responsibilities:
- execute NPU and host steps in order
- execute verification steps when verification is enabled
- use the compiler-produced expected tensors for comparisons

### 10.7 Verification Policy

Normal runtime mode:
- verify nothing by default
- runtime trusts the NPU unless verification is explicitly enabled

Debug verification mode:
- verify final outputs
- verify any tensors explicitly flagged by the user for checking

Verification points must therefore be user-annotatable.

### 10.8 Verification Targeting Semantic

Users need a semantic way to flag tensors/vectors for verification.

Current direction:
- expose a PyTorch-facing helper such as `mark_for_verify(tensor, "label")`
- compiler preserves that intent into the execution plan
- in v1, `mark_for_verify(...)` forces a segment boundary so the tensor is materialized at the host/runtime layer
- runtime verifies flagged tensors when debug verification is enabled

Compiler should reject impossible verification requests clearly if a flagged tensor cannot be materialized at the requested point.

### 10.9 Scope Constraints

This work is intentionally scoped to software only:
- no RTL edits
- compiler/runtime/orchestration changes only
- focus area is `software/compiler` and related software-side integration

### 10.10 Planned Implementation Order

1. Document the new segmented compiler/runtime design in-repo.
2. Add a new PyTorch-facing compiler path alongside the current compiler.
3. Implement FX capture, logical plan types, capability checker, and partitioner.
4. Implement host-emulation execution first for validation.
5. Add simulator runtime integration using the same execution plan.
6. Extend op coverage after `linear` / `matmul` is stable.

Open follow-on work remains for:
- whole-program compilation
- more advanced memory planning
- broader op coverage
- richer fallback semantics
- eventual direct PyTorch/backend integration if we decide to build that later

### 10.11 Open Annoyances And Risks

These are not soft TODOs. They are known architectural gaps that must stay visible during migration.

1. PyTorch quantization metadata ingestion now exists, but only for a narrow supported set.
- The new FX path can lower explicit quant/dequant boundaries and standard `torch.ao.nn.quantized.Linear` / `Conv2d` modules.
- It now reads scale metadata and synthesizes TinyNPU `multiplier` / `shift` for those supported modules.
- This is not broad quantization support. It does not mean arbitrary PyTorch quantized graphs are now compiler-ready.

2. TinyNPU NPU segments currently require symmetric zero-point-free quantization.
- Current NPU lowering is intentionally strict:
  - signed integer activations / weights
  - `zero_point = 0`
  - scale-driven integer rescale lowered into `multiplier` / `shift`
- If a PyTorch model uses asymmetric quantization or nonzero zero-points in a would-be NPU segment, the compiler must fail closed.
- We must not fake support for a quant scheme the hardware cannot actually implement.

3. `quant-by-claude.py` export is aligned with the new runtime direction, but the QAT model objects are not yet a native compiler frontend.
- The export stage computes `M0` / `shift` and produces weights/biases/manifests that the new JIT path can already run.
- The old training-time custom modules (`QConv2d`, `QLinear`) started as an export-only bridge.
- They are now convertible into compiler-ready modules via `tinynpu_quant.conversion`, so a trained PyTorch QAT model can be compiled without going through the old manifest/export path.
- This is still a narrow contract, not broad support for arbitrary QAT graphs.

4. `HostOp -> NpuSegment` re-entry quantization is only partially generalized.
- The architecture allows `NpuSegment -> HostOp -> NpuSegment`.
- Explicit helper-based re-entry exists and PyTorch quant/dequant module boundaries exist.
- The remaining gap is broad, automatic boundary quantization policy for larger mixed graphs.
- This is especially relevant for attention-style models where `softmax` and normalization stay on host.

5. Old `A/B/C/BIAS` role semantics are still present in lowering.
- The new compiler IR is logical-first.
- The lowering bridge still maps tensors into old backend roles.
- This is acceptable as a migration bridge, but it must remain quarantined to lowering.

6. Packing/layout transforms are not yet first-class IR.
- Current v1 correctness path is: read packed output -> materialize logical tensor -> repack for next segment.
- That is semantically clean, but not the final optimized story.
- We still need explicit layout-transform semantics if we want robust direct segment-to-segment reuse later.

7. Current simulator inspection proves packed output correctness only for the supported narrow path.
- We can now compare expected packed vectors vs actual UB vectors on RTL for simple segmented matmul chains.
- This does not yet cover broader op classes or host-op re-entry.

8. Real PyTorch model integration still needs a dedicated quantization toolkit layer.
- Most real PyTorch models will not call `npu_matmul(...)` directly, and they should not need to.
- We need a PyTorch-side preparation pipeline that owns:
  - PTQ / QAT setup
  - calibration
  - sensitivity analysis
  - conversion into a compiler-supported quantized inference graph
- That toolkit should become `tinynpu_quant`, not stay trapped in a monolithic MNIST script.

### 10.12 True Capabilities Right Now

The document should reflect what is actually proven, not just what is planned.

Current proven JIT/compiler/runtime capabilities:
- segmented runtime-plan IR (`NpuSegment`, `HostOp`, `VerifyTensor`)
- compiler-owned expected tensors and runtime-owned verification
- host-emulation backend
- simulator backend with packed vector capture
- explicit verification boundaries via `mark_for_verify(...)`
- explicit host quant/dequant boundaries
- export-backed MNIST conv/fc execution on the new path
- full exported MNIST chain on RTL matching the old path
- ordinary quantized `torch.ao.nn.quantized.Linear` lowered to TinyNPU and validated on RTL
- ordinary quantized `torch.ao.nn.quantized.Conv2d` lowered through host `im2col` and validated on RTL
- structured runtime debug trace showing:
  - host quantize / dequantize
  - host im2col / layout restore
  - NPU segment inputs / outputs
  - verification points
- first reusable PyTorch-side quantization toolkit pieces now live in `tinynpu_quant`:
  - fused-parameter math
  - per-layer quant config objects
  - reusable QAT layer modules
  - calibration helpers
  - sensitivity-analysis helpers
- compiler-ready QAT conversion exists:
  - `QConv2d` / `QLinear` -> `CompilerReadyConv2d` / `CompilerReadyLinear`
  - mixed precision has been validated on the narrow path (`W4A4` + `W16A16`)
- a fresh pure-PyTorch MNIST pipeline exists in `software/workload/mnist_tinynpu_pipeline.py`:
  - train FP32
  - initialize/fine-tune QAT
  - convert to compiler-ready model
  - compile with `compile_module(...)`
- trained-model host/RTL parity now exists for the new MNIST pipeline:
  - fresh trained checkpoint path matches host and RTL across multiple deterministic real MNIST samples
  - full tensor-by-tensor comparison is exercised in `verification/cocotb/test_jit_mnist_trained_pipeline.py`

Current non-capabilities:
- arbitrary PyTorch float models are not automatically prepared for TinyNPU
- arbitrary FX quantization forms are not supported
- asymmetric zero-point NPU segments are not supported
- transformer attention is not a first-class supported workload yet
- attention-softmax / normalization / residual-style mixed graphs will still rely on explicit host fallback boundaries
- broad automatic host-boundary scale inference is not solved generally yet
  - the fresh MNIST pipeline now uses a calibrated `conv3 -> mean` boundary scale instead of reusing the next NPU layer's activation scale
  - this recovered compiled-host MNIST accuracy from catastrophic loss back to near-QAT quality on the validated slice
  - the general rule is now explicit: "next layer activation scale" is only valid for direct quantized NPU-to-NPU chains, not for host-sensitive ops such as `mean`
  - the percentile-based host-boundary scale logic is now being extracted into `tinynpu_quant.calibration` so it becomes reusable calibration infrastructure instead of MNIST-only glue

### 10.13 Current Design Decisions

These are active policy decisions, not open design goals.

Hardware/compiler boundary:
- Do not change RTL architecture just to add new packing primitives at this stage.
- Prefer software/compiler fixes first while the existing hardware remains stable and validated.
- If future hardware changes happen, they should come only after the current software path clearly bottoms out.

Packing/layout policy:
- Keep `A/B/C/BIAS` buried as backend compatibility details only.
- Do not let `C` semantics leak upward into frontend IR, planning, or runtime-plan meaning.
- Repacking remains a software/runtime responsibility at boundaries for now.
- Fusing back-to-back NPU ops is already treated as a primary way to reduce packing/repacking overhead, and should continue to be preferred where legal.

Boundary precision policy:
- `int16` boundary outputs are a valid compiler/runtime simplification strategy when UB allows them.
- They are especially attractive for host-sensitive boundaries because they preserve information without requiring immediate requantization.
- They are not assumed to be universally legal; UB capacity must still be checked and overflow must fail explicitly.
- When `int16` boundary preservation is not legal, the fallback is calibrated `int8` boundary scaling rather than blindly reusing the next layer activation scale.

Model/workload tradeoff policy:
- A stricter contract such as "dynamic activations are always consumed as lhs-style inputs" is considered a valid simplifying direction for CNN/MLP-style flows.
- That simplification is acknowledged to be hostile to transformer-like workloads, so it is not being locked in as a universal architectural rule yet.
- Transformer-like mixed host/NPU graphs remain in scope, but they are not allowed to force premature RTL architectural changes before the software path is better understood.

### 10.14 `tinynpu_quant` Plan

The next major layer should be a PyTorch-side quantization toolkit living alongside the compiler, not inside a single script.

Target responsibilities:
- define TinyNPU-supported quantization contracts
- prepare PyTorch models for PTQ or QAT
- run calibration
- run sensitivity analysis
- convert a trained/prepared model into a compiler-supported quantized inference graph
- share fused-parameter math with the compiler/runtime (`multiplier` / `shift` synthesis)

Near-term migration path from `quant-by-claude.py`:
1. move shared fused-parameter math into `tinynpu_quant`
2. extract reusable QAT/PTQ utilities from the script into package modules
3. keep model-specific MNIST code thin and outside the shared quant toolkit
4. converge the toolkit output with what `compile_module(...)` expects

Progress against that plan today:
- step 1 is done
- step 2 has started:
  - fake-quant utilities
  - `LayerQuantConfig`
  - `QConv2d` / `QLinear`
  - calibration helpers
  - sensitivity helpers
- the remaining gap is turning those extracted pieces into a compiler-prepared-model pipeline instead of just a cleaner script foundation

### 10.15 Transformer Constraint

Transformer-style models should stay in design scope even while near-term validation is CNN/MLP-heavy.

Implications:
- `softmax` and normalization are host-fallback-critical ops unless hardware changes later
- quantize -> dequantize -> host op -> requantize boundaries must be first-class and inspectable
- QKV projections and FFN matmuls are natural TinyNPU segment candidates
- sensitivity analysis must eventually reason about mixed placement, not just per-layer bit width
- debug tooling must expose boundary tensors clearly because attention failures are numerically subtle

### 10.16 Immediate Next Steps

This section is here so the project can survive context compaction without losing the real sequence.

Short-term execution order:
1. Continue extracting `quant-by-claude.py` into `tinynpu_quant`.
   - next concrete extraction targets:
     - PTQ/QAT orchestration helpers
     - export helpers
     - model-agnostic conversion helpers
2. Keep `quant-by-claude.py` as a thin model-specific workflow script that imports shared logic from `tinynpu_quant`.
3. Define one compiler-supported prepared-model contract for PyTorch quantization.
   - near-term target: standard quantized `torch.ao` modules plus explicit quant/dequant boundaries
   - training-time custom modules are not the compiler contract unless we explicitly choose to support them
4. Add a dedicated RTL smoke test for the ordinary quantized `Conv2d` frontend path.
   - done for the current narrow quantized `Conv2d` path
5. After the quant toolkit is more complete, teach `compile_module(...)` to accept the prepared output of that toolkit directly.
6. Use the fresh trained MNIST pipeline as the main fidelity-debug target.
   - host and RTL are aligned on the trained checkpoint path
   - a calibrated `conv3 -> mean` boundary scale fixed the largest remaining MNIST fidelity loss
   - remaining work should generalize this host-boundary handling beyond the current MNIST-specific pipeline

Guardrails for the next steps:
- do not claim support for asymmetric zero-point NPU segments unless the hardware path is real
- keep transformer-like mixed host/NPU graphs in scope while designing the quant toolkit APIs
- do not duplicate fused-parameter math in scripts again; shared quant math must stay in `tinynpu_quant`
