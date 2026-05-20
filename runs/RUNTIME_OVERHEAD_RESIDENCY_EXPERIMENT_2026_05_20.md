# Runtime Overhead Residency Experiment - 2026-05-20

## Question

The CPU+NPU path is pure-compute faster, but small end-to-end runs are not much better than the ONNX CPU baseline. This experiment tests whether Runtime V2 is losing cycles by round-tripping intermediate NPU tensors through CPU memory instead of keeping them resident in shared SRAM.

## Experiment

Starting point: recovered four-segment is-zero MLP Runtime V2 program.

Original generated segment chain:

```text
fc1: stage x      -> run -> readback h1
fc2: stage h1     -> run -> readback h2
fc3: stage h2     -> run -> readback h3
fc4: stage h3     -> run -> readback final dq_out
```

Resident-handoff variant:

```text
fc1: stage x      -> run -> keep h1 in UB
fc2: use h1 in UB -> run -> keep h2 in UB
fc3: use h2 in UB -> run -> keep h3 in UB
fc4: use h3 in UB -> run -> readback final dq_out only
```

The experiment only changed generated segment descriptors:

- `fc1`, `fc2`, and `fc3` intermediate readbacks were removed.
- `fc2`, `fc3`, and `fc4` intermediate stages were removed.
- Final `fc4` readback stayed enabled.

This tests whether the NPU output layout for this row-vector MLP chain is already consumable by the next NPU segment. It passed, so the old readback/restage path was not required for correctness in this case.

## RTL Results

All runs used the same current RTL and current hard-sigmoid PPU contract. Runs were rebuilt with `TNPU_RUNTIME_V2_VERBOSE_STEPS=0` so timing is not polluted by diagnostic printing.

| Case | Preload | Cold body | Cold e2e | Warm body | Extrapolated 10x e2e | Output |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Original Runtime V2 | `26,781` | `54,833` | `81,614` | `54,833` | `575,111` | `0.500`, PASS |
| Resident handoff | `26,781` | `16,679` | `43,460` | `16,679` | `193,571` | `0.500`, PASS |

Improvement:

| Metric | Speedup |
| --- | ---: |
| Cold e2e | `1.88x` |
| Hot body | `3.29x` |
| Extrapolated 10x e2e | `2.97x` |

## Comparison To ONNX CPU

ONNX CPU baseline for the same tiny MLP family: `108,449` cycles.

Clock assumptions:

- CPU-only: `57.1 MHz`.
- CPU+NPU real-BRAM estimate: `39.17 MHz`.
- Clock adjustment factor: `0.686`.

| Case | Cycle speedup vs ONNX CPU | Clock-adjusted wall speedup vs ONNX CPU |
| --- | ---: | ---: |
| Original Runtime V2 cold | `1.33x` | `0.91x` |
| Original Runtime V2 hot | `1.98x` | `1.36x` |
| Resident handoff cold | `2.50x` | `1.71x` |
| Resident handoff hot | `6.50x` | `4.46x` |

This is the key result: with resident intermediate tensors, the CPU+NPU path beats the ONNX CPU baseline even after the current clock-frequency penalty. Without residency, the cold one-shot path is still slower in wall time than ONNX CPU.

## What Runtime V2 Is Doing Wrong

Runtime V2 currently treats every NPU segment boundary as a CPU-visible boundary:

1. Read the segment output from UB into a CPU tensor.
2. Later write that CPU tensor back into UB for the next segment.

For a chain of NPU-supported layers, that is unnecessary. It turns shared SRAM into a copy-back protocol instead of using it as local accelerator storage.

For the recovered MLP, the unnecessary round trips are exactly the difference between:

- original hot body: `54,833` cycles
- resident hot body: `16,679` cycles

So about `38k` cycles per token/run are avoidable on this small MLP.

## Real-Life Shared-SRAM Policy

In a real system, shared SRAM should not mean "CPU copies every intermediate." The better policy is:

- Static weights are loaded once and reused across many invocations.
- NPU-only intermediate activations stay resident in accelerator SRAM.
- CPU reads an intermediate only when a host op actually consumes it.
- CPU writes an intermediate only when a host op produced or modified it.
- The compiler tracks tensor residency and layout compatibility.

For this MLP chain, the producer output layout was directly reusable by the next NPU segment. That should become an explicit compiler/runtime contract, not an accidental generated-descriptor hack.

For transformer blocks, the same idea applies but needs layout-aware boundaries:

- Projection outputs consumed by another NPU matmul should stay resident.
- Values needed by CPU softmax/RoPE/RMSNorm must cross the host boundary.
- KV-cache append should write directly to the final cache layout when possible, not read back and scatter through CPU unless layout or RoPE requires it.

## Implementation Direction

The next compiler/runtime change should be a tensor-residency pass:

1. For each NPU segment output, mark `(tensor, UB address, layout, precision)` as resident.
2. For a later NPU segment input, if the same tensor is resident at the required address/layout/precision, omit the stage write.
3. If the tensor is not consumed by a host op, final output, or verifier, omit the readback.
4. If layout is not compatible, insert an explicit layout transform instead of silently round-tripping through CPU.
5. Keep final output and host-op boundaries explicit.

This directly targets the gap to ONNX: ONNX generated C does not pay accelerator boundary costs, so CPU+NPU must avoid unnecessary boundaries and use the NPU only when data reuse amortizes the remaining transfers.

## Implementation Update

A layout-aware compiler/emitter residency path was added after this experiment:

- Lowering now looks across segment boundaries. If a segment output is consumed only as the `lhs` of a later NPU segment, the producer writes it in A-layout. If it is consumed only as the `rhs`, the producer writes it in B-layout.
- The emitter now builds producer/consumer information for NPU segment outputs.
- A segment output can be kept resident only if all non-verify consumers are NPU segments.
- The producer and consumer UB symbols must match address, word count, precision, dtype, and layout role.
- Unsafe cross-role handoff, especially raw `C` output to `A` input, is explicitly rejected.

The initial MLP descriptor experiment showed the potential speedup, but follow-up exact matmul RTL probes showed that generic `C -> A` handoff is not layout-correct. The earlier MLP still produced the same coarse hard-sigmoid output, but that is not strong enough evidence for a general compiler optimization. The implemented fix therefore changes producer writeback layout first, then enables same-layout residency.

An exact `1x64 -> 1x64 -> 1x1` two-segment RTL probe now passes with resident handoff:

```text
preload.total cycles=9417
cold.body cycles=6397
cold.e2e cycles=15814
warm.avg.body cycles=6397
extrapolated.10x.e2e cycles=73387
Final outputs:
y shape=(1, 1)
  row 0: 2080
EXIT SUCCESS
```

This probe previously failed when the compiler blindly skipped the boundary without changing the producer layout. It now passes because the first segment writes the intermediate in A-layout and the second segment consumes it directly as A-layout input.

Remaining performance work for broader model coverage requires one of these layout mechanisms:

- emit the producer writeback directly in the consumer's required layout, e.g. A-layout writeback for the next segment;
- add a cheap in-NPU/shared-SRAM `C -> A` transform;
- or extend the consumer read path so it can consume C-layout activations directly.

The first item is implemented for simple single-consumer NPU-to-NPU chains. Multi-consumer cases, host-op boundaries, and final outputs still keep explicit readback/stage behavior.

Verification:

```bash
python3 -m py_compile software/compiler/tinynpu_jit/baremetal_emit_v2.py software/compiler/tests/test_baremetal_emit.py
pytest -q software/compiler/tests/test_baremetal_emit.py -k "resident or fused"
make -f Makefile.npu SIM_BUILD=sim_build_accept_iszero TOPLEVEL=tinynpu_top MODULE=test_jit_iszero_mlp_runtime CCACHE_DISABLE=1
```

Results:

- Python syntax: PASS.
- Emitter residency/fused-sigmoid tests: PASS, `5 passed, 74 deselected`.
- Runtime V2 MLP cocotb integration: PASS, host and RTL both produced `dq_out=-0.671875`.
- Exact `1x64` resident handoff bare-metal RTL probe: PASS, output `2080`.
