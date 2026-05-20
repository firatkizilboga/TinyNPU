# XFORM Q/DQ Utilization - 2026-05-20

## Objective

Use the NPU XFORM datapath for quantize/dequantize boundaries instead of leaving the cost in CPU-side runtime loops.

## Runtime And Compiler Change

Two XFORM Q paths now exist:

- Preferred compiler path: emit `XFORM Q_F16_I16` directly in the same segment IM stream before the consuming GEMM.
- Runtime fallback path: `TNPU_WRITE_XFORM_Q_F16_I16` stages FP16-bit tensors into the requested A/B layout and invokes the hardware XFORM Q instruction in-place through a scratch IM program.

Before this change, the Runtime V2 branch named `TNPU_WRITE_XFORM_Q_F16_I16` still used `tnpu_write_tensor_a_qf16_to_i16_fast`, a CPU-side FP16 decode plus INT16 quantize loop. That meant the compiler emitted an XFORM-looking boundary, but the hot runtime path did not actually exercise the hardware XFORM unit.

The preferred path is now:

```text
FP16-bit tensor in CPU memory
  -> raw A/B-layout write to shared SRAM by Runtime V2
  -> segment IM: hardware XFORM Q_F16_I16 in-place
  -> segment IM: GEMM consumes INT16 tensor
```

This preserves the existing compiler contract:

- source tensor dtype is `INT16`
- source tensor value encoding is `FP16_BITS`
- destination role is A or B
- precision is INT16
- zero point is 0

## RTL Measurement

Program: recovered resident-handoff four-layer is-zero MLP.

The previous resident-handoff result used CPU-side quantization for the first FP16 boundary:

| Runtime path | Preload | Hot body | Cold e2e | Extrapolated 10x e2e | Output |
| --- | ---: | ---: | ---: | ---: | --- |
| Resident handoff, CPU quant stage | `26,781` | `16,679` | `43,460` | `193,571` | `0.500`, PASS |
| Resident handoff, hardware XFORM Q | `26,788` | `9,280` | `36,068` | `119,588` | `0.500`, PASS |

Improvement from using hardware XFORM Q:

| Metric | Speedup |
| --- | ---: |
| Hot body | `1.80x` |
| Cold e2e | `1.20x` |
| Extrapolated 10x e2e | `1.62x` |

Verbose step timing for the hardware-XFORM run:

```text
preload.ub_image cycles=26417
preload.im_inner_fc1 cycles=87
preload.im_inner_fc2 cycles=87
preload.im_inner_fc3 cycles=87
preload.im_inner_fc4 cycles=87
segment.inner_fc1.stage cycles=4901
segment.inner_fc1.run cycles=940
segment.inner_fc1.readback cycles=6
segment.inner_fc2.stage cycles=4
segment.inner_fc2.run cycles=940
segment.inner_fc2.readback cycles=6
segment.inner_fc3.stage cycles=4
segment.inner_fc3.run cycles=940
segment.inner_fc3.readback cycles=6
segment.inner_fc4.stage cycles=4
segment.inner_fc4.run cycles=148
segment.inner_fc4.readback cycles=1203
```

The important change is the first segment stage:

- old CPU FP16 decode/quantize/pack stage: `11,449` cycles
- raw INT16 A-layout stage on similar 64-word input: `4,994` cycles
- new FP16-bit stage plus hardware XFORM Q: `4,901` cycles

So the previous `~6.5k` cycle CPU quantization tax is effectively removed for this boundary.

The resident MLP measurement above used the runtime fallback hardware-XFORM path, because it was rebuilt from the recovered generated program. A smaller current-compiler RTL probe verifies the preferred in-segment path:

```text
input:  [1.0, -2.0, 3.0, -4.0, 0.5, -0.5, 2.5, -1.5]
scale:  0.5
flow:   FP16-bit input -> in-segment XFORM Q -> identity GEMM
output: [2, -4, 6, -8, 1, -1, 5, -3]
status: EXIT SUCCESS
```

Verbose timing for that probe:

```text
preload.ub_image cycles=145
preload.im_seg0 cycles=119
segment.seg0.stage cycles=711
segment.seg0.run cycles=148
segment.seg0.readback cycles=1207
```

A second current-compiler RTL probe verifies the same in-segment path for an RHS/B-layout tensor:

```text
input:  B-layout FP16-bit identity matrix with diagonal value 1.0
scale:  0.5
flow:   FP16-bit RHS -> in-segment XFORM Q -> GEMM
output: [2, 4, 6, 8, 10, 12, 14, 16]
status: EXIT SUCCESS
```

Verbose timing for that RHS/B probe:

```text
preload.im_seg0 cycles=119
segment.seg0.stage cycles=4098
segment.seg0.run cycles=148
segment.seg0.readback cycles=1210
segment.seg0.npu cycles=11913
```

## DQ Status

Hardware XFORM DQ is present and passes RTL acceptance. The compiler folds dequantize-to-FP16 boundaries into NPU programs, including final outputs when the final output contract is FP16 bits instead of host float32.

Current limitation:

- `TNPU_READ_DEQUANTIZE_INT16_TO_FLOAT32` still performs CPU readback and float32 conversion.
- This is appropriate for final host-visible float outputs.
- It is not ideal for an internal boundary that only needs FP16 bits for another NPU/host-low-precision operation.

Final-output DQ can therefore be either:

- `INT16 -> FLOAT32` on CPU when the public output tensor is declared as float32;
- `INT16 -> FP16 bits` with hardware XFORM DQ when the public output tensor is declared as int16 storage with FP16-bit value encoding.

The policy should be:

- Use hardware XFORM Q for FP16-bit to INT16 NPU input boundaries.
- Use hardware XFORM DQ for INT16 to FP16-bit internal or final boundaries.
- Keep CPU float32 dequantize only for final outputs or host ops that truly require float32.

## Verification

Compiler/runtime regression:

```bash
pytest -q software/compiler/tests/test_baremetal_emit.py software/compiler/tests/test_rtl_runner.py -k "xform_q_f16_i16 or runtime_v2_xform_write or dequantize_fp16_to_xform or dequant_boundaries or rhs_fp16_quantize"
```

Result:

```text
6 passed, 79 deselected
```

The compiler tests now check that Q boundaries are emitted as segment-local XFORM instructions, not as runtime descriptor transforms:

```text
descriptor write transform: TNPU_WRITE_TRANSFORM_NONE
first segment IM opcode:    XFORM
first segment IM mode:      Q_F16_I16
```

XFORM RTL acceptance:

```bash
cd verification/cocotb
make -f Makefile.npu SIM_BUILD=sim_build_accept_xform TOPLEVEL=tinynpu_top MODULE=test_xform_shared CCACHE_DISABLE=1
```

Result:

```text
test_xform_shared.test_xform_q_f16_i16_one_word  PASS
test_xform_shared.test_xform_dq_i16_f16_one_word PASS
TESTS=2 PASS=2 FAIL=0 SKIP=0
```

Current compiler/runtime MLP cocotb integration:

```bash
cd verification/cocotb
make -f Makefile.npu SIM_BUILD=sim_build_accept_iszero TOPLEVEL=tinynpu_top MODULE=test_jit_iszero_mlp_runtime CCACHE_DISABLE=1
```

Result:

```text
host dq_out [[-0.671875]]
rtl dq_out [[-0.671875]]
test_jit_iszero_mlp_runtime PASS
```

Resident MLP RTL run:

```text
body.kind=npu_only
preload.total cycles=26788
cold.body cycles=9280
cold.e2e cycles=36068
warm.avg.body cycles=9280
extrapolated.10x.e2e cycles=119588
Final outputs:
dq_out shape=(1, 1)
  row 0: 0.500
EXIT SUCCESS
```

Current-compiler in-segment XFORM Q RTL probe:

```text
TinyNPU runtime v2 program: cv32e40p_xform_q_insegment_identity
preload.ub_image cycles=145
preload.im_seg0 cycles=119
segment.seg0.stage cycles=711
segment.seg0.run cycles=148
segment.seg0.readback cycles=1207
Final outputs:
y shape=(1, 8)
  row 0: 2 -4 6 -8 1 -1 5 -3
EXIT SUCCESS
```

Current-compiler final-output XFORM DQ RTL probe:

```text
TinyNPU runtime v2 program: cv32e40p_xform_dq_final_f16
preload.ub_image cycles=145
preload.im_seg0 cycles=119
segment.seg0.stage cycles=711
segment.seg0.run cycles=148
segment.seg0.readback cycles=1207
Final outputs:
y_f16 shape=(1, 8)
  row 0: 15360 -16384 16896 -15360 14336 -18432 16640 -16896
EXIT SUCCESS
```

Those output integers are FP16 bit patterns, e.g. `15360 = 0x3c00 = 1.0` and `-16384 = 0xc000 = -2.0`.

Current-compiler RHS/B in-segment XFORM Q RTL probe:

```text
TinyNPU runtime v2 program: cv32e40p_xform_q_rhs_insegment
preload.im_seg0 cycles=119
NpuSegment: seg0
segment.seg0.stage cycles=4098
segment.seg0.run cycles=148
segment.seg0.readback cycles=1210
segment.seg0.npu cycles=11913
Final outputs:
y shape=(1, 8)
  row 0: 2 4 6 8 10 12 14 16
EXIT SUCCESS
```

## Remaining Work

The main remaining gap is broader coverage, not the core Q/DQ mechanism:

- Q is currently inserted before GEMM for folded FP16-bit quantize boundaries on segment LHS/A and RHS/B inputs.
- DQ is inserted after GEMM for folded dequantize-to-FP16 internal and final boundaries.
- Final `INT16 -> FLOAT32` readback still stays on CPU by design, but final `INT16 -> FP16 bits` uses hardware XFORM DQ.
- The Python RTL simulator now supports A-layout readback so A-layout resident intermediates can still be verified in integration tests.
