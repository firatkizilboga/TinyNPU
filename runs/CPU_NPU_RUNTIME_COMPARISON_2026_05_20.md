# CPU, ONNX, and CPU+NPU Runtime Comparison - 2026-05-20

## Clock Context

- CPU-only routed point: `57.1 MHz`.
- CPU+NPU real-BRAM estimate: `39.17 MHz`.
- Clock adjustment factor: `39.17 / 57.1 = 0.686`.
- Therefore CPU+NPU needs `>1.46x` cycle speedup before it wins wall time against CPU-only.

## Direct Comparison Table

| Model + config | CPU Runtime | ONNX CPU | CPU+NPU Runtime |
| --- | ---: | ---: | ---: |
| MLP, 4-layer is-zero | `248,383` cycles | `108,449` cycles | `112,081` cycles |
| Conv4, no-gather historical | `1,054,736` cycles | `259,688` cycles | `170,528` cycles |
| TinyLM, T5 transformer body | not available | `674,863` cycles | not comparable |
| TinyLM, T9 transformer body | not available | `1,235,876` cycles | not comparable |
| QLlama decode, `d32 h8 nh4 nkv2 f32 T8` | `224,719` cycles | not available | `187,030` cold / `172,789` hot |
| QLlama decode, `d48 h8 nh6 nkv3 f48 T8` | `426,993` cycles | not available | `271,712` cold / `241,336` hot |
| QLlama decode, `d64 h16 nh4 nkv2 f64 T8` | `668,974` cycles | not available | `330,045` cold / `277,909` hot |
| GPT2 two-block, `d16 h16 nh1 f16 T4` | `729,486` cycles | not available | `522,997` cold / `498,500` hot |
| GPT2 two-block, `d24 h24 nh1 f24 T4` | `1,333,079` cycles | not available | `757,888` cold / `709,199` hot |

Notes:

- MLP CPU+NPU uses the recovered current hard-sigmoid PPU contract. The old CPU-only MLP expected DI-sigmoid output, so the cycle comparison is useful for runtime overhead, but the exact model semantics changed.
- Conv4 CPU+NPU is historical from the last good no-gather INT16 result. It is not a fresh current-state signoff.
- ONNX CPU is available for MLP, Conv4, and TinyLM only. QLlama/GPT2 comparisons are Runtime CPU-only vs Runtime CPU+NPU.

## Speedups

| Model + config | Cycle speedup vs CPU Runtime | Wall speedup after clock penalty | Speedup vs ONNX CPU |
| --- | ---: | ---: | ---: |
| MLP, 4-layer is-zero | `2.22x` | `1.52x` | `0.97x` |
| Conv4, no-gather historical | `6.19x` | `4.24x` | `1.52x` |
| QLlama decode, `d32 h8 nh4 nkv2 f32 T8` hot | `1.30x` | `0.89x` | not available |
| QLlama decode, `d48 h8 nh6 nkv3 f48 T8` hot | `1.77x` | `1.21x` | not available |
| QLlama decode, `d64 h16 nh4 nkv2 f64 T8` hot | `2.41x` | `1.65x` | not available |
| GPT2 two-block, `d16 h16 nh1 f16 T4` hot | `1.46x` | `1.00x` | not available |
| GPT2 two-block, `d24 h24 nh1 f24 T4` hot | `1.88x` | `1.29x` | not available |

## Why Small CPU+NPU Runs Lose

The recovered MLP run shows the runtime problem clearly:

| Component | Cycles | Share of CPU+NPU total |
| --- | ---: | ---: |
| Preload | `26,765` | `23.9%` |
| Segment staging | `26,431` | `23.6%` |
| Actual NPU run | `2,954` | `2.6%` |
| Readback | `27,188` | `24.3%` |
| Other segment overhead | included in segment totals | remaining overhead |
| Total | `112,081` | `100%` |

For this tiny MLP, the systolic body is only about `2.6%` of end-to-end CPU+NPU time. The runtime is dominated by preload, stage, readback, and per-segment overhead. That is why CPU+NPU can beat the internal Runtime CPU-only path but still fails to clearly beat the ONNX CPU baseline on tiny workloads.

## Scale Needed

With the current clock estimates, CPU+NPU needs `>1.46x` cycle speedup to beat CPU-only wall time.

Observed transformer scale points:

- QLlama decode starts winning wall time at `d48 h8 nh6 nkv3 f48 T8`: `1.77x` cycle speedup, `1.21x` wall speedup.
- QLlama decode is a clearer win at `d64 h16 nh4 nkv2 f64 T8`: `2.41x` cycle speedup, `1.65x` wall speedup.
- GPT2 two-block is break-even at `d16 h16 nh1 f16 T4`: `1.46x` cycle speedup, `1.00x` wall speedup.
- GPT2 two-block is a real win at `d24 h24 nh1 f24 T4`: `1.88x` cycle speedup, `1.29x` wall speedup.

Practical conclusion: tiny toy layers are too small to amortize CPU+NPU runtime overhead. The accelerator becomes defensible at transformer-like sizes around QLlama `d48+` decode or GPT2-like two-block `d24+`, and becomes cleaner at QLlama `d64 h16`.

## Runtime Improvements To Target

The next improvements should attack the runtime overhead, not the systolic math:

- Keep tensors resident across segments and avoid readback unless the CPU actually consumes the tensor.
- Fuse adjacent NPU segments where possible so IM launch, stage, and readback are paid once.
- Add direct NPU-to-NPU tensor handoff for intermediate activations.
- Batch or persist static weight preload across repeated decode tokens.
- Reduce host/NPU boundary crossings for quantize/dequantize and cache updates.
- Improve CPU+NPU timing from the current `39.17 MHz` estimate toward the CPU-only `57.1 MHz`; every MHz matters because the current design needs `1.46x` cycle speedup before wall-time parity.
