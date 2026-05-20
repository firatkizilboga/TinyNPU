# Scaled Conv vs MLP RTL Comparison

These runs answer whether scaling beyond the 51k-MAC ONNX-shape Conv improves the CPU+NPU story, and compare it against a similarly sized dense MLP.

| Workload | Shape/config | Logical MACs | Cold E2E cycles | Warm body cycles | 10x extrapolated E2E |
| --- | --- | ---: | ---: | ---: | ---: |
| Conv, ONNX-shape | `8x8x1 -> 6x6x16 -> 4x4x16 -> 2x2x16 -> scalar` | 51,328 | 124,838 | 113,721 | 1,148,327 |
| Conv, wide32 | `8x8x1 -> 6x6x32 -> 4x4x32 -> 2x2x32 -> scalar` | 194,816 | 253,018 | 212,717 | 2,167,471 |
| MLP, h256 | `1x256 -> 1x256 -> 1x256 -> 1x256 -> scalar` | 196,864 | 443,664 | 45,987 | 857,547 |

## Normalized Cycles

| Workload | Cold cycles/MAC | Warm cycles/MAC |
| --- | ---: | ---: |
| Conv, ONNX-shape | 2.43 | 2.22 |
| Conv, wide32 | 1.30 | 1.09 |
| MLP, h256 | 2.25 | 0.23 |

## Interpretation

The wider Conv scales in the right direction: warm normalized cost improves from `2.22` to `1.09` cycles/MAC as the NPU work grows. Cold E2E still pays preload, but even cold normalized cost improves from `2.43` to `1.30` cycles/MAC.

The MLP is the cleaner dense case. Its warm body is much better at `0.23` cycles/MAC because there are no host `im2col` operations and little per-layer CPU work. Cold E2E is dominated by static weight preload: `397,677` preload cycles versus only `45,987` warm body cycles.

Bottom line: scaling helps both, but MLP benefits much more from NPU residency/reuse. Conv still improves with scale, but it remains more exposed to staging/readback and `im2col` overhead.
