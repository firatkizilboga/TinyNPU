# Scaled MLP RTL Benchmark

Program: `cv32e40p_scaled_mlp_h256_v2`

Shape: `1x256 -> 1x256 -> 1x256 -> 1x256 -> 1 scalar`.

Repeat count: `3`.

| Workload | Logical MACs |
| --- | ---: |
| This MLP pipeline | 196,864 |

| MLP stage | Logical MACs |
| --- | ---: |
| fc1 | 65,536 |
| fc2 | 65,536 |
| fc3 | 65,536 |
| fc4 | 256 |

## RTL Cycles

| Counter | Cycles |
| --- | ---: |
| `preload.ub_image` | 397,329 |
| `preload.im_seg_fc1` | 87 |
| `preload.im_seg_fc2` | 87 |
| `preload.im_seg_fc3` | 87 |
| `preload.im_seg_fc4` | 87 |
| `preload.total` | 397,677 |
| `cold.body` | 45,987 |
| `cold.e2e` | 443,664 |
| `warm1.body` | 45,987 |
| `warm2.body` | 45,987 |
| `warm.avg.body` | 45,987 |
| `extrapolated.10x.e2e` | 857,547 |

## Status

`EXIT SUCCESS`: `True`
