# Scaled Conv RTL Benchmark

Program: `cv32e40p_wide32_conv_v2`

Variant: `wide32`

This is the next Conv scale point after the 16-channel ONNX-shape Conv4 run.

| Workload | Logical MACs |
| --- | ---: |
| This Conv pipeline | 194,816 |
| Is-zero MLP reference | 12,352 |

| Conv stage | Logical MACs |
| --- | ---: |
| conv1 | 10,368 |
| conv2 | 147,456 |
| conv3 | 36,864 |
| conv4 | 128 |

Shape chain: `8x8x1 -> 6x6x32 -> 4x4x32 -> 2x2x32 -> 1 scalar`.

Repeat count: `3`.

## RTL Cycles

| Counter | Cycles |
| --- | ---: |
| `preload.ub_image` | 39,953 |
| `preload.im_seg_conv1` | 87 |
| `preload.im_seg_conv2` | 87 |
| `preload.im_seg_conv3` | 87 |
| `preload.im_seg_conv4` | 87 |
| `preload.total` | 40,301 |
| `cold.body` | 212,717 |
| `cold.e2e` | 253,018 |
| `warm1.body` | 212,717 |
| `warm2.body` | 212,717 |
| `warm.avg.body` | 212,717 |
| `extrapolated.10x.e2e` | 2,167,471 |

## Status

`EXIT SUCCESS`: `True`
