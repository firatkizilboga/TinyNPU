# MLP-Scale Conv RTL Benchmark

Program: `cv32e40p_onnx_shape_conv_v2`

Variant: `onnx_shape`

This benchmark replaces the accidental 28x28 MNIST-style Conv run with a small valid-conv workload whose logical MAC count is close to the is-zero MLP.

The bad Conv run that motivated this benchmark used a 28x28 input and produced huge `im2col` tensors, including `(784, 144)` for later layers. That makes host-side lowering dominate before the NPU does useful work, so it is not comparable to the small is-zero MLP benchmark.

| Workload | Logical MACs |
| --- | ---: |
| This Conv pipeline | 51,328 |
| Is-zero MLP reference | 12,352 |

| Conv stage | Logical MACs |
| --- | ---: |
| conv1 | 5,184 |
| conv2 | 36,864 |
| conv3 | 9,216 |
| conv4 | 64 |

Shape chain: `8x8x1 -> 6x6x16 -> 4x4x16 -> 2x2x16 -> 1 scalar`.

Repeat count: `3`.

## RTL Cycles

| Counter | Cycles |
| --- | ---: |
| `cold.body` | 113,721 |
| `cold.e2e` | 124,838 |
| `extrapolated.10x.e2e` | 1,148,327 |
| `preload.im_seg_conv1` | 87 |
| `preload.im_seg_conv2` | 87 |
| `preload.im_seg_conv3` | 87 |
| `preload.im_seg_conv4` | 87 |
| `preload.total` | 11,117 |
| `preload.ub_image` | 10,769 |
| `warm.avg.body` | 113,721 |
| `warm1.body` | 113,721 |
| `warm2.body` | 113,721 |

## Status

`EXIT SUCCESS`: `True`
