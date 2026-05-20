# MLP-Scale Conv RTL Benchmark

Program: `cv32e40p_mlp_scale_conv_v2`

This benchmark replaces the accidental 28x28 MNIST-style Conv run with a small valid-conv workload whose logical MAC count is close to the is-zero MLP.

The bad Conv run that motivated this benchmark used a 28x28 input and produced huge `im2col` tensors, including `(784, 144)` for later layers. That makes host-side lowering dominate before the NPU does useful work, so it is not comparable to the small is-zero MLP benchmark.

| Workload | Logical MACs |
| --- | ---: |
| This Conv pipeline | 12,100 |
| Is-zero MLP reference | 12,352 |

| Conv stage | Logical MACs |
| --- | ---: |
| conv1 | 2,592 |
| conv2 | 9,216 |
| conv3 | 288 |
| conv4 | 4 |

Shape chain: `8x8x1 -> 6x6x8 -> 4x4x8 -> 2x2x1 -> 1 scalar`.

Repeat count: `3`.

## RTL Cycles

| Counter | Cycles |
| --- | ---: |
| `cold.body` | 63,032 |
| `cold.e2e` | 66,085 |
| `extrapolated.10x.e2e` | 633,373 |
| `preload.im_seg_conv1` | 87 |
| `preload.im_seg_conv2` | 87 |
| `preload.im_seg_conv3` | 87 |
| `preload.im_seg_conv4` | 87 |
| `preload.total` | 3,053 |
| `preload.ub_image` | 2,705 |
| `warm.avg.body` | 63,032 |
| `warm1.body` | 63,032 |
| `warm2.body` | 63,032 |

## Status

`EXIT SUCCESS`: `True`

## Single-Iteration Breakdown

This is from a separate verbose repeat-1 run of the same program.

| Step | Cycles |
| --- | ---: |
| `hostop.conv1_im2col` | 7,001 |
| `segment.seg_conv1.stage` | 6,145 |
| `segment.seg_conv1.run` | 346 |
| `segment.seg_conv1.readback` | 6,495 |
| `segment.seg_conv1.npu` | 20,041 |
| `hostop.conv2_im2col` | 15,834 |
| `segment.seg_conv2.stage` | 13,190 |
| `segment.seg_conv2.run` | 276 |
| `segment.seg_conv2.readback` | 2,714 |
| `segment.seg_conv2.npu` | 23,534 |
| `hostop.conv3_im2col` | 4,776 |
| `segment.seg_conv3.stage` | 5,455 |
| `segment.seg_conv3.run` | 156 |
| `segment.seg_conv3.readback` | 1,295 |
| `segment.seg_conv3.npu` | 13,961 |
| `hostop.conv4_im2col` | 1,155 |
| `segment.seg_conv4.stage` | 675 |
| `segment.seg_conv4.run` | 86 |
| `segment.seg_conv4.readback` | 1,211 |
| `segment.seg_conv4.npu` | 8,429 |

The corrected workload no longer has runaway host lowering, but it is still a very overhead-heavy Conv regime: only `864` cycles are spent in the four NPU `run` bodies, while host `im2col` totals `28,766` cycles and segment stage/readback/control overhead dominates E2E time.
