# Scaled Conv RTL Benchmark

Program: `cv32e40p_onnx_shape_conv_no_xform_compiler`

Variant: `onnx_shape`

This benchmark replaces the accidental 28x28 MNIST-style Conv run with a controlled valid-conv workload.

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

Repeat count: `1`.

## RTL Cycles

| Counter | Cycles |
| --- | ---: |
| `hostop.conv1_im2col` | 6,951 |
| `hostop.conv2_im2col` | 30,428 |
| `hostop.conv3_im2col` | 8,426 |
| `hostop.conv4_im2col` | 2,057 |
| `preload.im_seg_conv1` | 87 |
| `preload.im_seg_conv2` | 87 |
| `preload.im_seg_conv3` | 87 |
| `preload.im_seg_conv4` | 87 |
| `preload.ub_image` | 10,769 |
| `segment.seg_conv1.npu` | 27,253 |
| `segment.seg_conv1.readback` | 12,682 |
| `segment.seg_conv1.run` | 655 |
| `segment.seg_conv1.stage` | 6,865 |
| `segment.seg_conv2.npu` | 43,168 |
| `segment.seg_conv2.readback` | 5,234 |
| `segment.seg_conv2.run` | 795 |
| `segment.seg_conv2.stage` | 29,789 |
| `segment.seg_conv3.npu` | 21,580 |
| `segment.seg_conv3.readback` | 2,442 |
| `segment.seg_conv3.run` | 415 |
| `segment.seg_conv3.stage` | 11,373 |
| `segment.seg_conv4.npu` | 12,355 |
| `segment.seg_conv4.readback` | 1,202 |
| `segment.seg_conv4.run` | 145 |
| `segment.seg_conv4.stage` | 3,957 |

## Status

`EXIT SUCCESS`: `True`
