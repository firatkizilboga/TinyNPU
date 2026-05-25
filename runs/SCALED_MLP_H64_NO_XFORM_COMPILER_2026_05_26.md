# Scaled MLP RTL Benchmark

Program: `cv32e40p_scaled_mlp_h64_no_xform_compiler`

Shape: `1x64 -> 1x64 -> 1x64 -> 1x64 -> 1 scalar`.

Repeat count: `1`.

| Workload | Logical MACs |
| --- | ---: |
| This MLP pipeline | 12,352 |

| MLP stage | Logical MACs |
| --- | ---: |
| fc1 | 4,096 |
| fc2 | 4,096 |
| fc3 | 4,096 |
| fc4 | 64 |

## RTL Cycles

| Counter | Cycles |
| --- | ---: |
| `preload.im_seg_fc1` | 87 |
| `preload.im_seg_fc2` | 87 |
| `preload.im_seg_fc3` | 87 |
| `preload.im_seg_fc4` | 87 |
| `preload.ub_image` | 25,617 |
| `segment.seg_fc1.npu` | 11,689 |
| `segment.seg_fc1.readback` | 6 |
| `segment.seg_fc1.run` | 915 |
| `segment.seg_fc1.stage` | 3,957 |
| `segment.seg_fc2.npu` | 6,835 |
| `segment.seg_fc2.readback` | 6 |
| `segment.seg_fc2.run` | 915 |
| `segment.seg_fc2.stage` | 2 |
| `segment.seg_fc3.npu` | 6,835 |
| `segment.seg_fc3.readback` | 6 |
| `segment.seg_fc3.run` | 915 |
| `segment.seg_fc3.stage` | 2 |
| `segment.seg_fc4.npu` | 7,261 |
| `segment.seg_fc4.readback` | 1,202 |
| `segment.seg_fc4.run` | 145 |
| `segment.seg_fc4.stage` | 2 |

## Status

`EXIT SUCCESS`: `True`
