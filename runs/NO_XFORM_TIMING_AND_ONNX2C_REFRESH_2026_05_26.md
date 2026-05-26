# No-XFORM Timing and Baseline Refresh - 2026-05-26

Hardware XFORM is removed from the compiler-emitted path and from the TinyNPU
RTL. Quantize/dequantize boundaries are now software/runtime boundaries.

## Vivado Timing

Target device: `xc7a200tsbg484-1`.

| Design | Target | WNS | Approx period | Approx Fmax | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| NPU only, real BRAM | 50 MHz | -9.851 ns | 29.851 ns | 33.5 MHz | Critical path crosses control address logic into UB partial mask/data logic. |
| CPU+NPU, real BRAM | 50 MHz | -9.486 ns | 29.486 ns | 33.9 MHz | Same dominant NPU UB/control path; CPU does not materially change Fmax. |

## Refreshed ONNX2C CPU Baselines

| Workload | Result |
| --- | --- |
| Conv 16ch | `third_party_onnx2c_conv4 cycles=259688`, `EXIT SUCCESS` |
| Conv 32ch | `third_party_onnx2c_conv4 cycles=1013314`, `EXIT SUCCESS` |
| MLP h256 | Rerun timed out at 420 s before final printf; previous measured baseline remains `1662361` cycles. |

## Current No-XFORM QLlama Decode

| Config | CPU+NPU cold cycles | CPU+NPU hot cycles |
| --- | ---: | ---: |
| `d32 h8 nh4 nkv2 f32 T8` | 241,807 | 227,508 |
| `d64 h16 nh4 nkv2 f64 T8` | 398,705 | 346,518 |
| `d96 h16 nh6 nkv3 f96 T8` | 618,806 | 503,899 |
| `d128 h16 nh8 nkv4 f128 T8` | 862,327 | 660,124 |
| `d192 h16 nh12 nkv6 f192 T8` | 1,427,556 | 977,033 |

Using `57.1 MHz` for ONNX2C CPU and `33.9 MHz` for CPU+NPU, the current
no-XFORM path only beats ONNX2C at the largest measured QLlama decode point, and
only for the hot/resident case.
