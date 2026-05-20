# Measured Wall-Time Comparison

This table uses measured CV32 RTL cycle counts only. No extrapolated CPU/ONNX rows are included.

Clock assumptions:

- CPU / ONNX CPU baseline: `57.1 MHz`
- CPU+NPU baseline: `39.17 MHz`
- Wall speedup is computed from measured cycles and these clocks.

| Workload | Measured CPU/ONNX cycles | Measured CPU+NPU cycles | CPU/ONNX wall | CPU+NPU wall | Wall speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| Conv 16ch cold | 259,688 | 124,838 | 4,548.0 us | 3,187.1 us | 1.43x |
| Conv 16ch warm | 259,688 | 113,721 | 4,548.0 us | 2,903.3 us | 1.57x |
| Conv wide32 cold | 1,013,314 | 253,018 | 17,746.3 us | 6,459.5 us | 2.75x |
| Conv wide32 warm | 1,013,314 | 212,717 | 17,746.3 us | 5,430.6 us | 3.27x |
| MLP h256 cold | 1,662,361 | 443,664 | 29,113.2 us | 11,326.6 us | 2.57x |
| MLP h256 warm | 1,662,361 | 45,987 | 29,113.2 us | 1,174.0 us | 24.80x |

## Measurement Sources

| Measurement | Evidence |
| --- | --- |
| Conv 16ch CPU/ONNX | `third_party_onnx2c_conv4 cycles=259688` from earlier ONNX2C RTL baseline |
| Conv 16ch CPU+NPU | `runs/ONNX_SHAPE_CONV_BENCHMARK_2026_05_20.md`: cold `124,838`, warm `113,721` |
| Conv wide32 CPU/ONNX | `third_party_onnx2c_conv4_ch32`: `third_party_onnx2c_conv4 cycles=1013314`, `EXIT SUCCESS` |
| Conv wide32 CPU+NPU | `runs/WIDE32_CONV_BENCHMARK_2026_05_20.md`: cold `253,018`, warm `212,717` |
| MLP h256 CPU/ONNX | `third_party_onnx2c_mlp_h256`: `third_party_onnx2c_mlp cycles=1662361`, `EXIT SUCCESS` |
| MLP h256 CPU+NPU | `runs/SCALED_MLP_H256_BENCHMARK_2026_05_20.md`: cold `443,664`, warm `45,987` |

## Notes

The previous `10x` row is intentionally omitted here because it was not measured as a true 10-iteration ONNX CPU run. The NPU runtime report also labels its 10x number as extrapolated, so keeping that row would violate the “measurements only” requirement.

The measured trend still holds: scaling improves Conv wall speedup from `1.43x-1.57x` at 16 channels to `2.75x-3.27x` at 32 channels. The h256 MLP is dominated by preload when cold, but the warm resident path is very strong at `24.80x` wall speedup.
