# Clock-Adjusted Example Performance - 2026-05-18

## Clock Assumptions

- CPU-only OOC point: `57.1 MHz`, from `runs/vivado_cpu_only_a200_57p1mhz`.
- CPU+NPU real-BRAM estimate: `39.17 MHz`, from the 50 MHz CPU+NPU run with `WNS=-5.527 ns`.
- Clock adjustment factor: `39.17 / 57.1 = 0.686`.

This means a cycle-count speedup must be above about `1 / 0.686 = 1.46x` before CPU+NPU is faster in wall time than CPU-only.

## MLP

Source cycle data: last good four-layer is_zero MLP Runtime V2 result recorded in `runs/PERF_2026-04-08_INT16_INT8_INT4_FAIR.md`.

| Case | CPU cycles | CPU+NPU cycles | Cycle speedup | CPU time @57.1 MHz | CPU+NPU time @39.17 MHz | Clock-adjusted speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4-layer MLP 1x cold | 252,731 | 49,844 | 5.07x | 4.426 ms | 1.273 ms | 3.48x |
| 4-layer MLP 10x | 2,527,310 | 258,995 | 9.76x | 44.261 ms | 6.612 ms | 6.69x |

Note: current 2026-05-18 MLP runtime-v2 log failed output verification, so this is a clock-adjusted view of the last good MLP result, not a fresh current-state signoff.

## Conv

Source cycle data: last good INT16 multi-layer conv results recorded in `runs/PERF_2026-04-08_INT16_INT8_INT4_FAIR.md`.

| Case | CPU cycles | CPU+NPU cycles | Cycle speedup | CPU time @57.1 MHz | CPU+NPU time @39.17 MHz | Clock-adjusted speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Conv no-gather 1x cold | 1,054,736 | 170,528 | 6.19x | 18.472 ms | 4.354 ms | 4.24x |
| Conv no-gather 10x | 10,547,360 | 1,603,166 | 6.58x | 184.717 ms | 40.928 ms | 4.51x |

Note: old gather-on conv numbers are intentionally excluded here. The current RTL/compiler path does not have the old conv gather accelerator path; `scatter` names that still exist are KV-cache host helpers, not conv gather.

## Attention / QLlama

Source cycle data: current QLlama RTL runs in `runs/NPU_RECOVERY_AND_TIMING_2026_05_18.md`.

| Case | CPU cycles | CPU+NPU cycles | Cycle speedup | CPU time @57.1 MHz | CPU+NPU time @39.17 MHz | Clock-adjusted speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| QLlama prefill `d8/T8` cold | 261,341 | 214,731 | 1.22x | 4.577 ms | 5.482 ms | 0.83x |
| QLlama decode `d32/T8` cold | 224,719 | 187,030 | 1.20x | 3.936 ms | 4.775 ms | 0.82x |
| QLlama decode `d32/T8` hot | 224,719 | 172,789 | 1.30x | 3.936 ms | 4.411 ms | 0.89x |

Attention conclusion:

- In cycles, the current small QLlama attention-like block starts crossing parity.
- With the current real-BRAM CPU+NPU Fmax penalty, it does not yet beat CPU-only in wall time.
- It needs either larger attention/MLP shapes, lower host/readback overhead, or a CPU+NPU timing fix closer to 50 MHz.
