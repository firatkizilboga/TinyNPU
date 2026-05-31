# Vivado CPU+NPU Timing Refresh - 2026-05-31

Purpose: record the integrated TinyNPU timing point used for updated wall-clock comparisons after replacing the standalone sigmoid path with the hard-GELU gate form.

## Command

```sh
python3 scripts/vivado_timing.py cpu-npu --clock-ns 20.0 --workdir runs/vivado_cpu_npu_sigmoid_gate_50mhz_20260531
```

## Result

- Target device: `xc7a200tsbg484-1`.
- Target period: `20.000 ns` (`50.00 MHz`).
- Post-route WNS: `-0.408 ns`.
- Post-route TNS: `-4.147 ns`.
- Failing setup endpoints: `32`.
- Hold slack: `+0.058 ns`.
- Implied safe period: `20.408 ns`.
- Implied frequency used for wall-clock comparisons: `49.00 MHz`.

## Resource Utilization

| Resource | Used | Available | Utilization |
| --- | ---: | ---: | ---: |
| Slice LUTs | 49,180 | 133,800 | 36.76% |
| Slice Registers | 20,290 | 267,600 | 7.58% |
| Block RAM Tile | 272 | 365 | 74.52% |
| DSPs | 177 | 740 | 23.92% |

## Reports

- Timing: `runs/vivado_cpu_npu_sigmoid_gate_50mhz_20260531/reports/post_route_timing.rpt`
- Critical paths: `runs/vivado_cpu_npu_sigmoid_gate_50mhz_20260531/reports/post_route_critical_paths.rpt`
- Utilization: `runs/vivado_cpu_npu_sigmoid_gate_50mhz_20260531/reports/post_route_util.rpt`
