# RTL CPU+NPU Status - 2026-05-18

Environment:
- Server: Ubuntu 24.04, `/root/compiler-optimization`
- Verilator: 5.036
- RISC-V compiler: `riscv64-unknown-elf-gcc`, `-O3`, `rv32imfc/ilp32f`
- RTL build target: `cv32e40p_tb_vlt_npu`
- Git state: dirty worktree; these are current-uncommitted RTL/compiler results.

## Smoke/Linear Results

| Run | Status | Key cycles | Log |
| --- | --- | ---: | --- |
| GEMM 8x8x8 INT16, runtime input, Runtime V2 | PASS | `cold.e2e=2,976`, `cold.body=2,744`, `preload.total=232` | `runs/debug_gemm_current/logs/cv32e40p_gemm_8x8x8_int16_runtimein_v2.log` |
| GEMM 64x64x64 INT16, runtime input, Runtime V2 | PASS | `cold.e2e=154,609`, `cold.body=146,313`, `preload.total=8,296` | `runs/debug_gemm_current/logs/cv32e40p_gemm_64x64x64_int16_runtimein_v2.log` |

Conclusion: the current RTL can still execute verified INT16 NPU GEMM on both one-tile and 64x64 shapes.

## QLlama Decode d8 h8 nh1 nkv1 f8 ctx8

| Variant | Status | Cycles | Log |
| --- | --- | ---: | --- |
| CPU-only | PASS | `repeat.program.cpu.cold.total=41,382` | `runs/qllama_decode_d8_t8_cpu_2026_05_18.log` |
| NPU hybrid | FAIL verification | observed pre-failure total `66,965` = preload `1,878` + host `26,282` + segment.npu `38,805` | `runs/qllama_decode_d8_t8_npu_fail_2026_05_18.log` |

NPU hybrid completed all segments but final tensor did not match:
- Actual: `[28.355, -164.362, 28.585, -164.121, -186.809, 86.699, -26.136, 76.076]`
- Expected: `[53.012, 69.951, 125.366, -41.683, 138.066, -98.989, -62.198, -57.924]`

Current interpretation:
- CPU-only generated model/golden path is valid.
- Plain NPU GEMM is valid.
- Failure is likely in the legacy QLlama bare-metal emitter/runtime path or transformer-specific NPU data movement/readback, not in basic Verilator build or the systolic datapath.

## Is-Zero MLP

Existing firmware images were found under `external/cv32e40p/example_tb/core/custom` and run directly against the current RTL.

| Run | Status | Key cycles / observation | Log |
| --- | --- | --- | --- |
| `cv32e40p_iszero_mlp_multilayer_cpu_only_o3.hex` | PASS | `segment.segment_000.cpu=248,383`; fc breakdown: `78,161`, `78,705`, `80,739`, `1,600`; output matched `0.138` | `runs/iszero_mlp_multilayer_cpu_only_o3_current_rtl_2026_05_18.log` |
| `cv32e40p_iszero_mlp_multilayer_v2_repeat3.hex` | FAIL verification | preload `26,605`; output stayed `0.000`, expected `0.138` | `runs/iszero_mlp_multilayer_v2_repeat3_current_rtl_2026_05_18.log` |
| `cv32e40p_iszero_mlp_hardened_sweep.hex` | FAIL verification | output stayed `0.000`, expected `0.138` | `runs/iszero_mlp_hardened_sweep_current_rtl_2026_05_18.log` |
| Rebuilt `cv32e40p_iszero_mlp_multilayer_v2` from generated Runtime V2 source | TIMEOUT | `preload.ub_image=26,417`, then max-tick timeout before IM/segment output | `runs/iszero_mlp_multilayer_v2_rebuilt_current_rtl_2026_05_18.log` |

Current interpretation:
- The CPU baseline for the MLP is valid on the current RTL.
- The prebuilt NPU MLP firmware no longer produces the expected output. Rebuilding from the generated Runtime V2 source changes the failure to a timeout after UB preload, so this is a current runnable-path regression rather than just missing artifacts.

## Conv / Conv-Like Multilayer

Existing Conv firmware images were found and run directly. These did not complete within the current `VERILATOR_MAX_TICKS=3,000,000,000` limit.

| Run | Status | Last observed cycles | Log |
| --- | --- | --- | --- |
| `cv32e40p_fair_conv_multilayer_off_hardened_sweep.hex` | TIMEOUT | banner only before max-tick timeout | `runs/fair_conv_multilayer_off_hardened_sweep_current_rtl_2026_05_18.log` |
| `cv32e40p_fair_conv_multilayer_off_cpu_only_demo.hex` | TIMEOUT | `q_in=13,723`, `inner_conv1_im2col=2,230`, `segment_000.cpu=141,064`, `inner_conv2_im2col=31,880`; timed out during `segment_001` | `runs/fair_conv_multilayer_off_cpu_only_current_rtl_2026_05_18.log` |
| `cv32e40p_fair_conv_multilayer_off_cpu_repeat3_demo.hex` | TIMEOUT | first NPU segments before timeout: seg0 `npu=31,824`, seg0 CPU comparison `86,531`, seg1 `npu=61,097`; timed out after seg1 | `runs/fair_conv_multilayer_off_cpu_repeat3_current_rtl_2026_05_18.log` |

Current interpretation:
- Conv firmware is present, so this is no longer blocked by missing files.
- The old Conv runs are too slow/stale under the current Verilator max-tick budget to complete end-to-end.
- Partial CPU+NPU numbers are available for the repeat3 Conv firmware, but no final verification result was reached.

## GPT2 Two-Block Reuse d8 h8 nh1 f8 T4

| Run | Status | Last good point | Log |
| --- | --- | --- | --- |
| Runtime V2 two-block prefill+decode+reuse | FAIL runtime contract | after `p0_b0_seg_score`; `runtime failure: mul expects float-compatible inputs` | `runs/gpt2_two_block_reuse_d8_t4_fail_2026_05_18.log` |

Observed before failure:
- Preload cycles through IM images: `9,403`
- Host cycles before failure: `14,113`
- NPU segment cycles before failure: `35,887`

Current interpretation:
- Runtime V2 launches and executes early GPT2 NPU segments on current RTL.
- The failure is a compiler/runtime dtype contract issue: score scaling receives a non-float-compatible tensor.

## Missing Past Artifacts

The original source artifact directories used by `scripts/run_cv32e40p_hardened_sweep.py` are still missing:
- `runs/tinynpu_issue27_backup_2026_03_21/mnist_mlp_iszero_int16_smoke`
- `runs/mnist_tinynpu_pipeline`

However, prebuilt MLP and Conv firmware images were found in `external/cv32e40p/example_tb/core/custom` and were run directly above. If the source artifact directories are copied in, rerun the full scripted sweep:

```bash
python3 scripts/run_cv32e40p_hardened_sweep.py
```

## Fixes Applied During This Run

- Added picolibc `stdin/stdout/stderr` globals in `external/cv32e40p/example_tb/core/custom/syscalls.c` so generated benchmark programs link on Ubuntu 24.04 with `riscv64-unknown-elf-gcc`.
- Updated `scripts/run_gemm_v2_e2e.py` to use the shared toolchain/picolibc discovery path.
- Updated `scripts/run_cv32e40p_xform_q_f16_i16_demo.py` to use the shared Runtime V2 build helper, but the standalone XFORM demo still does not produce a useful pass/fail result yet.
