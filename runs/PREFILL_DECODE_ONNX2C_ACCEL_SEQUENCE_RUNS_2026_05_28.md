# Prefill+Decode ONNX2C vs Accelerated Sequence Runs - 2026-05-28

Purpose: collect persistent, reproducible RTL measurements for CPU-only ONNX2C
baselines and CPU+NPU accelerated paths using one firmware/runtime image per
experiment. Each image executes prefill followed by decode in one run and prints
section-level cycle counts where available.

## Script Updates

- `scripts/run_onnx2c_prefill_decode_sequence.py` builds CPU-only ONNX2C
  prefill+decode sequence images for QLlama-like and QGPT2-like blocks.
- `scripts/build_onnx2c_baremetal_baselines.py` now includes a GPT2 decode ONNX
  generator, so QGPT2 prefill+decode can be built in the same style as QLlama.
- `scripts/run_onnx2c_prefill_decode_sequence.py` supports `--timeout-s 0` to
  disable Python wall-time timeout.
- `scripts/run_onnx2c_prefill_decode_sequence.py` streams RTL stdout/stderr
  directly into the requested log path while the simulator runs.
- `scripts/run_cv32e40p_prefill_decode_sequence.py` was updated with the same
  streamed-log and `--timeout-s 0` behavior for accelerated CPU+NPU runs.
- Both sequence runners now support `--decode-tokens 2`. The accelerated runner
  builds a true second decode artifact whose cache length is `prompt_len + 2`;
  firmware copies decode0 K/V outputs into decode1 before running decode1.

## Completed ONNX2C CPU-only Sequence Runs

| Model | Config | Log | Total cycles | Prefill cycles | Decode cycles | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| QLlama-like | `d8 h8 nh1 nkv1 f8 T8` | `runs/onnx2c_qllama_prefill_decode_d8_smoke.log` | 38,320 | 33,312 | 5,007 | success |
| QGPT2-like | `d8 h8 nh1 nkv1 f8 T8` | `runs/onnx2c_qgpt2_prefill_decode_d8_smoke.log` | 38,787 | 33,862 | 4,924 | success |
| QLlama-like | `d64 h16 nh4 nkv2 f64 T8` | `runs/onnx2c_qllama_prefill_decode_d64_t8.log` | 2,045,307 | 1,814,089 | 231,217 | success |

## Long ONNX2C CPU-only Sequence Runs

| Model | Config | Log | Command | Status |
| --- | --- | --- | --- | --- |
| QLlama-like | `d192 h16 nh12 nkv6 f192 T8` | `runs/onnx2c_qllama_prefill_decode_d192_t8.log` | `PYTHONPATH=software/compiler python3 scripts/run_onnx2c_prefill_decode_sequence.py --model qllama --d-model 192 --d-head 16 --n-heads 12 --n-kv-heads 6 --ffn-dim 192 --prompt-len 8 --seed 0 --run-rtl --maxcycles 1000000000 --verilator-max-ticks 5000000000000 --timeout-s 1800 --log-path runs/onnx2c_qllama_prefill_decode_d192_t8.log` | timed out after 1800 s without a result line |
| QLlama-like | `d192 h16 nh12 nkv6 f192 T8` | `runs/onnx2c_qllama_prefill_decode_d192_t8_unlimited.log` | `PYTHONPATH=software/compiler python3 scripts/run_onnx2c_prefill_decode_sequence.py --model qllama --d-model 192 --d-head 16 --n-heads 12 --n-kv-heads 6 --ffn-dim 192 --prompt-len 8 --seed 0 --run-rtl --maxcycles 3000000000 --verilator-max-ticks 5000000000000 --timeout-s 0 --log-path runs/onnx2c_qllama_prefill_decode_d192_t8_unlimited.log` | running at note time |

## Build-only Sequence Artifacts

These artifacts link successfully and are ready for RTL execution.

| Model | Config | ELF |
| --- | --- | --- |
| QLlama-like ONNX2C | `d192 h16 nh12 nkv6 f192 T8` | `external/cv32e40p/example_tb/core/custom/third_party_onnx2c_qllama_prefill_decode_seq_d192_h16_nh12_nkv6_f192_t8_s0.elf` |
| QGPT2-like ONNX2C | `d192 h16 nh12 nkv12 f192 T8` | `external/cv32e40p/example_tb/core/custom/third_party_onnx2c_qgpt2_prefill_decode_seq_d192_h16_nh12_nkv12_f192_t8_s0.elf` |
| QLlama accelerated | `d192 h16 nh12 nkv6 f192 T8` | `external/cv32e40p/example_tb/core/custom/cv32e40p_qllama_prefill_decode_seq_d192_h16_nh12_nkv6_f192_t8_s0.elf` |
| QGPT2 accelerated | `d192 h16 nh12 nkv12 f192 T8` | `external/cv32e40p/example_tb/core/custom/cv32e40p_qgpt2_prefill_decode_seq_d192_h16_nh12_nkv12_f192_t8_s0.elf` |
| QLlama-like ONNX2C | `d8 h8 nh1 nkv1 f8 T8 decode_tokens=2` | `external/cv32e40p/example_tb/core/custom/third_party_onnx2c_qllama_prefill_decode2_seq_d8_h8_nh1_nkv1_f8_t8_s0.elf` |
| QGPT2-like ONNX2C | `d8 h8 nh1 nkv1 f8 T8 decode_tokens=2` | `external/cv32e40p/example_tb/core/custom/third_party_onnx2c_qgpt2_prefill_decode2_seq_d8_h8_nh1_nkv1_f8_t8_s0.elf` |
| QLlama accelerated | `d8 h8 nh1 nkv1 f8 T8 decode_tokens=2` | `external/cv32e40p/example_tb/core/custom/cv32e40p_qllama_prefill_decode2_seq_d8_h8_nh1_nkv1_f8_t8_s0.elf` |
| QGPT2 accelerated | `d8 h8 nh1 nkv1 f8 T8 decode_tokens=2` | `external/cv32e40p/example_tb/core/custom/cv32e40p_qgpt2_prefill_decode2_seq_d8_h8_nh1_nkv1_f8_t8_s0.elf` |
| QLlama-like ONNX2C | `d128 h16 nh8 nkv4 f128 T8 decode_tokens=2` | `external/cv32e40p/example_tb/core/custom/third_party_onnx2c_qllama_prefill_decode2_seq_d128_h16_nh8_nkv4_f128_t8_s0.elf` |
| QGPT2-like ONNX2C | `d128 h16 nh8 nkv8 f128 T8 decode_tokens=2` | `external/cv32e40p/example_tb/core/custom/third_party_onnx2c_qgpt2_prefill_decode2_seq_d128_h16_nh8_nkv8_f128_t8_s0.elf` |
| QLlama accelerated | `d128 h16 nh8 nkv4 f128 T8 decode_tokens=2` | `external/cv32e40p/example_tb/core/custom/cv32e40p_qllama_prefill_decode2_seq_d128_h16_nh8_nkv4_f128_t8_s0.elf` |
| QGPT2 accelerated | `d128 h16 nh8 nkv8 f128 T8 decode_tokens=2` | `external/cv32e40p/example_tb/core/custom/cv32e40p_qgpt2_prefill_decode2_seq_d128_h16_nh8_nkv8_f128_t8_s0.elf` |

## Large-RAM d288 Two-Token Build Artifacts

The default CV32E40P testbench RAM is 4 MiB. The sequence runners now accept
`--sim-ram-bytes` and `--sim-ram-addr-width` so large CPU firmware images can be
linked and run against a larger simulated CPU RAM without changing the NPU RTL
datapath or UB/IM sizes. The d288 two-token images below were built with
`--sim-ram-bytes 0x1000000 --sim-ram-addr-width 24`.

| Model | Config | Build log | ELF |
| --- | --- | --- | --- |
| QLlama-like ONNX2C | `d288 h32 nh9 nkv3 f288 T8 decode_tokens=2` | `runs/build_onnx2c_qllama_prefill_decode2_d288_t8_16m.log` | `external/cv32e40p/example_tb/core/custom/third_party_onnx2c_qllama_prefill_decode2_seq_d288_h32_nh9_nkv3_f288_t8_s0.elf` |
| QGPT2-like ONNX2C | `d288 h32 nh9 nkv9 f288 T8 decode_tokens=2` | `runs/build_onnx2c_qgpt2_prefill_decode2_d288_t8_16m.log` | `external/cv32e40p/example_tb/core/custom/third_party_onnx2c_qgpt2_prefill_decode2_seq_d288_h32_nh9_nkv9_f288_t8_s0.elf` |
| QLlama accelerated | `d288 h32 nh9 nkv3 f288 T8 decode_tokens=2` | `runs/build_accel_qllama_prefill_decode2_d288_t8_16m.log` | `external/cv32e40p/example_tb/core/custom/cv32e40p_qllama_prefill_decode2_seq_d288_h32_nh9_nkv3_f288_t8_s0.elf` |
| QGPT2 accelerated | `d288 h32 nh9 nkv9 f288 T8 decode_tokens=2` | `runs/build_accel_qgpt2_prefill_decode2_d288_t8_16m.log` | `external/cv32e40p/example_tb/core/custom/cv32e40p_qgpt2_prefill_decode2_seq_d288_h32_nh9_nkv9_f288_t8_s0.elf` |

## Build Failures Under Default 4 MiB CV32E40P RAM Image

| Model | Config | Failure |
| --- | --- | --- |
| QLlama accelerated | `d192 h16 nh12 nkv6 f192 T8 decode_tokens=2` | linker region `ram` overflowed by 535,152 bytes |
| QLlama accelerated | `d288 h32 nh9 nkv3 f288 T8 decode_tokens=2` | linker region `ram` overflowed by 5,099,664 bytes |
| QGPT2 accelerated | `d288 h32 nh9 nkv9 f288 T8 decode_tokens=2` | linker region `ram` overflowed by 5,618,576 bytes |

## Current Caveats

- The ONNX2C sequence timing measures independent ONNX2C calls in one firmware
  image. It is a CPU-only workload baseline with section timing; it does not yet
  feed prefill-produced K/V tensors into the decode ONNX graph.
- d192 and d288 two-token accelerated runs do not fit the default 4 MiB
  CV32E40P bare-metal RAM image because prefill, decode0, and decode1 are linked
  as separate generated programs with duplicated constants. d288 now links under
  the opt-in 16 MiB simulation RAM path. A compiler/runtime refactor that shares
  weights/constants across the sequence is still the better long-term fix.
- The older d192 one-decode ONNX2C uncapped run is still live with no firmware
  output in `runs/onnx2c_qllama_prefill_decode_d192_t8_unlimited.log`; it is not
  the two-token comparison target.
- Checkpoint commits are available through the alternate gitdir:
  `git --git-dir=/root/compiler-optimization/.git-real --work-tree=/root/compiler-optimization ...`.
  Current relevant commits are `69eb1d0` and `e332087`.
