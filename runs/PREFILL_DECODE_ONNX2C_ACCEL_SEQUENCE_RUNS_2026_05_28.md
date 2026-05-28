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

## Sequence Runner Debug Notes

| Case | Log | Result |
| --- | --- | --- |
| QLlama accelerated `d128 h16 nh8 nkv4 f128 T8 decode_tokens=2` old runner | `runs/accel_qllama_prefill_decode2_d128_t8_direct.log` | Failed before section timing: prefill exposes 9 outputs (`final + K/V cache for 4 KV heads`) but the generated runner only allocated 8 output descriptors. Fixed in commit `bcc7dc7` by raising the generated runner I/O descriptor capacity to 64. |
| QLlama accelerated `d128 h16 nh8 nkv4 f128 T8 decode_tokens=2` fixed runner | `runs/accel_qllama_prefill_decode2_d128_t8_fixed.log` | Aborted after roughly 6.5 h because it was started with the temporary `+verbose` flag. The log grew to 4.9 MiB of per-memory-access traffic and did not reach a complete section timing line. |

## Aborted / Invalid Long Runs

These runs were stopped at 2026-05-28 10:34 local time because they were stale
or invalid and were pinning the 32-core host at a load average above 70. They
should not be used for performance numbers.

| Case | Log | Why invalid |
| --- | --- | --- |
| QLlama ONNX2C `d192 h16 nh12 nkv6 f192 T8` one-decode | `runs/onnx2c_qllama_prefill_decode_d192_t8_unlimited.log` | Old one-decode baseline, not the requested two-decode sequence. It ran for about 8.1 h with no firmware result. |
| QLlama accelerated `d288 h32 nh9 nkv3 f288 T8 decode_tokens=2` | `runs/accel_qllama_prefill_decode2_d288_t8_16m_unlimited.log` | Started before the concise progress-marker and timeout discipline was in place. It ran for about 7.0 h with no firmware result. |
| QLlama accelerated `d128 h16 nh8 nkv4 f128 T8 decode_tokens=2` | `runs/accel_qllama_prefill_decode2_d128_t8_fixed.log` | Started with `+verbose`, which prints low-level memory traffic and makes the run unusably slow/noisy. It ran for about 6.6 h without a complete result. |
| QLlama accelerated `d8 h8 nh1 nkv1 f8 T8 decode_tokens=2` smoke | `runs/accel_qllama_prefill_decode2_d8_t8_fixed_smoke.log` | Timed out after 180 s while the host was overloaded by the invalid long runs. Rerun required on a clean host. |

## Clean Two-Decode Smoke Results

| Model | Config | Log | Result |
| --- | --- | --- | --- |
| QLlama accelerated | `d8 h8 nh1 nkv1 f8 T8 decode_tokens=2` | `runs/accel_qllama_prefill_decode2_d8_t8_fixed_sequence.log` | `EXIT SUCCESS`; prefill 717,659 cycles, prefill-to-decode handoff 8,148 cycles, decode0 256,086 cycles, decode0-to-decode1 handoff 8,068 cycles, decode1 256,783 cycles, e2e 1,265,362 cycles. |
| QLlama ONNX2C | `d8 h8 nh1 nkv1 f8 T8 decode_tokens=2` | `runs/onnx2c_qllama_prefill_decode2_d8_t8_sequence.log` | `EXIT SUCCESS`; total 44,710 cycles, prefill 33,310 cycles, decode0 4,969 cycles, decode1 5,243 cycles. ONNX2C is faster at this tiny smoke size, so this is not a reportable accelerator win. |

The clean smoke also validated two runner fixes needed before larger
measurements:

- Non-verbose Verilator stdout is now flushed after each pseudo-UART byte, so
  timeout-killed runs do not lose firmware progress markers.
- Decode-to-decode handoff copies the full mutated cache tensors
  (`k_cache_h*`/`v_cache_h*`) into the next decode image instead of trying to
  copy the one-token `_td` tensors as complete caches.
- The sequence runner owns the testbench timer for the whole image and disables
  the runtime's per-program timer reset, so `sequence.*.total` lines are valid.

Observability notes:

- Commit `71b6d1c` disables stdout buffering in generated sequence firmware and
  adds start markers for prefill, cache handoff, decode, and decode-to-decode
  handoff.
- Commit `840ed7a` briefly enabled `+verbose`, but that was too noisy because
  the CV32E40P testbench prints low-level memory traffic under that flag.
- Commit `2b44132` removes `+verbose` again. Future concise load visibility
  should use a dedicated testbench message, not the global verbose flag.

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
- The older d192 one-decode ONNX2C uncapped run was aborted with no firmware
  output in `runs/onnx2c_qllama_prefill_decode_d192_t8_unlimited.log`; it is not
  the two-token comparison target.
- Checkpoint commits are available through the alternate gitdir:
  `git --git-dir=/root/compiler-optimization/.git-real --work-tree=/root/compiler-optimization ...`.
  Current relevant commits are `69eb1d0` and `e332087`.
