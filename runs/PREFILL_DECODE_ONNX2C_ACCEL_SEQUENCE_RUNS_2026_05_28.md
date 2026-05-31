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
- Work in progress after the QGPT2 d128 ONNX2C run: the accelerated sequence
  runner is being converted from full-cache copy handoff to shared K/V cache
  backing storage. The intended model is pointer handoff for existing cache
  contents plus one-token append after each decode.

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

## Completed Large d288 Sequence Runs

| Model | Config | Log | Result |
| --- | --- | --- | --- |
| QGPT2 ONNX2C | `d288 h32 nh9 nkv9 f288 T8 decode_tokens=2` | `runs/onnx2c_qgpt2_prefill_decode2_d288_t8_16m_sequence_10h_20260528.log` | `EXIT SUCCESS`; total 41,802,669 cycles, prefill 33,371,930 cycles, decode0 4,211,679 cycles, decode1 4,217,823 cycles, checksum -116.647812. |
| QGPT2 accelerated shared-cache | `d288 h32 nh9 nkv9 f288 T8 decode_tokens=2` | `runs/accel_qgpt2_prefill_decode2_d288_t8_16m_sequence_20260529.log` | `EXIT SUCCESS`; total 11,324,259 cycles, prefill 6,160,032 cycles, prefill-to-decode handoff 1 cycle, decode0 2,570,421 cycles, decode0-to-decode1 handoff 1 cycle, decode1 2,576,148 cycles. Compared against ONNX2C, this is a 3.69x e2e cycle win and a 5.42x prefill cycle win. |
| QLlama ONNX2C | `d288 h32 nh9 nkv3 f288 T8 decode_tokens=2` | `runs/onnx2c_qllama_prefill_decode2_d288_t8_16m_sequence_long_20260529.log` | `EXIT SUCCESS`; total 39,139,676 cycles, prefill 31,248,397 cycles, decode0 3,940,790 cycles, decode1 3,949,251 cycles, checksum -41.8177147. |
| QLlama accelerated shared-cache | `d288 h32 nh9 nkv3 f288 T8 decode_tokens=2` | `runs/accel_qllama_prefill_decode2_d288_t8_16m_sequence_rerun_20260529.log` | `EXIT SUCCESS`; total 12,138,556 cycles, prefill 6,940,965 cycles, prefill-to-decode handoff 1 cycle, decode0 2,587,651 cycles, decode0-to-decode1 handoff 1 cycle, decode1 2,592,178 cycles. Compared against ONNX2C, this is a 3.22x e2e cycle win and a 4.50x prefill cycle win. With 66.0 MHz CPU-only and 50.17 MHz CPU+NPU timing, this is a 2.45x e2e wall-clock win and a 3.42x prefill wall-clock win. |

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
| QLlama accelerated shared-cache smoke | `d8 h8 nh1 nkv1 f8 T8 decode_tokens=2` | `runs/accel_qllama_prefill_decode2_d8_t8_shared_cache_smoke.log` | `EXIT SUCCESS`; prefill 401,503 cycles, prefill-to-decode handoff 1 cycle, decode0 241,694 cycles, decode0-to-decode1 handoff 1 cycle, decode1 242,107 cycles, e2e 902,111 cycles. This validates the shared-cache pointer handoff on the tiny QLlama sequence. |
| QLlama accelerated | `d128 h16 nh8 nkv4 f128 T8 decode_tokens=2` | `runs/accel_qllama_prefill_decode2_d128_t8_nodump_sequence.log` | `EXIT SUCCESS`; prefill 3,375,256 cycles, prefill-to-decode handoff 85,164 cycles, decode0 1,192,925 cycles, decode0-to-decode1 handoff 82,188 cycles, decode1 1,193,125 cycles, e2e 5,948,866 cycles. |
| QLlama ONNX2C | `d128 h16 nh8 nkv4 f128 T8 decode_tokens=2` | `runs/onnx2c_qllama_prefill_decode2_d128_t8_sequence.log` | `EXIT SUCCESS`; total 8,500,853 cycles, prefill 6,780,108 cycles, decode0 856,981 cycles, decode1 862,522 cycles. Accelerated e2e is 1.43x faster overall; accelerated prefill is 2.01x faster, while individual accelerated decode sections are slower. |
| QLlama accelerated shared-cache | `d128 h16 nh8 nkv4 f128 T8 decode_tokens=2` | `runs/accel_qllama_prefill_decode2_d128_t8_shared_cache_sequence.log` | `EXIT SUCCESS`; prefill 3,382,226 cycles, prefill-to-decode handoff 1 cycle, decode0 1,248,626 cycles, decode0-to-decode1 handoff 1 cycle, decode1 1,248,747 cycles, e2e 5,897,388 cycles. Compared against ONNX2C, this is a 1.44x e2e cycle win and a 2.00x prefill cycle win. Compared against the pre-shared-cache accelerated run, e2e improves by 51,478 cycles because full-cache copy handoff is removed, while decode bodies are slower in this configuration. |
| QGPT2 accelerated | `d128 h16 nh8 nkv8 f128 T8 decode_tokens=2` | `runs/accel_qgpt2_prefill_decode2_d128_t8_nodump_sequence.log` | `EXIT SUCCESS`; prefill 2,924,429 cycles, prefill-to-decode handoff 181,299 cycles, decode0 1,057,619 cycles, decode0-to-decode1 handoff 175,621 cycles, decode1 1,060,890 cycles, e2e 5,420,502 cycles. Compared against the ONNX2C row below, this is a 1.61x e2e cycle win and a 2.37x prefill cycle win. |
| QGPT2 ONNX2C | `d128 h16 nh8 nkv8 f128 T8 decode_tokens=2` | `runs/onnx2c_qgpt2_prefill_decode2_d128_t8_sequence.log` | `EXIT SUCCESS`; total 8,708,968 cycles, prefill 6,942,157 cycles, decode0 881,327 cycles, decode1 884,243 cycles. Accelerated e2e from the pre-shared-cache run is 1.61x faster overall; accelerated prefill is 2.37x faster, while individual accelerated decode sections are slower. |
| QGPT2 accelerated shared-cache | `d128 h16 nh8 nkv8 f128 T8 decode_tokens=2` | `runs/accel_qgpt2_prefill_decode2_d128_t8_shared_cache_sequence.log` | `EXIT SUCCESS`; prefill 2,915,943 cycles, prefill-to-decode handoff 1 cycle, decode0 1,169,403 cycles, decode0-to-decode1 handoff 1 cycle, decode1 1,172,647 cycles, e2e 5,275,649 cycles. Compared against ONNX2C, this is a 1.65x e2e cycle win and a 2.38x prefill cycle win. Compared against the pre-shared-cache accelerated run, e2e improves by 144,853 cycles because handoff copies are removed, even though decode bodies are slower in this configuration. |
| QGPT2 accelerated shared-cache smoke | `d8 h8 nh1 nkv1 f8 T8 decode_tokens=2` | `runs/accel_qgpt2_prefill_decode2_d8_t8_shared_cache_smoke.log` | Started after converting the handoff to shared cache backing. It stalled before the first sequence marker with only the boot character `S` in the log, likely because the first shared-cache implementation reserved oversized per-head buffers and paid that startup cost before firmware markers. |
| QGPT2 accelerated shared-cache exact-size smoke | `d8 h8 nh1 nkv1 f8 T8 decode_tokens=2` | `runs/accel_qgpt2_prefill_decode2_d8_t8_shared_cache_smoke_exact.log` | Command: `PYTHONPATH=software/compiler python3 scripts/run_cv32e40p_prefill_decode_sequence.py --model qgpt2 --d-model 8 --d-head 8 --n-heads 1 --ffn-dim 8 --prompt-len 8 --decode-tokens 2 --seed 0 --run-rtl --maxcycles 50000000 --verilator-max-ticks 1000000000000 --timeout-s 900 --log-path runs/accel_qgpt2_prefill_decode2_d8_t8_shared_cache_smoke_exact.log`. It currently records only the boot character `S`; next debug step is adding earlier firmware markers and reducing generated descriptor storage before rerunning. |
| QGPT2 accelerated shared-cache boot/timer-fixed smoke | `d8 h8 nh1 nkv1 f8 T8 decode_tokens=2` | `runs/accel_qgpt2_prefill_decode2_d8_t8_shared_cache_timerfix_smoke.log` | `EXIT SUCCESS`; prefill 364,293 cycles, prefill-to-decode handoff 1 cycle, decode0 214,691 cycles, decode0-to-decode1 handoff 1 cycle, decode1 215,545 cycles, e2e 811,287 cycles. This validates the shared-cache pointer handoff on the tiny QGPT2 sequence. |
| QGPT2 accelerated shared-cache `int16_t` backing experiment | `d8 h8 nh1 nkv1 f8 T8 decode_tokens=2` | `runs/accel_qgpt2_prefill_decode2_d8_t8_shared_cache_i16_smoke.log` | `EXIT FAILURE`; prefill verification failed. The generated C ABI stores logical `TNPU_DTYPE_INT16` tensor values in `int32_t` backing arrays, so shared-cache storage must also use `int32_t` backing even though the tensor dtype is INT16. |

## Reportable Wins

| Model | Config | Accelerated e2e | ONNX2C e2e | Speedup | Section notes |
| --- | --- | ---: | ---: | ---: | --- |
| QLlama | `d128 h16 nh8 nkv4 f128 T8 decode_tokens=2` | 5,948,866 | 8,500,853 | 1.43x | Prefill wins: 3,375,256 vs 6,780,108 cycles (2.01x). Decode0 and decode1 are slower on the accelerated path: 1,192,925 vs 856,981 and 1,193,125 vs 862,522 cycles. |
| QLlama shared-cache | `d128 h16 nh8 nkv4 f128 T8 decode_tokens=2` | 5,897,388 | 8,500,853 | 1.44x | Prefill wins: 3,382,226 vs 6,780,108 cycles (2.00x). Handoff is reduced to 1 cycle for prefill-to-decode and 1 cycle for decode0-to-decode1. Decode0 and decode1 remain slower than ONNX2C: 1,248,626 vs 856,981 and 1,248,747 vs 862,522 cycles. |
| QGPT2 | `d128 h16 nh8 nkv8 f128 T8 decode_tokens=2` | 5,420,502 | 8,708,968 | 1.61x | Prefill wins: 2,924,429 vs 6,942,157 cycles (2.37x). Decode0 and decode1 are slower on the accelerated path: 1,057,619 vs 881,327 and 1,060,890 vs 884,243 cycles. Handoff copy costs in this pre-shared-cache accelerated run were 181,299 and 175,621 cycles. |
| QGPT2 shared-cache | `d128 h16 nh8 nkv8 f128 T8 decode_tokens=2` | 5,275,649 | 8,708,968 | 1.65x | Prefill wins: 2,915,943 vs 6,942,157 cycles (2.38x). Handoff is reduced to 1 cycle for prefill-to-decode and 1 cycle for decode0-to-decode1. Decode0 and decode1 remain slower than ONNX2C: 1,169,403 vs 881,327 and 1,172,647 vs 884,243 cycles. |

## Wall-Clock Conversion For Completed Two-Decode Rows

Wall time uses the current routed timing assumptions from the report:
ONNX2C CPU-only at `66.0 MHz`, integrated CPU+NPU at `50.17 MHz`.

| Workload | ONNX2C wall | CPU+NPU wall | Wall speedup | Notes |
| --- | ---: | ---: | ---: | --- |
| QLlama shared-cache `d128 h16 nh8 nkv4 f128 T8 decode_tokens=2` e2e | 128.80 ms | 117.55 ms | 1.10x | Full prefill + decode0 + decode1 remains a wall-clock win after the lower CPU+NPU clock. |
| QLlama shared-cache `d128` prefill | 102.73 ms | 67.42 ms | 1.52x | Prefill is the useful section-level win. |
| QLlama shared-cache `d128` decode0 | 12.98 ms | 24.89 ms | 0.52x | Decode alone loses. |
| QLlama shared-cache `d128` decode1 | 13.07 ms | 24.89 ms | 0.53x | Decode alone loses. |
| QGPT2 shared-cache `d128 h16 nh8 nkv8 f128 T8 decode_tokens=2` e2e | 131.95 ms | 105.16 ms | 1.25x | Stronger full-sequence wall-clock win than QLlama at the same `d128` scale. |
| QGPT2 shared-cache `d128` prefill | 105.18 ms | 58.12 ms | 1.81x | Prefill is the useful section-level win. |
| QGPT2 shared-cache `d128` decode0 | 13.35 ms | 23.31 ms | 0.57x | Decode alone loses. |
| QGPT2 shared-cache `d128` decode1 | 13.40 ms | 23.37 ms | 0.57x | Decode alone loses. |
| QLlama shared-cache `d288 h32 nh9 nkv3 f288 T8 decode_tokens=2` e2e | 593.03 ms | 241.95 ms | 2.45x | Full prefill + decode0 + decode1 is a strong wall-clock win at the larger scale. |
| QLlama shared-cache `d288` prefill | 473.46 ms | 138.35 ms | 3.42x | Prefill is the dominant win. |
| QLlama shared-cache `d288` decode0 | 59.71 ms | 51.58 ms | 1.16x | Decode becomes a modest wall-clock win at this scale. |
| QLlama shared-cache `d288` decode1 | 59.84 ms | 51.67 ms | 1.16x | Decode remains a modest wall-clock win. |

The clean smoke also validated two runner fixes needed before larger
measurements:

- Non-verbose Verilator stdout is now flushed after each pseudo-UART byte, so
  timeout-killed runs do not lose firmware progress markers.
- The original clean accelerated sequence handoff copied the full mutated cache
  tensors (`k_cache_h*`/`v_cache_h*`) into the next decode image instead of
  trying to copy the one-token `_td` tensors as complete caches. This made the
  runs correct but made handoff expensive. A shared-cache pointer handoff is now
  being tested to remove that full-cache copy.
- The sequence runner owns the testbench timer for the whole image and disables
  the runtime's per-program timer reset, so `sequence.*.total` lines are valid.
- Sequence runs now disable full final tensor dumps in runtime v2. Kernel-level
  cycle lines are still printed, but sequence totals are no longer dominated by
  printing large tensors.

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
