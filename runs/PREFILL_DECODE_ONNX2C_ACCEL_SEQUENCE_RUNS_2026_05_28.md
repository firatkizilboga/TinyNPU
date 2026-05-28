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

## Current Caveats

- The sequence runners currently execute prefill plus one decode token. Extending
  the accelerated path to two decode tokens requires constructing a second decode
  artifact at `prompt_len + 1` and handing off the first decode cache into it,
  rather than re-running the same fixed-position decode image.
- The ONNX2C sequence timing currently measures two independent ONNX2C calls in
  one firmware image. It is a CPU-only workload baseline with section timing; it
  does not yet feed prefill-produced K/V tensors into the decode ONNX graph.
- Git checkpoint commits are blocked in this checkout because `git status`
  currently fails with:
  `fatal: not a git repository: /home/firatkizilboga/TinyNPU/.git/worktrees/compiler-optimization`.
