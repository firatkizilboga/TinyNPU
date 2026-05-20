# Transformer Scaling Sweep - 2026-05-18

## Clock Context

- CPU-only routed point: `57.1 MHz`.
- CPU+NPU real-BRAM routed estimate: `39.17 MHz`.
- Clock adjustment factor: `39.17 / 57.1 = 0.686`.
- A CPU+NPU cycle speedup must exceed about `1.46x` before it wins wall time against CPU-only.

## QLlama Decode

Command pattern:

```sh
python3 scripts/run_cv32e40p_qllama_block_benchmark.py \
  --mode decode --variant both --repeat-count 1 --prompt-len 8
```

Current RTL results:

| Config | CPU cycles | NPU cold cycles | NPU hot cycles | Cold cycle speedup | Hot cycle speedup | Cold wall speedup | Hot wall speedup | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `d8 h8 nh1 nkv1 f8 T8` | 41,382 | 73,768 | 71,896 | 0.56x | 0.58x | 0.38x | 0.40x | pass |
| `d16 h8 nh2 nkv1 f16 T8` | 86,032 | 110,081 | 105,951 | 0.78x | 0.81x | 0.54x | 0.56x | pass |
| `d32 h8 nh4 nkv2 f32 T8` | 224,719 | 187,030 | 172,789 | 1.20x | 1.30x | 0.82x | 0.89x | pass |
| `d48 h8 nh6 nkv3 f48 T8` | 426,993 | 271,712 | 241,336 | 1.57x | 1.77x | 1.08x | 1.21x | pass |
| `d64 h8 nh8 nkv4 f64 T8` | - | 363,424 | 310,783 | - | - | - | - | pass NPU-only correctness |
| `d64 h16 nh4 nkv2 f64 T8` | 668,974 | 330,045 | 277,909 | 2.03x | 2.41x | 1.39x | 1.65x | pass after disabling QLlama RoPE XFORM rewrite |

Interpretation:

- QLlama decode starts making cycle-count sense at `d32`.
- QLlama decode starts making wall-time sense, using the current FPGA clock estimates, at `d48`.
- The larger `d64 h16 nh4 nkv2 f64 T8` case now passes after QLlama was pinned to the host RoPE path instead of the NPU `XFORM ROPE_K16` rewrite.

### D64 Failure Isolation

Additional probes:

| Probe | Result | Meaning |
| --- | --- | --- |
| `d16 h16 nh1 nkv1 f16 T8` QLlama decode | now passes, NPU cold `111,617`, hot `106,539` | fixed by keeping QLlama RoPE on host |
| `d16 h16 nh1 nkv1 f16 T7` QLlama decode | previously failed before the fix | showed failure was not only the token-8 cache-block boundary |
| `d16 h16 nh1 nkv1 f16 T8` QLlama prefill | now passes, NPU cold `360,398`, hot `356,349` | fixed by keeping QLlama RoPE on host |
| `d32 h16 nh2 nkv1 f32 T8` QLlama prefill | now passes, NPU cold `648,578`, hot `635,746` | multi-head prefill coverage for `d_head=16` |
| isolated `1x16x16` INT16 GEMM | pass | raw projection-shaped K=16 systolic math works |
| isolated `1x16x9` INT16 GEMM | pass | raw score-shaped GEMM works |
| isolated `1x9x16` INT16 GEMM | pass | raw value-shaped GEMM works |

Current diagnosis:

- The failing surface was not the basic NPU GEMM datapath.
- The root cause was QLlama being silently rewritten from host dequantize -> RoPE -> quantize into NPU `XFORM ROPE_K16` for `d_head=16`.
- QLlama now disables that rewrite via plan metadata and keeps RoPE on the host. The standalone XFORM ROPE tests remain enabled, but QLlama no longer depends on that path for correctness.

## GPT2-Like Two-Block Prefill + Decode + Reuse

Command pattern:

```sh
python3 scripts/run_cv32e40p_gpt2_two_block_benchmark.py \
  --variant both --prompt-len 4 --repeat-count 1
```

This benchmark builds one current compiled artifact and runs:

- NPU path through runtime v2 with timed one-shot execution.
- CPU baseline through the existing CPU-only baseline emitter.

Current RTL results:

| Config | CPU cycles | NPU cold cycles | NPU hot cycles | Cold cycle speedup | Hot cycle speedup | Cold wall speedup | Hot wall speedup | Status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `d8 h8 nh1 f8 T4` | 301,937 | 297,773 | 288,252 | 1.01x | 1.05x | 0.69x | 0.72x | pass |
| `d16 h16 nh1 f16 T4` | 729,486 | 522,997 | 498,500 | 1.39x | 1.46x | 0.95x | 1.00x | pass |
| `d24 h24 nh1 f24 T4` | 1,333,079 | 757,888 | 709,199 | 1.76x | 1.88x | 1.21x | 1.29x | pass |
| `d16 h8 nh2 f16 T4` | - | - | - | - | - | - | - | compile fail: current two-block GPT2 builder only lowers the single-head path correctly |

Interpretation:

- The GPT2-like two-block chain is barely useful at `d8`.
- It reaches the current wall-time break-even region at `d16 h16 nh1`.
- It clearly wins at `d24 h24 nh1`.
- Multi-head GPT2 scaling is not validated in this manual two-block builder yet; the `d16 h8 nh2` case fails at compile time with a matmul dimension mismatch.

## Current Answer

- For QLlama decode, the first clean current point that makes sense after clock adjustment is `d48 h8 nh6 nkv3 f48 T8`.
- For GPT2-like two-block prefill+decode+reuse, the first clean current point that clearly makes sense after clock adjustment is `d24 h24 nh1 f24 T4`.
- The old conv gather path is not part of these results.
