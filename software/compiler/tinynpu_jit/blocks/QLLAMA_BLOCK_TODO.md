# QLlamaBlock TODO

## Runtime correctness

- [x] Fix prefill RoPE lowering so prompt tokens use per-token positions instead of applying `position=0` to the entire `[seq_len, head_dim]` matrix.
- [x] Fix decode host emulation cache initialization so prefilled `K` and `V` cache tensors keep their seeded data instead of being zero-initialized at runtime.
- [x] Fix compiled-artifact verification bookkeeping so `VerifyTensor.tensor_name` and `expected_tensors` use the same keying convention.
- [x] Add host-emulation checks for both prefill and decode paths, and make them fail on intermediate mismatches such as `q_rope`, `k_rope`, cache tensors, and final `out`.

## Llama compatibility

- [x] Document that local RoPE uses the Hugging Face split-halves convention, while raw Meta checkpoints use paired/interleaved rotation; note that raw Meta `Q/K` weights must be permuted on import.
- [x] Document the KV cache layout convention explicitly: `K` is stored as `(d_head, cache_len)` while `V` is stored as `(cache_len, d_head)`, with the `K` transpose happening inside cache scatter helpers and decode cache seeding.
- [x] Expose `rms_norm_eps` on `QLlamaBlockConfig` instead of hard-coding `1e-5`.
- [x] Expose structured RoPE config on `QLlamaBlockConfig`, including at least `rope_theta` and optional `rope_scaling`.
- [x] Change the Llama-3-oriented default `rope_theta` from `10000.0` to `500000.0`, or make the caller pass it explicitly.
- [ ] Add support for Llama 3.1+ `rope_scaling` (`rope_type="llama3"` and related scaling factors) if the block is intended to model those checkpoints faithfully.

## Quantization and import clarity

- [x] Document why `V` stays in projected INT16 space while `Q/K` round-trip through dequantize -> RoPE -> requantize.
- [x] Document that `reference_prefill()` and `reference_decode()` are hardware-faithful quantized references, not ideal floating-point Llama references.
- [ ] Revisit `QLlamaBlock.from_fp32()` so checkpoint import is scale-aware rather than plain `round()` to INT16.
- [x] Note that `QLlamaBlock.random()` uses tiny synthetic integer weights for plumbing tests and is not representative of real checkpoint statistics.

## Test coverage

- [x] Add a regression test that runs `build_llama_prefill_artifact(...).run_host_emulation(...)` and checks the compiled path against the stored reference output.
- [x] Add a regression test that runs `build_llama_decode_artifact(...).run_host_emulation(...)` and checks seeded cache contents before and after the decode append.
- [ ] Add a targeted test for HF-vs-Meta RoPE import compatibility so loader permutations are validated explicitly.
