# QGPT2Block TODO

## Runtime correctness

- [x] Add host-emulation checks for both prefill and decode paths, and make them fail on intermediate mismatches such as `x_norm1`, `x_norm2`, per-head `scores`, per-head `probs`, `ffn_fc_int`, and final `out`.
- [x] Verify decode host emulation cache initialization: `_add_decode_head_runtime_tensors` seeds `k_cache_h{h}.data` / `v_cache_h{h}.data` from `prefill_ref` in the Python builder, but confirm the compiled artifact actually surfaces that seeded data at runtime instead of zero-initializing the cache (same class of bug as QLlamaBlock).
- [x] Fix compiled-artifact verification bookkeeping if `VerifyTensor.tensor_name` and `expected_tensors` use different keying conventions (same pattern as QLlamaBlock: `plan.add_verification_step("out", "gpt2_prefill_out")` supplies two distinct strings).
- [ ] Audit the `DType.INT16`-declared `x_norm1` / `x_norm2` tensors that actually carry fp16 bits (`layernorm` emits `output_encoding="fp16_bits"`, `quant_x_norm*` reads `input_encoding="fp16_bits"`). Any new op added to the graph without reading the encoding attrs will misinterpret these tensors. Either split into two TensorSpec dtypes or promote `encoding` to a first-class TensorSpec field.

## GPT-2 compatibility

- [x] Expose `layer_norm_epsilon` on `QGPT2BlockConfig` instead of hard-coding `1e-5` in both references and the IR tail.
- [x] Document that the quantized path fuses `activation="h_gelu"` (piecewise-linear hard GELU) into `c_fc`, while real HF GPT-2 uses the tanh approximation (`_gpt2_gelu_tanh`). The fp32 reference (`reference_prefill_float` / `reference_decode_float`) uses tanh GELU; the quantized reference and the NPU IR use hard GELU. This is a silent numerical divergence from HF output on every FFN, every layer.
- [x] Document that positional information is added **outside** the block via `_materialize_inputs(token_emb, pos_emb, positions)`, so `x_in` already contains `wte + wpe[positions]`. Callers threading multi-step decode must increment the `pos_emb` slice themselves; the block has no position awareness.
- [x] Document that HF stores `attn.c_attn` as a single `(d_model, 3*d_model)` fused projection, while this implementation splits into `3 * n_heads` per-head `(d_model, d_head)` matmuls via `split_c_attn_weights` / `split_c_attn_biases_fp32`. Mathematically equivalent on output, but the weight layout differs and any importer must perform the split.
- [x] Document that the KV cache stores `K` pre-transposed as `(d_head, cache_len)` and `V` as `(cache_len, d_head)` (see `_add_decode_head_runtime_tensors` `k_base` / `v_base` seeding). Callers cannot pass raw `(cache_len, d_head)` K without transposition.
- [x] Rename `_layernorm_runtime_approx` — the function is numerically equivalent to `_layernorm_exact` up to ulp differences; the real "approximation" is the `_fp16_roundtrip` applied on top of it by the caller. Current naming suggests a different kernel where there isn't one.

## Quantization and import clarity

- [ ] Revisit `QGPT2Block.from_fp32()` so checkpoint import is scale-aware rather than plain `clip(round(weight), -32768, 32767)` to INT16. Real HF GPT-2 weights have magnitudes well within ±1 in fp32; raw rounding collapses almost everything to zero.
- [x] Document that `_quantize_bias_fp32` computes `round(bias / act_scale)` as INT32, which implicitly assumes `weight_scale == 1` (since the matmul's integer output scale is `input_scale * weight_scale = act_scale * 1`). Once scale-aware weight import lands, this must propagate the actual per-tensor or per-channel weight scale.
- [x] Document that `reference_prefill()` and `reference_decode()` are hardware-faithful quantized references (INT16 weights with `weight_scale=1`, INT32 biases at `act_scale`, fp16 softmax, fp16 layernorm roundtrip, hard GELU), while `reference_prefill_float()` and `reference_decode_float()` are pure-fp32 HF-GPT-2-faithful references (no quant, tanh GELU, fp32 layernorm). Use the float references as the architectural oracle and the quant references as the NPU oracle.
- [x] Note that `QGPT2Block.random()` uses tiny synthetic INT16 weights in `[-2, 3)` and fp32 biases in `[-0.05, 0.05]` for plumbing tests — not representative of real HF GPT-2 checkpoint statistics.
- [ ] Add an explicit HF state_dict -> `QGPT2Block.from_fp32` loader once `from_fp32` is scale-aware, so that `transformer.h.N.{ln_1,ln_2,attn.c_attn,attn.c_proj,mlp.c_fc,mlp.c_proj}` keys map cleanly onto block fields (including packing `ln_*.weight` and `ln_*.bias` into the `(2, d_model)` `ln_*_wb` layout).

## Test coverage

- [x] Add a regression test that runs `build_gpt2_prefill_artifact(...).run_host_emulation(...)` and checks the compiled path numerically against the stored `reference_prefill` output for all intermediates, not just `out`.
- [x] Add a regression test that runs `build_gpt2_decode_artifact(...).run_host_emulation(...)` and verifies seeded K/V cache contents before and after the decode append, and final `out` against `reference_decode`.
- [x] Add a test comparing `reference_prefill` to `reference_prefill_float` at a small shape to quantify and pin the combined error budget from quantization + fp16 layernorm + fp16 softmax + hard GELU. Regressions in any one of those host ops will move this delta.
- [ ] Add a test comparing `reference_prefill_float` output against `transformers.GPT2Model.h[0]` (or a hand-rolled equivalent) on identical small-shape inputs, to validate that the architectural skeleton matches HF and catch subtle deltas (LayerNorm epsilon placement, GELU formula, bias broadcasting) independent of quantization.
- [ ] Add a loader test for HF GPT-2 `state_dict()` -> `QGPT2Block.from_fp32` round-trip, once scale-aware import lands.
