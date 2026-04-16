# Attention Perf Findings

Date: 2026-04-16

## Summary

- The headline decode-attention speedup did not come from RoPE.
- The meaningful speedup came from the `softmax_f16 -> TNPU_WRITE_XFORM_Q_F16_I16 -> seg_value` handoff.
- Small decode shapes are still below crossover and can lose to the CPU because fixed segment overhead dominates.
- Scaled decode shapes can win clearly once the fast value-input path is used.

## Reproduced Results

### Small decode-attention point

Config:

- `d_model=16`
- `n_heads=1`
- `n_kv_heads=1`
- `d_head=16`
- `token_capacity=16`
- `token_indices=[0,1]`

Measured totals:

- NPU hot: `91,510`
- NPU cold: `95,396`
- CPU-only baseline: `75,516`

Interpretation:

- NPU loses on this tiny point.
- The loss is not because matmul compute is bad.
- The loss is driven by fixed staging/readback/control costs, especially on the score path.

Relevant breakdown:

- projections: NPU `17,542`, CPU `34,189`
- score path: NPU `40,162`, CPU `1,887`
- value path: NPU `18,392`, CPU `14,347`

### Scaled decode-attention point

Config:

- `d_head=16`
- `token_capacity=32`
- active tokens `[1,9,17,25]`

Current fast-path artifact:

- program: `cv32e40p_decode_attention_d16_t32_n4_s1_v2`
- `seg_score.npu = 8,933`
- `hostop.softmax_scores_f16 = 4,678`
- `seg_value.npu = 12,689`
- NPU hot total = `26,300`
- NPU cold total = `33,152`

CPU artifact:

- program: `cv32e40p_decode_scale_cpu_d16_t32_n4_s1`
- `seg_score.cpu = 56,272`
- `hostop.softmax_scores = 1,558`
- `hostop.quantize_probs = 19,391`
- `seg_value.cpu = 10,756`
- CPU total = `87,977`

Interpretation:

- NPU hot speedup: `3.35x`
- NPU cold speedup: `2.65x`

## Main Performance Conclusion

The decisive optimization is the fast value-input staging path:

- `softmax_f16`
- absorbed write transform
- `TNPU_WRITE_XFORM_Q_F16_I16`

This avoids the exact float32-to-int16 staging path before `seg_value`.

## Stale Artifact Trap

An older generated benchmark artifact reproduced a much worse `seg_value`:

- stale path `seg_value.npu = 32,690`
- current fast path `seg_value.npu = 12,689`

Root cause:

- stale artifact used `TNPU_HOST_SOFTMAX` plus `TNPU_WRITE_QUANTIZE_F32_TO_INT16`
- current fast artifact uses `softmax_f16` plus `TNPU_WRITE_XFORM_Q_F16_I16`

This means benchmark comparisons must use current `decode_attention_*_v2` artifacts rather than stale `decode_scale_*` NPU artifacts when evaluating the optimized attention path.

## RoPE Conclusion

- RoPE did not produce the measured headline speedup.
- Native K-RoPE may still be architecturally useful for a cleaner decode path.
- RoPE is not currently a first-order performance lever compared with boundary-path fixes.

## Practical Priority Order

1. Keep the `softmax_f16 + XFORM_Q_F16_I16` value-input path.
2. Reduce tiny decode segment fixed overheads.
3. Move more of the Q-side boundary path off the host if worthwhile.
4. Treat RoPE as secondary unless it blocks or simplifies a larger path.
