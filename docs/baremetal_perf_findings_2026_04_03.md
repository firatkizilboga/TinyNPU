## Bare-Metal Performance Findings (2026-04-03)

### CPU baseline cleanup

- The generated CV32 CPU baseline no longer pulls `__ashrdi3` / `__divdi3` into the hot matmul path.
- The shared runtime template now uses:
  - 32-bit DI-exp / DI-sigmoid helpers where possible
  - a custom 64-to-32 rounded requant shift helper
  - direct `int32_t*` tensor data access inside `host_matmul`

### `is_zero` MLP on `CV32E40P + TinyNPU`

Measured after the CPU baseline rewrite:

- `preload.ub_image = 128703`
- `preload.im_segment_000 = 783`
- `hostop.q_in = 26920`
- `segment.segment_000.npu = 46098`
- `segment.segment_000.cpu = 291911`
- `hostop.dq_out = 250`

Derived comparisons:

- Segment-only: NPU is about `6.33x` faster than CPU (`291911 / 46098`)
- Hot end-to-end excluding preload:
  - CPU: `319081`
  - NPU: `73268`
  - NPU is about `4.36x` faster
- Cold end-to-end including preload:
  - CPU: `319081`
  - NPU: `202754`
  - NPU is about `1.57x` faster

The old CPU baseline for the same segment was `1295886` cycles, so removing the heavy 64-bit helper path improved the CPU baseline by about `4.44x`.

### Repeated-run emitter support

- The bare-metal emitter now supports `repeat_count`.
- UB and IM preload stay outside the repeated inference body.
- The generated program accumulates:
  - repeated host totals
  - repeated NPU totals
  - repeated CPU-baseline totals
  - hot and cold averages
- A `10x` repeated `is_zero` run was emitted successfully, but the current `run-generated-iszero-mlp` Makefile target timed out at `VERILATOR_MAX_TICKS=10000000000` before completion. The emitted firmware itself started correctly.

### Conv path interpretation

From the fair conv benchmark summary:

- `host_remaining = 23649`
- `im2col = 19608`
- `layout_restore = 3588`

So `im2col + layout_restore = 23196`, which is about `98%` of host-remaining work.

Warm conv case-study numbers used in the report:

- `C_npu = 1778`
- `C_ovh = 35491`
- `C_host = 23644`

Using the current pack/unpack cost model, rough conv pack+unpack overhead is about `21.6k` cycles. If MMVR-side packing removed that whole portion of the offload path, the warm conv total would drop from about `60.9k` cycles to about `39.3k` cycles, which is roughly a `1.55x` runtime improvement. That would move the reported full-model speedup from about `6.05x` to about `9.4x`. This is an inference from the current model, not a measured RTL run.

### Architectural takeaway

- The main bottleneck in the fair conv path is not the MAC array.
- Two separate costs are both large:
  - host-side layout work (`im2col`, `layout_restore`)
  - offload overhead (packing, MMIO traffic, readback)
- Custom writeback helps preserve accelerator-native matrix layout, but it does not by itself eliminate `layout_restore` for conv-to-conv chains under the current partitioner.
- The most promising next software optimization is to fuse `layout_restore + next im2col`, or better, generate the next conv input directly from the previous conv matrix form without reconstructing a normal feature map first.
