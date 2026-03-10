## Summary
In long layer-by-layer MNIST cocotb flow, Unified Buffer word `addr=0` can be corrupted before execution, causing channel-local output mismatches (notably conv2 channels 0-3).

## Observed Behavior
- `test_mnist.py` previously failed at `conv2` with large mismatches while conv1 was bit-perfect.
- Debug comparison showed only UB word `0` diverged from compiler-produced payload.
- That word is the first packed bias word for conv2 (channels 0-3), consistent with mismatch pattern.

## Evidence
- Compiler payload expected at UB[0] for conv2 bias:
  - `[-4, 181, -298, -254]` packed into first 128-bit word.
- Runtime dump (`verification/cocotb/ub_dump_rtl.hex`) showed different value at UB[0], while other early words matched.
- Re-writing bias words immediately before `RUN` made conv2/conv3/fc bit-perfect.

## Reproduction
1. Use current MNIST artifacts (`npu_export/manifest.json`, `mnist_sample.npy`).
2. In `verification/cocotb/test_mnist.py`, temporarily disable the defensive bias reload block before `RUN`.
3. Run:
   - `make -C verification/cocotb -f Makefile.npu MODULE=test_mnist`
4. Observe:
   - conv1 passes, conv2 mismatch appears.
   - Inspect `verification/cocotb/ub_dump_rtl.hex`; UB word 0 differs from compiled conv2 bias word.

## Expected Behavior
- UB contents written by host interface remain stable until consumed.
- No silent corruption of `addr=0` (or any UB location).

## Current Mitigation
- `test_mnist.py` rewrites bias payload right before `RUN` to enforce deterministic behavior.

## Investigation Areas
- MMIO write ordering / doorbell timing.
- Control-unit latching behavior around command transitions.
- Host driver protocol assumptions for write/read interleaving.
- Any special-case behavior for UB base addresses.

## Root Cause (March 10, 2026)
- In `CTRL_READ_WAIT`, control continuously asserted `mmvr_wr_en`.
- In `mmio_interface.sv`, the internal `mmvr_wr_en` path had priority over host MMVR writes.
- Therefore, the first `WRITE_MEM` command issued while CU was still in `READ_WAIT` (common after UB debug reads) had its host payload overwritten by stale readback data.
- In MNIST flow, this first post-read write is `Bias` at `UB[0]`, producing the observed word-0 corruption and conv2 channel-0..3 mismatch.

## Fix (March 10, 2026)
- `mmio_interface.sv` now gives host MMVR writes priority over internal `mmvr_wr_en` updates.
- Added deterministic cocotb regression `test_mmio_readwrite_handoff.py`:
  - Repro sequence: `READ_MEM` -> immediate `WRITE_MEM`.
  - Asserts first post-read write data is preserved.
- `run_all_tests.sh` now runs this regression first.
