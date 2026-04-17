# UB Growth Findings

Current unified-buffer growth is not a pure capacity-only change.

## Current constraints

- RTL UB depth is set by `BUFFER_DEPTH` in `rtl/defines.sv`.
- UB and instruction memory are selected by a word-address split at `IM_BASE_ADDR = 0x8000`.
- Host/runtime shared-SRAM routing also uses the same `addr >= 0x8000` split.
- The shared-SRAM testbench model currently hardcodes `NPU_SHARED_WORDS = 8192`.
- The NPU address path is 16-bit wide end to end.

## Safe growth regime

Increasing UB is relatively low-risk if the resulting UB address range stays strictly below `0x8000`.

That requires coordinated updates in at least:

- `rtl/defines.sv`
- `external/cv32e40p/example_tb/core/mm_ram.sv`
- any tests that hardcode UB capacity as `0x8000`

The compiler side already attempts to read `BUFFER_DEPTH` from RTL defines, so moderate growth below the split should propagate cleanly once the hardware/testbench values match.

## Unsafe regime without memory-map changes

If UB grows to or beyond `0x8000` words, the current system breaks conceptually because:

- UB vs IM routing uses `addr >= 0x8000`
- runtime shared-SRAM helpers use the same split
- instruction memory addresses are computed as `addr - IM_BASE_ADDR`

At that point, UB growth becomes a memory-map redesign, not a parameter tweak.

## Practical conclusion

- `BUFFER_DEPTH > 8192` and `< 0x8000` is likely feasible with coordinated updates.
- `BUFFER_DEPTH >= 0x8000` requires moving `IM_BASE_ADDR` and updating routing/runtime/testbench logic accordingly.
- Any growth beyond the current 16-bit address assumptions would also require widening the address path.
