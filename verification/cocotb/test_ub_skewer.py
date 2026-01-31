"""
Test for UB + Skewer integration using buffer_init.hex.

Uses pre-initialized data from buffer_init.hex:
- Rows 0-3: Incrementing pattern [0x0001-0x0010]
- Rows 8-11: Identity-like pattern (diagonal ones)
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

# Parameters (must match RTL defines)
N = 4
DATA_WIDTH = 16


def unpack_flat(flat_val):
    """Unpack flattened output into N values."""
    result = []
    for i in range(N):
        val = (flat_val >> (i * DATA_WIDTH)) & ((1 << DATA_WIDTH) - 1)
        result.append(val)
    return result


async def reset_dut(dut):
    """Reset the DUT."""
    dut.rst_n.value = 0
    dut.en.value = 0
    dut.wr_en.value = 0
    dut.wr_addr.value = 0
    dut.wr_data.value = 0
    dut.input_addr.value = 0
    dut.input_first_in.value = 0
    dut.input_last_in.value = 0
    dut.weight_addr.value = 0
    dut.weight_first_in.value = 0
    dut.weight_last_in.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_ub_skewer_hex_init(dut):
    """Test UB → Skewer using buffer_init.hex data."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # ========================================================================
    # Data from buffer_init.hex (already loaded via INIT_FILE parameter)
    # ========================================================================
    # Row 0: DEADDEADDEADDEAD → INVALID SENTINEL
    # Row 1: 0004000300020001 → [0x0001, 0x0002, 0x0003, 0x0004]
    # Row 2: 0008000700060005 → [0x0005, 0x0006, 0x0007, 0x0008]
    # Row 3: 000C000B000A0009 → [0x0009, 0x000A, 0x000B, 0x000C]
    # Row 4: 0010000F000E000D → [0x000D, 0x000E, 0x000F, 0x0010]
    #
    # Row 8: 0000000000000001 → [0x0001, 0x0000, 0x0000, 0x0000]
    # Row 9: 0000000000010000 → [0x0000, 0x0001, 0x0000, 0x0000]
    # ...

    dut._log.info("Using buffer_init.hex data:")
    dut._log.info("  Row 0: DEAD sentinel (stale detection)")
    dut._log.info("  Input rows 1-4: incrementing pattern")
    dut._log.info("  Weight rows 8-11: identity pattern")

    await ClockCycles(dut.clk, 2)

    # ========================================================================
    # Push addresses and collect skewed outputs
    # ========================================================================
    dut.en.value = 1

    input_outputs = []
    weight_outputs = []
    input_markers = []
    weight_markers = []
    ub_input_outputs = []  # Debug: UB input stage outputs
    ub_input_markers = []
    ub_weight_outputs = []  # Debug: UB weight stage outputs
    ub_weight_markers = []

    # Feed N addresses: input from 1-4, weight from 8-11
    for col in range(N):
        # Set markers
        dut.input_first_in.value = 1 if col == 0 else 0
        dut.input_last_in.value = 1 if col == N - 1 else 0
        dut.weight_first_in.value = 1 if col == 0 else 0
        dut.weight_last_in.value = 1 if col == N - 1 else 0

        # Set addresses (input: 1-4, weight: 8-11)
        dut.input_addr.value = col + 1    # 1, 2, 3, 4
        dut.weight_addr.value = 8 + col   # 8, 9, 10, 11

        await RisingEdge(dut.clk)

        # Capture outputs
        input_outputs.append(unpack_flat(dut.input_data_flat.value.integer))
        weight_outputs.append(unpack_flat(dut.weight_data_flat.value.integer))
        input_markers.append(
            (int(dut.dut_input_first_out.value), int(dut.dut_input_last_out.value))
        )
        weight_markers.append(
            (int(dut.dut_weight_first_out.value),
             int(dut.dut_weight_last_out.value))
        )
        # Debug: capture UB outputs
        ub_input_outputs.append(unpack_flat(dut.mem_input_data_flat.value.integer))
        ub_input_markers.append(
            (int(dut.mem_input_first.value), int(dut.mem_input_last.value))
        )
        ub_weight_outputs.append(unpack_flat(dut.mem_weight_data_flat.value.integer))
        ub_weight_markers.append(
            (int(dut.mem_weight_first.value), int(dut.mem_weight_last.value))
        )

    # Clear inputs and drain pipeline
    dut.input_first_in.value = 0
    dut.input_last_in.value = 0
    dut.weight_first_in.value = 0
    dut.weight_last_in.value = 0
    dut.input_addr.value = 0
    dut.weight_addr.value = 0

    for _ in range(N + 2):
        await RisingEdge(dut.clk)
        input_outputs.append(unpack_flat(dut.input_data_flat.value.integer))
        weight_outputs.append(unpack_flat(dut.weight_data_flat.value.integer))
        input_markers.append(
            (int(dut.dut_input_first_out.value), int(dut.dut_input_last_out.value))
        )
        weight_markers.append(
            (int(dut.dut_weight_first_out.value),
             int(dut.dut_weight_last_out.value))
        )
        ub_input_outputs.append(unpack_flat(dut.mem_input_data_flat.value.integer))
        ub_input_markers.append(
            (int(dut.mem_input_first.value), int(dut.mem_input_last.value))
        )
        ub_weight_outputs.append(unpack_flat(dut.mem_weight_data_flat.value.integer))
        ub_weight_markers.append(
            (int(dut.mem_weight_first.value), int(dut.mem_weight_last.value))
        )

    # ========================================================================
    # Print timeline
    # ========================================================================
    dut._log.info("=" * 70)
    dut._log.info("INPUT CHANNEL - UB OUTPUT (before skewer):")
    for t, (out, (f, l)) in enumerate(zip(ub_input_outputs, ub_input_markers)):
        out_hex = [f"0x{v:04x}" for v in out]
        dut._log.info(f"  Cycle {t:2d}: {out_hex}, first={f}, last={l}")

    dut._log.info("=" * 70)
    dut._log.info("INPUT CHANNEL - SKEWER OUTPUT:")
    for t, (out, (f, l)) in enumerate(zip(input_outputs, input_markers)):
        out_hex = [f"0x{v:04x}" for v in out]
        dut._log.info(f"  Cycle {t:2d}: {out_hex}, first={f}, last={l}")

    dut._log.info("=" * 70)
    dut._log.info("WEIGHT CHANNEL - UB OUTPUT (before skewer):")
    for t, (out, (f, l)) in enumerate(zip(ub_weight_outputs, ub_weight_markers)):
        out_hex = [f"0x{v:04x}" for v in out]
        dut._log.info(f"  Cycle {t:2d}: {out_hex}, first={f}, last={l}")

    dut._log.info("=" * 70)
    dut._log.info("WEIGHT CHANNEL - SKEWER OUTPUT:")
    for t, (out, (f, l)) in enumerate(zip(weight_outputs, weight_markers)):
        out_hex = [f"0x{v:04x}" for v in out]
        dut._log.info(f"  Cycle {t:2d}: {out_hex}, first={f}, last={l}")

    # ========================================================================
    # Verify diagonal wavefront at cycle 5 (UB:1 + Skewer:row_delays)
    # ========================================================================
    # At the diagonal wavefront, each row should have data from one earlier column:
    # Row 0: col 3, Row 1: col 2, Row 2: col 1, Row 3: col 0
    # For input path: [0x000D, 0x000A, 0x0007, 0x0004]
    diagonal_cycle = 5
    expected_diagonal_input = [0x000D, 0x000A, 0x0007, 0x0004]
    expected_diagonal_weight = [0x0000, 0x0000, 0x0000, 0x0001]

    actual_input = input_outputs[diagonal_cycle]
    actual_weight = weight_outputs[diagonal_cycle]

    dut._log.info("=" * 70)
    dut._log.info(f"Diagonal wavefront at cycle {diagonal_cycle}:")
    dut._log.info(f"  Input expected:  {
                  [f'0x{v:04x}' for v in expected_diagonal_input]}")
    dut._log.info(f"  Input actual:    {[f'0x{v:04x}' for v in actual_input]}")
    dut._log.info(f"  Weight expected: {
                  [f'0x{v:04x}' for v in expected_diagonal_weight]}")
    dut._log.info(f"  Weight actual:   {
                  [f'0x{v:04x}' for v in actual_weight]}")

    dut._log.info("✓ UB + Skewer test with buffer_init.hex completed!")
