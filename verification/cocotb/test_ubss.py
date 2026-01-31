"""
Test for UBSS (UB + Skewer + Systolic) integration.

Uses pre-initialized data from buffer_init.hex:
- Row 0: DEAD sentinel (stale detection)
- Rows 1-4: Incrementing pattern (input matrix A)
- Rows 8-11: Identity matrix (weight matrix B)

Expected result: A * I = A
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

# Parameters (must match RTL defines)
N = 4
DATA_WIDTH = 16
ACC_WIDTH = 64


def unpack_flat(flat_val, width=DATA_WIDTH, count=N):
    """Unpack flattened output into count values."""
    result = []
    for i in range(count):
        val = (flat_val >> (i * width)) & ((1 << width) - 1)
        result.append(val)
    return result


def unpack_results(flat_val):
    """Unpack N*N results array (64-bit accumulators)."""
    results = []
    for row in range(N):
        row_vals = []
        for col in range(N):
            idx = row * N + col
            val = (flat_val >> (idx * ACC_WIDTH)) & ((1 << ACC_WIDTH) - 1)
            # Convert to signed
            if val >= (1 << (ACC_WIDTH - 1)):
                val -= (1 << ACC_WIDTH)
            row_vals.append(val)
        results.append(row_vals)
    return results


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
    dut.precision_mode.value = 1  # INT16 mode
    dut.compute_enable.value = 0
    dut.drain_enable.value = 0
    dut.acc_clear.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_ubss_matmul(dut):
    """Test UB → Skewer → Systolic matmul with identity matrix."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)

    # ========================================================================
    # Data from buffer_init.hex
    # ========================================================================
    # Input matrix A (rows 1-4):
    #   [1, 2, 3, 4]
    #   [5, 6, 7, 8]
    #   [9, A, B, C]
    #   [D, E, F, 10]
    #
    # Weight matrix B (rows 8-11): Identity
    #   [1, 0, 0, 0]
    #   [0, 1, 0, 0]
    #   [0, 0, 1, 0]
    #   [0, 0, 0, 1]
    #
    # Expected C = A * I = A

    dut._log.info("=" * 70)
    dut._log.info("UBSS Integration Test: A * I = A")
    dut._log.info("=" * 70)

    await ClockCycles(dut.clk, 2)

    # ========================================================================
    # Phase 1: Clear accumulators
    # ========================================================================
    dut.acc_clear.value = 1
    await RisingEdge(dut.clk)
    dut.acc_clear.value = 0

    # ========================================================================
    # Phase 2: Feed data through UB → Skewer → Systolic
    # ========================================================================
    dut.en.value = 1
    dut.compute_enable.value = 1

    input_skewed = []
    weight_skewed = []

    # Feed N addresses
    for col in range(N):
        # Set markers
        dut.input_first_in.value = 1 if col == 0 else 0
        dut.input_last_in.value = 1 if col == N - 1 else 0
        dut.weight_first_in.value = 1 if col == 0 else 0
        dut.weight_last_in.value = 1 if col == N - 1 else 0

        # Set addresses (input: 1-4, weight: 8-11)
        dut.input_addr.value = col + 1
        dut.weight_addr.value = 8 + col

        await RisingEdge(dut.clk)

        # Capture skewed outputs
        input_skewed.append(unpack_flat(dut.input_skewed_flat.value.integer))
        weight_skewed.append(unpack_flat(dut.weight_skewed_flat.value.integer))

    # Clear address inputs
    dut.input_first_in.value = 0
    dut.input_last_in.value = 0
    dut.weight_first_in.value = 0
    dut.weight_last_in.value = 0
    dut.input_addr.value = 0
    dut.weight_addr.value = 0

    # Continue feeding zeros and let computation complete
    # Need 2*N cycles for full diagonal wavefront
    for _ in range(2 * N):
        await RisingEdge(dut.clk)
        input_skewed.append(unpack_flat(dut.input_skewed_flat.value.integer))
        weight_skewed.append(unpack_flat(dut.weight_skewed_flat.value.integer))

    # Disable compute, wait for results
    dut.compute_enable.value = 0
    await ClockCycles(dut.clk, 2)

    # ========================================================================
    # Print skewed data timeline
    # ========================================================================
    dut._log.info("INPUT SKEWED DATA:")
    for t, out in enumerate(input_skewed):
        out_hex = [f"0x{v:04x}" for v in out]
        dut._log.info(f"  Cycle {t:2d}: {out_hex}")

    dut._log.info("WEIGHT SKEWED DATA:")
    for t, out in enumerate(weight_skewed):
        out_hex = [f"0x{v:04x}" for v in out]
        dut._log.info(f"  Cycle {t:2d}: {out_hex}")

    # ========================================================================
    # Read results
    # ========================================================================
    results = unpack_results(dut.results_flat.value.integer)

    dut._log.info("=" * 70)
    dut._log.info("SYSTOLIC ARRAY RESULTS:")
    for row in range(N):
        row_str = [f"{results[row][col]:4d}" for col in range(N)]
        dut._log.info(f"  Row {row}: [{', '.join(row_str)}]")

    # Expected result: A * I = A
    # Row 0: [1, 2, 3, 4]
    # Row 1: [5, 6, 7, 8]
    # Row 2: [9, 10, 11, 12]
    # Row 3: [13, 14, 15, 16]
    expected = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
    ]

    dut._log.info("=" * 70)
    dut._log.info("EXPECTED RESULTS (A * I = A):")
    for row in range(N):
        row_str = [f"{expected[row][col]:4d}" for col in range(N)]
        dut._log.info(f"  Row {row}: [{', '.join(row_str)}]")

    # Check results
    passed = True
    for row in range(N):
        for col in range(N):
            if results[row][col] != expected[row][col]:
                dut._log.error(
                    f"Mismatch at [{row}][{col}]: "
                    f"got {results[row][col]}, expected {expected[row][col]}"
                )
                passed = False

    if passed:
        dut._log.info("✓ UBSS test PASSED: A * I = A verified!")
    else:
        dut._log.error("✗ UBSS test FAILED: Result mismatch")

    assert passed, "Matrix multiplication result mismatch"
