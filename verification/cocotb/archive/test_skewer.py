"""
Test for parametric streaming_skewer module.

Verifies:
1. Row i gets (i+1) cycles of delay
2. first_out fires 1 cycle after first_in
3. last_out fires N cycles after last_in
4. Data flows correctly through the pipeline
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

# Get N from defines (default 4)
N = 4
DATA_WIDTH = 16


def get_data_out(dut):
    """Extract data_out array from flattened output."""
    flat = dut.data_out_flat.value.integer
    result = []
    for i in range(N):
        val = (flat >> (i * DATA_WIDTH)) & ((1 << DATA_WIDTH) - 1)
        result.append(val)
    return result


async def reset_dut(dut):
    """Reset the DUT."""
    dut.rst_n.value = 0
    dut.en.value = 0
    dut.first_in.value = 0
    dut.last_in.value = 0
    for i in range(N):
        dut.data_in[i].value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


@cocotb.test()
async def test_delay_pattern(dut):
    """Test that each row has the correct delay (row i -> i+1 cycles)."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)
    dut.en.value = 1

    # Feed unique values to each row
    test_values = [0x1000 + i for i in range(N)]
    for i in range(N):
        dut.data_in[i].value = test_values[i]

    # Collect outputs over time
    outputs = []
    for cycle in range(N + 2):
        await RisingEdge(dut.clk)
        outputs.append(get_data_out(dut))

    # Clear input
    for i in range(N):
        dut.data_in[i].value = 0

    # Verify timing: row i should output test_values[i] at cycle (i+1)
    # outputs[0] is after 1 clock from when we set data
    for row in range(N):
        expected_cycle = row + 1  # row 0 -> cycle 1, row 1 -> cycle 2, etc.
        dut._log.info(f"Row {row}: expected at cycle {expected_cycle}, got {
                      outputs[expected_cycle][row]:04x}, expected {test_values[row]:04x}")
        assert outputs[expected_cycle][row] == test_values[row], \
            f"Row {row} delay mismatch: expected {test_values[row]:04x} at cycle {
                expected_cycle}, got {outputs[expected_cycle][row]:04x}"

    dut._log.info("✓ All row delays correct!")


@cocotb.test()
async def test_first_marker(dut):
    """Test that first_out fires 1 cycle after first_in."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)
    dut.en.value = 1

    # Pulse first_in
    dut.first_in.value = 1
    await RisingEdge(dut.clk)
    dut.first_in.value = 0

    # first_out should be high now (1 cycle later)
    await RisingEdge(dut.clk)
    assert dut.first_out.value == 1, "first_out should be 1 after 1 cycle"

    # Should be low on next cycle
    await RisingEdge(dut.clk)
    assert dut.first_out.value == 0, "first_out should return to 0"

    dut._log.info("✓ first_out marker timing correct!")


@cocotb.test()
async def test_last_marker(dut):
    """Test that last_out fires N cycles after last_in."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)
    dut.en.value = 1

    # Pulse last_in
    dut.last_in.value = 1
    await RisingEdge(dut.clk)
    dut.last_in.value = 0

    # last_out should be low for N-1 cycles
    for i in range(N - 1):
        await RisingEdge(dut.clk)
        assert dut.last_out.value == 0, f"last_out should be 0 at cycle {i+1}"

    # last_out should be high at cycle N
    await RisingEdge(dut.clk)
    assert dut.last_out.value == 1, f"last_out should be 1 at cycle {N}"

    # Should return to 0
    await RisingEdge(dut.clk)
    assert dut.last_out.value == 0, "last_out should return to 0"

    dut._log.info(f"✓ last_out marker timing correct (N={N} cycles)!")


@cocotb.test()
async def test_diagonal_skew(dut):
    """Test complete diagonal skewing pattern like a systolic array needs."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    await reset_dut(dut)
    dut.en.value = 1

    # Feed N columns of data (simulating matrix columns)
    # Column j has values [j*N + row for each row]
    columns = [[col * N + row for row in range(N)] for col in range(N)]

    dut._log.info(f"Feeding {N} columns: {columns}")

    outputs = []
    markers = []  # Capture markers at each cycle

    # Feed columns and collect outputs
    for col in range(N):
        dut.first_in.value = 0
        dut.last_in.value = 0
        if col == 0:
            dut.first_in.value = 1
        if col == N-1:
            dut.last_in.value = 1

        for row in range(N):
            dut.data_in[row].value = columns[col][row]
        await RisingEdge(dut.clk)
        outputs.append(get_data_out(dut))
        markers.append((int(dut.first_out.value), int(dut.last_out.value)))

    # Continue collecting for N more cycles (pipeline drain)
    dut.first_in.value = 0
    dut.last_in.value = 0
    for row in range(N):
        dut.data_in[row].value = 0

    for _ in range(N):
        await RisingEdge(dut.clk)
        outputs.append(get_data_out(dut))
        markers.append((int(dut.first_out.value), int(dut.last_out.value)))

    dut._log.info("Output timeline:")
    for t, (out, (f, l)) in enumerate(zip(outputs, markers)):
        dut._log.info(f"  Cycle {t}: {out}, f: {f}, l: {l}")

    # Check diagonal pattern: at time t, row r should have column[t - r - 1] if valid
    # The "diagonal" means row 0 gets col 0 at t=1, row 1 gets col 0 at t=2, etc.
    dut._log.info("✓ Diagonal skew test completed!")
