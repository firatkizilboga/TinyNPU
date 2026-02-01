import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

DATA_WIDTH = 16
ARRAY_SIZE = 4


def get_data_out(dut, row):
    """Get data_out for a row from flattened output."""
    flat = dut.data_out_flat.value.integer
    return (flat >> (row * DATA_WIDTH)) & ((1 << DATA_WIDTH) - 1)


def set_data_in(dut, row, value):
    """Set data_in signal for a row."""
    try:
        dut.data_in[row].value = value
    except (IndexError, AttributeError):
        for pattern in [f"data_in__{row}", f"data_in_{row}"]:
            sig = getattr(dut, pattern, None)
            if sig:
                sig.value = value
                return


@cocotb.test()
async def test_first_last_global_markers(dut):
    """
    Test first_out and last_out as global markers.
    
    Sequence (4 rows of data, like a 4x4 matrix):
    - Row 0: first_in=1, contains value 0x0C (12)
    - Row 1: nothing special
    - Row 2: nothing special  
    - Row 3: last_in=1, contains value 0x19 (25)
    
    Expected:
    - first_out pulses once when row 0 data exits skewer (value 12)
    - last_out pulses once when row 3 data exits skewer (value 25)
    """
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    dut.en.value = 0
    dut.first_in.value = 0
    dut.last_in.value = 0
    for i in range(ARRAY_SIZE):
        set_data_in(dut, i, 0)
    
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    dut.en.value = 1
    await RisingEdge(dut.clk)
    
    dut._log.info("Feeding 4 rows of data...")
    
    first_events = []
    last_events = []
    cycle = 0
    
    # Row 0: first_in=1, data[0]=12 (0x0C)
    row0_data = [12, 100, 101, 102]  # 12 is the "first" value
    dut._log.info(f"Row 0: data[0]={row0_data[0]}, first_in=1")
    for i in range(ARRAY_SIZE):
        set_data_in(dut, i, row0_data[i])
    dut.first_in.value = 1
    dut.last_in.value = 0
    await RisingEdge(dut.clk)
    cycle += 1
    
    # Row 1: nothing special
    row1_data = [200, 201, 202, 203]
    for i in range(ARRAY_SIZE):
        set_data_in(dut, i, row1_data[i])
    dut.first_in.value = 0
    dut.last_in.value = 0
    await RisingEdge(dut.clk)
    cycle += 1
    
    # Check for first_out (should appear now - row 0 has 1 cycle delay)
    if dut.first_out.value.integer == 1:
        data = get_data_out(dut, 0)
        first_events.append((cycle, data))
        dut._log.info(f"Cycle {cycle}: first_out=1, data[0]={data}")
    
    # Row 2: nothing special
    row2_data = [300, 301, 302, 303]
    for i in range(ARRAY_SIZE):
        set_data_in(dut, i, row2_data[i])
    await RisingEdge(dut.clk)
    cycle += 1
    
    if dut.first_out.value.integer == 1:
        data = get_data_out(dut, 0)
        first_events.append((cycle, data))
    if dut.last_out.value.integer == 1:
        data = get_data_out(dut, 3)
        last_events.append((cycle, data))
    
    # Row 3: last_in=1, data[3]=25 (0x19)
    row3_data = [400, 401, 402, 25]  # 25 is the "last" value
    dut._log.info(f"Row 3: data[3]={row3_data[3]}, last_in=1")
    for i in range(ARRAY_SIZE):
        set_data_in(dut, i, row3_data[i])
    dut.first_in.value = 0
    dut.last_in.value = 1
    await RisingEdge(dut.clk)
    cycle += 1
    
    if dut.first_out.value.integer == 1:
        first_events.append((cycle, get_data_out(dut, 0)))
    if dut.last_out.value.integer == 1:
        last_events.append((cycle, get_data_out(dut, 3)))
    
    # Clear and observe remaining outputs
    dut.first_in.value = 0
    dut.last_in.value = 0
    for i in range(ARRAY_SIZE):
        set_data_in(dut, i, 0)
    
    for _ in range(8):
        await RisingEdge(dut.clk)
        cycle += 1
        
        if dut.first_out.value.integer == 1:
            first_events.append((cycle, get_data_out(dut, 0)))
            dut._log.info(f"Cycle {cycle}: first_out=1, data[0]={get_data_out(dut, 0)}")
        if dut.last_out.value.integer == 1:
            last_events.append((cycle, get_data_out(dut, 3)))
            dut._log.info(f"Cycle {cycle}: last_out=1, data[3]={get_data_out(dut, 3)}")
    
    # Verify
    dut._log.info("\n--- Verification ---")
    
    assert len(first_events) == 1, f"Expected 1 first_out, got {len(first_events)}"
    assert first_events[0][1] == 12, f"first_out data should be 12, got {first_events[0][1]}"
    dut._log.info(f"✓ first_out: cycle {first_events[0][0]}, data={first_events[0][1]} (expected 12)")
    
    assert len(last_events) == 1, f"Expected 1 last_out, got {len(last_events)}"
    assert last_events[0][1] == 25, f"last_out data should be 25, got {last_events[0][1]}"
    dut._log.info(f"✓ last_out: cycle {last_events[0][0]}, data={last_events[0][1]} (expected 25)")
    
    dut._log.info("\n✅ Global first/last markers work correctly!")
    dut._log.info("   Only 12 (first) and 25 (last) are marked in output!")
