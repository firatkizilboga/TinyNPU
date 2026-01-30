import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

MODE_INT8 = 1

@cocotb.test()
async def test_pe(dut):
    """Test PE - simple version that works"""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset with compute_enable already HIGH
    dut.rst_n.value = 0
    dut.precision_mode.value = MODE_INT8
    dut.compute_enable.value = 1  # Enable BEFORE releasing reset
    dut.drain_enable.value = 0
    dut.acc_clear.value = 0
    dut.input_from_left.value = 0
    dut.data_from_top.value = 0
    
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1  # Release reset
    await RisingEdge(dut.clk)
    
    # Set data
    dut.input_from_left.value = 0x0302
    dut.data_from_top.value = 0x0405
    
    # Wait for result (we know it appears in cycle 3 from diagnostic)
    await RisingEdge(dut.clk)  # Cycle 1: latch
    await RisingEdge(dut.clk)  # Cycle 2: compute  
    await RisingEdge(dut.clk)  # Cycle 3: result ready
    
    result = int(dut.acc_out.value.signed_integer)
    dut._log.info(f"Result: {result}")
    assert result == 22, f"Expected 22, got {result}"
    
    dut._log.info("✓ TEST PASSED!")


@cocotb.test()
async def test_pe_drain(dut):
    """Test PE drain mode - accumulator propagation"""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    dut.precision_mode.value = MODE_INT8
    dut.compute_enable.value = 1
    dut.drain_enable.value = 0
    dut.acc_clear.value = 0
    dut.input_from_left.value = 0
    dut.data_from_top.value = 0
    
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    # Compute a value first
    dut.input_from_left.value = 0x0505
    dut.data_from_top.value = 0x0202
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    
    result = int(dut.acc_out.value.signed_integer)
    dut._log.info(f"Computed: {result}")
    assert result == 20, f"Expected 20 (5*2 + 5*2), got {result}"
    
    # Test drain - load value from top
    dut.compute_enable.value = 0
    dut.drain_enable.value = 1
    test_value = 999999
    dut.data_from_top.value = test_value
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)  # Extra cycle for stability
    
    result = int(dut.acc_out.value.signed_integer)
    dut._log.info(f"Drained: {result}")
    assert result == test_value, f"Expected {test_value}, got {result}"
    
    # Check propagation to bottom
    output = int(dut.data_to_bottom.value.signed_integer)
    assert output == test_value, f"Output propagation failed"
    
    dut._log.info("✓ DRAIN TEST PASSED!")
