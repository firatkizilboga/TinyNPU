import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

@cocotb.test()
async def test_data_flow(dut):
    """Simple data flow test: Check if result_valid asserts after computation."""
    
    # Start Clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    dut.compute_enable.value = 0
    dut.drain_enable.value = 0
    dut.acc_clear.value = 0
    dut.precision_mode.value = 2 # MODE_INT16
    
    for i in range(4):
        dut.input_data[i].value = 0
        dut.weight_data[i].value = 0
        
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    dut._log.info("Starting computation flow...")
    
    # Enable compute
    dut.compute_enable.value = 1
    
    # Feed some dummy data for a few cycles
    for i in range(10):
        dut.input_data[0].value = 1
        dut.weight_data[0].value = 1
        await RisingEdge(dut.clk)
        
    dut.input_data[0].value = 0
    dut.weight_data[0].value = 0
    
    # Wait for results
    dut._log.info("Waiting for result_valid...")
    
    valid_seen = False
    for i in range(50):
        if dut.result_valid.value:
            valid_seen = True
            dut._log.info(f"✅ result_valid asserted at cycle {i}!")
            break
        await RisingEdge(dut.clk)
        
    if valid_seen:
        dut._log.info("TEST PASSED: Data flow verified via result_valid signal.")
    else:
        dut._log.error("TEST FAILED: result_valid never went high.")
        
    assert valid_seen, "result_valid signal did not assert"
