import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

MODE_INT16 = 2

@cocotb.test()
async def test_pe_row1_col0_debug(dut):
    """Debug test focusing ONLY on PE[1][0] to see exactly what's happening"""
    
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.rst_n.value = 0
    dut.precision_mode.value = MODE_INT16
    dut.compute_enable.value = 1
    dut.drain_enable.value = 0
    dut.acc_clear.value = 0
    
    for i in range(4):
        dut.input_data[i].value = 0
        dut.weight_data[i].value = 0
    
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    dut._log.info("="*80)
    dut._log.info("DEBUGGING PE[1][0] - Expected: 3×5 + 4×7 = 43")
    dut._log.info("="*80)
    
    # We want PE[1][0] to compute: 3×5 + 4×7 = 43
    # Row 1 needs inputs: [3, 4]
    # Col 0 needs weights: [5, 7]
    
    # Skewed injection for row 1 (delay by 1 cycle):
    schedule = [
        # Cycle, input_data[1], weight_data[0], description
        (0, 0, 0, "Cycle 0: Both skewed, inject bubbles"),
        (1, 3, 5, "Cycle 1: Row 1 starts (3), Col 0 starts (5)"),
        (2, 4, 7, "Cycle 2: Second pair (4,  7)"),
        (3, 0, 0, "Cycle 3: Clear inputs"),
        (4, 0, 0, "Cycle 4: Settling"),
        (5, 0, 0, "Cycle 5: Settling"),
        (6, 0, 0, "Cycle 6: Settling"),
        (7, 0, 0, "Cycle 7: Settling"),
    ]
    
    for cycle_num, inp, wgt, desc in schedule:
        dut._log.info(f"\n{'='*70}")
        dut._log.info(f"CYCLE {cycle_num}: {desc}")
        dut._log.info(f"{'='*70}")
        
        # Set inputs
        dut.input_data[1].value = inp
        dut.weight_data[0].value = wgt
        dut._log.info(f"INJECTING: input_data[1]={inp}, weight_data[0]={wgt}")
        
        # Wait for clock edge
        await RisingEdge(dut.clk)
        
        # Read PE[1][0] state AFTER the clock edge
        pe = dut.gen_rows[1].gen_cols[0].pe_inst
        
        acc = int(pe.accumulator.value.to_signed())
        inp_latch = int(pe.input_latch.value)
        wgt_latch = int(pe.weight_latch.value.signed)
        inp_from_left = int(pe.input_from_left.value)
        data_from_top = int(pe.data_from_top.value.to_signed())
        
        dut._log.info(f"PE[1][0] STATE AFTER CYCLE {cycle_num}:")
        dut._log.info(f"  input_from_left (boundary) = {inp_from_left}")
        dut._log.info(f"  data_from_top (boundary)   = {data_from_top} (lower 16 bits = {data_from_top & 0xFFFF})")
        dut._log.info(f"  input_latch (registered)   = {inp_latch}")
        dut._log.info(f"  weight_latch (registered)  = {wgt_latch}")
        dut._log.info(f"  accumulator                = {acc}")
        
        # Calculate expected accumulation
        if cycle_num >= 2:
            expected_partial = inp_latch * wgt_latch
            dut._log.info(f"  Current MAC would be: {inp_latch} × {wgt_latch} = {expected_partial}")
    
    # Final check
    pe = dut.gen_rows[1].gen_cols[0].pe_inst
    final_acc = int(pe.accumulator.value.to_signed())
    
    dut._log.info(f"\n{'='*70}")
    dut._log.info(f"FINAL RESULT: PE[1][0] accumulator = {final_acc}")
    dut._log.info(f"EXPECTED: 43 (3×5 + 4×7 = 15 + 28)")
    
    if final_acc == 43:
        dut._log.info(f"✅ TEST PASSED!")
    else:
        dut._log.error(f"❌ TEST FAILED! Got {final_acc}, expected 43")
        dut._log.error(f"Difference: {final_acc - 43}")
        
        # Hypothesis testing
        if final_acc == 20:  # 3×5 + 1×5
            dut._log.error("Hypothesis: Got 3×5 + 1×5 = 20 (input 4 didn't arrive, got 1 instead)")
        elif final_acc == 40:  # 3×5 + 5×5
            dut._log.error("Hypothesis: Got 3×5 + 5×5 = 40 (input 4 didn't arrive, got 5 instead)")
        elif final_acc == 30:  # 3×5 + 3×5
            dut._log.error("Hypothesis: Got 3×5 + 3×5 = 30 (weight 7 didn't arrive, still have 5)")
        
    assert final_acc == 43, f"PE[1][0] = {final_acc}, expected 43"
