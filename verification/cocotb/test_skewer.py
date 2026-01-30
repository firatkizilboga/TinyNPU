import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

MODE_INT16 = 2

@cocotb.test()
async def test_skewer_with_array(dut):
    """Test skewer module feeding systolic array"""
    
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    
    # Test matrices: 2x2 matmul
    A = [[1, 2],
         [3, 4]]
    
    B = [[5, 6],
         [7, 8]]
    
    # Expected: C = [[19, 22], [43, 50]]
    C_expected = [[19, 22],
                  [43, 50]]
    
    # Reset
    dut.rst_n.value = 0
    dut.start.value = 0
    
    # Load matrices into skewer
    for r in range(2):
        for k in range(2):
            dut.A[r][k].value = A[r][k]
    
    for k in range(2):
        for c in range(2):
            dut.B[k][c].value = B[k][c]
    
    # Pad unused positions with zeros
    for r in range(2, 4):
        for k in range(4):
            dut.A[r][k].value = 0
    
    for k in range(2, 4):
        for c in range(4):
            dut.B[k][c].value = 0
    
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    dut._log.info("Starting skewed injection test")
    
    # Start skewer
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for skewer to complete
    while int(dut.busy.value) == 1:
        await RisingEdge(dut.clk)
    
    # Wait for done signal
    await RisingEdge(dut.clk)
    assert int(dut.done.value) == 1, "Done signal should be high"
    
    dut._log.info("Skewer completed, waiting for results to settle...")
    
    # Wait additional cycles for PE pipeline (2 cycles latency) + settling
    for _ in range(5):
        await RisingEdge(dut.clk)
    
    # Read results
    dut._log.info("\nResults:")
    C_actual = []
    all_correct = True
    
    for r in range(2):
        row = []
        for c in range(2):
            pe = dut.array_inst.gen_rows[r].gen_cols[c].pe_inst
            acc = int(pe.accumulator.value.to_signed())
            row.append(acc)
            
            if acc != C_expected[r][c]:
                all_correct = False
                dut._log.error(f"  PE[{r}][{c}]: got {acc}, expected {C_expected[r][c]}")
            else:
                dut._log.info(f"  PE[{r}][{c}]: {acc} ✓")
        C_actual.append(row)
    
    dut._log.info(f"\nActual:   {C_actual}")
    dut._log.info(f"Expected: {C_expected}")
    
    assert all_correct, f"Matrix multiplication failed! Got {C_actual}, expected {C_expected}"
    
    dut._log.info("\n✓ SKEWER + ARRAY TEST PASSED!")
