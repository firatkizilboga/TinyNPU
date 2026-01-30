import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

MODE_INT16 = 2

@cocotb.test()
async def test_2x2_matmul_fixed(dut):
    """Test 2x2 matrix multiplication after fix"""
    
    clock = Clock(dut.clk, 10, unit="ns")
    cocotb.start_soon(clock.start())
    
    A = [[1, 2],
         [3, 4]]
    
    B = [[5, 6],
         [7, 8]]
    
    C_expected = [[19, 22],
                  [43, 50]]
    
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
    
    dut._log.info("Testing 2x2 matrix multiplication with skewed injection (FIXED)")
    
    K = 2
    N = 2
    total_cycles = 2 * (N - 1) + K + 3  # Increased for row 1
    
    for cycle in range(total_cycles):
        # Inject with skewing
        for r in range(N):
            k = cycle - r
            if 0 <= k < K:
                dut.input_data[r].value = A[r][k]
            else:
                dut.input_data[r].value = 0
        
        for c in range(N):
            k = cycle - c
            if 0 <= k < K:
                dut.weight_data[c].value = B[k][c]
            else:
                dut.weight_data[c].value = 0
        
        await RisingEdge(dut.clk)
    
    # Clear and settle - need more time for row 1
    for i in range(4):
        dut.input_data[i].value = 0
        dut.weight_data[i].value = 0
    
    for _ in range(10):  # More settling
        await RisingEdge(dut.clk)
    
    # Check results
    dut._log.info("\nResults:")
    all_correct = True
    C_actual = []
    
    for r in range(N):
        row = []
        for c in range(N):
            pe = dut.gen_rows[r].gen_cols[c].pe_inst
            acc = int(pe.accumulator.value.to_signed())
            row.append(acc)
            if acc != C_expected[r][c]:
                all_correct = False
                dut._log.error(f"  PE[{r}][{c}]: got {acc}, expected {C_expected[r][c]}")
            else:
                dut._log.info(f"  PE[{r}][{c}]: {acc} ✓")
        C_actual.append(row)
    
    assert all_correct, f"Got {C_actual}, expected {C_expected}"
    
    dut._log.info("\n✓ 2x2 MATRIX MULTIPLICATION WORKS!")
