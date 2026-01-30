import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

MODE_INT16 = 2

@cocotb.test()
async def test_2x2_matmul_corrected(dut):
    """Test 2x2 matrix multiplication with PROPER data injection
    
    KEY FIX: Only inject to the LEFTMOST column boundary (input_data).
    Data propagates HORIZONTALLY through the array via PE interconnections.
    Do NOT continuously re-inject into the same row - that overwrites propagating data!
    """
    
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
    
    dut._log.info("Testing 2x2 matrix multiplication (CORRECTED INJECTION)")
    
    K = 2
    N = 2
    
    # Total cycles needed:
    # - Skew delay: N-1 cycles to start all rows
    # - K cycles to inject all K elements
    # - Pipeline delay: ~2-3 cycles for data to settle
    total_cycles = (N - 1) + K + 4
    
    dut._log.info(f"Injecting data for {total_cycles} cycles...")
    
    for cycle in range(total_cycles):
        dut._log.info(f"\n=== Cycle {cycle} ===")
        
        # Inject INPUTS (A matrix) with skewing - ONLY at left boundary
        for r in range(N):
            k = cycle - r  # Row r starts at cycle r (skewed)
            if 0 <= k < K:
                value = A[r][k]
                dut.input_data[r].value = value
                dut._log.info(f"  input_data[{r}] = A[{r}][{k}] = {value}")
            else:
                dut.input_data[r].value = 0
                dut._log.info(f"  input_data[{r}] = 0 (bubble)")
        
        # Inject WEIGHTS (B matrix) with skewing - ONLY at top boundary
        for c in range(N):
            k = cycle - c  # Column c starts at cycle c (skewed)
            if 0 <= k < K:
                value = B[k][c]
                dut.weight_data[c].value = value
                dut._log.info(f"  weight_data[{c}] = B[{k}][{c}] = {value}")
            else:
                dut.weight_data[c].value = 0
                dut._log.info(f"  weight_data[{c}] = 0 (bubble)")
        
        await RisingEdge(dut.clk)
        
        # Monitor PE states every cycle
        if cycle >= 2:  # After initial pipeline fill
            for r in range(N):
                for c in range(N):
                    pe = dut.gen_rows[r].gen_cols[c].pe_inst
                    acc = int(pe.accumulator.value.to_signed())
                    inp = int(pe.input_latch.value)
                    wgt = int(pe.weight_latch.value.signed)
                    dut._log.info(f"  PE[{r}][{c}]: acc={acc}, inp_latch={inp}, wgt_latch={wgt}")
    
    # Clear inputs and allow settling
    dut._log.info("\nClearing inputs and settling...")
    for i in range(N):
        dut.input_data[i].value = 0
        dut.weight_data[i].value = 0
    
    for _ in range(5):
        await RisingEdge(dut.clk)
    
    # Check final results
    dut._log.info("\n=== FINAL RESULTS ===")
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
                dut._log.error(f"  PE[{r}][{c}]: got {acc}, expected {C_expected[r][c]} ✗")
            else:
                dut._log.info(f"  PE[{r}][{c}]: {acc} ✓")
        C_actual.append(row)
    
    dut._log.info(f"\nExpected:\n{C_expected}")
    dut._log.info(f"Got:\n{C_actual}")
    
    assert all_correct, f"Matrix multiplication failed!\nGot:\n{C_actual}\nExpected:\n{C_expected}"
    
    dut._log.info("\n✅ 2x2 MATRIX MULTIPLICATION WORKS CORRECTLY!")
