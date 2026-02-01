import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

MODE_INT16 = 2
ACC_WIDTH = 64
ARRAY_SIZE = 4

@cocotb.test()
async def test_numeric_matmul(dut):
    """Test 4x4 matrix multiplication numerically using flattened output port."""
    
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Define 4x4 test matrices
    A = [[1,  2,  3,  4],
         [5,  6,  7,  8],
         [9,  10, 11, 12],
         [13, 14, 15, 16]]
    
    B = [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]]
    
    # Expected result: C = A × I = A (identity matrix multiplication)
    C_expected = [[1,  2,  3,  4],
                  [5,  6,  7,  8],
                  [9,  10, 11, 12],
                  [13, 14, 15, 16]]
    
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
    
    dut._log.info("Injecting skewed data...")
    
    K = 4  # Dot product length
    N = 4  # Matrix size
    
    # Total cycles: skew delay (N-1) + K products + pipeline settling
    total_cycles = (N - 1) + K + 5
    
    for cycle in range(total_cycles):
        # Inject inputs (A matrix) with row skewing
        for r in range(N):
            k = cycle - r  # Row r starts at cycle r
            if 0 <= k < K:
                dut.input_data[r].value = A[r][k]
            else:
                dut.input_data[r].value = 0
        
        # Inject weights (B matrix) with column skewing
        for c in range(N):
            k = cycle - c  # Column c starts at cycle c
            if 0 <= k < K:
                dut.weight_data[c].value = B[k][c]
            else:
                dut.weight_data[c].value = 0
        
        await RisingEdge(dut.clk)
    
    # Clear inputs and settle
    for i in range(N):
        dut.input_data[i].value = 0
        dut.weight_data[i].value = 0
    
    # Wait for computation to finish and result_valid to assert
    for _ in range(50):
        await RisingEdge(dut.clk)
        
    dut._log.info("Checking results from flattened port...")
    
    # Read the flattened result vector
    # results_flat is a huge integer
    # Width = 4 * 4 * 64 = 1024 bits
    flat_val = dut.results_flat.value
    
    # Convert to big integer
    try:
        flat_int = flat_val.integer
    except ValueError:
        dut._log.error(f"Results Flat is 'X' or 'Z': {flat_val}")
        assert False, "Simulation outputs are undefined!"

    C_actual = []
    all_correct = True
    
    # Unpack manually
    # Layout: Row0_Col0, Row0_Col1, ..., Row1_Col0, ...
    # LSB is Row0_Col0
    mask = (1 << ACC_WIDTH) - 1
    
    for r in range(N):
        row = []
        for c in range(N):
            # Calculate bit offset
            idx = r * N + c
            offset = idx * ACC_WIDTH
            
            # Extract ACC_WIDTH bits
            val_unsigned = (flat_int >> offset) & mask
            
            # Convert to signed 64-bit
            if val_unsigned & (1 << (ACC_WIDTH - 1)):
                val_signed = val_unsigned - (1 << ACC_WIDTH)
            else:
                val_signed = val_unsigned
                
            row.append(val_signed)
            
            if val_signed != C_expected[r][c]:
                all_correct = False
                dut._log.error(f"Mismatch at [{r}][{c}]: Expected {C_expected[r][c]}, Got {val_signed}")
        
        C_actual.append(row)
        dut._log.info(f"Row {r}: {row}")

    assert all_correct, f"Matrix mismatch!\nExpected:\n{C_expected}\nGot:\n{C_actual}"
    dut._log.info("✅ NUMERICAL TEST PASSED!")
