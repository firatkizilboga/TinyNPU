"""
Test for UBSS with K=12 inner dimension.
Matrix multiplication: A(4x12) x B(12x4) = C(4x4)

Uses pre-initialized data from buffer_init_k12.hex:
- Row 0: DEAD sentinel
- Rows 1-12: Matrix A columns (incrementing pattern)
- Rows 16-27: Matrix B rows (all ones)

Expected result: Each C[i][j] = sum of row i of A = 78, 222, 366, 510
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

# Parameters
N = 4  # Systolic array size
K = 12  # Inner dimension
DATA_WIDTH = 16
ACC_WIDTH = 64


def unpack_results(flat_val):
    """Unpack N*N results array (64-bit accumulators)."""
    results = []
    for row in range(N):
        row_vals = []
        for col in range(N):
            idx = row * N + col
            val = (flat_val >> (idx * ACC_WIDTH)) & ((1 << ACC_WIDTH) - 1)
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
async def test_ubss_k12(dut):
    """Test UB → Skewer → Systolic matmul with K=12."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Memory layout from gen_test_data.py:
    # A columns at rows 1-12 (K=12 columns)
    # B rows at rows 16-27 (K=12 rows)
    A_START = 1
    B_START = 16
    
    # Expected result: C = A × B (computed by gen_test_data.py)
    # Using random matrices, result is:
    expected = [
        [239, 189, 218, 326],
        [154, 165, 139, 225],
        [268, 331, 305, 403],
        [232, 327, 293, 409],
    ]
    
    # Clear accumulators
    dut.acc_clear.value = 1
    await RisingEdge(dut.clk)
    dut.acc_clear.value = 0
    
    # Enable UB and compute
    dut.en.value = 1
    dut.compute_enable.value = 1
    
    # Feed K=12 cycles of data
    for k in range(K):
        dut.input_addr.value = A_START + k
        dut.weight_addr.value = B_START + k
        
        # Set first/last markers
        dut.input_first_in.value = 1 if k == 0 else 0
        dut.input_last_in.value = 1 if k == K-1 else 0
        dut.weight_first_in.value = 1 if k == 0 else 0
        dut.weight_last_in.value = 1 if k == K-1 else 0
        
        await RisingEdge(dut.clk)
    
    # Clear markers
    dut.input_first_in.value = 0
    dut.input_last_in.value = 0
    dut.weight_first_in.value = 0
    dut.weight_last_in.value = 0
    
    # Wait for data to propagate through skewers and systolic array
    # Skewer delay: up to N-1 cycles
    # Systolic propagation: up to 2*(N-1) cycles
    drain_time = 3 * N
    await ClockCycles(dut.clk, drain_time)
    
    # Disable compute
    dut.compute_enable.value = 0
    await ClockCycles(dut.clk, 2)
    
    # Read results
    results = unpack_results(dut.results_flat.value.integer)
    
    # Print results
    dut._log.info("=" * 70)
    dut._log.info(f"SYSTOLIC ARRAY RESULTS (K={K}):")
    for i, row in enumerate(results):
        dut._log.info(f"  Row {i}: [{row[0]:4}, {row[1]:4}, {row[2]:4}, {row[3]:4}]")
    dut._log.info("=" * 70)
    dut._log.info("EXPECTED RESULTS:")
    for i, row in enumerate(expected):
        dut._log.info(f"  Row {i}: [{row[0]:4}, {row[1]:4}, {row[2]:4}, {row[3]:4}]")
    
    # Verify
    passed = True
    for i in range(N):
        for j in range(N):
            if results[i][j] != expected[i][j]:
                dut._log.error(f"Mismatch at [{i}][{j}]: got {results[i][j]}, expected {expected[i][j]}")
                passed = False
    
    if passed:
        dut._log.info(f"✓ K={K} test PASSED!")
    else:
        raise AssertionError(f"K={K} test FAILED")
