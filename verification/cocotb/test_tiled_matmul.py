"""
Test tiled matrix multiplication: A(13×17) × B(17×24) = C(13×24)

This test demonstrates how to perform large matrix multiplication on a 4×4 systolic array
by breaking the computation into tiles and accumulating partial results.

Tiling strategy:
- A (13×17) → 4 row tiles × 5 K-tiles
- B (17×24) → 5 K-tiles × 6 column tiles
- Total: 4×6 = 24 output tiles, each requiring 5 K-tile passes
"""

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import json

# Parameters
N = 4  # Systolic array size
TILE_SIZE = 4
DATA_WIDTH = 16
ACC_WIDTH = 64


def unpack_results(flat_val):
    """Unpack N×N results array (64-bit accumulators)."""
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
    
    # CU Port (Idle)
    dut.cu_req.value = 0
    dut.cu_wr_en.value = 0
    dut.cu_addr.value = 0
    dut.cu_wdata.value = 0
    
    # SA Port (Idle)
    dut.sa_input_addr.value = 0
    dut.sa_input_first.value = 0
    dut.sa_input_last.value = 0
    dut.sa_weight_addr.value = 0
    dut.sa_weight_first.value = 0
    dut.sa_weight_last.value = 0
    
    dut.precision_mode.value = 1  # INT16 mode
    dut.compute_enable.value = 0
    dut.drain_enable.value = 0
    dut.acc_clear.value = 0
    
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def compute_tile(dut, a_addrs, b_addrs, clear_acc=True):
    """
    Compute one output tile by accumulating K-tiles.
    """
    assert len(a_addrs) == len(b_addrs), "Must have same number of A and B tiles"
    n_k_tiles = len(a_addrs)
    
    # Clear accumulators if requested
    if clear_acc:
        dut.acc_clear.value = 1
        await RisingEdge(dut.clk)
        dut.acc_clear.value = 0
    
    # Enable compute
    dut.en.value = 1
    dut.compute_enable.value = 1
    
    # Process each K-tile
    for k_idx, (a_addr, b_addr) in enumerate(zip(a_addrs, b_addrs)):
        for cycle in range(TILE_SIZE):
            dut.sa_input_addr.value = a_addr + cycle
            dut.sa_weight_addr.value = b_addr + cycle
            
            # Set first/last markers for this K-tile
            dut.sa_input_first.value = 1 if (k_idx == 0 and cycle == 0) else 0
            dut.sa_input_last.value  = 1 if (k_idx == n_k_tiles-1 and cycle == TILE_SIZE-1) else 0
            dut.sa_weight_first.value = 1 if (k_idx == 0 and cycle == 0) else 0
            dut.sa_weight_last.value  = 1 if (k_idx == n_k_tiles-1 and cycle == TILE_SIZE-1) else 0
            
            await RisingEdge(dut.clk)
    
    # Clear markers
    dut.sa_input_first.value = 0
    dut.sa_input_last.value = 0
    dut.sa_weight_first.value = 0
    dut.sa_weight_last.value = 0
    
    # Wait for computation to complete
    drain_time = 3 * N
    await ClockCycles(dut.clk, drain_time)
    
    # Disable compute
    dut.compute_enable.value = 0
    await ClockCycles(dut.clk, 2)
    
    # Read results
    results = unpack_results(dut.results_flat.value.integer)
    return results


@cocotb.test()
async def test_tiled_matmul(dut):
    """Test tiled matrix multiplication: 13×17 × 17×24."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    await reset_dut(dut)
    
    # Load metadata
    with open('buffer_init_tiled_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    m = metadata['m']
    k = metadata['k']
    n = metadata['n']
    n_m_tiles = metadata['n_m_tiles']
    n_k_tiles = metadata['n_k_tiles']
    n_n_tiles = metadata['n_n_tiles']
    golden = metadata['golden']
    
    dut._log.info("=" * 80)
    dut._log.info(f"TILED MATRIX MULTIPLICATION: {m}×{k} × {k}×{n} = {m}×{n}")
    dut._log.info("=" * 80)
    
    # Compute each output tile
    all_passed = True
    result_matrix = [[0] * n for _ in range(m)]
    
    for i in range(n_m_tiles):
        for j in range(n_n_tiles):
            dut._log.info(f"\nComputing output tile C[{i}][{j}]...")
            
            a_addrs = []
            b_addrs = []
            for k_tile in range(n_k_tiles):
                a_key = f"{i},{k_tile}"
                b_key = f"{k_tile},{j}"
                a_addr = metadata['a_tiles'][a_key]
                b_addr = metadata['b_tiles'][b_key]
                a_addrs.append(a_addr)
                b_addrs.append(b_addr)
            
            tile_result = await compute_tile(dut, a_addrs, b_addrs, clear_acc=True)
            
            r_start = i * TILE_SIZE
            r_end = min(r_start + TILE_SIZE, m)
            c_start = j * TILE_SIZE
            c_end = min(c_start + TILE_SIZE, n)
            
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    computed = tile_result[r - r_start][c - c_start]
                    expected = golden[r][c]
                    result_matrix[r][c] = computed
                    if computed != expected:
                        dut._log.error(f"MISMATCH at C[{r}][{c}]: got {computed}, exp {expected}")
                        all_passed = False
            
    if all_passed:
        dut._log.info(f"✓ TILED MATMUL TEST PASSED!")
    else:
        raise AssertionError("TILED MATMUL TEST FAILED")