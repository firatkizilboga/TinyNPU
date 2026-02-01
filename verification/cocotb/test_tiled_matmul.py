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


async def compute_tile(dut, a_addrs, b_addrs, clear_acc=True):
    """
    Compute one output tile by accumulating K-tiles.
    
    Args:
        dut: Device under test
        a_addrs: List of addresses for A tiles (one per K-tile)
        b_addrs: List of addresses for B tiles (one per K-tile)
        clear_acc: Whether to clear accumulators before computation
    
    Returns:
        4×4 result matrix
    """
    assert len(a_addrs) == len(b_addrs), "Must have same number of A and B tiles"
    n_k_tiles = len(a_addrs)
    
    # Clear accumulators if requested
    if clear_acc:
        dut.acc_clear.value = 1
        await RisingEdge(dut.clk)
        dut.acc_clear.value = 0
    
    # Enable UB and compute
    dut.en.value = 1
    dut.compute_enable.value = 1
    
    # Process each K-tile
    for k_idx, (a_addr, b_addr) in enumerate(zip(a_addrs, b_addrs)):
        # Feed TILE_SIZE cycles for this K-tile
        for cycle in range(TILE_SIZE):
            dut.input_addr.value = a_addr + cycle
            dut.weight_addr.value = b_addr + cycle
            
            # Set first/last markers for this K-tile
            dut.input_first_in.value = 1 if (k_idx == 0 and cycle == 0) else 0
            dut.input_last_in.value = 1 if (k_idx == n_k_tiles-1 and cycle == TILE_SIZE-1) else 0
            dut.weight_first_in.value = 1 if (k_idx == 0 and cycle == 0) else 0
            dut.weight_last_in.value = 1 if (k_idx == n_k_tiles-1 and cycle == TILE_SIZE-1) else 0
            
            await RisingEdge(dut.clk)
    
    # Clear markers
    dut.input_first_in.value = 0
    dut.input_last_in.value = 0
    dut.weight_first_in.value = 0
    dut.weight_last_in.value = 0
    
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
    dut._log.info(f"Tiles: {n_m_tiles} (M) × {n_k_tiles} (K) × {n_n_tiles} (N)")
    dut._log.info(f"Total output tiles: {n_m_tiles} × {n_n_tiles} = {n_m_tiles * n_n_tiles}")
    dut._log.info(f"Passes per output tile: {n_k_tiles}")
    dut._log.info(f"Total passes: {n_m_tiles * n_n_tiles * n_k_tiles}")
    dut._log.info("=" * 80)
    
    # Compute each output tile
    all_passed = True
    result_matrix = [[0] * n for _ in range(m)]
    
    for i in range(n_m_tiles):
        for j in range(n_n_tiles):
            dut._log.info(f"\nComputing output tile C[{i}][{j}]...")
            
            # Gather addresses for all K-tiles
            a_addrs = []
            b_addrs = []
            for k_tile in range(n_k_tiles):
                a_key = f"{i},{k_tile}"
                b_key = f"{k_tile},{j}"
                a_addr = metadata['a_tiles'][a_key]
                b_addr = metadata['b_tiles'][b_key]
                a_addrs.append(a_addr)
                b_addrs.append(b_addr)
                dut._log.info(f"  K-tile {k_tile}: A[{i}][{k_tile}] @ {a_addr}, B[{k_tile}][{j}] @ {b_addr}")
            
            # Compute this output tile
            tile_result = await compute_tile(dut, a_addrs, b_addrs, clear_acc=True)
            
            # Extract the valid portion of the result (may be padded)
            r_start = i * TILE_SIZE
            r_end = min(r_start + TILE_SIZE, m)
            c_start = j * TILE_SIZE
            c_end = min(c_start + TILE_SIZE, n)
            
            # Verify against golden reference
            tile_passed = True
            for r in range(r_start, r_end):
                for c in range(c_start, c_end):
                    tile_r = r - r_start
                    tile_c = c - c_start
                    computed = tile_result[tile_r][tile_c]
                    expected = golden[r][c]
                    result_matrix[r][c] = computed
                    
                    if computed != expected:
                        dut._log.error(
                            f"  MISMATCH at C[{r}][{c}]: "
                            f"got {computed}, expected {expected}"
                        )
                        tile_passed = False
                        all_passed = False
            
            if tile_passed:
                dut._log.info(f"  ✓ Tile C[{i}][{j}] PASSED")
            else:
                dut._log.error(f"  ✗ Tile C[{i}][{j}] FAILED")
    
    # Print final results
    dut._log.info("\n" + "=" * 80)
    dut._log.info("FINAL RESULT MATRIX:")
    for r in range(m):
        row_str = "  [" + ", ".join(f"{result_matrix[r][c]:4}" for c in range(n)) + "]"
        dut._log.info(row_str)
    
    dut._log.info("\nGOLDEN REFERENCE:")
    for r in range(m):
        row_str = "  [" + ", ".join(f"{golden[r][c]:4}" for c in range(n)) + "]"
        dut._log.info(row_str)
    dut._log.info("=" * 80)
    
    if all_passed:
        dut._log.info(f"✓ TILED MATMUL TEST PASSED!")
    else:
        raise AssertionError("TILED MATMUL TEST FAILED")
