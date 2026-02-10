import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Timer
import numpy as np
import random

# Opcodes
OP_MATMUL = 0x2
OP_HALT   = 0x1
CMD_WRITE_MEM = 0x01
CMD_RUN       = 0x03
REG_CMD    = 0x04
REG_ADDR   = 0x08
REG_ARG    = 0x0C
REG_MMVR   = 0x10

TILE_SIZE = 4

async def write_reg(dut, addr, data, width=8):
    for i in range(width // 8):
        dut.host_addr.value = addr + i
        dut.host_wr_data.value = (data >> (i * 8)) & 0xFF
        dut.host_wr_en.value = 1
        await RisingEdge(dut.clk)
        dut.host_wr_en.value = 0

async def load_matmul_instruction(dut, inst_idx, a_base, b_base, c_base, m, k, n):
    base_word_addr = 0x8000 + (inst_idx * 4)
    # Word 3: Opcode, A_Base, B_Base, C_Base
    # C_Base is [215:200] -> Bits [23:8] of Word 3
    word3 = (OP_MATMUL << 60) | (a_base << 40) | (b_base << 24) | (c_base << 8)
    # Word 2: M, K, N
    word2 = (m << 40) | (k << 24) | (n << 8)
    
    await write_reg(dut, REG_CMD, CMD_WRITE_MEM)
    await write_reg(dut, REG_ADDR, base_word_addr + 0, 16); await write_reg(dut, REG_MMVR, 0, 64)
    await write_reg(dut, REG_ADDR, base_word_addr + 1, 16); await write_reg(dut, REG_MMVR, 0, 64)
    await write_reg(dut, REG_ADDR, base_word_addr + 2, 16); await write_reg(dut, REG_MMVR, word2, 64)
    await write_reg(dut, REG_ADDR, base_word_addr + 3, 16); await write_reg(dut, REG_MMVR, word3, 64)

def pack_tile(tile, as_cols=True):
    # Pack 4x4 tile into 4 64-bit words
    # A-matrix: Pack COLUMNS (to feed rows of systolic array) -> Wait, logic check.
    # In 'pe.sv', inputs flow horizontally. Row 0 gets input[0].
    # So we need to feed Row 0 at Cycle 0.
    # If we store vectors in memory: Mem[Addr] = [El3, El2, El1, El0].
    # Row 0 of array gets bits [15:0] (El0).
    # Row 1 gets bits [31:16] (El1).
    # So Mem[Addr] should contain one column of the matrix if we want to feed array rows in parallel?
    # NO. 
    # Array Input Port is [4 x 16-bit].
    # input_data[0] is Row 0 input.
    # input_data[1] is Row 1 input.
    # So Mem[Addr] MUST contain [Row3_Val, Row2_Val, Row1_Val, Row0_Val].
    # Which corresponds to A[0][k], A[1][k], A[2][k], A[3][k].
    # This is a COLUMN of A.
    
    # B-matrix (Weights):
    # Flows Vertically.
    # weight_data[0] is Col 0.
    # Mem[Addr] MUST contain [Col3_Val, Col2_Val, Col1_Val, Col0_Val].
    # This corresponds to B[k][0], B[k][1], B[k][2], B[k][3].
    # This is a ROW of B.
    
    words = []
    if as_cols: # For Matrix A (Inputs)
        for c in range(4): # 4 columns in a tile
            packed = 0
            for r in range(4): # 4 rows packed into one word
                val = int(tile[r, c]) & 0xFFFF
                packed |= (val << (r * 16))
            words.append(packed)
    else: # For Matrix B (Weights) -> as_rows
        for r in range(4): # 4 rows in a tile
            packed = 0
            for c in range(4): # 4 cols packed into one word
                val = int(tile[r, c]) & 0xFFFF
                packed |= (val << (c * 16))
            words.append(packed)
    return words

@cocotb.test()
async def test_stress_matmul(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    dut.host_wr_en.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    # --- 1. Generate Data ---
    M, K, N_DIM = 35, 17, 69
    
    dut._log.info(f"Generating Random Matrices: A({M}x{K}) * B({K}x{N_DIM})")
    # Small values to avoid overflow for now (Accumulator is 64-bit so safe, but easier to debug)
    A_ref = np.random.randint(1, 5, size=(M, K), dtype=np.int32)
    B_ref = np.random.randint(1, 5, size=(K, N_DIM), dtype=np.int32)
    C_ref = A_ref @ B_ref
    
    # Tile Counts
    m_tiles = (M + 3) // 4
    k_tiles = (K + 3) // 4
    n_tiles = (N_DIM + 3) // 4
    
    dut._log.info(f"Tiling: {m_tiles}x{n_tiles} output tiles (K={k_tiles})")
    
    # Pad matrices to multiples of 4
    A_pad = np.pad(A_ref, ((0, m_tiles*4 - M), (0, k_tiles*4 - K)))
    B_pad = np.pad(B_ref, ((0, k_tiles*4 - K), (0, n_tiles*4 - N_DIM)))
    
    # --- 2. Load Data ---
    a_base = 0x0000
    b_base = 0x1000 # 4K words offset
    c_base = 0x0200 # Output base (safe area between A and B)
    
    dut._log.info("Loading Matrix A...")
    # Layout: Row-Major of Tiles.
    # Tile(0,0), Tile(0,1)... Wait.
    # Logic in control_unit: 
    # ub_addr = mm_a_base + (m_idx * mm_k_total * 4) + (k_idx * 4) + cycle_cnt;
    # This implies A is stored: [Tile(0,0)], [Tile(0,1)], ... [Tile(0, K-1)], [Tile(1,0)]...
    # Row-major of tiles. Correct.
    
    current_addr = a_base
    for m_i in range(m_tiles):
        for k_i in range(k_tiles):
            # Extract 4x4 tile
            tile = A_pad[m_i*4 : (m_i+1)*4, k_i*4 : (k_i+1)*4]
            packed_words = pack_tile(tile, as_cols=True) # A needs columns packed
            
            # Write 4 words
            await write_reg(dut, REG_CMD, CMD_WRITE_MEM)
            for i, w in enumerate(packed_words):
                await write_reg(dut, REG_ADDR, current_addr + i, 16)
                await write_reg(dut, REG_MMVR, w, 64)
            current_addr += 4

    dut._log.info("Loading Matrix B...")
    # Logic in control_unit:
    # ub_w_addr = mm_b_base + (k_idx * mm_n_total * 4) + (n_idx * 4) + cycle_cnt;
    # This implies B is stored: [Tile(0,0)], [Tile(0,1)]...
    # Where tile index is (k_idx, n_idx).
    
    current_addr = b_base
    for k_i in range(k_tiles):
        for n_i in range(n_tiles):
            tile = B_pad[k_i*4 : (k_i+1)*4, n_i*4 : (n_i+1)*4]
            packed_words = pack_tile(tile, as_cols=False) # B needs rows packed
            
            await write_reg(dut, REG_CMD, CMD_WRITE_MEM)
            for i, w in enumerate(packed_words):
                await write_reg(dut, REG_ADDR, current_addr + i, 16)
                await write_reg(dut, REG_MMVR, w, 64)
            current_addr += 4
            
    # --- 3. Load Program ---
    dut._log.info("Loading Program...")
    await load_matmul_instruction(dut, 0, a_base, b_base, c_base, m_tiles, k_tiles, n_tiles)
    
    # HALT at PC=1
    await write_reg(dut, REG_CMD, CMD_WRITE_MEM)
    await write_reg(dut, REG_ADDR, 0x8004 + 3, 16)
    await write_reg(dut, REG_MMVR, OP_HALT << 60, 64)
    
    # --- 4. Run ---
    dut._log.info("Starting Execution...")
    await write_reg(dut, REG_CMD, CMD_RUN)
    await write_reg(dut, REG_ARG, 0, 32)
    await write_reg(dut, REG_MMVR, 0, 64)
    
    # --- 5. Wait for Completion ---
    # We expect 162 output tiles. Each generates an 'all_done' pulse from the array.
    # The sequencer manages this. The program only HALTs after all tiles are done.
    # So we should actually wait for HALT status, not just all_done.
    
    dut._log.info(f"Waiting for {m_tiles * n_tiles} tiles to process...")
    
    # Wait for HALT status
    for i in range(50000): # Increased timeout
        await RisingEdge(dut.clk)
        
        # Check Status Reg every 100 cycles to save sim time
        if i % 100 == 0:
            dut.host_addr.value = 0x00 # REG_STATUS
            await Timer(1, units="ps")
            if dut.host_rd_data.value == 0xFF:
                dut._log.info(f"HALT Detected at cycle {i}!")
                break
    else:
        raise AssertionError("Timeout waiting for HALT")
        
    # --- 6. Verify Results ---
    # Since we don't have writeback, we can't read memory.
    # We rely on the verification 'results_flat' port which exposes the accumulator array.
    # BUT: The array only holds the LAST calculated tile (Tile[M-1][N-1]).
    # To verify EVERYTHING, we would need to snoop the bus or add writeback.
    # For now, let's verify the FINAL tile: C_ref[Last 4 rows][Last 4 cols].
    
    # Wait a bit for pipeline drain
    await ClockCycles(dut.clk, 10)
    
    dut._log.info("Verifying Final Tile...")
    flat_val = dut.results_flat.value.integer
    
    # Extract last 4x4 block from C_ref (padded)
    # The TPU calculates padded result.
    last_m_tile = m_tiles - 1
    last_n_tile = n_tiles - 1
    
    expected_tile = np.zeros((4,4), dtype=int)
    # Calculate expected for the last tile including padding
    # Re-calculate C_pad using A_pad and B_pad
    C_pad = A_pad @ B_pad
    expected_tile = C_pad[last_m_tile*4 : (last_m_tile+1)*4, last_n_tile*4 : (last_n_tile+1)*4]
    
    all_good = True
    for r in range(4):
        for c in range(4):
            # Unpack 64-bit values from 1024-bit vector
            idx = r * 4 + c
            val = (flat_val >> (idx * 64)) & ((1 << 64) - 1)
            # Signed conversion
            if val >= (1 << 63): val -= (1 << 64)
            
            exp = expected_tile[r, c]
            if val != exp:
                dut._log.error(f"Mismatch at local tile [{r}][{c}]: Got {val}, Exp {exp}")
                all_good = False
                
    if all_good:
        dut._log.info("✅ STRESS TEST PASSED (Final Tile Verified)")
    else:
        raise AssertionError("Stress Test Failed")
