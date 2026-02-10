import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Timer
import numpy as np

# Constants
N = 4
DATA_WIDTH = 16
ACC_WIDTH = 64

def pack_vector(vals):
    res = 0
    for i, v in enumerate(vals):
        res |= (int(v) & 0xFFFF) << (i * 16)
    return res

def unpack_vector(val):
    return [(val >> (i * 16)) & 0xFFFF for i in range(N)]

@cocotb.test()
async def test_ubss_isolation(dut):
    """Isolated test for UBSS: Loading -> Compute -> Drain -> PPU Capture"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    # 1. Reset
    dut.rst_n.value = 0
    dut.en.value = 1
    dut.cu_req.value = 0
    dut.cu_wr_en.value = 0
    dut.compute_enable.value = 0
    dut.drain_enable.value = 0
    dut.acc_clear.value = 0
    dut.precision_mode.value = 2 # MODE_INT16
    dut.ppu_wb_en.value = 0
    dut.ppu_capture_en.value = 0
    dut.ppu_cycle_idx.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # 2. Load Data into UB via cu interface
    # Matrix A (4x4): Identity * 2
    # Matrix B (4x4): Sequence 1..16
    # We want to check if A * B = 2 * B
    mat_a = np.eye(4, dtype=np.uint16) * 2
    mat_b = np.arange(1, 17, dtype=np.uint16).reshape(4,4)
    
    # IMPORTANT: 
    # Based on hardware analysis:
    # A (Left) needs to be stored COLUMN-MAJOR in UB.
    # B (Top) needs to be stored ROW-MAJOR in UB.
    
    dut._log.info("Loading Matrix A (Column-Major)...")
    for i in range(4):
        col = mat_a[:, i]
        word = pack_vector(col)
        dut.cu_req.value = 1
        dut.cu_wr_en.value = 1
        dut.cu_addr.value = 0x00 + i
        dut.cu_wdata.value = word
        await RisingEdge(dut.clk)
    
    dut._log.info("Loading Matrix B (Row-Major)...")
    for i in range(4):
        row = mat_b[i, :]
        word = pack_vector(row)
        dut.cu_req.value = 1
        dut.cu_wr_en.value = 1
        dut.cu_addr.value = 0x10 + i
        dut.cu_wdata.value = word
        await RisingEdge(dut.clk)
        
    dut.cu_req.value = 0
    dut.cu_wr_en.value = 0
    await RisingEdge(dut.clk)

    # 3. Start Compute
    dut._log.info("Starting Computation...")
    dut.acc_clear.value = 1
    await RisingEdge(dut.clk)
    dut.acc_clear.value = 0
    
    dut.compute_enable.value = 1
    for i in range(4):
        dut.sa_input_addr.value = 0x00 + i
        dut.sa_weight_addr.value = 0x10 + i
        dut.sa_input_first.value = 1 if i == 0 else 0
        dut.sa_weight_first.value = 1 if i == 0 else 0
        dut.sa_input_last.value = 1 if i == 3 else 0
        dut.sa_weight_last.value = 1 if i == 3 else 0
        await RisingEdge(dut.clk)
    
    dut.compute_enable.value = 1 # Keep enabled to flush skewers
    dut.sa_input_first.value = 0
    dut.sa_weight_first.value = 0
    dut.sa_input_last.value = 0
    dut.sa_weight_last.value = 0
    
    # Wait for all_done
    for _ in range(100):
        await RisingEdge(dut.clk)
        if int(dut.all_done.value) == 1:
            dut._log.info("Computation Complete (all_done detected)")
            break
    else:
        raise AssertionError("Timeout waiting for all_done")

    # 4. Drain into PPU
    dut._log.info("Draining Results into PPU...")
    dut.compute_enable.value = 0
    dut.drain_enable.value = 1
    dut.ppu_capture_en.value = 1
    
    for i in range(4):
        dut.ppu_cycle_idx.value = i
        await RisingEdge(dut.clk)
    
    dut.drain_enable.value = 0
    dut.ppu_capture_en.value = 0
    await RisingEdge(dut.clk)

    # 5. Verify PPU results by reading back via MMIO mux
    # We set ppu_wb_en = 1 to select PPU data on ub_final_wdata
    # Actually, we can just look at dut.ppu_wdata directly or use the mux
    dut._log.info("Verifying Captured Results in PPU Storage...")
    
    # We want to check C = 2 * B
    expected_c = mat_b * 2
    
    for i in range(4):
        # We need to be careful: which row did PPU capture at which index?
        # In ppu.sv: storage[cycle_idx] <= acc_in
        # In ubss.sv: acc_in is sa_results[3] (bottom row)
        # When draining:
        # Cycle 0: Row 3 arrives at bottom. -> storage[0] = Row 3
        # Cycle 1: Row 2 arrives at bottom. -> storage[1] = Row 2
        # Cycle 2: Row 1 arrives at bottom. -> storage[2] = Row 1
        # Cycle 3: Row 0 arrives at bottom. -> storage[3] = Row 0
        
        dut.ppu_cycle_idx.value = i
        await Timer(1, units="ns") # Combinational stable
        
        captured_word = int(dut.ppu_wdata.value)
        actual_row = unpack_vector(captured_word)
        
        row_idx = 3 - i
        expected_row = expected_c[row_idx, :].tolist()
        
        dut._log.info(f"PPU Storage[{i}] (Matrix Row {row_idx}): Got {actual_row}, Expected {expected_row}")
        assert actual_row == expected_row, f"Result mismatch at row {row_idx}"

    dut._log.info("✅ UBSS ISOLATION TEST PASSED!")
