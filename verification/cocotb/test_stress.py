import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles, Timer
import numpy as np
import stress_driver

REG_STATUS = 0x00
REG_CMD    = 0x04
REG_ADDR   = 0x08
REG_ARG    = 0x0C
REG_MMVR   = 0x10

async def write_reg(dut, addr, data, width=8):
    num_bytes = width // 8
    for i in range(num_bytes):
        dut.host_addr.value = addr + i
        dut.host_wr_data.value = (int(data) >> (i * 8)) & 0xFF
        dut.host_wr_en.value = 1
        await RisingEdge(dut.clk)
    dut.host_wr_en.value = 0

async def read_ub_vector(dut, addr):
    await write_reg(dut, REG_ADDR, addr, 16)
    await write_reg(dut, REG_CMD, 2, 8)
    await write_reg(dut, REG_MMVR, 0, 64)
    for _ in range(500):
        dut.host_addr.value = REG_STATUS
        await RisingEdge(dut.clk)
        if int(dut.host_rd_data.value) == 0x02:
            break
    else:
        raise AssertionError(f"Timeout waiting for read at addr {addr}")
    res_bytes = []
    for i in range(8):
        dut.host_addr.value = REG_MMVR + i
        await RisingEdge(dut.clk)
        res_bytes.append(int(dut.host_rd_data.value))
    res = []
    for i in range(4):
        val = res_bytes[i*2] | (res_bytes[i*2+1] << 8)
        res.append(val)
    return res

@cocotb.test()
async def test_stress(dut):
    """Stress Test: Dynamic Dimensions"""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    M, K, N = stress_driver.M_RAW, getattr(stress_driver, 'K_RAW', '?'), stress_driver.N_RAW
    dut._log.info(f"Stress Test: {M}x{K} x {K}x{N} Matrix Multiplication")

    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    dut._log.info(f"Loading {len(stress_driver.DRIVER_MESSAGES)} MMIO messages...")
    for reg, val in stress_driver.DRIVER_MESSAGES:
        width = 64 if reg == REG_MMVR else (32 if reg == REG_ARG else (16 if reg == REG_ADDR else 8))
        await write_reg(dut, reg, val, width)

    dut._log.info("Execution triggered. Waiting for HALT...")
    for cycle in range(200000):
        dut.host_addr.value = REG_STATUS
        await RisingEdge(dut.clk)
        status = int(dut.host_rd_data.value)
        if status == 0xFF:
            dut._log.info(f"HALT detected at cycle {cycle}")
            break
    else:
        raise AssertionError("Timeout waiting for HALT")

    dut._log.info("Verifying Results...")
    expected_c = np.array(stress_driver.EXPECTED_C)
    m_tiles = stress_driver.M_TILES
    n_tiles = stress_driver.N_TILES
    c_base  = stress_driver.C_BASE
    
    # We will reconstruct the actual matrix from the tiled UB data
    actual_c = np.zeros((m_tiles * 4, n_tiles * 4), dtype=np.uint64)
    
    for m in range(m_tiles):
        for n in range(n_tiles):
            # Each tile (4x4) consists of 4 rows (each row is one 64-bit word)
            tile_addr = c_base + (m * n_tiles * 4) + (n * 4)
            for r in range(4):
                word = await read_ub_vector(dut, tile_addr + r)
                # Word contains 4 elements for Row r of the tile
                actual_c[m*4 + r, n*4 : n*4 + 4] = word
                
    # Slice actual_c to original dimensions
    actual_c_trimmed = actual_c[:stress_driver.M_RAW, :stress_driver.N_RAW]
    
    # Check
    if np.array_equal(actual_c_trimmed, expected_c):
        dut._log.info(f"✅ STRESS TEST PASSED: {M}x{K} x {K}x{N} Verified!")
    else:
        # Find first mismatch
        mismatches = np.where(actual_c_trimmed != expected_c)
        r, c = mismatches[0][0], mismatches[1][0]
        dut._log.error(f"Mismatch at [{r}][{c}]: Got {actual_c_trimmed[r,c]}, Expected {expected_c[r,c]}")
        assert False, "Stress test result mismatch"