import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
import chain_driver

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
    for _ in range(100):
        dut.host_addr.value = REG_STATUS
        await RisingEdge(dut.clk)
        if int(dut.host_rd_data.value) == 0x02: break
    res_bytes = []
    for i in range(8):
        dut.host_addr.value = REG_MMVR + i
        await RisingEdge(dut.clk)
        res_bytes.append(int(dut.host_rd_data.value))
    return [res_bytes[i*2] | (res_bytes[i*2+1] << 8) for i in range(4)]

@cocotb.test()
async def test_chain(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    
    dut._log.info("Loading Chain Test Program...")
    for reg, val in chain_driver.DRIVER_MESSAGES:
        width = 64 if reg == REG_MMVR else (32 if reg == REG_ARG else (16 if reg == REG_ADDR else 8))
        await write_reg(dut, reg, val, width)

    dut._log.info("Waiting for HALT...")
    # Timeout increased for 4 sequential ops
    for _ in range(50000):
        dut.host_addr.value = REG_STATUS
        await RisingEdge(dut.clk)
        if int(dut.host_rd_data.value) == 0xFF: break
    else: raise AssertionError("Timeout")

    # Verify Final Result
    dim = chain_driver.DIM
    dut._log.info(f"Verifying Final A_out ({dim}x{dim})...")
    
    actual_a = np.zeros((dim, dim), dtype=np.uint64)
    tiles_m = dim // 4
    tiles_n = dim // 4
    base_addr = chain_driver.ADDR_A_OUT
    
    for m in range(tiles_m):
        for n in range(tiles_n):
            tile_idx = (m * tiles_n) + n
            tile_addr = base_addr + (tile_idx * 4)
            for r in range(4):
                row_vals = await read_ub_vector(dut, tile_addr + r)
                actual_a[m*4 + r, n*4 : n*4 + 4] = row_vals
    
    expected = np.array(chain_driver.EXPECTED_A_OUT)
    
    if np.array_equal(actual_a, expected):
        dut._log.info("✅ Chain Test PASSED!")
    else:
        dut._log.error(f"Mismatch!\nExpected:\n{expected}\nGot:\n{actual_a}")
        mismatches = np.where(actual_a != expected)
        r, c = mismatches[0][0], mismatches[1][0]
        dut._log.error(f"First mismatch at [{r}][{c}]: Exp {expected[r,c]} vs Got {actual_a[r,c]}")
        assert False