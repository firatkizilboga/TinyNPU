import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
import multi_driver

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
    # Wait for Data Valid
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
async def test_multi(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    
    dut._log.info("Loading Program and Data...")
    for reg, val in multi_driver.DRIVER_MESSAGES:
        width = 64 if reg == REG_MMVR else (32 if reg == REG_ARG else (16 if reg == REG_ADDR else 8))
        await write_reg(dut, reg, val, width)

    dut._log.info("Waiting for HALT...")
    for _ in range(10000):
        dut.host_addr.value = REG_STATUS
        await RisingEdge(dut.clk)
        if int(dut.host_rd_data.value) == 0xFF: break
    else: raise AssertionError("Timeout")

    # Verify D (Result of A*B)
    dut._log.info("Verifying D = A*B...")
    actual_d = np.zeros((4,4), dtype=np.uint64)
    for r in range(4):
        actual_d[r, :] = await read_ub_vector(dut, multi_driver.ADDR_D + r)
    
    if np.array_equal(actual_d, multi_driver.EXPECTED_D):
        dut._log.info("✅ D is Correct!")
    else:
        dut._log.error(f"D Mismatch! Got:\n{actual_d}\nExpected:\n{np.array(multi_driver.EXPECTED_D)}")
        assert False

    # Verify E (Result of C*D)
    dut._log.info("Verifying E = C*D...")
    actual_e = np.zeros((4,4), dtype=np.uint64)
    for r in range(4):
        actual_e[r, :] = await read_ub_vector(dut, multi_driver.ADDR_E + r)
    
    if np.array_equal(actual_e, multi_driver.EXPECTED_E):
        dut._log.info("✅ E is Correct!")
    else:
        dut._log.error(f"E Mismatch! Got:\n{actual_e}\nExpected:\n{np.array(multi_driver.EXPECTED_E)}")
        assert False