import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

try:
    from compiler_driver import DRIVER_MESSAGES
except ImportError:
    DRIVER_MESSAGES = []

# MMIO Regs (matching defines.sv)
REG_STATUS = 0x00
REG_CMD    = 0x04
REG_ADDR   = 0x08
REG_ARG    = 0x0C
REG_MMVR   = 0x10

async def write_reg(dut, addr, data, width=8):
    """Writes to MMIO with correct byte-steering."""
    num_bytes = width // 8
    for i in range(num_bytes):
        dut.host_addr.value = addr + i
        dut.host_wr_data.value = (int(data) >> (i * 8)) & 0xFF
        dut.host_wr_en.value = 1
        await RisingEdge(dut.clk)
    dut.host_wr_en.value = 0

async def read_ub_vector(dut, addr):
    """Reads a 64-bit vector from UB via MMIO."""
    # 1. Set Address (16-bit)
    await write_reg(dut, REG_ADDR, addr, 16)
    # 2. Set Command READ (8-bit)
    await write_reg(dut, REG_CMD, 0x02, 8) 
    # 3. Trigger via MMVR Doorbell (write to byte 7)
    await write_reg(dut, REG_MMVR, 0, 64)
    
    # 4. Wait for DATA_VALID (Status 0x02)
    for _ in range(100):
        dut.host_addr.value = REG_STATUS
        await RisingEdge(dut.clk)
        if int(dut.host_rd_data.value) == 0x02:
            break
    else:
        dut._log.error("Timeout waiting for DATA_VALID")
            
    # 5. Read 8 bytes from MMVR
    val = 0
    for i in range(8):
        dut.host_addr.value = REG_MMVR + i
        await RisingEdge(dut.clk)
        val |= (int(dut.host_rd_data.value) << (i * 8))
    return val

async def check_results(dut):
    """Common check: Matrix C should be Matrix A * 3."""
    dut._log.info("Checking Results in UB...")
    # Matrix C is at 0x0008 (4 vectors)
    expected_a = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [1, 1, 1, 1],
        [2, 2, 2, 2]
    ]
    
    for i in range(4):
        actual_packed = await read_ub_vector(dut, 8 + i)
        # Unpack 4x 16-bit
        actual = [
            (actual_packed >> 0) & 0xFFFF,
            (actual_packed >> 16) & 0xFFFF,
            (actual_packed >> 32) & 0xFFFF,
            (actual_packed >> 48) & 0xFFFF,
        ]
        expected = [x * 3 for x in expected_a[i]]
        dut._log.info(f"Row {i}: Expected {expected}, Got {actual}")
        assert actual == expected, f"Result mismatch at row {i}"

@cocotb.test()
async def test_preloaded(dut):
    """Case 1: Data and Instructions already in memory."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    await cocotb.triggers.ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Start Execution (CMD_RUN @ ARG=0x8000)
    dut._log.info("Starting Preloaded Execution...")
    await write_reg(dut, REG_ARG, 0x8000, 32)
    await write_reg(dut, REG_CMD, 0x03, 8) # CMD_RUN
    await write_reg(dut, REG_MMVR, 0, 64) # Trigger doorbell

    # Wait for HALT (Status 0xFF)
    for i in range(2000):
        dut.host_addr.value = REG_STATUS
        await RisingEdge(dut.clk)
        status = int(dut.host_rd_data.value)
        if status == 0xFF:
            dut._log.info(f"HALT detected at cycle {i*10}")
            break
        if status == 0xFE:
            raise AssertionError("TPU entered ERROR state")
        await cocotb.triggers.ClockCycles(dut.clk, 10)
    else:
        raise AssertionError("Timeout waiting for HALT")

    await check_results(dut)

@cocotb.test()
async def test_host_driven(dut):
    """Case 2: Host loads everything via MMIO."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    await cocotb.triggers.ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    
    dut._log.info(f"Loading {len(DRIVER_MESSAGES)} MMIO messages...")
    for reg, val in DRIVER_MESSAGES:
        # Determine width based on reg
        width = 8
        if reg == REG_ADDR: width = 16
        elif reg == REG_ARG: width = 32
        elif reg == REG_MMVR: width = 64
        
        await write_reg(dut, reg, val, width)
        
    dut._log.info("Execution triggered by driver. Waiting for completion...")
    
    # Wait for HALT
    for i in range(5000):
        dut.host_addr.value = REG_STATUS
        await RisingEdge(dut.clk)
        status = int(dut.host_rd_data.value)
        if status == 0xFF:
            dut._log.info(f"HALT detected at cycle {i*10}")
            break
        if status == 0xFE:
            raise AssertionError("TPU entered ERROR state")
        await cocotb.triggers.ClockCycles(dut.clk, 10)
    else:
        raise AssertionError("Timeout waiting for HALT")

    await check_results(dut)