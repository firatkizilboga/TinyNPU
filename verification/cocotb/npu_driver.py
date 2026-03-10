import cocotb
from cocotb.triggers import RisingEdge

REG_STATUS = 0x00
REG_CMD    = 0x04
REG_ADDR   = 0x08
REG_ARG    = 0x0C
REG_MMVR   = 0x10

async def write_reg(dut, addr, data, width=8):
    """Writes a value of arbitrary width to a register (byte by byte)."""
    num_bytes = (width + 7) // 8
    for i in range(num_bytes):
        dut.host_addr.value = addr + i
        dut.host_wr_data.value = (int(data) >> (i * 8)) & 0xFF
        dut.host_wr_en.value = 1
        await RisingEdge(dut.clk)
    dut.host_wr_en.value = 0

async def read_ub_vector(dut, addr, array_size):
    """Reads a full vector from the Unified Buffer via MMVR."""
    mmvr_bytes = (array_size * 16) // 8
    doorbell_addr = REG_MMVR + mmvr_bytes - 1
    
    await write_reg(dut, REG_ADDR, addr, 16)
    await write_reg(dut, REG_CMD, 2, 8) # CMD_READ_MEM
    await write_reg(dut, doorbell_addr, 0, 8) # Trigger
    
    # Wait for Data Valid
    for _ in range(100):
        dut.host_addr.value = REG_STATUS
        await RisingEdge(dut.clk)
        if int(dut.host_rd_data.value) == 0x02: break
    else: raise AssertionError(f"Timeout waiting for read at {addr}")
    
    res_bytes = []
    for i in range(mmvr_bytes):
        dut.host_addr.value = REG_MMVR + i
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk) # Wait for combinational/sync read logic to update data
        res_bytes.append(int(dut.host_rd_data.value))
    
    res = []
    for i in range(array_size):
        val = res_bytes[i*2] | (res_bytes[i*2+1] << 8)
        res.append(val)
    return res
