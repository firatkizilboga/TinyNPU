import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

async def write_byte(dut, addr, data):
    dut.host_addr.value = addr
    dut.host_wr_data.value = data
    dut.host_wr_en.value = 1
    await RisingEdge(dut.clk)
    dut.host_wr_en.value = 0

async def read_byte(dut, addr):
    dut.host_addr.value = addr
    await Timer(1, units="ns")
    return dut.host_rd_data.value.integer

@cocotb.test()
async def test_mmio_basic(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    dut.host_wr_en.value = 0
    dut.status_in.value = 0xAB
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # 1. Status Read
    val = await read_byte(dut, 0x00)
    assert val == 0xAB

    # 2. CMD Reg
    await write_byte(dut, 0x04, 0x55)
    await Timer(1, units="ns") # Let combinational paths settle
    assert dut.cmd_out.value == 0x55

    # 3. ADDR Reg (16-bit)
    await write_byte(dut, 0x08, 0x12)
    await write_byte(dut, 0x09, 0x34)
    await Timer(1, units="ns")
    assert dut.addr_out.value == 0x3412

    # 4. ARG Reg (32-bit)
    await write_byte(dut, 0x0C, 0x11)
    await write_byte(dut, 0x0D, 0x22)
    await write_byte(dut, 0x0E, 0x33)
    await write_byte(dut, 0x0F, 0x44)
    await Timer(1, units="ns")
    assert dut.arg_out.value == 0x44332211

    # 5. MMVR and Doorbell
    for i in range(7):
        await write_byte(dut, 0x10 + i, i + 1)
        assert dut.doorbell_pulse.value == 0

    # Write 8th byte - pulse should happen on next rising edge
    dut.host_addr.value = 0x17
    dut.host_wr_data.value = 0x88
    dut.host_wr_en.value = 1
    await RisingEdge(dut.clk)
    dut.host_wr_en.value = 0
    
    # Pulse is registered, should be visible NOW (right after RisingEdge)
    # Verilator might need a tiny delay to propagate in the simulator view
    await Timer(1, units="ps") 
    assert dut.doorbell_pulse.value == 1, f"Doorbell not 1, got {dut.doorbell_pulse.value}"
    await RisingEdge(dut.clk)
    await Timer(1, units="ps")
    assert dut.doorbell_pulse.value == 0, f"Doorbell not 0, got {dut.doorbell_pulse.value}"

    actual_mmvr = dut.mmvr_out.value.integer
    assert (actual_mmvr & 0xFF) == 1
    assert (actual_mmvr >> 56) == 0x88
    
    dut._log.info("✅ MMIO UNIT TEST PASSED")