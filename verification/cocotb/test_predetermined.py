import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# Opcodes & Constants (Matching defines.sv)
CMD_RUN = 0x03
REG_CMD = 0x04
REG_ARG = 0x0C
REG_MMVR = 0x10


async def write_reg(dut, addr, data, width=8):
    for i in range(width // 8):
        dut.host_addr.value = addr + i
        dut.host_wr_data.value = (data >> (i * 8)) & 0xFF
        dut.host_wr_en.value = 1
        await RisingEdge(dut.clk)
        dut.host_wr_en.value = 0


@cocotb.test()
async def test_predetermined(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    dut.host_wr_en.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # Give it a few cycles
    await cocotb.triggers.ClockCycles(dut.clk, 10)

    # Start Execution from address 0
    dut._log.info("Triggering CMD_RUN...")
    await write_reg(dut, REG_CMD, CMD_RUN)
    await write_reg(dut, REG_ARG, 0, 32)
    await write_reg(dut, REG_MMVR, 0, 64)

    # Monitor for completion
    for i in range(500):
        await RisingEdge(dut.clk)
        if dut.all_done.value == 1:
            dut._log.info(f"ALL DONE at cycle {i}!")
            # The result should be at 0x300 in UB
            # Since C = A * B = I * 5s = 5s
            return

    raise AssertionError("Timeout waiting for all_done")
