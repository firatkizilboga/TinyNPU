import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# Opcodes
OP_NOP  = 0x0
OP_HALT = 0x1
OP_MOVE = 0x3

CMD_WRITE_MEM = 0x01
CMD_RUN       = 0x03

# MMIO Offsets
REG_STATUS = 0x00
REG_CMD    = 0x04
REG_ADDR   = 0x08
REG_ARG    = 0x0C
REG_MMVR   = 0x10

async def write_reg(dut, addr, data, width=8):
    for i in range(width // 8):
        dut.host_addr.value = addr + i
        dut.host_wr_data.value = (data >> (i * 8)) & 0xFF
        dut.host_wr_en.value = 1
        await RisingEdge(dut.clk)
        dut.host_wr_en.value = 0

async def load_instruction(dut, inst_idx, opcode, arg1=0, arg2=0, arg3=0):
    base_word_addr = 0x8000 + (inst_idx * 4)
    # Word 3 (MSB) contains Opcode[255:252]
    # For MOVE: Src[247:232], Dest[231:216], Len[215:200]
    if opcode == OP_MOVE:
        word3 = (opcode << 60) | (arg1 << 40) | (arg2 << 24) | (arg3 << 8)
    else:
        word3 = opcode << 60
        
    await write_reg(dut, REG_CMD, CMD_WRITE_MEM)
    await write_reg(dut, REG_ADDR, base_word_addr + 3, 16)
    await write_reg(dut, REG_MMVR, word3, 64)

@cocotb.test()
async def test_move_execution(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    dut.host_wr_en.value = 0
    dut.ub_rdata.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    # 1. Host loads data into UB[0x10]
    dut._log.info("Host: Writing 0xABCDEFA1 to UB[0x10]")
    await write_reg(dut, REG_CMD, CMD_WRITE_MEM)
    await write_reg(dut, REG_ADDR, 0x0010, 16)
    await write_reg(dut, REG_MMVR, 0xABCDEFA1, 64)
    await RisingEdge(dut.clk) # Wait for write to complete

    # 2. Host loads program: MOVE(0x10 -> 0x20, len=1), HALT
    dut._log.info("Loading Program: MOVE(0x10->0x20, 1), HALT")
    await load_instruction(dut, 0, OP_MOVE, 0x10, 0x20, 1)
    await load_instruction(dut, 1, OP_HALT)

    # 3. Trigger Run
    dut._log.info("Triggering Run...")
    await write_reg(dut, REG_CMD, CMD_RUN)
    await write_reg(dut, REG_ARG, 0, 32)
    await write_reg(dut, REG_MMVR, 0, 64)

    # 4. Mock Memory Response & Verify Move
    # We need to watch for the TPU reading from 0x10
    # and then writing that same data to 0x20.
    
    move_completed = False
    captured_data = 0
    
    for i in range(100):
        await RisingEdge(dut.clk)
        
        # If TPU is reading from 0x10, provide the data on the next cycle
        if dut.ub_wr_en.value == 0 and dut.ub_addr.value == 0x10:
            dut.ub_rdata.value = 0xABCDEFA1
            dut._log.info(f"Memory: Providing 0xABCDEFA1 for address 0x10 at cycle {i}")
            
        # If TPU is writing to 0x20, check the data
        if dut.ub_wr_en.value == 1 and dut.ub_addr.value == 0x20:
            assert dut.ub_wdata.value == 0xABCDEFA1
            dut._log.info(f"Success: TPU wrote 0xABCDEFA1 to UB[0x20] at cycle {i}")
            move_completed = True
            
        # Check for HALT
        dut.host_addr.value = REG_STATUS
        await Timer(1, units="ps")
        if dut.host_rd_data.value == 0xFF:
            dut._log.info(f"Halt detected at cycle {i}")
            break

    assert move_completed, "MOVE operation never completed"
    dut._log.info("✅ MOVE EXECUTION TEST PASSED")