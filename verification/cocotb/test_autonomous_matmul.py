import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# Opcodes & Constants (Matching defines.sv)
OP_HALT   = 0x1
OP_MATMUL = 0x2
CMD_WRITE_MEM = 0x01
CMD_RUN       = 0x03
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

async def load_matmul_instruction(dut, inst_idx, a_base, b_base, m, k, n):
    base_word_addr = 0x8000 + (inst_idx * 4)
    
    # Matches control_unit.sv decoding:
    # Word 3 (MSB): [255:192] -> Opcode(4), Flags(4), A_Base(16), B_Base(16), Out_Base(16), Bias_Base(16)
    # We simplified in test to: Opcode, A, B
    
    # A_Base: [247:232], B_Base: [231:216]
    word3 = (OP_MATMUL << 60) | (a_base << 40) | (b_base << 24)
    
    # Word 2: [191:128] -> M[183:168], K[167:152], N[151:136]
    word2 = (m << 40) | (k << 24) | (n << 8)
    
    word1 = 0
    word0 = 0
    
    await write_reg(dut, REG_CMD, CMD_WRITE_MEM)
    
    dut._log.info(f"Writing Word 0 to {hex(base_word_addr)}")
    await write_reg(dut, REG_ADDR, base_word_addr + 0, 16)
    await write_reg(dut, REG_MMVR, word0, 64)
    await Timer(10, units="ns")

    dut._log.info(f"Writing Word 1 to {hex(base_word_addr+1)}")
    await write_reg(dut, REG_ADDR, base_word_addr + 1, 16)
    await write_reg(dut, REG_MMVR, word1, 64)
    await Timer(10, units="ns")

    dut._log.info(f"Writing Word 2 to {hex(base_word_addr+2)}")
    await write_reg(dut, REG_ADDR, base_word_addr + 2, 16)
    await write_reg(dut, REG_MMVR, word2, 64)
    await Timer(10, units="ns")

    dut._log.info(f"Writing Word 3 to {hex(base_word_addr+3)}")
    await write_reg(dut, REG_ADDR, base_word_addr + 3, 16)
    await write_reg(dut, REG_MMVR, word3, 64)
    await Timer(10, units="ns")

@cocotb.test()
async def test_autonomous_matmul(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    dut.rst_n.value = 0
    dut.host_wr_en.value = 0
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)
    
    # 1. Load Data
    a_base = 0x100
    b_base = 0x200
    
    # Load A: 2 tiles (8x4)
    # Tile 0: All 1s, Tile 1: All 2s
    for i in range(4):
        await write_reg(dut, REG_CMD, CMD_WRITE_MEM)
        await write_reg(dut, REG_ADDR, a_base + i, 16)
        # Packed 16-bit: 4 x 0x0001
        await write_reg(dut, REG_MMVR, 0x0001000100010001, 64)
        
    for i in range(4):
        await write_reg(dut, REG_CMD, CMD_WRITE_MEM)
        await write_reg(dut, REG_ADDR, a_base + 4 + i, 16)
        # Packed 16-bit: 4 x 0x0002
        await write_reg(dut, REG_MMVR, 0x0002000200020002, 64)

    # Load B: 1 tile (4x4)
    # Identity matrix: 
    # Row 0: 1 0 0 0 -> 0x0000000000000001 (Little Endian?)
    # Input is 4x16-bit. 
    # Word = [El3, El2, El1, El0]
    b_data = [
        0x0000000000000001,
        0x0000000000010000,
        0x0000000100000000,
        0x0001000000000000
    ]
    for i in range(4):
        await write_reg(dut, REG_CMD, CMD_WRITE_MEM)
        await write_reg(dut, REG_ADDR, b_base + i, 16)
        await write_reg(dut, REG_MMVR, b_data[i], 64)
        
    # 2. Load Instruction (M=2, K=1, N=1)
    # This means 2 output tiles.
    dut._log.info("Loading Instruction...")
    await load_matmul_instruction(dut, 0, a_base, b_base, 2, 1, 1)
    
    # 3. Load HALT at PC=1 (Addr 0x8004)
    await write_reg(dut, REG_CMD, CMD_WRITE_MEM)
    await write_reg(dut, REG_ADDR, 0x8004 + 3, 16)
    await write_reg(dut, REG_MMVR, OP_HALT << 60, 64)
    
    # 4. Run
    dut._log.info("Running...")
    await write_reg(dut, REG_CMD, CMD_RUN)
    await write_reg(dut, REG_ARG, 0, 32)
    await write_reg(dut, REG_MMVR, 0, 64)
    
    # 5. Monitor
    for i in range(200):
        await RisingEdge(dut.clk)
        if dut.all_done.value == 1:
            dut._log.info(f"ALL DONE at cycle {i}!")
            return
            
    raise AssertionError("Timeout waiting for all_done")
