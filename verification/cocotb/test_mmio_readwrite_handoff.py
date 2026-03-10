import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles

import npu_driver


def pack_word(vec16):
    word = 0
    for i, v in enumerate(vec16):
        word |= (v & 0xFFFF) << (i * 16)
    return word


async def write_ub_word(dut, addr, word):
    await npu_driver.write_reg(dut, npu_driver.REG_ADDR, addr, 16)
    await npu_driver.write_reg(dut, npu_driver.REG_CMD, 0x01, 8)  # WRITE_MEM
    await npu_driver.write_reg(dut, npu_driver.REG_MMVR, word, 128)


async def read_ub_word(dut, addr):
    vec = await npu_driver.read_ub_vector(dut, addr, 8)
    return pack_word(vec)


@cocotb.test()
async def test_mmio_readwrite_handoff(dut):
    """Regression: first WRITE_MEM after READ_WAIT must not use stale MMVR data."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    src_addr = 5
    dst_addr = 0
    src_word = 0x112233445566778899AABBCCDDEEFF00
    dst_word = 0xFEDCBA98765432100123456789ABCDEF

    # Seed source word and verify it.
    await write_ub_word(dut, src_addr, src_word)
    got_src = await read_ub_word(dut, src_addr)
    assert got_src == src_word, (
        f"Seed readback mismatch at src_addr={src_addr}: "
        f"expected 0x{src_word:032x}, got 0x{got_src:032x}"
    )

    # Critical sequence: while CU is in READ_WAIT after readback above,
    # issue first WRITE_MEM. This previously wrote stale MMVR contents.
    await write_ub_word(dut, dst_addr, dst_word)

    got_dst = await read_ub_word(dut, dst_addr)
    assert got_dst == dst_word, (
        "MMIO handoff bug: first WRITE_MEM after READ_WAIT used stale MMVR data. "
        f"expected 0x{dst_word:032x}, got 0x{got_dst:032x}"
    )
