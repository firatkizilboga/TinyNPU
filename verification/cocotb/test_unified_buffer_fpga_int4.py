import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge


DATA_WIDTH = 16
ARRAY_SIZE = 8
BUFFER_WIDTH = DATA_WIDTH * ARRAY_SIZE


def _lane_mask(nibble_offset: int) -> int:
    mask = 0
    for lane in range(ARRAY_SIZE):
        mask |= 0xF << (lane * DATA_WIDTH + nibble_offset * 4)
    return mask


def _lane_data(base: int, nibble_offset: int) -> int:
    data = 0
    for lane in range(ARRAY_SIZE):
        data |= ((base + lane) & 0xF) << (lane * DATA_WIDTH + nibble_offset * 4)
    return data


def _expected_word(bases: tuple[int, int, int, int]) -> int:
    word = 0
    for lane in range(ARRAY_SIZE):
        value = 0
        for nibble_offset, base in enumerate(bases):
            value |= ((base + lane) & 0xF) << (nibble_offset * 4)
        word |= value << (lane * DATA_WIDTH)
    return word


async def _read_word(dut, addr: int) -> tuple[int, int]:
    dut.input_addr.value = addr
    dut.weight_addr.value = addr
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    return dut.input_data.value.integer, dut.weight_data.value.integer


@cocotb.test()
async def test_fpga_bram_int4_partial_writes_pack_to_bytes(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst_n.value = 0
    dut.wr_en.value = 0
    dut.wr_mask.value = 0
    dut.wr_addr.value = 0
    dut.wr_data.value = 0
    dut.input_first_in.value = 0
    dut.input_last_in.value = 0
    dut.weight_first_in.value = 0
    dut.weight_last_in.value = 0
    dut.input_addr.value = 0
    dut.weight_addr.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)

    addresses = list(range(8))
    bases = (1, 9, 3, 11)

    # Model packed C-layout INT4 writeback: each pass writes one nibble into
    # the same 16-bit lanes across a small group of UB rows.
    for nibble_offset, base in enumerate(bases):
        for addr in addresses:
            dut.wr_en.value = 1
            dut.wr_addr.value = addr
            dut.wr_mask.value = _lane_mask(nibble_offset)
            dut.wr_data.value = _lane_data(base + addr, nibble_offset)
            await RisingEdge(dut.clk)

    dut.wr_en.value = 0
    dut.wr_mask.value = 0
    dut.wr_data.value = 0
    await RisingEdge(dut.clk)

    for addr in addresses:
        expected = _expected_word(tuple(base + addr for base in bases))
        input_word, weight_word = await _read_word(dut, addr)
        assert input_word == expected, f"input addr {addr}: expected 0x{expected:x}, got 0x{input_word:x}"
        assert weight_word == expected, f"weight addr {addr}: expected 0x{expected:x}, got 0x{weight_word:x}"
