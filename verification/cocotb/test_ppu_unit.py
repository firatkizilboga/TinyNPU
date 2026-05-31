import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge, Timer


ARRAY_SIZE = 8


def _pack_i32(values):
    word = 0
    for idx, value in enumerate(values):
        word |= (int(value) & 0xFFFF_FFFF) << (idx * 32)
    return word


def _unpack_i16_word(word):
    values = []
    for idx in range(ARRAY_SIZE):
        raw = (int(word) >> (idx * 16)) & 0xFFFF
        values.append(raw - 0x10000 if raw & 0x8000 else raw)
    return values


def _unpack_u16_word(word):
    return [(int(word) >> (idx * 16)) & 0xFFFF for idx in range(ARRAY_SIZE)]


def _clip_i16(value):
    return max(-32768, min(32767, int(value)))


def _round_shift_signed(value, shift):
    value = int(value)
    shift = int(shift)
    if shift <= 0:
        return value
    rounder = 1 << (shift - 1)
    if value >= 0:
        return (value + rounder) >> shift
    return -(((-value) + rounder) >> shift)


def _ppu_rescale(acc, *, multiplier=1, shift=0, bias=0):
    value = int(acc) + int(bias)
    product = value * int(multiplier)
    if shift > 0:
        product += 1 << (shift - 1)
    return _clip_i16(product >> shift)


def _ppu_sigmoid(value, *, shift, precision=2):
    p_out = 4 if precision == 0 else 8 if precision == 1 else 16
    qmax = (1 << (p_out - 1)) - 1
    scale = 1 << int(shift) if shift <= 15 else 0
    if scale <= 0 or qmax <= 0:
        return 0
    gate = ((int(value) * 218 + 64) >> 7) + (3 * scale)
    gate = min(max(gate, 0), 6 * scale)
    div6 = ((qmax * gate * 10923) + (1 << 15)) >> 16
    if shift > 0:
        div6 = (div6 + (1 << (int(shift) - 1))) >> int(shift)
    return _clip_i16(div6)


def _ppu_h_gelu(value, *, x_scale_shift=7):
    scale = 1 << int(x_scale_shift) if x_scale_shift < 31 else 0
    gate = ((int(value) * 218 + 64) >> 7) + (3 * scale)
    gate = min(max(gate, 0), 6 * scale)

    gelu_div6 = ((int(value) * gate * 10923) + (1 << 15)) >> 16
    return _clip_i16(_round_shift_signed(gelu_div6, int(x_scale_shift)))


async def _reset(dut):
    dut.rst_n.value = 0
    dut.capture_en.value = 0
    dut.bias_en.value = 0
    dut.bias_clear.value = 0
    dut.ppu_cycle_idx.value = 0
    dut.shift.value = 0
    dut.multiplier.value = 1
    dut.activation.value = 0
    dut.h_gelu_x_scale_shift.value = 7
    dut.precision.value = 2
    dut.write_offset.value = 0
    dut.output_layout.value = 0
    dut.writeback_mode.value = 0
    dut.cache_lane_idx.value = 0
    dut.bias_in.value = 0
    for idx in range(ARRAY_SIZE):
        dut.acc_in[idx].value = 0
    await ClockCycles(dut.clk, 6)
    dut.rst_n.value = 1
    await RisingEdge(dut.clk)


async def _load_bias(dut, values):
    dut.bias_en.value = 1
    dut.bias_in.value = _pack_i32(values[:4])
    await RisingEdge(dut.clk)
    dut.bias_in.value = _pack_i32(values[4:])
    await RisingEdge(dut.clk)
    dut.bias_en.value = 0
    await RisingEdge(dut.clk)


async def _capture_row0(
    dut,
    acc_values,
    *,
    activation=0,
    multiplier=1,
    shift=0,
    h_gelu_x_scale_shift=7,
    precision=2,
):
    for idx, value in enumerate(acc_values):
        dut.acc_in[idx].value = int(value)
    dut.activation.value = activation
    dut.h_gelu_x_scale_shift.value = h_gelu_x_scale_shift
    dut.multiplier.value = multiplier
    dut.shift.value = shift
    dut.precision.value = precision
    dut.ppu_cycle_idx.value = ARRAY_SIZE - 1
    dut.capture_en.value = 1
    await RisingEdge(dut.clk)
    dut.capture_en.value = 0
    for _ in range(16):
        await RisingEdge(dut.clk)
        if int(dut.done.value):
            break
    assert int(dut.done.value), "PPU did not complete captured row"
    await RisingEdge(dut.clk)
    dut.ppu_cycle_idx.value = 0
    await Timer(1, units="ns")
    return _unpack_i16_word(dut.ub_wdata.value)


@cocotb.test()
async def test_ppu_int16_bias_relu_saturation(dut):
    """PPU unit acceptance: bias, rescale, ReLU, saturation, and INT16 writeback."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await _reset(dut)

    biases = [10, -20, 30, -40, 1000, -1000, 0, 1]
    await _load_bias(dut, biases)

    raw = [1, 2, -3, -4, 40000, -40000, 32767, -32768]
    got = await _capture_row0(dut, raw)
    assert got == [11, -18, 27, -44, 32767, -32768, 32767, -32767]

    got_relu = await _capture_row0(dut, raw, activation=1)
    assert got_relu == [11, 0, 27, 0, 32767, 0, 32767, 0]

    got_shift = await _capture_row0(dut, [3, -3, 5, -5, 7, -7, 9, -9], multiplier=2, shift=1)
    assert got_shift == [13, -23, 35, -45, 1007, -1007, 9, -8]


@cocotb.test()
async def test_ppu_int8_int4_saturation_packing(dut):
    """Narrow precision modes still saturate and pack into the selected lane."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await _reset(dut)

    await _capture_row0(
        dut,
        [-200, -129, -128, -1, 0, 1, 127, 200],
        precision=1,
    )
    dut.ppu_cycle_idx.value = 0
    await Timer(1, units="ns")
    assert _unpack_u16_word(dut.ub_wdata.value) == [0x80, 0x80, 0x80, 0xFF, 0x00, 0x01, 0x7F, 0x7F]

    await _capture_row0(
        dut,
        [-20, -9, -8, -1, 0, 1, 7, 20],
        precision=0,
    )
    dut.ppu_cycle_idx.value = 0
    await Timer(1, units="ns")
    assert _unpack_u16_word(dut.ub_wdata.value) == [0x8, 0x8, 0x8, 0xF, 0x0, 0x1, 0x7, 0x7]


@cocotb.test()
async def test_ppu_sigmoid_activation(dut):
    """PPU sigmoid mode remains live through the staged activation pipeline."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await _reset(dut)

    shift = 4
    sigmoid_shift = 10
    rescaled_inputs = [-128, -64, -1, 0, 1, 64, 128, 256]
    acc_values = [value << shift for value in rescaled_inputs]

    got = await _capture_row0(
        dut,
        acc_values,
        activation=2,
        multiplier=1,
        shift=shift,
        h_gelu_x_scale_shift=sigmoid_shift,
    )
    expected = [_ppu_sigmoid(_ppu_rescale(acc, shift=shift), shift=sigmoid_shift) for acc in acc_values]
    assert got == expected


@cocotb.test()
async def test_ppu_h_gelu_activation(dut):
    """PPU hard-GELU mode remains live through activation, saturation, and writeback."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await _reset(dut)

    x_scale_shift = 7
    acc_values = [-1024, -128, -1, 0, 1, 128, 512, 1024]

    got = await _capture_row0(
        dut,
        acc_values,
        activation=3,
        multiplier=1,
        shift=0,
        h_gelu_x_scale_shift=x_scale_shift,
    )
    expected = [_ppu_h_gelu(_ppu_rescale(acc), x_scale_shift=x_scale_shift) for acc in acc_values]
    assert got == expected


@cocotb.test()
async def test_ppu_done_waits_for_full_tile(dut):
    """The control unit must not see done until the final captured row is stored."""
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    await _reset(dut)

    for cycle_idx in range(ARRAY_SIZE):
        for lane in range(ARRAY_SIZE):
            dut.acc_in[lane].value = cycle_idx * 100 + lane
        dut.ppu_cycle_idx.value = cycle_idx
        dut.capture_en.value = 1
        await RisingEdge(dut.clk)

    dut.capture_en.value = 0
    for _ in range(8):
        await RisingEdge(dut.clk)
        assert int(dut.done.value) == 0, "PPU signaled done before the tile drained"

    for _ in range(8):
        await RisingEdge(dut.clk)
        if int(dut.done.value):
            break
    assert int(dut.done.value), "PPU did not signal done for the final captured row"

    await RisingEdge(dut.clk)
    for row in range(ARRAY_SIZE):
        dut.ppu_cycle_idx.value = row
        await Timer(1, units="ns")
        expected = [(ARRAY_SIZE - 1 - row) * 100 + lane for lane in range(ARRAY_SIZE)]
        assert _unpack_i16_word(dut.ub_wdata.value) == expected
