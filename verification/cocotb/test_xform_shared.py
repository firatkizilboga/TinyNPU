import os
import sys

import cocotb
import numpy as np
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, FallingEdge, ReadOnly, RisingEdge


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import npu_driver


BUFFER_WIDTH = 128
LANES_PER_WORD = 4
IM_BASE_ADDR = 0x9000
CHUNKS_PER_INST = 2

REG_STATUS = npu_driver.REG_STATUS
REG_CMD = npu_driver.REG_CMD
REG_ARG = npu_driver.REG_ARG
REG_MMVR = npu_driver.REG_MMVR

CMD_RUN = 0x03
STATUS_BUSY = 0x01
STATUS_HALTED = 0xFF
DOORBELL_ADDR = REG_MMVR + (BUFFER_WIDTH // 8) - 1

OP_HALT = 0x1
OP_XFORM = 0x4
XFORM_MODE_Q_F16_I16 = 0x1
XFORM_MODE_DQ_I16_F16 = 0x2


async def _reset(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    dut.host_wr_en.value = 0
    dut.host_shared_addr.value = 0
    dut.host_shared_lane.value = 0
    dut.host_shared_wr_data.value = 0
    dut.host_shared_wr_be.value = 0
    dut.host_shared_wr_en.value = 0
    dut.host_shared_rd_en.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


def _pack_halt():
    return (OP_HALT & 0xF) << 252


def _pack_xform(mode, src, dest, count, multiplier, shift):
    instr = 0
    instr |= (OP_XFORM & 0xF) << 252
    instr |= (mode & 0xF) << 248
    instr |= (src & 0xFFFF) << 232
    instr |= (dest & 0xFFFF) << 216
    instr |= (count & 0xFFFF) << 200
    instr |= (multiplier & 0xFFFF) << 184
    instr |= (shift & 0xFF) << 176
    return instr


def _pack_xform_q(src, dest, count, multiplier, shift):
    return _pack_xform(XFORM_MODE_Q_F16_I16, src, dest, count, multiplier, shift)


def _pack_xform_dq(src, dest, count, multiplier, shift):
    return _pack_xform(XFORM_MODE_DQ_I16_F16, src, dest, count, multiplier, shift)


def _pack_i16_word(values):
    word = 0
    for idx, value in enumerate(values):
        word |= (int(value) & 0xFFFF) << (idx * 16)
    return word


def _unpack_i16_word(lanes):
    word = 0
    for idx, lane in enumerate(lanes):
        word |= (int(lane) & 0xFFFF_FFFF) << (idx * 32)
    values = []
    for idx in range(8):
        raw = (word >> (idx * 16)) & 0xFFFF
        values.append(raw - 0x10000 if raw & 0x8000 else raw)
    return values


def _unpack_u16_word(lanes):
    word = 0
    for idx, lane in enumerate(lanes):
        word |= (int(lane) & 0xFFFF_FFFF) << (idx * 32)
    return [(word >> (idx * 16)) & 0xFFFF for idx in range(8)]


async def _shared_write_word(dut, addr, lanes):
    assert int(dut.host_shared_allow.value) == 1
    for lane_idx, value in enumerate(lanes):
        dut.host_shared_addr.value = addr
        dut.host_shared_lane.value = lane_idx
        dut.host_shared_wr_data.value = int(value) & 0xFFFF_FFFF
        dut.host_shared_wr_be.value = 0xF
        dut.host_shared_wr_en.value = 1
        await RisingEdge(dut.clk)
    dut.host_shared_wr_en.value = 0
    dut.host_shared_wr_be.value = 0
    await RisingEdge(dut.clk)


async def _shared_read_word(dut, addr):
    lanes = []
    for lane_idx in range(LANES_PER_WORD):
        dut.host_shared_addr.value = addr
        dut.host_shared_lane.value = lane_idx
        dut.host_shared_rd_en.value = 1
        await RisingEdge(dut.clk)
        await ReadOnly()
        lanes.append(int(dut.host_shared_rd_data.value))
        await FallingEdge(dut.clk)
    dut.host_shared_rd_en.value = 0
    await RisingEdge(dut.clk)
    return lanes


async def _load_instruction_shared(dut, inst_idx, instr):
    base = IM_BASE_ADDR + inst_idx * CHUNKS_PER_INST
    for chunk_idx in range(CHUNKS_PER_INST):
        chunk = (instr >> (chunk_idx * BUFFER_WIDTH)) & ((1 << BUFFER_WIDTH) - 1)
        lanes = [(chunk >> (lane * 32)) & 0xFFFF_FFFF for lane in range(LANES_PER_WORD)]
        await _shared_write_word(dut, base + chunk_idx, lanes)


async def _run_until_halt(dut):
    await npu_driver.write_reg(dut, REG_ARG, IM_BASE_ADDR, 32)
    await npu_driver.write_reg(dut, REG_CMD, CMD_RUN, 8)
    await npu_driver.write_reg(dut, DOORBELL_ADDR, 0, 8)
    saw_busy = False
    for _ in range(200_000):
        dut.host_addr.value = REG_STATUS
        await RisingEdge(dut.clk)
        status = int(dut.host_rd_data.value)
        if status == STATUS_BUSY:
            saw_busy = True
        if saw_busy and status == STATUS_HALTED:
            return
    raise AssertionError("XFORM program did not halt")


@cocotb.test()
async def test_xform_q_f16_i16_one_word(dut):
    await _reset(dut)

    src_addr = 0
    dest_addr = 4
    multiplier = 16
    shift = 0
    values_f32 = np.array([1.0, -2.0, 3.5, -4.25, 0.5, -0.5, 2.5, -1.5], dtype=np.float32)
    values_f16 = values_f32.astype(np.float16)
    expected = np.clip(np.rint(values_f16.astype(np.float32) * multiplier), -32768, 32767).astype(np.int16).tolist()
    src_word = _pack_i16_word(values_f16.view(np.uint16).astype(np.int32).tolist())
    src_lanes = [(src_word >> (lane * 32)) & 0xFFFF_FFFF for lane in range(LANES_PER_WORD)]

    await _shared_write_word(dut, src_addr, src_lanes)
    await _load_instruction_shared(dut, 0, _pack_xform_q(src_addr, dest_addr, 1, multiplier, shift))
    await _load_instruction_shared(dut, 1, _pack_halt())
    await _run_until_halt(dut)

    got = _unpack_i16_word(await _shared_read_word(dut, dest_addr))
    assert got == expected, f"XFORM Q mismatch: got {got}, expected {expected}"


@cocotb.test()
async def test_xform_dq_i16_f16_one_word(dut):
    await _reset(dut)

    src_addr = 0
    dest_addr = 4
    multiplier = 16
    shift = 4
    values_i16 = np.array([1, -2, 3, -4, 8, -16, 32, -64], dtype=np.int16)
    expected = values_i16.astype(np.float16).view(np.uint16).astype(np.int32).tolist()
    src_word = _pack_i16_word(values_i16.tolist())
    src_lanes = [(src_word >> (lane * 32)) & 0xFFFF_FFFF for lane in range(LANES_PER_WORD)]

    await _shared_write_word(dut, src_addr, src_lanes)
    await _load_instruction_shared(dut, 0, _pack_xform_dq(src_addr, dest_addr, 1, multiplier, shift))
    await _load_instruction_shared(dut, 1, _pack_halt())
    await _run_until_halt(dut)

    got = _unpack_u16_word(await _shared_read_word(dut, dest_addr))
    assert got == expected, f"XFORM DQ mismatch: got {got}, expected {expected}"
