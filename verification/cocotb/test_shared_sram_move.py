"""
test_shared_sram_move.py

Verifies the host shared SRAM port end-to-end:
  1. CPU writes a 128-bit vector to UB[0x0000] via host_shared_* signals.
  2. NPU executes MOVE(src=0x0000, dest=0x0004, count=1) + HALT.
  3. CPU reads back from UB[0x0004] via host_shared_rd_en and verifies it
     matches what was written.

This is the minimal proof that host_shared_allow gates correctly and that
the write/read lane logic is wired through ubss → unified_buffer properly.
"""

import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles, RisingEdge

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import npu_driver

# ------------------------------------------------------------------ constants

ARRAY_SIZE     = 8
BUFFER_WIDTH   = 128          # bits per UB word
LANES_PER_WORD = BUFFER_WIDTH // 32   # 4 lanes of 32 bits each

IM_BASE_ADDR   = 0x8000
CHUNKS_PER_INST = 2           # 256-bit instruction / 128-bit chunk

# MMIO addresses (from npu_driver)
REG_STATUS = npu_driver.REG_STATUS
REG_CMD    = npu_driver.REG_CMD
REG_ADDR   = npu_driver.REG_ADDR
REG_ARG    = npu_driver.REG_ARG
REG_MMVR   = npu_driver.REG_MMVR

CMD_WRITE_MEM = 0x01
CMD_RUN       = 0x03
STATUS_BUSY   = 0x01
STATUS_HALTED = 0xFF

DOORBELL_ADDR = REG_MMVR + (BUFFER_WIDTH // 8) - 1  # last MMVR byte

OP_HALT = 0x1
OP_MOVE = 0x3

# ------------------------------------------------------------------ helpers

async def _reset(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())

    dut.rst_n.value          = 0
    dut.host_wr_en.value     = 0
    # initialise shared port to safe idle state
    dut.host_shared_addr.value   = 0
    dut.host_shared_lane.value   = 0
    dut.host_shared_wr_data.value = 0
    dut.host_shared_wr_be.value  = 0
    dut.host_shared_wr_en.value  = 0
    dut.host_shared_rd_en.value  = 0

    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)


def _pack_move(src: int, dest: int, count: int) -> int:
    """Build a 256-bit MOVE instruction integer."""
    instr = 0
    instr |= (OP_MOVE  & 0xF)     << 252
    instr |= (src      & 0xFFFF)  << 232
    instr |= (dest     & 0xFFFF)  << 216
    instr |= (count    & 0xFFFF)  << 200
    return instr


def _pack_halt() -> int:
    """Build a 256-bit HALT instruction integer."""
    return (OP_HALT & 0xF) << 252


async def _load_instruction(dut, inst_idx: int, instr: int):
    """Write a 256-bit instruction into IM at instruction slot inst_idx."""
    base = IM_BASE_ADDR + inst_idx * CHUNKS_PER_INST
    for ci in range(CHUNKS_PER_INST):
        chunk = (instr >> (ci * BUFFER_WIDTH)) & ((1 << BUFFER_WIDTH) - 1)
        await npu_driver.write_reg(dut, REG_ADDR, base + ci, 16)
        await npu_driver.write_reg(dut, REG_CMD,  CMD_WRITE_MEM, 8)
        await npu_driver.write_reg(dut, REG_MMVR, chunk, BUFFER_WIDTH)


async def _load_instruction_shared(dut, inst_idx: int, instr: int):
    """Write a 256-bit instruction into IM via shared SRAM host port."""
    base = IM_BASE_ADDR + inst_idx * CHUNKS_PER_INST
    for ci in range(CHUNKS_PER_INST):
        chunk = (instr >> (ci * BUFFER_WIDTH)) & ((1 << BUFFER_WIDTH) - 1)
        lanes = [((chunk >> (lane * 32)) & 0xFFFF_FFFF) for lane in range(LANES_PER_WORD)]
        await _shared_write_word(dut, base + ci, lanes)


async def _run_until_halt(dut):
    """Trigger execution from IM_BASE_ADDR and wait for STATUS_HALTED."""
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
    raise AssertionError("Timeout: NPU never reached HALTED status.")


async def _shared_write_word(dut, ub_addr: int, lanes: list[int]):
    """
    Write one 128-bit UB word via host_shared port.
    lanes: list of 4 uint32 values [lane0, lane1, lane2, lane3].
    Each lane is written in a separate clock cycle.
    """
    assert len(lanes) == LANES_PER_WORD
    # Check allow before first write (NPU must be idle)
    allow = int(dut.host_shared_allow.value)
    assert allow == 1, f"host_shared_allow=0 before write — NPU not idle?"

    for lane_idx, value in enumerate(lanes):
        dut.host_shared_addr.value    = ub_addr
        dut.host_shared_lane.value    = lane_idx
        dut.host_shared_wr_data.value = value & 0xFFFF_FFFF
        dut.host_shared_wr_be.value   = 0xF          # all 4 bytes enabled
        dut.host_shared_wr_en.value   = 1
        await RisingEdge(dut.clk)

    # De-assert write enable
    dut.host_shared_wr_en.value = 0
    dut.host_shared_wr_be.value = 0
    await RisingEdge(dut.clk)


async def _shared_read_word(dut, ub_addr: int) -> list[int]:
    """
    Read one 128-bit UB word via host_shared combinational read tap.
    Returns list of 4 uint32 lane values.
    """
    lanes = []
    for lane_idx in range(LANES_PER_WORD):
        dut.host_shared_addr.value  = ub_addr
        dut.host_shared_lane.value  = lane_idx
        dut.host_shared_rd_en.value = 1
        # combinational output; sample after one clock so signal propagates
        await RisingEdge(dut.clk)
        lanes.append(int(dut.host_shared_rd_data.value))

    dut.host_shared_rd_en.value = 0
    await RisingEdge(dut.clk)
    return lanes


# ------------------------------------------------------------------ test

@cocotb.test()
async def test_cpu_write_npu_move_cpu_verify(dut):
    """
    CPU writes vector to UB[0x0000] via shared SRAM port.
    NPU moves it to UB[0x0004].
    CPU reads back from UB[0x0004] and verifies.
    """
    await _reset(dut)

    # --- Step 1: CPU writes test vector to UB[0x0000] ---
    src_addr  = 0x0000
    dest_addr = 0x0004

    # Arbitrary but distinctive values so bit-flips are caught
    write_lanes = [0xDEAD_BEEF, 0xCAFE_F00D, 0x1234_5678, 0xABCD_EF01]
    dut._log.info(f"CPU: writing {[hex(v) for v in write_lanes]} → UB[{src_addr:#06x}]")
    await _shared_write_word(dut, src_addr, write_lanes)

    # Sanity: read back from src before NPU runs to confirm write landed
    readback_src = await _shared_read_word(dut, src_addr)
    dut._log.info(f"CPU readback src before run: {[hex(v) for v in readback_src]}")
    assert readback_src == write_lanes, \
        f"Src write didn't land: got {[hex(v) for v in readback_src]}"

    # --- Step 2: load MOVE(0x0000→0x0004, count=1) + HALT into IM ---
    move_instr = _pack_move(src=src_addr, dest=dest_addr, count=1)
    halt_instr = _pack_halt()

    dut._log.info("CPU: loading MOVE + HALT into IM via shared SRAM")
    await _load_instruction_shared(dut, inst_idx=0, instr=move_instr)
    await _load_instruction_shared(dut, inst_idx=1, instr=halt_instr)

    # --- Step 3: run NPU ---
    dut._log.info("CPU: triggering NPU run")
    await _run_until_halt(dut)
    dut._log.info("NPU: halted")

    # --- Step 4: CPU reads destination via shared SRAM port ---
    readback_dest = await _shared_read_word(dut, dest_addr)
    dut._log.info(f"CPU readback dest after run: {[hex(v) for v in readback_dest]}")

    assert readback_dest == write_lanes, (
        f"MOVE result mismatch at UB[{dest_addr:#06x}]:\n"
        f"  expected {[hex(v) for v in write_lanes]}\n"
        f"  got      {[hex(v) for v in readback_dest]}"
    )
    dut._log.info("PASS: host_shared write → NPU MOVE → host_shared read verified.")
