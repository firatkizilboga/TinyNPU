import json
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles

import npu_driver
from perf_utils import CTRL_STATE_NAMES, diff_counters, unpack_flat_counters


def read_perf_snapshot(dut):
    cu = dut.u_brain.u_cu
    return {
        "total": int(cu.perf_total_cycles.value),
        "cycles": unpack_flat_counters(int(cu.perf_state_cycles_flat.value)),
        "entries": unpack_flat_counters(int(cu.perf_state_entries_flat.value)),
    }


@cocotb.test()
async def test_perf_observability(dut):
    """PERF_ENABLE=1 should expose per-state counters without changing behavior."""
    assert int(dut.PERF_ENABLE.value) == 1, "Run this test with -GPERF_ENABLE=1"

    npu_file = os.environ.get("NPU_FILE", "simple_chain.npu")
    if not os.path.exists(npu_file):
        raise FileNotFoundError(f"NPU program file not found: {npu_file}")

    with open(npu_file, "r") as f:
        prog = json.load(f)

    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

    config = prog["config"]
    buffer_width = config["buffer_width"]
    im_base = config["im_base"]

    # Load UB
    for addr, word_hex in enumerate(prog["ub"]):
        word = int(word_hex, 16)
        await npu_driver.write_reg(dut, npu_driver.REG_ADDR, addr, 16)
        await npu_driver.write_reg(dut, npu_driver.REG_CMD, 0x01, 8)
        await npu_driver.write_reg(dut, npu_driver.REG_MMVR, word, buffer_width)

    # Load IM
    inst_width = 256
    num_chunks = inst_width // buffer_width
    for i, inst_hex in enumerate(prog["im"]):
        inst = int(inst_hex, 16)
        for c in range(num_chunks):
            chunk = (inst >> (c * buffer_width)) & ((1 << buffer_width) - 1)
            await npu_driver.write_reg(dut, npu_driver.REG_ADDR, im_base + (i * num_chunks) + c, 16)
            await npu_driver.write_reg(dut, npu_driver.REG_CMD, 0x01, 8)
            await npu_driver.write_reg(dut, npu_driver.REG_MMVR, chunk, buffer_width)

    # Let the final host write retire so the RUN-phase snapshot starts cleanly.
    await ClockCycles(dut.clk, 2)
    before = read_perf_snapshot(dut)

    # Run
    doorbell_addr = npu_driver.REG_MMVR + (buffer_width // 8) - 1
    await npu_driver.write_reg(dut, npu_driver.REG_ARG, im_base, 32)
    await npu_driver.write_reg(dut, npu_driver.REG_CMD, 0x03, 8)
    await npu_driver.write_reg(dut, doorbell_addr, 0, 8)

    for _ in range(100000):
        dut.host_addr.value = npu_driver.REG_STATUS
        await RisingEdge(dut.clk)
        if int(dut.host_rd_data.value) == 0xFF:
            break
    else:
        raise AssertionError("Timeout waiting for HALT")

    after = read_perf_snapshot(dut)
    delta_total = after["total"] - before["total"]
    delta_cycles = diff_counters(after["cycles"], before["cycles"])
    delta_entries = diff_counters(after["entries"], before["entries"])

    dut._log.info("PERF state breakdown for RUN phase:")
    for name in CTRL_STATE_NAMES:
        dut._log.info(f"{name}: cycles={delta_cycles[name]} entries={delta_entries[name]}")

    assert delta_total > 0, "Total cycle counter did not advance"
    assert sum(delta_cycles.values()) == delta_total, "Per-state cycles do not sum to total delta"

    expected_active = [
        "CTRL_FETCH",
        "CTRL_DECODE",
        "CTRL_EXEC_MATMUL",
        "CTRL_MM_CLEAR",
        "CTRL_MM_FEED",
        "CTRL_MM_WAIT",
        "CTRL_MM_DRAIN_SA",
        "CTRL_MM_WRITEBACK",
    ]
    expected_quiet = [
        "CTRL_HOST_READ",
        "CTRL_READ_WAIT",
        "CTRL_EXEC_MOVE",
        "CTRL_MM_LOAD_BIAS",
    ]

    for name in expected_active:
        assert delta_cycles[name] > 0, f"Expected non-zero cycles for {name}"
        assert delta_entries[name] > 0, f"Expected non-zero entries for {name}"

    assert delta_entries["CTRL_HALT"] > 0, "Expected HALT to be entered"
    for name in expected_quiet:
        assert delta_cycles[name] == 0, f"Expected zero cycles for {name}"
        assert delta_entries[name] == 0, f"Expected zero entries for {name}"
