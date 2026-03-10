import json
import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

import npu_driver
from software.workload.mnist_npu_compiler import (
    compile_mnist_layer,
    get_im2col_matrix,
    prepare_activation_for_hw,
)


def _unpack_output(dut, out_sym):
    actual = np.zeros(out_sym.shape, dtype=np.int32)
    p = 1 << (2 - out_sym.precision)
    bits = 16 // p
    array_size = 8
    mask = (1 << bits) - 1
    m_tiles = (out_sym.shape[0] + 7) // 8
    n_tiles = (out_sym.shape[1] + 7) // 8
    mt_phys = (m_tiles + p - 1) // p

    async def _read():
        for mtp in range(mt_phys):
            for nt in range(n_tiles):
                tile_addr = out_sym.addr + (mtp * n_tiles * array_size) + (nt * array_size)
                for row_in_tile in range(array_size):
                    vec = await npu_driver.read_ub_vector(dut, tile_addr + row_in_tile, array_size)
                    for lane in range(array_size):
                        word = vec[lane]
                        col_idx = nt * array_size + lane
                        for bit_idx in range(p):
                            mt = mtp * p + bit_idx
                            row_idx = mt * array_size + row_in_tile
                            if row_idx < out_sym.shape[0] and col_idx < out_sym.shape[1]:
                                val = (word >> (bit_idx * bits)) & mask
                                if val & (1 << (bits - 1)):
                                    val -= (1 << bits)
                                actual[row_idx, col_idx] = val
        return actual

    return _read()


@cocotb.test()
async def test_mnist_full_chain_old_compare(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1

    export_dir = os.path.join(project_root, "mnist_mixed_export")
    with open(os.path.join(export_dir, "manifest.json")) as f:
        manifest = json.load(f)

    rng = np.random.default_rng(7)
    current_data = rng.integers(0, 256, size=(28, 28, 1), dtype=np.int16)
    final_vector = None

    for layer in manifest["layers"]:
        name = layer["name"]
        if layer["type"] == "conv2d":
            input_matrix = get_im2col_matrix(
                current_data,
                layer["kernel_size"],
                layer["kernel_size"],
                layer["stride"],
                layer["padding"],
            )
        else:
            gap = np.mean(current_data, axis=(0, 1))
            input_matrix = gap.reshape(-1, 1).astype(np.int16)

        prog = compile_mnist_layer(name, layer, input_matrix, export_dir=export_dir)
        binary = prog.last_compiled

        for addr, word in enumerate(binary["ub"]):
            await npu_driver.write_reg(dut, npu_driver.REG_ADDR, addr, 16)
            await npu_driver.write_reg(dut, npu_driver.REG_CMD, 0x01, 8)
            await npu_driver.write_reg(dut, npu_driver.REG_MMVR, word, 128)

        im_base = prog.im_base_addr
        for addr, inst in enumerate(binary["im"]):
            await npu_driver.write_reg(dut, npu_driver.REG_ADDR, im_base + addr * 2, 16)
            await npu_driver.write_reg(dut, npu_driver.REG_CMD, 0x01, 8)
            await npu_driver.write_reg(dut, npu_driver.REG_MMVR, inst & ((1 << 128) - 1), 128)
            await npu_driver.write_reg(dut, npu_driver.REG_ADDR, im_base + addr * 2 + 1, 16)
            await npu_driver.write_reg(dut, npu_driver.REG_CMD, 0x01, 8)
            await npu_driver.write_reg(dut, npu_driver.REG_MMVR, inst >> 128, 128)

        bias_sym = prog.symbols.get("Bias")
        if bias_sym is not None:
            for off in range(bias_sym.word_count):
                word = binary["ub"][bias_sym.addr + off]
                await npu_driver.write_reg(dut, npu_driver.REG_ADDR, bias_sym.addr + off, 16)
                await npu_driver.write_reg(dut, npu_driver.REG_CMD, 0x01, 8)
                await npu_driver.write_reg(dut, npu_driver.REG_MMVR, word, 128)

        await npu_driver.write_reg(dut, npu_driver.REG_ARG, im_base, 32)
        await npu_driver.write_reg(dut, npu_driver.REG_CMD, 0x03, 8)
        await npu_driver.write_reg(dut, npu_driver.REG_MMVR + (128 // 8) - 1, 0, 8)

        for _ in range(1000000):
            dut.host_addr.value = npu_driver.REG_STATUS
            await RisingEdge(dut.clk)
            await RisingEdge(dut.clk)
            if int(dut.host_rd_data.value) == 0xFF:
                break
        else:
            raise AssertionError(f"Layer {name} timed out")

        actual = await _unpack_output(dut, prog.symbols["Output"])
        if layer["type"] == "conv2d":
            out_h = ((current_data.shape[0] + 2 * layer["padding"] - layer["kernel_size"]) // layer["stride"]) + 1
            out_w = ((current_data.shape[1] + 2 * layer["padding"] - layer["kernel_size"]) // layer["stride"]) + 1
            current_data = actual.reshape(out_h, out_w, layer["out_channels"])
        else:
            final_vector = actual.reshape(-1)

    assert final_vector is not None
    dut._log.info(f"OLD RTL final vector: {final_vector.tolist()}")
    dut._log.info(f"OLD RTL predicted class: {int(np.argmax(final_vector))}")
