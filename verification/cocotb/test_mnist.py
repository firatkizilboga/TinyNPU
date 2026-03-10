import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
import json
import os
import sys

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

import npu_driver
from software.workload.mnist_npu_compiler import (
    compile_mnist_layer,
    get_im2col_matrix,
    prepare_activation_for_hw,
)
from verification.golden_model.model import GoldenModel

@cocotb.test()
async def test_mnist_full(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    
    gm = GoldenModel()
    export_dir = "../../npu_export"
    with open(os.path.join(export_dir, 'manifest.json'), 'r') as f:
        manifest = json.load(f)
    
    # Image in [0, 255]
    img = np.load("../../mnist_sample.npy").transpose(1, 2, 0).astype(np.int16)
    current_data = img
    
    dut._log.info("--- Starting Bit-Perfect MNIST NPU Verification ---")

    reload_bias_before_run = os.getenv("TINYNPU_DEFENSIVE_BIAS_RELOAD", "0") == "1"

    for i, layer in enumerate(manifest['layers']):
        name = layer['name']
        ltype = layer['type']
        dut._log.info(f"Processing Layer {i+1}: {name} ({ltype})")
        
        if ltype == 'conv2d':
            input_matrix = get_im2col_matrix(current_data, layer['kernel_size'], layer['kernel_size'], layer['stride'], layer['padding'])
            input_hw = prepare_activation_for_hw(input_matrix, layer.get('a_bits', 16))
            w = np.load(os.path.join(export_dir, f"{name}_weights_gemm.npy"))
            b = np.load(os.path.join(export_dir, f"{name}_bias.npy"))
            conv_out_prec = f"INT{int(layer.get('a_bits', 16))}"
            # Golden logic
            expected = gm.matmul(input_hw, w.T, b,
                                multiplier=layer['M0'], shift=layer['shift'], 
                                activation=True, out_precision=conv_out_prec)
        else:
            gap = np.mean(current_data, axis=(0, 1))
            input_matrix = gap.reshape(-1, 1).astype(np.int16)
            input_hw = prepare_activation_for_hw(input_matrix, layer.get('a_bits', 16))
            w = np.load(os.path.join(export_dir, f"{name}_weights.npy"))
            b = np.load(os.path.join(export_dir, f"{name}_bias.npy"))
            # Golden logic
            expected = gm.matmul(w, input_hw, b, 
                                multiplier=layer['M0'], shift=layer['shift'], 
                                activation=False, out_precision='INT16')
            
        prog = compile_mnist_layer(name, layer, input_matrix, export_dir=export_dir)
        binary = prog.last_compiled
        for sname, sym in prog.symbols.items():
            dut._log.info(f"Symbol {sname}: addr={sym.addr}, shape={sym.shape}, role={sym.storage_role}")
        
        # Load UB
        for addr, word in enumerate(binary['ub']):
            await npu_driver.write_reg(dut, npu_driver.REG_ADDR, addr, 16)
            await npu_driver.write_reg(dut, npu_driver.REG_CMD, 0x01, 8)
            await npu_driver.write_reg(dut, npu_driver.REG_MMVR, word, 128)
            
        im_base = prog.im_base_addr
        for addr, inst in enumerate(binary['im']):
            await npu_driver.write_reg(dut, npu_driver.REG_ADDR, im_base + addr*2, 16)
            await npu_driver.write_reg(dut, npu_driver.REG_CMD, 0x01, 8)
            await npu_driver.write_reg(dut, npu_driver.REG_MMVR, inst & ((1 << 128) - 1), 128)
            await npu_driver.write_reg(dut, npu_driver.REG_ADDR, im_base + addr*2 + 1, 16)
            await npu_driver.write_reg(dut, npu_driver.REG_CMD, 0x01, 8)
            await npu_driver.write_reg(dut, npu_driver.REG_MMVR, inst >> 128, 128)

        # Historical mitigation for UB word-0 instability.
        # Default is OFF; set TINYNPU_DEFENSIVE_BIAS_RELOAD=1 to re-enable.
        if reload_bias_before_run:
            bias_sym = prog.symbols.get("Bias")
            if bias_sym is not None:
                for off in range(bias_sym.word_count):
                    word = binary["ub"][bias_sym.addr + off]
                    await npu_driver.write_reg(dut, npu_driver.REG_ADDR, bias_sym.addr + off, 16)
                    await npu_driver.write_reg(dut, npu_driver.REG_CMD, 0x01, 8)
                    await npu_driver.write_reg(dut, npu_driver.REG_MMVR, word, 128)

        # Run
        await npu_driver.write_reg(dut, npu_driver.REG_ARG, im_base, 32)
        await npu_driver.write_reg(dut, npu_driver.REG_CMD, 0x03, 8)
        await npu_driver.write_reg(dut, npu_driver.REG_MMVR + (128 // 8) - 1, 0, 8)
        
        for _ in range(1000000):
            dut.host_addr.value = npu_driver.REG_STATUS
            await RisingEdge(dut.clk)
            await RisingEdge(dut.clk) # Wait for data
            if int(dut.host_rd_data.value) == 0xFF: break
        else: raise AssertionError(f"Layer {name} timed out")
        
        # DUMP UB for debug
        ub_dump = []
        for addr in range(1024):
            vec = await npu_driver.read_ub_vector(dut, addr, 8)
            word = 0
            for i, v in enumerate(vec): word |= (v & 0xFFFF) << (i*16)
            ub_dump.append(word)
        with open("ub_dump_rtl.hex", "w") as f:
            for w in ub_dump: f.write(f"{w:032x}\n")
        
        # Unpack Output
        out_sym = prog.symbols["Output"]
        actual = np.zeros(out_sym.shape, dtype=np.int32)
        p = 1 << (2 - out_sym.precision)
        bits, array_size = 16 // p, 8
        mask = (1 << bits) - 1
        m_tiles, n_tiles = (out_sym.shape[0] + 7) // 8, (out_sym.shape[1] + 7) // 8
        mt_phys = (m_tiles + p - 1) // p
        
        for mtp in range(mt_phys):
            for nt in range(n_tiles):
                tile_addr = out_sym.addr + (mtp * n_tiles * array_size) + (nt * array_size)
                for i in range(array_size):
                    vec = await npu_driver.read_ub_vector(dut, tile_addr + i, array_size)
                    for lane in range(array_size):
                        word = vec[lane]
                        col_idx = nt * array_size + lane
                        for bit_idx in range(p):
                            mt = mtp * p + bit_idx
                            row_idx = mt * array_size + i
                            if row_idx < out_sym.shape[0] and col_idx < out_sym.shape[1]:
                                val = (word >> (bit_idx * bits)) & mask
                                if val & (1 << (bits - 1)): val -= (1 << bits)
                                actual[row_idx, col_idx] = val
        
        if np.array_equal(actual, expected):
            dut._log.info(f"✅ Layer {name} is BIT-PERFECT!")
        else:
            diff = actual != expected
            dut._log.error(f"❌ Layer {name} mismatch! {np.sum(diff)} pixels differ.")
            idx = np.where(diff); r, c = idx[0][0], idx[1][0]
            dut._log.error(f"First mismatch at ({r},{c}): Golden {expected[r,c]}, NPU {actual[r,c]}")
            assert False, f"Hardware divergence at layer {name}"

        # Update data for next layer
        if ltype == 'conv2d':
            OH = OW = int(np.sqrt(actual.shape[0]))
            current_data = actual.reshape(OH, OW, layer['out_channels'])
        else:
            predictions = actual.flatten()
            predicted_class = np.argmax(predictions)
            dut._log.info(f"--- Final Prediction: {predicted_class} ---")
            break
