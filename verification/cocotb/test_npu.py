import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import numpy as np
import json
import os
import npu_driver

@cocotb.test()
async def test_npu(dut):
    # 1. Load the NPU program file
    npu_file = os.environ.get("NPU_FILE", "test.npu")
    if not os.path.exists(npu_file):
        raise FileNotFoundError(f"NPU program file not found: {npu_file}")
    
    with open(npu_file, "r") as f:
        prog = json.load(f)
    
    config = prog["config"]
    array_size = config["array_size"]
    buffer_width = config["buffer_width"]
    im_base = config["im_base"]
    
    # 2. Setup Clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1
    
    dut._log.info(f"--- Running NPU Test: {npu_file} ---")
    
    # 3. Load Unified Buffer (UB)
    dut._log.info(f"Loading UB ({len(prog['ub'])} words)...")
    for addr, word_hex in enumerate(prog["ub"]):
        word = int(word_hex, 16)
        await npu_driver.write_reg(dut, npu_driver.REG_ADDR, addr, 16)
        await npu_driver.write_reg(dut, npu_driver.REG_CMD, 0x01, 8) # WRITE_MEM
        await npu_driver.write_reg(dut, npu_driver.REG_MMVR, word, buffer_width)
        
    # 4. Load Instruction Memory (IM)
    dut._log.info(f"Loading IM ({len(prog['im'])} words)...")
    inst_width = 256
    num_chunks = inst_width // buffer_width
    
    for i, inst_hex in enumerate(prog["im"]):
        inst = int(inst_hex, 16)
        if num_chunks == 1:
            await npu_driver.write_reg(dut, npu_driver.REG_ADDR, im_base + i, 16)
            await npu_driver.write_reg(dut, npu_driver.REG_CMD, 0x01, 8)
            await npu_driver.write_reg(dut, npu_driver.REG_MMVR, inst, buffer_width)
        else:
            for c in range(num_chunks):
                chunk = (inst >> (c * buffer_width)) & ((1 << buffer_width) - 1)
                await npu_driver.write_reg(dut, npu_driver.REG_ADDR, im_base + (i * num_chunks) + c, 16)
                await npu_driver.write_reg(dut, npu_driver.REG_CMD, 0x01, 8)
                await npu_driver.write_reg(dut, npu_driver.REG_MMVR, chunk, buffer_width)

    # 5. Execute
    dut._log.info(f"Triggering execution at {im_base:#x}...")
    doorbell_addr = npu_driver.REG_MMVR + (buffer_width // 8) - 1
    await npu_driver.write_reg(dut, npu_driver.REG_ARG, im_base, 32)
    await npu_driver.write_reg(dut, npu_driver.REG_CMD, 0x03, 8) # RUN
    await npu_driver.write_reg(dut, doorbell_addr, 0, 8) # Trigger

    # 6. Wait for HALT (0xFF)
    for _ in range(100000):
        dut.host_addr.value = npu_driver.REG_STATUS
        await RisingEdge(dut.clk)
        if int(dut.host_rd_data.value) == 0xFF:
            break
    else:
        raise AssertionError("Timeout waiting for HALT")
    
    dut._log.info("Execution finished. Verifying results...")

    # 7. Verify Results
    for name, expected_list in prog["expected"].items():
        sym_info = prog["symbols"][name]
        addr = sym_info["addr"]
        shape = sym_info["shape"] # [M, N]
        role = sym_info.get("role", "C")
        precision = sym_info.get("precision", 2) # Default INT16
        
        expected = np.array(expected_list, dtype=np.int32)
        actual = np.zeros(shape, dtype=np.int32)
        
        p = 1 << (2 - precision) # Elements per 16-bit word
        bits = 16 // p
        mask = (1 << bits) - 1
        
        m_tiles = (shape[0] + array_size - 1) // array_size
        n_tiles = (shape[1] + array_size - 1) // array_size
        
        # Physical tiles count depends on role and precision
        if role == 'C':
            mt_phys = (m_tiles + p - 1) // p
            for mtp in range(mt_phys):
                for nt in range(n_tiles):
                    tile_addr = addr + (mtp * n_tiles * array_size) + (nt * array_size)
                    for i in range(array_size):
                        vec = await npu_driver.read_ub_vector(dut, tile_addr + i, array_size)
                        # Word i contains elements from multiple M-tiles
                        for lane in range(array_size):
                            word = vec[lane]
                            col_idx = nt * array_size + lane
                            for bit_idx in range(p):
                                # logical M-tile index
                                mt = mtp * p + bit_idx
                                row_idx = mt * array_size + i
                                if row_idx < shape[0] and col_idx < shape[1]:
                                    val = (word >> (bit_idx * bits)) & mask
                                    # Sign extend if needed
                                    if val & (1 << (bits - 1)):
                                        val -= (1 << bits)
                                    actual[row_idx, col_idx] = val
        else:
            # For A/B, just handle basic tiling for now (used for debugging)
            # Mixed precision for A/B is more complex, but we mainly care about C
            for mt in range(m_tiles):
                for nt in range(n_tiles):
                    tile_idx = (mt * n_tiles) + nt
                    tile_addr = addr + (tile_idx * array_size)
                    for i in range(array_size):
                        vec = await npu_driver.read_ub_vector(dut, tile_addr + i, array_size)
                        row_idx = mt * array_size + i
                        if row_idx < shape[0]:
                            start_col = nt * array_size
                            end_col = min(start_col + array_size, shape[1])
                            num_elements = end_col - start_col
                            actual[row_idx, start_col:end_col] = vec[:num_elements]
        
                if np.array_equal(actual, expected):
        
                    dut._log.info(f"✅ Symbol '{name}' matched!")
        
                else:
        
                    diff = actual != expected
        
                    num_mismatches = np.sum(diff)
        
                    dut._log.error(f"❌ Mismatch in symbol '{name}'! {num_mismatches} elements differ.")
        
                    
        
                    # Find first mismatch
        
                    idx = np.where(diff)
        
                    r, c = idx[0][0], idx[1][0]
        
                    dut._log.error(f"First mismatch at ({r}, {c}): Expected {expected[r,c]}, Actual {actual[r,c]}")
        
                    
        
                    # dut._log.error(f"Expected:\n{expected}")
        
                    # dut._log.error(f"Actual:\n{actual}")
        
                    assert False, f"Result mismatch for {name}"
        
        
        
            dut._log.info("✅ All verifications passed!")
        
        