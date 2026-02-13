import numpy as np
import sys
import os

# Add compiler to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../compiler"))
from tinynpu import TinyNPUProgram, PrecisionMode

def saturate_to_uint16(x):
    """Simulates HW PPU: Saturate to INT16 then view as UINT16"""
    clipped = np.clip(x, -32768, 32767).astype(np.int16)
    return clipped.view(np.uint16)

def generate_int16_chain_test():
    prog = TinyNPUProgram()
    sz = 16 
    
    print(f"Generating Controlled INT16 Chain Test (3 steps) for {sz}x{sz}...")

    # Step 1: A * W1 + B1 -> C1
    # Inputs: [-20, 20], Weights: [-5, 5]
    # Max dot product: 20 * 5 * 16 = 1600
    A = np.random.randint(-20, 20, size=(sz, sz)).astype(np.int16)
    W1 = np.random.randint(-5, 5, size=(sz, sz)).astype(np.int16)
    B1 = np.random.randint(-100, 100, size=(1, sz)).astype(np.int16)
    
    prog.declare_data("A", A)
    prog.declare_data("W1", W1)
    prog.declare_data("B1", B1)
    prog.declare_data("C1", np.zeros((sz, sz), dtype=np.uint16))
    
    prog.matmul("A", "W1", "C1", "B1", shift=0, multiplier=1, 
                in_precision=PrecisionMode.INT16, out_precision=PrecisionMode.INT16)

    raw_matmul = np.matmul(A.astype(np.int32), W1.astype(np.int32))
    print("Step 1 Raw matmul Col 0 (Rows 0-15):")
    print(raw_matmul[:, 0])
    C1_acc = raw_matmul + B1
    C1_logical = np.clip(C1_acc, -32768, 32767).astype(np.int16)
    print("C1_logical Tile(0,0) first column (Rows 0-7):")
    print(C1_logical[0:8, 0])
    
    # Step 2: C1 * W2 + B2 -> C2
    # C1 is around [-1700, 1700]. Weights: [-4, 4].
    # Max dot product: 1700 * 4 * 16 = 108,800.
    # We use shift=2 to keep it around 27,200.
    W2 = np.random.randint(-4, 4, size=(sz, sz)).astype(np.int16)
    B2 = np.random.randint(-500, 500, size=(1, sz)).astype(np.int16)
    prog.declare_data("W2", W2)
    prog.declare_data("B2", B2)
    prog.declare_data("C2", np.zeros((sz, sz), dtype=np.uint16))
    
    prog.matmul("C1", "W2", "C2", "B2", shift=2, multiplier=1,
                in_precision=PrecisionMode.INT16, out_precision=PrecisionMode.INT16)
                
    C2_acc = np.matmul(C1_logical.astype(np.int32), W2.astype(np.int32)) + B2
    C2_logical = np.clip(C2_acc >> 2, -32768, 32767).astype(np.int16)

    # Step 3: C2 * W3 + B3 -> Final
    # C2 is around [-32768, 32767]. Weights: [-1, 1] (sparse-ish)
    W3 = np.random.randint(-1, 2, size=(sz, sz)).astype(np.int16)
    B3 = np.random.randint(-1000, 1000, size=(1, sz)).astype(np.int16)
    prog.declare_data("W3", W3)
    prog.declare_data("B3", B3)
    prog.declare_data("Final", np.zeros((sz, sz), dtype=np.uint16))
    
    prog.matmul("C2", "W3", "Final", "B3", shift=4, multiplier=1,
                in_precision=PrecisionMode.INT16, out_precision=PrecisionMode.INT16)
                
    Final_acc = np.matmul(C2_logical.astype(np.int32), W3.astype(np.int32)) + B3
    Final_golden = saturate_to_uint16(Final_acc >> 4)
    
    prog.halt()
    prog.add_expected_result("Final", Final_golden)
    
    output_path = "int16_chain.npu"
    prog.save_npu(output_path)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    generate_int16_chain_test()
