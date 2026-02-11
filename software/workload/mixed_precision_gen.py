import numpy as np
import sys
import os

# Add compiler to path
sys.path.append(os.path.join(os.getcwd(), "../compiler"))
from tinynpu import TinyNPUProgram, PrecisionMode

def generate_mixed_test():
    prog = TinyNPUProgram()
    sz = prog.array_size
    dim = sz # 8x8 for 8x8 array
    
    print(f"Generating Mixed Precision Packing + Bias test for {sz}x{sz} array...")
    
    # 1. Declare Data
    X  = np.random.randint(0, 5, size=(dim, dim), dtype=np.uint16)
    W1 = np.random.randint(0, 3, size=(dim, dim), dtype=np.uint16)
    W2 = np.random.randint(0, 3, size=(dim, dim), dtype=np.uint16)
    B1 = np.random.randint(0, 20, size=(1, dim), dtype=np.uint16)
    B2 = np.random.randint(0, 20, size=(1, dim), dtype=np.uint16)
    
    prog.declare_data("Input", X) 
    prog.declare_data("W1", W1)
    prog.declare_data("W2", W2)
    prog.declare_data("B1", B1)
    prog.declare_data("B2", B2)
    
    # Declare Output space (Role 'C')
    prog.declare_data("PackedOutput", np.zeros((dim, dim), dtype=np.uint16))
    
    # 2. Add Instructions
    
    # Layer 1: INT16 -> INT8 (Low Byte) + Bias
    prog.matmul("W1", "Input", "PackedOutput", bias_name="B1", 
                precision=PrecisionMode.INT8, write_offset=0, multiplier=2, shift=1)
    
    # Layer 2: INT16 -> INT8 (High Byte) + Bias
    prog.matmul("W2", "Input", "PackedOutput", bias_name="B2", 
                precision=PrecisionMode.INT8, write_offset=1, multiplier=16384, shift=14)
    
    prog.halt()
    
    # 3. Calculate Golden Model (NumPy)
    def layer_op(w, x, b, m, s):
        # High precision bias addition
        acc = np.matmul(w.astype(np.int64), x.astype(np.int64)) + b.astype(np.int64)
        rescaled = acc * m
        shifted = rescaled >> s
        # Saturate to 8-bit signed range
        return np.clip(shifted, -128, 127).astype(np.int8)
    
    Y1 = layer_op(W1, X, B1, 2, 1)
    Y2 = layer_op(W2, X, B2, 16384, 14)
    
    # Construct expected packed buffer (INT16 words)
    expected = (Y2.astype(np.uint16) << 8) | (Y1.astype(np.uint16) & 0xFF)
    
    prog.add_expected_result("PackedOutput", expected)
    
    # 4. Save
    output_path = "mixed_test.npu"
    prog.save_npu(output_path)
    print(f"Mixed precision test generated: {output_path}")

if __name__ == "__main__":
    generate_mixed_test()