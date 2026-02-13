import numpy as np
import sys
import os

# Add compiler to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "../compiler"))
from tinynpu import TinyNPUProgram

def generate_bias_test():
    prog = TinyNPUProgram()
    sz = prog.array_size
    dim = sz # 1x1 tile for simplicity
    
    print(f"Generating BIAS test for {sz}x{sz} array...")
    
    # 1. Declare Data
    A = np.random.randint(0, 10, size=(dim, dim), dtype=np.uint16)
    B = np.random.randint(0, 10, size=(dim, dim), dtype=np.uint16)
    # Bias is a vector of size N (one for each column)
    # But in TinyNPU, everything is stored as tiles. 
    # A 1xN vector is a 1xN matrix.
    bias = np.random.randint(0, 100, size=(1, dim), dtype=np.uint16)
    
    prog.declare_data("A", A)
    prog.declare_data("B", B)
    prog.declare_data("Bias", bias)
    
    # 2. Add Instructions
    # Out = A * B + Bias
    prog.matmul("A", "B", "Out", bias_name="Bias")
    prog.halt()
    
    # 3. Calculate Expected
    # A * B + Bias (Standard broadcasting in NumPy)
    acc = np.matmul(A.astype(np.int64), B.astype(np.int64)) + bias.astype(np.int64)
    expected = (acc & 0xFFFF).astype(np.uint16)
    
    prog.add_expected_result("Out", expected)
    
    # 4. Save
    output_path = "bias_test.npu"
    prog.save_npu(output_path)
    print(f"Test generated: {output_path}")

if __name__ == "__main__":
    generate_bias_test()
