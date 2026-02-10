import numpy as np
import sys
import os

# Add compiler to path
sys.path.append(os.path.join(os.getcwd(), "../../software/compiler"))
from tinynpu import TinyNPUProgram

def generate_chain_test():
    # Chain: A_out = W4 @ W3 @ W2 @ W1 @ A_in
    # Dimensions: All 16x16 for simplicity (4x4 tiles)
    
    dim = 16
    
    # Input Activation A (Random values)
    # Using small values 0,1
    A_in = np.random.randint(0, 2, size=(dim, dim), dtype=np.uint16)
    
    # Weights W1..W4
    W1 = np.random.randint(0, 2, size=(dim, dim), dtype=np.uint16)
    W2 = np.random.randint(0, 2, size=(dim, dim), dtype=np.uint16)
    W3 = np.random.randint(0, 2, size=(dim, dim), dtype=np.uint16)
    W4 = np.random.randint(0, 2, size=(dim, dim), dtype=np.uint16)
    
    # Calculate Expected Result (Python uses standard MatMul order)
    # Hardware PPU truncates to 16-bit at each step!
    # So we must simulate that truncation.
    
    # A1 = W1 @ A
    A1_exp = np.matmul(W1.astype(np.uint64), A_in.astype(np.uint64))
    A1_exp &= 0xFFFF # Truncate to 16-bit
    
    # A2 = W2 @ A1
    A2_exp = np.matmul(W2.astype(np.uint64), A1_exp)
    A2_exp &= 0xFFFF
    
    # A3 = W3 @ A2
    A3_exp = np.matmul(W3.astype(np.uint64), A2_exp)
    A3_exp &= 0xFFFF
    
    # A4 = W4 @ A3
    A4_exp = np.matmul(W4.astype(np.uint64), A3_exp)
    A4_exp &= 0xFFFF
    
    print("Generating Chain Test...")
    print(f"A_in shape: {A_in.shape}")
    print(f"Final Expected A_out:\n{A4_exp}")

    prog = TinyNPUProgram()
    
    # Declare Data
    prog.declare_data("A_in", A_in)
    prog.declare_data("W1", W1)
    prog.declare_data("W2", W2)
    prog.declare_data("W3", W3)
    prog.declare_data("W4", W4)
    
    # Hardware Execution:
    # A_next = W * A_curr
    # W is Left (Col-Major), A is Top (Row-Major)
    
    prog.matmul("W1", "A_in", "A1")
    prog.matmul("W2", "A1", "A2")
    prog.matmul("W3", "A2", "A3")
    prog.matmul("W4", "A3", "A_out")
    
    prog.halt()

    prog.generate_driver_source("chain_driver.py")
    
    # Append Verification Data
    with open("chain_driver.py", "a") as f:
        f.write("\n# Verification Data\n")
        f.write(f"EXPECTED_A_OUT = {A4_exp.tolist()}\n")
        f.write(f"ADDR_A_OUT = SYMBOLS['A_out']\n")
        f.write(f"DIM = {dim}\n")

if __name__ == "__main__":
    generate_chain_test()
