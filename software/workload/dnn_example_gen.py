import numpy as np
import sys
import os

# Add compiler to path
sys.path.append(os.path.join(os.getcwd(), "../compiler"))
from tinynpu import TinyNPUProgram

def generate_dnn_example():
    prog = TinyNPUProgram()
    sz = prog.array_size
    dim = sz * 4 # 32x32 for 8x8 array (4x4 tiles)
    
    print(f"Generating 3-Layer DNN example for {sz}x{sz} array...")
    
    # 1. Declare Data
    # Using small random integers to prevent massive accumulation overflow
    X  = np.random.randint(0, 4, size=(dim, dim), dtype=np.uint16)
    W1 = np.random.randint(0, 3, size=(dim, dim), dtype=np.uint16)
    B1 = np.random.randint(0, 10, size=(1, dim), dtype=np.uint16)
    
    W2 = np.random.randint(0, 3, size=(dim, dim), dtype=np.uint16)
    B2 = np.random.randint(0, 10, size=(1, dim), dtype=np.uint16)
    
    W3 = np.random.randint(0, 3, size=(dim, dim), dtype=np.uint16)
    B3 = np.random.randint(0, 10, size=(1, dim), dtype=np.uint16)
    
    # Initial Activation starts in "Act_Ping"
    # To chain results Row-Major -> Row-Major, Act_Ping must be Role B
    prog.declare_data("Act_Ping", X) 
    prog.declare_data("W1", W1)
    prog.declare_data("B1", B1)
    prog.declare_data("W2", W2)
    prog.declare_data("B2", B2)
    prog.declare_data("W3", W3)
    prog.declare_data("B3", B3)
    
    # 2. Execution Chain (Ping-Pong Memory Reuse)
    
    # Layer 1: Pong = W1 * Ping + B1
    prog.matmul("W1", "Act_Ping", "Act_Pong", bias_name="B1")
    
    # Layer 2: Ping = W2 * Pong + B2
    prog.matmul("W2", "Act_Pong", "Act_Ping", bias_name="B2")
    
    # Layer 3: Pong = W3 * Ping + B3
    prog.matmul("W3", "Act_Ping", "Act_Pong", bias_name="B3")
    
    # Final Move: Result = Act_Pong
    prog.move("Act_Pong", "Result")
    
    prog.halt()
    
    # 3. Calculate Golden Model (NumPy)
    # Hardware performs: (A_col_major * B_row_major) + Bias_row_major
    # Truncating to 16-bit at every layer.
    
    def layer_op(w, x, b):
        return (np.matmul(w.astype(np.uint64), x.astype(np.uint64)) + b) & 0xFFFF
    
    Y1 = layer_op(W1, X, B1)
    Y2 = layer_op(W2, Y1, B2)
    Y3 = layer_op(W3, Y2, B3)
    
    prog.add_expected_result("Result", Y3)
    
    # 4. Save
    output_path = "dnn_example.npu"
    prog.save_npu(output_path)
    print(f"DNN test generated: {output_path}")

if __name__ == "__main__":
    generate_dnn_example()
