from tinynpu import TinyNPUProgram
import numpy as np
import sys
import os

# Add compiler to path
sys.path.append(os.path.join(os.getcwd(), "../compiler"))


def generate_complex_chain():
    prog = TinyNPUProgram()
    sz = prog.array_size
    dim = sz * 2  # 32x32 for 16x16 array

    print(f"Generating Ping-Pong Chain test for {sz}x{sz} array...")

    # 1. Declare Data
    X = np.random.randint(0, 5, size=(dim, dim), dtype=np.uint16)
    W1 = np.random.randint(0, 3, size=(dim, dim), dtype=np.uint16)
    W2 = np.random.randint(0, 3, size=(dim, dim), dtype=np.uint16)
    W3 = np.random.randint(0, 3, size=(dim, dim), dtype=np.uint16)

    # Declare Data
    # Ping starts as Input X
    prog.declare_data("Ping", X)
    prog.declare_data("W1", W1)
    prog.declare_data("W2", W2)
    prog.declare_data("W3", W3)

    # 2. Execution Chain
    # Strategy: Use Output (Row-Major) as Input B (Row-Major) for next layer

    prog.matmul("W1", "Ping", "Pong")

    prog.matmul("W2", "Pong", "Ping")

    prog.matmul("W3", "Ping", "Pong")

    prog.halt()

    Pong_exp = np.matmul(W1.astype(np.uint64), X.astype(np.uint64)) & 0xFFFF
    Ping_exp = np.matmul(W2.astype(np.uint64),
                         Pong_exp.astype(np.uint64)) & 0xFFFF
    Pong_final_exp = np.matmul(
        W3.astype(np.uint64), Ping_exp.astype(np.uint64)) & 0xFFFF

    prog.add_expected_result("Pong", Pong_final_exp)

    output_path = "complex_chain.npu"
    prog.save_npu(output_path)
    print(f"Test generated: {output_path}")


if __name__ == "__main__":
    generate_complex_chain()
