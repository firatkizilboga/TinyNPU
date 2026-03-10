from tinynpu import TinyNPUProgram
import numpy as np
import sys
import os

# Add compiler to path
sys.path.append(os.path.join(os.getcwd(), "../compiler"))


def generate_sample_test():
    prog = TinyNPUProgram()
    sz = prog.array_size
    dim = sz * 2  # 2x2 tiles

    print(f"Generating {dim}x{dim} test for {sz}x{sz} array...")

    # 1. Declare Data
    A = np.random.randint(0, 10, size=(dim, dim), dtype=np.uint16)
    W1 = np.random.randint(0, 5, size=(dim, dim), dtype=np.uint16)
    W2 = np.random.randint(0, 5, size=(dim, dim), dtype=np.uint16)

    prog.declare_data("A", A)
    prog.declare_data("W1", W1)
    prog.declare_data("W2", W2)

    # 2. Add Instructions
    prog.matmul("W1", "A", "A1")
    prog.matmul("W2", "A1", "A2")
    prog.halt()

    # 3. Calculate Expected (Simulation with truncation)
    A1_exp = np.matmul(W1.astype(np.uint64), A.astype(np.uint64)) & 0xFFFF
    A2_exp = np.matmul(W2.astype(np.uint64), A1_exp.astype(np.uint64)) & 0xFFFF

    prog.add_expected_result("A2", A2_exp)

    # 4. Save
    output_path = "simple_chain.npu"
    prog.save_npu(output_path)
    print(f"Test generated: {output_path}")


def generate_move_test():
    prog = TinyNPUProgram()
    sz = prog.array_size
    dim = sz * 2

    print(f"Generating MOVE test for {sz}x{sz} array...")

    # 1. Declare Data
    A = np.random.randint(0, 100, size=(dim, dim), dtype=np.uint16)
    prog.declare_data("Source", A)

    # 2. Add Instructions
    # Copy Source to Target
    prog.move("Source", "Target")
    prog.halt()

    # 3. Expected result is identical
    prog.add_expected_result("Target", A)

    # 4. Save
    output_path = "move_test.npu"
    prog.save_npu(output_path)
    print(f"Test generated: {output_path}")


if __name__ == "__main__":
    generate_move_test()
    # generate_sample_test()
