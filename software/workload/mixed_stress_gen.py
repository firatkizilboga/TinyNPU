import numpy as np
import sys
import os

# Add compiler to path
sys.path.append(os.path.join(os.getcwd(), "../compiler"))
from tinynpu import TinyNPUProgram, PrecisionMode

def generate_mixed_stress_test():
    prog = TinyNPUProgram()
    sz = prog.array_size

    print(f"Generating Multi-Tile INT8 Packing Stress Test for {sz}x{sz} array...")

    # Multi-tile matmul: W(16x16) * X(16x16) + B -> Output(16x16)
    # INT8 output: 2 N-tiles pack into 1 physical tile
    # Logical: m_tiles=2, k_tiles=2, n_tiles=2
    # Physical output: 16x8 words (n_total_packed=1)
    M, K, N = sz * 2, sz * 2, sz * 2  # 16x16 x 16x16 -> 16x16

    W = np.random.randint(0, 5, size=(M, K), dtype=np.uint16)
    X = np.random.randint(0, 10, size=(K, N), dtype=np.uint16)
    B = np.random.randint(0, 50, size=(1, N), dtype=np.uint16)

    prog.declare_data("W", W)
    prog.declare_data("X", X)
    prog.declare_data("B", B)

    # Declare output with LOGICAL shape (16x16)
    prog.declare_data("PackedOutput", np.zeros((M, N), dtype=np.uint16))

    shift, multiplier = 3, 5
    prog.matmul("W", "X", "PackedOutput", bias_name="B",
                out_precision=PrecisionMode.INT8, shift=shift, multiplier=multiplier)
    prog.halt()

    # Golden model
    acc = np.matmul(W.astype(np.int64), X.astype(np.int64)) + B.astype(np.int64)
    rescaled = acc * multiplier
    rounding_offset = 1 << (shift - 1)
    shifted = (rescaled + rounding_offset) >> shift
    Y = np.clip(shifted, -128, 127).astype(np.int8)

    prog.add_expected_result("PackedOutput", Y)

    output_path = "mixed_stress.npu"
    prog.save_npu(output_path)
    print(f"Mixed stress test generated: {output_path}")

if __name__ == "__main__":
    generate_mixed_stress_test()
