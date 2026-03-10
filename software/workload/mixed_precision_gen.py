import numpy as np
import sys
import os

# Add compiler to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "../compiler"))
from tinynpu import TinyNPUProgram, PrecisionMode

def generate_mixed_test():
    prog = TinyNPUProgram()
    sz = prog.array_size

    print(f"Generating INT8 Tile Packing test for {sz}x{sz} array...")

    # Single matmul: W(8x8) * X(8x16) + B -> Output(8x16)
    # INT8 output: 2 N-tiles pack into 1 physical tile
    # Physical output: 8x8 words, each holding 2 INT8 values
    M, K, N = sz, sz, sz * 2  # 8x8 x 8x16 -> 8x16

    W = np.random.randint(0, 3, size=(M, K), dtype=np.uint16)
    X = np.random.randint(0, 5, size=(K, N), dtype=np.uint16)
    B = np.random.randint(0, 20, size=(1, N), dtype=np.uint16)

    prog.declare_data("W", W)
    prog.declare_data("X", X)
    prog.declare_data("B", B)

    # Declare output with LOGICAL shape (8x16)
    prog.declare_data("PackedOutput", np.zeros((M, N), dtype=np.uint16))

    shift, multiplier = 1, 2
    prog.matmul("W", "X", "PackedOutput", bias_name="B",
                out_precision=PrecisionMode.INT8, shift=shift, multiplier=multiplier)
    prog.halt()

    # Golden model
    acc = np.matmul(W.astype(np.int64), X.astype(np.int64)) + B.astype(np.int64)
    rescaled = acc * multiplier
    shifted = rescaled >> shift
    Y = np.clip(shifted, -128, 127).astype(np.int8)

    prog.add_expected_result("PackedOutput", Y)

    output_path = "mixed_test.npu"
    prog.save_npu(output_path)
    print(f"Mixed precision test generated: {output_path}")

if __name__ == "__main__":
    generate_mixed_test()
