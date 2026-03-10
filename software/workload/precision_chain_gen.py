
from tinynpu import TinyNPUProgram, PrecisionMode
import numpy as np
import sys
import os

# Add compiler to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "../compiler"))


def generate_precision_chain_test():
    prog = TinyNPUProgram()
    sz = prog.array_size
    dim = sz * 2  # 16x16 for 8x8 array

    print(
        f"Generating Precision Chain Test (INT8->16->4->16->8->4->8) for {dim}x{dim} logical...")

    # Initial Data (Logical)
    X = np.random.randint(-128, 127, size=(dim, dim)).astype(np.int16)
    W1 = np.random.randint(-128, 127, size=(dim, dim)).astype(np.int16)
    B1 = np.random.randint(-100, 100, size=(1, dim)).astype(np.int16)

    prog.declare_data("X", X)
    prog.declare_data("W1", W1)
    prog.declare_data("B1", B1)

    prog.declare_data("C1", np.zeros((dim, dim), dtype=np.uint16))

    # Step 1: INT8 -> INT16
    prog.matmul("X", "W1", "C1", "B1", shift=0, multiplier=1,
                in_precision=PrecisionMode.INT8, out_precision=PrecisionMode.INT16)

    # Step 2: INT16 -> INT4
    W2 = np.random.randint(-1000, 1000, size=(dim, dim)).astype(np.int16)
    B2 = np.random.randint(-100, 100, size=(1, dim)).astype(np.int16)
    prog.declare_data("W2", W2)
    prog.declare_data("B2", B2)
    prog.declare_data("C2", np.zeros((dim, dim), dtype=np.uint16))
    prog.matmul("C1", "W2", "C2", "B2", shift=4, multiplier=1,
                in_precision=PrecisionMode.INT16, out_precision=PrecisionMode.INT4)

    # Step 3: INT4 -> INT8
    W3 = np.random.randint(-8, 7, size=(dim, dim)).astype(np.int16)
    B3 = np.random.randint(-50, 50, size=(1, dim)).astype(np.int16)
    prog.declare_data("W3", W3)
    prog.declare_data("B3", B3)
    prog.declare_data("Final", np.zeros((dim, dim), dtype=np.uint16))
    prog.matmul("C2", "W3", "Final", "B3", shift=2, multiplier=1,
                in_precision=PrecisionMode.INT4, out_precision=PrecisionMode.INT8)

    prog.halt()

    # --- Golden Model ---
    def layer_op(a, w, b, shift, mult, out_prec):
        acc = np.matmul(a.astype(np.int64), w.astype(np.int64)) + b.astype(np.int64)
        rescaled = acc * mult
        if shift > 0:
            rounding_offset = 1 << (shift - 1)
            val = (rescaled + rounding_offset) >> shift
        else:
            val = rescaled
        
        if out_prec == PrecisionMode.INT16:
            return np.clip(val, -32768, 32767).astype(np.int16)
        elif out_prec == PrecisionMode.INT8:
            val = np.clip(val, -128, 127)
            return val.astype(np.int16)
        elif out_prec == PrecisionMode.INT4:
            val = np.clip(val, -8, 7)
            return val.astype(np.int16)
        return val.astype(np.int16)
    

    Y1 = layer_op(X, W1, B1, 0, 1, PrecisionMode.INT16)
    Y2 = layer_op(Y1, W2, B2, 4, 1, PrecisionMode.INT4)
    Y3 = layer_op(Y2, W3, B3, 2, 1, PrecisionMode.INT8)

    # Verification: Testbench expects a LOGICAL [M, N] matrix for Role C.
    # We should NOT manually pack it for Role C, as the testbench handles
    # reading from UB and comparing against this logical 2D array.
    # Note: Mixed Precision Test (mixed_precision_gen.py) was special because
    # it manually declared a "PackedOutput" symbol.

    prog.add_expected_result("Final", Y3)

    output_path = "precision_chain.npu"
    prog.save_npu(output_path)
    print(f"Test generated: {output_path}")


if __name__ == "__main__":
    generate_precision_chain_test()
