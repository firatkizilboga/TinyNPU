import numpy as np
import sys
import os

# Add compiler to path
sys.path.append(os.path.join(os.getcwd(), "software/compiler"))
from tinynpu import TinyNPUProgram, PrecisionMode

def generate_relu_test():
    prog = TinyNPUProgram()
    sz = prog.array_size

    print(f"Generating ReLU Activation Test for {sz}x{sz} array...")

    # Single matmul: W(8x8) * X(8x8) + B -> Output(8x8)
    M, K, N = sz, sz, sz

    # Use random data with negative values to test ReLU
    W = np.random.randint(-10, 10, size=(M, K)).astype(np.int16)
    X = np.random.randint(-10, 10, size=(K, N)).astype(np.int16)
    B = np.random.randint(-20, 20, size=(1, N)).astype(np.int16)

    prog.declare_data("W", W, role='A')
    prog.declare_data("X", X, role='B')
    prog.declare_data("B", B, role='BIAS')
    prog.declare_data("Output", np.zeros((M, N), dtype=np.int16), role='C')

    shift, multiplier = 0, 1
    # activation=1 triggers ReLU
    prog.matmul("W", "X", "Output", bias_name="B",
                shift=shift, multiplier=multiplier, activation=1)
    prog.halt()

    # Golden model
    acc = np.matmul(W.astype(np.int64), X.astype(np.int64)) + B.astype(np.int64)
    rescaled = acc * multiplier
    shifted = rescaled >> shift
    
    # ReLU: max(0, x)
    relu_out = np.maximum(0, shifted)
    
    # Saturation to INT16
    Y = np.clip(relu_out, -32768, 32767).astype(np.int16)

    prog.add_expected_result("Output", Y)

    output_path = "relu_test.npu"
    prog.save_npu(output_path)
    print(f"ReLU test generated: {output_path}")

if __name__ == "__main__":
    generate_relu_test()
