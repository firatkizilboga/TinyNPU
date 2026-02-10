import numpy as np
import sys
import os

# Add compiler to path
sys.path.append(os.path.join(os.getcwd(), "../../software/compiler"))
from tinynpu import TinyNPUProgram

def generate_multi_test():
    # Matrices: 
    # A(4,4) * B(4,4) = D(4,4)
    # C(4,4) * D(4,4) = E(4,4)
    
    A = np.random.randint(0, 5, size=(4,4), dtype=np.uint16)
    B = np.random.randint(0, 5, size=(4,4), dtype=np.uint16)
    C = np.random.randint(0, 5, size=(4,4), dtype=np.uint16)
    
    D_expected = np.matmul(A.astype(np.uint64), B.astype(np.uint64))
    # Match Hardware: E = C * D
    E_expected = np.matmul(C.astype(np.uint64), D_expected)
    
    print(f"Matrix A:\n{A}")
    print(f"Matrix B:\n{B}")
    print(f"Matrix C:\n{C}")
    print(f"D (A*B) Expected:\n{D_expected}")
    print(f"E (C*D) Expected:\n{E_expected}")

    prog = TinyNPUProgram()
    prog.declare_data("A", A)
    prog.declare_data("B", B)
    prog.declare_data("C", C)
    
    # D = A * B
    prog.matmul("A", "B", "D")
    
    # E = C * D
    # C is Left (Column-Major), D is Top (Row-Major from PPU)
    prog.matmul("C", "D", "E")
    
    prog.halt()

    prog.generate_driver_source("multi_driver.py")
    
    # Metadata for verification
    with open("multi_driver.py", "a") as f:
        f.write("\n# Verification Data\n")
        f.write(f"EXPECTED_D = {D_expected.tolist()}\n")
        f.write(f"EXPECTED_E = {E_expected.tolist()}\n")
        f.write(f"ADDR_D = SYMBOLS['D']\n")
        f.write(f"ADDR_E = SYMBOLS['E']\n")

if __name__ == "__main__":
    generate_multi_test()
