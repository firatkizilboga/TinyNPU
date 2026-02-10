import numpy as np
import sys
import os

# Add compiler to path
sys.path.append(os.path.join(os.getcwd(), "../../software/compiler"))
from tinynpu import TinyNPUProgram

def generate_stress_test():
    M, K, N = 64, 16, 32
    m_tiles = (M + 3) // 4
    k_tiles = (K + 3) // 4
    n_tiles = (N + 3) // 4
    
    print(f"Generating Stress Test: {M}x{K} x {K}x{N} ({m_tiles}x{k_tiles}x{n_tiles} tiles)")

    mat_a = np.random.randint(0, 10, size=(M, K), dtype=np.uint16)
    mat_b = np.random.randint(0, 10, size=(K, N), dtype=np.uint16)
    expected_c = np.matmul(mat_a.astype(np.uint64), mat_b.astype(np.uint64))

    prog = TinyNPUProgram()
    prog.declare_data("A", mat_a)
    prog.declare_data("B", mat_b)
    prog.matmul(tile_a="A", tile_b="B", out="C", m=m_tiles, k=k_tiles, n=n_tiles)
    prog.halt()

    # Generate the driver using the new compiler method
    prog.generate_driver_source("stress_driver.py")
    
    # Append the test-specific metadata (Expected Results)
    # The compiler generates the generic driver part, we add the verification data.
    with open("stress_driver.py", "a") as f:
        f.write(f"\n# Test Specific Data\n")
        f.write(f"EXPECTED_C = {expected_c.tolist()}\n")
        f.write(f"C_BASE = SYMBOLS['C']\n") # Use the symbol map
        f.write(f"M_TILES = {m_tiles}\n")
        f.write(f"N_TILES = {n_tiles}\n")
        f.write(f"M_RAW = {M}\n")
        f.write(f"K_RAW = {K}\n")
        f.write(f"N_RAW = {N}\n")

if __name__ == "__main__": generate_stress_test()
