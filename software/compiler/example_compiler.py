import numpy as np
from tinynpu import TinyNPUProgram

def test_compiler():
    print("Initializing TinyNPU Smart Compiler Example...")
    prog = TinyNPUProgram()

    # Inputs (Row Major naturally)
    mat_a = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ], dtype=np.uint16)
    
    # Weights (User provides as standard Row Major, Compiler will transpose!)
    mat_b = np.array([
        [5, 0, 0, 0],
        [0, 5, 0, 0],
        [0, 0, 5, 0],
        [0, 0, 0, 5]
    ], dtype=np.uint16)

    print("Declaring Data (No order specified)...")
    prog.declare_data("MatrixA", mat_a)
    prog.declare_data("MatrixB", mat_b)
    
    print("Adding Instructions...")
    # Compiler sees MatrixB is the RHS of a MATMUL
    prog.matmul(tile_a="MatrixA", tile_b="MatrixB", out="MatrixC", m=1, k=1, n=1)
    prog.halt()

    print("Compiling (Inferring Layouts)...")
    binary = prog.compile()
    asm = prog.to_assembly()

    print("\n--- Assembly (Notice inferred orders) ---")
    print(asm)
    
    print("\n--- UB Hex (Matrix B - Should be transposed) ---")
    # Matrix B is @ 0x0004
    # Original Row 0: 5, 0, 0, 0
    # If transposed, Column 0: 5, 0, 0, 0 (Wait, identity is same)
    # Let's use a non-symmetric matrix for B to be sure
    pass

def test_compiler_asymmetric():
    print("\n\nTesting with Asymmetric Weights for Transpose Verification...")
    prog = TinyNPUProgram()
    
    a = np.ones((4,4), dtype=np.uint16)
    b = np.array([
        [1, 2, 3, 4],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=np.uint16)
    
    prog.declare_data("A", a)
    prog.declare_data("B", b)
    prog.matmul("A", "B", "C", 1, 1, 1)
    
    binary = prog.compile()
    
    # B starts at index 4 (A is 4 words)
    # Original B: Row 0 is [1,2,3,4]. 
    # If Row-Major: 0x0004000300020001
    # If Column-Major (Transposed): 
    #   Col 0: [1, 0, 0, 0] -> 0x0000000000000001
    #   Col 1: [2, 0, 0, 0] -> 0x0000000000000002
    
    b_word0 = binary['ub'][4]
    b_word1 = binary['ub'][5]
    
    print(f"B Word 0 (Col 0): 0x{b_word0:016X}")
    print(f"B Word 1 (Col 1): 0x{b_word1:016X}")
    
    if b_word0 == 1 and b_word1 == 2:
        print("SUCCESS: Compiler automatically transposed Weights (B) to Column-Major!")
    else:
        print("FAILURE: Transpose logic incorrect.")

if __name__ == "__main__":
    test_compiler()
    test_compiler_asymmetric()
