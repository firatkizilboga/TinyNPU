import numpy as np
import sys
import os

# Add compiler to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../software/compiler'))
from tinynpu import TinyNPUProgram

def generate():
    prog = TinyNPUProgram()
    
    # 4x4 Matrices
    mat_a = np.array([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [1, 1, 1, 1],
        [2, 2, 2, 2]
    ], dtype=np.uint16)
    
    mat_b = np.eye(4, dtype=np.uint16) * 3
    
    prog.declare_data("A", mat_a)
    prog.declare_data("B", mat_b)
    
    # C = A * B. Results should be A * 3
    prog.matmul("A", "B", "C", 1, 1, 1)
    prog.halt()
    
    binary = prog.compile()
    
    # Write Hex Files (Preloaded Case)
    with open("compiler_im.hex", "w") as f:
        f.write(binary['im_hex'])
    
    with open("compiler_ub.hex", "w") as f:
        f.write(binary['ub_hex'])
        
    # Generate Driver Sequence (Host Case)
    messages = prog.generate_host_driver()
    with open("compiler_driver.py", "w") as f:
        f.write("DRIVER_MESSAGES = [\n")
        for reg, val in messages:
            f.write(f"    ({int(reg)}, {int(val)}),\n")
        f.write("]\n")
        
    print("Generated compiler_im.hex, compiler_ub.hex, and compiler_driver.py")

if __name__ == "__main__":
    generate()