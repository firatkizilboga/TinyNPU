# TinyNPU Compiler

A Python-based compiler for the TinyNPU architecture.

## Usage

```python
from tinynpu import TinyNPUProgram
import numpy as np

# Initialize Program
prog = TinyNPUProgram()

# Define Data (Matrices)
# Data is automatically packed into 64-bit words (4x 16-bit elements per row)
mat_a = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
], dtype=np.uint16)

addr_a = prog.declare_data("MatrixA", mat_a)

# Add Instructions
# M, K, N are tile counts (not raw dimensions)
prog.matmul(tile_a="MatrixA", tile_b="MatrixB", out="MatrixC", m=1, k=1, n=1)
prog.halt()

# Compile
asm = prog.to_assembly()
binary = prog.compile()

print(asm)
# binary['im_hex'] contains the hex string for Instruction Memory
# binary['ub_hex'] contains the hex string for Unified Buffer
```

## Structure

*   `tinynpu/`: The compiler package.
    *   `program.py`: Main `TinyNPUProgram` class.
    *   `isa.py`: Instruction packing logic and Opcode definitions.
*   `example_compiler.py`: A running example script.
