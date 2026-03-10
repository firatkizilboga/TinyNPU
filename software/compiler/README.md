# TinyNPU Compiler

A Python-based compiler for the TinyNPU architecture.

## New Segmented JIT Path

The repository now also contains a new experimental compiler/runtime path under `tinynpu_jit/`.

This path is designed around:
- PyTorch/FX as the frontend
- segmented execution plans instead of whole-program lowering
- explicit `HostOp(...)` boundaries for unsupported ops
- compiler-owned expected tensors for runtime verification
- `mark_for_verify(...)` as a host-visible verification boundary

Current scope:
- TinyNPU lowering: explicit `matmul`, exported `linear`, and exported `conv2d` lowered through host-side `im2col`
- PyTorch quant boundary support: `torch.ao.nn.quantized.Quantize` / `DeQuantize`, plus `QuantStub` / `DeQuantStub` when explicit qparams are attached
- host-emulation runtime for compile/run/verify validation
- async simulator runtime entrypoint that targets the existing cocotb driver path
- deterministic compile behavior with explicit failure if a segment exceeds UB capacity

Key entry points:

```python
from tinynpu_jit import compile_module, mark_for_verify, quantize_for_npu

# compile_module(...) requires torch to be installed locally
# mark_for_verify(...) is intended to let users request runtime-visible
# verification points in the execution plan. In v1 it forces a segment
# boundary so the tensor can be materialized and checked on the host.
# quantize_for_npu(...) is the current explicit source of truth for
# HostOp -> NpuSegment re-entry quantization.
# npu_matmul(...) and im2col_for_npu(...) are transitional exported-model
# helpers, not the intended final public frontend for ordinary PyTorch models.
```

Examples:
- `example_segmented_jit.py`: manual execution-plan construction
- `example_torch_jit.py`: traced PyTorch module with `mark_for_verify(...)`
- `software/workload/mnist_npu_compiler.py::compile_mnist_layer_jit(...)`: exported MNIST `linear` and `conv2d` layer examples on the new PyTorch JIT path
- `software/workload/jit_test_gen.py::build_simple_chain_artifact(...)`: migrated old simple-chain workload on the new JIT path
- `software/workload/jit_hostop_chain.py::build_hostop_chain_artifact(...)`: mixed `NpuSegment -> HostOp(softmax) -> quantize_for_npu -> NpuSegment` workload
- `software/workload/jit_qdq_chain.py::build_qdq_chain_artifact(...)`: PyTorch quant/dequant module chain lowered into explicit host quantize/dequantize steps plus NPU segments
- `software/workload/inspect_simple_chain.py`: prints the segmented plan, logical previews, and packed output vectors for the migrated simple-chain artifact

Simulator smoke test:
- `cd verification/cocotb && MODULE=test_jit_runtime make -f Makefile.npu`
- Migrated simple-chain RTL test: `cd verification/cocotb && MODULE=test_jit_simple_chain make -f Makefile.npu`
- Simple-chain RTL vector inspection: `cd verification/cocotb && MODULE=test_jit_simple_chain_inspect make -f Makefile.npu`
- Mixed host/NPU chain RTL test: `cd verification/cocotb && MODULE=test_jit_hostop_chain make -f Makefile.npu`
- Quant/dequant stub RTL test: `cd verification/cocotb && MODULE=test_jit_qdq_chain make -f Makefile.npu`
- Exported MNIST conv RTL smoke test: `cd verification/cocotb && MODULE=test_jit_mnist_conv1 make -f Makefile.npu`

Current limitation:
- `torch` is an optional dependency and is not bundled by this repository today
- the simulator backend is exercised on the current narrow segmented matmul path, but broader mixed-op coverage is still incomplete
- standard PyTorch `nn.Linear` row-major normalization is not fully hardened yet; the traced frontend is currently validated on explicit `torch.matmul`
- the legacy `tinynpu/` package remains in place during migration

Open risks:
- ordinary PyTorch models still do not carry quantization metadata natively through the new path
- exported workloads like MNIST currently use export-backed quant config through transitional helpers such as `npu_matmul(...)`
- quant/dequant module support exists for explicit qparams, but broad FX-native quant metadata ingestion is still incomplete
- `HostOp -> NpuSegment` re-entry quantization now has a first explicit path through `quantize_for_npu(...)`, but broader quant config ingestion is still an active gap
- old `A/B/C/BIAS` role semantics still exist in lowering as a migration bridge

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
