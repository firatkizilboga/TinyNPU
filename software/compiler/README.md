# TinyNPU Compiler

A Python-based compiler for the TinyNPU architecture.

## New Segmented JIT Path

The repository now also contains a new experimental compiler/runtime path under `tinynpu_jit/`.
It also now contains the beginnings of a PyTorch-side quantization toolkit under `tinynpu_quant/`.

This path is designed around:
- PyTorch/FX as the frontend
- segmented execution plans instead of whole-program lowering
- explicit `HostOp(...)` boundaries for unsupported ops
- registry-backed host-op dispatch in `tinynpu_jit.host_ops` so execution and benchmark accounting share one contract
- compiler-owned expected tensors for runtime verification
- `mark_for_verify(...)` as a host-visible verification boundary

Current scope:
- TinyNPU lowering: explicit `matmul`, exported `linear`, exported `conv2d` lowered through host-side `im2col`, plus ordinary quantized `torch.ao.nn.quantized.Linear`
- PyTorch quant boundary support: `torch.ao.nn.quantized.Quantize` / `DeQuantize`, plus `QuantStub` / `DeQuantStub` when explicit qparams are attached
- ordinary quantized `torch.ao.nn.quantized.Conv2d` lowered through host-side `im2col`
- host-side attention primitives: `softmax`, `transpose`, and scalar `scale` host ops (usable between NPU matmul segments)
- host-emulation runtime for compile/run/verify validation
- async simulator runtime entrypoint that targets the existing cocotb driver path
- structured runtime debug tracing for host ops, NPU segments, verification, and simulator IO overlays
- deterministic compile behavior with explicit failure if a segment exceeds UB capacity

Performance note for attention-heavy (GPT/LLaMA-style) graphs:
- `pure_acceleration_speedup` compares CPU-equivalent compute vs. NPU compute only.
- `integration_adjusted_speedup` also includes NPU interface overhead.
- `end_to_end_speedup` includes host intrinsic work too (`softmax`, `scale`, etc.), which is the most realistic headline metric for mixed host/NPU attention blocks.

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
# ordinary quantized torch.ao modules are now the preferred direction for
# the real frontend contract, within the current supported quantization subset.
```

Examples:
- `example_segmented_jit.py`: manual execution-plan construction
- `example_torch_jit.py`: traced PyTorch module with `mark_for_verify(...)`
- `software/workload/mnist_npu_compiler.py::compile_mnist_layer_jit(...)`: exported MNIST `linear` and `conv2d` layer examples on the new PyTorch JIT path
- `software/workload/jit_test_gen.py::build_simple_chain_artifact(...)`: migrated old simple-chain workload on the new JIT path
- `software/workload/jit_hostop_chain.py::build_hostop_chain_artifact(...)`: mixed `NpuSegment -> HostOp(softmax) -> quantize_for_npu -> NpuSegment` workload
- `software/workload/jit_qdq_chain.py::build_qdq_chain_artifact(...)`: PyTorch quant/dequant module chain lowered into explicit host quantize/dequantize steps plus NPU segments
- `software/workload/jit_quantized_modules.py`: ordinary quantized `Linear` / `Conv2d` PyTorch module examples
- `software/workload/jit_qat_compiler_ready.py`: compiler-ready QAT conversion example
- `software/workload/mnist_tinynpu_pipeline.py`: fresh train -> QAT -> convert -> compile pipeline for MNIST
- `software/workload/benchmark_jit_multitile_matmul.py`: host-side benchmark accounting report for the new multi-tile JIT matmul
- `software/workload/inspect_simple_chain.py`: prints the segmented plan, logical previews, and packed output vectors for the migrated simple-chain artifact
- `tinynpu_quant.calibration.collect_tensor_percentile_scale(...)`: reusable percentile-based host-boundary calibration helper

### Attention block lowering (GPT/LLaMA-style)

A practical attention block decomposition on TinyNPU is:

1. `Q @ K^T` on NPU (`NpuSegment` + `MatMulOp`)
2. divide by `sqrt(d_k)` on host (`HostOp(kind="scale")`)
3. masked/unmasked `softmax` on host (`HostOp(kind="softmax")`)
4. (optional) quantize host result for NPU re-entry (`HostOp(kind="quantize")`)
5. `P @ V` on NPU (`NpuSegment` + `MatMulOp`)

This keeps high-density GEMMs on the accelerator while lowering nonlinear attention normalization on host ops. See `tests/test_attention_lowering.py` for a minimal end-to-end execution-plan example.

PyTorch quantization toolkit:
- `software/compiler/tinynpu_quant/`: extracted reusable quantization utilities
- `quant-by-claude.py`: still the MNIST-specific workflow script, but now backed by shared `tinynpu_quant` helpers instead of owning all quant logic itself

Simulator smoke test:
- `cd verification/cocotb && MODULE=test_jit_runtime make -f Makefile.npu`
- Migrated simple-chain RTL test: `cd verification/cocotb && MODULE=test_jit_simple_chain make -f Makefile.npu`
- Simple-chain RTL vector inspection: `cd verification/cocotb && MODULE=test_jit_simple_chain_inspect make -f Makefile.npu`
- Mixed host/NPU chain RTL test: `cd verification/cocotb && MODULE=test_jit_hostop_chain make -f Makefile.npu`
- Quant/dequant stub RTL test: `cd verification/cocotb && MODULE=test_jit_qdq_chain make -f Makefile.npu`
- Quantized PyTorch `Linear` RTL test: `cd verification/cocotb && MODULE=test_jit_quantized_linear make -f Makefile.npu`
- Quantized PyTorch `Conv2d` RTL test: `cd verification/cocotb && MODULE=test_jit_quantized_conv make -f Makefile.npu`
- Compiler-ready QAT RTL test: `cd verification/cocotb && MODULE=test_jit_qat_compiler_ready make -f Makefile.npu`
- Exported MNIST conv RTL smoke test: `cd verification/cocotb && MODULE=test_jit_mnist_conv1 make -f Makefile.npu`
- Full exported MNIST JIT RTL chain: `cd verification/cocotb && MODULE=test_jit_mnist_full_chain make -f Makefile.npu`
- Fresh trained MNIST pipeline RTL compare: `cd verification/cocotb && MODULE=test_jit_mnist_trained_pipeline make -f Makefile.npu`
- Perf-enabled JIT benchmark RTL test: `cd verification/cocotb && SIM_BUILD=sim_build_perf USER_EXTRA_ARGS=-GPERF_ENABLE=1 MODULE=test_jit_benchmark_multitile_matmul make -f Makefile.npu`

Current limitation:
- `torch` is an optional dependency and is not bundled by this repository today
- the simulator backend is exercised on the segmented matmul path, full exported MNIST chain, and ordinary quantized `Linear`; broader mixed-op coverage is still incomplete
- ordinary float `nn.Linear` / `nn.Conv2d` are not yet the trusted frontend contract; current trusted direct frontend is explicit quantized `torch.ao` modules plus export-backed helpers
- fresh trained-model host/RTL parity is validated on the new MNIST pipeline path
- the fresh MNIST pipeline now calibrates the `conv3 -> mean` host boundary instead of blindly reusing the FC input scale; this restores compiled-host accuracy to near-QAT quality on the validated slice
- that host-boundary calibration helper now lives in `tinynpu_quant` instead of remaining MNIST-specific code
- the legacy `tinynpu/` package remains in place during migration
- reproducible synthesis characterization script: `scripts/synth_tinynpu_yosys.py`
- the new benchmark path is an architectural model:
  - `cpu_replaced_cycles` counts only work the NPU replaces
  - `npu_overhead_cycles` counts only accelerator-induced packing/interface overhead
  - `npu_compute_cycles` comes from RTL perf counters and requires `PERF_ENABLE=1`

Open risks:
- only a narrow PyTorch quantization subset is accepted for NPU segments today:
  - signed quantized tensors
  - symmetric `zero_point = 0`
  - scale-derived `multiplier` / `shift`
- exported workloads like MNIST still use export-backed quant config through transitional helpers such as `npu_matmul(...)`
- ordinary quantized `Conv2d` lowering is validated on RTL for the current narrow quantized frontend path, but broader mixed-op graphs are still incomplete
- a reusable PyTorch-side quantization toolkit is not built yet; `quant-by-claude.py` is still a script, not the final `tinynpu_quant` package
- old `A/B/C/BIAS` role semantics still exist in lowering as a migration bridge

Current working policy:
- do not change RTL architecture just to add new packing primitives at this stage
- prefer fused NPU segments first to reduce repacking overhead in software
- prefer `int16` boundary preservation when UB allows it, especially for host-sensitive boundaries
- otherwise use calibrated boundary `int8` scales instead of blindly reusing the next layer activation scale

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
