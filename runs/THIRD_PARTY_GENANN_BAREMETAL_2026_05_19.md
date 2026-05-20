# Third-Party Bare-Metal NN Baseline - genann - 2026-05-19

Purpose: prove that a lightweight third-party neural-network library can be
compiled into a RISC-V bare-metal binary and run on the cv32e40p RTL testbench
without an operating system.

Library: `genann` from https://github.com/codeplea/genann

License: zlib

## What Runs

The runner links upstream `genann.c` and calls `genann_run()` from a standalone
bare-metal C program. It does not use TinyNPU IR, lowering, runtime, NPU
execution, or the internal CPU-only fallback path.

The test model is a deterministic 64-input, 1-hidden-layer, 16-hidden-neuron,
1-output MLP using genann's feed-forward implementation. The MNIST `is_zero`
sample input is embedded into the generated C source.

To keep the program bare-metal friendly, the runner statically initializes the
public `genann` struct and buffers rather than calling `genann_init()`. This
avoids genann's 4096-entry sigmoid lookup initialization, which is extremely
slow on RV32 without a double-precision FPU. Inference still calls the upstream
third-party `genann_run()` and upstream activation functions.

## Command

```bash
python3 scripts/build_genann_baremetal_baseline.py --run-rtl --maxcycles 20000000 --verilator-max-ticks 20000000000 --timeout-s 300
```

## RTL Result

```text
third_party_genann_mlp sample_index=0 label=0 cycles=331914 output=0.548233738918 pred=1 weights=1057
EXIT SUCCESS
```

Generated artifacts:

```text
generated/third_party_genann_mlp.c
external/cv32e40p/example_tb/core/custom/third_party_genann_mlp.elf
external/cv32e40p/example_tb/core/custom/third_party_genann_mlp.hex
```

## Interpretation

This is the first true third-party bare-metal NN runtime proof in the repo. It
is not sufficient for Conv or TinyLM by itself; genann is only a small
feed-forward ANN library. For a third-party Conv/Transformer baseline, the next
step is still TFLite Micro or microTVM.
