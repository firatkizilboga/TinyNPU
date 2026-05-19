# External FP32 CPU Baselines - 2026-05-19

Purpose: get a CPU-only benchmark path that does not use the TinyNPU compiler,
lowering, runtime, or CPU-only fallback. The current implementation emits
standalone C99 fp32/libm programs and links them directly into the cv32e40p
bare-metal testbench.

TVM/microTVM was attempted first, but the machine only has Python 3.12 and the
`apache-tvm` wheel was unavailable for this environment. Rather than block on a
source build, the first usable baseline is plain C/libm. This is still
independent from the TinyNPU software stack.

## Commands

```bash
python3 scripts/build_external_fp32_cpu_baselines.py --model iszero_mlp --run-rtl --maxcycles 20000000 --verilator-max-ticks 20000000000 --timeout-s 300
python3 scripts/build_external_fp32_cpu_baselines.py --model conv --run-rtl --maxcycles 30000000 --verilator-max-ticks 30000000000 --timeout-s 420
python3 scripts/build_external_fp32_cpu_baselines.py --model tiny_lm --run-rtl --maxcycles 80000000 --verilator-max-ticks 120000000000 --timeout-s 1200
```

## Results

| Model | Program | RTL result |
| --- | --- | --- |
| is_zero MLP fp32 C/libm | `external_fp32_cpu_iszero_mlp` | `cycles=111980`, output `0.487469882`, pred `0`, `EXIT SUCCESS` |
| 4-layer conv fp32 C/libm | `external_fp32_cpu_conv` | `cycles=252398`, output `0.487502635`, pred `0`, `EXIT SUCCESS` |
| TinyStories tiny LM fp32 C/libm | `external_fp32_cpu_tiny_lm` | `cycles=1340351`, next token `she`, `EXIT SUCCESS` |

TinyLM configuration is the trained d32/f32 toy model in
`runs/tinystories_word_lm_d32_t17_qat_int16_long`. The standalone C program
uses the exported fp32 weights, prompt `there was a little girl named lily .`,
and prompt length 9.

## Scope

This is a true bare-metal cv32e40p CPU path in the sense that the measured
program is standalone fp32 C linked with libc/libm and run through the RTL
testbench. It is not TVM/TFLM yet. If we need a named external ML compiler in
the thesis, the next step is building TVM from source or using a Python version
that has a compatible `apache-tvm` wheel.
