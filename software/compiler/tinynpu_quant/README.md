# TinyNPU Quantization Toolkit

This package is the intended home for the PyTorch-side quantization pipeline that feeds the TinyNPU compiler.

Current contents:
- `compute_fused_params(...)`: shared fused-parameter math for TinyNPU `multiplier` / `shift`

Planned responsibilities:
- PTQ / QAT preparation helpers
- calibration utilities
- sensitivity analysis utilities
- conversion from prepared PyTorch models into compiler-supported quantized inference graphs

Current status:
- `quant-by-claude.py` still owns most of the training/QAT/export workflow
- this package is the start of migrating shared quantization logic out of that script
- the compiler now also has a matching runtime/compiler-side rescale synthesizer in `tinynpu_jit/quantization.py`

Important constraint:
- current TinyNPU NPU lowering supports only symmetric zero-point-free quantization for actual NPU segments
