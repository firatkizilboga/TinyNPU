# TinyNPU Quantization Toolkit

This package is the intended home for the PyTorch-side quantization pipeline that feeds the TinyNPU compiler.

Current contents:
- `compute_fused_params(...)`: shared fused-parameter math for TinyNPU `multiplier` / `shift`
- `LayerQuantConfig`: reusable per-layer bit-width config object
- `QConv2d` / `QLinear`: reusable QAT layer modules
- `collect_input_activation_maxes(...)`: calibration helper
- `single_layer_bit_drop_sensitivity(...)`: reusable sensitivity helper

Planned responsibilities:
- PTQ / QAT preparation helpers
- calibration utilities
- sensitivity analysis utilities
- conversion from prepared PyTorch models into compiler-supported quantized inference graphs

Current status:
- `quant-by-claude.py` still owns the model-specific MNIST workflow
- this package now owns the first reusable quantization primitives extracted from that script
- the compiler now also has a matching runtime/compiler-side rescale synthesizer in `tinynpu_jit/quantization.py`

Important constraint:
- current TinyNPU NPU lowering supports only symmetric zero-point-free quantization for actual NPU segments
