# TinyNPU Perf Snapshot (2026-04-08)

All values below are `mcycle` deltas printed by the generated bare-metal firmware.

## INT8 pad+irregular (`cv32e40p_convstream_int8_pad_irreg_demo`)

- `-Os`: preload.ub=`1700`, preload.im=`223`, stage=`21305`, run=`93`, readback=`4700`, npu=`30234`
- `-O3`: preload.ub=`773`, preload.im=`101`, stage=`3242`, run=`97`, readback=`1410`, npu=`8785`

## INT4 pad+irregular (`cv32e40p_convstream_int4_pad_irreg_demo`)

- `-Os`: preload.ub=`1347`, preload.im=`231`, stage=`30172`, run=`84`, readback=`6858`, npu=`41319`
- `-O3`: preload.ub=`581`, preload.im=`101`, stage=`6678`, run=`90`, readback=`2828`, npu=`13648`

## INT16 pad+irregular (`cv32e40p_convstream_int16_pad_irreg_demo`, `-O3`)

- preload.ub=`1157`, preload.im=`101`, stage=`2414`, run=`113`, readback=`794`, npu=`7457`

## Fair convstream shape sweep

Shape used: `H=6, W=6, C=7, K=3, S=1, P=1, OC=7`

- INT16: stage=`28669`, run=`525`, readback=`18387`, npu=`51899`
- INT8: stage=`36343`, run=`364`, readback=`21288`, npu=`62233`
- INT4: stage=`46302`, run=`284`, readback=`21816`, npu=`72719`

## INT16 conv CPU-vs-NPU, conv_stream OFF (`cv32e40p_int16_cpu_vs_npu_conv_demo`)

- preload.ub=`1029`, preload.im=`33`
- hostop.im2col=`16003`
- segment.stage=`35616`, segment.run=`529`, segment.readback=`3451`, segment.npu=`43838`
- segment.cpu=`265626`
- E2E to output-ready:
  - NPU path = `1029 + 33 + 16003 + 43838 = 60903`
  - CPU path = `1029 + 33 + 16003 + 265626 = 282691`

## INT16 conv, conv_stream ON same shape (`cv32e40p_int16_convstream_same_shape_demo`)

- preload.ub=`1029`, preload.im=`32`
- segment.stage=`4083`, segment.run=`529`, segment.readback=`3292`, segment.npu=`12118`
- E2E to output-ready: `1029 + 32 + 12118 = 13179`

## INT16 linear CPU-vs-NPU (`cv32e40p_int16_cpu_vs_npu_linear_demo`)

- preload.ub=`1029`, preload.im=`32`
- segment.stage=`30313`, segment.run=`530`, segment.readback=`3319`, segment.npu=`38525`
- segment.cpu=`265001`

## Notes

- The runs above were generated through Runtime V1 emit (`emit_cv32e40p_c`) with `TINYNPU_USE_SHARED_SRAM=1`.
- These measurements include readback in `segment.*.npu` (stage + run + readback).
