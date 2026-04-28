#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
import sys
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
GEMM_CSV = ROOT / "runs" / "gemm_v2_e2e_compile_lhs" / "gemm_v2_e2e_results.csv"
PERF_MD = ROOT / "runs" / "PERF_2026-04-08_INT16_INT8_INT4_FAIR.md"
ARRAY_SIZE = 8


@dataclass(frozen=True)
class MatmulShape:
    name: str
    m: int
    k: int
    n: int
    count: int = 1

    @property
    def macs(self) -> int:
        return int(self.count * self.m * self.k * self.n)

    @property
    def weights(self) -> int:
        return int(self.count * self.k * self.n)


@dataclass(frozen=True)
class NpuTileModel:
    dtype: str
    fixed_cycles: float
    kstep_tile_cycles: float
    output_tile_cycles: float

    def body_cycles(self, shape: MatmulShape) -> float:
        mt = math.ceil(shape.m / ARRAY_SIZE)
        kt = math.ceil(shape.k / ARRAY_SIZE)
        nt = math.ceil(shape.n / ARRAY_SIZE)
        per = self.fixed_cycles + self.kstep_tile_cycles * (mt * kt * nt) + self.output_tile_cycles * (mt * nt)
        return float(shape.count) * per


@dataclass(frozen=True)
class StepCostRow:
    index: str
    kind: str
    name: str
    detail: str
    cpu_saved_cycles: float = 0.0
    cpu_count_cycles: float = 0.0
    host_cycles: float = 0.0
    npu_body_cycles: float = 0.0
    transfer_cycles: float = 0.0
    preload_cycles: float = 0.0
    primitive_host_cycles: float = 0.0
    primitive_npu_cycles: float = 0.0
    primitive_cpu_replaced_cycles: float = 0.0
    macs: int = 0
    words: int = 0
    rope_xforms: int = 0

    @property
    def hybrid_hot_cycles(self) -> float:
        return self.host_cycles + self.npu_body_cycles + self.transfer_cycles

    @property
    def hybrid_cold_cycles(self) -> float:
        return self.hybrid_hot_cycles + self.preload_cycles


@dataclass(frozen=True)
class QLlamaRtlCase:
    point: str
    mode: str
    d_model: int
    d_head: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden_dim: int
    prompt_len: int
    measured_npu_cold: float
    measured_npu_hot: float | None = None
    measured_cpu: float | None = None
    cpu_lower_bound: float | None = None

    @property
    def label(self) -> str:
        return f"{self.point}-{self.mode}"


@dataclass(frozen=True)
class QLlamaPlanFeatures:
    label: str
    mode: str
    d_model: int
    d_head: int
    n_heads: int
    n_kv_heads: int
    ffn_hidden_dim: int
    prompt_len: int
    steps: int
    host_steps: int
    segments: int
    matmul_ops: int
    rope_xforms: int
    ub_static_words: int
    ub_peak_words: int
    ub_capacity_words: int
    preload_cycles: float
    host_cycles: float
    npu_hot_cycles: float
    npu_non_setup_cycles: float
    cpu_replaced_cycles: float

    @property
    def fits_current_ub(self) -> bool:
        return not self.ub_capacity_words or self.ub_peak_words <= self.ub_capacity_words


@dataclass(frozen=True)
class QLlamaStructuredFit:
    cpu_host_scale: float
    cpu_matmul_scale: float
    cpu_fixed_cycles: float
    npu_host_scale: float
    npu_non_setup_scale: float
    npu_segment_setup_cycles: float

    def predict_cpu(self, features: QLlamaPlanFeatures) -> float:
        return (
            self.cpu_host_scale * features.host_cycles
            + self.cpu_matmul_scale * features.cpu_replaced_cycles
            + self.cpu_fixed_cycles
        )

    def predict_npu_hot(self, features: QLlamaPlanFeatures) -> float:
        return (
            self.npu_host_scale * features.host_cycles
            + self.npu_non_setup_scale * features.npu_non_setup_cycles
            + self.npu_segment_setup_cycles * float(features.segments)
        )

    def predict_npu_cold(self, features: QLlamaPlanFeatures) -> float:
        return self.predict_npu_hot(features) + features.preload_cycles


@dataclass(frozen=True)
class RuntimeCalibration:
    name: str
    npu_segment_launch_cycles: float = 0.0
    npu_body_scale: float = 1.0
    host_expensive_prefill_scale: float = 1.0
    host_expensive_decode_slope: float = 0.0
    host_expensive_decode_intercept: float = 1.0
    host_expensive_decode_cap: float = 1.0
    source: str = "raw GEMM/body fit only"

    def host_expensive_decode_scale(self, d_model: int) -> float:
        return min(
            self.host_expensive_decode_cap,
            max(1.0, self.host_expensive_decode_slope * float(d_model) + self.host_expensive_decode_intercept),
        )

    def calibrated_host_cycles(self, row: StepCostRow, *, host_expensive_scale: float) -> float:
        kind = row.detail.split()[0] if row.detail else ""
        if kind in {"rmsnorm", "rope", "silu", "softmax_f16"}:
            return row.host_cycles * host_expensive_scale
        return row.host_cycles

    def calibrated_npu_body_cycles(self, row: StepCostRow) -> float:
        if row.kind != "NPU":
            return row.npu_body_cycles
        return self.npu_segment_launch_cycles + self.npu_body_scale * row.npu_body_cycles

    def calibrated_hot_cycles(self, row: StepCostRow, *, host_expensive_scale: float) -> float:
        return self.calibrated_host_cycles(row, host_expensive_scale=host_expensive_scale) + self.calibrated_npu_body_cycles(row) + row.transfer_cycles

    def calibrated_cold_cycles(self, row: StepCostRow, *, host_expensive_scale: float) -> float:
        return self.calibrated_hot_cycles(row, host_expensive_scale=host_expensive_scale) + row.preload_cycles


@dataclass(frozen=True)
class HardwarePrimitives:
    # Scalar RV32IM-ish integer primitives.
    # Calibrated from the saved single-linear, MLP, and conv CPU segment runs:
    # these costs predict those three independent CPU baselines within ~5%.
    load: float = 2.0
    store: float = 2.0
    iadd: float = 2.0
    imul: float = 6.0
    ishift: float = 2.0
    branch: float = 4.0
    clamp: float = 2.0
    bitop: float = 1.0
    nonlinear: float = 16.0

    # Hardware FPU primitives. Build is rv32imfc (ilp32f) so cv32e40p has the
    # F extension enabled with the default FPU config (FPU_ADDMUL_LAT=0,
    # FPU_OTHERS_LAT=0; iterative fdiv/fsqrt). Single-precision IEEE 754 is
    # native:
    #   fadd.s/fsub.s/fmul.s/fmadd.s    ~1 EX cycle
    #   fcvt.*/fsgnj.*/fcmp             1-2 cycles
    #   fdiv.s/fsqrt.s                  iterative, ~12-15 cycles
    # These are NOT softfloat. Earlier values (30/40/60/80) were a mistake and
    # caused the model to over-predict host op cost roughly 5-10x.
    #
    # The libm transcendental calls (expf/logf/sinf/cosf) DO still go through
    # software polynomial approximations, but those use the hardware FPU
    # internally — so each call is ~10-20 hw FP ops plus range-reduction
    # branches, giving ~80-150 cyc total instead of the softfloat 250+ range.
    f_add: float = 1.0
    f_sub: float = 1.0
    f_mul: float = 1.0
    f_div: float = 13.0
    f_sqrt: float = 13.0
    # libm via hw FPU; placeholder until isolated bench. Real value depends
    # on which expf/sinf implementation newlib/picolibc ships.
    f_exp: float = 100.0
    f_trig: float = 100.0

    # TinyNPU primitives. Output writeback is initialized from the existing GEMM
    # output-tile term (~698 cycles / 64 padded tile elements), not fitted to
    # QLlama/GPT block totals.
    array_size: int = ARRAY_SIZE
    pipeline_fill: float = ARRAY_SIZE
    pipeline_drain: float = ARRAY_SIZE
    kstep_per_tile: float = ARRAY_SIZE
    # Derived from the 3 calibration GEMMs (64x64x64, 96x64x96, 128x128x128),
    # all of which have full 8x8 output tiles. Skinny-output behavior is
    # unconstrained by this data and currently overestimated for partial tiles.
    output_writeback_per_padded_elem: float = 10.918
    # IM word decode and one-time per-segment cmd setup, derived from saved
    # logs (one matmul+halt segment is 87 cycles, four-matmul+halt is 183).
    segment_cmd_decode_per_im_word: float = 32.0
    segment_cmd_setup: float = 23.0
    # RoPE XFORM cycles = phase*half_count + exit, derived from the XFORM
    # microcode: each (cos,sin) pair drives a two-cycle multiply-add and the
    # state machine exits one cycle after the last pair.
    rope_xform_phase_cycles: float = 6.0
    rope_xform_exit_cycles: float = 1.0
    # Per-segment runtime overhead. **Measured value, not fitted.** From the
    # QLlama A prefill Verilator run (runs/kernel_cycles.json), per-segment
    # residual after subtracting stage + run + readback was ~4172 cycles,
    # consistent across all 6 segments (qkv, score, value, o_proj, ffn_up,
    # ffn_down). This represents npu_run() driver call + completion poll +
    # MMIO arbitration; not yet decomposed into sub-primitives. The earlier
    # block-rtl value of 6955 was over-fit to QGPT2 d8 segment counts that
    # include extra IM decoding the per-segment overhead does not.
    # Should be revisited if larger-d_model segments show systematic
    # deviation. Override with `--npu-segment-overhead-cycles`.
    # Updated 2026-04-26: mean across 7 RTL-measured runs (QLlama A/B/C × prefill+decode + d=64 prefill) is 4231 cyc/segment, range 4121-4399. Constant across d_model 8-64.
    npu_segment_unmodeled_overhead: float = 4231.0
    npu_h_gelu_per_elem: float = 11.5
    npu_sigmoid_per_elem: float = 16.0

    # Memory/interface primitives.
    # `ub_word_latency` is the static-preload bandwidth (large transfer
    # regime), fit from the GEMM CSV.
    ub_word_latency: float = 16.032
    # `dma_setup` is the per-segment DMA/MMIO stage cost — fixed-cost
    # descriptor + arbitration. **Measured ~1824 cyc/segment** on QLlama A
    # (runs/kernel_cycles.json), independent of word count for the small
    # transfers in that block. Larger transfers should add `words *
    # ub_word_latency` on top, but for now this captures the dominant
    # per-segment fixed cost the previous model missed entirely.
    dma_setup: float = 1824.0
    # Per-segment readback overhead, similar measured value (~1983 cyc avg).
    dma_readback_setup: float = 1983.0

    # CPU reference matmul overheads. These make small decode matmuls pay setup
    # and per-output costs instead of using a flat cycles/MAC.
    cpu_matmul_setup: float = 200.0
    cpu_output_setup: float = 10.0


DEFAULT_PRIMITIVES = HardwarePrimitives()


RAW_CALIBRATION = RuntimeCalibration(name="raw")
PRIMITIVE_CALIBRATION = RuntimeCalibration(
    name="primitive",
    source="shape-driven HardwarePrimitives formulas; constants are microbench placeholders/report-derived where noted",
)

# Fitted from saved RTL evidence:
# - NPU segment launch/body correction: QGPT2 d8 per-segment counts in commit
#   c5568b3 plus the reported is_zero MLP segment.
# - Host scalar correction: weighted fit to QLlama A/B/C CPU/NPU RTL totals in
#   commit 034f7e4 after applying the segment correction above.
BLOCK_RTL_CALIBRATION = RuntimeCalibration(
    name="block-rtl",
    npu_segment_launch_cycles=6938.327683164376,
    npu_body_scale=0.7947830294136037,
    host_expensive_prefill_scale=10.0,
    host_expensive_decode_slope=0.6625,
    host_expensive_decode_intercept=-1.95,
    host_expensive_decode_cap=10.0,
    source="QGPT2 d8 segment RTL + is_zero MLP + QLlama A/B/C structured host/segment RTL",
)


def runtime_calibration(name: str) -> RuntimeCalibration:
    if name == "primitive":
        return PRIMITIVE_CALIBRATION
    if name == "raw":
        return RAW_CALIBRATION
    if name == "block-rtl":
        return BLOCK_RTL_CALIBRATION
    raise ValueError(f"unknown runtime calibration {name}")


def _solve_3x3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    a = [row[:] + [rhs] for row, rhs in zip(matrix, vector, strict=True)]
    for col in range(3):
        pivot = max(range(col, 3), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("Singular 3x3 fit matrix.")
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        for j in range(col, 4):
            a[col][j] /= scale
        for row in range(3):
            if row == col:
                continue
            factor = a[row][col]
            for j in range(col, 4):
                a[row][j] -= factor * a[col][j]
    return [a[row][3] for row in range(3)]


def _solve_linear_system(matrix: list[list[float]], vector: list[float]) -> list[float]:
    size = len(vector)
    if len(matrix) != size or any(len(row) != size for row in matrix):
        raise ValueError("linear solve expects a square matrix")
    a = [row[:] + [rhs] for row, rhs in zip(matrix, vector, strict=True)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda row: abs(a[row][col]))
        if abs(a[pivot][col]) < 1e-12:
            raise ValueError("Singular fit matrix.")
        if pivot != col:
            a[col], a[pivot] = a[pivot], a[col]
        scale = a[col][col]
        for j in range(col, size + 1):
            a[col][j] /= scale
        for row in range(size):
            if row == col:
                continue
            factor = a[row][col]
            for j in range(col, size + 1):
                a[row][j] -= factor * a[col][j]
    return [a[row][size] for row in range(size)]


def _least_squares(matrix: list[list[float]], vector: list[float]) -> list[float]:
    if not matrix:
        raise ValueError("least-squares fit needs at least one row")
    cols = len(matrix[0])
    if any(len(row) != cols for row in matrix):
        raise ValueError("least-squares rows have inconsistent widths")
    normal = [
        [sum(row[i] * row[j] for row in matrix) for j in range(cols)]
        for i in range(cols)
    ]
    rhs = [sum(row[i] * value for row, value in zip(matrix, vector, strict=True)) for i in range(cols)]
    return _solve_linear_system(normal, rhs)


def load_gemm_rows(path: Path = GEMM_CSV) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fit_npu_models(rows: list[dict[str, str]]) -> dict[str, NpuTileModel]:
    models: dict[str, NpuTileModel] = {}
    for dtype in sorted({row["dtype"] for row in rows}):
        subset = [row for row in rows if row["dtype"] == dtype]
        if len(subset) != 3:
            raise ValueError(f"Expected exactly three calibration rows for {dtype}, got {len(subset)}.")
        matrix: list[list[float]] = []
        vector: list[float] = []
        for row in subset:
            m, k, n = int(row["m"]), int(row["k"]), int(row["n"])
            mt = math.ceil(m / ARRAY_SIZE)
            kt = math.ceil(k / ARRAY_SIZE)
            nt = math.ceil(n / ARRAY_SIZE)
            matrix.append([1.0, float(mt * kt * nt), float(mt * nt)])
            vector.append(float(row["cold_npu"]))
        fixed, kstep, output = _solve_3x3(matrix, vector)
        models[dtype] = NpuTileModel(dtype=dtype, fixed_cycles=fixed, kstep_tile_cycles=kstep, output_tile_cycles=output)
    return models


def infer_preload_cycles_per_word(rows: list[dict[str, str]]) -> float:
    samples: list[float] = []
    for row in rows:
        m, k, n = int(row["m"]), int(row["k"]), int(row["n"])
        precision_pack = {"int16": 1, "int8": 2, "int4": 4}[row["dtype"]]
        mt = math.ceil(m / ARRAY_SIZE)
        kt = math.ceil((k // precision_pack) / ARRAY_SIZE)
        nt = math.ceil(n / ARRAY_SIZE)
        static_words = mt * kt * ARRAY_SIZE + kt * nt * ARRAY_SIZE
        # All calibration logs use one IM preload at 87 cycles.
        samples.append((float(row["preload_total"]) - 87.0) / float(static_words))
    return sum(samples) / len(samples)


def _extract_int(pattern: str, text: str) -> int:
    match = re.search(pattern, text)
    if not match:
        raise ValueError(f"Could not find pattern: {pattern}")
    return int(match.group(1))


def infer_cpu_cycles_per_mac(path: Path = PERF_MD) -> tuple[float, list[tuple[str, float]]]:
    text = path.read_text()
    single_linear_cpu = _extract_int(r"warm\.avg\.cpu=`([0-9]+)`", text)
    single_linear_macs = 36 * 63 * 6

    mlp_cpu = _extract_int(r"segment\.segment_000\.cpu=`([0-9]+)`", text)
    mlp_macs = (64 * 64) + (64 * 64) + (64 * 64) + (64 * 1)

    conv_cpu = _extract_int(r"cpu_segments\.total=`([0-9]+)`", text)
    conv_macs = (36 * 9 * 16) + (16 * 144 * 16) + (4 * 144 * 16) + (1 * 64 * 1)

    samples = [
        ("single_linear_36x63x6", single_linear_cpu / single_linear_macs),
        ("four_layer_mlp", mlp_cpu / mlp_macs),
        ("four_layer_conv_im2col_matmuls", conv_cpu / conv_macs),
    ]
    return sum(value for _, value in samples) / len(samples), samples


def preload_words_for_weights(shapes: list[MatmulShape], dtype: str) -> int:
    pack = {"int16": 1, "int8": 2, "int4": 4}[dtype]
    words = 0
    for shape in shapes:
        kt = math.ceil((shape.k // pack) / ARRAY_SIZE)
        nt = math.ceil(shape.n / ARRAY_SIZE)
        words += shape.count * kt * nt * ARRAY_SIZE
    return int(words)


def tinyllama_linear_shapes(prompt_len: int, *, decode_context: int) -> tuple[list[MatmulShape], list[MatmulShape]]:
    d_model = 2048
    d_head = 64
    n_heads = 32
    n_kv_heads = 4
    kv_dim = n_kv_heads * d_head
    ffn = 5632
    layers = 22

    per_layer_prefill = [
        MatmulShape("q_proj", prompt_len, d_model, d_model, layers),
        MatmulShape("k_proj", prompt_len, d_model, kv_dim, layers),
        MatmulShape("v_proj", prompt_len, d_model, kv_dim, layers),
        MatmulShape("o_proj", prompt_len, d_model, d_model, layers),
        MatmulShape("gate_proj", prompt_len, d_model, ffn, layers),
        MatmulShape("up_proj", prompt_len, d_model, ffn, layers),
        MatmulShape("down_proj", prompt_len, ffn, d_model, layers),
        MatmulShape("score_qk", prompt_len, d_head, prompt_len, layers * n_heads),
        MatmulShape("value_av", prompt_len, prompt_len, d_head, layers * n_heads),
    ]
    per_layer_decode = [
        MatmulShape("q_proj", 1, d_model, d_model, layers),
        MatmulShape("k_proj", 1, d_model, kv_dim, layers),
        MatmulShape("v_proj", 1, d_model, kv_dim, layers),
        MatmulShape("o_proj", 1, d_model, d_model, layers),
        MatmulShape("gate_proj", 1, d_model, ffn, layers),
        MatmulShape("up_proj", 1, d_model, ffn, layers),
        MatmulShape("down_proj", 1, ffn, d_model, layers),
        MatmulShape("score_qk", 1, d_head, decode_context, layers * n_heads),
        MatmulShape("value_av", 1, decode_context, d_head, layers * n_heads),
    ]
    return per_layer_prefill, per_layer_decode


def summarize_workload(
    label: str,
    shapes: list[MatmulShape],
    *,
    npu_models: dict[str, NpuTileModel],
    cpu_cycles_per_mac: float,
    preload_cycles_per_word: float,
    linear_dtype: str,
    attention_dtype: str,
    stream_weights: bool,
) -> dict[str, float]:
    macs = float(sum(shape.macs for shape in shapes))
    body = 0.0
    for shape in shapes:
        dtype = attention_dtype if shape.name.startswith(("score_", "value_")) else linear_dtype
        body += npu_models[dtype].body_cycles(shape)
    cpu = macs * cpu_cycles_per_mac
    weight_words = preload_words_for_weights(
        [shape for shape in shapes if not shape.name.startswith(("score_", "value_"))],
        linear_dtype,
    )
    preload = weight_words * preload_cycles_per_word if stream_weights else 0.0
    hybrid = body + preload
    return {
        "label": label,
        "macs": macs,
        "cpu_cycles": cpu,
        "npu_body_cycles": body,
        "weight_preload_cycles": preload,
        "hybrid_cycles": hybrid,
        "speedup": cpu / hybrid if hybrid else float("inf"),
    }


def print_report_case_validation(
    *,
    npu_models: dict[str, NpuTileModel],
    cpu_cycles_per_mac: float,
    calibration: RuntimeCalibration,
) -> None:
    cases = [
        {
            "name": "single_linear_36x63x6",
            "shapes": [MatmulShape("linear", 36, 63, 6)],
            "activations": ["none"],
            "measured_warm_npu": 37766.0,
            "measured_cold_npu": 42083.0,
            "measured_cpu_segment": 265345.0,
            "preload_cycles": 42083.0 - 37766.0,
            "note": "small-N skinny output; not represented by GEMM calibration shapes",
        },
        {
            "name": "iszero_mlp_4layer",
            "shapes": [
                MatmulShape("fc1", 1, 64, 64),
                MatmulShape("fc2", 1, 64, 64),
                MatmulShape("fc3", 1, 64, 64),
                MatmulShape("fc4", 1, 64, 1),
            ],
            "activations": ["relu", "relu", "h_gelu", "sigmoid"],
            "measured_warm_npu": 23239.0,
            "measured_cold_npu": 49844.0,
            "measured_cpu_segment": 239229.0,
            "measured_cpu_e2e": 252731.0,
            "preload_cycles": 26605.0,
            "note": "tested Runtime V2 repeat3 case from the report",
        },
    ]

    print("\nValidation against saved small RTL/report cases:")
    print("  source: runs/PERF_2026-04-08_INT16_INT8_INT4_FAIR.md")
    print(
        "  "
        + f"{'case':<24} {'model_warm':>11} {'meas_warm':>11} {'err':>8} "
        + f"{'model_cold':>11} {'meas_cold':>11} {'err':>8} "
        + f"{'model_cpu':>11} {'meas_cpu':>11} {'err':>8}"
    )
    for case in cases:
        shapes = list(case["shapes"])
        activations = list(case["activations"])
        model_body = sum(npu_models["int16"].body_cycles(shape) for shape in shapes)
        model_warm = model_body
        if calibration.name == "primitive":
            model_warm = DEFAULT_PRIMITIVES.npu_segment_unmodeled_overhead
            model_warm += sum(primitive_npu_matmul_body_cycles(shape) for shape in shapes)
            model_warm += sum(
                primitive_npu_activation_cycles(act, shape.count * shape.m * shape.n)
                for shape, act in zip(shapes, activations, strict=True)
            )
        elif calibration.name != "raw":
            model_warm = calibration.npu_segment_launch_cycles + calibration.npu_body_scale * model_body
        model_cold = model_warm + float(case["preload_cycles"])
        macs = sum(shape.macs for shape in shapes)
        model_cpu = (
            sum(
                primitive_cpu_matmul_cycles(shape, activation=act)
                for shape, act in zip(shapes, activations, strict=True)
            )
            if calibration.name == "primitive"
            else float(macs) * cpu_cycles_per_mac
        )
        measured_cpu = float(case["measured_cpu_segment"])

        def err(model: float, measured: float) -> str:
            return f"{((model - measured) / measured) * 100.0:+.1f}%"

        print(
            "  "
            + f"{case['name']:<24} {fmt(model_warm):>11} {fmt(float(case['measured_warm_npu'])):>11} "
            + f"{err(model_warm, float(case['measured_warm_npu'])):>8} "
            + f"{fmt(model_cold):>11} {fmt(float(case['measured_cold_npu'])):>11} "
            + f"{err(model_cold, float(case['measured_cold_npu'])):>8} "
            + f"{fmt(model_cpu):>11} {fmt(measured_cpu):>11} {err(model_cpu, measured_cpu):>8}"
        )
        print(f"    note: {case['note']}")


def _compiled_plan_cycle_totals(
    rows: list[StepCostRow],
    *,
    cpu_replaced_saved_model: float,
    calibration: RuntimeCalibration,
    host_expensive_scale: float,
) -> dict[str, float]:
    host_raw = sum(row.host_cycles for row in rows)
    npu_raw = sum(row.npu_body_cycles for row in rows)
    transfer = sum(row.transfer_cycles for row in rows)
    preload = sum(row.preload_cycles for row in rows)
    primitive_host = sum(row.primitive_host_cycles for row in rows)
    primitive_npu_hot = sum(row.primitive_npu_cycles for row in rows if row.kind == "NPU")
    primitive_cpu_replaced = sum(row.primitive_cpu_replaced_cycles for row in rows)
    if calibration.name == "primitive":
        return {
            "host_raw": host_raw,
            "npu_raw": npu_raw,
            "transfer": transfer,
            "preload": preload,
            "host_cal": primitive_host,
            "npu_cal": primitive_npu_hot,
            "hybrid_hot_raw": host_raw + npu_raw + transfer,
            "hybrid_cold_raw": host_raw + npu_raw + transfer + preload,
            "hybrid_hot_cal": primitive_host + primitive_npu_hot,
            "hybrid_cold_cal": primitive_host + primitive_npu_hot + preload,
            "cpu_hot_raw": host_raw + cpu_replaced_saved_model,
            "cpu_hot_cal": primitive_host + primitive_cpu_replaced,
        }
    host_cal = sum(calibration.calibrated_host_cycles(row, host_expensive_scale=host_expensive_scale) for row in rows)
    npu_cal = sum(calibration.calibrated_npu_body_cycles(row) for row in rows if row.kind == "NPU")
    return {
        "host_raw": host_raw,
        "npu_raw": npu_raw,
        "transfer": transfer,
        "preload": preload,
        "host_cal": host_cal,
        "npu_cal": npu_cal,
        "hybrid_hot_raw": host_raw + npu_raw + transfer,
        "hybrid_cold_raw": host_raw + npu_raw + transfer + preload,
        "hybrid_hot_cal": host_cal + npu_cal + transfer,
        "hybrid_cold_cal": host_cal + npu_cal + transfer + preload,
        "cpu_hot_raw": host_raw + cpu_replaced_saved_model,
        "cpu_hot_cal": host_cal + cpu_replaced_saved_model,
    }


def _err_pct(model: float, measured: float) -> str:
    return f"{((model - measured) / measured) * 100.0:+.1f}%"


def _artifact_d_model(artifact: Any) -> int:
    spec = artifact.plan.tensors.get("x_in")
    if spec is not None and spec.shape:
        return int(spec.shape[-1])
    for name in artifact.plan.inputs:
        input_spec = artifact.plan.tensors.get(name)
        if input_spec is not None and input_spec.shape:
            return int(input_spec.shape[-1])
    return 0


def _host_expensive_scale_for_plan(artifact: Any, label: str, calibration: RuntimeCalibration) -> float:
    if calibration.name == "raw":
        return 1.0
    mode = "decode" if "decode" in label else "prefill"
    if mode == "decode":
        return calibration.host_expensive_decode_scale(_artifact_d_model(artifact))
    return calibration.host_expensive_prefill_scale


def print_qllama_calibration_validation(
    *,
    npu_models: dict[str, NpuTileModel],
    cpu_cycles_per_mac: float,
    preload_cycles_per_word: float,
    calibration: RuntimeCalibration,
) -> None:
    _compiler_path()
    from tinynpu_jit import NpuSegment, VerificationMode, five_stage_in_order_model

    cases = [
        ("A", "prefill", 8, 8, 1, 1, 8, 171710.0, None, 224843.0, None),
        ("A", "decode", 8, 8, 1, 1, 8, 60591.0, None, 35852.0, None),
        ("B", "prefill", 16, 8, 2, 1, 16, 289401.0, None, 534748.0, None),
        ("B", "decode", 16, 8, 2, 1, 16, 89772.0, None, 77480.0, None),
        ("C", "prefill", 32, 8, 4, 2, 32, 549512.0, 536302.0, None, 1339611.0),
        ("C", "decode", 32, 8, 4, 2, 32, 149867.0, 136650.0, 209120.0, None),
    ]

    print("\nQLlama RTL calibration validation:")
    print("  source: commit 034f7e4 QLlamaBlock RTL O3 fast-sim findings")
    print(f"  calibration={calibration.name}: {calibration.source}")
    print(
        "  "
        + f"{'case':<10} {'model_hybrid_cold':>17} {'meas_total':>13} {'err':>8} "
        + f"{'model_npu_hot':>13} {'meas_npu_hot':>12} {'err':>8} "
        + f"{'model_cpu':>11} {'meas/lb_cpu':>11} {'err':>8}"
    )
    for point, mode, d_model, d_head, n_heads, n_kv_heads, ffn_hidden_dim, meas_npu_cold, meas_npu_hot, meas_cpu, cpu_lower in cases:
        args = argparse.Namespace(
            d_model=d_model,
            d_head=d_head,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            ffn_hidden_dim=ffn_hidden_dim,
            prompt_len=8,
            seed=0,
        )
        artifact = _build_qllama_artifact(args, mode)
        macs = 0
        for step in artifact.plan.steps:
            if isinstance(step, NpuSegment):
                for op in step.ops:
                    macs += _op_shape(artifact, op).macs
        result = artifact.run_host_emulation(
            {},
            verification=VerificationMode.OFF,
            benchmark=True,
            cost_model=five_stage_in_order_model(),
        )
        bench = result.benchmark.to_dict() if result.benchmark is not None else {"entries": []}
        rows = compiled_plan_step_cost_rows(
            artifact,
            npu_models=npu_models,
            cpu_cycles_per_mac=cpu_cycles_per_mac,
            preload_cycles_per_word=preload_cycles_per_word,
            attention_dtype="int16",
            benchmark_entries=list(bench.get("entries", [])),
        )
        totals = _compiled_plan_cycle_totals(
            rows,
            cpu_replaced_saved_model=float(macs) * cpu_cycles_per_mac,
            calibration=calibration,
            host_expensive_scale=_host_expensive_scale_for_plan(artifact, mode, calibration),
        )
        cpu_ref = meas_cpu if meas_cpu is not None else cpu_lower
        cpu_err = _err_pct(totals["cpu_hot_cal"], cpu_ref) if meas_cpu is not None else "lower-bound"
        npu_hot_ref = meas_npu_hot if meas_npu_hot is not None else 0.0
        npu_hot_model = totals["hybrid_hot_cal"]
        print(
            "  "
            + f"{point}-{mode:<8} {fmt(totals['hybrid_cold_cal']):>14} {fmt(meas_npu_cold):>13} "
            + f"{_err_pct(totals['hybrid_cold_cal'], meas_npu_cold):>8} "
            + f"{fmt(npu_hot_model):>13} {(fmt(npu_hot_ref) if meas_npu_hot is not None else '-'):>12} "
            + f"{(_err_pct(npu_hot_model, npu_hot_ref) if meas_npu_hot is not None else '-'):>8} "
            + f"{fmt(totals['cpu_hot_cal']):>11} {fmt(cpu_ref):>11} {cpu_err:>8}"
        )


def fmt(value: float) -> str:
    if abs(value) >= 1e9:
        return f"{value / 1e9:.3f}B"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.3f}M"
    if abs(value) >= 1e3:
        return f"{value / 1e3:.3f}K"
    return f"{value:.3f}"


def _compiler_path() -> None:
    compiler_path = ROOT / "software" / "compiler"
    if str(compiler_path) not in sys.path:
        sys.path.insert(0, str(compiler_path))


def _build_qllama_artifact(args: argparse.Namespace, mode: str) -> Any:
    _compiler_path()
    from tinynpu_jit.blocks import build_llama_decode_artifact, build_llama_prefill_artifact

    kwargs = {
        "d_model": args.d_model,
        "d_head": args.d_head,
        "n_heads": args.n_heads,
        "n_kv_heads": args.n_kv_heads,
        "ffn_hidden_dim": args.ffn_hidden_dim,
        "prompt_len": args.prompt_len,
        "seed": args.seed,
    }
    if mode == "prefill":
        artifact, _, _ = build_llama_prefill_artifact(**kwargs)
        return artifact
    if mode == "decode":
        artifact, _, _, _ = build_llama_decode_artifact(**kwargs)
        return artifact
    raise ValueError(f"unsupported compiled plan mode {mode}")


def _is_attention_op(segment_name: str, op_name: str) -> bool:
    return segment_name in {"seg_score", "seg_value"} or "qk" in op_name or "av" in op_name


def _op_shape(artifact: Any, op: Any) -> MatmulShape:
    lhs_shape = tuple(int(dim) for dim in artifact.plan.tensors[op.lhs].shape)
    rhs_shape = tuple(int(dim) for dim in artifact.plan.tensors[op.rhs].shape)
    if len(lhs_shape) != 2 or len(rhs_shape) != 2:
        raise ValueError(f"Only rank-2 matmul ops are supported, got {lhs_shape} x {rhs_shape}")
    return MatmulShape(op.name, lhs_shape[0], lhs_shape[1], rhs_shape[1])


def _tensor_elements(artifact: Any, tensor_name: str) -> int:
    spec = artifact.plan.tensors[tensor_name]
    return int(math.prod(int(dim) for dim in spec.shape))


def _tensor_last_dim(artifact: Any, tensor_name: str) -> int:
    spec = artifact.plan.tensors[tensor_name]
    return int(spec.shape[-1]) if spec.shape else 1


def _tensor_rows(artifact: Any, tensor_name: str) -> int:
    spec = artifact.plan.tensors[tensor_name]
    if not spec.shape:
        return 1
    last = int(spec.shape[-1])
    return max(1, int(math.prod(int(dim) for dim in spec.shape)) // max(last, 1))


def primitive_npu_matmul_body_cycles(shape: MatmulShape, primitives: HardwarePrimitives = DEFAULT_PRIMITIVES) -> float:
    mt = math.ceil(shape.m / primitives.array_size)
    kt = math.ceil(shape.k / primitives.array_size)
    nt = math.ceil(shape.n / primitives.array_size)
    output_tiles = mt * nt
    padded_output_elems = output_tiles * primitives.array_size * primitives.array_size
    per = (
        output_tiles * (primitives.pipeline_fill + primitives.pipeline_drain)
        + output_tiles * kt * primitives.kstep_per_tile
        + padded_output_elems * primitives.output_writeback_per_padded_elem
    )
    return float(shape.count) * per


def primitive_npu_activation_cycles(
    activation: str,
    out_elems: int,
    primitives: HardwarePrimitives = DEFAULT_PRIMITIVES,
) -> float:
    if activation == "h_gelu":
        return float(out_elems) * primitives.npu_h_gelu_per_elem
    if activation == "sigmoid":
        return float(out_elems) * primitives.npu_sigmoid_per_elem
    return 0.0


@dataclass(frozen=True)
class BlockEstimate:
    """Shape-driven analytical estimate for one transformer block.

    Every field is in cycles. The decomposition matches HardwarePrimitives so
    that the contribution of each primitive constant can be inspected and an
    error bar derived for primitives that are still placeholders.
    """

    label: str
    npu_matmul_body: float
    npu_writeback: float
    npu_rope_xforms: float
    npu_activation: float
    npu_segment_cmd: float
    npu_segment_unmodeled: float
    npu_transfer: float
    npu_static_preload: float
    host_rmsnorm: float
    host_quant: float
    host_dequant: float
    host_softmax: float
    host_silu: float
    host_residual: float
    host_score_scale: float
    host_ffn_mul: float
    host_concat: float
    host_kv_scatter: float
    host_causal_mask: float
    cpu_baseline_matmul: float
    cpu_baseline_host: float
    macs: int
    n_segments: int
    n_layers: int

    @property
    def npu_body_total(self) -> float:
        return (
            self.npu_matmul_body
            + self.npu_writeback
            + self.npu_rope_xforms
            + self.npu_activation
        )

    @property
    def npu_segment_total(self) -> float:
        return self.npu_segment_cmd + self.npu_segment_unmodeled

    @property
    def host_total(self) -> float:
        return (
            self.host_rmsnorm
            + self.host_quant
            + self.host_dequant
            + self.host_softmax
            + self.host_silu
            + self.host_residual
            + self.host_score_scale
            + self.host_ffn_mul
            + self.host_concat
            + self.host_kv_scatter
            + self.host_causal_mask
        )

    @property
    def hybrid_hot(self) -> float:
        return self.npu_body_total + self.npu_segment_total + self.npu_transfer + self.host_total

    @property
    def hybrid_cold(self) -> float:
        return self.hybrid_hot + self.npu_static_preload

    @property
    def cpu_baseline(self) -> float:
        return self.cpu_baseline_matmul + self.cpu_baseline_host


def _matmul_body_components(
    shape: MatmulShape,
    primitives: HardwarePrimitives,
) -> tuple[float, float]:
    """Split the matmul body into (compute, writeback) components."""
    mt = math.ceil(shape.m / primitives.array_size)
    kt = math.ceil(shape.k / primitives.array_size)
    nt = math.ceil(shape.n / primitives.array_size)
    output_tiles = mt * nt
    padded_output_elems = output_tiles * primitives.array_size * primitives.array_size
    compute = (
        output_tiles * (primitives.pipeline_fill + primitives.pipeline_drain)
        + output_tiles * kt * primitives.kstep_per_tile
    )
    writeback = padded_output_elems * primitives.output_writeback_per_padded_elem
    return float(shape.count) * compute, float(shape.count) * writeback


def analytical_block_estimate(
    *,
    label: str,
    mode: str,                  # "prefill" or "decode"
    d_model: int,
    d_head: int,
    n_heads: int,
    n_kv_heads: int,
    ffn_hidden_dim: int,
    seq_len: int,               # T for prefill, decode_context for decode (KV history depth)
    n_layers: int = 1,
    dtype: str = "int16",
    cpu_cycles_per_mac: float = 19.43,
    primitives: HardwarePrimitives = DEFAULT_PRIMITIVES,
    preload_words_per_layer: int | None = None,
) -> BlockEstimate:
    """Predict cycles for `n_layers` blocks of the given configuration.

    All shape-driven; does not require a compiled artifact. Uses the same
    primitive constants as the compiled-plan path, so this estimate is what
    the model "would say" about a TinyLlama block once that block compiles.

    The block layout assumed here mirrors the QLlama reference plan:
      RMSNorm -> quantize -> [seg_qkv] -> RoPE-XFORM (on Q,K) ->
        per-head: dequant_score, score_scale, softmax, quant_probs,
                  causal_mask, [seg_score], [seg_value]
      concat -> [seg_o_proj] -> dequant -> residual -> RMSNorm -> quantize
      [seg_ffn_up] (gate_proj + up_proj fused) -> dequant_gate, dequant_up,
                  silu, mul, quantize -> [seg_ffn_down] -> dequant -> residual
      KV cache scatter (write or matrix-form)
    """
    M = seq_len if mode == "prefill" else 1
    T = seq_len  # KV history length (== prompt for prefill, == ctx for decode)
    kv_dim = n_kv_heads * d_head
    out_dim = n_heads * d_head

    # ---- NPU matmul shapes ----
    # Projections fuse Q/K/V into one segment in the QLlama plan; we still
    # account for each as a separate matmul body since the array runs them
    # back-to-back.
    matmuls: list[tuple[MatmulShape, str]] = [
        (MatmulShape("q_proj",   M, d_model, out_dim), "none"),
        (MatmulShape("k_proj",   M, d_model, kv_dim),  "none"),
        (MatmulShape("v_proj",   M, d_model, kv_dim),  "none"),
        # Per-head score and value matmuls
        (MatmulShape("score_qk", M, d_head, T, n_heads), "none"),
        (MatmulShape("value_av", M, T, d_head, n_heads), "none"),
        (MatmulShape("o_proj",   M, out_dim, d_model), "none"),
        (MatmulShape("gate_proj", M, d_model, ffn_hidden_dim), "none"),
        (MatmulShape("up_proj",   M, d_model, ffn_hidden_dim), "none"),
        (MatmulShape("down_proj", M, ffn_hidden_dim, d_model), "none"),
    ]

    matmul_body = 0.0
    writeback = 0.0
    macs = 0
    for shape, _act in matmuls:
        compute_part, writeback_part = _matmul_body_components(shape, primitives)
        matmul_body += compute_part
        writeback += writeback_part
        macs += shape.macs

    # ---- RoPE XFORMs (one per token row, applied to Q heads + K heads) ----
    # Prefill: M token rows, applied per (n_heads + n_kv_heads) head per row.
    # Decode: 1 token row, applied per (n_heads + n_kv_heads) head.
    rope_xform_count = (n_heads + n_kv_heads) * M
    rope_cycles = primitive_npu_rope_xform_cycles(d_head, rope_xform_count, primitives)

    # ---- Segment count ----
    # Mirrors the QLlama plan: qkv, score, value, o_proj, ffn_up, ffn_down.
    n_segments = 6
    # IM word counts roughly: qkv ~= 3 + n_kv_heads, score ~= n_heads, etc.
    # Conservative estimate based on QLlama A/B/C plans.
    im_words_total = 3 * (n_heads + n_kv_heads) + 4 * 2  # ~per-segment IM
    segment_cmd = float(n_segments) * primitives.segment_cmd_setup + \
                  float(im_words_total) * primitives.segment_cmd_decode_per_im_word
    segment_unmodeled = float(n_segments) * primitives.npu_segment_unmodeled_overhead

    # ---- Transfers (segment stage + readback) ----
    # Bytes moved across the UB boundary per segment, assuming the segment
    # body's lhs and (small) outputs are streamed; constant weights stay
    # resident.
    pack = {"int16": 1, "int8": 2, "int4": 4}[dtype]
    elem_words_in = M * d_model // primitives.array_size
    elem_words_qkv_out = M * (out_dim + 2 * kv_dim) // primitives.array_size
    elem_words_score_out = n_heads * M * T // primitives.array_size
    elem_words_value_out = n_heads * M * d_head // primitives.array_size
    elem_words_o_out = M * d_model // primitives.array_size
    elem_words_ffn_up_out = M * 2 * ffn_hidden_dim // primitives.array_size
    elem_words_ffn_down_out = M * d_model // primitives.array_size
    transfer_words = max(
        1,
        elem_words_in + elem_words_qkv_out + elem_words_score_out + elem_words_value_out
        + elem_words_o_out + elem_words_ffn_up_out + elem_words_ffn_down_out,
    )
    transfer_cycles = primitives.dma_setup * n_segments + transfer_words * primitives.ub_word_latency

    # ---- Static preload (constant weights resident in UB) ----
    # If the user supplied an explicit per-layer preload, use it; otherwise
    # estimate from the projection/FFN weight footprint. RoPE tables and biases
    # are small relative to weights; ignored here.
    if preload_words_per_layer is None:
        weight_words = 0
        for shape, _act in matmuls:
            if shape.name in {"score_qk", "value_av"}:
                continue  # not constant
            kt = math.ceil((shape.k // pack) / primitives.array_size)
            nt = math.ceil(shape.n / primitives.array_size)
            weight_words += int(shape.count) * kt * nt * primitives.array_size
    else:
        weight_words = int(preload_words_per_layer)
    static_preload = float(weight_words) * primitives.ub_word_latency

    # ---- Host op element counts (per layer, parameterised by shape) ----
    # rmsnorm: rows=M, width=d_model, applied twice (pre-attn, pre-ffn)
    rmsnorm_rows = M
    rmsnorm_width = d_model
    h_rmsnorm = 2 * rmsnorm_rows * (
        rmsnorm_width * (2 * primitives.load + primitives.store + 2 * primitives.f_mul + primitives.f_add)
        + primitives.f_sqrt + primitives.f_div
    )

    # quantize: input to attn (M*d_model), ffn input (M*d_model),
    #           probs after softmax (n_heads * M*T)
    quant_per_elem = (
        primitives.load + primitives.f_mul + primitives.f_add
        + primitives.clamp + primitives.store + primitives.branch
    )
    quant_elems = 2 * M * d_model + n_heads * M * T
    h_quant = quant_elems * quant_per_elem

    # dequantize: q,k,v outputs (M*out_dim + 2*M*kv_dim), o_proj (M*d_model),
    #             gate (M*ffn), up (M*ffn), ffn_down (M*d_model),
    #             scores per head (n_heads * M*T)
    dequant_elems = (
        M * out_dim + 2 * M * kv_dim
        + M * d_model
        + 2 * M * ffn_hidden_dim
        + M * d_model
        + n_heads * M * T
    )
    h_dequant = dequant_elems * quant_per_elem  # same primitive form

    # softmax: per head, on M*T (scores after causal mask)
    softmax_per_row = (
        T * (primitives.load + primitives.branch + primitives.store)
        + T * (primitives.load + primitives.f_sub + primitives.f_exp + primitives.f_add + primitives.store)
        + T * (primitives.load + primitives.f_div + primitives.store)
    )
    h_softmax = n_heads * M * softmax_per_row

    # silu on M*ffn_hidden_dim
    silu_per_elem = (
        primitives.load + primitives.f_exp + primitives.f_add
        + primitives.f_div + primitives.f_mul + primitives.store + primitives.branch
    )
    h_silu = M * ffn_hidden_dim * silu_per_elem

    # residual add: 2 per layer, on M*d_model
    add_per_elem = 2 * primitives.load + primitives.f_add + primitives.store + primitives.branch
    h_residual = 2 * M * d_model * add_per_elem

    # score scale mul: per head on M*T
    mul_per_elem = 2 * primitives.load + primitives.f_mul + primitives.store + primitives.branch
    h_score_scale = n_heads * M * T * mul_per_elem

    # ffn elementwise mul (gate * up): M*ffn_hidden_dim
    h_ffn_mul = M * ffn_hidden_dim * mul_per_elem

    # concat heads back: ~(n_heads-1) ops on M*d_head each (memcpy-ish)
    concat_per_elem = primitives.load + primitives.store + primitives.iadd + primitives.branch
    h_concat = max(0, n_heads - 1) * M * d_head * concat_per_elem

    # KV cache scatter: 2 ops, write M*kv_dim each (decode is M=1, prefill is M=T)
    scatter_per_elem = primitives.load + primitives.store
    h_kv_scatter = 2 * M * kv_dim * scatter_per_elem

    # causal mask: per head on M*T (one branch per element)
    mask_per_elem = primitives.load + primitives.branch + primitives.store
    h_causal_mask = n_heads * M * T * mask_per_elem

    # ---- CPU baselines ----
    cpu_baseline_matmul = 0.0
    for shape, act in matmuls:
        cpu_baseline_matmul += primitive_cpu_matmul_cycles(
            shape, activation=act, primitives=primitives
        )
    # If the user wants a strict "MACs * 19.43" comparison instead, sum macs.
    cpu_baseline_host = (
        h_rmsnorm + h_quant + h_dequant + h_softmax + h_silu
        + h_residual + h_score_scale + h_ffn_mul + h_concat
        + h_kv_scatter + h_causal_mask
    )

    return BlockEstimate(
        label=label,
        npu_matmul_body=matmul_body * n_layers,
        npu_writeback=writeback * n_layers,
        npu_rope_xforms=rope_cycles * n_layers,
        npu_activation=0.0,
        npu_segment_cmd=segment_cmd * n_layers,
        npu_segment_unmodeled=segment_unmodeled * n_layers,
        npu_transfer=transfer_cycles * n_layers,
        npu_static_preload=static_preload * n_layers,
        host_rmsnorm=h_rmsnorm * n_layers,
        host_quant=h_quant * n_layers,
        host_dequant=h_dequant * n_layers,
        host_softmax=h_softmax * n_layers,
        host_silu=h_silu * n_layers,
        host_residual=h_residual * n_layers,
        host_score_scale=h_score_scale * n_layers,
        host_ffn_mul=h_ffn_mul * n_layers,
        host_concat=h_concat * n_layers,
        host_kv_scatter=h_kv_scatter * n_layers,
        host_causal_mask=h_causal_mask * n_layers,
        cpu_baseline_matmul=cpu_baseline_matmul * n_layers,
        cpu_baseline_host=cpu_baseline_host * n_layers,
        macs=macs * n_layers,
        n_segments=n_segments * n_layers,
        n_layers=n_layers,
    )


def print_block_estimate(est: BlockEstimate) -> None:
    print(f"\n{est.label} (n_layers={est.n_layers}):")
    print(f"  macs={fmt(est.macs)}  segments={est.n_segments}")
    print("  NPU body breakdown:")
    print(f"    matmul_compute = {fmt(est.npu_matmul_body)}")
    print(f"    writeback      = {fmt(est.npu_writeback)}")
    print(f"    rope_xforms    = {fmt(est.npu_rope_xforms)}")
    print(f"    activation     = {fmt(est.npu_activation)}")
    print(f"    --- body total = {fmt(est.npu_body_total)}")
    print("  NPU segment overhead:")
    print(f"    cmd_decode     = {fmt(est.npu_segment_cmd)}")
    print(f"    unmodeled      = {fmt(est.npu_segment_unmodeled)}  (=0 means honest gap)")
    print(f"  NPU transfer     = {fmt(est.npu_transfer)}")
    print(f"  NPU static preload (cold) = {fmt(est.npu_static_preload)}")
    print("  Host op breakdown:")
    print(f"    rmsnorm        = {fmt(est.host_rmsnorm)}")
    print(f"    softmax        = {fmt(est.host_softmax)}")
    print(f"    silu           = {fmt(est.host_silu)}")
    print(f"    quantize       = {fmt(est.host_quant)}")
    print(f"    dequantize     = {fmt(est.host_dequant)}")
    print(f"    residual_add   = {fmt(est.host_residual)}")
    print(f"    score_scale    = {fmt(est.host_score_scale)}")
    print(f"    ffn_mul        = {fmt(est.host_ffn_mul)}")
    print(f"    concat_heads   = {fmt(est.host_concat)}")
    print(f"    kv_scatter     = {fmt(est.host_kv_scatter)}")
    print(f"    causal_mask    = {fmt(est.host_causal_mask)}")
    print(f"    --- host total = {fmt(est.host_total)}")
    print(f"  hybrid_hot       = {fmt(est.hybrid_hot)}")
    print(f"  hybrid_cold      = {fmt(est.hybrid_cold)}")
    print(f"  cpu_baseline     = {fmt(est.cpu_baseline)} "
          f"(matmul={fmt(est.cpu_baseline_matmul)}, host={fmt(est.cpu_baseline_host)})")
    if est.hybrid_hot > 0:
        print(f"  speedup hot/cold = "
              f"{est.cpu_baseline / est.hybrid_hot:.2f}x / "
              f"{est.cpu_baseline / est.hybrid_cold:.2f}x")


def primitive_npu_rope_xform_cycles(
    d_head: int,
    xform_count: int,
    primitives: HardwarePrimitives = DEFAULT_PRIMITIVES,
) -> float:
    if xform_count <= 0:
        return 0.0
    n_tiles = math.ceil(float(d_head) / float(primitives.array_size))
    half_count = max(1, n_tiles // 2)
    return float(xform_count) * (
        primitives.rope_xform_phase_cycles * float(half_count)
        + primitives.rope_xform_exit_cycles
    )


def primitive_cpu_matmul_cycles(
    shape: MatmulShape,
    *,
    activation: str = "none",
    bias: bool = True,
    requant: bool = True,
    primitives: HardwarePrimitives = DEFAULT_PRIMITIVES,
) -> float:
    out_elems = int(shape.count * shape.m * shape.n)
    macs = int(shape.macs)
    reads = out_elems * (2 * shape.k)
    muls = macs
    adds = macs + out_elems * (2 * shape.k)
    branches = macs
    writes = out_elems
    shifts = 0
    clamps = 0
    nonlinear = 0
    if bias:
        reads += out_elems
        adds += out_elems
    if requant:
        muls += out_elems
        shifts += out_elems
    if activation == "relu":
        clamps += out_elems
    elif activation in {"sigmoid", "h_gelu"}:
        nonlinear += out_elems
    clamps += out_elems
    adds += out_elems * 2
    branches += out_elems
    return (
        reads * primitives.load
        + writes * primitives.store
        + adds * primitives.iadd
        + muls * primitives.imul
        + branches * primitives.branch
        + shifts * primitives.ishift
        + clamps * primitives.clamp
        + nonlinear * primitives.nonlinear
    )


def primitive_host_op_cycles(
    artifact: Any,
    step: Any,
    primitives: HardwarePrimitives = DEFAULT_PRIMITIVES,
) -> float:
    if not step.outputs:
        return 0.0
    out = step.outputs[0]
    elems = float(_tensor_elements(artifact, out))
    rows = float(_tensor_rows(artifact, out))
    width = float(_tensor_last_dim(artifact, out))
    kind = str(step.kind)

    if kind in {"alias", "reshape", "transpose", "slice_row"}:
        return elems * (primitives.load + primitives.store + primitives.branch)
    if kind in {"k_cache_scatter_write", "v_cache_scatter_write", "k_cache_scatter_matrix", "v_cache_scatter_matrix"}:
        return elems * (primitives.load + primitives.store)
    if kind in {"concat_lastdim2", "layout_restore"}:
        return elems * (primitives.load + primitives.store + primitives.iadd + primitives.branch)
    if kind == "add":
        return elems * (2.0 * primitives.load + primitives.f_add + primitives.store + primitives.branch)
    if kind == "mul":
        return elems * (2.0 * primitives.load + primitives.f_mul + primitives.store + primitives.branch)
    if kind in {"quantize", "dequantize", "requantize"}:
        return elems * (
            primitives.load
            + primitives.f_mul
            + primitives.f_add
            + primitives.clamp
            + primitives.store
            + primitives.branch
        )
    if kind == "rmsnorm":
        return rows * (
            width * (2.0 * primitives.load + primitives.store + 2.0 * primitives.f_mul + primitives.f_add)
            + primitives.f_sqrt
            + primitives.f_div
        )
    if kind == "layernorm":
        return rows * (
            width * (
                3.0 * primitives.load
                + primitives.store
                + 3.0 * primitives.f_add
                + 3.0 * primitives.f_mul
            )
            + primitives.f_sqrt
            + primitives.f_div
        )
    if kind == "softmax_f16":
        return rows * (
            width * (primitives.load + primitives.branch + primitives.store)
            + width * (primitives.load + primitives.f_sub + primitives.f_exp + primitives.f_add + primitives.store)
            + width * (primitives.load + primitives.f_div + primitives.store)
        )
    if kind == "silu":
        return elems * (
            primitives.load
            + primitives.f_exp
            + primitives.f_add
            + primitives.f_div
            + primitives.f_mul
            + primitives.store
            + primitives.branch
        )
    if kind == "rope":
        half_elems = elems / 2.0
        return (
            half_elems * (primitives.f_trig * 2.0)
            + elems * (primitives.load + 2.0 * primitives.f_mul + primitives.f_add + primitives.store)
        )
    if kind == "causal_mask":
        return elems * (primitives.load + primitives.branch + primitives.store)
    return elems * (primitives.load + primitives.store + primitives.iadd + primitives.branch)


def _compiled_segment_io_words(artifact: Any, segment: Any) -> tuple[int, int]:
    _compiler_path()
    from tinynpu_jit import TensorKind

    segment_artifact = artifact.segment_artifacts[segment.name]
    produced_inside = {op.out for op in segment.ops}
    write_words = 0
    for tensor_name in segment.inputs:
        spec = artifact.plan.tensors[tensor_name]
        if spec.kind == TensorKind.CONSTANT or tensor_name in produced_inside:
            continue
        symbol = segment_artifact.symbol_table.get(tensor_name)
        if symbol is not None:
            write_words += int(symbol["word_count"])

    read_words = 0
    for tensor_name in segment.outputs:
        symbol = segment_artifact.symbol_table.get(tensor_name)
        if symbol is not None:
            read_words += int(symbol["word_count"])
    return write_words, read_words


def _benchmark_entries_by_step(entries: list[dict[str, Any]]) -> dict[str, deque[dict[str, Any]]]:
    by_step: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for entry in entries:
        by_step[str(entry["step"])].append(entry)
    return by_step


def _segment_costs(
    artifact: Any,
    segment: Any,
    *,
    npu_models: dict[str, NpuTileModel],
    cpu_cycles_per_mac: float,
    preload_cycles_per_word: float,
    attention_dtype: str,
    cpu_count_cycles: float,
    primitives: HardwarePrimitives = DEFAULT_PRIMITIVES,
) -> StepCostRow:
    macs = 0
    npu_body = 0.0
    primitive_npu_body = 0.0
    primitive_cpu = 0.0
    shapes: list[str] = []
    rope_xforms = 0
    for op in segment.ops:
        shape = _op_shape(artifact, op)
        op_rope_xforms = len(op.rope_xforms())
        macs += shape.macs
        rope_xforms += op_rope_xforms
        dtype = attention_dtype if _is_attention_op(segment.name, op.name) else str(op.in_dtype.value)
        if dtype not in npu_models:
            dtype = "int16"
        npu_body += npu_models[dtype].body_cycles(shape)
        out_elems = shape.count * shape.m * shape.n
        primitive_npu_body += primitive_npu_matmul_body_cycles(shape, primitives)
        primitive_npu_body += primitive_npu_activation_cycles(op.activation, out_elems, primitives)
        primitive_npu_body += primitive_npu_rope_xform_cycles(shape.n, op_rope_xforms, primitives)
        primitive_cpu += primitive_cpu_matmul_cycles(
            shape,
            activation=op.activation,
            bias=op.bias is not None,
            primitives=primitives,
        )
        shapes.append(f"{op.name}:{shape.m}x{shape.k}x{shape.n}/{dtype}")

    stage_words, readback_words = _compiled_segment_io_words(artifact, segment)
    transfer_words = stage_words + readback_words
    im_words = int(artifact.segment_artifacts[segment.name].im_words)
    primitive_transfer = primitives.dma_setup + float(transfer_words) * primitives.ub_word_latency
    primitive_launch = (
        primitives.npu_segment_unmodeled_overhead
        + primitives.segment_cmd_setup
        + float(im_words) * primitives.segment_cmd_decode_per_im_word
    )
    return StepCostRow(
        index="",
        kind="NPU",
        name=segment.name,
        detail=f"ops={len(segment.ops)} stage={stage_words} read={readback_words} " + ";".join(shapes),
        cpu_saved_cycles=float(macs) * cpu_cycles_per_mac,
        cpu_count_cycles=float(cpu_count_cycles),
        npu_body_cycles=npu_body,
        transfer_cycles=float(transfer_words) * preload_cycles_per_word,
        primitive_npu_cycles=primitive_launch + primitive_npu_body + primitive_transfer,
        primitive_cpu_replaced_cycles=primitive_cpu,
        macs=macs,
        words=transfer_words,
        rope_xforms=rope_xforms,
    )


def compiled_plan_step_cost_rows(
    artifact: Any,
    *,
    npu_models: dict[str, NpuTileModel],
    cpu_cycles_per_mac: float,
    preload_cycles_per_word: float,
    attention_dtype: str,
    benchmark_entries: list[dict[str, Any]],
    primitives: HardwarePrimitives = DEFAULT_PRIMITIVES,
) -> list[StepCostRow]:
    _compiler_path()
    from tinynpu_jit import HostOp, NpuSegment, VerifyTensor

    rows: list[StepCostRow] = []
    entries_by_step = _benchmark_entries_by_step(benchmark_entries)

    static_ub_words = len(artifact.static_ub_image or [])
    if static_ub_words:
        rows.append(
            StepCostRow(
                index="pre",
                kind="PRELOAD",
                name="static_ub",
                detail="constant weights/bias/tables resident in UB image",
                preload_cycles=float(static_ub_words) * preload_cycles_per_word,
                # Static preload is already primitive-shaped by word latency.
                primitive_npu_cycles=0.0,
                words=static_ub_words,
            )
        )

    for step in artifact.plan.steps:
        if not isinstance(step, NpuSegment):
            continue
        im_words = int(artifact.segment_artifacts[step.name].im_words)
        rows.append(
            StepCostRow(
                index="pre",
                kind="PRELOAD",
                name=f"{step.name}.im",
                detail=f"instruction memory words={im_words}",
                preload_cycles=23.0 + 32.0 * float(im_words),
                words=im_words,
            )
        )

    for index, step in enumerate(artifact.plan.steps):
        if isinstance(step, NpuSegment):
            entry = entries_by_step[step.name].popleft() if entries_by_step.get(step.name) else {}
            row = _segment_costs(
                artifact,
                step,
                npu_models=npu_models,
                cpu_cycles_per_mac=cpu_cycles_per_mac,
                preload_cycles_per_word=preload_cycles_per_word,
                attention_dtype=attention_dtype,
                cpu_count_cycles=float(entry.get("cycles", 0)),
                primitives=primitives,
            )
            rows.append(replace(row, index=str(index)))
            continue
        if isinstance(step, HostOp):
            entry = entries_by_step[step.name].popleft() if entries_by_step.get(step.name) else {}
            bucket = str(entry.get("bucket", "unmeasured"))
            cycles = float(entry.get("cycles", 0))
            rows.append(
                StepCostRow(
                    index=str(index),
                    kind="HOST",
                    name=step.name,
                    detail=f"{step.kind} bucket={bucket}",
                    host_cycles=cycles if bucket == "host_intrinsic" else 0.0,
                    cpu_count_cycles=cycles if bucket != "host_intrinsic" else 0.0,
                    primitive_host_cycles=primitive_host_op_cycles(artifact, step, primitives) if bucket == "host_intrinsic" else 0.0,
                )
            )
            continue
        if isinstance(step, VerifyTensor):
            rows.append(
                StepCostRow(
                    index=str(index),
                    kind="VERIFY",
                    name=step.label,
                    detail=step.tensor_name,
                )
            )
    return rows


def qllama_saved_rtl_cases() -> list[QLlamaRtlCase]:
    return [
        QLlamaRtlCase("A", "prefill", 8, 8, 1, 1, 8, 8, 171710.0, measured_cpu=224843.0),
        QLlamaRtlCase("A", "decode", 8, 8, 1, 1, 8, 8, 60591.0, measured_cpu=35852.0),
        QLlamaRtlCase("B", "prefill", 16, 8, 2, 1, 16, 8, 289401.0, measured_cpu=534748.0),
        QLlamaRtlCase("B", "decode", 16, 8, 2, 1, 16, 8, 89772.0, measured_cpu=77480.0),
        QLlamaRtlCase("C", "prefill", 32, 8, 4, 2, 32, 8, 549512.0, measured_npu_hot=536302.0, cpu_lower_bound=1339611.0),
        QLlamaRtlCase("C", "decode", 32, 8, 4, 2, 32, 8, 149867.0, measured_npu_hot=136650.0, measured_cpu=209120.0),
    ]


def _qllama_args_from_case(case: QLlamaRtlCase) -> argparse.Namespace:
    return argparse.Namespace(
        d_model=case.d_model,
        d_head=case.d_head,
        n_heads=case.n_heads,
        n_kv_heads=case.n_kv_heads,
        ffn_hidden_dim=case.ffn_hidden_dim,
        prompt_len=case.prompt_len,
        seed=0,
    )


def _qllama_args_from_namespace(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        d_model=args.d_model,
        d_head=args.d_head,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        ffn_hidden_dim=args.ffn_hidden_dim,
        prompt_len=args.prompt_len,
        seed=args.seed,
    )


def _qllama_plan_features(
    label: str,
    mode: str,
    args: argparse.Namespace,
    *,
    npu_models: dict[str, NpuTileModel],
    cpu_cycles_per_mac: float,
    preload_cycles_per_word: float,
    attention_dtype: str,
) -> QLlamaPlanFeatures:
    _compiler_path()
    from tinynpu_jit import HostOp, NpuSegment, VerificationMode, five_stage_in_order_model

    artifact = _build_qllama_artifact(args, mode)
    result = artifact.run_host_emulation(
        {},
        verification=VerificationMode.OFF,
        benchmark=True,
        cost_model=five_stage_in_order_model(),
    )
    bench = result.benchmark.to_dict() if result.benchmark is not None else {"entries": []}
    rows = compiled_plan_step_cost_rows(
        artifact,
        npu_models=npu_models,
        cpu_cycles_per_mac=cpu_cycles_per_mac,
        preload_cycles_per_word=preload_cycles_per_word,
        attention_dtype=attention_dtype,
        benchmark_entries=list(bench.get("entries", [])),
    )

    host_steps = 0
    segments = 0
    matmul_ops = 0
    rope_xforms = 0
    for step in artifact.plan.steps:
        if isinstance(step, HostOp):
            host_steps += 1
        elif isinstance(step, NpuSegment):
            segments += 1
            for op in step.ops:
                matmul_ops += 1
                rope_xforms += len(op.rope_xforms())

    memory_report = artifact.memory_report
    ub_peak = int(memory_report.total_ub_peak) if memory_report is not None else 0
    ub_static = int(memory_report.static_zone_end) if memory_report is not None else len(artifact.static_ub_image or [])
    ub_capacity = max((int(sp.ub_capacity) for sp in memory_report.segments), default=0) if memory_report is not None else 0

    host_cycles = sum(row.primitive_host_cycles for row in rows)
    npu_hot = sum(row.primitive_npu_cycles for row in rows if row.kind == "NPU")
    cpu_replaced = sum(row.primitive_cpu_replaced_cycles for row in rows)
    preload_cycles = sum(row.preload_cycles for row in rows)
    npu_non_setup = npu_hot - float(segments) * DEFAULT_PRIMITIVES.npu_segment_unmodeled_overhead

    return QLlamaPlanFeatures(
        label=label,
        mode=mode,
        d_model=int(args.d_model),
        d_head=int(args.d_head),
        n_heads=int(args.n_heads),
        n_kv_heads=int(args.n_kv_heads),
        ffn_hidden_dim=int(args.ffn_hidden_dim),
        prompt_len=int(args.prompt_len),
        steps=len(artifact.plan.steps),
        host_steps=host_steps,
        segments=segments,
        matmul_ops=matmul_ops,
        rope_xforms=rope_xforms,
        ub_static_words=ub_static,
        ub_peak_words=ub_peak,
        ub_capacity_words=ub_capacity,
        preload_cycles=preload_cycles,
        host_cycles=host_cycles,
        npu_hot_cycles=npu_hot,
        npu_non_setup_cycles=npu_non_setup,
        cpu_replaced_cycles=cpu_replaced,
    )


def _fit_qllama_structured_model(training: list[tuple[QLlamaRtlCase, QLlamaPlanFeatures]]) -> QLlamaStructuredFit:
    cpu_matrix: list[list[float]] = []
    cpu_vector: list[float] = []
    for case, features in training:
        if case.measured_cpu is None:
            continue
        cpu_matrix.append([features.host_cycles, features.cpu_replaced_cycles, 1.0])
        cpu_vector.append(case.measured_cpu)
    cpu_host_scale, cpu_matmul_scale, cpu_fixed = _least_squares(cpu_matrix, cpu_vector)

    npu_matrix: list[list[float]] = []
    npu_vector: list[float] = []
    for case, features in training:
        npu_matrix.append([features.host_cycles, features.npu_non_setup_cycles, float(features.segments)])
        npu_vector.append(case.measured_npu_cold - features.preload_cycles)
    npu_host_scale, npu_non_setup_scale, npu_segment_setup = _least_squares(npu_matrix, npu_vector)

    return QLlamaStructuredFit(
        cpu_host_scale=cpu_host_scale,
        cpu_matmul_scale=cpu_matmul_scale,
        cpu_fixed_cycles=cpu_fixed,
        npu_host_scale=npu_host_scale,
        npu_non_setup_scale=npu_non_setup_scale,
        npu_segment_setup_cycles=npu_segment_setup,
    )


def _print_qllama_fit_row(
    case_label: str,
    features: QLlamaPlanFeatures,
    fit: QLlamaStructuredFit,
    *,
    measured_npu_cold: float | None = None,
    measured_cpu: float | None = None,
    cpu_lower_bound: float | None = None,
) -> None:
    npu_cold = fit.predict_npu_cold(features)
    npu_hot = fit.predict_npu_hot(features)
    cpu = fit.predict_cpu(features)
    speedup = cpu / npu_cold if npu_cold else float("inf")
    npu_err = _err_pct(npu_cold, measured_npu_cold) if measured_npu_cold is not None else "-"
    if measured_cpu is not None:
        cpu_ref = fmt(measured_cpu)
        cpu_err = _err_pct(cpu, measured_cpu)
    elif cpu_lower_bound is not None:
        cpu_ref = f">{fmt(cpu_lower_bound)}"
        cpu_err = "lower-bound"
    else:
        cpu_ref = "-"
        cpu_err = "-"
    print(
        "  "
        + f"{case_label[:34]:<34} {features.steps:>5} {features.host_steps:>5} {features.segments:>4} "
        + f"{features.rope_xforms:>6} {fmt(features.ub_peak_words):>8} "
        + f"{fmt(npu_hot):>11} {fmt(npu_cold):>11} "
        + f"{(fmt(measured_npu_cold) if measured_npu_cold is not None else '-'):>11} {npu_err:>8} "
        + f"{fmt(cpu):>11} {cpu_ref:>11} {cpu_err:>11} {speedup:>7.2f}x"
    )


def print_qllama_structured_fit(
    args: argparse.Namespace,
    *,
    npu_models: dict[str, NpuTileModel],
    cpu_cycles_per_mac: float,
    preload_cycles_per_word: float,
) -> None:
    cases = qllama_saved_rtl_cases()
    training: list[tuple[QLlamaRtlCase, QLlamaPlanFeatures]] = []
    for case in cases:
        training.append(
            (
                case,
                _qllama_plan_features(
                    case.label,
                    case.mode,
                    _qllama_args_from_case(case),
                    npu_models=npu_models,
                    cpu_cycles_per_mac=cpu_cycles_per_mac,
                    preload_cycles_per_word=preload_cycles_per_word,
                    attention_dtype=args.attention_dtype,
                ),
            )
        )
    fit = _fit_qllama_structured_model(training)
    max_train_xforms = max(features.rope_xforms for _, features in training)

    print("\nQLlama structured sweep fit (DIAGNOSTIC — NOT a forward-path model):")
    print("  warning: this is a least-squares fit on saved QLlama A/B/C RTL points.")
    print("    With 3 free parameters per equation and 4-6 training points the")
    print("    reported residuals are fit error, NOT generalization error.")
    print("    All training points have rope_xforms=0; any extrapolation row that")
    print("    has rope_xforms > 0 is outside the fit envelope. Use this output")
    print("    to sanity-check that the structural primitive model lands in a")
    print("    similar regime, NOT to read off speedup numbers.")
    print("  source: saved QLlamaBlock RTL O3 fast-sim points from commit 034f7e4")
    print(
        "  NPU cold fit: preload + "
        f"{fit.npu_host_scale:.3f}*host_primitive + "
        f"{fit.npu_non_setup_scale:.3f}*npu_non_setup + "
        f"{fit.npu_segment_setup_cycles:.1f}*segments"
    )
    print(
        "  CPU fit: "
        f"{fit.cpu_host_scale:.3f}*host_primitive + "
        f"{fit.cpu_matmul_scale:.3f}*cpu_replaced + "
        f"{fit.cpu_fixed_cycles:.1f}"
    )
    if max_train_xforms == 0:
        print(
            "  note: all saved fit points have rope_xforms=0; RoPE-XFORM uses the RTL source formula "
            "(6*half_count+1 cycles), not a learned QLlama RTL coefficient."
        )

    print(
        "  "
        + f"{'case':<34} {'steps':>5} {'host':>5} {'seg':>4} {'xform':>6} {'ub_peak':>8} "
        + f"{'npu_hot':>11} {'npu_cold':>11} {'meas_npu':>11} {'npu_err':>8} "
        + f"{'cpu_fit':>11} {'cpu_ref':>11} {'cpu_err':>11} {'speed':>8}"
    )
    for case, features in training:
        _print_qllama_fit_row(
            case.label,
            features,
            fit,
            measured_npu_cold=case.measured_npu_cold,
            measured_cpu=case.measured_cpu,
            cpu_lower_bound=case.cpu_lower_bound,
        )

    target_args = _qllama_args_from_namespace(args)
    targets: list[tuple[str, argparse.Namespace]] = [
        ("target", target_args),
        (
            "d64_h16_nh4_nkv2_f64_t8",
            argparse.Namespace(d_model=64, d_head=16, n_heads=4, n_kv_heads=2, ffn_hidden_dim=64, prompt_len=8, seed=args.seed),
        ),
        (
            "d128_h16_nh8_nkv2_f128_t8",
            argparse.Namespace(d_model=128, d_head=16, n_heads=8, n_kv_heads=2, ffn_hidden_dim=128, prompt_len=8, seed=args.seed),
        ),
    ]
    seen: set[tuple[int, int, int, int, int, int]] = set()
    print("\nQLlama fit extrapolation:")
    print(
        "  "
        + f"{'target':<34} {'steps':>5} {'host':>5} {'seg':>4} {'xform':>6} {'ub_peak':>8} "
        + f"{'npu_hot':>11} {'npu_cold':>11} {'meas_npu':>11} {'npu_err':>8} "
        + f"{'cpu_fit':>11} {'cpu_ref':>11} {'cpu_err':>11} {'speed':>8}"
    )
    for target_name, target in targets:
        key = (target.d_model, target.d_head, target.n_heads, target.n_kv_heads, target.ffn_hidden_dim, target.prompt_len)
        if key in seen:
            continue
        seen.add(key)
        for mode in ("prefill", "decode"):
            label = f"{target_name}-{mode}"
            try:
                features = _qllama_plan_features(
                    label,
                    mode,
                    target,
                    npu_models=npu_models,
                    cpu_cycles_per_mac=cpu_cycles_per_mac,
                    preload_cycles_per_word=preload_cycles_per_word,
                    attention_dtype=args.attention_dtype,
                )
            except Exception as exc:
                print("  " + f"{label[:34]:<34} build failed: {exc}")
                continue
            _print_qllama_fit_row(label, features, fit)
            if max_train_xforms == 0 and features.rope_xforms:
                print(
                    "    caution: "
                    + f"{label} has {features.rope_xforms} RoPE XFORMs outside the saved QLlama fit envelope."
                )


def _print_step_cost_table(rows: list[StepCostRow], *, calibration: RuntimeCalibration, host_expensive_scale: float) -> None:
    headers = ["idx", "kind", "name", "cpu_saved", "cpu_count", "host", "npu", "xfer", "preload", "hot", "cold", "detail"]
    print("\n  Step cost breakdown:")
    if calibration.name == "raw":
        print(
            "  "
            + f"{headers[0]:>4} {headers[1]:<8} {headers[2]:<28} {headers[3]:>10} {headers[4]:>10} "
            + f"{headers[5]:>9} {headers[6]:>9} {headers[7]:>9} {headers[8]:>9} "
            + f"{headers[9]:>9} {headers[10]:>9} {headers[11]}"
        )
    else:
        print(
            "  "
            + f"{headers[0]:>4} {headers[1]:<8} {headers[2]:<28} {headers[3]:>10} {headers[4]:>10} "
            + f"{headers[5]:>9} {headers[6]:>9} {headers[7]:>9} {headers[8]:>9} "
            + f"{'raw_hot':>9} {'cal_hot':>9} {'cal_cold':>9} {headers[11]}"
        )
    for row in rows:
        if calibration.name == "raw":
            print(
                "  "
                + f"{row.index:>4} {row.kind:<8} {row.name[:28]:<28} "
                + f"{fmt(row.cpu_saved_cycles):>10} {fmt(row.cpu_count_cycles):>10} "
                + f"{fmt(row.host_cycles):>9} {fmt(row.npu_body_cycles):>9} "
                + f"{fmt(row.transfer_cycles):>9} {fmt(row.preload_cycles):>9} "
                + f"{fmt(row.hybrid_hot_cycles):>9} {fmt(row.hybrid_cold_cycles):>9} "
                + row.detail[:120]
            )
        else:
            if calibration.name == "primitive":
                cal_hot = row.primitive_host_cycles + row.primitive_npu_cycles
                cal_cold = cal_hot + row.preload_cycles
            else:
                cal_hot = calibration.calibrated_hot_cycles(row, host_expensive_scale=host_expensive_scale)
                cal_cold = calibration.calibrated_cold_cycles(row, host_expensive_scale=host_expensive_scale)
            print(
                "  "
                + f"{row.index:>4} {row.kind:<8} {row.name[:28]:<28} "
                + f"{fmt(row.cpu_saved_cycles):>10} {fmt(row.cpu_count_cycles):>10} "
                + f"{fmt(row.host_cycles):>9} {fmt(row.npu_body_cycles):>9} "
                + f"{fmt(row.transfer_cycles):>9} {fmt(row.preload_cycles):>9} "
                + f"{fmt(row.hybrid_hot_cycles):>9} {fmt(cal_hot):>9} "
                + f"{fmt(cal_cold):>9} "
                + row.detail[:120]
            )


def summarize_compiled_plan(
    label: str,
    artifact: Any,
    *,
    npu_models: dict[str, NpuTileModel],
    cpu_cycles_per_mac: float,
    preload_cycles_per_word: float,
    attention_dtype: str,
    show_step_costs: bool,
    calibration: RuntimeCalibration,
) -> None:
    _compiler_path()
    from tinynpu_jit import HostOp, NpuSegment, VerificationMode, five_stage_in_order_model

    step_counts: Counter[str] = Counter()
    host_kind_counts: Counter[str] = Counter()
    matmul_ops = 0
    rope_xforms = 0
    macs = 0
    npu_body = 0.0
    stage_words = 0
    readback_words = 0

    for step in artifact.plan.steps:
        if isinstance(step, HostOp):
            step_counts["host"] += 1
            host_kind_counts[str(step.kind)] += 1
        elif isinstance(step, NpuSegment):
            step_counts["segment"] += 1
            writes, reads = _compiled_segment_io_words(artifact, step)
            stage_words += writes
            readback_words += reads
            for op in step.ops:
                matmul_ops += 1
                rope_xforms += len(op.rope_xforms())
                shape = _op_shape(artifact, op)
                macs += shape.macs
                dtype = attention_dtype if _is_attention_op(step.name, op.name) else str(op.in_dtype.value)
                if dtype not in npu_models:
                    dtype = "int16"
                npu_body += npu_models[dtype].body_cycles(shape)
        else:
            step_counts[step.__class__.__name__] += 1

    result = artifact.run_host_emulation(
        {},
        verification=VerificationMode.OFF,
        benchmark=True,
        cost_model=five_stage_in_order_model(),
    )
    bench = result.benchmark.to_dict() if result.benchmark is not None else {"totals": {}}
    totals = bench["totals"]
    cost_rows = compiled_plan_step_cost_rows(
        artifact,
        npu_models=npu_models,
        cpu_cycles_per_mac=cpu_cycles_per_mac,
        preload_cycles_per_word=preload_cycles_per_word,
        attention_dtype=attention_dtype,
        benchmark_entries=list(bench.get("entries", [])),
    )
    host_intrinsic = int(totals.get("host_intrinsic_cycles", 0))
    cpu_replaced_count_model = int(totals.get("cpu_replaced_cycles", 0))
    cpu_replaced_saved_model = float(macs) * cpu_cycles_per_mac

    static_ub_words = len(artifact.static_ub_image or [])
    im_cycles = 0.0
    im_instructions = 0
    for step in artifact.plan.steps:
        if isinstance(step, NpuSegment):
            im_count = int(artifact.segment_artifacts[step.name].im_words)
            im_instructions += im_count
            # Calibrated from saved logs: one matmul+halt segment is 87 cycles,
            # four-matmul+halt segment is 183 cycles.
            im_cycles += 23.0 + 32.0 * float(im_count)
    static_preload_cycles = static_ub_words * preload_cycles_per_word + im_cycles
    transfer_words = stage_words + readback_words
    transfer_cycles = transfer_words * preload_cycles_per_word
    hybrid_hot = host_intrinsic + npu_body + transfer_cycles
    hybrid_cold = static_preload_cycles + hybrid_hot
    cpu_e2e_saved_model = host_intrinsic + cpu_replaced_saved_model
    calibrated_totals = _compiled_plan_cycle_totals(
        cost_rows,
        cpu_replaced_saved_model=cpu_replaced_saved_model,
        calibration=calibration,
        host_expensive_scale=_host_expensive_scale_for_plan(artifact, label, calibration),
    )
    host_expensive_scale = _host_expensive_scale_for_plan(artifact, label, calibration)

    memory_report = artifact.memory_report
    ub_peak = int(memory_report.total_ub_peak) if memory_report is not None else 0
    ub_static = int(memory_report.static_zone_end) if memory_report is not None else static_ub_words
    ub_capacity = max((int(sp.ub_capacity) for sp in memory_report.segments), default=0) if memory_report is not None else 0
    fits = ub_peak <= ub_capacity if ub_capacity else True

    print(f"\nCompiled QLlama {label} plan estimate:")
    print(
        f"  steps={sum(step_counts.values())} "
        f"(host={step_counts['host']}, segments={step_counts['segment']}, verify={step_counts['VerifyTensor']}) "
        f"matmul_ops={matmul_ops} rope_xforms={rope_xforms}"
    )
    if host_kind_counts:
        host_kinds = ", ".join(f"{kind}:{count}" for kind, count in sorted(host_kind_counts.items()))
        print(f"  host_kinds={host_kinds}")
    print(
        f"  ub_static={ub_static} words, ub_peak={ub_peak}/{ub_capacity} words, "
        f"fits_current_ub={fits}, im_instructions={im_instructions}"
    )
    print(
        f"  transfer_words: stage={stage_words}, readback={readback_words}, total={transfer_words}"
    )
    print(
        f"  cycles: host_model={fmt(host_intrinsic)}, npu_body_model={fmt(npu_body)}, "
        f"transfer_model={fmt(transfer_cycles)}, preload_model={fmt(static_preload_cycles)}"
    )
    if calibration.name != "raw":
        segment_hot_cal = (
            calibrated_totals["npu_cal"]
            if calibration.name == "primitive"
            else calibrated_totals["npu_cal"] + calibrated_totals["transfer"]
        )
        print(
            f"  calibrated({calibration.name}): host={fmt(calibrated_totals['host_cal'])}, "
            f"segment_hot={fmt(segment_hot_cal)}, "
            f"hybrid_hot={fmt(calibrated_totals['hybrid_hot_cal'])}, "
            f"hybrid_cold={fmt(calibrated_totals['hybrid_cold_cal'])}"
            + ("" if calibration.name == "primitive" else f", expensive_host_scale={host_expensive_scale:.2f}x")
        )
    print(
        f"  cpu_segments_saved_model={fmt(cpu_replaced_saved_model)} "
        f"(count_model={fmt(cpu_replaced_count_model)})"
    )
    print(
        f"  raw e2e: cpu_hot_model={fmt(cpu_e2e_saved_model)}, "
        f"hybrid_hot_model={fmt(hybrid_hot)}, hybrid_cold_model={fmt(hybrid_cold)}, "
        f"hot_speedup={cpu_e2e_saved_model / hybrid_hot:.2f}x, "
        f"cold_speedup={cpu_e2e_saved_model / hybrid_cold:.2f}x"
    )
    if calibration.name != "raw":
        print(
            f"  calibrated e2e: cpu_hot_model={fmt(calibrated_totals['cpu_hot_cal'])}, "
            f"hybrid_hot_model={fmt(calibrated_totals['hybrid_hot_cal'])}, "
            f"hybrid_cold_model={fmt(calibrated_totals['hybrid_cold_cal'])}, "
            f"hot_speedup={calibrated_totals['cpu_hot_cal'] / calibrated_totals['hybrid_hot_cal']:.2f}x, "
            f"cold_speedup={calibrated_totals['cpu_hot_cal'] / calibrated_totals['hybrid_cold_cal']:.2f}x"
        )
    if show_step_costs:
        _print_step_cost_table(cost_rows, calibration=calibration, host_expensive_scale=host_expensive_scale)


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer a TinyNPU analytical performance model from saved RTL runs.")
    parser.add_argument("--prompt-len", type=int, default=128)
    parser.add_argument("--decode-context", type=int, default=128)
    parser.add_argument("--dtype", choices=("int16", "int8", "int4"), default="int16")
    parser.add_argument("--attention-dtype", choices=("int16",), default="int16")
    parser.add_argument("--compiled-plan", choices=("none", "qllama-prefill", "qllama-decode", "qllama-both"), default="none")
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--d-head", type=int, default=8)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--ffn-hidden-dim", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-step-costs", action="store_true", help="Hide per-ExecutionPlan-step analytical costs.")
    parser.add_argument("--calibration", choices=("primitive", "raw", "block-rtl"), default="primitive")
    parser.add_argument("--validate-report-cases", action="store_true", help="Compare the fitted model against saved small report cases.")
    parser.add_argument("--validate-block-calibration", action="store_true", help="Compare calibrated QLlama estimates against saved RTL commit data.")
    parser.add_argument("--fit-qllama-sweep", action="store_true", help="DIAGNOSTIC ONLY: least-squares fit on saved QLlama RTL points. Not a forward-path model.")
    parser.add_argument("--analytical-block", action="store_true",
                        help="Run the shape-driven analytical block estimator. Uses HardwarePrimitives only; no compiled artifact required.")
    parser.add_argument("--analytical-target", choices=("tinyllama-22", "qllama-c", "custom"), default="tinyllama-22",
                        help="Predefined block configuration to estimate.")
    parser.add_argument("--analytical-prompt-len", type=int, default=128)
    parser.add_argument("--analytical-decode-context", type=int, default=128)
    args = parser.parse_args()
    calibration = runtime_calibration(args.calibration)

    rows = load_gemm_rows()
    npu_models = fit_npu_models(rows)
    preload_cycles_per_word = infer_preload_cycles_per_word(rows)
    cpu_cycles_per_mac, cpu_samples = infer_cpu_cycles_per_mac()
    print("Fitted NPU body models from runs/gemm_v2_e2e_compile_lhs:")
    for dtype, fitted in sorted(npu_models.items()):
        print(
            f"  {dtype}: cycles = {fitted.fixed_cycles:.1f}"
            f" + {fitted.kstep_tile_cycles:.3f}*kstep_tiles"
            f" + {fitted.output_tile_cycles:.3f}*output_tiles"
        )
    print(f"Fitted static preload: {preload_cycles_per_word:.3f} cycles per 128-bit UB word")
    print("CPU INT16 fallback samples from report:")
    for name, value in cpu_samples:
        print(f"  {name}: {value:.2f} cycles/MAC")
    print(f"CPU model average: {cpu_cycles_per_mac:.2f} cycles/MAC")
    print(f"Runtime calibration: {calibration.name} ({calibration.source})")
    if args.validate_report_cases:
        print_report_case_validation(
            npu_models=npu_models,
            cpu_cycles_per_mac=cpu_cycles_per_mac,
            calibration=calibration,
        )
    if args.validate_block_calibration:
        print_qllama_calibration_validation(
            npu_models=npu_models,
            cpu_cycles_per_mac=cpu_cycles_per_mac,
            preload_cycles_per_word=preload_cycles_per_word,
            calibration=calibration,
        )
    if args.fit_qllama_sweep:
        print_qllama_structured_fit(
            args,
            npu_models=npu_models,
            cpu_cycles_per_mac=cpu_cycles_per_mac,
            preload_cycles_per_word=preload_cycles_per_word,
        )

    if args.analytical_block:
        configs = {
            "tinyllama-22": dict(d_model=2048, d_head=64, n_heads=32, n_kv_heads=4, ffn_hidden_dim=5632, n_layers=22),
            "qllama-c":     dict(d_model=32, d_head=8, n_heads=4, n_kv_heads=2, ffn_hidden_dim=32, n_layers=1),
            "custom":       dict(
                d_model=args.d_model, d_head=args.d_head, n_heads=args.n_heads,
                n_kv_heads=args.n_kv_heads, ffn_hidden_dim=args.ffn_hidden_dim, n_layers=1,
            ),
        }
        cfg = configs[args.analytical_target]
        print(f"\n=== Analytical block estimate ({args.analytical_target}) ===")
        print(f"config: {cfg}")
        print("All numbers are derived from HardwarePrimitives. Errors are attributable")
        print("to specific primitives (writeback / softfloat / segment overhead).")

        prefill = analytical_block_estimate(
            label=f"{args.analytical_target} prefill T={args.analytical_prompt_len}",
            mode="prefill",
            seq_len=args.analytical_prompt_len,
            cpu_cycles_per_mac=cpu_cycles_per_mac,
            **cfg,
        )
        decode = analytical_block_estimate(
            label=f"{args.analytical_target} decode ctx={args.analytical_decode_context}",
            mode="decode",
            seq_len=args.analytical_decode_context,
            cpu_cycles_per_mac=cpu_cycles_per_mac,
            **cfg,
        )
        print_block_estimate(prefill)
        print_block_estimate(decode)
        print()
        print("Primitive uncertainty notes:")
        print(f"  softfloat (F_EXP={DEFAULT_PRIMITIVES.f_exp}, F_DIV={DEFAULT_PRIMITIVES.f_div}, "
              f"F_SQRT={DEFAULT_PRIMITIVES.f_sqrt}, F_TRIG={DEFAULT_PRIMITIVES.f_trig}) — placeholder.")
        print(f"  npu_segment_unmodeled_overhead={DEFAULT_PRIMITIVES.npu_segment_unmodeled_overhead} — explicit gap.")
        print(f"  output_writeback_per_padded_elem={DEFAULT_PRIMITIVES.output_writeback_per_padded_elem} — full-tile-only fit.")
        print("  The above three primitives drive most of the QLlama validation residual.")

    prefill_shapes, decode_shapes = tinyllama_linear_shapes(args.prompt_len, decode_context=args.decode_context)
    summaries = [
        summarize_workload(
            f"TinyLlama prefill T={args.prompt_len} resident",
            prefill_shapes,
            npu_models=npu_models,
            cpu_cycles_per_mac=cpu_cycles_per_mac,
            preload_cycles_per_word=preload_cycles_per_word,
            linear_dtype=args.dtype,
            attention_dtype=args.attention_dtype,
            stream_weights=False,
        ),
        summarize_workload(
            f"TinyLlama prefill T={args.prompt_len} streamed weights",
            prefill_shapes,
            npu_models=npu_models,
            cpu_cycles_per_mac=cpu_cycles_per_mac,
            preload_cycles_per_word=preload_cycles_per_word,
            linear_dtype=args.dtype,
            attention_dtype=args.attention_dtype,
            stream_weights=True,
        ),
        summarize_workload(
            f"TinyLlama decode ctx={args.decode_context} resident",
            decode_shapes,
            npu_models=npu_models,
            cpu_cycles_per_mac=cpu_cycles_per_mac,
            preload_cycles_per_word=preload_cycles_per_word,
            linear_dtype=args.dtype,
            attention_dtype=args.attention_dtype,
            stream_weights=False,
        ),
        summarize_workload(
            f"TinyLlama decode ctx={args.decode_context} streamed weights",
            decode_shapes,
            npu_models=npu_models,
            cpu_cycles_per_mac=cpu_cycles_per_mac,
            preload_cycles_per_word=preload_cycles_per_word,
            linear_dtype=args.dtype,
            attention_dtype=args.attention_dtype,
            stream_weights=True,
        ),
    ]
    print("\nTinyLlama analytical estimates, matmul-only plus optional weight streaming:")
    print(f"  note: projection/MLP dtype={args.dtype}; attention score/value dtype={args.attention_dtype}.")
    print("  note: scalar host ops and DRAM protocol are not calibrated in the shape-only section.")
    for row in summaries:
        print(
            f"  {row['label']}: macs={fmt(row['macs'])}, "
            f"cpu={fmt(row['cpu_cycles'])} cyc, "
            f"npu_body={fmt(row['npu_body_cycles'])} cyc, "
            f"weight_load={fmt(row['weight_preload_cycles'])} cyc, "
            f"hybrid={fmt(row['hybrid_cycles'])} cyc, "
            f"speedup={row['speedup']:.2f}x"
        )

    if args.compiled_plan != "none":
        modes = ["prefill", "decode"] if args.compiled_plan == "qllama-both" else [args.compiled_plan.replace("qllama-", "")]
        print("\nCompiled-plan section uses the actual QLlama ExecutionPlan after compiler rewrites.")
        print("  transfer/preload cycles are calibrated estimates; step counts and UB words are from the compiled artifact.")
        for mode in modes:
            artifact = _build_qllama_artifact(args, mode)
            summarize_compiled_plan(
                mode,
                artifact,
                npu_models=npu_models,
                cpu_cycles_per_mac=cpu_cycles_per_mac,
                preload_cycles_per_word=preload_cycles_per_word,
                attention_dtype=args.attention_dtype,
                show_step_costs=not args.no_step_costs,
                calibration=calibration,
            )


if __name__ == "__main__":
    main()
