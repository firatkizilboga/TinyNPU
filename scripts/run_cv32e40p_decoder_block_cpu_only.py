from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))

from tinynpu_jit import (  # noqa: E402
    DType,
    ExecutionPlan,
    HostOp,
    MatMulOp,
    NpuSegment,
    TensorKind,
    TensorSpec,
    compile_plan,
    write_cv32e40p_c,
)
from tinynpu_jit.golden import GoldenModel  # noqa: E402

from run_cv32e40p_b_append_demo import (  # noqa: E402
    CORE_DIR,
    CUSTOM_DIR,
    GENERATED_DIR,
    TNPU_RISCV_MABI,
    TNPU_RISCV_MARCH,
    _run,
    _sanitize,
    _toolchain_prefix,
    _toolchain_root,
)


def _rand_i16(rng: np.random.Generator, shape: tuple[int, ...], low: int = -2, high: int = 3) -> np.ndarray:
    return rng.integers(low, high, size=shape, endpoint=False, dtype=np.int16)


def build_plan(
    *,
    d_model: int = 16,
    d_head: int = 16,
    ffn_dim: int = 32,
    token_count: int = 16,
    seed: int = 0,
    act_scale: float = 1.0 / 32.0,
    attn_scale: float = 1.0 / 256.0,
    rope_position: int = 7,
    rope_theta: float = 10000.0,
) -> ExecutionPlan:
    if d_model <= 0 or d_model % 8 != 0:
        raise ValueError("d_model must be a positive multiple of 8")
    if d_head <= 0 or d_head % 8 != 0:
        raise ValueError("d_head must be a positive multiple of 8")
    if ffn_dim <= 0 or ffn_dim % 8 != 0:
        raise ValueError("ffn_dim must be a positive multiple of 8")
    if token_count <= 0 or token_count % 8 != 0:
        raise ValueError("token_count must be a positive multiple of 8")

    rng = np.random.default_rng(seed)

    x = _rand_i16(rng, (1, d_model))
    rms1_w = rng.uniform(0.5, 1.5, size=(d_model,)).astype(np.float32)
    rms2_w = rng.uniform(0.5, 1.5, size=(d_model,)).astype(np.float32)

    w_q = _rand_i16(rng, (d_model, d_head))
    w_o = _rand_i16(rng, (d_head, d_model))
    w_gate = _rand_i16(rng, (d_model, ffn_dim))
    w_up = _rand_i16(rng, (d_model, ffn_dim))
    w_down = _rand_i16(rng, (ffn_dim, d_model))

    k_cache = _rand_i16(rng, (d_head, token_count))
    v_cache = _rand_i16(rng, (token_count, d_head))

    tensors = {
        "x": TensorSpec("x", x.shape, DType.INT16, TensorKind.CONSTANT, data=x),
        "rms1_w": TensorSpec("rms1_w", rms1_w.shape, DType.FLOAT32, TensorKind.CONSTANT, data=rms1_w),
        "rms2_w": TensorSpec("rms2_w", rms2_w.shape, DType.FLOAT32, TensorKind.CONSTANT, data=rms2_w),
        "w_q": TensorSpec("w_q", w_q.shape, DType.INT16, TensorKind.CONSTANT, data=w_q),
        "w_o": TensorSpec("w_o", w_o.shape, DType.INT16, TensorKind.CONSTANT, data=w_o),
        "w_gate": TensorSpec("w_gate", w_gate.shape, DType.INT16, TensorKind.CONSTANT, data=w_gate),
        "w_up": TensorSpec("w_up", w_up.shape, DType.INT16, TensorKind.CONSTANT, data=w_up),
        "w_down": TensorSpec("w_down", w_down.shape, DType.INT16, TensorKind.CONSTANT, data=w_down),
        "k_cache": TensorSpec("k_cache", k_cache.shape, DType.INT16, TensorKind.CONSTANT, data=k_cache),
        "v_cache": TensorSpec("v_cache", v_cache.shape, DType.INT16, TensorKind.CONSTANT, data=v_cache),
        "x_norm1": TensorSpec("x_norm1", (1, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "x_norm1_q": TensorSpec("x_norm1_q", (1, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        "q_int": TensorSpec("q_int", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE),
        "q_f": TensorSpec("q_f", (1, d_head), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "q_rope": TensorSpec("q_rope", (1, d_head), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "q_rope_q": TensorSpec("q_rope_q", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE),
        "scores": TensorSpec("scores", (1, token_count), DType.INT16, TensorKind.INTERMEDIATE),
        "probs": TensorSpec("probs", (1, token_count), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "probs_q": TensorSpec("probs_q", (1, token_count), DType.INT16, TensorKind.INTERMEDIATE),
        "attn_int": TensorSpec("attn_int", (1, d_head), DType.INT16, TensorKind.INTERMEDIATE),
        "o_int": TensorSpec("o_int", (1, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        "o_f": TensorSpec("o_f", (1, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "resid1": TensorSpec("resid1", (1, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "x_norm2": TensorSpec("x_norm2", (1, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "x_norm2_q": TensorSpec("x_norm2_q", (1, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        "gate_int": TensorSpec("gate_int", (1, ffn_dim), DType.INT16, TensorKind.INTERMEDIATE),
        "up_int": TensorSpec("up_int", (1, ffn_dim), DType.INT16, TensorKind.INTERMEDIATE),
        "gate_f": TensorSpec("gate_f", (1, ffn_dim), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "up_f": TensorSpec("up_f", (1, ffn_dim), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "gate_act": TensorSpec("gate_act", (1, ffn_dim), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "ffn_hidden": TensorSpec("ffn_hidden", (1, ffn_dim), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "ffn_hidden_q": TensorSpec("ffn_hidden_q", (1, ffn_dim), DType.INT16, TensorKind.INTERMEDIATE),
        "ffn_out_int": TensorSpec("ffn_out_int", (1, d_model), DType.INT16, TensorKind.INTERMEDIATE),
        "ffn_out_f": TensorSpec("ffn_out_f", (1, d_model), DType.FLOAT32, TensorKind.INTERMEDIATE),
        "out": TensorSpec("out", (1, d_model), DType.FLOAT32, TensorKind.OUTPUT, is_final_output=True),
    }

    steps = [
        HostOp("rmsnorm1", "rmsnorm", inputs=["x", "rms1_w"], outputs=["x_norm1"], attrs={"eps": 1.0e-5}),
        HostOp(
            "quant_x_norm1",
            "quantize",
            inputs=["x_norm1"],
            outputs=["x_norm1_q"],
            attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16},
        ),
        NpuSegment(
            "seg_q_proj",
            [MatMulOp("op_q_proj", "x_norm1_q", "w_q", "q_int", in_dtype=DType.INT16, out_dtype=DType.INT16)],
            inputs=["x_norm1_q"],
            outputs=["q_int"],
        ),
        HostOp("dequant_q", "dequantize", inputs=["q_int"], outputs=["q_f"], attrs={"scale": act_scale, "zero_point": 0}),
        HostOp(
            "rope_q",
            "rope",
            inputs=["q_f"],
            outputs=["q_rope"],
            attrs={"head_dim": d_head, "position": rope_position, "theta": rope_theta},
        ),
        HostOp(
            "quant_q_rope",
            "quantize",
            inputs=["q_rope"],
            outputs=["q_rope_q"],
            attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16},
        ),
        NpuSegment(
            "seg_scores",
            [MatMulOp("op_qk", "q_rope_q", "k_cache", "scores", in_dtype=DType.INT16, out_dtype=DType.INT16)],
            inputs=["q_rope_q"],
            outputs=["scores"],
        ),
        HostOp("softmax_scores", "softmax", inputs=["scores"], outputs=["probs"], attrs={"axis": -1}),
        HostOp(
            "quant_probs",
            "quantize",
            inputs=["probs"],
            outputs=["probs_q"],
            attrs={"scale": attn_scale, "zero_point": 0, "dtype": DType.INT16},
        ),
        NpuSegment(
            "seg_value",
            [
                MatMulOp("op_av", "probs_q", "v_cache", "attn_int", shift=8, in_dtype=DType.INT16, out_dtype=DType.INT16),
                MatMulOp("op_o_proj", "attn_int", "w_o", "o_int", in_dtype=DType.INT16, out_dtype=DType.INT16),
            ],
            inputs=["probs_q"],
            outputs=["o_int"],
        ),
        HostOp("dequant_o", "dequantize", inputs=["o_int"], outputs=["o_f"], attrs={"scale": 1.0, "zero_point": 0}),
        HostOp("residual1", "add", inputs=["x", "o_f"], outputs=["resid1"]),
        HostOp("rmsnorm2", "rmsnorm", inputs=["resid1", "rms2_w"], outputs=["x_norm2"], attrs={"eps": 1.0e-5}),
        HostOp(
            "quant_x_norm2",
            "quantize",
            inputs=["x_norm2"],
            outputs=["x_norm2_q"],
            attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16},
        ),
        NpuSegment(
            "seg_ffn_up",
            [
                MatMulOp("op_gate_proj", "x_norm2_q", "w_gate", "gate_int", in_dtype=DType.INT16, out_dtype=DType.INT16),
                MatMulOp("op_up_proj", "x_norm2_q", "w_up", "up_int", in_dtype=DType.INT16, out_dtype=DType.INT16),
            ],
            inputs=["x_norm2_q"],
            outputs=["gate_int", "up_int"],
        ),
        HostOp("dequant_gate", "dequantize", inputs=["gate_int"], outputs=["gate_f"], attrs={"scale": act_scale, "zero_point": 0}),
        HostOp("dequant_up", "dequantize", inputs=["up_int"], outputs=["up_f"], attrs={"scale": act_scale, "zero_point": 0}),
        HostOp("silu_gate", "silu", inputs=["gate_f"], outputs=["gate_act"]),
        HostOp("ffn_mul", "mul", inputs=["gate_act", "up_f"], outputs=["ffn_hidden"]),
        HostOp(
            "quant_ffn_hidden",
            "quantize",
            inputs=["ffn_hidden"],
            outputs=["ffn_hidden_q"],
            attrs={"scale": act_scale, "zero_point": 0, "dtype": DType.INT16},
        ),
        NpuSegment(
            "seg_ffn_down",
            [MatMulOp("op_down_proj", "ffn_hidden_q", "w_down", "ffn_out_int", in_dtype=DType.INT16, out_dtype=DType.INT16)],
            inputs=["ffn_hidden_q"],
            outputs=["ffn_out_int"],
        ),
        HostOp("dequant_ffn_out", "dequantize", inputs=["ffn_out_int"], outputs=["ffn_out_f"], attrs={"scale": act_scale, "zero_point": 0}),
        HostOp("residual2", "add", inputs=["resid1", "ffn_out_f"], outputs=["out"]),
    ]

    plan = ExecutionPlan(tensors=tensors, steps=steps, inputs=[], outputs=["out"])
    plan.add_verification_step("out", "decoder_block_out")
    return plan


def build_artifact(**kwargs):
    plan = build_plan(**kwargs)
    golden = GoldenModel()
    expected = golden.softmax(np.array([[0.0]], dtype=np.float32), axis=-1)  # placeholder to keep golden in scope
    result = compile_plan(plan, {}).run_host_emulation({}, verification="off") if False else None
    # Direct host-side expected computation keeps this script independent of the native-cache emulation gap.
    tensors = plan.tensors
    act_scale = float(kwargs.get("act_scale", 1.0 / 32.0))
    attn_scale = float(kwargs.get("attn_scale", 1.0 / 256.0))
    rope_position = int(kwargs.get("rope_position", 7))
    rope_theta = float(kwargs.get("rope_theta", 10000.0))
    d_head = int(kwargs["d_head"])

    x = np.array(tensors["x"].data, dtype=np.int16, copy=True)
    rms1_w = np.array(tensors["rms1_w"].data, dtype=np.float32, copy=True)
    rms2_w = np.array(tensors["rms2_w"].data, dtype=np.float32, copy=True)
    w_q = np.array(tensors["w_q"].data, dtype=np.int16, copy=True)
    w_o = np.array(tensors["w_o"].data, dtype=np.int16, copy=True)
    w_gate = np.array(tensors["w_gate"].data, dtype=np.int16, copy=True)
    w_up = np.array(tensors["w_up"].data, dtype=np.int16, copy=True)
    w_down = np.array(tensors["w_down"].data, dtype=np.int16, copy=True)
    k_cache = np.array(tensors["k_cache"].data, dtype=np.int16, copy=True)
    v_cache = np.array(tensors["v_cache"].data, dtype=np.int16, copy=True)

    x_f = x.astype(np.float32)
    eps = np.float32(1.0e-5)
    rms1 = np.sqrt(np.mean(np.square(x_f, dtype=np.float32), axis=-1, keepdims=True, dtype=np.float32) + eps).astype(np.float32)
    x_norm1 = ((x_f / rms1) * rms1_w.reshape(1, -1)).astype(np.float32)
    x_norm1_q = golden.quantize(x_norm1, scale=act_scale, zero_point=0, out_dtype=DType.INT16)
    q_int = golden.matmul(x_norm1_q, w_q, out_dtype=DType.INT16)
    q_f = golden.dequantize(q_int, scale=act_scale, zero_point=0)

    half = d_head // 2
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, half, dtype=np.float32) / np.float32(half)))
    angles = np.full((q_f.shape[0], 1), np.float32(rope_position), dtype=np.float32) * inv_freq.reshape(1, -1)
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    q_rope = np.array(q_f, dtype=np.float32, copy=True)
    first = q_f[..., :half]
    second = q_f[..., half:d_head]
    q_rope[..., :half] = first * cos - second * sin
    q_rope[..., half:d_head] = second * cos + first * sin
    q_rope_q = golden.quantize(q_rope, scale=act_scale, zero_point=0, out_dtype=DType.INT16)

    scores = golden.matmul(q_rope_q, k_cache, out_dtype=DType.INT16)
    probs = golden.softmax(scores, axis=-1).astype(np.float32)
    probs_q = golden.quantize(probs, scale=attn_scale, zero_point=0, out_dtype=DType.INT16)
    attn_int = golden.matmul(probs_q, v_cache, shift=8, out_dtype=DType.INT16)
    o_int = golden.matmul(attn_int, w_o, out_dtype=DType.INT16)
    o_f = golden.dequantize(o_int, scale=1.0, zero_point=0)
    resid1 = (x_f + o_f).astype(np.float32)

    rms2 = np.sqrt(np.mean(np.square(resid1, dtype=np.float32), axis=-1, keepdims=True, dtype=np.float32) + eps).astype(np.float32)
    x_norm2 = ((resid1 / rms2) * rms2_w.reshape(1, -1)).astype(np.float32)
    x_norm2_q = golden.quantize(x_norm2, scale=act_scale, zero_point=0, out_dtype=DType.INT16)
    gate_int = golden.matmul(x_norm2_q, w_gate, out_dtype=DType.INT16)
    up_int = golden.matmul(x_norm2_q, w_up, out_dtype=DType.INT16)
    gate_f = golden.dequantize(gate_int, scale=act_scale, zero_point=0)
    up_f = golden.dequantize(up_int, scale=act_scale, zero_point=0)
    gate_act = (gate_f / (np.float32(1.0) + np.exp(-gate_f))).astype(np.float32)
    ffn_hidden = (gate_act * up_f).astype(np.float32)
    ffn_hidden_q = golden.quantize(ffn_hidden, scale=act_scale, zero_point=0, out_dtype=DType.INT16)
    ffn_out_int = golden.matmul(ffn_hidden_q, w_down, out_dtype=DType.INT16)
    ffn_out_f = golden.dequantize(ffn_out_int, scale=act_scale, zero_point=0)
    out = (resid1 + ffn_out_f).astype(np.float32)

    artifact = compile_plan(plan, {"out": out})
    return artifact, out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--d-head", type=int, default=16)
    parser.add_argument("--ffn-dim", type=int, default=32)
    parser.add_argument("--token-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    artifact, expected = build_artifact(
        d_model=args.d_model,
        d_head=args.d_head,
        ffn_dim=args.ffn_dim,
        token_count=args.token_count,
        seed=args.seed,
    )

    program_name = (
        f"cv32e40p_decoder_block_cpu_only_d{args.d_model}_h{args.d_head}"
        f"_f{args.ffn_dim}_t{args.token_count}_s{args.seed}"
    )
    program_path = GENERATED_DIR / f"{program_name}.c"
    GENERATED_DIR.mkdir(exist_ok=True)
    write_cv32e40p_c(
        artifact,
        {},
        program_path,
        program_name=program_name,
        cpu_only_baseline=True,
    )

    prefix = _toolchain_prefix()
    gcc = f"{prefix}gcc"
    objcopy = f"{prefix}objcopy"
    toolchain_root = _toolchain_root(prefix)
    include_dir = toolchain_root / "riscv32-unknown-elf" / "include"
    lib_dir = toolchain_root / "riscv32-unknown-elf" / "lib"
    elf_path = CUSTOM_DIR / f"{program_name}.elf"
    hex_path = CUSTOM_DIR / f"{program_name}.hex"

    build_env = dict(os.environ)
    build_env["CCACHE_DISABLE"] = "1"
    build_env["TMPDIR"] = "/tmp"
    _run(["make", "verilator-build-npu"], cwd=CORE_DIR, env=build_env)
    _run(
        [
            gcc,
            f"-march={TNPU_RISCV_MARCH}",
            f"-mabi={TNPU_RISCV_MABI}",
            "-o",
            str(elf_path),
            "-w",
            "-O3",
            "-g",
            "-nostdlib",
            "-T",
            "custom/link.ld",
            "-static",
            "custom/crt0.S",
            str(program_path),
            "mem_stall/mem_stall.c",
            "custom/syscalls.c",
            "custom/vectors.S",
            "-I",
            str(include_dir),
            "-I",
            "mem_stall",
            "-L",
            str(lib_dir),
            "-lc",
            "-lm",
            "-lgcc",
        ],
        cwd=CORE_DIR,
        env=build_env,
    )
    _run([objcopy, "-O", "verilog", str(elf_path), str(hex_path)], cwd=CORE_DIR, env=build_env)

    env = dict(os.environ)
    env["VERILATOR_MAX_TICKS"] = "3000000000"
    proc = _run(
        [
            str(CORE_DIR / "obj_dir" / "cv32e40p_tb_vlt_npu"),
            "+verilator+noassert",
            f"+firmware={hex_path}",
            "+maxcycles=2000000",
        ],
        cwd=CORE_DIR,
        env=env,
        capture=True,
    )
    print(f"program={program_name}")
    print(
        f"d_model={args.d_model} d_head={args.d_head} "
        f"ffn_dim={args.ffn_dim} token_count={args.token_count} seed={args.seed}"
    )
    print(f"expected_checksum={float(np.array(expected, dtype=np.float32).sum()):.6f}")
    print(proc.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
