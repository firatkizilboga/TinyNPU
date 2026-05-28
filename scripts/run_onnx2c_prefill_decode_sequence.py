from __future__ import annotations

import argparse
from datetime import datetime
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))

from build_onnx2c_baremetal_baselines import (  # noqa: E402
    _c_array,
    _deterministic_llama_input,
    _generate_c_from_onnx,
    _generate_onnx,
    _read_entry_signature,
)
from tinynpu_jit.rtl_runner import (  # noqa: E402
    CORE_DIR,
    CUSTOM_DIR,
    GENERATED_DIR,
    RUNTIME_DIR,
    TNPU_RISCV_MABI,
    TNPU_RISCV_MARCH,
    run_checked,
    toolchain_include_lib_dirs,
    toolchain_prefix,
)


def _linker_script_for_ram_bytes(ram_bytes: int | None) -> str:
    if ram_bytes is None or ram_bytes <= 0:
        return "custom/link.ld"
    if ram_bytes < 0x400000:
        raise ValueError("--sim-ram-bytes must be at least the default 0x400000")
    base = CORE_DIR / "custom" / "link.ld"
    generated = CORE_DIR / "custom" / f"link_ram_{ram_bytes:x}.ld"
    text = base.read_text()
    old = "LENGTH = 0x400000"
    if old not in text:
        raise RuntimeError(f"could not find default RAM length in {base}")
    generated.write_text(text.replace(old, f"LENGTH = 0x{ram_bytes:x}", 1))
    return f"custom/{generated.name}"


def _canonical_model(name: str) -> str:
    if name in {"qllama", "llama"}:
        return "qllama"
    if name in {"qgpt2", "gpt2"}:
        return "qgpt2"
    raise ValueError(f"unsupported model {name}")


def _model_onnx_names(model: str) -> tuple[str, str, str]:
    if model == "qllama":
        return "llama_prefill", "llama_decode", "QLlama"
    if model == "qgpt2":
        return "gpt2_prefill", "gpt2_decode", "QGPT2"
    raise ValueError(f"unsupported model {model}")


def _text_from_completed(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def _default_rtl_log_path(stem: str) -> Path:
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    return RUNS_DIR / f"{stem}_rtl_{timestamp}.log"


def _write_rtl_log(
    path: Path,
    *,
    command: list[str] | None,
    stdout: str | bytes | None,
    stderr: str | bytes | None,
    status: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"status={status}\n"
        f"command={' '.join(command or [])}\n"
        "\n--- stdout ---\n"
        f"{_text_from_completed(stdout)}"
        "\n--- stderr ---\n"
        f"{_text_from_completed(stderr)}"
    )


def _run_rtl_streamed(
    *,
    hex_path: Path,
    maxcycles: int,
    verilator_max_ticks: int,
    timeout_s: int | None,
    log_path: Path,
) -> tuple[int, str]:
    cmd = [
        str(CORE_DIR / "obj_dir" / "cv32e40p_tb_vlt_npu"),
        "+verilator+noassert",
        f"+firmware={hex_path}",
        f"+maxcycles={maxcycles}",
    ]
    env = dict(os.environ)
    env["VERILATOR_MAX_TICKS"] = str(verilator_max_ticks)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("status=running\n")
        log.write(f"command={' '.join(cmd)}\n")
        log.write("\n--- stdout/stderr ---\n")
        log.flush()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(CORE_DIR),
                env=env,
                check=False,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            log.write(f"\nstatus=timeout:{timeout_s}\n")
            return 124, "timeout"
        log.write(f"\nstatus={'success' if proc.returncode == 0 else f'failed:{proc.returncode}'}\n")
        return proc.returncode, "success" if proc.returncode == 0 else f"failed:{proc.returncode}"


def _render_sequence_wrapper(
    *,
    model_label: str,
    prefill_c_name: str,
    decode_c_names: list[str],
    prefill_func: str,
    decode_funcs: list[str],
    prefill_signature: str,
    decode_signatures: list[str],
    d_model: int,
    prompt_len: int,
) -> str:
    token = _deterministic_llama_input(d_model)[0]
    prefill_input = np.vstack([token + np.float32((i % 7) - 3) / np.float32(128.0) for i in range(prompt_len)])
    decode_inputs = [
        _deterministic_llama_input(d_model) + np.float32(i) / np.float32(256.0)
        for i in range(len(decode_funcs))
    ]
    decode_decls = "\n".join(
        f"void {func}({signature});"
        for func, signature in zip(decode_funcs, decode_signatures, strict=True)
    )
    decode_arrays = "\n".join(_c_array(f"decode{i}_input_flat", values) for i, values in enumerate(decode_inputs))
    decode_outputs = "\n".join(f"static float decode{i}_output[1][{int(d_model)}];" for i in range(len(decode_funcs)))
    decode_ptrs = "\n".join(
        f"    const float (*decode{i}_input)[{int(d_model)}] = (const float (*)[{int(d_model)}])decode{i}_input_flat;"
        for i in range(len(decode_funcs))
    )
    decode_calls: list[str] = []
    for i, func in enumerate(decode_funcs):
        decode_calls.append(
            f"""    t0 = read_mcycle32();
    {func}(decode{i}_input, decode{i}_output);
    t1 = read_mcycle32();
    uint32_t decode{i}_cycles = t0 - t1;"""
        )
    decode_checksum = "\n".join(
        f"""    for (int c = 0; c < {int(d_model)}; ++c) {{
        checksum += decode{i}_output[0][c];
    }}"""
        for i in range(len(decode_funcs))
    )
    decode_cycle_args = ", ".join(f"(unsigned long)decode{i}_cycles" for i in range(len(decode_funcs)))
    decode_cycle_fields = " ".join(f"decode{i}_cycles=%lu" for i in range(len(decode_funcs)))
    last_decode = len(decode_funcs) - 1
    return f"""// Bare-metal prefill+decode sequence baseline generated by third-party onnx2c.
// onnx2c: https://github.com/kraiskil/onnx2c
// Generated model sources: {prefill_c_name}, {", ".join(decode_c_names)}
#include <stdint.h>
#include <stdio.h>

uint32_t runtime_cycle_start;
uint32_t runtime_cycle_post_bss;
uint32_t runtime_cycle_post_init;
uint32_t runtime_cycle_pre_main;

#define TIMER_CTRL  ((volatile uint32_t *) 0x15000000u)
#define TIMER_VALUE ((volatile uint32_t *) 0x15000004u)
#define TIMER_COUNT ((volatile uint32_t *) 0x15001000u)

static inline uint32_t read_mcycle32(void) {{ return *TIMER_COUNT; }}

static void reset_timer(void)
{{
    *TIMER_CTRL = 0u;
    *TIMER_VALUE = 0xFFFFFFFFu;
    while (*TIMER_COUNT == 0u) {{ }}
}}

void {prefill_func}({prefill_signature});
{decode_decls}

{_c_array("prefill_input_flat", prefill_input)}
{decode_arrays}
static float prefill_output[{int(prompt_len)}][{int(d_model)}];
{decode_outputs}

int main(void)
{{
    const float (*prefill_input)[{int(d_model)}] = (const float (*)[{int(d_model)}])prefill_input_flat;
{decode_ptrs}
    reset_timer();
    uint32_t total_start = read_mcycle32();
    uint32_t t0 = read_mcycle32();
    {prefill_func}(prefill_input, prefill_output);
    uint32_t t1 = read_mcycle32();
    uint32_t prefill_cycles = t0 - t1;
{chr(10).join(decode_calls)}
    uint32_t total_cycles = total_start - t1;
    float checksum = 0.0f;
    for (int r = 0; r < {int(prompt_len)}; ++r) {{
        for (int c = 0; c < {int(d_model)}; ++c) {{
            checksum += prefill_output[r][c];
        }}
    }}
{decode_checksum}
    printf("third_party_onnx2c_{model_label.lower()}_prefill_decode{len(decode_funcs)} d_model={int(d_model)} prompt_len={int(prompt_len)} cycles=%lu prefill_cycles=%lu {decode_cycle_fields} checksum=%.9g first=%.9g last=%.9g\\n",
           (unsigned long)total_cycles, (unsigned long)prefill_cycles, {decode_cycle_args},
           (double)checksum, (double)prefill_output[0][0], (double)decode{last_decode}_output[0][{int(d_model) - 1}]);
    puts("EXIT SUCCESS");
    return 0;
}}
"""


def _build_elf_and_hex(
    *,
    program_name: str,
    wrapper_source: str,
    prefill_c: Path,
    decode_cs: list[Path],
    linker_script: str,
    sim_ram_addr_width: int | None,
) -> tuple[Path, Path, Path]:
    GENERATED_DIR.mkdir(exist_ok=True)
    CUSTOM_DIR.mkdir(exist_ok=True)
    wrapper_path = GENERATED_DIR / f"{program_name}.c"
    wrapper_path.write_text(wrapper_source)

    prefix = toolchain_prefix()
    gcc = f"{prefix}gcc"
    objcopy = f"{prefix}objcopy"
    include_dir, lib_dir = toolchain_include_lib_dirs(prefix)
    elf_path = CUSTOM_DIR / f"{program_name}.elf"
    hex_path = CUSTOM_DIR / f"{program_name}.hex"
    build_env = dict(os.environ)
    build_env["CCACHE_DISABLE"] = "1"
    build_env["TMPDIR"] = "/tmp"
    if sim_ram_addr_width is not None:
        extra_flags = build_env.get("VERILATOR_EXTRA_FLAGS", "--x-assign fast --x-initial fast --inline-mult 0")
        build_env["VERILATOR_EXTRA_FLAGS"] = f"{extra_flags} -GRAM_ADDR_WIDTH={sim_ram_addr_width}"
    run_checked(["make", "verilator-build-npu"], cwd=CORE_DIR, env=build_env)
    run_checked(
        [
            gcc,
            f"-march={TNPU_RISCV_MARCH}",
            f"-mabi={TNPU_RISCV_MABI}",
            "-o",
            str(elf_path),
            "-w",
            "-O3",
            "-g",
            "-DNDEBUG",
            "-nostdlib",
            "-T",
            linker_script,
            "-static",
            "custom/crt0.S",
            str(wrapper_path),
            str(prefill_c),
            *(str(path) for path in decode_cs),
            "mem_stall/mem_stall.c",
            "custom/syscalls.c",
            "custom/vectors.S",
            "-I",
            str(include_dir),
            "-I",
            "mem_stall",
            "-I",
            str(RUNTIME_DIR),
            "-L",
            str(lib_dir),
            "-lc",
            "-lm",
            "-lgcc",
        ],
        cwd=CORE_DIR,
        env=build_env,
    )
    run_checked([objcopy, "-O", "verilog", str(elf_path), str(hex_path)], cwd=CORE_DIR, env=build_env)
    return wrapper_path, elf_path, hex_path


def build_sequence(args: argparse.Namespace) -> tuple[Path, list[Path], Path, list[Path], Path, Path, Path]:
    prefill_model, decode_model, model_label = _model_onnx_names(args.model)
    stem = (
        f"third_party_onnx2c_{args.model}_prefill_decode{args.decode_tokens}_seq_d{args.d_model}_h{args.d_head}_"
        f"nh{args.n_heads}_nkv{args.n_kv_heads}_f{args.ffn_dim}_t{args.prompt_len}_s{args.seed}"
    )
    prefill_onnx = GENERATED_DIR / f"{stem}_prefill.onnx"
    decode_onnx = [GENERATED_DIR / f"{stem}_decode{i}.onnx" for i in range(args.decode_tokens)]
    prefill_c = GENERATED_DIR / f"{stem}_prefill_model.c"
    decode_c = [GENERATED_DIR / f"{stem}_decode{i}_model.c" for i in range(args.decode_tokens)]
    prefill_func = f"onnx2c_{args.model}_prefill_seq"
    decode_func = [f"onnx2c_{args.model}_decode{i}_seq" for i in range(args.decode_tokens)]

    _generate_onnx(
        prefill_model,
        prefill_onnx,
        prompt_len=args.prompt_len,
        mlp_hidden=64,
        conv_channels=16,
        d_model=args.d_model,
        d_head=args.d_head,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        ffn_hidden_dim=args.ffn_dim,
        seed=args.seed,
    )
    for i, path in enumerate(decode_onnx):
        _generate_onnx(
            decode_model,
            path,
            prompt_len=args.prompt_len + i,
            mlp_hidden=64,
            conv_channels=16,
            d_model=args.d_model,
            d_head=args.d_head,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            ffn_hidden_dim=args.ffn_dim,
            seed=args.seed,
        )
    _generate_c_from_onnx(prefill_onnx, prefill_c, func_name=prefill_func)
    for path, c_path, func in zip(decode_onnx, decode_c, decode_func, strict=True):
        _generate_c_from_onnx(path, c_path, func_name=func)
    prefill_signature = _read_entry_signature(prefill_c, prefill_func)
    decode_signature = [
        _read_entry_signature(c_path, func)
        for c_path, func in zip(decode_c, decode_func, strict=True)
    ]
    wrapper = _render_sequence_wrapper(
        model_label=model_label,
        prefill_c_name=prefill_c.name,
        decode_c_names=[path.name for path in decode_c],
        prefill_func=prefill_func,
        decode_funcs=decode_func,
        prefill_signature=prefill_signature,
        decode_signatures=decode_signature,
        d_model=args.d_model,
        prompt_len=args.prompt_len,
    )
    wrapper_path, elf_path, hex_path = _build_elf_and_hex(
        program_name=stem,
        wrapper_source=wrapper,
        prefill_c=prefill_c,
        decode_cs=decode_c,
        linker_script=_linker_script_for_ram_bytes(args.sim_ram_bytes),
        sim_ram_addr_width=args.sim_ram_addr_width,
    )
    return prefill_onnx, decode_onnx, prefill_c, decode_c, wrapper_path, elf_path, hex_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/run one ONNX2C CPU-only image that executes prefill then decode.")
    parser.add_argument("--model", choices=("qllama", "qgpt2", "llama", "gpt2"), required=True)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-head", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    parser.add_argument("--ffn-dim", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, choices=(1, 2), default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-rtl", action="store_true")
    parser.add_argument("--maxcycles", type=int, default=200_000_000)
    parser.add_argument("--verilator-max-ticks", type=int, default=1_000_000_000_000)
    parser.add_argument("--timeout-s", type=int, default=900, help="Python wall-time timeout. Use 0 or a negative value to disable it.")
    parser.add_argument("--dump-stdout", action="store_true")
    parser.add_argument("--log-path", type=Path, default=None)
    parser.add_argument("--sim-ram-bytes", type=lambda value: int(value, 0), default=0, help="Optional larger bare-metal RAM size for generated linker script, e.g. 0x1000000.")
    parser.add_argument("--sim-ram-addr-width", type=int, default=None, help="Optional Verilator tb RAM_ADDR_WIDTH override. Use with --sim-ram-bytes when the image exceeds 4 MiB.")
    args = parser.parse_args()
    args.model = _canonical_model(args.model)
    if args.n_kv_heads is None:
        args.n_kv_heads = args.n_heads if args.model == "qgpt2" else max(1, args.n_heads // 2)
    if args.model == "qgpt2" and args.n_kv_heads != args.n_heads:
        raise SystemExit("GPT-2 uses full multi-head KV; pass --n-kv-heads equal to --n-heads or omit it.")
    if args.d_model != args.n_heads * args.d_head:
        raise SystemExit("ONNX2C transformer baselines expect --d-model == --n-heads * --d-head.")
    timeout_s = None if args.timeout_s <= 0 else args.timeout_s

    prefill_onnx, decode_onnx, prefill_c, decode_c, wrapper_path, elf_path, hex_path = build_sequence(args)
    print(f"prefill_onnx={prefill_onnx}")
    for idx, path in enumerate(decode_onnx):
        print(f"decode{idx}_onnx={path}")
    print(f"prefill_generated_c={prefill_c}")
    for idx, path in enumerate(decode_c):
        print(f"decode{idx}_generated_c={path}")
    print(f"wrapper={wrapper_path}")
    print(f"elf={elf_path}")
    print(f"hex={hex_path}")

    if args.run_rtl:
        stem = Path(hex_path).stem
        log_path = args.log_path or _default_rtl_log_path(stem)
        rc, status = _run_rtl_streamed(
            hex_path=hex_path,
            maxcycles=args.maxcycles,
            verilator_max_ticks=args.verilator_max_ticks,
            timeout_s=timeout_s,
            log_path=log_path,
        )
        print(f"rtl_log={log_path}")
        log_text = log_path.read_text(errors="replace")
        if args.dump_stdout:
            print(log_text)
        else:
            for line in log_text.splitlines():
                if "third_party_onnx2c_" in line or line.startswith("EXIT "):
                    print(line)
        if status.startswith("timeout"):
            print(f"RTL run timed out after {args.timeout_s}s", file=sys.stderr)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
