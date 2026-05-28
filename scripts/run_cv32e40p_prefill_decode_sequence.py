from __future__ import annotations

import argparse
from datetime import datetime
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"
if str(REPO_ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "software" / "compiler"))

from tinynpu_jit import RunnerConfig, emit_cv32e40p_program_v2  # noqa: E402
from tinynpu_jit.blocks.gpt2_block import (  # noqa: E402
    build_decode_artifact as build_gpt2_decode_artifact,
    build_prefill_artifact as build_gpt2_prefill_artifact,
    extend_kv_cache as extend_gpt2_kv_cache,
)
from tinynpu_jit.blocks.llama_block import (  # noqa: E402
    build_decode_artifact as build_llama_decode_artifact,
    build_prefill_artifact as build_llama_prefill_artifact,
    extend_kv_cache as extend_llama_kv_cache,
)
from tinynpu_jit.rtl_runner import (  # noqa: E402
    CORE_DIR,
    CUSTOM_DIR,
    GENERATED_DIR,
    RUNTIME_DIR,
    TNPU_RISCV_MABI,
    TNPU_RISCV_MARCH,
    run_checked,
    runtime_cflags,
    sanitize_program_symbol,
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


def _combined_runner_source(
    *,
    prefill_symbol: str,
    decode_symbols: list[str],
    model: str,
    n_cache_heads: int,
) -> str:
    extern_decodes = "\n".join(f"extern const TnpuProgram {symbol};" for symbol in decode_symbols)

    prefill_to_decode = []
    for head in range(n_cache_heads):
        prefill_to_decode.append(
            f"""    if (copy_k_cache(&{prefill_symbol}, "prefill_k_cache_h{head}", &{decode_symbols[0]}, "k_cache_h{head}") != 0) return EXIT_FAILURE;
    if (copy_v_cache(&{prefill_symbol}, "prefill_v_cache_h{head}", &{decode_symbols[0]}, "v_cache_h{head}") != 0) return EXIT_FAILURE;"""
        )

    decode_run_blocks: list[str] = []
    for idx, symbol in enumerate(decode_symbols):
        if idx > 0:
            prior = decode_symbols[idx - 1]
            handoff = []
            for head in range(n_cache_heads):
                handoff.append(
                    f"""    if (copy_k_cache(&{prior}, "k_cache_h{head}_td", &{symbol}, "k_cache_h{head}") != 0) return EXIT_FAILURE;
    if (copy_v_cache(&{prior}, "v_cache_h{head}_td", &{symbol}, "v_cache_h{head}") != 0) return EXIT_FAILURE;"""
                )
            decode_run_blocks.append(
                f"""    seq_print_marker("sequence.decode{idx - 1}_to_decode{idx}_handoff.start");
    t0 = seq_read_mcycle32();
{chr(10).join(handoff)}
    t1 = seq_read_mcycle32();
    seq_print_delta("sequence.decode{idx - 1}_to_decode{idx}_handoff.total", t0, t1);
"""
            )
        decode_run_blocks.append(
            f"""    seq_print_marker("sequence.decode{idx}.start");
    if (prepare_io(&{symbol}, ins, ip, outs, op) != 0) return EXIT_FAILURE;
    t0 = seq_read_mcycle32();
    rc = tinynpu_run(&{symbol}, ip, op, NULL, 0u);
    t1 = seq_read_mcycle32();
    seq_print_delta("sequence.decode{idx}.total", t0, t1);
    if (rc != 0) return EXIT_FAILURE;
"""
        )

    return f"""#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "tinynpu_runtime_v2.h"

extern const TnpuProgram {prefill_symbol};
{extern_decodes}

static inline uint32_t seq_read_mcycle32(void)
{{
    return *((volatile uint32_t *)0x15001000u);
}}

static void seq_print_delta(const char *label, uint32_t start, uint32_t end)
{{
    printf("%s cycles=%lu\\n", label, (unsigned long)(start - end));
}}

static void seq_print_marker(const char *label)
{{
    puts(label);
}}

static int streq(const char *a, const char *b)
{{
    while (*a != 0 && *b != 0) {{
        if (*a != *b) return 0;
        ++a;
        ++b;
    }}
    return *a == *b;
}}

static const TnpuTensorDesc *find_desc(const TnpuProgram *program, const char *name)
{{
    for (uint32_t i = 0; i < program->tensor_count; ++i) {{
        if (streq(program->tensors[i].name, name)) {{
            return &program->tensors[i];
        }}
    }}
    printf("missing tensor: %s in %s\\n", name, program->name);
    return NULL;
}}

static int prepare_io(const TnpuProgram *program, TnpuTensor *ins, const TnpuTensor **ip, TnpuTensor *outs, const TnpuTensor **op)
{{
    if (program->input_count > 64u) return 1;
    if (program->output_count > 64u) return 1;
    for (uint32_t i = 0; i < program->input_count; ++i) {{
        uint16_t t = program->input_tensor_indices[i];
        ins[i].data = program->tensors[t].data;
        ins[i].desc = &program->tensors[t];
        ins[i].elem_count = program->tensors[t].elem_count;
        ip[i] = &ins[i];
    }}
    for (uint32_t i = 0; i < program->output_count; ++i) {{
        uint16_t t = program->output_tensor_indices[i];
        outs[i].data = program->tensors[t].data;
        outs[i].desc = &program->tensors[t];
        outs[i].elem_count = program->tensors[t].elem_count;
        op[i] = &outs[i];
    }}
    return 0;
}}

static int copy_k_cache(const TnpuProgram *src_program, const char *src_name, const TnpuProgram *dst_program, const char *dst_name)
{{
    const TnpuTensorDesc *src = find_desc(src_program, src_name);
    const TnpuTensorDesc *dst = find_desc(dst_program, dst_name);
    if (src == NULL || dst == NULL) return 1;
    if (src->dtype != TNPU_DTYPE_INT16 || dst->dtype != TNPU_DTYPE_INT16) return 1;
    const int d_head = src->shape[0];
    const int prompt_len = src->shape[1];
    if (dst->shape[0] != d_head || dst->shape[1] < prompt_len) return 1;
    int16_t *src_data = (int16_t *)src->data;
    int16_t *dst_data = (int16_t *)dst->data;
    for (uint32_t i = 0; i < dst->elem_count; ++i) {{
        dst_data[i] = 0;
    }}
    for (int row = 0; row < d_head; ++row) {{
        for (int token = 0; token < prompt_len; ++token) {{
            dst_data[row * dst->shape[1] + token] = src_data[row * src->shape[1] + token];
        }}
    }}
    return 0;
}}

static int copy_v_cache(const TnpuProgram *src_program, const char *src_name, const TnpuProgram *dst_program, const char *dst_name)
{{
    const TnpuTensorDesc *src = find_desc(src_program, src_name);
    const TnpuTensorDesc *dst = find_desc(dst_program, dst_name);
    if (src == NULL || dst == NULL) return 1;
    if (src->dtype != TNPU_DTYPE_INT16 || dst->dtype != TNPU_DTYPE_INT16) return 1;
    const int prompt_len = src->shape[0];
    const int d_head = src->shape[1];
    if (dst->shape[0] < prompt_len || dst->shape[1] != d_head) return 1;
    int16_t *src_data = (int16_t *)src->data;
    int16_t *dst_data = (int16_t *)dst->data;
    for (uint32_t i = 0; i < dst->elem_count; ++i) {{
        dst_data[i] = 0;
    }}
    for (int token = 0; token < prompt_len; ++token) {{
        for (int col = 0; col < d_head; ++col) {{
            dst_data[token * d_head + col] = src_data[token * d_head + col];
        }}
    }}
    return 0;
}}

int main(void)
{{
    setbuf(stdout, NULL);
    TnpuTensor ins[64];
    const TnpuTensor *ip[64];
    TnpuTensor outs[64];
    const TnpuTensor *op[64];
    uint32_t total_start = seq_read_mcycle32();

    puts("prefill_decode_sequence: {model}");
    seq_print_marker("sequence.prefill.start");
    if (prepare_io(&{prefill_symbol}, ins, ip, outs, op) != 0) return EXIT_FAILURE;
    uint32_t t0 = seq_read_mcycle32();
    int rc = tinynpu_run(&{prefill_symbol}, ip, op, NULL, 0u);
    uint32_t t1 = seq_read_mcycle32();
    seq_print_delta("sequence.prefill.total", t0, t1);
    if (rc != 0) return EXIT_FAILURE;

    seq_print_marker("sequence.cache_handoff.start");
    t0 = seq_read_mcycle32();
{chr(10).join(prefill_to_decode)}
    t1 = seq_read_mcycle32();
    seq_print_delta("sequence.cache_handoff.total", t0, t1);

{chr(10).join(decode_run_blocks)}

    uint32_t total_end = seq_read_mcycle32();
    seq_print_delta("sequence.e2e.total", total_start, total_end);
    return EXIT_SUCCESS;
}}
"""


def _build_sequence_elf_and_hex(
    *,
    program_name: str,
    prefill_source: str,
    decode_sources: list[str],
    runner_source: str,
    linker_script: str,
    sim_ram_addr_width: int | None,
) -> tuple[Path, list[Path], Path, Path, Path]:
    GENERATED_DIR.mkdir(exist_ok=True)
    prefill_path = GENERATED_DIR / f"{program_name}_prefill_program.c"
    decode_paths = [GENERATED_DIR / f"{program_name}_decode{i}_program.c" for i in range(len(decode_sources))]
    runner_path = GENERATED_DIR / f"{program_name}_runner.c"
    prefill_path.write_text(prefill_source)
    for path, source in zip(decode_paths, decode_sources, strict=True):
        path.write_text(source)
    runner_path.write_text(runner_source)

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
    cfg = RunnerConfig(repeat_count=1, dump_final_outputs=True, verbose_steps=True)
    run_checked(
        [
            gcc,
            f"-march={TNPU_RISCV_MARCH}",
            f"-mabi={TNPU_RISCV_MABI}",
            "-o",
            str(elf_path),
            *runtime_cflags(cfg, extra_cflags=["-ffast-math", "-fno-builtin-printf"]),
            "-T",
            linker_script,
            "-static",
            "custom/crt0.S",
            str(runner_path),
            str(prefill_path),
            *(str(path) for path in decode_paths),
            str(RUNTIME_DIR / "tinynpu_runtime_v2.c"),
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
    return prefill_path, decode_paths, runner_path, elf_path, hex_path


def _build_artifacts(args: argparse.Namespace):
    if args.model == "qllama":
        prefill, state, _ = build_llama_prefill_artifact(
            d_model=args.d_model,
            d_head=args.d_head,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            ffn_hidden_dim=args.ffn_dim,
            prompt_len=args.prompt_len,
            seed=args.seed,
            expose_kv_cache_outputs=True,
        )
        decode0, _, prefill_ref, decode0_ref = build_llama_decode_artifact(
            d_model=args.d_model,
            d_head=args.d_head,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads,
            ffn_hidden_dim=args.ffn_dim,
            prompt_len=args.prompt_len,
            seed=args.seed,
            state=state,
        )
        decodes = [decode0]
        if args.decode_tokens >= 2:
            decode1_cache = extend_llama_kv_cache(prefill_ref, decode0_ref)
            decode1, _, _, _ = build_llama_decode_artifact(
                d_model=args.d_model,
                d_head=args.d_head,
                n_heads=args.n_heads,
                n_kv_heads=args.n_kv_heads,
                ffn_hidden_dim=args.ffn_dim,
                prompt_len=args.prompt_len,
                seed=args.seed,
                state=state,
                cache_ref=decode1_cache,
                x_decode_in=state["x_decode2_in"],
            )
            decodes.append(decode1)
        return prefill, decodes, args.n_kv_heads

    prefill, state, _ = build_gpt2_prefill_artifact(
        d_model=args.d_model,
        d_head=args.d_head,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        prompt_len=args.prompt_len,
        seed=args.seed,
    )
    decode0, _, prefill_ref, decode0_ref = build_gpt2_decode_artifact(
        d_model=args.d_model,
        d_head=args.d_head,
        n_heads=args.n_heads,
        ffn_dim=args.ffn_dim,
        prompt_len=args.prompt_len,
        seed=args.seed,
        state=state,
    )
    decodes = [decode0]
    if args.decode_tokens >= 2:
        decode1_cache = extend_gpt2_kv_cache(prefill_ref, decode0_ref)
        decode1, _, _, _ = build_gpt2_decode_artifact(
            d_model=args.d_model,
            d_head=args.d_head,
            n_heads=args.n_heads,
            ffn_dim=args.ffn_dim,
            prompt_len=args.prompt_len,
            seed=args.seed,
            state=state,
            cache_ref=decode1_cache,
            x_decode_in=state["x_decode2_in"],
        )
        decodes.append(decode1)
    return prefill, decodes, args.n_heads


def _canonical_model(name: str) -> str:
    if name in {"qllama", "llama"}:
        return "qllama"
    if name in {"qgpt2", "gpt2"}:
        return "qgpt2"
    raise ValueError(f"unsupported model {name}")


def _model_label(name: str) -> str:
    return "QLlama" if name == "qllama" else "QGPT2"


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
    out_text = _text_from_completed(stdout)
    err_text = _text_from_completed(stderr)
    cmd_text = " ".join(command or [])
    path.write_text(
        f"status={status}\n"
        f"command={cmd_text}\n"
        "\n--- stdout ---\n"
        f"{out_text}"
        "\n--- stderr ---\n"
        f"{err_text}"
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
        "+verbose",
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
        status = "success" if proc.returncode == 0 else f"failed:{proc.returncode}"
        log.write(f"\nstatus={status}\n")
        return proc.returncode, status


def main() -> int:
    parser = argparse.ArgumentParser(description="Build/run one CV32E40P image that executes prefill then decode with firmware-side KV handoff.")
    parser.add_argument(
        "--model",
        choices=("qllama", "qgpt2", "llama", "gpt2"),
        required=True,
        help="Model block family. Use qllama/qgpt2; llama/gpt2 are accepted as aliases.",
    )
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--d-head", type=int, default=16)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-kv-heads", type=int, default=None)
    parser.add_argument("--ffn-dim", type=int, default=64)
    parser.add_argument("--prompt-len", type=int, default=8)
    parser.add_argument("--decode-tokens", type=int, choices=(1, 2), default=1, help="Number of decode tokens to run after prefill.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-rtl", action="store_true")
    parser.add_argument("--maxcycles", type=int, default=20_000_000)
    parser.add_argument("--verilator-max-ticks", type=int, default=1_000_000_000_000)
    parser.add_argument("--timeout-s", type=int, default=900, help="Python wall-time timeout. Use 0 or a negative value to disable it.")
    parser.add_argument("--dump-stdout", action="store_true")
    parser.add_argument("--log-path", type=Path, default=None, help="Persistent RTL stdout/stderr log path. Defaults to runs/<program>_rtl_<timestamp>.log.")
    parser.add_argument("--sim-ram-bytes", type=lambda value: int(value, 0), default=0, help="Optional larger bare-metal RAM size for generated linker script, e.g. 0x1000000.")
    parser.add_argument("--sim-ram-addr-width", type=int, default=None, help="Optional Verilator tb RAM_ADDR_WIDTH override. Use with --sim-ram-bytes when the image exceeds 4 MiB.")
    args = parser.parse_args()
    args.model = _canonical_model(args.model)
    if args.n_kv_heads is None:
        args.n_kv_heads = args.n_heads if args.model == "qgpt2" else max(1, args.n_heads // 2)
    timeout_s = None if args.timeout_s <= 0 else args.timeout_s

    if args.model == "qgpt2" and args.n_kv_heads != args.n_heads:
        raise SystemExit("GPT-2 uses full multi-head KV; pass --n-kv-heads equal to --n-heads or omit it.")

    prefill_artifact, decode_artifacts, n_cache_heads = _build_artifacts(args)
    stem = (
        f"cv32e40p_{args.model}_prefill_decode{args.decode_tokens}_seq_d{args.d_model}_h{args.d_head}_"
        f"nh{args.n_heads}_nkv{args.n_kv_heads}_f{args.ffn_dim}_t{args.prompt_len}_s{args.seed}"
    )
    prefill_name = f"{stem}_prefill"
    decode_names = [f"{stem}_decode{i}" for i in range(len(decode_artifacts))]
    prefill_symbol = sanitize_program_symbol(prefill_name)
    decode_symbols = [sanitize_program_symbol(name) for name in decode_names]
    prefill_source = emit_cv32e40p_program_v2(prefill_artifact, {}, program_name=prefill_name)
    decode_sources = [
        emit_cv32e40p_program_v2(artifact, {}, program_name=name)
        for artifact, name in zip(decode_artifacts, decode_names, strict=True)
    ]
    linker_script = _linker_script_for_ram_bytes(args.sim_ram_bytes)
    runner_source = _combined_runner_source(
        prefill_symbol=prefill_symbol,
        decode_symbols=decode_symbols,
        model=_model_label(args.model),
        n_cache_heads=n_cache_heads,
    )
    prefill_path, decode_paths, runner_path, elf_path, hex_path = _build_sequence_elf_and_hex(
        program_name=stem,
        prefill_source=prefill_source,
        decode_sources=decode_sources,
        runner_source=runner_source,
        linker_script=linker_script,
        sim_ram_addr_width=args.sim_ram_addr_width,
    )
    print(f"prefill_program={prefill_path}")
    for idx, path in enumerate(decode_paths):
        print(f"decode{idx}_program={path}")
    print(f"runner={runner_path}")
    print(f"elf={elf_path}")
    print(f"hex={hex_path}")

    if args.run_rtl:
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
                if (
                    "prefill_decode_sequence:" in line
                    or "sequence." in line
                    or "VerifyTensor:" in line
                    or "verify." in line
                    or line.startswith("EXIT ")
                ):
                    print(line)
        if status.startswith("timeout"):
            print(f"RTL run timed out after {args.timeout_s}s", file=sys.stderr)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
