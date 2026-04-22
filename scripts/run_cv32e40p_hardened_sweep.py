from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "software" / "compiler") not in sys.path:
    sys.path.insert(0, str(ROOT / "software" / "compiler"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from tinynpu_jit import emit_cv32e40p_program_v2  # noqa: E402
from software.workload.mnist_mlp_feature_benchmark import build_compiled_artifact_from_run as build_mlp_artifact  # noqa: E402
from software.workload.mnist_tinynpu_pipeline import build_compiled_artifact_from_run as build_conv_artifact  # noqa: E402
from run_cv32e40p_b_append_demo import RunnerConfig, build_v2_elf_and_hex, run_vlt_npu  # noqa: E402
from run_cv32e40p_gpt2_two_block_reuse_jit_demo import build_artifact as build_gpt2_two_block_artifact  # noqa: E402


MLP_RUN_DIR = ROOT / "runs" / "tinynpu_issue27_backup_2026_03_21" / "mnist_mlp_iszero_int16_smoke"
CONV_RUN_DIR = ROOT / "runs" / "mnist_tinynpu_pipeline"


def _build_hex_from_artifact(artifact, *, program_name: str, runtime_inputs: dict[str, object] | None = None) -> Path:
    source = emit_cv32e40p_program_v2(
        artifact,
        runtime_inputs or {},
        program_name=program_name,
    )
    _, _, _, hex_path = build_v2_elf_and_hex(
        program_name,
        source,
        runner_config=RunnerConfig(
            repeat_count=1,
            dump_final_outputs=True,
            verbose_steps=False,
        ),
    )
    return hex_path


def _run_case(name: str, hex_path: Path, *, maxcycles: int, max_ticks: int, timeout_s: int) -> str:
    result = run_vlt_npu(
        hex_path,
        maxcycles=maxcycles,
        verilator_max_ticks=max_ticks,
        timeout_s=timeout_s,
    )
    output = result.stdout.strip()
    print(f"===== {name} =====")
    if output:
        print(output)
    else:
        print("(no stdout)")
    print()
    return output


def main() -> int:
    mlp_artifact, mlp_example, _, _ = build_mlp_artifact(str(MLP_RUN_DIR), dequantize_output=True)
    mlp_hex = _build_hex_from_artifact(
        mlp_artifact,
        program_name="cv32e40p_iszero_mlp_hardened_sweep",
        runtime_inputs={mlp_artifact.plan.inputs[0]: mlp_example},
    )
    _run_case("MLP", mlp_hex, maxcycles=500_000, max_ticks=10_000_000_000, timeout_s=240)

    conv_artifact, conv_example, _ = build_conv_artifact(str(CONV_RUN_DIR), dequantize_output=True)
    conv_hex = _build_hex_from_artifact(
        conv_artifact,
        program_name="cv32e40p_fair_conv_multilayer_off_hardened_sweep",
        runtime_inputs={conv_artifact.plan.inputs[0]: conv_example},
    )
    _run_case("CONV", conv_hex, maxcycles=1_000_000, max_ticks=10_000_000_000, timeout_s=900)

    gpt2_artifact = build_gpt2_two_block_artifact(prompt_len=8)
    gpt2_hex = _build_hex_from_artifact(gpt2_artifact, program_name="cv32e40p_gpt2_two_block_reuse_d8_h8_nh1_f8_t8_hardened_sweep")
    _run_case("GPT2", gpt2_hex, maxcycles=1_000_000, max_ticks=30_000_000_000, timeout_s=420)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
