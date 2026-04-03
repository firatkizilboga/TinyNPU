from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPILER_ROOT = ROOT / "software" / "compiler"
for path in (ROOT, COMPILER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from software.compiler.tinynpu_jit import VerificationMode, run_host_emulation, write_cv32e40p_c  # noqa: E402
from software.workload.mnist_mlp_feature_benchmark import build_compiled_artifact_from_run  # noqa: E402


RUN_DIR = ROOT / "runs" / "tinynpu_issue27_backup_2026_03_21" / "mnist_mlp_iszero_int16_smoke"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-baseline", action="store_true", help="Emit timed CPU baseline matmul path for each NPU segment.")
    parser.add_argument(
        "--no-cpu-verify",
        action="store_true",
        help="Disable CPU-vs-NPU segment output cross-checks when CPU baseline emission is enabled.",
    )
    args = parser.parse_args()

    artifact, sample_image, sample_label, summary = build_compiled_artifact_from_run(
        str(RUN_DIR),
        dequantize_output=True,
    )
    inputs = {artifact.plan.inputs[0]: sample_image}
    host_result = run_host_emulation(
        artifact,
        inputs,
        verification=VerificationMode.FINAL,
        debug=False,
    )
    generated_dir = ROOT / "generated"
    generated_dir.mkdir(exist_ok=True)
    output_path = generated_dir / "cv32e40p_iszero_mlp_demo.c"
    write_cv32e40p_c(
        artifact,
        inputs,
        output_path,
        program_name="cv32e40p_iszero_mlp_demo",
        emit_cpu_baseline=args.cpu_baseline,
        verify_cpu_baseline=args.cpu_baseline and not args.no_cpu_verify,
    )
    prediction = int(float(host_result.tensors[artifact.plan.outputs[0]].reshape(-1)[0]) >= 0.5)
    print(output_path)
    print(f"sample_label={sample_label}")
    print(f"sample_pred={prediction}")
    print(f"qat_acc={float(summary.get('qat_acc', 0.0))}")
    print(f"steps={[type(step).__name__ for step in artifact.plan.steps]}")


if __name__ == "__main__":
    main()
