from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMPILER_ROOT = PROJECT_ROOT / "software" / "compiler"
for path in (PROJECT_ROOT, COMPILER_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from software.workload.mnist_tinynpu_pipeline import (  # noqa: E402
    TinyMNISTFP32,
    VerificationMode,
    apply_compiler_boundary_overrides,
    build_compiler_ready_mnist_model,
    build_signed_w16a16_configs,
    build_signed_w8a8_configs,
    compile_module,
    evaluate_compiled_artifact,
    evaluate_model,
    finetune_qat,
    get_mnist_loaders,
    initialize_qat_from_fp32,
    run_host_emulation,
    run_pipeline,
)


DEFAULT_DATA_DIR = "/home/firatkizilboga/compiler-optimization/data"


def run_gelu_experiment(
    *,
    run_dir: str,
    data_dir: str = DEFAULT_DATA_DIR,
    fp32_epochs: int = 20,
    qat_epochs: int = 5,
    compiled_eval_samples: int = 256,
    int16_only: bool = False,
):
    layer_configs = build_signed_w16a16_configs() if int16_only else build_signed_w8a8_configs()
    summary, artifact = run_pipeline(
        run_dir=run_dir,
        data_dir=data_dir,
        fp32_epochs=fp32_epochs,
        qat_epochs=qat_epochs,
        compiled_eval_samples=compiled_eval_samples,
        activation="gelu",
        output_activation="none",
        use_h_gelu_approx=True,
        layer_configs=layer_configs,
    )
    summary["int16_only"] = bool(int16_only)
    summary["compiled_vs_qat_drop"] = summary["qat_acc"] - summary["compiled_host_acc"]

    run_path = Path(run_dir)
    with open(run_path / "gelu_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    return summary, artifact


def run_gelu_experiment_from_fp32(
    *,
    run_dir: str,
    fp32_checkpoint: str,
    data_dir: str = DEFAULT_DATA_DIR,
    qat_epochs: int = 5,
    qat_lr: float = 1e-4,
    compiled_eval_samples: int = 256,
    int16_only: bool = False,
):
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    fp32_ckpt = Path(fp32_checkpoint)
    if not fp32_ckpt.exists():
        raise FileNotFoundError(f"Missing fp32 checkpoint: {fp32_ckpt}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading_fp32={fp32_ckpt}", flush=True)
    print(f"device={device}", flush=True)

    train_loader, test_loader, test_loader_single, train_ds, test_ds = get_mnist_loaders(data_dir)
    layer_configs = build_signed_w16a16_configs() if int16_only else build_signed_w8a8_configs()

    fp32_model = TinyMNISTFP32(activation="gelu", output_activation="none").to(device)
    fp32_model.load_state_dict(torch.load(fp32_ckpt, map_location=device))
    fp32_model.eval()

    fp32_acc = evaluate_model(fp32_model, test_loader, device, progress_label="stage=fp32_eval")
    print(f"stage=fp32_eval acc={fp32_acc * 100:.2f}%", flush=True)

    calib_loader = DataLoader(Subset(train_ds, range(1000)), batch_size=256, shuffle=False)
    print("stage=qat_init", flush=True)
    qat_model = initialize_qat_from_fp32(
        fp32_model,
        layer_configs=layer_configs,
        calib_loader=calib_loader,
        device=device,
        activation="gelu",
        output_activation="none",
        use_h_gelu_approx=True,
    )
    apply_compiler_boundary_overrides(qat_model, calib_dataset=test_ds, activation="gelu")

    print("stage=ptq_eval", flush=True)
    ptq_acc = evaluate_model(qat_model, test_loader, device, progress_label="stage=ptq_eval")
    print(f"stage=ptq_eval_done acc={ptq_acc * 100:.2f}%", flush=True)

    print("stage=qat_finetune", flush=True)
    qat_model = finetune_qat(
        qat_model,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        calib_dataset=test_ds,
        epochs=qat_epochs,
        lr=qat_lr,
        activation="gelu",
        output_activation="none",
    )
    torch.save(qat_model.state_dict(), run_path / "qat.pt")

    qat_acc = evaluate_model(qat_model, test_loader, device, progress_label="stage=qat_eval")
    print(f"stage=qat_eval acc={qat_acc * 100:.2f}%", flush=True)

    print("stage=compiler_ready", flush=True)
    compiler_ready = build_compiler_ready_mnist_model(
        qat_model,
        calib_dataset=test_ds,
        activation="gelu",
        output_activation="none",
    )

    example = test_ds[0][0].unsqueeze(0)
    print("stage=compile_module", flush=True)
    artifact = compile_module(compiler_ready, (example,))

    print("stage=compiled_eval", flush=True)
    compiled_acc = evaluate_compiled_artifact(
        artifact,
        test_loader_single,
        max_samples=compiled_eval_samples,
        progress_label="stage=compiled_eval",
    )
    print(f"stage=compiled_eval_done acc={compiled_acc * 100:.2f}%", flush=True)

    sample_image, sample_label = test_ds[0]
    print("stage=sample_debug", flush=True)
    sample_result = run_host_emulation(
        artifact,
        {artifact.plan.inputs[0]: sample_image.unsqueeze(0).numpy()},
        VerificationMode.DEBUG,
        debug=True,
    )
    sample_logits = sample_result.tensors[artifact.plan.outputs[0]].reshape(-1).tolist()

    summary = {
        "device": device,
        "activation": "gelu",
        "output_activation": "none",
        "use_h_gelu_approx": True,
        "int16_only": bool(int16_only),
        "fp32_checkpoint": str(fp32_ckpt),
        "fp32_acc": fp32_acc,
        "ptq_acc": ptq_acc,
        "qat_acc": qat_acc,
        "compiled_host_acc_samples": compiled_eval_samples,
        "compiled_host_acc": compiled_acc,
        "sample_label": int(sample_label),
        "sample_pred": int(torch.tensor(sample_logits).argmax().item()),
        "sample_logits": sample_logits,
        "debug_kinds": [event["kind"] for event in sample_result.debug_trace],
        "compiled_vs_qat_drop": qat_acc - compiled_acc,
    }
    with open(run_path / "gelu_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary, artifact


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TinyNPU GELU/h-GELU MNIST experiment.")
    parser.add_argument("--run-dir", default="runs/mnist_tinynpu_gelu", help="Output directory for checkpoints and summary.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="MNIST data directory.")
    parser.add_argument("--fp32-epochs", type=int, default=20, help="FP32 training epochs for a fresh run.")
    parser.add_argument("--qat-epochs", type=int, default=5, help="QAT fine-tuning epochs.")
    parser.add_argument("--qat-lr", type=float, default=1e-4, help="QAT learning rate for resume mode.")
    parser.add_argument("--compiled-eval-samples", type=int, default=256, help="Number of compiled-host samples to evaluate.")
    parser.add_argument("--int16-only", action="store_true", help="Use W16A16 QAT/compiler configs instead of W8A8.")
    parser.add_argument(
        "--resume-fp32-from",
        default=None,
        help="Existing FP32 checkpoint to reuse instead of retraining FP32.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if args.resume_fp32_from:
        summary, _ = run_gelu_experiment_from_fp32(
            run_dir=args.run_dir,
            fp32_checkpoint=args.resume_fp32_from,
            data_dir=args.data_dir,
            qat_epochs=args.qat_epochs,
            qat_lr=args.qat_lr,
            compiled_eval_samples=args.compiled_eval_samples,
            int16_only=args.int16_only,
        )
    else:
        summary, _ = run_gelu_experiment(
            run_dir=args.run_dir,
            data_dir=args.data_dir,
            fp32_epochs=args.fp32_epochs,
            qat_epochs=args.qat_epochs,
            compiled_eval_samples=args.compiled_eval_samples,
            int16_only=args.int16_only,
        )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
