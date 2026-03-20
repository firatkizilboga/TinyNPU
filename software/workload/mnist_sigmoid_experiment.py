from __future__ import annotations

import json
from pathlib import Path

from software.workload.mnist_tinynpu_pipeline import run_pipeline


def run_sigmoid_experiment(
    *,
    run_dir: str,
    data_dir: str = "/home/firatkizilboga/compiler-optimization/data",
    fp32_epochs: int = 20,
    qat_epochs: int = 5,
    compiled_eval_samples: int = 256,
):
    summary, artifact = run_pipeline(
        run_dir=run_dir,
        data_dir=data_dir,
        fp32_epochs=fp32_epochs,
        qat_epochs=qat_epochs,
        compiled_eval_samples=compiled_eval_samples,
        activation="relu",
        output_activation="sigmoid",
        use_di_sigmoid_approx=True,
    )
    summary["compiled_vs_qat_drop"] = summary["qat_acc"] - summary["compiled_host_acc"]

    run_path = Path(run_dir)
    with open(run_path / "sigmoid_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    return summary, artifact


if __name__ == "__main__":
    summary, _ = run_sigmoid_experiment(run_dir="runs/mnist_tinynpu_sigmoid")
    print(json.dumps(summary, indent=2))
