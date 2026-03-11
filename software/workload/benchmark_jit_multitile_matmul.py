from __future__ import annotations

from software.compiler.tinynpu_jit import (
    VerificationMode,
    five_stage_in_order_model,
    ideal_issue_1_model,
    run_host_emulation,
    unpipelined_scalar_model,
)
from software.workload.jit_multitile_matmul import build_multitile_matmul_artifact


def main() -> None:
    artifact, _ = build_multitile_matmul_artifact()
    result = run_host_emulation(
        artifact,
        {},
        verification=VerificationMode.OFF,
        debug=False,
        benchmark=True,
    )
    print(artifact.format_benchmark_report(result))
    print(
        artifact.format_benchmark_comparison(
            result,
            [unpipelined_scalar_model(), ideal_issue_1_model(), five_stage_in_order_model()],
        )
    )
    print("For RTL-backed NPU cycle data run:")
    print("  cd verification/cocotb && CCACHE_DISABLE=1 USER_EXTRA_ARGS=-GPERF_ENABLE=1 MODULE=test_jit_benchmark_multitile_matmul make -f Makefile.npu")


if __name__ == "__main__":
    main()
