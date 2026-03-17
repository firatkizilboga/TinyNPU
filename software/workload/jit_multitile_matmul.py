from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from software.compiler.tinynpu_jit import (
    CompiledArtifact,
    DType,
    ExecutionPlan,
    MatMulOp,
    NpuSegment,
    TensorKind,
    TensorSpec,
    VerificationMode,
    VerifyTensor,
    compile_plan,
    run_host_emulation,
)
from software.compiler.tinynpu_jit.golden import GoldenModel


@dataclass(frozen=True)
class JitMatmulBenchmarkCase:
    name: str
    m: int
    k: int
    n: int
    in_dtype: DType = DType.INT16
    out_dtype: DType = DType.INT16
    lhs_runtime_input: bool = True
    seed: int = 0

    @property
    def total_macs(self) -> int:
        return int(self.m * self.k * self.n)


def default_gemm_benchmark_cases() -> list[JitMatmulBenchmarkCase]:
    shapes = [
        ("gemm_64x64x64", 64, 64, 64),
        ("gemm_96x64x96", 96, 64, 96),
        ("gemm_128x128x128", 128, 128, 128),
    ]
    dtypes = [DType.INT16, DType.INT8, DType.INT4]
    cases: list[JitMatmulBenchmarkCase] = []
    seed = 0
    for shape_name, m, k, n in shapes:
        for dtype in dtypes:
            seed += 1
            cases.append(
                JitMatmulBenchmarkCase(
                    name=f"{shape_name}_{dtype.value}",
                    m=m,
                    k=k,
                    n=n,
                    in_dtype=dtype,
                    out_dtype=DType.INT16,
                    lhs_runtime_input=True,
                    seed=seed,
                )
            )
    return cases


def _default_multitile_case() -> JitMatmulBenchmarkCase:
    return JitMatmulBenchmarkCase(
        name="multitile_default",
        m=13,
        k=17,
        n=19,
        in_dtype=DType.INT16,
        out_dtype=DType.INT16,
        lhs_runtime_input=False,
        seed=0,
    )


def _operand_bound(case: JitMatmulBenchmarkCase) -> int:
    dtype_bound = {
        DType.INT4: 7,
        DType.INT8: 127,
        DType.INT16: 32767,
    }[case.in_dtype]
    # Keep outputs mostly within INT16 range so the benchmark is not dominated
    # by requant/clamp saturation when out_dtype is left at INT16.
    safe_bound = max(1, int(np.sqrt(30000.0 / max(case.k, 1))))
    return max(1, min(dtype_bound, safe_bound))


def _generate_operands(case: JitMatmulBenchmarkCase) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(case.seed)
    bound = _operand_bound(case)
    lhs = rng.integers(-bound, bound + 1, size=(case.m, case.k), dtype=np.int32).astype(np.int16)
    rhs = rng.integers(-bound, bound + 1, size=(case.k, case.n), dtype=np.int32).astype(np.int16)
    return lhs, rhs


def build_configured_matmul_artifact(
    case: JitMatmulBenchmarkCase,
) -> tuple[CompiledArtifact, dict[str, np.ndarray], np.ndarray]:
    lhs, rhs = _generate_operands(case)
    expected = GoldenModel().matmul(lhs, rhs, out_dtype=case.out_dtype)

    lhs_kind = TensorKind.INPUT if case.lhs_runtime_input else TensorKind.CONSTANT
    lhs_data = None if case.lhs_runtime_input else lhs
    tensors = {
        "lhs": TensorSpec("lhs", lhs.shape, case.in_dtype, lhs_kind, data=lhs_data),
        "rhs": TensorSpec("rhs", rhs.shape, case.in_dtype, TensorKind.CONSTANT, data=rhs),
        "out": TensorSpec(
            "out",
            expected.shape,
            case.out_dtype,
            TensorKind.OUTPUT,
            is_final_output=True,
        ),
    }
    plan = ExecutionPlan(
        tensors=tensors,
        steps=[
            NpuSegment(
                "segment_000",
                [
                    MatMulOp(
                        "mm_benchmark",
                        "lhs",
                        "rhs",
                        "out",
                        in_dtype=case.in_dtype,
                        out_dtype=case.out_dtype,
                    )
                ],
                ["lhs", "rhs"],
                ["out"],
            ),
            VerifyTensor("out", "out", is_final_output=True),
        ],
        inputs=["lhs"] if case.lhs_runtime_input else [],
        outputs=["out"],
        metadata={
            "shape": {
                "lhs": list(lhs.shape),
                "rhs": list(rhs.shape),
                "out": list(expected.shape),
            },
            "benchmark_case": {
                "name": case.name,
                "m": case.m,
                "k": case.k,
                "n": case.n,
                "in_dtype": case.in_dtype.value,
                "out_dtype": case.out_dtype.value,
                "lhs_runtime_input": case.lhs_runtime_input,
                "total_macs": case.total_macs,
            },
        },
    )
    inputs = {"lhs": lhs} if case.lhs_runtime_input else {}
    return compile_plan(plan, {"out": expected}), inputs, expected


def build_multitile_matmul_artifact():
    """
    Build a single-segment matmul whose logical M/K/N all exceed ARRAY_SIZE=8.

    Shape choice:
    - lhs: 13 x 17
    - rhs: 17 x 19
    - out: 13 x 19

    This forces:
    - M tiles > 1
    - K tiles > 1
    - N tiles > 1
    """
    case = _default_multitile_case()
    artifact, _, expected = build_configured_matmul_artifact(case)
    return artifact, expected


def smoke_run_multitile_matmul():
    artifact, expected = build_multitile_matmul_artifact()
    result = run_host_emulation(artifact, {}, VerificationMode.DEBUG, debug=True)
    output = result.tensors[artifact.plan.outputs[0]]
    return {
        "expected": expected,
        "output": output,
        "verified": result.verified,
        "debug_kinds": [event["kind"] for event in result.debug_trace],
    }


if __name__ == "__main__":
    summary = smoke_run_multitile_matmul()
    print("verified", summary["verified"])
    print("output_shape", list(summary["output"].shape))
    print("debug_kinds", summary["debug_kinds"])
