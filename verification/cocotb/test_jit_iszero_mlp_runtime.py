import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import ClockCycles
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.append(project_root)

RUN_DIR = os.path.join(
    project_root,
    "runs",
    "tinynpu_issue27_backup_2026_03_21",
    "mnist_mlp_iszero_int16_smoke",
)

from software.workload.mnist_mlp_feature_benchmark import (  # noqa: E402
    TASK_IS_ZERO,
    build_compiled_artifact_from_run,
)
from software.compiler.tinynpu_jit import (  # noqa: E402
    DType,
    IRBuilder,
    TensorKind,
    TensorSpec,
    VerificationMode,
    compile_plan,
    run_host_emulation,
    run_sim,
)


def _build_synthetic_artifact():
    rng = np.random.default_rng(7)
    sample = rng.integers(-16, 17, size=(1, 8), dtype=np.int16)
    w1 = rng.integers(-2, 3, size=(8, 8), dtype=np.int16)
    w2 = rng.integers(-2, 3, size=(8, 8), dtype=np.int16)
    w3 = rng.integers(-2, 3, size=(8, 1), dtype=np.int16)

    b = IRBuilder()
    b.add_tensor(TensorSpec("x", (1, 8), DType.INT16, TensorKind.INPUT, metadata={"storage_role": "A"}))
    b.add_tensor(TensorSpec("w1", (8, 8), DType.INT16, TensorKind.CONSTANT, data=w1))
    b.add_tensor(TensorSpec("w2", (8, 8), DType.INT16, TensorKind.CONSTANT, data=w2))
    b.add_tensor(TensorSpec("w3", (8, 1), DType.INT16, TensorKind.CONSTANT, data=w3))
    b.add_tensor(TensorSpec("h1", (1, 8), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}))
    b.add_tensor(TensorSpec("h2", (1, 8), DType.INT16, TensorKind.INTERMEDIATE, metadata={"storage_role": "A"}))
    b.add_tensor(TensorSpec("out_i", (1, 1), DType.INT16, TensorKind.INTERMEDIATE))
    b.add_tensor(TensorSpec("dq_out", (1, 1), DType.FLOAT32, TensorKind.OUTPUT, is_final_output=True))
    b.segment(
        "fc1_relu",
        ops=[b.matmul("op_fc1", "x", "w1", "h1", activation="relu", in_dtype=DType.INT16, out_dtype=DType.INT16)],
        inputs=["x"],
        outputs=["h1"],
    )
    b.segment(
        "fc2_hgelu",
        ops=[b.matmul("op_fc2", "h1", "w2", "h2", activation="h_gelu", h_gelu_x_scale_shift=7, in_dtype=DType.INT16, out_dtype=DType.INT16)],
        inputs=["h1"],
        outputs=["h2"],
    )
    b.segment(
        "fc3",
        ops=[b.matmul("op_fc3", "h2", "w3", "out_i", in_dtype=DType.INT16, out_dtype=DType.INT16)],
        inputs=["h2"],
        outputs=["out_i"],
    )
    b.host("dequant_out", "dequantize", inputs=["out_i"], outputs=["dq_out"], attrs={"scale": 1.0 / 64.0, "zero_point": 0})
    plan = b.finalize(inputs=["x"], outputs=["dq_out"])
    artifact = compile_plan(plan, {})
    return artifact, sample, -1, {"task": TASK_IS_ZERO, "source": "synthetic_ir"}


@cocotb.test()
async def test_jit_iszero_mlp_runtime(dut):
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 5)
    dut.rst_n.value = 1

    if os.path.exists(os.path.join(RUN_DIR, "summary.json")):
        artifact, sample, label, _summary = build_compiled_artifact_from_run(
            RUN_DIR,
            dequantize_output=True,
        )
    else:
        artifact, sample, label, _summary = _build_synthetic_artifact()
    inputs = {"x": np.array(sample, copy=True)}

    host_result = run_host_emulation(
        artifact,
        inputs,
        VerificationMode.DEBUG,
    )
    rtl_result = await run_sim(
        artifact,
        inputs,
        dut=dut,
        verification=VerificationMode.OFF,
        debug=True,
    )

    host_out = host_result.tensors["dq_out"]
    rtl_out = rtl_result.tensors["dq_out"]
    print("label", label)
    print("host dq_out", host_out)
    print("rtl dq_out", rtl_out)
    print("host debug trace", host_result.debug_trace)
    print("rtl debug trace", rtl_result.debug_trace)
    assert np.allclose(rtl_out, host_out, rtol=1e-5, atol=1e-6)
