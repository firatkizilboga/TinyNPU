import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from tinynpu_jit.rtl_runner import RunnerConfig, render_runner_source, runtime_cflags, sanitize_program_symbol


def test_render_runner_source_uses_single_run_by_default():
    source = render_runner_source("demo_program")

    assert "extern const TnpuProgram demo_program;" in source
    assert "return tinynpu_run(program, ip, op, NULL, 0u);" in source
    assert "tinynpu_run_repeat" not in source
    assert "tinynpu_set_force_mmio(1);" not in source
    assert 'puts("' not in source


def test_render_runner_source_supports_repeat_banner_and_mmio():
    source = render_runner_source(
        "demo_program",
        RunnerConfig(repeat_count=3, force_mmio=True, banner="runner banner"),
    )

    assert 'puts("runner banner");' in source
    assert "tinynpu_set_force_mmio(1);" in source
    assert "return tinynpu_run_repeat(program, ip, op, NULL, 0u, 3u);" in source


def test_runtime_cflags_respect_runner_config():
    cflags = runtime_cflags(
        RunnerConfig(dump_final_outputs=False, verbose_steps=False),
        extra_cflags=["-DTEST_FLAG=1"],
    )

    assert "-O3" in cflags
    assert "-DTINYNPU_USE_SHARED_SRAM=1" in cflags
    assert "-DTNPU_RUNTIME_V2_DUMP_FINAL_OUTPUTS=0" in cflags
    assert "-DTNPU_RUNTIME_V2_VERBOSE_STEPS=0" in cflags
    assert "-DTEST_FLAG=1" in cflags


def test_sanitize_program_symbol_normalizes_c_identifier():
    assert sanitize_program_symbol("cv32e40p-demo/v2") == "cv32e40p_demo_v2"


def test_runtime_v2_has_no_hardware_xform_qdq_path():
    runtime_source = Path(__file__).parents[1] / "tinynpu_jit" / "tinynpu_runtime_v2.c"
    runtime_header = Path(__file__).parents[1] / "tinynpu_jit" / "tinynpu_runtime_v2.h"
    source = runtime_source.read_text()
    header = runtime_header.read_text()

    assert "TNPU_WRITE_XFORM_Q_F32_I16" not in source
    assert "TNPU_READ_XFORM_DQ_I16_F32" not in source
    assert "TNPU_WRITE_XFORM_Q_F32_I16" not in header
    assert "TNPU_READ_XFORM_DQ_I16_F32" not in header
    assert "tnpu_write_tensor_quantized_via_xform" not in source
    assert "tnpu_write_tensor_a_qf16_to_i16_fast" not in source
