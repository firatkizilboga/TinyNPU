from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_synth_module():
    script_path = Path(__file__).resolve().parents[3] / "scripts" / "synth_tinynpu_yosys.py"
    spec = importlib.util.spec_from_file_location("synth_tinynpu_yosys", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_stats_extracts_summary_fields():
    synth = _load_synth_module()
    log = """
=== tinynpu_top ===

   Number of cells:
     227 DSP48E1
     9495 FDCE
     1840 CARRY4
     3408 MUXF7
     1057 MUXF8
     264 LUT1
     8156 LUT2
     4158 LUT3
     6430 LUT4
     4493 LUT5
     10036 LUT6

Estimated number of LCs: 25117
End of script.
"""
    stats = synth.parse_stats(log)
    assert stats["lcs"] == "25117"
    assert stats["DSP48E1"] == "227"
    assert stats["FDCE"] == "9495"
    assert stats["LUT6"] == "10036"


def test_stage_rtl_strips_debug_blocks_and_can_blackbox_memories(tmp_path):
    synth = _load_synth_module()
    staged = tmp_path / "rtl_abstract"

    synth.stage_rtl(staged, abstract_memory=True)

    top_text = (staged / "tinynpu_top.sv").read_text()
    ppu_text = (staged / "ppu.sv").read_text()
    ub_text = (staged / "unified_buffer.sv").read_text()
    im_text = (staged / "instruction_memory.sv").read_text()

    assert "$test$plusargs(\"trace\")" not in top_text
    assert "$display" not in ppu_text
    assert "(* blackbox *) module unified_buffer" in ub_text
    assert "(* blackbox *) module instruction_memory" in im_text


def test_stage_rtl_keeps_real_memory_rtl_when_not_abstracted(tmp_path):
    synth = _load_synth_module()
    staged = tmp_path / "rtl_full"

    synth.stage_rtl(staged, abstract_memory=False)

    ub_text = (staged / "unified_buffer.sv").read_text()
    im_text = (staged / "instruction_memory.sv").read_text()

    assert "(* blackbox *) module unified_buffer" not in ub_text
    assert "(* blackbox *) module instruction_memory" not in im_text
