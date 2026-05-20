import pytest

from software.compiler.tinynpu_quant.fused_params import MAX_PPU_RESCALE_SHIFT, synthesize_rescale


def test_synthesize_rescale_defaults_to_ppu_shift_limit():
    params = synthesize_rescale(1.0 / float(1 << MAX_PPU_RESCALE_SHIFT))

    assert params.multiplier == 1
    assert params.shift == MAX_PPU_RESCALE_SHIFT


def test_synthesize_rescale_rejects_scales_that_need_larger_ppu_shift():
    with pytest.raises(ValueError, match=f"shift<={MAX_PPU_RESCALE_SHIFT}"):
        synthesize_rescale(1.0 / float(1 << (MAX_PPU_RESCALE_SHIFT + 1)))
