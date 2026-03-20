import math
import os
import sys


TESTS_DIR = os.path.dirname(__file__)
COMPILER_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, COMPILER_DIR)

from tinynpu_jit.golden import GoldenModel, h_gelu


PARAMS = {"x_scale_shift": 7, "slope_num": 218, "slope_shift": 7}


def _dequant(x_int: int, *, x_scale_shift: int) -> float:
    return x_int / float(1 << x_scale_shift)


def _float_h_gelu(x: float) -> float:
    gate = min(max((1.702 * x) + 3.0, 0.0), 6.0) / 6.0
    return x * gate


def test_h_gelu_matches_expected_regions():
    assert h_gelu(0, **PARAMS) == 0
    assert h_gelu(-256, **PARAMS) == 0  # about -2.0 in the default domain
    assert h_gelu(256, **PARAMS) == 256  # about +2.0 in the default domain


def test_h_gelu_has_expected_negative_shoulder():
    assert h_gelu(-256, **PARAMS) == 0
    assert h_gelu(-192, **PARAMS) < 0
    assert h_gelu(-64, **PARAMS) < 0


def test_h_gelu_tracks_float_shape():
    for x_in in [-384, -256, -192, -128, -64, -32, 0, 32, 64, 128, 192, 256, 384]:
        got = _dequant(h_gelu(x_in, **PARAMS), x_scale_shift=PARAMS["x_scale_shift"])
        want = _float_h_gelu(_dequant(x_in, x_scale_shift=PARAMS["x_scale_shift"]))
        assert abs(got - want) < 0.04, (x_in, got, want)


def test_h_gelu_accepts_custom_integer_slope():
    base = h_gelu(96, **PARAMS)
    tuned = h_gelu(96, x_scale_shift=7, slope_num=224, slope_shift=7)
    assert tuned >= base


def test_h_gelu_validates_parameters():
    try:
        h_gelu(1, x_scale_shift=-1)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for negative x_scale_shift.")

    try:
        h_gelu(1, slope_num=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for non-positive slope_num.")


def test_golden_model_exposes_same_h_gelu_helper():
    golden = GoldenModel()
    assert golden.h_gelu(120, **PARAMS) == h_gelu(120, **PARAMS)
