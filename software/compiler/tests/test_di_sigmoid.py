import math
import os
import sys


TESTS_DIR = os.path.dirname(__file__)
COMPILER_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, COMPILER_DIR)

from tinynpu_jit.golden import GoldenModel, di_sigmoid
from tinynpu_jit.ir import DType


def _dequant_prob(y_int: int, *, p_out: int) -> float:
    return y_int / float((1 << (p_out - 1)) - 1)


def test_di_sigmoid_is_monotone():
    params = {"m_i": 128, "k_i": 12, "p_out": 8}
    ys = [di_sigmoid(x_in, **params) for x_in in range(-128, 129)]
    for prev, curr in zip(ys, ys[1:]):
        assert prev <= curr, f"DI-Sigmoid should be non-decreasing: {prev} !<= {curr}"


def test_di_sigmoid_tracks_float_sigmoid_shape():
    params = {"m_i": 128, "k_i": 12, "p_out": 8}
    m_f = params["m_i"] + (params["m_i"] >> 1) - (params["m_i"] >> 4)
    scale = m_f / float(1 << params["k_i"])

    for x_in in [-64, -32, -16, -8, -4, 0, 4, 8, 16, 32, 64]:
        got = _dequant_prob(di_sigmoid(x_in, **params), p_out=params["p_out"])
        want = 1.0 / (1.0 + math.exp(-(x_in * scale)))
        assert abs(got - want) < 0.12, (x_in, got, want)


def test_di_sigmoid_is_approximately_symmetric():
    params = {"m_i": 128, "k_i": 12, "p_out": 8}
    full_scale = (1 << (params["p_out"] - 1)) - 1
    for x_in in [0, 4, 8, 16, 32, 64]:
        pos = di_sigmoid(x_in, **params)
        neg = di_sigmoid(-x_in, **params)
        assert abs((pos + neg) - full_scale) <= 2, (x_in, pos, neg, full_scale)


def test_di_sigmoid_handles_optional_smoothing():
    base = di_sigmoid(16, m_i=128, k_i=12, p_out=8, alpha_smooth=1)
    smoothed = di_sigmoid(16, m_i=128, k_i=12, p_out=8, alpha_smooth=2)
    assert smoothed < base


def test_di_sigmoid_supports_large_shift_values():
    assert di_sigmoid(0, m_i=26845, k_i=35, p_out=8, alpha_smooth=1) > 0


def test_golden_model_exposes_same_di_sigmoid_helper():
    golden = GoldenModel()
    assert golden.di_sigmoid(12, m_i=128, k_i=12, p_out=8) == di_sigmoid(12, m_i=128, k_i=12, p_out=8)


def test_golden_matmul_accepts_sigmoid_activation():
    golden = GoldenModel()
    lhs = [[4, -2], [1, 3]]
    rhs = [[2, 1], [-1, 2]]
    out = golden.matmul(lhs, rhs, multiplier=128, shift=12, activation="sigmoid", out_dtype=DType.INT8)
    assert out.shape == (2, 2)
    assert out.min() >= 0
    assert out.max() <= 127
