import os
import sys


TESTS_DIR = os.path.dirname(__file__)
COMPILER_DIR = os.path.dirname(TESTS_DIR)
sys.path.insert(0, COMPILER_DIR)

from tinynpu_jit.golden import GoldenModel, di_exp


def test_di_exp_matches_hand_worked_examples():
    # m_f = 128 + 64 - 8 = 184
    # s_f = 184 / 2^12 ~= 0.0449
    # t = round(-1 / s_f) = -22
    params = {"m_i": 128, "k_i": 12}
    expected = {
        0: 22,
        -1: 21,
        -22: 11,
        -23: 10,
        -44: 5,
        -45: 5,
    }

    for x_in, want in expected.items():
        assert di_exp(x_in, **params) == want


def test_di_exp_handles_multiple_integer_parameter_sets():
    parameter_sets = [
        {"m_i": 128, "k_i": 12},
        {"m_i": 92, "k_i": 10},
        {"m_i": 64, "k_i": 8},
    ]
    for params in parameter_sets:
        ys = [di_exp(x_in, **params) for x_in in range(-64, 1)]
        assert all(y >= 0 for y in ys)
        for prev, curr in zip(ys, ys[1:]):
            assert prev <= curr, f"DI-Exp should be non-decreasing as input increases: {prev} !<= {curr}"


def test_di_exp_is_monotone_over_negative_domain():
    params = {"m_i": 128, "k_i": 12}
    xs = list(range(-128, 1))
    ys = [di_exp(x_in, **params) for x_in in xs]

    for prev, curr in zip(ys, ys[1:]):
        assert prev <= curr, f"DI-Exp should be non-decreasing as input increases: {prev} !<= {curr}"


def test_di_exp_stays_non_negative():
    params = {"m_i": 128, "k_i": 12}
    for x_in in range(-256, 1):
        assert di_exp(x_in, **params) >= 0


def test_di_exp_rejects_degenerate_fixed_parameters():
    try:
        di_exp(-8, m_i=1024, k_i=4)
    except ValueError as exc:
        assert "period to zero" in str(exc)
    else:
        raise AssertionError("Expected degenerate fixed parameters to raise ValueError.")


def test_golden_model_exposes_same_di_exp_helper():
    golden = GoldenModel()
    assert golden.di_exp(-23, m_i=128, k_i=12) == di_exp(-23, m_i=128, k_i=12)
