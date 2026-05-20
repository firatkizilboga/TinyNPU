import pytest

from tinynpu.isa import MatMul, pack_matmul
from tinynpu.program import TinyNPUProgram


def test_matmul_accepts_supported_h_gelu_shift():
    instr = MatMul("a", "b", "c", h_gelu_x_scale_shift=15)

    assert instr.h_gelu_x_scale_shift == 15


def test_matmul_accepts_supported_rescale_shift():
    instr = MatMul("a", "b", "c", shift=63)

    assert instr.shift == 63


def test_matmul_rejects_unsupported_rescale_shift():
    with pytest.raises(ValueError, match="shift=64"):
        MatMul("a", "b", "c", shift=64)


def test_matmul_rejects_unsupported_h_gelu_shift():
    with pytest.raises(ValueError, match="h_gelu_x_scale_shift=16"):
        MatMul("a", "b", "c", h_gelu_x_scale_shift=16)


def test_pack_matmul_rejects_unsupported_rescale_shift():
    with pytest.raises(ValueError, match="shift=64"):
        pack_matmul(2, 0, 0, 0, 1, 1, 1, shift=64)


def test_pack_matmul_rejects_unsupported_h_gelu_shift():
    with pytest.raises(ValueError, match="h_gelu_x_scale_shift=16"):
        pack_matmul(2, 0, 0, 0, 1, 1, 1, h_gelu_x_scale_shift=16)


def test_program_matmul_rejects_unsupported_h_gelu_shift():
    program = TinyNPUProgram()
    program.declare_data("a", [[1]], role="A")
    program.declare_data("b", [[1]], role="B")

    with pytest.raises(ValueError, match="h_gelu_x_scale_shift=16"):
        program.matmul("a", "b", "c", activation=3, h_gelu_x_scale_shift=16)
