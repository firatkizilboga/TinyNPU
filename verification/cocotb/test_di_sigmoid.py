import cocotb
from cocotb.triggers import Timer

from tinynpu_jit.golden import di_sigmoid


M_I = 128
K_I = 12
P_OUT = 8
ALPHA_SMOOTH = 1


@cocotb.test()
async def test_di_sigmoid_matches_golden(dut):
    dut.m_i.value = M_I
    dut.k_i.value = K_I
    dut.p_out.value = P_OUT
    dut.alpha_smooth.value = ALPHA_SMOOTH

    for x_in in range(-128, 129):
        dut.x_in.value = x_in
        await Timer(1, units="ns")
        got = int(dut.y_out.value)
        want = di_sigmoid(x_in, m_i=M_I, k_i=K_I, p_out=P_OUT, alpha_smooth=ALPHA_SMOOTH)
        assert got == want, f"x_in={x_in}: got {got}, want {want}"


@cocotb.test()
async def test_di_sigmoid_is_monotone(dut):
    dut.m_i.value = M_I
    dut.k_i.value = K_I
    dut.p_out.value = P_OUT
    dut.alpha_smooth.value = ALPHA_SMOOTH

    prev = None
    for x_in in range(-128, 129):
        dut.x_in.value = x_in
        await Timer(1, units="ns")
        got = int(dut.y_out.value)
        if prev is not None:
            assert prev <= got, f"non-monotone around x_in={x_in}: prev={prev}, got={got}"
        prev = got


@cocotb.test()
async def test_di_sigmoid_supports_large_shift_values(dut):
    dut.m_i.value = 26845
    dut.k_i.value = 35
    dut.p_out.value = 8
    dut.alpha_smooth.value = 1

    for x_in in (-4, -1, 0, 1, 4):
        dut.x_in.value = x_in
        await Timer(1, units="ns")
        got = int(dut.y_out.value)
        want = di_sigmoid(x_in, m_i=26845, k_i=35, p_out=8, alpha_smooth=1)
        assert got == want, f"x_in={x_in}: got {got}, want {want}"
