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
