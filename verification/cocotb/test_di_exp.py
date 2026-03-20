import cocotb
from cocotb.triggers import Timer

from tinynpu_jit.golden import di_exp_fixed


M_I = 128
K_I = 12


@cocotb.test()
async def test_di_exp_matches_golden_over_negative_range(dut):
    for x_in in range(-128, 1):
        dut.x_in.value = x_in
        await Timer(1, units="ns")
        got = int(dut.y_out.value)
        want = di_exp_fixed(x_in, m_i=M_I, k_i=K_I)
        assert got == want, f"x_in={x_in}: got {got}, want {want}"


@cocotb.test()
async def test_di_exp_matches_selected_edge_cases(dut):
    vectors = [-128, -64, -45, -44, -23, -22, -1, 0]
    for x_in in vectors:
        dut.x_in.value = x_in
        await Timer(1, units="ns")
        got = int(dut.y_out.value)
        want = di_exp_fixed(x_in, m_i=M_I, k_i=K_I)
        assert got == want, f"x_in={x_in}: got {got}, want {want}"


@cocotb.test()
async def test_di_exp_output_is_monotone(dut):
    prev = None
    for x_in in range(-128, 1):
        dut.x_in.value = x_in
        await Timer(1, units="ns")
        got = int(dut.y_out.value)
        if prev is not None:
            assert prev <= got, f"non-monotone around x_in={x_in}: prev={prev}, got={got}"
        prev = got
