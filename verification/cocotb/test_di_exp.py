import cocotb
from cocotb.triggers import Timer

from tinynpu_jit.golden import di_exp


PARAMETER_SETS = [(128, 12), (92, 10), (64, 8)]


@cocotb.test()
async def test_di_exp_matches_golden_over_negative_range(dut):
    for m_i, k_i in PARAMETER_SETS:
        dut.m_i.value = m_i
        dut.k_i.value = k_i
        for x_in in range(-128, 1):
            dut.x_in.value = x_in
            await Timer(1, units="ns")
            got = int(dut.y_out.value)
            want = di_exp(x_in, m_i=m_i, k_i=k_i)
            assert got == want, f"m_i={m_i} k_i={k_i} x_in={x_in}: got {got}, want {want}"


@cocotb.test()
async def test_di_exp_matches_selected_edge_cases(dut):
    vectors = [-128, -64, -45, -44, -23, -22, -1, 0]
    for m_i, k_i in PARAMETER_SETS:
        dut.m_i.value = m_i
        dut.k_i.value = k_i
        for x_in in vectors:
            dut.x_in.value = x_in
            await Timer(1, units="ns")
            got = int(dut.y_out.value)
            want = di_exp(x_in, m_i=m_i, k_i=k_i)
            assert got == want, f"m_i={m_i} k_i={k_i} x_in={x_in}: got {got}, want {want}"


@cocotb.test()
async def test_di_exp_output_is_monotone(dut):
    for m_i, k_i in PARAMETER_SETS:
        dut.m_i.value = m_i
        dut.k_i.value = k_i
        prev = None
        for x_in in range(-128, 1):
            dut.x_in.value = x_in
            await Timer(1, units="ns")
            got = int(dut.y_out.value)
            if prev is not None:
                assert prev <= got, f"m_i={m_i} k_i={k_i} non-monotone around x_in={x_in}: prev={prev}, got={got}"
            prev = got
