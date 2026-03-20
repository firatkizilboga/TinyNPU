import cocotb
from cocotb.triggers import Timer

from tinynpu_jit.golden import h_gelu


X_SCALE_SHIFT = 7
SLOPE_NUM = 218
SLOPE_SHIFT = 7


@cocotb.test()
async def test_h_gelu_matches_golden(dut):
    dut.x_scale_shift.value = X_SCALE_SHIFT
    dut.slope_num.value = SLOPE_NUM
    dut.slope_shift.value = SLOPE_SHIFT

    for x_in in range(-512, 513):
        dut.x_in.value = x_in
        await Timer(1, units="ns")
        got = int(dut.y_out.value.signed_integer)
        want = h_gelu(x_in, x_scale_shift=X_SCALE_SHIFT, slope_num=SLOPE_NUM, slope_shift=SLOPE_SHIFT)
        assert got == want, f"x_in={x_in}: got {got}, want {want}"


@cocotb.test()
async def test_h_gelu_has_expected_regions(dut):
    dut.x_scale_shift.value = X_SCALE_SHIFT
    dut.slope_num.value = SLOPE_NUM
    dut.slope_shift.value = SLOPE_SHIFT

    expectations = {
        -256: 0,
        -192: h_gelu(-192, x_scale_shift=X_SCALE_SHIFT, slope_num=SLOPE_NUM, slope_shift=SLOPE_SHIFT),
        -64: h_gelu(-64, x_scale_shift=X_SCALE_SHIFT, slope_num=SLOPE_NUM, slope_shift=SLOPE_SHIFT),
        0: 0,
        256: 256,
    }
    for x_in, want in expectations.items():
        dut.x_in.value = x_in
        await Timer(1, units="ns")
        got = int(dut.y_out.value.signed_integer)
        assert got == want, f"x_in={x_in}: got {got}, want {want}"
