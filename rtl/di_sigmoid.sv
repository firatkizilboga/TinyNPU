module di_sigmoid #(
    parameter int INPUT_WIDTH = 16,
    parameter int OUTPUT_WIDTH = 8
) (
    input  logic signed [ INPUT_WIDTH-1:0] x_in,
    input  logic        [             15:0] m_i,
    input  logic        [              7:0] k_i,
    input  logic        [              7:0] p_out,
    input  logic        [              7:0] alpha_smooth,
    output logic        [OUTPUT_WIDTH-1:0] y_out
);

  logic signed [INPUT_WIDTH-1:0] x_smoothed;
  logic signed [INPUT_WIDTH-1:0] exp_arg;
  logic        [31:0] exp_zero;
  logic        [31:0] exp_term;
  logic               use_exp_term_numer;

  di_exp #(
      .INPUT_WIDTH(INPUT_WIDTH),
      .OUTPUT_WIDTH(32)
  ) u_di_exp_zero (
      .x_in('0),
      .m_i(m_i),
      .k_i(k_i),
      .y_out(exp_zero)
  );

  di_exp #(
      .INPUT_WIDTH(INPUT_WIDTH),
      .OUTPUT_WIDTH(32)
  ) u_di_exp_term (
      .x_in(exp_arg),
      .m_i(m_i),
      .k_i(k_i),
      .y_out(exp_term)
  );

  always_comb begin
    x_smoothed = '0;
    exp_arg = '0;
    use_exp_term_numer = 1'b0;

    if (alpha_smooth != 0 && p_out != 0 && p_out < 31) begin
      x_smoothed = $signed(x_in) / $signed({{(INPUT_WIDTH-8){1'b0}}, alpha_smooth});
      if (x_smoothed >= 0) begin
        exp_arg = -x_smoothed;
      end else begin
        exp_arg = x_smoothed;
        use_exp_term_numer = 1'b1;
      end
    end
  end

  always_comb begin
    int unsigned numer;
    int unsigned denom;
    int unsigned out_scale;
    int unsigned result_i;

    numer = '0;
    denom = '0;
    out_scale = '0;
    result_i = '0;

    if (alpha_smooth == 0 || p_out == 0 || p_out >= 31) begin
      y_out = '0;
    end else begin
      // Scalar DI-Sigmoid built from DI-Exp:
      // sigma(x) = 1 / (1 + exp(-x)) for x >= 0,
      // sigma(x) = exp(x) / (1 + exp(x)) for x < 0.
      numer = use_exp_term_numer ? int'(exp_term) : int'(exp_zero);
      denom = int'(exp_zero) + int'(exp_term);
      out_scale = (1 << (p_out - 1)) - 1;
      result_i = (denom > 0) ? ((numer * out_scale + (denom >> 1)) / denom) : 0;

      if (result_i > ((1 << OUTPUT_WIDTH) - 1)) begin
        y_out = OUTPUT_WIDTH'((1 << OUTPUT_WIDTH) - 1);
      end else begin
        y_out = OUTPUT_WIDTH'(result_i);
      end
    end
  end

endmodule
