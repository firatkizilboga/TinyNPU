module di_exp #(
    parameter int INPUT_WIDTH = 16,
    parameter int OUTPUT_WIDTH = 16
) (
    input  logic signed [ INPUT_WIDTH-1:0] x_in,
    input  logic        [             15:0] m_i,
    input  logic        [              7:0] k_i,
    output logic        [OUTPUT_WIDTH-1:0] y_out
);

  localparam int unsigned Y_MAX = (1 << OUTPUT_WIDTH) - 1;

  always_comb begin
    int unsigned m_f;
    int unsigned t_mag;
    int unsigned period_numer;
    int signed t_val;
    int signed q_i;
    int signed r_i;
    int signed unshifted_exp;
    int signed result_i;
    int signed x_ext;

    m_f = '0;
    t_mag = '0;
    period_numer = '0;
    t_val = '0;
    q_i = '0;
    r_i = '0;
    unshifted_exp = '0;
    result_i = '0;
    x_ext = '0;

    m_f = int'(m_i) + (int'(m_i) >> 1) - (int'(m_i) >> 4);
    period_numer = (k_i < 31) ? (32'd1 << k_i) : 0;

    // I-LLM Algorithm 1, step 1: derive the negative period t from the
    // runtime integer scale pair (m_i, k_i).
    t_mag = (m_f > 0 && period_numer > 0) ? ((period_numer + (m_f >> 1)) / m_f) : 0;

    if (m_f == 0 || t_mag == 0) begin
      y_out = '0;
    end else begin
      x_ext = int'($signed(x_in));
      t_val = -$signed(t_mag);

      // I-LLM Algorithm 1, steps 2-3: x = q_i * t + r_i.
      q_i = x_ext / t_val;
      r_i = x_ext - (q_i * t_val);

      // I-LLM Algorithm 1, step 4: affine approximation over the remainder.
      unshifted_exp = (r_i >>> 1) - t_val;

      // I-LLM Algorithm 1, step 5: apply the power-of-two decay.
      result_i = unshifted_exp >>> q_i;

      if (result_i <= 0) begin
        y_out = '0;
      end else if (result_i > $signed(Y_MAX)) begin
        y_out = OUTPUT_WIDTH'(Y_MAX);
      end else begin
        y_out = OUTPUT_WIDTH'(result_i);
      end
    end
  end

endmodule
