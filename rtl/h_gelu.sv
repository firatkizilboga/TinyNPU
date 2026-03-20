module h_gelu #(
    parameter int INPUT_WIDTH = 16,
    parameter int OUTPUT_WIDTH = 16
) (
    input  logic signed [ INPUT_WIDTH-1:0] x_in,
    input  logic        [              7:0] x_scale_shift,
    input  logic        [             15:0] slope_num,
    input  logic        [              7:0] slope_shift,
    output logic signed [OUTPUT_WIDTH-1:0] y_out
);

  function automatic longint signed round_shift_signed(input longint signed value, input int shamt);
    longint signed rounder;
    begin
      if (shamt <= 0) begin
        round_shift_signed = value;
      end else begin
        rounder = 64'sd1 <<< (shamt - 1);
        if (value >= 0) begin
          round_shift_signed = (value + rounder) >>> shamt;
        end else begin
          round_shift_signed = -(((-value) + rounder) >>> shamt);
        end
      end
    end
  endfunction

  function automatic longint signed round_div_signed(input longint signed numer, input longint signed denom);
    longint signed half;
    begin
      if (denom <= 0) begin
        round_div_signed = 0;
      end else begin
        half = denom >>> 1;
        if (numer >= 0) begin
          round_div_signed = (numer + half) / denom;
        end else begin
          round_div_signed = -(((-numer) + half) / denom);
        end
      end
    end
  endfunction

  always_comb begin
    longint signed scale_denom;
    longint signed three_int;
    longint signed six_int;
    longint signed slope_term;
    longint signed gate_int;
    longint signed result_i;
    longint signed max_out;
    longint signed min_out;

    scale_denom = 0;
    three_int = 0;
    six_int = 0;
    slope_term = 0;
    gate_int = 0;
    result_i = 0;
    max_out = (64'sd1 <<< (OUTPUT_WIDTH - 1)) - 1;
    min_out = -(64'sd1 <<< (OUTPUT_WIDTH - 1));

    if (slope_num == 0 || x_scale_shift >= 31 || slope_shift >= 31) begin
      y_out = '0;
    end else begin
      // Integer hard-GELU:
      // y = x * ReLU6(1.702x + 3) / 6
      // with x represented in the domain x_real ~= x_in / 2^x_scale_shift.
      scale_denom = 64'sd1 <<< x_scale_shift;
      three_int = 64'sd3 * scale_denom;
      six_int = 64'sd6 * scale_denom;

      slope_term = round_shift_signed($signed(x_in) * $signed({1'b0, slope_num}), int'(slope_shift));
      gate_int = slope_term + three_int;
      if (gate_int < 0) begin
        gate_int = 0;
      end else if (gate_int > six_int) begin
        gate_int = six_int;
      end

      result_i = round_div_signed($signed(x_in) * gate_int, six_int);

      if (result_i > max_out) begin
        y_out = OUTPUT_WIDTH'(max_out);
      end else if (result_i < min_out) begin
        y_out = OUTPUT_WIDTH'(min_out);
      end else begin
        y_out = OUTPUT_WIDTH'(result_i);
      end
    end
  end

endmodule
