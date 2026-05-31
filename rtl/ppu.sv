`include "defines.sv"

module ppu (
    input logic clk,
    input logic rst_n,

    // Control from sequencer
    input logic                           capture_en,
    input logic                           bias_en,
    input logic                           bias_clear,
    input logic [$clog2(`ARRAY_SIZE)-1:0] ppu_cycle_idx,
    input logic [                    7:0] shift,
    input logic [                   15:0] multiplier,
    input logic [                    7:0] activation,
    input logic [                    7:0] h_gelu_x_scale_shift,
    input logic [                    1:0] precision,
    input logic [                    1:0] write_offset,
    input output_layout_t                 output_layout,
    input writeback_mode_t                writeback_mode,
    input logic [$clog2(`ARRAY_SIZE)-1:0] cache_lane_idx,

    // Data from Unified Buffer (for Bias Loading)
    input logic [`BUFFER_WIDTH-1:0] bias_in,

    // Data from Systolic Array (Bottom Row)
    input logic signed [`ACC_WIDTH-1:0] acc_in[`ARRAY_SIZE-1:0],

    // Completion handshake for the control unit. done pulses when the last
    // captured row has reached storage and writeback can start safely.
    output logic busy,
    output logic done,

    // Output to Unified Buffer (64-bit vector)
    output logic [`BUFFER_WIDTH-1:0] ub_wdata
);

  localparam int INT4_WORDS_PER_TILE = (`ARRAY_SIZE / 4);
  localparam int INT8_WORDS_PER_TILE = (`ARRAY_SIZE / 2);
  localparam int PPU_ACC_WIDTH = 48;
  localparam int PPU_PRODUCT_WIDTH = PPU_ACC_WIDTH + 16;
  localparam int PPU_SHIFT_WIDTH = $clog2(PPU_PRODUCT_WIDTH);
  localparam int PPU_ACT_WIDTH = 48;
  localparam int PPU_GELU_DIV_WIDTH = 32;
  // Hardware contract: H-GELU scale shifts are compiler-limited to 0..15.
  // Keeping this narrow avoids a full 8-bit dynamic barrel shifter in the
  // activation path without changing any generated model configuration.
  localparam int PPU_HGELU_SHIFT_WIDTH = 4;
  localparam int PPU_SIGMOID_SHIFT_WIDTH = 6;
  localparam logic signed [PPU_ACC_WIDTH-1:0] PPU_ACC_MAX = {1'b0, {(PPU_ACC_WIDTH - 1) {1'b1}}};
  localparam logic signed [PPU_ACC_WIDTH-1:0] PPU_ACC_MIN = {1'b1, {(PPU_ACC_WIDTH - 1) {1'b0}}};
  localparam logic [PPU_SHIFT_WIDTH-1:0] PPU_SHIFT_MAX = PPU_SHIFT_WIDTH'(PPU_PRODUCT_WIDTH - 1);
  localparam logic [PPU_HGELU_SHIFT_WIDTH-1:0] PPU_HGELU_SHIFT_MAX = {PPU_HGELU_SHIFT_WIDTH{1'b1}};

  // Internal storage for one full tile (quantized to 16-bit)
  logic [15:0] storage [`ARRAY_SIZE-1:0] [`ARRAY_SIZE-1:0];

  // Internal storage for bias vector (ARRAY_SIZE elements, 32-bit each)
  logic [31:0] bias_reg[`ARRAY_SIZE-1:0];

  // Internal state to track which bias word is being loaded (0 or 1)
  logic bias_word_toggle;

  logic valid_s0, valid_s1, valid_s2, valid_s3, valid_s4, valid_s5, valid_s6, valid_s7, valid_s8, valid_s9;
  logic last_s0, last_s1, last_s2, last_s3, last_s4, last_s5, last_s6, last_s7, last_s8, last_s9;
  logic [`ARRAY_SIZE-1:0] row_we_s0, row_we_s1, row_we_s2, row_we_s3, row_we_s4;
  logic [`ARRAY_SIZE-1:0] row_we_s5, row_we_s6, row_we_s7, row_we_s8, row_we_s9;
  logic [PPU_SHIFT_WIDTH-1:0] shift_s0, shift_s1, shift_s2, shift_s3, shift_s4, shift_s5, shift_s6, shift_s7;
  logic [PPU_SIGMOID_SHIFT_WIDTH-1:0] sigmoid_shift_s0, sigmoid_shift_s1, sigmoid_shift_s2, sigmoid_shift_s3;
  logic [PPU_SIGMOID_SHIFT_WIDTH-1:0] sigmoid_shift_s4, sigmoid_shift_s5, sigmoid_shift_s6, sigmoid_shift_s7;
  logic sigmoid_shift_valid_s0, sigmoid_shift_valid_s1, sigmoid_shift_valid_s2, sigmoid_shift_valid_s3;
  logic sigmoid_shift_valid_s4, sigmoid_shift_valid_s5, sigmoid_shift_valid_s6, sigmoid_shift_valid_s7;
  logic [7:0] activation_s0, activation_s1, activation_s2, activation_s3, activation_s4, activation_s5, activation_s6, activation_s7;
  logic [PPU_HGELU_SHIFT_WIDTH-1:0] h_gelu_x_scale_shift_s0, h_gelu_x_scale_shift_s1, h_gelu_x_scale_shift_s2;
  logic [PPU_HGELU_SHIFT_WIDTH-1:0] h_gelu_x_scale_shift_s3, h_gelu_x_scale_shift_s4, h_gelu_x_scale_shift_s5, h_gelu_x_scale_shift_s6, h_gelu_x_scale_shift_s7;
  logic [1:0] precision_s0, precision_s1, precision_s2, precision_s3, precision_s4, precision_s5, precision_s6, precision_s7, precision_s8;
  logic [15:0] multiplier_s0, multiplier_s1, multiplier_s2;

  logic signed [PPU_ACC_WIDTH-1:0] biased_s0[`ARRAY_SIZE-1:0];
  logic signed [PPU_PRODUCT_WIDTH-1:0] product_s1[`ARRAY_SIZE-1:0];
  logic signed [PPU_PRODUCT_WIDTH:0] rounded_s2[`ARRAY_SIZE-1:0];
  logic signed [PPU_PRODUCT_WIDTH-1:0] shifted_s3[`ARRAY_SIZE-1:0];
  logic signed [15:0] rescaled_s4[`ARRAY_SIZE-1:0];
  logic signed [15:0] act_passthrough_s5[`ARRAY_SIZE-1:0];
  logic signed [PPU_ACT_WIDTH-1:0] act_x_s5[`ARRAY_SIZE-1:0];
  logic signed [PPU_ACT_WIDTH-1:0] sigmoid_numer_s5[`ARRAY_SIZE-1:0];
  logic signed [PPU_ACT_WIDTH-1:0] gelu_gate_s5[`ARRAY_SIZE-1:0];
  logic signed [15:0] act_passthrough_s6[`ARRAY_SIZE-1:0];
  logic signed [PPU_ACT_WIDTH-1:0] act_x_s6[`ARRAY_SIZE-1:0];
  logic signed [PPU_ACT_WIDTH-1:0] sigmoid_rounded_s6[`ARRAY_SIZE-1:0];
  logic signed [PPU_ACT_WIDTH-1:0] gelu_prod_s6[`ARRAY_SIZE-1:0];
  logic signed [15:0] act_passthrough_s7[`ARRAY_SIZE-1:0];
  logic signed [PPU_ACT_WIDTH-1:0] sigmoid_rounded_s7[`ARRAY_SIZE-1:0];
  logic signed [PPU_GELU_DIV_WIDTH-1:0] gelu_div6_s7[`ARRAY_SIZE-1:0];
  logic signed [15:0] activated_s8[`ARRAY_SIZE-1:0];
  logic [15:0] quantized_s9[`ARRAY_SIZE-1:0];
  logic [15:0] quantized_w_s9[`ARRAY_SIZE-1:0];
  logic signed [PPU_PRODUCT_WIDTH:0] shifted_wide_s3[`ARRAY_SIZE-1:0];
  logic signed [PPU_ACT_WIDTH-1:0] sigmoid_qmax_s4;

  function automatic logic [PPU_SHIFT_WIDTH-1:0] ppu_effective_shift(input logic [7:0] raw_shift);
    begin
      if (raw_shift >= PPU_PRODUCT_WIDTH) begin
        ppu_effective_shift = PPU_SHIFT_MAX;
      end else begin
        ppu_effective_shift = raw_shift[PPU_SHIFT_WIDTH-1:0];
      end
    end
  endfunction

  function automatic logic [PPU_HGELU_SHIFT_WIDTH-1:0] ppu_effective_h_gelu_shift(input logic [7:0] raw_shift);
    begin
      if (raw_shift > PPU_HGELU_SHIFT_MAX) begin
        ppu_effective_h_gelu_shift = PPU_HGELU_SHIFT_MAX;
      end else begin
        ppu_effective_h_gelu_shift = raw_shift[PPU_HGELU_SHIFT_WIDTH-1:0];
      end
    end
  endfunction

  function automatic logic [PPU_SIGMOID_SHIFT_WIDTH-1:0] ppu_effective_sigmoid_shift(input logic [7:0] raw_shift);
    begin
      ppu_effective_sigmoid_shift = raw_shift[PPU_SIGMOID_SHIFT_WIDTH-1:0];
    end
  endfunction

  assign busy = capture_en | valid_s0 | valid_s1 | valid_s2 | valid_s3 | valid_s4 | valid_s5 | valid_s6 | valid_s7 | valid_s8 | valid_s9;

  always_comb begin
    unique case (precision_s4)
      MODE_INT4:  sigmoid_qmax_s4 = PPU_ACT_WIDTH'(7);
      MODE_INT8:  sigmoid_qmax_s4 = PPU_ACT_WIDTH'(127);
      default:    sigmoid_qmax_s4 = PPU_ACT_WIDTH'(32767);
    endcase
  end

  always_comb begin
    for (int i = 0; i < `ARRAY_SIZE; i++) begin
      logic signed [15:0] activated;
      logic signed [3:0]  sat4;
      logic signed [7:0]  sat8;

      activated = activated_s8[i];

      unique case (precision_s8)
        2'b00: begin  // INT4: [-8, 7]
          if (activated > 16'sd7) sat4 = 4'sd7;
          else if (activated < -16'sd8) sat4 = -4'sd8;
          else sat4 = activated[3:0];
          quantized_w_s9[i] = 16'(unsigned'(sat4));
        end
        2'b01: begin  // INT8: [-128, 127]
          if (activated > 16'sd127) sat8 = 8'sd127;
          else if (activated < -16'sd128) sat8 = -8'sd128;
          else sat8 = activated[7:0];
          quantized_w_s9[i] = 16'(unsigned'(sat8));
        end
        default: begin  // INT16: [-32768, 32767]
          quantized_w_s9[i] = activated;
        end
      endcase
    end
  end

  always_comb begin
    for (int i = 0; i < `ARRAY_SIZE; i++) begin
      shifted_wide_s3[i] = rounded_s2[i] >>> shift_s2;
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int r = 0; r < `ARRAY_SIZE; r++) begin
        for (int c = 0; c < `ARRAY_SIZE; c++) storage[r][c] <= '0;
      end
      for (int i = 0; i < `ARRAY_SIZE; i++) bias_reg[i] <= '0;
      bias_word_toggle <= 1'b0;
      valid_s0 <= 1'b0;
      valid_s1 <= 1'b0;
      valid_s2 <= 1'b0;
      valid_s3 <= 1'b0;
      valid_s4 <= 1'b0;
      valid_s5 <= 1'b0;
      valid_s6 <= 1'b0;
      valid_s7 <= 1'b0;
      valid_s8 <= 1'b0;
      valid_s9 <= 1'b0;
      last_s0 <= 1'b0;
      last_s1 <= 1'b0;
      last_s2 <= 1'b0;
      last_s3 <= 1'b0;
      last_s4 <= 1'b0;
      last_s5 <= 1'b0;
      last_s6 <= 1'b0;
      last_s7 <= 1'b0;
      last_s8 <= 1'b0;
      last_s9 <= 1'b0;
      row_we_s0 <= '0;
      row_we_s1 <= '0;
      row_we_s2 <= '0;
      row_we_s3 <= '0;
      row_we_s4 <= '0;
      row_we_s5 <= '0;
      row_we_s6 <= '0;
      row_we_s7 <= '0;
      row_we_s8 <= '0;
      row_we_s9 <= '0;
      shift_s0 <= '0;
      shift_s1 <= '0;
      shift_s2 <= '0;
      shift_s3 <= '0;
      shift_s4 <= '0;
      shift_s5 <= '0;
      shift_s6 <= '0;
      shift_s7 <= '0;
      sigmoid_shift_s0 <= '0;
      sigmoid_shift_s1 <= '0;
      sigmoid_shift_s2 <= '0;
      sigmoid_shift_s3 <= '0;
      sigmoid_shift_s4 <= '0;
      sigmoid_shift_s5 <= '0;
      sigmoid_shift_s6 <= '0;
      sigmoid_shift_s7 <= '0;
      sigmoid_shift_valid_s0 <= 1'b0;
      sigmoid_shift_valid_s1 <= 1'b0;
      sigmoid_shift_valid_s2 <= 1'b0;
      sigmoid_shift_valid_s3 <= 1'b0;
      sigmoid_shift_valid_s4 <= 1'b0;
      sigmoid_shift_valid_s5 <= 1'b0;
      sigmoid_shift_valid_s6 <= 1'b0;
      sigmoid_shift_valid_s7 <= 1'b0;
      activation_s0 <= '0;
      activation_s1 <= '0;
      activation_s2 <= '0;
      activation_s3 <= '0;
      activation_s4 <= '0;
      activation_s5 <= '0;
      activation_s6 <= '0;
      activation_s7 <= '0;
      h_gelu_x_scale_shift_s0 <= '0;
      h_gelu_x_scale_shift_s1 <= '0;
      h_gelu_x_scale_shift_s2 <= '0;
      h_gelu_x_scale_shift_s3 <= '0;
      h_gelu_x_scale_shift_s4 <= '0;
      h_gelu_x_scale_shift_s5 <= '0;
      h_gelu_x_scale_shift_s6 <= '0;
      h_gelu_x_scale_shift_s7 <= '0;
      precision_s0 <= MODE_INT16;
      precision_s1 <= MODE_INT16;
      precision_s2 <= MODE_INT16;
      precision_s3 <= MODE_INT16;
      precision_s4 <= MODE_INT16;
      precision_s5 <= MODE_INT16;
      precision_s6 <= MODE_INT16;
      precision_s7 <= MODE_INT16;
      precision_s8 <= MODE_INT16;
      multiplier_s0 <= '0;
      multiplier_s1 <= '0;
      multiplier_s2 <= '0;
      done <= 1'b0;
      for (int i = 0; i < `ARRAY_SIZE; i++) begin
        biased_s0[i] <= '0;
        product_s1[i] <= '0;
        rounded_s2[i] <= '0;
        shifted_s3[i] <= '0;
        rescaled_s4[i] <= '0;
        act_passthrough_s5[i] <= '0;
        act_x_s5[i] <= '0;
        sigmoid_numer_s5[i] <= '0;
        gelu_gate_s5[i] <= '0;
        act_passthrough_s6[i] <= '0;
        act_x_s6[i] <= '0;
        sigmoid_rounded_s6[i] <= '0;
        gelu_prod_s6[i] <= '0;
        act_passthrough_s7[i] <= '0;
        sigmoid_rounded_s7[i] <= '0;
        gelu_div6_s7[i] <= '0;
        activated_s8[i] <= '0;
        quantized_s9[i] <= '0;
      end
    end else begin
      done <= valid_s9 && last_s9;

      if (bias_clear) begin
        for (int i = 0; i < `ARRAY_SIZE; i++) bias_reg[i] <= '0;
        bias_word_toggle <= 1'b0;
      end else if (bias_en) begin
        if (bias_word_toggle == 1'b0) begin
          for (int i = 0; i < 4; i++) bias_reg[i] <= bias_in[i*32+:32];
          bias_word_toggle <= 1'b1;
        end else begin
          for (int i = 0; i < 4; i++) bias_reg[i+4] <= bias_in[i*32+:32];
          bias_word_toggle <= 1'b0;
        end
      end

      valid_s0 <= capture_en;
      valid_s1 <= valid_s0;
      valid_s2 <= valid_s1;
      valid_s3 <= valid_s2;
      valid_s4 <= valid_s3;
      valid_s5 <= valid_s4;
      valid_s6 <= valid_s5;
      valid_s7 <= valid_s6;
      valid_s8 <= valid_s7;
      valid_s9 <= valid_s8;
      last_s0 <= capture_en && (ppu_cycle_idx == (`ARRAY_SIZE - 1));
      last_s1 <= last_s0;
      last_s2 <= last_s1;
      last_s3 <= last_s2;
      last_s4 <= last_s3;
      last_s5 <= last_s4;
      last_s6 <= last_s5;
      last_s7 <= last_s6;
      last_s8 <= last_s7;
      last_s9 <= last_s8;

      if (capture_en) begin
        row_we_s0 <= `ARRAY_SIZE'(1) << ((`ARRAY_SIZE - 1) - ppu_cycle_idx);
        shift_s0 <= ppu_effective_shift(shift);
        sigmoid_shift_s0 <= ppu_effective_sigmoid_shift(h_gelu_x_scale_shift);
        // Sigmoid now reuses the hard-GELU clipped gate, so it follows the
        // same bounded activation-domain shift contract.
        sigmoid_shift_valid_s0 <= (h_gelu_x_scale_shift <= 8'd15);
        activation_s0 <= activation;
        h_gelu_x_scale_shift_s0 <= ppu_effective_h_gelu_shift(h_gelu_x_scale_shift);
        precision_s0 <= precision;
        multiplier_s0 <= multiplier;
        for (int i = 0; i < `ARRAY_SIZE; i++) begin
          logic signed [`ACC_WIDTH:0] biased_wide;
          biased_wide = $signed(acc_in[i]) + $signed({{33{bias_reg[i][31]}}, bias_reg[i]});
          if (biased_wide > $signed(PPU_ACC_MAX)) begin
            biased_s0[i] <= PPU_ACC_MAX;
          end else if (biased_wide < $signed(PPU_ACC_MIN)) begin
            biased_s0[i] <= PPU_ACC_MIN;
          end else begin
            biased_s0[i] <= biased_wide[PPU_ACC_WIDTH-1:0];
          end
        end
      end else begin
        row_we_s0 <= '0;
      end

      row_we_s1 <= row_we_s0;
      shift_s1 <= shift_s0;
      sigmoid_shift_s1 <= sigmoid_shift_s0;
      sigmoid_shift_valid_s1 <= sigmoid_shift_valid_s0;
      activation_s1 <= activation_s0;
      h_gelu_x_scale_shift_s1 <= h_gelu_x_scale_shift_s0;
      precision_s1 <= precision_s0;
      multiplier_s1 <= multiplier_s0;
      for (int i = 0; i < `ARRAY_SIZE; i++) begin
        product_s1[i] <= biased_s0[i] * $signed({1'b0, multiplier_s0});
      end

      row_we_s2 <= row_we_s1;
      shift_s2 <= shift_s1;
      sigmoid_shift_s2 <= sigmoid_shift_s1;
      sigmoid_shift_valid_s2 <= sigmoid_shift_valid_s1;
      activation_s2 <= activation_s1;
      h_gelu_x_scale_shift_s2 <= h_gelu_x_scale_shift_s1;
      precision_s2 <= precision_s1;
      multiplier_s2 <= multiplier_s1;
      for (int i = 0; i < `ARRAY_SIZE; i++) begin
        if (shift_s1 != '0) begin
          rounded_s2[i] <= product_s1[i] + $signed({1'b0, (PPU_PRODUCT_WIDTH'(1) << (shift_s1 - 1'b1))});
        end else begin
          rounded_s2[i] <= {product_s1[i][PPU_PRODUCT_WIDTH-1], product_s1[i]};
        end
      end

      row_we_s3 <= row_we_s2;
      shift_s3 <= shift_s2;
      sigmoid_shift_s3 <= sigmoid_shift_s2;
      sigmoid_shift_valid_s3 <= sigmoid_shift_valid_s2;
      activation_s3 <= activation_s2;
      h_gelu_x_scale_shift_s3 <= h_gelu_x_scale_shift_s2;
      precision_s3 <= precision_s2;
      for (int i = 0; i < `ARRAY_SIZE; i++) begin
        shifted_s3[i] <= shifted_wide_s3[i][PPU_PRODUCT_WIDTH-1:0];
      end

      row_we_s4 <= row_we_s3;
      shift_s4 <= shift_s3;
      sigmoid_shift_s4 <= sigmoid_shift_s3;
      sigmoid_shift_valid_s4 <= sigmoid_shift_valid_s3;
      activation_s4 <= activation_s3;
      h_gelu_x_scale_shift_s4 <= h_gelu_x_scale_shift_s3;
      precision_s4 <= precision_s3;
      for (int i = 0; i < `ARRAY_SIZE; i++) begin
        if (shifted_s3[i] > PPU_PRODUCT_WIDTH'(32767)) begin
          rescaled_s4[i] <= 16'sd32767;
        end else if (shifted_s3[i] < -PPU_PRODUCT_WIDTH'(32768)) begin
          rescaled_s4[i] <= -16'sd32768;
        end else begin
          rescaled_s4[i] <= shifted_s3[i][15:0];
        end
      end

      row_we_s5 <= row_we_s4;
      shift_s5 <= shift_s4;
      sigmoid_shift_s5 <= sigmoid_shift_s4;
      sigmoid_shift_valid_s5 <= sigmoid_shift_valid_s4;
      activation_s5 <= activation_s4;
      h_gelu_x_scale_shift_s5 <= h_gelu_x_scale_shift_s4;
      precision_s5 <= precision_s4;
      for (int i = 0; i < `ARRAY_SIZE; i++) begin
        logic signed [PPU_ACT_WIDTH-1:0] x_ext;
        logic signed [PPU_ACT_WIDTH-1:0] scale;
        logic signed [PPU_ACT_WIDTH-1:0] gate;

        x_ext = PPU_ACT_WIDTH'($signed(rescaled_s4[i]));
        scale = PPU_ACT_WIDTH'(1) <<< h_gelu_x_scale_shift_s4;
        gate = ((x_ext * PPU_ACT_WIDTH'(218) + PPU_ACT_WIDTH'(64)) >>> 7) + (PPU_ACT_WIDTH'(3) * scale);
        if (gate < 0) gate = 0;
        else if (gate > (PPU_ACT_WIDTH'(6) * scale)) gate = PPU_ACT_WIDTH'(6) * scale;

        act_passthrough_s5[i] <= rescaled_s4[i];
        act_x_s5[i] <= x_ext;
        gelu_gate_s5[i] <= gate;
        unique case (activation_s4)
          8'd1: sigmoid_numer_s5[i] <= '0;
          8'd2: begin
            if (!sigmoid_shift_valid_s4 || sigmoid_qmax_s4 <= 0) begin
              sigmoid_numer_s5[i] <= '0;
            end else begin
              sigmoid_numer_s5[i] <= sigmoid_qmax_s4 * gate;
            end
          end
          default: sigmoid_numer_s5[i] <= '0;
        endcase
      end

      row_we_s6 <= row_we_s5;
      shift_s6 <= shift_s5;
      sigmoid_shift_s6 <= sigmoid_shift_s5;
      sigmoid_shift_valid_s6 <= sigmoid_shift_valid_s5;
      activation_s6 <= activation_s5;
      h_gelu_x_scale_shift_s6 <= h_gelu_x_scale_shift_s5;
      precision_s6 <= precision_s5;
      for (int i = 0; i < `ARRAY_SIZE; i++) begin
        act_passthrough_s6[i] <= act_passthrough_s5[i];
        act_x_s6[i] <= act_x_s5[i];
        sigmoid_rounded_s6[i] <= sigmoid_shift_valid_s5 ? sigmoid_numer_s5[i] : '0;
        gelu_prod_s6[i] <= act_x_s5[i] * gelu_gate_s5[i];
      end

      row_we_s7 <= row_we_s6;
      shift_s7 <= shift_s6;
      sigmoid_shift_s7 <= sigmoid_shift_s6;
      sigmoid_shift_valid_s7 <= sigmoid_shift_valid_s6;
      activation_s7 <= activation_s6;
      h_gelu_x_scale_shift_s7 <= h_gelu_x_scale_shift_s6;
      precision_s7 <= precision_s6;
      for (int i = 0; i < `ARRAY_SIZE; i++) begin
        act_passthrough_s7[i] <= act_passthrough_s6[i];
        sigmoid_rounded_s7[i] <= (sigmoid_rounded_s6[i] * PPU_ACT_WIDTH'(10923) + (PPU_ACT_WIDTH'(1) <<< 15)) >>> 16;
        gelu_div6_s7[i] <= (gelu_prod_s6[i] * PPU_ACT_WIDTH'(10923) + (PPU_ACT_WIDTH'(1) <<< 15)) >>> 16;
      end

      row_we_s8 <= row_we_s7;
      precision_s8 <= precision_s7;
      for (int i = 0; i < `ARRAY_SIZE; i++) begin
        logic signed [PPU_GELU_DIV_WIDTH-1:0] shifted_gelu;

        if (h_gelu_x_scale_shift_s7 > 0) begin
          shifted_gelu = (gelu_div6_s7[i] >= 0)
              ? ((gelu_div6_s7[i] + (PPU_GELU_DIV_WIDTH'(1) <<< (h_gelu_x_scale_shift_s7 - 1))) >>> h_gelu_x_scale_shift_s7)
              : -(((-gelu_div6_s7[i]) + (PPU_GELU_DIV_WIDTH'(1) <<< (h_gelu_x_scale_shift_s7 - 1))) >>> h_gelu_x_scale_shift_s7);
        end else begin
          shifted_gelu = gelu_div6_s7[i];
        end

        unique case (activation_s7)
          8'd1: activated_s8[i] <= (act_passthrough_s7[i] < 0) ? '0 : act_passthrough_s7[i];
          8'd2: begin
            logic signed [PPU_ACT_WIDTH-1:0] sigmoid_out;
            if (!sigmoid_shift_valid_s7) begin
              sigmoid_out = '0;
            end else if (sigmoid_shift_s7 == '0) begin
              sigmoid_out = sigmoid_rounded_s7[i];
            end else begin
              sigmoid_out = (sigmoid_rounded_s7[i] + (PPU_ACT_WIDTH'(1) <<< (sigmoid_shift_s7 - 1'b1))) >>> sigmoid_shift_s7;
            end
            if (sigmoid_out > PPU_ACT_WIDTH'(32767)) activated_s8[i] <= 16'sd32767;
            else if (sigmoid_out < -PPU_ACT_WIDTH'(32768)) activated_s8[i] <= -16'sd32768;
            else activated_s8[i] <= sigmoid_out[15:0];
          end
          8'd3: begin
            if (shifted_gelu > PPU_GELU_DIV_WIDTH'(32767)) activated_s8[i] <= 16'sd32767;
            else if (shifted_gelu < -PPU_GELU_DIV_WIDTH'(32768)) activated_s8[i] <= -16'sd32768;
            else activated_s8[i] <= shifted_gelu[15:0];
          end
          default: activated_s8[i] <= act_passthrough_s7[i];
        endcase
      end

      row_we_s9 <= row_we_s8;
      for (int i = 0; i < `ARRAY_SIZE; i++) begin
        quantized_s9[i] <= quantized_w_s9[i];
      end

      if (valid_s9) begin
        for (int r = 0; r < `ARRAY_SIZE; r++) begin
          if (row_we_s9[r]) begin
            for (int i = 0; i < `ARRAY_SIZE; i++) begin
              storage[r][i] <= quantized_s9[i];
            end
          end
        end
      end
    end
  end

  always_comb begin
    ub_wdata = '0;
    if (writeback_mode == WB_MODE_V_CACHE_APPEND_INT16) begin
      for (int col = 0; col < `ARRAY_SIZE; col++) begin
        ub_wdata[col*16 +: 16] = storage[0][col];
      end
    end else if (writeback_mode == WB_MODE_K_CACHE_APPEND_INT16) begin
      ub_wdata[cache_lane_idx*16 +: 16] = storage[0][ppu_cycle_idx];
    end else if (output_layout == OUT_LAYOUT_A) begin
      unique case (precision)
        2'b00: begin
          if (ppu_cycle_idx < INT4_WORDS_PER_TILE) begin
            for (int row = 0; row < `ARRAY_SIZE; row++) begin
              logic [15:0] packed_word;
              packed_word = '0;
              for (int nib = 0; nib < 4; nib++) begin
                packed_word |= (storage[row][(ppu_cycle_idx * 4) + nib] & 16'h000F) << (nib * 4);
              end
              ub_wdata[row*16 +: 16] = packed_word;
            end
          end
        end
        2'b01: begin
          if (ppu_cycle_idx < INT8_WORDS_PER_TILE) begin
            for (int row = 0; row < `ARRAY_SIZE; row++) begin
              logic [15:0] packed_word;
              packed_word = '0;
              packed_word[7:0]  = storage[row][(ppu_cycle_idx * 2)][7:0];
              packed_word[15:8] = storage[row][(ppu_cycle_idx * 2) + 1][7:0];
              ub_wdata[row*16 +: 16] = packed_word;
            end
          end
        end
        default: begin
          for (int row = 0; row < `ARRAY_SIZE; row++) begin
            ub_wdata[row*16 +: 16] = storage[row][ppu_cycle_idx];
          end
        end
      endcase
    end else if (output_layout == OUT_LAYOUT_B) begin
      unique case (precision)
        2'b00: begin
          if (ppu_cycle_idx < INT4_WORDS_PER_TILE) begin
            for (int col = 0; col < `ARRAY_SIZE; col++) begin
              logic [15:0] packed_word;
              packed_word = '0;
              for (int nib = 0; nib < 4; nib++) begin
                packed_word |= (storage[(ppu_cycle_idx * 4) + nib][col] & 16'h000F) << (nib * 4);
              end
              ub_wdata[col*16 +: 16] = packed_word;
            end
          end
        end
        2'b01: begin
          if (ppu_cycle_idx < INT8_WORDS_PER_TILE) begin
            for (int col = 0; col < `ARRAY_SIZE; col++) begin
              logic [15:0] packed_word;
              packed_word = '0;
              packed_word[7:0]  = storage[(ppu_cycle_idx * 2)][col][7:0];
              packed_word[15:8] = storage[(ppu_cycle_idx * 2) + 1][col][7:0];
              ub_wdata[col*16 +: 16] = packed_word;
            end
          end
        end
        default: begin
          for (int col = 0; col < `ARRAY_SIZE; col++) begin
            ub_wdata[col*16 +: 16] = storage[ppu_cycle_idx][col];
          end
        end
      endcase
    end else begin
      unique case (precision)
        2'b00: begin
          for (int col = 0; col < `ARRAY_SIZE; col++) begin
            ub_wdata[col*16 +: 16] = (storage[ppu_cycle_idx][col] & 16'h000F) << (write_offset * 4);
          end
        end
        2'b01: begin
          for (int col = 0; col < `ARRAY_SIZE; col++) begin
            ub_wdata[col*16 +: 16] = (storage[ppu_cycle_idx][col] & 16'h00FF) << (write_offset * 8);
          end
        end
        default: begin
          for (int col = 0; col < `ARRAY_SIZE; col++) begin
            ub_wdata[col*16 +: 16] = storage[ppu_cycle_idx][col];
          end
        end
      endcase
    end
  end

endmodule
