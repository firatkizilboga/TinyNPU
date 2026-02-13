`include "defines.sv"

module ppu (
    input logic clk,
    input logic rst_n,

    // Control from sequencer
    input logic                           capture_en,
    input logic                           bias_en,
    input logic                           bias_clear,
    input logic [$clog2(`ARRAY_SIZE)-1:0] cycle_idx,
    input logic [                    7:0] shift,
    input logic [                   15:0] multiplier,
    input logic [                    7:0] activation,
    input logic [                    1:0] precision,
    input logic [                    1:0] write_offset,

    // Data from Unified Buffer (for Bias Loading)
    input logic [`BUFFER_WIDTH-1:0] bias_in,

    // Data from Systolic Array (Bottom Row)
    input logic signed [`ACC_WIDTH-1:0] acc_in[`ARRAY_SIZE-1:0],

    // Output to Unified Buffer (64-bit vector)
    output logic [`BUFFER_WIDTH-1:0] ub_wdata
);

  // Internal storage for one full tile (quantized to 16-bit)
  logic [15:0] storage [`ARRAY_SIZE-1:0] [`ARRAY_SIZE-1:0];

  // Internal storage for bias vector (ARRAY_SIZE elements, 32-bit each)
  logic [31:0] bias_reg[`ARRAY_SIZE-1:0];

  // Internal state to track which bias word is being loaded (0 or 1)
  logic bias_word_toggle;

  // Logic for quantization pipeline
  logic [15:0] quantized_row[`ARRAY_SIZE-1:0];

  always_comb begin
    for (int i = 0; i < `ARRAY_SIZE; i++) begin
      logic signed [64:0] biased_acc;
      logic signed [80:0] rescaled;
      logic signed [81:0] rounded;
      logic signed [80:0] shifted;
      logic signed [3:0]  sat4;
      logic signed [7:0]  sat8;
      logic signed [15:0] sat16;
      logic [15:0]        result_val;

      // A. High-Precision Bias Addition (32-bit Bias)
      biased_acc = $signed(acc_in[i]) + $signed({{33{bias_reg[i][31]}}, bias_reg[i]});

      // B. Rescale (Multiplier)
      rescaled = biased_acc * $signed({1'b0, multiplier});

      // C. Rounding and Shift Right
      if (shift > 0) begin
        rounded = rescaled + $signed({1'b0, (81'd1 << (shift - 1))});
        shifted = rounded >>> shift;
      end else begin
        shifted = rescaled;
      end

      // D. Activation (ReLU)
      if (activation[0]) begin
        if (shifted < 0) shifted = 0;
      end

      // E. Precision Saturation and Alignment
      unique case (precision)
        2'b00: begin  // INT4
          if (shifted > 7)       sat4 = 7;
          else if (shifted < -8) sat4 = -8;
          else                   sat4 = shifted[3:0];
          result_val = (16'(unsigned'(sat4))) << (write_offset * 4);
        end
        2'b01: begin  // INT8
          if (shifted > 127)       sat8 = 127;
          else if (shifted < -128) sat8 = -128;
          else                     sat8 = shifted[7:0];
          result_val = (16'(unsigned'(sat8))) << (write_offset * 8);
        end
        default: begin  // INT16
          if (shifted > 32767)       sat16 = 32767;
          else if (shifted < -32768) sat16 = -32768;
          else                       sat16 = shifted[15:0];
          result_val = unsigned'(sat16);
        end
      endcase
      
      quantized_row[i] = result_val;
    end
  end

  // Capture and Quantization Logic
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int r = 0; r < `ARRAY_SIZE; r++) begin
        for (int c = 0; c < `ARRAY_SIZE; c++) storage[r][c] <= '0;
      end
      for (int i = 0; i < `ARRAY_SIZE; i++) bias_reg[i] <= '0;
      bias_word_toggle <= 1'b0;
    end else begin
      // 1. Handle Bias Loading
      if (bias_clear) begin
        for (int i = 0; i < `ARRAY_SIZE; i++) bias_reg[i] <= '0;
        bias_word_toggle <= 1'b0;
      end else if (bias_en) begin
        if (bias_word_toggle == 1'b0) begin
          // Load Word 0: Cols 0-3
          for (int i = 0; i < 4; i++) bias_reg[i] <= bias_in[i*32+:32];
          bias_word_toggle <= 1'b1;
        end else begin
          // Load Word 1: Cols 4-7
          for (int i = 0; i < 4; i++) bias_reg[i+4] <= bias_in[i*32+:32];
          bias_word_toggle <= 1'b0;
        end
      end

      // 2. Handle Data Capture
      if (capture_en) begin
        for (int i = 0; i < `ARRAY_SIZE; i++) begin
          storage[cycle_idx][i] <= quantized_row[i];
        end
      end
    end
  end

  // Output Selection
  generate
    for (genvar i = 0; i < `ARRAY_SIZE; i++) begin : gen_output
      assign ub_wdata[i*16+:16] = storage[cycle_idx][i];
    end
  endgenerate

endmodule
