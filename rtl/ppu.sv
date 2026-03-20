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
  logic signed [81:0] pre_activation_row[`ARRAY_SIZE-1:0];
  logic signed [15:0] sigmoid_in_row[`ARRAY_SIZE-1:0];
  logic [15:0] sigmoid_out_row[`ARRAY_SIZE-1:0];
  logic [7:0] sigmoid_p_out;

  always_comb begin
    unique case (precision)
      2'b00: sigmoid_p_out = 8'd4;
      2'b01: sigmoid_p_out = 8'd8;
      default: sigmoid_p_out = 8'd16;
    endcase

    for (int i = 0; i < `ARRAY_SIZE; i++) begin
      logic signed [64:0] biased_acc;
      logic signed [81:0] rescaled; // 65-bit * 17-bit = 82 bits
      logic signed [82:0] rounded;  // 83 bits for carry
      logic signed [81:0] shifted;

      biased_acc = $signed(acc_in[i]) + $signed({{33{bias_reg[i][31]}}, bias_reg[i]});
      rescaled = biased_acc * $signed({1'b0, multiplier});

      if (shift > 0) begin
        rounded = rescaled + $signed({1'b0, (82'd1 << (shift - 1))});
        shifted = 82'(signed'(rounded >>> shift));
      end else begin
        shifted = rescaled;
      end

      pre_activation_row[i] = shifted;

      if (shifted > 32767) begin
        sigmoid_in_row[i] = 16'sd32767;
      end else if (shifted < -32768) begin
        sigmoid_in_row[i] = -16'sd32768;
      end else begin
        sigmoid_in_row[i] = shifted[15:0];
      end
    end
  end

  generate
    for (genvar i = 0; i < `ARRAY_SIZE; i++) begin : gen_sigmoid
      di_sigmoid #(
          .INPUT_WIDTH(16),
          .OUTPUT_WIDTH(16)
      ) u_di_sigmoid (
          .x_in(sigmoid_in_row[i]),
          .m_i(multiplier),
          .k_i(shift),
          .p_out(sigmoid_p_out),
          .alpha_smooth(8'd1),
          .y_out(sigmoid_out_row[i])
      );
    end
  endgenerate

  always_comb begin
    for (int i = 0; i < `ARRAY_SIZE; i++) begin
      logic signed [81:0] activated;
      logic signed [3:0]  sat4;
      logic signed [7:0]  sat8;
      logic signed [15:0] sat16;
      logic [15:0]        result_val;

      activated = pre_activation_row[i];

      // Activation stage: 0 = none, 1 = ReLU, 2 = sigmoid.
      if (activation == 8'd1) begin
        if (activated < 0) activated = 0;
      end else if (activation == 8'd2) begin
        activated = 82'(signed'({1'b0, sigmoid_out_row[i]}));
      end

      // Precision saturation and packed writeback.
      unique case (precision)
        2'b00: begin  // INT4: [-8, 7]
          if (activated > 7)       sat4 = 7;
          else if (activated < -8) sat4 = -8;
          else                     sat4 = activated[3:0];
          result_val = (16'(unsigned'(sat4))) << (write_offset * 4);
        end
        2'b01: begin  // INT8: [-128, 127]
          if (activated > 127)       sat8 = 127;
          else if (activated < -128) sat8 = -128;
          else                       sat8 = activated[7:0];
          result_val = (16'(unsigned'(sat8))) << (write_offset * 8);
        end
        default: begin  // INT16: [-32768, 32767]
          if (activated > 32767)       sat16 = 32767;
          else if (activated < -32768) sat16 = -32768;
          else                         sat16 = activated[15:0];
          result_val = 16'(unsigned'(sat16));
        end
      endcase
      
      quantized_row[i] = result_val;
    end
  end

  always_ff @(posedge clk) begin
    if (rst_n && capture_en) begin
`ifdef PPU_DEBUG_CAPTURE
        $display("PPU CAPTURE: cycle=%d | B0=%d B1=%d B2=%d B3=%d B4=%d B5=%d B6=%d B7=%d | Lane0: acc=%d, res=%x", 
            ppu_cycle_idx, bias_reg[0], bias_reg[1], bias_reg[2], bias_reg[3],
            bias_reg[4], bias_reg[5], bias_reg[6], bias_reg[7],
            acc_in[0], quantized_row[0]);
        $fflush();
`endif
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

      if (capture_en) begin
        // Drain outputs row N-1 first (cycle 0), row 0 last (cycle N-1).
        // Reverse the index so storage[0] = row 0, storage[N-1] = row N-1.
        for (int i = 0; i < `ARRAY_SIZE; i++) begin
          storage[(`ARRAY_SIZE-1) - ppu_cycle_idx][i] <= quantized_row[i];
        end
      end
    end
  end

  generate
    for (genvar i = 0; i < `ARRAY_SIZE; i++) begin : gen_output
      assign ub_wdata[i*16+:16] = storage[ppu_cycle_idx][i];
    end
  endgenerate

endmodule
