`include "defines.sv"

module ppu (
    input  logic clk,
    input  logic rst_n,

    // Control from sequencer
    input  logic                           capture_en,
    input  logic                           bias_en,
    input  logic                           bias_clear,
    input  logic [$clog2(`ARRAY_SIZE)-1:0] cycle_idx,
    input  logic [ 7:0]                    shift,
    input  logic [15:0]                    multiplier,
    input  logic [ 7:0]                    activation,
    input  logic [ 1:0]                    precision,
    input  logic [ 1:0]                    write_offset,

    // Data from Unified Buffer (for Bias Loading)
    input  logic [`BUFFER_WIDTH-1:0]       bias_in,

    // Data from Systolic Array (Bottom Row)
    input  logic signed [`ACC_WIDTH-1:0]   acc_in [`ARRAY_SIZE-1:0],

    // Output to Unified Buffer (64-bit vector)
    output logic [`BUFFER_WIDTH-1:0]       ub_wdata
);

    // Internal storage for one full tile (quantized to 16-bit)
    logic [15:0] storage [`ARRAY_SIZE-1:0][`ARRAY_SIZE-1:0];
    
    // Internal storage for bias vector (ARRAY_SIZE elements)
    logic [15:0] bias_reg [`ARRAY_SIZE-1:0];

    // Capture and Quantization Logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int r=0; r<`ARRAY_SIZE; r++) begin
                for (int c=0; c<`ARRAY_SIZE; c++) storage[r][c] <= '0;
            end
            for (int i=0; i<`ARRAY_SIZE; i++) bias_reg[i] <= '0;
        end else begin
            // 1. Handle Bias Loading/Clearing
            if (bias_clear) begin
                for (int i=0; i<`ARRAY_SIZE; i++) bias_reg[i] <= '0;
            end else if (bias_en) begin
                for (int i=0; i<`ARRAY_SIZE; i++) begin
                    bias_reg[i] <= bias_in[i*16 +: 16];
                end
            end

            // 2. Handle Data Capture with Quantization
            if (capture_en) begin
                for (int i=0; i<`ARRAY_SIZE; i++) begin
                    // Local variables for calculation
                    logic signed [`ACC_WIDTH:0] biased_acc;
                    logic signed [80:0] rescaled;
                    logic signed [80:0] shifted;
                    logic [15:0] result_val;
                    logic signed [3:0]  sat4;
                    logic signed [7:0]  sat8;
                    logic signed [15:0] sat16;

                    // A. High-Precision Bias Addition
                    biased_acc = acc_in[i] + $signed(bias_reg[i]);
                    
                    // B. Rescale (Multiplier)
                    rescaled = biased_acc * $signed({1'b0, multiplier});
                    
                    // C. Shift Right
                    shifted = rescaled >>> shift;
                    
                    // D. Precision Saturation and Alignment
                    unique case (precision)
                        2'b00: begin // INT4
                            if (shifted > 81'sd7)  sat4 = 4'sd7;
                            else if (shifted < -81'sd8) sat4 = -4'sd8;
                            else sat4 = shifted[3:0];
                            result_val = (16'(unsigned'(sat4))) << (write_offset * 4);
                        end
                        2'b01: begin // INT8
                            if (shifted > 81'sd127)  sat8 = 8'sd127;
                            else if (shifted < -81'sd128) sat8 = -8'sd128;
                            else sat8 = shifted[7:0];
                            result_val = (16'(unsigned'(sat8))) << (write_offset * 8);
                        end
                        default: begin // INT16
                            if (shifted > 81'sd32767)  sat16 = 16'sh7FFF;
                            else if (shifted < -81'sd32768) sat16 = 16'sh8000;
                            else sat16 = shifted[15:0];
                            result_val = unsigned'(sat16);
                        end
                    endcase
                    
                    storage[cycle_idx][i] <= result_val;
                end
            end
        end
    end

    // Output Selection: Present quantized rows
    generate
        for (genvar i = 0; i < `ARRAY_SIZE; i++) begin : gen_output
            assign ub_wdata[i*16 +: 16] = storage[cycle_idx][i];
        end
    endgenerate

endmodule
