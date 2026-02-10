`include "defines.sv"

module ppu (
    input  logic clk,
    input  logic rst_n,

    // Control from sequencer
    input  logic                           capture_en,
    input  logic                           bias_en,
    input  logic                           bias_clear,
    input  logic [$clog2(`ARRAY_SIZE)-1:0] cycle_idx,

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

    // Capture Logic
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

            // 2. Handle Data Capture (Addition happens HERE in high precision)
            if (capture_en) begin
                for (int i=0; i<`ARRAY_SIZE; i++) begin
                    // Add 16-bit bias to 64-bit accumulator, then truncate
                    // This is "Before Quantization" addition.
                    logic signed [`ACC_WIDTH-1:0] biased_acc;
                    biased_acc = acc_in[i] + $signed(bias_reg[i]);
                    storage[cycle_idx][i] <= biased_acc[15:0];
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
