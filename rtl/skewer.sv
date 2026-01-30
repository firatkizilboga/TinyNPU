`include "defines.sv"

// ============================================================================
// Streaming Skewer Module
// ============================================================================
// Converts streaming matrix columns/rows into skewed timing for Systolic Array.
//
// Behavior:
// - Row 0: 1 cycle delay (base pipeline stage)
// - Row 1: 2 cycles delay
// - Row i: i+1 cycles delay
//
// This allows the user to feed a full vector (column/row) every cycle, 
// while the hardware automatically manages the staggered timing required 
// by the output-stationary array.

module streaming_skewer #(
    parameter N = `ARRAY_SIZE,
    parameter DATA_WIDTH = `DATA_WIDTH
) (
    input  logic clk,
    input  logic rst_n,
    input  logic en,
    
    input  logic [DATA_WIDTH-1:0] data_in [N-1:0],
    output logic [DATA_WIDTH-1:0] data_out [N-1:0]
);

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : gen_skew_rows
            
            // Chain of 'i+1' Flip-Flops
            logic [DATA_WIDTH-1:0] chain [i+1:0]; 
            
            assign chain[0] = data_in[i];
            
            for (genvar d = 0; d < i + 1; d++) begin : gen_depth
                always_ff @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        chain[d+1] <= '0;
                    end else if (en) begin
                        chain[d+1] <= chain[d];
                    end
                end
            end
            
            assign data_out[i] = chain[i+1];
        end
    endgenerate

endmodule