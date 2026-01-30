`include "defines.sv"

// ============================================================================
// Streaming Skewer Module
// ============================================================================
// Converts streaming matrix columns/rows into skewed timing for Systolic Array.
// 
// Marker signals (global, not per-row):
// - first_in:  Pulse when feeding row 0 (first data entering)
// - first_out: Pulse when row 0 data exits (computation starts at PE[0][0])
// - last_in:   Pulse when feeding row N-1 (last data entering)
// - last_out:  Pulse when row N-1 data exits (all data loaded at PE[N-1][N-1])
//
// Behavior:
// - Row 0: 1 cycle delay
// - Row i: i+1 cycles delay

module streaming_skewer #(
    parameter N = `ARRAY_SIZE,
    parameter DATA_WIDTH = `DATA_WIDTH
) (
    input  logic clk,
    input  logic rst_n,
    input  logic en,
    
    // Data interface
    input  logic [DATA_WIDTH-1:0] data_in  [N-1:0],
    output logic [DATA_WIDTH-1:0] data_out [N-1:0],
    
    // Marker signals (single pulses, not arrays)
    input  logic first_in,          // Pulse: feeding first row (row 0)
    input  logic last_in,           // Pulse: feeding last row (row N-1)
    output logic first_out,         // Pulse: row 0 data exits
    output logic last_out,          // Pulse: row N-1 data exits
    
    // Flattened outputs for Verilator verification
    output logic [N*DATA_WIDTH-1:0] data_out_flat
);

    // ========================================================================
    // Row 0: 1 stage delay
    // ========================================================================
    logic [DATA_WIDTH-1:0] d0_s0;
    logic f0;  // first marker for row 0
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            d0_s0 <= '0;
            f0    <= 1'b0;
        end else if (en) begin
            d0_s0 <= data_in[0];
            f0    <= first_in;
        end
    end
    
    assign data_out[0] = d0_s0;
    assign first_out   = f0;  // Row 0 exits → computation starts

    // ========================================================================
    // Row 1: 2 stage delay  
    // ========================================================================
    logic [DATA_WIDTH-1:0] d1_s0, d1_s1;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            d1_s0 <= '0;
            d1_s1 <= '0;
        end else if (en) begin
            d1_s0 <= data_in[1];
            d1_s1 <= d1_s0;
        end
    end
    
    assign data_out[1] = d1_s1;

    // ========================================================================
    // Row 2: 3 stage delay
    // ========================================================================
    logic [DATA_WIDTH-1:0] d2_s0, d2_s1, d2_s2;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            d2_s0 <= '0;
            d2_s1 <= '0;
            d2_s2 <= '0;
        end else if (en) begin
            d2_s0 <= data_in[2];
            d2_s1 <= d2_s0;
            d2_s2 <= d2_s1;
        end
    end
    
    assign data_out[2] = d2_s2;

    // ========================================================================
    // Row 3: 4 stage delay + last marker
    // ========================================================================
    logic [DATA_WIDTH-1:0] d3_s0, d3_s1, d3_s2, d3_s3;
    logic [3:0] l3;  // last marker chain for row 3
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            d3_s0 <= '0;
            d3_s1 <= '0;
            d3_s2 <= '0;
            d3_s3 <= '0;
            l3    <= '0;
        end else if (en) begin
            d3_s0 <= data_in[3];
            d3_s1 <= d3_s0;
            d3_s2 <= d3_s1;
            d3_s3 <= d3_s2;
            l3    <= {l3[2:0], last_in};
        end
    end
    
    assign data_out[3] = d3_s3;
    assign last_out    = l3[3];  // Row 3 exits → all data loaded

    // ========================================================================
    // Flattened outputs for Verilator
    // ========================================================================
    assign data_out_flat = {d3_s3, d2_s2, d1_s1, d0_s0};

endmodule