`include "defines.sv"

// ============================================================================
// Streaming Skewer Module (Fully Parametric)
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
    // Parametric delay pipeline using generate
    // ========================================================================
    // Row i gets (i+1) stages of delay
    
    genvar row;
    generate
        for (row = 0; row < N; row++) begin : delay_row
            localparam STAGES = row + 1;
            
            // Shift register for data
            logic [DATA_WIDTH-1:0] shift_reg [STAGES-1:0];
            
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    for (int s = 0; s < STAGES; s++) begin
                        shift_reg[s] <= '0;
                    end
                end else if (en) begin
                    shift_reg[0] <= data_in[row];
                    for (int s = 1; s < STAGES; s++) begin
                        shift_reg[s] <= shift_reg[s-1];
                    end
                end
            end
            
            // Output is the last stage
            assign data_out[row] = shift_reg[STAGES-1];
        end
    endgenerate

    // ========================================================================
    // Marker signal pipelines
    // ========================================================================
    // Markers arrive from UB already delayed by 1 cycle (same as data).
    // They need to go through the SAME delay as the corresponding row.
    // first_out: aligned with row 0 output (1 stage delay)
    // last_out: aligned with row N-1 output (N stages delay)
    
    // First marker: goes through row 0's delay (1 stage)
    logic first_marker;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            first_marker <= 1'b0;
        end else if (en) begin
            first_marker <= first_in;
        end
    end
    
    assign first_out = first_marker;

    // Last marker: goes through row N-1's delay (N stages)
    logic [N-1:0] last_marker_chain;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            last_marker_chain <= '0;
        end else if (en) begin
            last_marker_chain <= {last_marker_chain[N-2:0], last_in};
        end
    end
    
    assign last_out = last_marker_chain[N-1];

    // ========================================================================
    // Flattened outputs for Verilator
    // ========================================================================
    genvar i;
    generate
        for (i = 0; i < N; i++) begin : flatten_out
            assign data_out_flat[(i+1)*DATA_WIDTH-1 -: DATA_WIDTH] = data_out[i];
        end
    endgenerate

endmodule