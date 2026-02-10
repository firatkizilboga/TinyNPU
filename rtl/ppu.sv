`include "defines.sv"

module ppu (
    input  logic clk,
    input  logic rst_n,

    // Control
    input  logic       capture_en,
    // Automatically calculate required width for the counter
    input  logic [$clog2(`ARRAY_SIZE)-1:0] cycle_idx,

    // Data from Systolic Array (Bottom row vertical bus)
    input  logic signed [`ACC_WIDTH-1:0] acc_in [`ARRAY_SIZE-1:0],
    
    // Output to Unified Buffer
    output logic [`BUFFER_WIDTH-1:0] ub_wdata
);

    // Parameterized Tile Buffer: [Column][Row]
    logic [15:0] tile_buffer [`ARRAY_SIZE-1:0][`ARRAY_SIZE-1:0];

    // ------------------------------------------------------------------------
    // Capture Logic (Row-Major In)
    // ------------------------------------------------------------------------
    integer c;
    always_ff @(posedge clk) begin
        if (capture_en) begin
            // We receive rows in reverse order: Row (N-1) down to Row 0.
            // acc_in[c] holds the value for Column 'c' of the current Row.
            for (c = 0; c < `ARRAY_SIZE; c++) begin
                tile_buffer[c][`ARRAY_SIZE - 1 - cycle_idx] <= acc_in[c][15:0];
            end
        end
    end

    // ------------------------------------------------------------------------
    // Writeback Logic (Column-Major Out)
    // ------------------------------------------------------------------------
    // Use generate to construct the wide output vector from the selected column.
    // We map rows 0..N-1 to the output vector segments.
    
    generate
        genvar r;
        for (r = 0; r < `ARRAY_SIZE; r++) begin : gen_out
             // Element 0 (Row 0) goes to LSBs, Element N-1 (Row N-1) goes to MSBs
             assign ub_wdata[(r*16) +: 16] = tile_buffer[cycle_idx][r];
        end
    endgenerate

endmodule