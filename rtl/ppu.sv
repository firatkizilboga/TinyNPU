`include "defines.sv"

module ppu (
    input  logic clk,
    input  logic rst_n,

    // Control from sequencer
    input  logic                           capture_en,
    input  logic [$clog2(`ARRAY_SIZE)-1:0] cycle_idx,

    // Data from Systolic Array (Bottom Row)
    input  logic signed [`ACC_WIDTH-1:0]   acc_in [`ARRAY_SIZE-1:0],

    // Output to Unified Buffer (64-bit vector)
    output logic [`BUFFER_WIDTH-1:0]       ub_wdata
);

    // Internal storage for one full 4x4 tile (quantized to 16-bit)
    logic [15:0] storage [`ARRAY_SIZE-1:0][`ARRAY_SIZE-1:0];

    // Capture Logic: Store bottom row into storage[cycle_idx]
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (int r=0; r<`ARRAY_SIZE; r++) begin
                for (int c=0; c<`ARRAY_SIZE; c++) begin
                    storage[r][c] <= '0;
                end
            end
        end else if (capture_en) begin
            for (int i=0; i<`ARRAY_SIZE; i++) begin
                // Simple truncation quantization
                storage[cycle_idx][i] <= acc_in[i][15:0];
            end
        end
    end

    // Output Selection: Present storage[cycle_idx] to UB
    // This assumes the CU sets cycle_idx during WRITEBACK state as well.
    assign ub_wdata = {storage[cycle_idx][3], 
                       storage[cycle_idx][2], 
                       storage[cycle_idx][1], 
                       storage[cycle_idx][0]};

endmodule