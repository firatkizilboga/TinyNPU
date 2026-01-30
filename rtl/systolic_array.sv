`include "defines.sv"

module systolic_array (
    input  logic clk,
    input  logic rst_n,
    
    // Input data interface (Matrix A, flows horizontally, 16-bit)
    input  logic [`DATA_WIDTH-1:0] input_data [`ARRAY_SIZE-1:0],
    
    // Weight data interface (Matrix B, flows vertically, 16-bit from UB)
    // Will be zero-extended to 64-bit internally for vertical bus
    // During drain mode, pass zeros from outside
    input  logic [`DATA_WIDTH-1:0] weight_data [`ARRAY_SIZE-1:0],
    
    // Control signals
    input  precision_mode_t precision_mode,  // Bit-width mode for all PEs
    input  logic compute_enable,             // Enable computation in all PEs
    input  logic drain_enable,               // Enable drain mode (shift accumulators)
    input  logic acc_clear,                  // Clear all accumulators
    
    // Output interface - Read accumulator results
    output logic signed [`ACC_WIDTH-1:0] results [`ARRAY_SIZE-1:0][`ARRAY_SIZE-1:0],
    // Flattened output for Verilator/Verification access
    output logic [(`ARRAY_SIZE * `ARRAY_SIZE * `ACC_WIDTH)-1:0] results_flat,
    output logic result_valid
);

    // ------------------------------------------------------------------------
    // Flatten Output for Verification
    // ------------------------------------------------------------------------
    // Pack 2D array [Row][Col] into 1D vector
    // Layout: Row0_Col0, Row0_Col1, ..., Row1_Col0, ...
    generate
        genvar fr, fc;
        for (fr = 0; fr < `ARRAY_SIZE; fr++) begin : gen_flat_rows
            for (fc = 0; fc < `ARRAY_SIZE; fc++) begin : gen_flat_cols
                assign results_flat[((fr * `ARRAY_SIZE + fc) * `ACC_WIDTH) +: `ACC_WIDTH] = results[fr][fc];
            end
        end
    endgenerate

    // ------------------------------------------------------------------------
    // PE Interconnection Wires
    // ------------------------------------------------------------------------
    // Horizontal data flow (inputs): [row][col] to [row][col+1] - 16-bit
    // Extra column (+1) for boundary inputs
    logic [`DATA_WIDTH-1:0] horizontal_bus [`ARRAY_SIZE-1:0][`ARRAY_SIZE:0];
    
    // Vertical data flow (weights/accumulators): [row][col] to [row+1][col] - 64-bit
    // Extra row (+1) for boundary inputs
    logic signed [`ACC_WIDTH-1:0] vertical_bus [`ARRAY_SIZE:0][`ARRAY_SIZE-1:0];
    
    // Accumulator outputs from each PE
    logic signed [`ACC_WIDTH-1:0] pe_accumulators [`ARRAY_SIZE-1:0][`ARRAY_SIZE-1:0];
    
    // ------------------------------------------------------------------------
    // Boundary Input Assignment (Leftmost Column and Topmost Row)
    // ------------------------------------------------------------------------
    // External inputs connect to the left edge and top edge
    genvar r, c;
    generate
        // Connect input data to leftmost column (col=0 boundary)
        for (r = 0; r < `ARRAY_SIZE; r++) begin : input_boundary
            assign horizontal_bus[r][0] = input_data[r];
        end
        
        // Connect weight data to topmost row (row=0 boundary)
        // Zero-extend 16-bit weight data to 64-bit vertical bus
        for (c = 0; c < `ARRAY_SIZE; c++) begin : weight_boundary
            assign vertical_bus[0][c] = {{(`ACC_WIDTH-`DATA_WIDTH){1'b0}}, weight_data[c]};
        end
    endgenerate
    
    // ------------------------------------------------------------------------
    // PE Array Instantiation (N×N Grid)
    // ------------------------------------------------------------------------
    generate
        for (r = 0; r < `ARRAY_SIZE; r++) begin : gen_rows
            for (c = 0; c < `ARRAY_SIZE; c++) begin : gen_cols
                pe pe_inst (
                    .clk               (clk),
                    .rst_n             (rst_n),
                    
                    // Horizontal dataflow (16-bit inputs)
                    .input_from_left   (horizontal_bus[r][c]),
                    .input_to_right    (horizontal_bus[r][c+1]),
                    
                    // Vertical dataflow (64-bit weights/accumulators)
                    .data_from_top     (vertical_bus[r][c]),
                    .data_to_bottom    (vertical_bus[r+1][c]),
                    
                    // Control (broadcast to all PEs)
                    .precision_mode    (precision_mode),
                    .compute_enable    (compute_enable),
                    .drain_enable      (drain_enable),
                    .acc_clear         (acc_clear),
                    
                    // Accumulator output (stationary)
                    .acc_out           (pe_accumulators[r][c])
                );
            end
        end
    endgenerate
    
    // ------------------------------------------------------------------------
    // Output Assignment
    // ------------------------------------------------------------------------
    generate
        for (r = 0; r < `ARRAY_SIZE; r++) begin : output_rows
            for (c = 0; c < `ARRAY_SIZE; c++) begin : output_cols
                assign results[r][c] = pe_accumulators[r][c];
            end
        end
    endgenerate
    
    // ------------------------------------------------------------------------
    // Result Valid Generation
    // ------------------------------------------------------------------------
    // Pipeline delay chain to track when results are valid
    // Depth = 2*ARRAY_SIZE for skewed injection + propagation
    localparam VALID_DELAY = 2 * `ARRAY_SIZE;
    logic [VALID_DELAY-1:0] valid_pipeline;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_pipeline <= '0;
        end else begin
            // Shift in the compute_enable signal
            valid_pipeline <= {valid_pipeline[VALID_DELAY-2:0], compute_enable};
        end
    end
    
    assign result_valid = valid_pipeline[VALID_DELAY-1];

endmodule
