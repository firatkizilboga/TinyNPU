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
    
    // Data flow markers (from skewer - indicates first/last valid data)
    input  logic input_first,     // First valid input row arrived
    input  logic input_last,      // Last valid input row arrived
    input  logic weight_first,    // First valid weight column arrived
    input  logic weight_last,     // Last valid weight column arrived
    
    // Control signals
    input  precision_mode_t precision_mode,  // Bit-width mode for all PEs
    input  logic compute_enable,             // Enable computation in all PEs
    input  logic drain_enable,               // Enable drain mode (shift accumulators)
    input  logic acc_clear,                  // Clear all accumulators
    
    // Output interface - Read accumulator results
    output logic signed [`ACC_WIDTH-1:0] results [`ARRAY_SIZE-1:0][`ARRAY_SIZE-1:0],
    // Flattened output for Verilator/Verification access
    output logic [(`ARRAY_SIZE * `ARRAY_SIZE * `ACC_WIDTH)-1:0] results_flat,
    output logic result_valid,
    
    // Marker outputs (expose for external control/debug)
    output logic computation_started,  // Pulses when first valid data enters array
    output logic computation_done,     // Pulses when last valid data enters array (row 3 col 0)
    output logic all_done              // Pulses when last valid data exits PE[3][3] (all computation complete)
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
    
    // Horizontal last marker flow: [row][col] to [row][col+1]
    // Extra column (+1) for boundary inputs and outputs
    logic last_marker_bus [`ARRAY_SIZE-1:0][`ARRAY_SIZE:0];
    
    // Vertical data flow (weights/accumulators): [row][col] to [row+1][col] - 64-bit
    // Extra row (+1) for boundary inputs
    logic signed [`ACC_WIDTH-1:0] vertical_bus [`ARRAY_SIZE:0][`ARRAY_SIZE-1:0];
    
    // Accumulator outputs from each PE
    logic signed [`ACC_WIDTH-1:0] pe_accumulators [`ARRAY_SIZE-1:0][`ARRAY_SIZE-1:0];
    
    // Valid mesh for systolic wavefront propagation
    // Horizontal: [row][col] to [row][col+1], extra column for boundary
    // Vertical: [row][col] to [row+1][col], extra row for boundary
    logic valid_h_bus [`ARRAY_SIZE-1:0][`ARRAY_SIZE:0];
    logic valid_v_bus [`ARRAY_SIZE:0][`ARRAY_SIZE-1:0];
    
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
        
        // Connect last marker to leftmost column (col=0 boundary)
        // Only row 3 (last row) gets the input_last marker
        for (r = 0; r < `ARRAY_SIZE; r++) begin : last_boundary
            assign last_marker_bus[r][0] = (r == `ARRAY_SIZE-1) ? input_last : 1'b0;
        end
        
        // Connect weight data to topmost row (row=0 boundary)
        // Zero-extend 16-bit weight data to 64-bit vertical bus
        for (c = 0; c < `ARRAY_SIZE; c++) begin : weight_boundary
            assign vertical_bus[0][c] = {{(`ACC_WIDTH-`DATA_WIDTH){1'b0}}, weight_data[c]};
        end
        
        // Connect valid signals at boundaries
        // All rows get valid_h from compute_enable at left edge
        // All cols get valid_v from compute_enable at top edge
        for (r = 0; r < `ARRAY_SIZE; r++) begin : valid_h_boundary
            assign valid_h_bus[r][0] = compute_enable;
        end
        for (c = 0; c < `ARRAY_SIZE; c++) begin : valid_v_boundary
            assign valid_v_bus[0][c] = compute_enable;
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
                    
                    // Horizontal last marker
                    .last_in           (last_marker_bus[r][c]),
                    .last_out          (last_marker_bus[r][c+1]),
                    
                    // Valid mesh (wavefront propagation)
                    .valid_h_in        (valid_h_bus[r][c]),
                    .valid_h_out       (valid_h_bus[r][c+1]),
                    .valid_v_in        (valid_v_bus[r][c]),
                    .valid_v_out       (valid_v_bus[r+1][c]),
                    
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
    
    // ------------------------------------------------------------------------
    // Computation Start/Done Markers
    // ------------------------------------------------------------------------
    // computation_started: fires when BOTH input and weight first markers arrive
    // computation_done: fires when BOTH input and weight last markers arrive (enters array)
    // all_done: fires when last marker exits PE[3][3] (all computation complete)
    
    always_ff @(posedge clk) begin : debugBlock
        if (compute_enable) begin
            $display("=== [Systolic Array @ %0t] compute_enable=1 ===", $time);
            $display("  Input (horizontal, enters left edge):");
            $display("    Row0=%04h  Row1=%04h  Row2=%04h  Row3=%04h", 
                     input_data[0], input_data[1], input_data[2], input_data[3]);
            $display("  Weight (vertical, enters top edge):");
            $display("    Col0=%04h  Col1=%04h  Col2=%04h  Col3=%04h", 
                     weight_data[0], weight_data[1], weight_data[2], weight_data[3]);
            $display("  Markers: input_first=%b input_last=%b weight_first=%b weight_last=%b",
                     input_first, input_last, weight_first, weight_last);
        end
    end 

    assign computation_started = input_first & weight_first;
    assign computation_done    = input_last & weight_last;
    assign all_done            = last_marker_bus[`ARRAY_SIZE-1][`ARRAY_SIZE];

endmodule
