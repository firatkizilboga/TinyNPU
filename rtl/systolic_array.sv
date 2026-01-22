`include "defines.sv"

// ============================================================================
// Systolic Array Module
// ============================================================================
// 2D mesh of Processing Elements (PEs) for matrix multiplication
// Weight-stationary dataflow: C = A * B
// - Weights (B) are pre-loaded and stay in PEs
// - Inputs (A) flow horizontally through rows
// - Partial sums accumulate within each PE
// - Results (C) read from PE accumulators after computation

module systolic_array (
    input  logic clk,
    input  logic rst_n,
    
    // Weight loading interface
    input  logic weight_load_enable,
    input  logic signed [`DATA_WIDTH-1:0] weight_in [`ARRAY_HEIGHT-1:0],
    
    // Input data interface
    input  logic data_valid,
    input  logic signed [`DATA_WIDTH-1:0] data_in [`ARRAY_HEIGHT-1:0],
    
    // Control signals
    input  logic acc_enable,
    input  logic acc_clear,
    
    // Output interface
    output logic signed [`ACC_WIDTH-1:0] results [`ARRAY_HEIGHT-1:0][`ARRAY_WIDTH-1:0],
    output logic result_valid
);

    // ------------------------------------------------------------------------
    // Internal Wiring - PE Interconnections
    // ------------------------------------------------------------------------
    // Data flows horizontally (left to right)
    logic signed [`DATA_WIDTH-1:0] data_horizontal [`ARRAY_HEIGHT-1:0][`ARRAY_WIDTH:0];
    
    // Weights flow vertically (top to bottom) during loading
    logic signed [`DATA_WIDTH-1:0] weight_vertical [`ARRAY_HEIGHT:0][`ARRAY_WIDTH-1:0];
    
    // Accumulator outputs from each PE
    logic signed [`ACC_WIDTH-1:0] pe_acc [`ARRAY_HEIGHT-1:0][`ARRAY_WIDTH-1:0];
    
    // ------------------------------------------------------------------------
    // Input Assignment - Connect external inputs to first column/row
    // ------------------------------------------------------------------------
    genvar row, col;
    generate
        // Connect data inputs to leftmost column
        for (row = 0; row < `ARRAY_HEIGHT; row++) begin : data_input_conn
            assign data_horizontal[row][0] = data_in[row];
        end
        
        // Connect weight inputs to topmost row
        for (col = 0; col < `ARRAY_WIDTH; col++) begin : weight_input_conn
            assign weight_vertical[0][col] = weight_in[col];
        end
    endgenerate
    
    // ------------------------------------------------------------------------
    // PE Array Instantiation
    // ------------------------------------------------------------------------
    generate
        for (row = 0; row < `ARRAY_HEIGHT; row++) begin : pe_rows
            for (col = 0; col < `ARRAY_WIDTH; col++) begin : pe_cols
                pe pe_inst (
                    .clk           (clk),
                    .rst_n         (rst_n),
                    
                    // Horizontal data flow (input activations)
                    .data_in       (data_horizontal[row][col]),
                    .data_out      (data_horizontal[row][col+1]),
                    
                    // Vertical weight flow (during loading)
                    .weight_in     (weight_vertical[row][col]),
                    .weight_out    (weight_vertical[row+1][col]),
                    
                    // Control signals (broadcast to all PEs)
                    .weight_load   (weight_load_enable),
                    .acc_enable    (acc_enable),
                    .acc_clear     (acc_clear),
                    
                    // Accumulator output
                    .acc_out       (pe_acc[row][col])
                );
            end
        end
    endgenerate
    
    // ------------------------------------------------------------------------
    // Output Assignment
    // ------------------------------------------------------------------------
    generate
        for (row = 0; row < `ARRAY_HEIGHT; row++) begin : output_row
            for (col = 0; col < `ARRAY_WIDTH; col++) begin : output_col
                assign results[row][col] = pe_acc[row][col];
            end
        end
    endgenerate
    
    // ------------------------------------------------------------------------
    // Result Valid Signal Generation
    // ------------------------------------------------------------------------
    // Delay chain to account for pipeline depth
    // Valid signal propagates with the data through the array
    logic [2*`ARRAY_WIDTH-1:0] valid_delay;
    
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            valid_delay <= '0;
        end else begin
            valid_delay <= {valid_delay[2*`ARRAY_WIDTH-2:0], data_valid};
        end
    end
    
    assign result_valid = valid_delay[2*`ARRAY_WIDTH-1];

endmodule
