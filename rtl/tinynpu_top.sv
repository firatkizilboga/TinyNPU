`include "defines.sv"

// ============================================================================
// TinyNPU Top-Level Module
// ============================================================================
// Structural integration: Unified Buffer → Streaming Skewer → Systolic Array
// First/last markers flow through the pipeline for state orchestration.

module tinynpu_top (
    input logic clk,
    input logic rst_n,

    // ========================================================================
    // Unified Buffer Write Interface (for loading data)
    // ========================================================================
    input  logic                      ub_wr_en,
    input  logic [`ADDR_WIDTH-1:0]    ub_wr_addr,
    input  logic [`BUFFER_WIDTH-1:0]  ub_wr_data,

    // ========================================================================
    // Unified Buffer Read Control (driven by external control unit)
    // ========================================================================
    input  logic                      input_first,      // Mark first input row
    input  logic                      input_last,       // Mark last input row
    input  logic [`ADDR_WIDTH-1:0]    input_addr,
    input  logic                      weight_first,     // Mark first weight row
    input  logic                      weight_last,      // Mark last weight row
    input  logic [`ADDR_WIDTH-1:0]    weight_addr,

    // ========================================================================
    // Skewer Control
    // ========================================================================
    input  logic                      skewer_en,

    // ========================================================================
    // Systolic Array Control
    // ========================================================================
    input  precision_mode_t           precision_mode,
    input  logic                      compute_enable,
    input  logic                      drain_enable,
    input  logic                      acc_clear,

    // ========================================================================
    // Result Outputs
    // ========================================================================
    output logic signed [`ACC_WIDTH-1:0] results [`ARRAY_SIZE-1:0][`ARRAY_SIZE-1:0],
    output logic [(`ARRAY_SIZE * `ARRAY_SIZE * `ACC_WIDTH)-1:0] results_flat,
    output logic                      result_valid,

    // ========================================================================
    // Marker Outputs (for external orchestration)
    // ========================================================================
    output logic                      input_first_out,   // First input hit row 0 PE
    output logic                      input_last_out,    // Last input hit row 3 PE
    output logic                      weight_first_out,  // First weight hit row 0 PE
    output logic                      weight_last_out    // Last weight hit row 3 PE
);

    // ========================================================================
    // Internal Wires
    // ========================================================================
    
    // Buffer outputs
    logic [`BUFFER_WIDTH-1:0] buffer_input_data;
    logic [`BUFFER_WIDTH-1:0] buffer_weight_data;
    logic                     buffer_input_first;
    logic                     buffer_input_last;
    logic                     buffer_weight_first;
    logic                     buffer_weight_last;
    
    // Unpacked vectors for skewer
    logic [`DATA_WIDTH-1:0] input_vector  [`ARRAY_SIZE-1:0];
    logic [`DATA_WIDTH-1:0] weight_vector [`ARRAY_SIZE-1:0];
    
    // Skewer outputs to systolic array
    logic [`DATA_WIDTH-1:0] skewed_input  [`ARRAY_SIZE-1:0];
    logic [`DATA_WIDTH-1:0] skewed_weight [`ARRAY_SIZE-1:0];

    // ========================================================================
    // Unpack: Buffer (64-bit) → Vector (4 × 16-bit)
    // ========================================================================
    generate
        genvar i;
        for (i = 0; i < `ARRAY_SIZE; i++) begin : gen_unpack
            assign input_vector[i]  = buffer_input_data[i*`DATA_WIDTH +: `DATA_WIDTH];
            assign weight_vector[i] = buffer_weight_data[i*`DATA_WIDTH +: `DATA_WIDTH];
        end
    endgenerate

    // ========================================================================
    // Unified Buffer
    // ========================================================================
    unified_buffer u_buffer (
        .clk              (clk),
        .rst_n            (rst_n),
        .wr_en            (ub_wr_en),
        .wr_addr          (ub_wr_addr),
        .wr_data          (ub_wr_data),
        .input_first_in   (input_first),
        .input_last_in    (input_last),
        .input_addr       (input_addr),
        .input_first_out  (buffer_input_first),
        .input_last_out   (buffer_input_last),
        .input_data       (buffer_input_data),
        .weight_first_in  (weight_first),
        .weight_last_in   (weight_last),
        .weight_addr      (weight_addr),
        .weight_first_out (buffer_weight_first),
        .weight_last_out  (buffer_weight_last),
        .weight_data      (buffer_weight_data)
    );

    // ========================================================================
    // Streaming Skewer for Inputs (Matrix A)
    // ========================================================================
    streaming_skewer #(
        .N(`ARRAY_SIZE),
        .DATA_WIDTH(`DATA_WIDTH)
    ) u_input_skewer (
        .clk           (clk),
        .rst_n         (rst_n),
        .en            (skewer_en),
        .data_in       (input_vector),
        .data_out      (skewed_input),
        .first_in      (buffer_input_first),
        .last_in       (buffer_input_last),
        .first_out     (input_first_out),
        .last_out      (input_last_out),
        .data_out_flat ()  // Unused at top level
    );

    // ========================================================================
    // Streaming Skewer for Weights (Matrix B)
    // ========================================================================
    streaming_skewer #(
        .N(`ARRAY_SIZE),
        .DATA_WIDTH(`DATA_WIDTH)
    ) u_weight_skewer (
        .clk           (clk),
        .rst_n         (rst_n),
        .en            (skewer_en),
        .data_in       (weight_vector),
        .data_out      (skewed_weight),
        .first_in      (buffer_weight_first),
        .last_in       (buffer_weight_last),
        .first_out     (weight_first_out),
        .last_out      (weight_last_out),
        .data_out_flat ()  // Unused at top level
    );

    // ========================================================================
    // Systolic Array
    // ========================================================================
    systolic_array u_systolic (
        .clk            (clk),
        .rst_n          (rst_n),
        .input_data     (skewed_input),
        .weight_data    (skewed_weight),
        .precision_mode (precision_mode),
        .compute_enable (compute_enable),
        .drain_enable   (drain_enable),
        .acc_clear      (acc_clear),
        .results        (results),
        .results_flat   (results_flat),
        .result_valid   (result_valid)
    );

endmodule
