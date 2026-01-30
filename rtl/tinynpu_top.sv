`include "defines.sv"

// ============================================================================
// TinyNPU Top-Level Module
// ============================================================================
// Structural integration: Unified Buffer → Streaming Skewer → Systolic Array
// Control unit is external — this module just wires the datapath.

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
    // Unified Buffer Read Addresses (driven by external control unit)
    // ========================================================================
    input  logic [`ADDR_WIDTH-1:0]    input_addr,
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
    output logic                      result_valid
);

    // ========================================================================
    // Internal Wires
    // ========================================================================
    
    // Buffer outputs (packed)
    logic [`BUFFER_WIDTH-1:0] buffer_input_data;
    logic [`BUFFER_WIDTH-1:0] buffer_weight_data;
    
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
        .clk         (clk),
        .rst_n       (rst_n),
        .wr_en       (ub_wr_en),
        .wr_addr     (ub_wr_addr),
        .wr_data     (ub_wr_data),
        .input_addr  (input_addr),
        .input_data  (buffer_input_data),
        .weight_addr (weight_addr),
        .weight_data (buffer_weight_data)
    );

    // ========================================================================
    // Streaming Skewer for Inputs (Matrix A)
    // ========================================================================
    streaming_skewer #(
        .N(`ARRAY_SIZE),
        .DATA_WIDTH(`DATA_WIDTH)
    ) u_input_skewer (
        .clk      (clk),
        .rst_n    (rst_n),
        .en       (skewer_en),
        .data_in  (input_vector),
        .data_out (skewed_input)
    );

    // ========================================================================
    // Streaming Skewer for Weights (Matrix B)
    // ========================================================================
    streaming_skewer #(
        .N(`ARRAY_SIZE),
        .DATA_WIDTH(`DATA_WIDTH)
    ) u_weight_skewer (
        .clk      (clk),
        .rst_n    (rst_n),
        .en       (skewer_en),
        .data_in  (weight_vector),
        .data_out (skewed_weight)
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
