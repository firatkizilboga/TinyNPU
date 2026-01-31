`include "defines.sv"

// ============================================================================
// UB + Skewer Test Wrapper
// ============================================================================
// Connects Unified Buffer to two Skewers (input path + weight path)
// for verifying data flow before systolic array integration.

module ub_skewer_wrapper #(
    parameter N = `ARRAY_SIZE,
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter ADDR_WIDTH = `ADDR_WIDTH,
    parameter BUFFER_WIDTH = `BUFFER_WIDTH,
    parameter string INIT_FILE = "buffer_init.hex"
) (
    input logic clk,
    input logic rst_n,
    input logic en,

    // Write interface (for loading UB)
    input logic                    wr_en,
    input logic [  ADDR_WIDTH-1:0] wr_addr,
    input logic [BUFFER_WIDTH-1:0] wr_data,

    // Input path address interface
    input logic [ADDR_WIDTH-1:0] input_addr,
    input logic                  input_first_in,
    input logic                  input_last_in,

    // Weight path address interface
    input logic [ADDR_WIDTH-1:0] weight_addr,
    input logic                  weight_first_in,
    input logic                  weight_last_in,



    // Input skewer outputs (flattened for Verilator)
    output logic [N*DATA_WIDTH-1:0] input_data_flat,
    output logic                    dut_input_first_out,
    output logic                    dut_input_last_out,

    // UB debug outputs (before skewer)
    output logic [N*DATA_WIDTH-1:0] mem_input_data_flat,
    output logic                    mem_input_first,
    output logic                    mem_input_last,

    // Weight skewer outputs (flattened for Verilator)
    output logic [N*DATA_WIDTH-1:0] weight_data_flat,
    output logic                    dut_weight_first_out,
    output logic                    dut_weight_last_out
);

  // ========================================================================
  // Internal wires
  // ========================================================================

  // UB outputs (before skewing)
  logic [BUFFER_WIDTH-1:0] ub_input_data;
  logic [BUFFER_WIDTH-1:0] ub_weight_data;
  logic                    ub_input_first;
  logic                    ub_input_last;
  logic                    ub_weight_first;
  logic                    ub_weight_last;

  // Unpacked arrays for skewer interface
  logic [  DATA_WIDTH-1:0] input_unpacked  [N-1:0];
  logic [  DATA_WIDTH-1:0] weight_unpacked [N-1:0];
  logic [  DATA_WIDTH-1:0] input_skewed    [N-1:0];
  logic [  DATA_WIDTH-1:0] weight_skewed   [N-1:0];

  // ========================================================================
  // Unified Buffer Instance
  // ========================================================================
  unified_buffer #(
      .INIT_FILE(INIT_FILE)
  ) ub_inst (
      .clk  (clk),
      .rst_n(rst_n),

      // Write interface
      .wr_en  (wr_en),
      .wr_addr(wr_addr),
      .wr_data(wr_data),

      // Input read interface
      .input_first_in (input_first_in),
      .input_last_in  (input_last_in),
      .input_addr     (input_addr),
      .input_first_out(ub_input_first),
      .input_last_out (ub_input_last),
      .input_data     (ub_input_data),

      // Weight read interface
      .weight_first_in (weight_first_in),
      .weight_last_in  (weight_last_in),
      .weight_addr     (weight_addr),
      .weight_first_out(ub_weight_first),
      .weight_last_out (ub_weight_last),
      .weight_data     (ub_weight_data)
  );

  // ========================================================================
  // Unpack UB output into arrays
  // ========================================================================
  genvar i;
  generate
    for (i = 0; i < N; i++) begin : unpack
      assign input_unpacked[i]  = ub_input_data[(i+1)*DATA_WIDTH-1-:DATA_WIDTH];
      assign weight_unpacked[i] = ub_weight_data[(i+1)*DATA_WIDTH-1-:DATA_WIDTH];
    end
  endgenerate

  // Debug: expose UB outputs directly
  assign mem_input_data_flat = ub_input_data;
  assign mem_input_first = ub_input_first;
  assign mem_input_last = ub_input_last;

  // ========================================================================
  // Input Skewer Instance
  // ========================================================================
  streaming_skewer #(
      .N         (N),
      .DATA_WIDTH(DATA_WIDTH)
  ) input_skewer (
      .clk          (clk),
      .rst_n        (rst_n),
      .en           (en),
      .data_in      (input_unpacked),
      .data_out     (input_skewed),
      .first_in     (ub_input_first),
      .last_in      (ub_input_last),
      .first_out    (dut_input_first_out),
      .last_out     (dut_input_last_out),
      .data_out_flat(input_data_flat)
  );

  // ========================================================================
  // Weight Skewer Instance
  // ========================================================================
  streaming_skewer #(
      .N         (N),
      .DATA_WIDTH(DATA_WIDTH)
  ) weight_skewer (
      .clk          (clk),
      .rst_n        (rst_n),
      .en           (en),
      .data_in      (weight_unpacked),
      .data_out     (weight_skewed),
      .first_in     (ub_weight_first),
      .last_in      (ub_weight_last),
      .first_out    (dut_weight_first_out),
      .last_out     (dut_weight_last_out),
      .data_out_flat(weight_data_flat)
  );

endmodule
