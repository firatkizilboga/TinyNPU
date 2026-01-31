`include "defines.sv"

module unified_buffer #(
    parameter string INIT_FILE = ""  // Hex file path (empty = no init)
) (
    input logic clk,
    input logic rst_n,

    // Write interface (for loading data)
    input logic                     wr_en,
    input logic [  `ADDR_WIDTH-1:0] wr_addr,
    input logic [`BUFFER_WIDTH-1:0] wr_data,

    // Read interface for inputs (feeds systolic array horizontally)
    input  logic                     input_first_in,   // First data marker
    input  logic                     input_last_in,    // Last data marker
    input  logic [  `ADDR_WIDTH-1:0] input_addr,
    output logic                     input_first_out,  // Delayed to match data
    output logic                     input_last_out,
    output logic [`BUFFER_WIDTH-1:0] input_data,

    // Read interface for weights (feeds systolic array vertically)
    input  logic                     weight_first_in,
    input  logic                     weight_last_in,
    input  logic [  `ADDR_WIDTH-1:0] weight_addr,
    output logic                     weight_first_out,
    output logic                     weight_last_out,
    output logic [`BUFFER_WIDTH-1:0] weight_data
);

  // Memory: BUFFER_DEPTH rows × BUFFER_WIDTH bits per row
  logic [`BUFFER_WIDTH-1:0] memory[`BUFFER_DEPTH-1:0];

  // Write logic (negedge for timing hygiene)
  always_ff @(negedge clk) begin
    if (wr_en) begin
      memory[wr_addr] <= wr_data;
    end
  end

  // Read logic - data and markers have same 1-cycle latency
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      input_data       <= '0;
      input_first_out  <= 1'b0;
      input_last_out   <= 1'b0;
      weight_data      <= '0;
      weight_first_out <= 1'b0;
      weight_last_out  <= 1'b0;
    end else begin
      input_data       <= memory[input_addr];
      input_first_out  <= input_first_in;
      input_last_out   <= input_last_in;
      weight_data      <= memory[weight_addr];
      weight_first_out <= weight_first_in;
      weight_last_out  <= weight_last_in;
    end
  end

  // Initialize memory from file for simulation (if provided)
  initial begin
    if (INIT_FILE != "") begin
      $readmemh(INIT_FILE, memory);
    end
  end

endmodule

