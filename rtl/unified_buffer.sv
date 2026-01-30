`include "defines.sv"

module unified_buffer (
    input logic clk,
    input logic rst_n,

    // Write interface (for loading data)
    input logic                     wr_en,
    input logic [  `ADDR_WIDTH-1:0] wr_addr,
    input logic [`BUFFER_WIDTH-1:0] wr_data,  // 64-bit: 4 × 16-bit packed

    // Read interface for inputs (feeds systolic array horizontally)
    input  logic [  `ADDR_WIDTH-1:0] input_addr,
    output logic [`BUFFER_WIDTH-1:0] input_data,  // 64-bit: 4 × 16-bit packed

    // Read interface for weights (feeds systolic array vertically)
    input  logic [  `ADDR_WIDTH-1:0] weight_addr,
    output logic [`BUFFER_WIDTH-1:0] weight_data   // 64-bit: 4 × 16-bit packed
);

  // Memory: BUFFER_DEPTH rows × BUFFER_WIDTH bits per row
  logic [`BUFFER_WIDTH-1:0] memory[`BUFFER_DEPTH-1:0];

  // Write logic
  always_ff @(negedge clk) begin
    if (wr_en) begin
      memory[wr_addr] <= wr_data;
    end
  end

  // Read logic
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      input_data  <= '0;
      weight_data <= '0;
    end else begin
      input_data  <= memory[input_addr];
      weight_data <= memory[weight_addr];
    end
  end

  initial begin
    $readmemh("buffer_init.hex", memory);
  end

endmodule
