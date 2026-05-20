`include "defines.sv"

// One byte-wide synchronous RAM macro wrapper.
//
// TINYNPU_FPGA_BRAM without TINYNPU_VIVADO_BRAM keeps this as a blackbox for
// lightweight open-source resource runs. Vivado timing uses the routeable BRAM
// template below so the full design can place and route with real memories.
`ifdef TINYNPU_FPGA_BRAM
`ifndef TINYNPU_VIVADO_BRAM
(* blackbox *) module tinynpu_byte_ram #(
    parameter int DEPTH = `BUFFER_DEPTH,
    parameter int ADDR_BITS = `ADDR_WIDTH
) (
    input  logic                 clk,
    input  logic                 wr_en,
    input  logic [ADDR_BITS-1:0] wr_addr,
    input  logic [7:0]           wr_data,
    input  logic [ADDR_BITS-1:0] rd_addr,
    output logic [7:0]           rd_data
);
endmodule
`else
module tinynpu_byte_ram #(
    parameter int DEPTH = `BUFFER_DEPTH,
    parameter int ADDR_BITS = `ADDR_WIDTH
) (
    input  logic                 clk,
    input  logic                 wr_en,
    input  logic [ADDR_BITS-1:0] wr_addr,
    input  logic [7:0]           wr_data,
    input  logic [ADDR_BITS-1:0] rd_addr,
    output logic [7:0]           rd_data
);
  (* ram_style = "block" *) logic [7:0] mem[0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (wr_en) begin
      mem[wr_addr] <= wr_data;
    end
    rd_data <= mem[rd_addr];
  end
endmodule
`endif
`else
module tinynpu_byte_ram #(
    parameter int DEPTH = `BUFFER_DEPTH,
    parameter int ADDR_BITS = `ADDR_WIDTH
) (
    input  logic                 clk,
    input  logic                 wr_en,
    input  logic [ADDR_BITS-1:0] wr_addr,
    input  logic [7:0]           wr_data,
    input  logic [ADDR_BITS-1:0] rd_addr,
    output logic [7:0]           rd_data
);
  logic [7:0] mem[0:DEPTH-1];

  always_ff @(posedge clk) begin
    if (wr_en) begin
      mem[wr_addr] <= wr_data;
    end
    rd_data <= mem[rd_addr];
  end
endmodule
`endif
