`include "defines.sv"

// One byte-wide synchronous RAM macro wrapper.
//
// In FPGA synthesis mode this is intentionally a blackbox: the implementation
// should be bound to vendor BRAM/URAM IP or generated memory macros. Keeping the
// macro boundary prevents open-source synthesis from exploding the 4-8 Mib UB
// into flip-flops/LUTRAM while preserving the correct memory architecture.
`ifdef TINYNPU_FPGA_BRAM
(* blackbox *) module tinynpu_byte_ram (
    input  logic                   clk,
    input  logic                   wr_en,
    input  logic [`ADDR_WIDTH-1:0] wr_addr,
    input  logic [7:0]             wr_data,
    input  logic [`ADDR_WIDTH-1:0] rd_addr,
    output logic [7:0]             rd_data
);
endmodule
`else
module tinynpu_byte_ram (
    input  logic                   clk,
    input  logic                   wr_en,
    input  logic [`ADDR_WIDTH-1:0] wr_addr,
    input  logic [7:0]             wr_data,
    input  logic [`ADDR_WIDTH-1:0] rd_addr,
    output logic [7:0]             rd_data
);
  logic [7:0] mem[`BUFFER_DEPTH-1:0];

  always_ff @(posedge clk) begin
    if (wr_en) begin
      mem[wr_addr] <= wr_data;
    end
    rd_data <= mem[rd_addr];
  end
endmodule
`endif
