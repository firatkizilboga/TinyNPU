`include "defines.sv"

module unified_buffer #(
    parameter INIT_FILE = ""  // Hex file path (empty = no init)
) (
    input logic clk,
    input logic rst_n,

    // Write interface (for loading data)
    input logic                     wr_en,
    input logic [`BUFFER_WIDTH-1:0] wr_mask, // Bit-mask for partial writes
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

`ifdef TINYNPU_FPGA_BRAM
  // FPGA BRAM mode: replicate storage for 2R1W and split each 128-bit row into
  // byte-wide RAM macros. This deliberately rejects sub-byte masks. INT4 packed
  // writeback needs a staged read-modify-write path before it can be
  // FPGA-realistic.
  // TODO(int4-fpga): add an INT4 writeback packer that accumulates four
  // nibbles per 16-bit lane in registers, then commits a full byte/halfword to
  // UB. Do not reintroduce nibble write enables into the large FPGA memory.
  localparam int UB_BYTES = `BUFFER_WIDTH / 8;
  logic [UB_BYTES-1:0] input_byte_wr_en;
  logic [UB_BYTES-1:0] weight_byte_wr_en;
  logic [7:0]          input_byte_rd_data [UB_BYTES-1:0];
  logic [7:0]          weight_byte_rd_data[UB_BYTES-1:0];

  generate
    genvar byte_idx;
    for (byte_idx = 0; byte_idx < UB_BYTES; byte_idx++) begin : g_byte_banks
      assign input_byte_wr_en[byte_idx] = wr_en && (wr_mask[byte_idx*8 +: 8] == 8'hFF);
      assign weight_byte_wr_en[byte_idx] = wr_en && (wr_mask[byte_idx*8 +: 8] == 8'hFF);

      tinynpu_byte_ram u_input_bank (
          .clk    (clk),
          .wr_en  (input_byte_wr_en[byte_idx]),
          .wr_addr(wr_addr),
          .wr_data(wr_data[byte_idx*8 +: 8]),
          .rd_addr(input_addr),
          .rd_data(input_byte_rd_data[byte_idx])
      );

      tinynpu_byte_ram u_weight_bank (
          .clk    (clk),
          .wr_en  (weight_byte_wr_en[byte_idx]),
          .wr_addr(wr_addr),
          .wr_data(wr_data[byte_idx*8 +: 8]),
          .rd_addr(weight_addr),
          .rd_data(weight_byte_rd_data[byte_idx])
      );
    end
  endgenerate

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
      for (int b = 0; b < UB_BYTES; b++) begin
        input_data[b*8 +: 8]  <= input_byte_rd_data[b];
        weight_data[b*8 +: 8] <= weight_byte_rd_data[b];
      end
      input_first_out  <= input_first_in;
      input_last_out   <= input_last_in;
      weight_first_out <= weight_first_in;
      weight_last_out  <= weight_last_in;
    end
  end

`else
  // Functional model: supports arbitrary bit masks, including INT4 nibble
  // writeback. This is convenient for RTL simulation but is not a direct FPGA
  // BRAM template.
  logic [`BUFFER_WIDTH-1:0] memory [`BUFFER_DEPTH-1:0];
  logic [`BUFFER_WIDTH-1:0] weight_bank[`BUFFER_DEPTH-1:0];

  always_ff @(posedge clk) begin
    if (wr_en) begin
      memory[wr_addr]      <= (memory[wr_addr]      & ~wr_mask) | (wr_data & wr_mask);
      weight_bank[wr_addr] <= (weight_bank[wr_addr] & ~wr_mask) | (wr_data & wr_mask);
    end
  end

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
      weight_data      <= weight_bank[weight_addr];
      weight_first_out <= weight_first_in;
      weight_last_out  <= weight_last_in;
    end
  end
`endif

`ifndef TINYNPU_FPGA_BRAM
  // Initialize memory from file for simulation (if provided)
  initial begin
    if (INIT_FILE != "") begin
      $readmemh(INIT_FILE, memory);
      $readmemh(INIT_FILE, weight_bank);
    end
  end

  // Dump memory at end of simulation for debugging
  final begin
      $writememh("ub_dump_rtl.hex", memory);
  end
`endif

endmodule
