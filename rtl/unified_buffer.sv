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
  // byte-wide RAM macros. Sub-byte INT4 writes are merged in a small staging
  // CAM, then committed as full-byte writes so the large UB RAM never needs
  // nibble write enables.
  localparam int UB_BYTES = `BUFFER_WIDTH / 8;
  localparam int PARTIAL_ENTRIES = 16;
  localparam int PARTIAL_IDX_W = $clog2(PARTIAL_ENTRIES);

  logic [UB_BYTES-1:0] byte_commit_wr_en;
  logic [7:0]          byte_commit_wr_data[UB_BYTES-1:0];
  logic [7:0]          input_byte_rd_data [UB_BYTES-1:0];
  logic [7:0]          weight_byte_rd_data[UB_BYTES-1:0];
  logic [`BUFFER_WIDTH-1:0] byte_commit_mask;
  logic [`BUFFER_WIDTH-1:0] byte_commit_word;
  logic [`BUFFER_WIDTH-1:0] input_data_raw;
  logic [`BUFFER_WIDTH-1:0] weight_data_raw;

  logic                         wr_req_valid;
  logic [`BUFFER_WIDTH-1:0]     wr_req_mask;
  logic [  `ADDR_WIDTH-1:0]     wr_req_addr;
  logic [`BUFFER_WIDTH-1:0]     wr_req_data;

  logic                         partial_valid[PARTIAL_ENTRIES-1:0];
  logic [`ADDR_WIDTH-1:0]       partial_addr [PARTIAL_ENTRIES-1:0];
  logic [`BUFFER_WIDTH-1:0]     partial_data [PARTIAL_ENTRIES-1:0];
  logic [`BUFFER_WIDTH-1:0]     partial_mask [PARTIAL_ENTRIES-1:0];

  logic                         wr_has_partial_mask;
  logic                         partial_match_found;
  logic                         partial_free_found;
  logic [PARTIAL_IDX_W-1:0]     partial_match_idx;
  logic [PARTIAL_IDX_W-1:0]     partial_free_idx;
  logic [PARTIAL_IDX_W-1:0]     partial_update_idx;
  logic [`BUFFER_WIDTH-1:0]     partial_selected_data;
  logic [`BUFFER_WIDTH-1:0]     partial_selected_mask;
  logic [`BUFFER_WIDTH-1:0]     partial_merged_data;
  logic [`BUFFER_WIDTH-1:0]     partial_merged_mask;
  logic [`BUFFER_WIDTH-1:0]     partial_next_data;
  logic [`BUFFER_WIDTH-1:0]     partial_next_mask;
  logic                         partial_update_needed;

  always_comb begin
    wr_has_partial_mask  = 1'b0;
    partial_match_found  = 1'b0;
    partial_free_found   = 1'b0;
    partial_match_idx    = '0;
    partial_free_idx     = '0;
    partial_selected_data = '0;
    partial_selected_mask = '0;

    for (int b = 0; b < UB_BYTES; b++) begin
      if (wr_req_mask[b*8 +: 8] != 8'h00 && wr_req_mask[b*8 +: 8] != 8'hFF) begin
        wr_has_partial_mask = 1'b1;
      end
    end

    for (int e = 0; e < PARTIAL_ENTRIES; e++) begin
      if (partial_valid[e] && partial_addr[e] == wr_req_addr && !partial_match_found) begin
        partial_match_found = 1'b1;
        partial_match_idx = e[PARTIAL_IDX_W-1:0];
      end
      if (!partial_valid[e] && !partial_free_found) begin
        partial_free_found = 1'b1;
        partial_free_idx = e[PARTIAL_IDX_W-1:0];
      end
    end

    partial_update_idx = partial_match_found ? partial_match_idx : partial_free_idx;
    if (partial_match_found) begin
      partial_selected_data = partial_data[partial_match_idx];
      partial_selected_mask = partial_mask[partial_match_idx];
    end

    partial_merged_data = (partial_selected_data & ~wr_req_mask) | (wr_req_data & wr_req_mask);
    partial_merged_mask = partial_selected_mask | wr_req_mask;
    partial_next_mask = partial_merged_mask;
    byte_commit_mask = '0;
    byte_commit_word = '0;

    for (int b = 0; b < UB_BYTES; b++) begin
      byte_commit_wr_en[b] = wr_req_valid && (wr_req_mask[b*8 +: 8] != 8'h00);
      byte_commit_wr_data[b] = partial_merged_data[b*8 +: 8];
      if (byte_commit_wr_en[b]) begin
        byte_commit_mask[b*8 +: 8] = 8'hFF;
        byte_commit_word[b*8 +: 8] = byte_commit_wr_data[b];
      end
      if (wr_req_mask[b*8 +: 8] != 8'h00 && partial_merged_mask[b*8 +: 8] == 8'hFF) begin
        partial_next_mask[b*8 +: 8] = 8'h00;
      end
    end

    partial_next_data = partial_merged_data & partial_next_mask;
    partial_update_needed = wr_req_valid && (partial_match_found || wr_has_partial_mask);
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wr_req_valid <= 1'b0;
      wr_req_addr  <= '0;
      wr_req_data  <= '0;
      wr_req_mask  <= '0;
      for (int e = 0; e < PARTIAL_ENTRIES; e++) begin
        partial_valid[e] <= 1'b0;
        partial_addr[e]  <= '0;
        partial_data[e]  <= '0;
        partial_mask[e]  <= '0;
      end
    end else begin
      wr_req_valid <= wr_en;
      wr_req_addr  <= wr_addr;
      wr_req_data  <= wr_data;
      wr_req_mask  <= wr_mask;

      if (partial_update_needed) begin
        if (!partial_match_found && !partial_free_found) begin
`ifndef SYNTHESIS
          $fatal(1, "unified_buffer INT4 partial-byte packer overflow");
`endif
        end else if (partial_next_mask == '0) begin
          partial_valid[partial_update_idx] <= 1'b0;
          partial_data[partial_update_idx]  <= '0;
          partial_mask[partial_update_idx]  <= '0;
        end else begin
          partial_valid[partial_update_idx] <= 1'b1;
          partial_addr[partial_update_idx]  <= wr_req_addr;
          partial_data[partial_update_idx]  <= partial_next_data;
          partial_mask[partial_update_idx]  <= partial_next_mask;
        end
      end
    end
  end

  generate
    genvar byte_idx;
    for (byte_idx = 0; byte_idx < UB_BYTES; byte_idx++) begin : g_byte_banks
      tinynpu_byte_ram u_input_bank (
          .clk    (clk),
          .wr_en  (byte_commit_wr_en[byte_idx]),
          .wr_addr(wr_req_addr),
          .wr_data(byte_commit_wr_data[byte_idx]),
          .rd_addr(input_addr),
          .rd_data(input_byte_rd_data[byte_idx])
      );

      tinynpu_byte_ram u_weight_bank (
          .clk    (clk),
          .wr_en  (byte_commit_wr_en[byte_idx]),
          .wr_addr(wr_req_addr),
          .wr_data(byte_commit_wr_data[byte_idx]),
          .rd_addr(weight_addr),
          .rd_data(weight_byte_rd_data[byte_idx])
      );
    end
  endgenerate

  // Byte RAM rd_data is already registered, so expose it directly to keep the
  // same one-cycle read latency as the functional UB model.
  always_comb begin
    input_data_raw  = '0;
    weight_data_raw = '0;
    input_data      = '0;
    weight_data     = '0;
    if (rst_n) begin
      for (int b = 0; b < UB_BYTES; b++) begin
        input_data_raw[b*8 +: 8]  = input_byte_rd_data[b];
        weight_data_raw[b*8 +: 8] = weight_byte_rd_data[b];
      end
      input_data = input_data_raw;
      weight_data = weight_data_raw;
      if (wr_req_valid && input_addr == wr_req_addr) begin
        input_data = (input_data_raw & ~byte_commit_mask) | (byte_commit_word & byte_commit_mask);
      end
      if (wr_req_valid && weight_addr == wr_req_addr) begin
        weight_data = (weight_data_raw & ~byte_commit_mask) | (byte_commit_word & byte_commit_mask);
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      input_first_out  <= 1'b0;
      input_last_out   <= 1'b0;
      weight_first_out <= 1'b0;
      weight_last_out  <= 1'b0;
    end else begin
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
