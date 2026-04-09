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
    input  logic                     conv_stream_gather_en,
    input  logic [  `ADDR_WIDTH-1:0] conv_stream_lane_word_addr[`ARRAY_SIZE-1:0][3:0],
    input  logic [$clog2(`ARRAY_SIZE)-1:0] conv_stream_lane_word_lane[`ARRAY_SIZE-1:0][3:0],
    input  logic [1:0]               conv_stream_lane_subidx[`ARRAY_SIZE-1:0][3:0],
    input  logic [3:0]               conv_stream_lane_valid[`ARRAY_SIZE-1:0],
    input  logic [1:0]               conv_stream_in_precision,
    output logic                     input_first_out,  // Delayed to match data
    output logic                     input_last_out,
    output logic [`BUFFER_WIDTH-1:0] input_data,
    output logic [`BUFFER_WIDTH-1:0] input_data_comb,

    // Read interface for weights (feeds systolic array vertically)
    input  logic                     weight_first_in,
    input  logic                     weight_last_in,
    input  logic [  `ADDR_WIDTH-1:0] weight_addr,
    output logic                     weight_first_out,
    output logic                     weight_last_out,
    output logic [`BUFFER_WIDTH-1:0] weight_data,

    // Optional host shared read tap.
    input  logic [  `ADDR_WIDTH-1:0] host_addr,
    output logic [`BUFFER_WIDTH-1:0] host_data_comb
);

  // Memory: BUFFER_DEPTH rows × BUFFER_WIDTH bits per row
  logic [`BUFFER_WIDTH-1:0] memory[`BUFFER_DEPTH-1:0];
  logic [`BUFFER_WIDTH-1:0] conv_stream_gather_word;
  logic [`BUFFER_WIDTH-1:0] conv_stream_lane_word;
  logic [15:0]              conv_stream_lane_elem16;
  logic [15:0]              conv_stream_lane_elem_unpacked;
  logic [15:0]              conv_stream_lane_packed;

  function automatic logic [15:0] conv_stream_unpack_elem(
      input logic [15:0] packed_word,
      input logic [1:0]  precision,
      input logic [1:0]  subidx
  );
    logic [15:0] unpacked;
    logic [7:0] byte_val;
    logic [3:0] nibble_val;
    begin
      unpacked = packed_word;
      unique case (precision)
        2'b00: begin  // INT4
          unique case (subidx)
            2'd0: nibble_val = packed_word[3:0];
            2'd1: nibble_val = packed_word[7:4];
            2'd2: nibble_val = packed_word[11:8];
            default: nibble_val = packed_word[15:12];
          endcase
          unpacked = {{12{nibble_val[3]}}, nibble_val};
        end
        2'b01: begin  // INT8
          unique case (subidx[0])
            1'b0: byte_val = packed_word[7:0];
            default: byte_val = packed_word[15:8];
          endcase
          unpacked = {{8{byte_val[7]}}, byte_val};
        end
        default: begin  // INT16
          unpacked = packed_word;
        end
      endcase
      return unpacked;
    end
  endfunction

  // Port A combinational read tap (used by CU when arbiter maps CU onto Port A).
  assign input_data_comb = memory[input_addr];
  assign host_data_comb  = memory[host_addr];

  // Conv-stream gather: build one synthetic Role-A word by selecting one
  // element per output lane from potentially different UB words.
  always_comb begin
    conv_stream_gather_word = '0;
    conv_stream_lane_word = '0;
    conv_stream_lane_elem16 = '0;
    conv_stream_lane_elem_unpacked = '0;
    conv_stream_lane_packed = '0;
    for (int lane = 0; lane < `ARRAY_SIZE; lane++) begin
      conv_stream_lane_packed = '0;
      unique case (conv_stream_in_precision)
        2'b00: begin  // INT4: gather four logical K-elements into one packed lane
          for (int comp = 0; comp < 4; comp++) begin
            if (conv_stream_lane_valid[lane][comp]) begin
              conv_stream_lane_word = memory[conv_stream_lane_word_addr[lane][comp]];
              conv_stream_lane_elem16 =
                  conv_stream_lane_word[(conv_stream_lane_word_lane[lane][comp] * 16) +: 16];
              conv_stream_lane_elem_unpacked = conv_stream_unpack_elem(
                  conv_stream_lane_elem16,
                  conv_stream_in_precision,
                  conv_stream_lane_subidx[lane][comp]
              );
              conv_stream_lane_packed[(comp * 4) +: 4] = conv_stream_lane_elem_unpacked[3:0];
            end
          end
        end
        2'b01: begin  // INT8: gather two logical K-elements into one packed lane
          for (int comp = 0; comp < 2; comp++) begin
            if (conv_stream_lane_valid[lane][comp]) begin
              conv_stream_lane_word = memory[conv_stream_lane_word_addr[lane][comp]];
              conv_stream_lane_elem16 =
                  conv_stream_lane_word[(conv_stream_lane_word_lane[lane][comp] * 16) +: 16];
              conv_stream_lane_elem_unpacked = conv_stream_unpack_elem(
                  conv_stream_lane_elem16,
                  conv_stream_in_precision,
                  conv_stream_lane_subidx[lane][comp]
              );
              conv_stream_lane_packed[(comp * 8) +: 8] = conv_stream_lane_elem_unpacked[7:0];
            end
          end
        end
        default: begin  // INT16
          if (conv_stream_lane_valid[lane][0]) begin
            conv_stream_lane_word = memory[conv_stream_lane_word_addr[lane][0]];
            conv_stream_lane_elem16 =
                conv_stream_lane_word[(conv_stream_lane_word_lane[lane][0] * 16) +: 16];
            conv_stream_lane_packed = conv_stream_unpack_elem(
                conv_stream_lane_elem16,
                conv_stream_in_precision,
                conv_stream_lane_subidx[lane][0]
            );
          end
        end
      endcase
      conv_stream_gather_word[(lane * 16) +: 16] = conv_stream_lane_packed;
    end
  end

  // Write logic (negedge for timing hygiene)
  always_ff @(negedge clk) begin
    if (wr_en) begin
      // Bit-masked write: only update bits where wr_mask is high
      memory[wr_addr] <= (memory[wr_addr] & ~wr_mask) | (wr_data & wr_mask);
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
      input_data       <= conv_stream_gather_en ? conv_stream_gather_word : memory[input_addr];
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

  // Dump memory at end of simulation for debugging
  final begin
      $writememh("ub_dump_rtl.hex", memory);
  end

endmodule
