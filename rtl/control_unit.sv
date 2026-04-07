`include "defines.sv"

module control_unit #(
    parameter PERF_ENABLE = 0
) (
    input logic clk,
    input logic rst_n,

    // Interface to MMIO
    input  logic [`HOST_DATA_WIDTH-1:0] cmd_in,
    input  logic [     `ADDR_WIDTH-1:0] addr_in,
    input  logic [      `ARG_WIDTH-1:0] arg_in,
    input  logic [   `BUFFER_WIDTH-1:0] mmvr_in,
    input  logic                        doorbell_pulse,
    output logic [`HOST_DATA_WIDTH-1:0] status_out,
    output logic                        mmvr_wr_en,
    output logic [   `BUFFER_WIDTH-1:0] mmvr_out,

    // Instruction Memory Interface
    output logic                     im_wr_en,
    output logic [  `ADDR_WIDTH-1:0] im_addr,
    output logic [`BUFFER_WIDTH-1:0] im_wdata,
    input  logic [  `INST_WIDTH-1:0] im_rdata,

    // Unified Buffer Interface (Data)
    output logic                     ub_req,
    output logic                     ub_wr_en,
    output logic [  `ADDR_WIDTH-1:0] ub_addr,
    output logic [  `ADDR_WIDTH-1:0] ub_w_addr,
    output logic [`BUFFER_WIDTH-1:0] ub_wdata,
    input  logic [`BUFFER_WIDTH-1:0] ub_rdata,
    // Conv-stream gather path (lane-wise UB reads for on-the-fly windowing)
    output logic                     conv_stream_gather_en,
    output logic [  `ADDR_WIDTH-1:0] conv_stream_lane_word_addr[`ARRAY_SIZE-1:0],
    output logic [$clog2(`ARRAY_SIZE)-1:0] conv_stream_lane_word_lane[`ARRAY_SIZE-1:0],
    output logic [1:0]               conv_stream_lane_subidx[`ARRAY_SIZE-1:0],
    output logic [  `ARRAY_SIZE-1:0] conv_stream_lane_valid,
    output logic [1:0]               conv_stream_in_precision,

    // Systolic Control & Markers
    output logic acc_clear,
    output logic compute_enable,
    output logic drain_enable,     // New: Control PPU/Drain
    output logic sa_input_first,
    output logic sa_input_last,
    output logic sa_weight_first,
    output logic sa_weight_last,

    // PPU Control
    output logic                           ppu_wb_en,
    output logic                           ppu_bias_en,
    output logic                           ppu_bias_clear,
    output logic [$clog2(`ARRAY_SIZE)-1:0] ppu_cycle_idx,
    output logic                           ppu_capture_en,
    output logic [ 7:0]                    ppu_shift,
    output logic [15:0]                    ppu_multiplier,
    output logic [ 7:0]                    ppu_activation,
    output logic [ 7:0]                    ppu_h_gelu_x_scale_shift,
    output logic [ 1:0]                    ppu_in_precision,
    output logic [ 1:0]                    ppu_out_precision,
    output logic [ 1:0]                    ppu_write_offset,
    output output_layout_t                 ppu_output_layout,

    input logic all_done_in
);

  typedef enum logic [3:0] {
    CTRL_IDLE,
    CTRL_HOST_WRITE,
    CTRL_HOST_READ,
    CTRL_READ_WAIT,
    CTRL_FETCH,
    CTRL_DECODE,
    CTRL_EXEC_MOVE,
    CTRL_EXEC_MATMUL,
    CTRL_MM_CLEAR,
    CTRL_MM_FEED,
    CTRL_MM_WAIT,
    CTRL_MM_LOAD_BIAS,  // New: Load second half of 32-bit bias
    CTRL_MM_DRAIN_SA,   // Drain array into PPU
    CTRL_MM_WRITEBACK,  // Write PPU to Buffer
    CTRL_HALT
  } ctrl_state_t;

  ctrl_state_t state, next_state;
  logic [`ADDR_WIDTH-1:0] pc, pc_next;
  localparam int CTRL_STATE_COUNT = 15;
  localparam int PERF_COUNTER_WIDTH = 64;
  localparam int PERF_COUNTERS_FLAT_WIDTH = CTRL_STATE_COUNT * PERF_COUNTER_WIDTH;

  // Safety Latches for MMIO inputs
  logic [`HOST_DATA_WIDTH-1:0] latched_cmd;
  logic [     `ADDR_WIDTH-1:0] latched_addr;
  logic [   `BUFFER_WIDTH-1:0] latched_mmvr;
  logic [      `ARG_WIDTH-1:0] latched_arg;

  // Register to break combinational loop on memory read
  logic [   `BUFFER_WIDTH-1:0] ub_rdata_reg;

  // --- Internal Registers for MOVE ---
  logic [`ADDR_WIDTH-1:0] move_src, move_src_next;
  logic [`ADDR_WIDTH-1:0] move_dest, move_dest_next;
  logic [`ADDR_WIDTH-1:0] move_count, move_count_next;
  logic move_phase, move_phase_next;

  // --- Internal Registers for MATMUL ---
  logic [`ADDR_WIDTH-1:0] mm_a_base, mm_b_base, mm_c_base, mm_bias_base;
  logic [15:0] mm_m_total, mm_k_total, mm_n_total;
  logic [ 7:0] mm_shift;
  logic [15:0] mm_multiplier;
  logic [ 7:0] mm_activation;
  logic [ 7:0] mm_h_gelu_x_scale_shift;
  logic [ 1:0] mm_in_precision;
  logic [ 1:0] mm_out_precision;
  logic [ 1:0] mm_write_offset;
  output_layout_t mm_output_layout;
  logic        mm_conv_stream_en;
  logic [ 7:0] mm_conv_kernel;
  logic [ 7:0] mm_conv_stride;
  logic [ 7:0] mm_conv_padding;
  logic [15:0] mm_conv_in_h;
  logic [15:0] mm_conv_in_w;
  logic [14:0] mm_conv_in_c;
  logic [15:0] m_idx, n_idx, k_idx;

  // Flexible counter width for NxN support
  logic [$clog2(`ARRAY_SIZE)-1:0] cycle_cnt, cycle_next;

  logic [15:0] m_next, n_next, k_next;
  logic [PERF_COUNTER_WIDTH-1:0] perf_total_cycles;
  logic [PERF_COUNTER_WIDTH-1:0] perf_state_cycles [0:CTRL_STATE_COUNT-1];
  logic [PERF_COUNTER_WIDTH-1:0] perf_state_entries[0:CTRL_STATE_COUNT-1];
  logic [PERF_COUNTERS_FLAT_WIDTH-1:0] perf_state_cycles_flat;
  logic [PERF_COUNTERS_FLAT_WIDTH-1:0] perf_state_entries_flat;
  logic [3:0] perf_state_id;

  assign perf_state_id = state;

  generate
    for (genvar perf_idx = 0; perf_idx < CTRL_STATE_COUNT; perf_idx++) begin : gen_perf_flatten
      assign perf_state_cycles_flat[(perf_idx * PERF_COUNTER_WIDTH) +: PERF_COUNTER_WIDTH] = perf_state_cycles[perf_idx];
      assign perf_state_entries_flat[(perf_idx * PERF_COUNTER_WIDTH) +: PERF_COUNTER_WIDTH] = perf_state_entries[perf_idx];
    end
  endgenerate

  // --- Sequential State ---
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= CTRL_IDLE;
      pc    <= '0;
      ub_rdata_reg <= '0;
      {latched_cmd, latched_addr, latched_mmvr, latched_arg} <= '0;
      {move_src, move_dest, move_count, move_phase} <= '0;
      {mm_a_base, mm_b_base, mm_c_base, mm_bias_base, mm_m_total, mm_k_total, mm_n_total} <= '0;
      {mm_shift, mm_multiplier, mm_activation, mm_h_gelu_x_scale_shift, mm_in_precision, mm_out_precision, mm_write_offset} <= '0;
      mm_output_layout <= OUT_LAYOUT_C;
      mm_conv_stream_en <= 1'b0;
      mm_conv_kernel <= '0;
      mm_conv_stride <= '0;
      mm_conv_padding <= '0;
      mm_conv_in_h <= '0;
      mm_conv_in_w <= '0;
      mm_conv_in_c <= '0;
      {m_idx, n_idx, k_idx, cycle_cnt} <= '0;
      perf_total_cycles <= '0;
      for (int perf_i = 0; perf_i < CTRL_STATE_COUNT; perf_i++) begin
        perf_state_cycles[perf_i] <= '0;
        perf_state_entries[perf_i] <= '0;
      end
      if (PERF_ENABLE) begin
        perf_state_entries[CTRL_IDLE] <= 64'd1;
      end
    end else begin
      state <= next_state;
      pc    <= pc_next;
      ub_rdata_reg <= ub_rdata;


      // Doorbell Latching
      if ((state == CTRL_IDLE || state == CTRL_READ_WAIT || state == CTRL_HALT) && doorbell_pulse) begin
        latched_cmd  <= cmd_in;
        latched_addr <= addr_in;
        latched_mmvr <= mmvr_in;
        // Only latch ARG if it's a RUN command to avoid corruption from memory data
        if (cmd_in == `CMD_RUN) begin
            latched_arg  <= arg_in;
        end
      end

      // MOVE Update
      move_src   <= move_src_next;
      move_dest  <= move_dest_next;
      move_count <= move_count_next;
      move_phase <= move_phase_next;

      // MATMUL Latch from Instruction Memory
      if (state == CTRL_DECODE && im_rdata[255:252] == ISA_OP_MATMUL) begin
        mm_a_base <= im_rdata[247:232];
        mm_b_base <= im_rdata[231:216];
        mm_c_base <= im_rdata[215:200];  // Load C Base
        mm_m_total <= im_rdata[183:168];
        mm_k_total <= im_rdata[167:152];
        mm_n_total <= im_rdata[151:136];
        mm_bias_base <= im_rdata[135:120];
        mm_shift      <= im_rdata[119:112];
        mm_multiplier <= im_rdata[111:96];
        mm_activation <= im_rdata[95:88];
        mm_out_precision  <= im_rdata[87:86];
        mm_write_offset   <= im_rdata[85:84];
        mm_in_precision   <= im_rdata[83:82];
        mm_h_gelu_x_scale_shift <= im_rdata[81:74];
        mm_output_layout  <= output_layout_t'(im_rdata[73:72]);
        mm_conv_stream_en <= im_rdata[71];
        mm_conv_kernel <= im_rdata[70:63];
        mm_conv_stride <= im_rdata[62:55];
        mm_conv_padding <= im_rdata[54:47];
        mm_conv_in_h <= im_rdata[46:31];
        mm_conv_in_w <= im_rdata[30:15];
        mm_conv_in_c <= im_rdata[14:0];
        m_idx <= '0;
        n_idx <= '0;
        k_idx <= '0;
        cycle_cnt <= '0;
      end else begin
        m_idx <= m_next;
        n_idx <= n_next;
        k_idx <= k_next;
        cycle_cnt <= cycle_next;
      end

      if (PERF_ENABLE) begin
        perf_total_cycles <= perf_total_cycles + 64'd1;
        perf_state_cycles[state] <= perf_state_cycles[state] + 64'd1;
        if (state != next_state) begin
          perf_state_entries[next_state] <= perf_state_entries[next_state] + 64'd1;
        end
      end
    end
  end

    // Tile packing: derive physical address and byte lane from m_idx and precision.
    // This aligns Row-Major Output C with Row-Major Input B for chaining.
    logic [ 1:0] packed_write_offset;
    logic [15:0] m_idx_packed;
    logic [15:0] m_total_packed;
    logic [15:0] n_idx_packed;
    logic [15:0] n_total_packed;
    logic [$clog2(`ARRAY_SIZE)-1:0]   a_word_base;
    logic [$clog2(`ARRAY_SIZE+1)-1:0] a_words_per_tile;
    logic        wb_valid_cycle;
    logic [15:0] conv_row_lin;
    logic [15:0] conv_out_h;
    logic [15:0] conv_out_w;
    logic [31:0] conv_out_elems;
    logic [15:0] conv_c_phys;
    logic [15:0] conv_c_tiles;
    logic [15:0] conv_k_window_idx;
    logic [15:0] conv_c_tile_idx;
    logic [15:0] conv_k_channel_phys;
    logic [1:0]  conv_k_subidx;
    logic [15:0] conv_oy;
    logic [15:0] conv_ox;
    logic [15:0] conv_ky;
    logic [15:0] conv_kx;
    logic [15:0] conv_in_y;
    logic [15:0] conv_in_x;
    logic [15:0] conv_in_row_lin;
    logic [15:0] conv_input_addr;
    logic        conv_row_valid;
    logic [2:0]  conv_pack_factor;
    logic [31:0] conv_window_elems;
    logic [31:0] conv_k_logical;
    logic [15:0] conv_k_channel;
    int signed conv_num_h_i;
    int signed conv_num_w_i;
    logic [15:0] conv_lane_row_lin;
    logic [15:0] conv_lane_oy;
    logic [15:0] conv_lane_ox;
    logic [15:0] conv_lane_in_y;
    logic [15:0] conv_lane_in_x;
    logic [15:0] conv_lane_in_row_lin;
    logic [15:0] conv_lane_col_tile;
    logic [15:0] conv_lane_col_lane;
    logic [15:0] conv_lane_row_tile;
    logic [$clog2(`ARRAY_SIZE)-1:0] conv_lane_row_lane;
    logic [15:0] conv_lane_window_idx;
    logic [15:0] conv_lane_kx;
    logic [15:0] conv_lane_ky;
    logic [15:0] conv_lane_k_channel_phys;
    logic [1:0]  conv_lane_subidx;
    int signed conv_lane_in_y_i;
    int signed conv_lane_in_x_i;
    logic conv_lane_valid_i;

    always_comb begin
        unique case (mm_out_precision)
            2'b00: begin // INT4: 4 tiles per word
                m_idx_packed     = m_idx >> 2;
                packed_write_offset = m_idx[1:0];
                m_total_packed   = (mm_m_total + 16'd3) >> 2;
                n_idx_packed     = n_idx >> 2;
                n_total_packed   = (mm_n_total + 16'd3) >> 2;
                a_word_base      = (n_idx[1:0] * (`ARRAY_SIZE / 4));
                a_words_per_tile = (`ARRAY_SIZE / 4);
            end
            2'b01: begin // INT8: 2 tiles per word
                m_idx_packed     = m_idx >> 1;
                packed_write_offset = {1'b0, m_idx[0]};
                m_total_packed   = (mm_m_total + 16'd1) >> 1;
                n_idx_packed     = n_idx >> 1;
                n_total_packed   = (mm_n_total + 16'd1) >> 1;
                a_word_base      = (n_idx[0] * (`ARRAY_SIZE / 2));
                a_words_per_tile = (`ARRAY_SIZE / 2);
            end
            default: begin // INT16: 1 tile per word
                m_idx_packed     = m_idx;
                packed_write_offset = 2'b0;
                m_total_packed   = mm_m_total;
                n_idx_packed     = n_idx;
                n_total_packed   = mm_n_total;
                a_word_base      = '0;
                a_words_per_tile = `ARRAY_SIZE;
            end
        endcase
        wb_valid_cycle = (mm_output_layout != OUT_LAYOUT_A) || (cycle_cnt < a_words_per_tile);

        conv_row_lin = (m_idx * `ARRAY_SIZE) + cycle_cnt;
        conv_out_h = 16'd0;
        conv_out_w = 16'd0;
        conv_num_h_i = int'(mm_conv_in_h) + (int'(mm_conv_padding) * 2) - int'(mm_conv_kernel);
        conv_num_w_i = int'(mm_conv_in_w) + (int'(mm_conv_padding) * 2) - int'(mm_conv_kernel);
        if (mm_conv_stride != 0 && mm_conv_kernel != 0 && conv_num_h_i >= 0 && conv_num_w_i >= 0) begin
            conv_out_h = (conv_num_h_i / int'(mm_conv_stride)) + 1;
            conv_out_w = (conv_num_w_i / int'(mm_conv_stride)) + 1;
        end
        conv_out_elems = conv_out_h * conv_out_w;
        unique case (mm_in_precision)
            2'b00: conv_pack_factor = 3'd4; // INT4
            2'b01: conv_pack_factor = 3'd2; // INT8
            default: conv_pack_factor = 3'd1; // INT16
        endcase
        conv_c_phys = (mm_conv_in_c + conv_pack_factor - 1) / conv_pack_factor;
        conv_c_tiles = (conv_c_phys + `ARRAY_SIZE - 1) / `ARRAY_SIZE;
        conv_k_window_idx = (conv_c_tiles != 0) ? (k_idx / conv_c_tiles) : 16'd0;
        conv_c_tile_idx = (conv_c_tiles != 0) ? (k_idx % conv_c_tiles) : 16'd0;
        conv_ky = (mm_conv_kernel != 0) ? (conv_k_window_idx / mm_conv_kernel) : 16'd0;
        conv_kx = (mm_conv_kernel != 0) ? (conv_k_window_idx % mm_conv_kernel) : 16'd0;
        conv_oy = (conv_out_w != 0) ? (conv_row_lin / conv_out_w) : 16'd0;
        conv_ox = (conv_out_w != 0) ? (conv_row_lin % conv_out_w) : 16'd0;
        conv_in_y = (conv_oy * mm_conv_stride) + conv_ky - mm_conv_padding;
        conv_in_x = (conv_ox * mm_conv_stride) + conv_kx - mm_conv_padding;
        conv_in_row_lin = (conv_in_y * mm_conv_in_w) + conv_in_x;
        conv_input_addr = mm_a_base
            + ((conv_in_row_lin / `ARRAY_SIZE) * (conv_c_tiles * `ARRAY_SIZE))
            + (conv_c_tile_idx * `ARRAY_SIZE)
            + (conv_in_row_lin % `ARRAY_SIZE);
        conv_row_valid = (conv_row_lin < conv_out_elems);

        conv_window_elems = mm_conv_kernel * mm_conv_kernel;
        conv_k_logical = (k_idx * `ARRAY_SIZE) + cycle_cnt;
        conv_k_channel = (conv_window_elems != 0) ? (conv_k_logical / conv_window_elems) : 16'd0;
        conv_k_channel_phys = conv_k_channel / conv_pack_factor;
        conv_k_subidx = conv_k_channel % conv_pack_factor;

        // Lane-wise gather metadata for conv_stream on raw matrix_hwc input.
        // This matches software im2col logical order: channel -> ky -> kx.
        conv_stream_gather_en = mm_conv_stream_en && (mm_conv_stride != 0) && (mm_conv_kernel != 0);
        conv_stream_in_precision = mm_in_precision;
        conv_stream_lane_valid = '0;
        for (int lane = 0; lane < `ARRAY_SIZE; lane++) begin
            conv_stream_lane_word_addr[lane] = '0;
            conv_stream_lane_word_lane[lane] = '0;
            conv_stream_lane_subidx[lane] = '0;
            if (conv_stream_gather_en && conv_window_elems != 0) begin
                conv_lane_row_lin = (m_idx * `ARRAY_SIZE) + lane;
                conv_lane_valid_i = (conv_lane_row_lin < conv_out_elems)
                    && (conv_k_logical < (conv_window_elems * mm_conv_in_c))
                    && (conv_k_channel < mm_conv_in_c);
                if (conv_lane_valid_i) begin
                    conv_lane_oy = (conv_out_w != 0) ? (conv_lane_row_lin / conv_out_w) : 16'd0;
                    conv_lane_ox = (conv_out_w != 0) ? (conv_lane_row_lin % conv_out_w) : 16'd0;
                    conv_lane_window_idx = conv_k_logical % conv_window_elems;
                    conv_lane_kx = conv_lane_window_idx % mm_conv_kernel;
                    conv_lane_ky = conv_lane_window_idx / mm_conv_kernel;
                    conv_lane_in_y_i = int'(conv_lane_oy) * int'(mm_conv_stride) + int'(conv_lane_ky) - int'(mm_conv_padding);
                    conv_lane_in_x_i = int'(conv_lane_ox) * int'(mm_conv_stride) + int'(conv_lane_kx) - int'(mm_conv_padding);
                    conv_lane_valid_i = (conv_lane_in_y_i >= 0) && (conv_lane_in_y_i < int'(mm_conv_in_h))
                        && (conv_lane_in_x_i >= 0) && (conv_lane_in_x_i < int'(mm_conv_in_w));
                end
                if (conv_lane_valid_i) begin
                    conv_lane_in_y = conv_lane_in_y_i[15:0];
                    conv_lane_in_x = conv_lane_in_x_i[15:0];
                    conv_lane_in_row_lin = (conv_lane_in_y * mm_conv_in_w) + conv_lane_in_x;
                    conv_lane_row_tile = conv_lane_in_row_lin / `ARRAY_SIZE;
                    conv_lane_row_lane = conv_lane_in_row_lin % `ARRAY_SIZE;
                    conv_lane_k_channel_phys = conv_k_channel_phys;
                    conv_lane_subidx = conv_k_subidx;
                    conv_lane_col_tile = conv_lane_k_channel_phys / `ARRAY_SIZE;
                    conv_lane_col_lane = conv_lane_k_channel_phys % `ARRAY_SIZE;
                    conv_stream_lane_word_addr[lane] = mm_a_base
                        + ((conv_lane_row_tile * conv_c_tiles + conv_lane_col_tile) * `ARRAY_SIZE)
                        + conv_lane_col_lane;
                    conv_stream_lane_word_lane[lane] = conv_lane_row_lane;
                    conv_stream_lane_subidx[lane] = conv_lane_subidx;
                    conv_stream_lane_valid[lane] = 1'b1;
                end
            end
        end
    end

    always_comb begin
        next_state = state;
        pc_next = pc;
        status_out = `STATUS_IDLE;
        im_wr_en = 1'b0;
        im_addr = '0;
        im_wdata = latched_mmvr;
        ub_req = 1'b0;
        ub_wr_en = 1'b0;
        ub_addr = '0;
        ub_w_addr = '0;
        ub_wdata = latched_mmvr;
        acc_clear = 1'b0;
        compute_enable = 1'b0;
        drain_enable = 1'b0;
        sa_input_first = 1'b0;
        sa_input_last = 1'b0;
        sa_weight_first = 1'b0;
        sa_weight_last = 1'b0;
        ppu_capture_en = 1'b0;
        ppu_cycle_idx = cycle_cnt;
        ppu_wb_en = 1'b0;
        ppu_bias_en = 1'b0;
        ppu_bias_clear = 1'b0;
        ppu_shift = mm_shift;
        ppu_multiplier = mm_multiplier;
        ppu_activation = mm_activation;
        ppu_h_gelu_x_scale_shift = mm_h_gelu_x_scale_shift;
        ppu_in_precision = mm_in_precision;
        ppu_out_precision = mm_out_precision;
        ppu_write_offset = (mm_output_layout == OUT_LAYOUT_A) ? 2'b0 : packed_write_offset;
        ppu_output_layout = mm_output_layout;
        mmvr_wr_en = 1'b0;
        mmvr_out = '0;

    move_src_next = move_src;
    move_dest_next = move_dest;
    move_count_next = move_count;
    move_phase_next = move_phase;
    m_next = m_idx;
    n_next = n_idx;
    k_next = k_idx;
    cycle_next = cycle_cnt;

    case (state)
      CTRL_IDLE: begin
        if (doorbell_pulse) begin
          if (cmd_in == `CMD_WRITE_MEM) next_state = CTRL_HOST_WRITE;
          else if (cmd_in == `CMD_READ_MEM) next_state = CTRL_HOST_READ;
          else if (cmd_in == `CMD_RUN) begin
            pc_next = arg_in[`ADDR_WIDTH-1:0];
            next_state = CTRL_FETCH;
          end
          status_out = `STATUS_BUSY;
        end
      end

      CTRL_HOST_WRITE: begin
        status_out = `STATUS_BUSY;
        if (latched_addr >= `IM_BASE_ADDR) begin
          im_wr_en = 1'b1;
          im_addr  = latched_addr;
        end else begin
          ub_req   = 1'b1;
          ub_wr_en = 1'b1;
          ub_addr  = latched_addr;
        end
        next_state = CTRL_IDLE;
      end

      CTRL_HOST_READ: begin
        status_out = `STATUS_BUSY;
        if (latched_addr >= `IM_BASE_ADDR) begin
          im_addr = latched_addr;
        end else begin
          ub_req  = 1'b1;
          ub_addr = latched_addr;
        end
        next_state = CTRL_READ_WAIT;
      end

      CTRL_READ_WAIT: begin
        status_out = doorbell_pulse ? `STATUS_BUSY : `STATUS_DATA_VALID;
        mmvr_wr_en = 1'b1;
        // Hold addresses to keep combinational data valid
        if (latched_addr >= `IM_BASE_ADDR) begin
          im_addr = latched_addr;
          mmvr_out = im_rdata;
        end else begin
          ub_req  = 1'b1;
          ub_addr = latched_addr;
          mmvr_out = ub_rdata;
        end
        
        if (doorbell_pulse) begin
          if (cmd_in == `CMD_WRITE_MEM) next_state = CTRL_HOST_WRITE;
          else if (cmd_in == `CMD_READ_MEM) next_state = CTRL_HOST_READ;
          else if (cmd_in == `CMD_RUN) begin
            pc_next = arg_in[`ADDR_WIDTH-1:0];
            next_state = CTRL_FETCH;
          end else begin
            next_state = CTRL_IDLE;
          end
        end
      end

      CTRL_FETCH: begin
        status_out = `STATUS_BUSY;
        im_addr = pc;
        next_state = CTRL_DECODE;
      end

      CTRL_DECODE: begin
        status_out = `STATUS_BUSY;
        case (im_rdata[255:252])
          ISA_OP_HALT: begin
            pc_next = pc + `INST_CHUNKS;
            next_state = CTRL_HALT;
          end
          ISA_OP_NOP: begin
            pc_next = pc + `INST_CHUNKS;
            next_state = CTRL_FETCH;
          end
          ISA_OP_MOVE: begin
            move_src_next = im_rdata[247:232];
            move_dest_next = im_rdata[231:216];
            move_count_next = im_rdata[215:200];
            move_phase_next = 1'b0;
            next_state = CTRL_EXEC_MOVE;
          end
          ISA_OP_MATMUL: begin
            ppu_bias_clear = 1'b1;
            next_state = CTRL_EXEC_MATMUL;
          end
          default: next_state = CTRL_HALT;
        endcase
      end

      CTRL_EXEC_MOVE: begin
        status_out = `STATUS_BUSY;
        ub_req = 1'b1;

        if (move_count == 0) begin
          pc_next = pc + `INST_CHUNKS;
          next_state = CTRL_FETCH;
        end else if (move_phase == 1'b0) begin
          ub_addr = move_src;
          move_phase_next = 1'b1;
        end else begin
          ub_addr = move_dest;
          ub_wdata = ub_rdata_reg;
          ub_wr_en = 1'b1;
          move_src_next = move_src + 1;
          move_dest_next = move_dest + 1;
          move_count_next = move_count - 1;
          move_phase_next = 1'b0;
        end
      end

      CTRL_EXEC_MATMUL: begin
        status_out = `STATUS_BUSY;
        if (m_idx >= mm_m_total) begin
          pc_next = pc + `INST_CHUNKS;
          next_state = CTRL_FETCH;
        end else if (n_idx >= mm_n_total) begin
          m_next = m_idx + 1;
          n_next = '0;
          ppu_bias_clear = 1'b1;
        end else begin
          next_state = CTRL_MM_CLEAR;
        end
      end

      CTRL_MM_CLEAR: begin
        status_out = `STATUS_BUSY;
        acc_clear = 1'b1;
        k_next = '0;
        cycle_next = '0;
        if (mm_bias_base != 16'hFFFF) begin
            next_state = CTRL_MM_LOAD_BIAS;
        end else begin
            next_state = CTRL_MM_FEED;
        end
      end

      CTRL_MM_LOAD_BIAS: begin
        status_out = `STATUS_BUSY;
        ub_req = 1'b1;
        ppu_bias_en = 1'b1;
        if (cycle_cnt == 0) begin
            ub_addr = mm_bias_base + (n_idx * 2);
            cycle_next = 1;
        end else begin
            ub_addr = mm_bias_base + (n_idx * 2) + 1;
            cycle_next = 0;
            next_state = CTRL_MM_FEED;
        end
      end

      CTRL_MM_FEED: begin
        status_out = `STATUS_BUSY;
        compute_enable = 1'b1;

        if (mm_conv_stream_en) begin
          // Conv-stream gather drives UB reads lane-wise. Keep Port-A address
          // parked on a valid base row to avoid out-of-range combinational
          // lookups when padding introduces negative coordinates.
          ub_addr = mm_a_base;
        end else begin
          ub_addr = mm_a_base + (m_idx * mm_k_total * `ARRAY_SIZE) + (k_idx * `ARRAY_SIZE) + cycle_cnt;
        end
        ub_w_addr = mm_b_base + (k_idx * mm_n_total * `ARRAY_SIZE) + (n_idx * `ARRAY_SIZE) + cycle_cnt;

        if (k_idx == 0 && cycle_cnt == 0) begin
          sa_input_first  = 1'b1;
          sa_weight_first = 1'b1;
        end
        if (k_idx == mm_k_total - 1 && cycle_cnt == (`ARRAY_SIZE - 1)) begin
          sa_input_last  = 1'b1;
          sa_weight_last = 1'b1;
        end

        if (cycle_cnt == (`ARRAY_SIZE - 1)) begin
          cycle_next = '0;
          if (k_idx == mm_k_total - 1) begin
            next_state = CTRL_MM_WAIT;
          end else begin
            k_next = k_idx + 1;
          end
        end else begin
          cycle_next = cycle_cnt + 1;
        end
      end

      CTRL_MM_WAIT: begin
        status_out = `STATUS_BUSY;
        compute_enable = 1'b1;  // Flush skewers
        
        if (all_done_in) begin
          cycle_next = '0;
          next_state = CTRL_MM_DRAIN_SA;
        end
      end

      CTRL_MM_DRAIN_SA: begin
        status_out = `STATUS_BUSY;
        drain_enable = 1'b1;
        ppu_capture_en = 1'b1;
        
        if (cycle_cnt == (`ARRAY_SIZE - 1)) begin
          cycle_next = '0;
          next_state = CTRL_MM_WRITEBACK;
        end else begin
          cycle_next = cycle_cnt + 1;
        end
      end

      CTRL_MM_WRITEBACK: begin
        status_out = `STATUS_BUSY;
        ub_req = 1'b1;
        ub_wr_en = wb_valid_cycle;
        ppu_wb_en = wb_valid_cycle;

        if (mm_output_layout == OUT_LAYOUT_A) begin
          ub_addr = mm_c_base + (m_idx * n_total_packed * `ARRAY_SIZE) + (n_idx_packed * `ARRAY_SIZE) + a_word_base + cycle_cnt;
        end else begin
          ub_addr = mm_c_base + (m_idx_packed * mm_n_total * `ARRAY_SIZE) + (n_idx * `ARRAY_SIZE) + cycle_cnt;
        end

        if (cycle_cnt == (`ARRAY_SIZE - 1)) begin
          cycle_next = '0;
          n_next = n_idx + 1;
          next_state = CTRL_EXEC_MATMUL;
        end else begin
          cycle_next = cycle_cnt + 1;
        end
      end

      CTRL_HALT: begin
        status_out = doorbell_pulse ? `STATUS_BUSY : `STATUS_HALTED;
        if (doorbell_pulse) begin
          if (cmd_in == `CMD_WRITE_MEM) next_state = CTRL_HOST_WRITE;
          else if (cmd_in == `CMD_READ_MEM) next_state = CTRL_HOST_READ;
          else if (cmd_in == `CMD_RUN) begin
            pc_next = arg_in[`ADDR_WIDTH-1:0];
            next_state = CTRL_FETCH;
          end else begin
            next_state = CTRL_IDLE;
          end
        end
      end

      default: next_state = CTRL_IDLE;
    endcase
  end
endmodule
