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
    output writeback_mode_t                ppu_writeback_mode,
    output logic [$clog2(`ARRAY_SIZE)-1:0] ppu_cache_lane_idx,
    input  logic                           ppu_busy,
    input  logic                           ppu_done,

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
    CTRL_EXEC_XFORM,
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
  localparam int CTRL_STATE_COUNT = 16;
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

  // --- Internal Registers for XFORM ---
  logic [`ADDR_WIDTH-1:0] xform_src, xform_src_next;
  logic [`ADDR_WIDTH-1:0] xform_dest, xform_dest_next;
  logic [`ADDR_WIDTH-1:0] xform_count, xform_count_next;
  logic [2:0] xform_phase, xform_phase_next;
  xform_mode_t xform_mode, xform_mode_next;
  logic [15:0] xform_multiplier, xform_multiplier_next;
  logic [7:0] xform_shift, xform_shift_next;

  // --- Registered XFORM unit ---
  // Hardware XFORM is intentionally limited to word-local Q/DQ transforms.
  // RoPE is a host/compiler responsibility because it needs cross-word pairing
  // and trigonometric tables that do not fit the simple streaming datapath.
  logic xform_unit_start;
  logic xform_unit_done;
  logic [`BUFFER_WIDTH-1:0] xform_unit_in_word;
  logic [`BUFFER_WIDTH-1:0] xform_unit_in_word_hi;
  logic [`BUFFER_WIDTH-1:0] xform_unit_out_word;
  logic [`BUFFER_WIDTH-1:0] xform_unit_out_word_hi;
  logic [`BUFFER_WIDTH-1:0] xform_tmp_word, xform_tmp_word_next;

  // --- Internal Registers for MATMUL ---
  logic [`ADDR_WIDTH-1:0] mm_a_base, mm_b_base, mm_c_base, mm_bias_base;
  logic [`ADDR_WIDTH-1:0] mm_output_word_offset;
  logic [`ADDR_WIDTH-1:0] mm_b_word_offset;
  b_read_mode_t mm_b_read_mode;
  logic [15:0] mm_m_total, mm_k_total, mm_n_total;
  logic [ 7:0] mm_shift;
  logic [15:0] mm_multiplier;
  logic [ 7:0] mm_activation;
  logic [ 7:0] mm_h_gelu_x_scale_shift;
  logic [ 1:0] mm_in_precision;
  logic [ 1:0] mm_out_precision;
  logic [ 1:0] mm_write_offset;
  output_layout_t mm_output_layout;
  writeback_mode_t mm_writeback_mode;
  logic [15:0] m_idx, n_idx, k_idx;

  // Flexible counter width for NxN support plus PPU pipeline drain wait.
  localparam int CYCLE_CNT_WIDTH = $clog2(`ARRAY_SIZE + 16);
  logic [CYCLE_CNT_WIDTH-1:0] cycle_cnt, cycle_next;

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
      {xform_src, xform_dest, xform_count, xform_phase} <= '0;
      xform_mode <= XFORM_MODE_NONE;
      xform_multiplier <= '0;
      xform_shift <= '0;
      xform_tmp_word <= '0;
      {mm_a_base, mm_b_base, mm_c_base, mm_bias_base, mm_output_word_offset, mm_b_word_offset, mm_m_total, mm_k_total, mm_n_total} <= '0;
      {mm_shift, mm_multiplier, mm_activation, mm_h_gelu_x_scale_shift, mm_in_precision, mm_out_precision, mm_write_offset} <= '0;
      mm_output_layout <= OUT_LAYOUT_C;
      mm_writeback_mode <= WB_MODE_NORMAL;
      mm_b_read_mode <= B_READ_MODE_NORMAL;
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

      // XFORM Update
      xform_src <= xform_src_next;
      xform_dest <= xform_dest_next;
      xform_count <= xform_count_next;
      xform_phase <= xform_phase_next;
      xform_mode <= xform_mode_next;
      xform_multiplier <= xform_multiplier_next;
      xform_shift <= xform_shift_next;
      xform_tmp_word <= xform_tmp_word_next;

      // MATMUL Latch from Instruction Memory
      if (state == CTRL_DECODE && im_rdata[255:252] == ISA_OP_MATMUL) begin
        mm_writeback_mode <= writeback_mode_t'(im_rdata[251:248]);
        mm_a_base <= im_rdata[247:232];
        mm_b_base <= im_rdata[231:216];
        mm_c_base <= im_rdata[215:200];  // Load C Base
        mm_output_word_offset <= im_rdata[199:184];
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
        mm_b_word_offset <= im_rdata[71:56];
        mm_b_read_mode <= b_read_mode_t'(im_rdata[55:52]);
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

    // Packed writeback: A layout compresses the N dimension, B layout compresses
    // the M dimension, and C layout uses packed write_offset updates in-place.
    logic [ 1:0] packed_write_offset;
    logic [15:0] m_idx_packed;
    logic [15:0] m_total_packed;
    logic [15:0] n_idx_packed;
    logic [15:0] n_total_packed;
    logic [$clog2(`ARRAY_SIZE)-1:0]   a_word_base;
    logic [$clog2(`ARRAY_SIZE)-1:0]   b_word_base;
    logic [$clog2(`ARRAY_SIZE+1)-1:0] a_words_per_tile;
    logic [$clog2(`ARRAY_SIZE+1)-1:0] b_words_per_tile;
    logic        wb_valid_cycle;
    logic [$clog2(`ARRAY_SIZE)-1:0] cache_lane_idx;
    logic [`ADDR_WIDTH-1:0] mm_c_effective_base;
    logic        packed_c_writeback;
    logic [15:0] packed_c_group_base;
    logic [15:0] packed_c_group_limit;
    xform_unit u_xform (
        .clk(clk),
        .rst_n(rst_n),
        .start(xform_unit_start),
        .mode(xform_mode),
        .in_word(xform_unit_in_word),
        .in_word_hi(xform_unit_in_word_hi),
        .multiplier(xform_multiplier),
        .shift(xform_shift),
        .done(xform_unit_done),
        .out_word(xform_unit_out_word),
        .out_word_hi(xform_unit_out_word_hi)
    );

    always_comb begin
        unique case (mm_out_precision)
            2'b00: begin // INT4: 4 tiles per word
                m_idx_packed     = m_idx >> 2;
                packed_write_offset = m_idx[1:0];
                m_total_packed   = (mm_m_total + 16'd3) >> 2;
                n_idx_packed     = n_idx >> 2;
                n_total_packed   = (mm_n_total + 16'd3) >> 2;
                a_word_base      = (n_idx[1:0] * (`ARRAY_SIZE / 4));
                b_word_base      = (m_idx[1:0] * (`ARRAY_SIZE / 4));
                a_words_per_tile = (`ARRAY_SIZE / 4);
                b_words_per_tile = (`ARRAY_SIZE / 4);
            end
            2'b01: begin // INT8: 2 tiles per word
                m_idx_packed     = m_idx >> 1;
                packed_write_offset = {1'b0, m_idx[0]};
                m_total_packed   = (mm_m_total + 16'd1) >> 1;
                n_idx_packed     = n_idx >> 1;
                n_total_packed   = (mm_n_total + 16'd1) >> 1;
                a_word_base      = (n_idx[0] * (`ARRAY_SIZE / 2));
                b_word_base      = (m_idx[0] * (`ARRAY_SIZE / 2));
                a_words_per_tile = (`ARRAY_SIZE / 2);
                b_words_per_tile = (`ARRAY_SIZE / 2);
            end
            default: begin // INT16: 1 tile per word
                m_idx_packed     = m_idx;
                packed_write_offset = 2'b0;
                m_total_packed   = mm_m_total;
                n_idx_packed     = n_idx;
                n_total_packed   = mm_n_total;
                a_word_base      = '0;
                b_word_base      = '0;
                a_words_per_tile = `ARRAY_SIZE;
                b_words_per_tile = `ARRAY_SIZE;
            end
        endcase
        cache_lane_idx = mm_output_word_offset[$clog2(`ARRAY_SIZE)-1:0];
        unique case (mm_output_layout)
            OUT_LAYOUT_A: wb_valid_cycle = (cycle_cnt < a_words_per_tile);
            OUT_LAYOUT_B: wb_valid_cycle = (cycle_cnt < b_words_per_tile);
            default:      wb_valid_cycle = 1'b1;
        endcase
        if (mm_writeback_mode == WB_MODE_V_CACHE_APPEND_INT16) begin
            wb_valid_cycle = (mm_out_precision == MODE_INT16) && (cycle_cnt == cache_lane_idx);
        end
        mm_c_effective_base = mm_c_base + mm_output_word_offset;

        packed_c_writeback = (mm_writeback_mode == WB_MODE_NORMAL) &&
                             (mm_output_layout == OUT_LAYOUT_C) &&
                             (mm_out_precision != MODE_INT16);
        unique case (mm_out_precision)
            MODE_INT4: begin
                packed_c_group_base  = {m_idx[15:2], 2'b00};
                packed_c_group_limit = {m_idx[15:2], 2'b00} + 16'd4;
            end
            MODE_INT8: begin
                packed_c_group_base  = {m_idx[15:1], 1'b0};
                packed_c_group_limit = {m_idx[15:1], 1'b0} + 16'd2;
            end
            default: begin
                packed_c_group_base  = m_idx;
                packed_c_group_limit = m_idx + 16'd1;
            end
        endcase
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
        ppu_cycle_idx = cycle_cnt[$clog2(`ARRAY_SIZE)-1:0];
        ppu_wb_en = 1'b0;
        ppu_bias_en = 1'b0;
        ppu_bias_clear = 1'b0;
        ppu_shift = mm_shift;
        ppu_multiplier = mm_multiplier;
        ppu_activation = mm_activation;
        ppu_h_gelu_x_scale_shift = mm_h_gelu_x_scale_shift;
        ppu_in_precision = mm_in_precision;
        ppu_out_precision = mm_out_precision;
        ppu_write_offset = (mm_output_layout == OUT_LAYOUT_C) ? packed_write_offset : 2'b0;
        ppu_output_layout = mm_output_layout;
        ppu_writeback_mode = mm_writeback_mode;
        ppu_cache_lane_idx = cache_lane_idx;
        mmvr_wr_en = 1'b0;
        mmvr_out = '0;
        xform_unit_start = 1'b0;
        xform_unit_in_word = ub_rdata;
        xform_unit_in_word_hi = '0;

    move_src_next = move_src;
    move_dest_next = move_dest;
    move_count_next = move_count;
    move_phase_next = move_phase;
    xform_src_next = xform_src;
    xform_dest_next = xform_dest;
    xform_count_next = xform_count;
    xform_phase_next = xform_phase;
    xform_mode_next = xform_mode;
    xform_multiplier_next = xform_multiplier;
    xform_shift_next = xform_shift;
    xform_tmp_word_next = xform_tmp_word;
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
          ISA_OP_XFORM: begin
            xform_mode_next = xform_mode_t'(im_rdata[251:248]);
            xform_src_next = im_rdata[247:232];
            xform_dest_next = im_rdata[231:216];
            xform_count_next = im_rdata[215:200];
            xform_multiplier_next = im_rdata[199:184];
            xform_shift_next = im_rdata[183:176];
            xform_phase_next = 3'd0;
            next_state = CTRL_EXEC_XFORM;
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
          ub_wdata = ub_rdata;
          ub_wr_en = 1'b1;
          move_src_next = move_src + 1;
          move_dest_next = move_dest + 1;
          move_count_next = move_count - 1;
          move_phase_next = 1'b0;
        end
      end

      CTRL_EXEC_XFORM: begin
        status_out = `STATUS_BUSY;
        ub_req = 1'b1;

`ifdef TINYNPU_DISABLE_XFORM
        pc_next = pc + `INST_CHUNKS;
        next_state = CTRL_FETCH;
`else
        if (xform_count == 0) begin
          pc_next = pc + `INST_CHUNKS;
          next_state = CTRL_FETCH;
        end else begin
          if (xform_mode == XFORM_MODE_Q_F32_I16) begin
            unique case (xform_phase)
              3'd0: begin
                ub_addr = xform_src;
                xform_phase_next = 3'd1;
              end
              3'd1: begin
                xform_tmp_word_next = ub_rdata;
                ub_addr = xform_src + 1;
                xform_phase_next = 3'd2;
              end
              3'd2: begin
                xform_unit_in_word = xform_tmp_word;
                xform_unit_in_word_hi = ub_rdata;
                xform_unit_start = 1'b1;
                xform_phase_next = 3'd3;
              end
              3'd3: begin
                if (xform_unit_done) begin
                  ub_addr = xform_dest;
                  ub_wr_en = 1'b1;
                  ub_wdata = xform_unit_out_word;
                  xform_src_next = xform_src + 2;
                  xform_dest_next = xform_dest + 1;
                  xform_count_next = xform_count - 1;
                  xform_phase_next = 3'd0;
                end
              end
              default: xform_phase_next = 3'd0;
            endcase
          end else if (xform_mode == XFORM_MODE_DQ_I16_F32) begin
            unique case (xform_phase)
              3'd0: begin
                ub_addr = xform_src;
                xform_phase_next = 3'd1;
              end
              3'd1: begin
                xform_unit_in_word = ub_rdata;
                xform_unit_start = 1'b1;
                xform_phase_next = 3'd2;
              end
              3'd2: begin
                if (xform_unit_done) begin
                  ub_addr = xform_dest;
                  ub_wr_en = 1'b1;
                  ub_wdata = xform_unit_out_word;
                  xform_phase_next = 3'd3;
                end
              end
              3'd3: begin
                ub_addr = xform_dest + 1;
                ub_wr_en = 1'b1;
                ub_wdata = xform_unit_out_word_hi;
                xform_src_next = xform_src + 1;
                xform_dest_next = xform_dest + 2;
                xform_count_next = xform_count - 1;
                xform_phase_next = 3'd0;
              end
              default: xform_phase_next = 3'd0;
            endcase
          end else begin
            pc_next = pc + `INST_CHUNKS;
            next_state = CTRL_FETCH;
          end
        end
`endif
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
        if (cycle_cnt == 0) begin
            // Issue first synchronous bias read.
            ub_addr = mm_bias_base + (n_idx * 2);
            cycle_next = 1;
        end else if (cycle_cnt == 1) begin
            // Capture first bias word while issuing the second read.
            ppu_bias_en = 1'b1;
            ub_addr = mm_bias_base + (n_idx * 2) + 1;
            cycle_next = 2;
        end else begin
            // Capture second bias word; both bias reads are now resident in PPU.
            ppu_bias_en = 1'b1;
            cycle_next = 0;
            next_state = CTRL_MM_FEED;
        end
      end

      CTRL_MM_FEED: begin
        status_out = `STATUS_BUSY;
        compute_enable = 1'b1;
        ub_addr = mm_a_base + (m_idx * mm_k_total * `ARRAY_SIZE) + (k_idx * `ARRAY_SIZE) + cycle_cnt;
        if (mm_b_read_mode == B_READ_MODE_K_CACHE_INT16) begin
          ub_w_addr = mm_b_base + mm_b_word_offset + (n_idx * mm_k_total * `ARRAY_SIZE) + (k_idx * `ARRAY_SIZE) + cycle_cnt;
        end else begin
          ub_w_addr = mm_b_base + mm_b_word_offset + (k_idx * mm_n_total * `ARRAY_SIZE) + (n_idx * `ARRAY_SIZE) + cycle_cnt;
        end

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
        drain_enable = (cycle_cnt < `ARRAY_SIZE);
        ppu_capture_en = (cycle_cnt < `ARRAY_SIZE);
        
        if (cycle_cnt >= `ARRAY_SIZE && ppu_done) begin
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

        if (mm_writeback_mode == WB_MODE_V_CACHE_APPEND_INT16) begin
          ub_addr = mm_c_effective_base + (n_idx * `ARRAY_SIZE);
        end else if (mm_writeback_mode == WB_MODE_K_CACHE_APPEND_INT16) begin
          ub_addr = mm_c_base + (mm_output_word_offset & ~(`ARRAY_SIZE - 1)) + (n_idx * `ARRAY_SIZE) + cycle_cnt;
        end else begin
          unique case (mm_output_layout)
            OUT_LAYOUT_A: begin
              ub_addr = mm_c_effective_base + (m_idx * n_total_packed * `ARRAY_SIZE) + (n_idx_packed * `ARRAY_SIZE) + a_word_base + cycle_cnt;
            end
            OUT_LAYOUT_B: begin
              ub_addr = mm_c_effective_base + (m_idx_packed * mm_n_total * `ARRAY_SIZE) + (n_idx * `ARRAY_SIZE) + b_word_base + cycle_cnt;
            end
            default: begin
              ub_addr = mm_c_effective_base + (m_idx_packed * mm_n_total * `ARRAY_SIZE) + (n_idx * `ARRAY_SIZE) + cycle_cnt;
            end
          endcase
        end

        if (cycle_cnt == (`ARRAY_SIZE - 1)) begin
          cycle_next = '0;
          if (packed_c_writeback) begin
            if (((m_idx + 16'd1) < mm_m_total) && ((m_idx + 16'd1) < packed_c_group_limit)) begin
              // Keep the same N tile while producing the next packed M lane.
              m_next = m_idx + 16'd1;
            end else if ((n_idx + 16'd1) < mm_n_total) begin
              // Finish this packed M group across all N tiles before moving on.
              m_next = packed_c_group_base;
              n_next = n_idx + 16'd1;
            end else begin
              m_next = packed_c_group_limit;
              n_next = '0;
            end
          end else begin
            n_next = n_idx + 1;
          end
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
