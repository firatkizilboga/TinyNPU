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
  logic xform_phase, xform_phase_next;
  xform_mode_t xform_mode, xform_mode_next;
  logic [15:0] xform_multiplier, xform_multiplier_next;
  logic [7:0] xform_shift, xform_shift_next;

  // --- ROPE Buffer Registers (used by XFORM_MODE_ROPE_K16) ---
  logic [`BUFFER_WIDTH-1:0] rope_k_lo_buf;    // K[0..half-1] latched from UB
  logic [`BUFFER_WIDTH-1:0] rope_k_hi_buf;    // K[half..d-1] latched from UB
  logic [`BUFFER_WIDTH-1:0] rope_cos_buf;     // cos table word latched from UB
  logic [`BUFFER_WIDTH-1:0] rope_k_hi_rot_buf; // rotated K_hi latched for phase-5 write

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
      {xform_src, xform_dest, xform_count, xform_phase} <= '0;
      xform_mode <= XFORM_MODE_NONE;
      xform_multiplier <= '0;
      xform_shift <= '0;
      {mm_a_base, mm_b_base, mm_c_base, mm_bias_base, mm_output_word_offset, mm_b_word_offset, mm_m_total, mm_k_total, mm_n_total} <= '0;
      {mm_shift, mm_multiplier, mm_activation, mm_h_gelu_x_scale_shift, mm_in_precision, mm_out_precision, mm_write_offset} <= '0;
      mm_output_layout <= OUT_LAYOUT_C;
      mm_writeback_mode <= WB_MODE_NORMAL;
      mm_b_read_mode <= B_READ_MODE_NORMAL;
      {m_idx, n_idx, k_idx, cycle_cnt} <= '0;
      {rope_k_lo_buf, rope_k_hi_buf, rope_cos_buf, rope_k_hi_rot_buf} <= '0;
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

      // ROPE_K16: latch UB reads into rotation buffers each phase
      // cycle_cnt here is the CURRENT phase (already updated to next value via cycle_next)
      // ub_rdata_reg captured this cycle holds data fetched in the previous phase's address.
      if (state == CTRL_EXEC_XFORM && xform_mode == XFORM_MODE_ROPE_K16) begin
        case (cycle_cnt)
          3'd1: rope_k_lo_buf <= ub_rdata_reg; // phase 0 addressed K_lo; latch now
          3'd2: rope_k_hi_buf <= ub_rdata_reg; // phase 1 addressed K_hi
          3'd3: rope_cos_buf  <= ub_rdata_reg; // phase 2 addressed cos
          3'd4: rope_k_hi_rot_buf <= rope_k_hi_rot_w; // sin in ub_rdata_reg; latch K_hi_rot
          default: ;
        endcase
      end

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
    logic [`BUFFER_WIDTH-1:0] xform_word_q_f16_i16;

    function automatic logic signed [63:0] round_shift_right_signed(
        input logic signed [63:0] value,
        input int unsigned shift
    );
        logic signed [63:0] abs_value;
        logic signed [63:0] rounded;
        begin
            if (shift == 0) begin
                round_shift_right_signed = value;
            end else if (shift >= 63) begin
                round_shift_right_signed = '0;
            end else if (value >= 0) begin
                round_shift_right_signed = (value + (64'sd1 <<< (shift - 1))) >>> shift;
            end else begin
                abs_value = -value;
                rounded = (abs_value + (64'sd1 <<< (shift - 1))) >>> shift;
                round_shift_right_signed = -rounded;
            end
        end
    endfunction

    function automatic logic signed [15:0] clip_i16(
        input logic signed [63:0] value
    );
        begin
            if (value > 64'sd32767) begin
                clip_i16 = 16'sh7fff;
            end else if (value < -64'sd32768) begin
                clip_i16 = 16'sh8000;
            end else begin
                clip_i16 = value[15:0];
            end
        end
    endfunction

    function automatic logic signed [15:0] quantize_lane_q_f16_i16(
        input logic [15:0] fp16,
        input logic [15:0] multiplier,
        input logic [7:0] shift
    );
        logic sign;
        logic [4:0] exp_bits;
        logic [9:0] frac_bits;
        logic signed [63:0] mant;
        logic signed [63:0] scaled;
        logic signed [63:0] qvalue;
        int exp2;
        int left_shift;
        int right_shift;
        begin
            sign = fp16[15];
            exp_bits = fp16[14:10];
            frac_bits = fp16[9:0];
            qvalue = 64'sd0;
            if (multiplier == 16'd0) begin
                quantize_lane_q_f16_i16 = 16'sd0;
            end else if (exp_bits == 5'h1f) begin
                quantize_lane_q_f16_i16 = sign ? -16'sd32768 : 16'sd32767;
            end else if (exp_bits == 5'd0 && frac_bits == 10'd0) begin
                quantize_lane_q_f16_i16 = 16'sd0;
            end else begin
                if (exp_bits == 5'd0) begin
                    // subnormal: frac * 2^-24
                    mant = $signed({2'b00, frac_bits});
                    exp2 = -24;
                end else begin
                    // normal: (1024 + frac) * 2^(exp-25)
                    mant = $signed({2'b01, frac_bits});
                    exp2 = $signed({1'b0, exp_bits}) - 25;
                end

                scaled = mant * $signed({1'b0, multiplier});
                if (exp2 >= $signed({1'b0, shift})) begin
                    left_shift = exp2 - $signed({1'b0, shift});
                    if (left_shift >= 47) begin
                        qvalue = 64'sh7fffffffffffffff;
                    end else begin
                        qvalue = scaled <<< left_shift;
                    end
                end else begin
                    right_shift = $signed({1'b0, shift}) - exp2;
                    qvalue = round_shift_right_signed(scaled, right_shift);
                end

                if (sign) begin
                    qvalue = -qvalue;
                end
                quantize_lane_q_f16_i16 = clip_i16(qvalue);
            end
        end
    endfunction

    always_comb begin
        xform_word_q_f16_i16 = ub_rdata_reg;
        for (int lane = 0; lane < `ARRAY_SIZE; lane++) begin
            xform_word_q_f16_i16[lane*16 +: 16] =
                quantize_lane_q_f16_i16(
                    ub_rdata_reg[lane*16 +: 16],
                    xform_multiplier,
                    xform_shift
                );
        end
    end

    // --- RoPE INT16 Q14 rotation combinational ---
    // Valid during CTRL_EXEC_XFORM phase 4 (ub_rdata_reg = sin word).
    // rope_k_lo_rot_w[j] = clip_i16((K_lo[j]*cos[j] - K_hi[j]*sin[j]) >> 14)
    // rope_k_hi_rot_w[j] = clip_i16((K_hi[j]*cos[j] + K_lo[j]*sin[j]) >> 14)
    logic [`BUFFER_WIDTH-1:0] rope_k_lo_rot_w;
    logic [`BUFFER_WIDTH-1:0] rope_k_hi_rot_w;

    always_comb begin
        rope_k_lo_rot_w = '0;
        rope_k_hi_rot_w = '0;
        for (int rj = 0; rj < `ARRAY_SIZE; rj++) begin
            automatic logic signed [15:0] klo_v, khi_v, cos_v, sin_v;
            automatic logic signed [31:0] lo_prod, hi_prod;
            klo_v   = $signed(rope_k_lo_buf[rj*16 +: 16]);
            khi_v   = $signed(rope_k_hi_buf[rj*16 +: 16]);
            cos_v   = $signed(rope_cos_buf[rj*16 +: 16]);
            sin_v   = $signed(ub_rdata_reg[rj*16 +: 16]); // sin valid in phase 4
            lo_prod = klo_v * cos_v - khi_v * sin_v;
            hi_prod = khi_v * cos_v + klo_v * sin_v;
            rope_k_lo_rot_w[rj*16 +: 16] = clip_i16($signed(lo_prod) >>> 14);
            rope_k_hi_rot_w[rj*16 +: 16] = clip_i16($signed(hi_prod) >>> 14);
        end
    end

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
        ppu_write_offset = (mm_output_layout == OUT_LAYOUT_C) ? packed_write_offset : 2'b0;
        ppu_output_layout = mm_output_layout;
        ppu_writeback_mode = mm_writeback_mode;
        ppu_cache_lane_idx = cache_lane_idx;
        mmvr_wr_en = 1'b0;
        mmvr_out = '0;

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
            xform_phase_next = 1'b0;
            k_next = '0;      // pair index for ROPE_K16
            cycle_next = '0;  // phase index for ROPE_K16
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
          ub_wdata = ub_rdata_reg;
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

        if (xform_mode == XFORM_MODE_ROPE_K16) begin
          // 6-phase per K-pair: k_idx = pair index (0..half_count-1)
          // xform_count = half_count, xform_multiplier = cs_base (cos/sin table addr)
          // Both K and cos/sin use stride-8 (one valid word per ARRAY_SIZE-word tile).
          // K stored as OUT_LAYOUT_C for m=1: valid word at src + n_tile*8 + 0.
          // cos/sin stored as C-layout [1, d_head] with data[0, 0:half]=cos, data[0, half:d]=sin:
          //   cos[k] at cs + k*8, sin[k] at cs + (k + half_count)*8
          if (k_idx >= xform_count) begin
            pc_next = pc + `INST_CHUNKS;
            next_state = CTRL_FETCH;
          end else begin
            case (cycle_cnt)
              3'd0: begin // read K_lo tile (stride-8)
                ub_addr = xform_src + (k_idx << 3);
                cycle_next = 3'd1;
              end
              3'd1: begin // K_lo in ub_rdata_reg; read K_hi tile
                ub_addr = xform_src + ((k_idx + xform_count) << 3);
                cycle_next = 3'd2;
              end
              3'd2: begin // K_hi in ub_rdata_reg; read cos word (stride-8)
                ub_addr = xform_multiplier + (k_idx << 3);
                cycle_next = 3'd3;
              end
              3'd3: begin // cos in ub_rdata_reg; read sin word (stride-8)
                ub_addr = xform_multiplier + ((k_idx + xform_count) << 3);
                cycle_next = 3'd4;
              end
              3'd4: begin // sin is now in ub_rdata_reg; write rotated K_lo (stride-8)
                ub_addr = xform_dest + (k_idx << 3);
                ub_wr_en = 1'b1;
                ub_wdata = rope_k_lo_rot_w;
                cycle_next = 3'd5;
              end
              3'd5: begin // write rotated K_hi (stride-8, pre-latched)
                ub_addr = xform_dest + ((k_idx + xform_count) << 3);
                ub_wr_en = 1'b1;
                ub_wdata = rope_k_hi_rot_buf;
                k_next = k_idx + 1;
                cycle_next = 3'd0;
              end
              default: cycle_next = 3'd0;
            endcase
          end
        end else if (xform_count == 0) begin
          pc_next = pc + `INST_CHUNKS;
          next_state = CTRL_FETCH;
        end else if (xform_phase == 1'b0) begin
          ub_addr = xform_src;
          xform_phase_next = 1'b1;
        end else begin
          ub_addr = xform_dest;
          ub_wr_en = 1'b1;
          unique case (xform_mode)
            XFORM_MODE_Q_F16_I16: ub_wdata = xform_word_q_f16_i16;
            default: ub_wdata = ub_rdata_reg;
          endcase
          xform_src_next = xform_src + 1;
          xform_dest_next = xform_dest + 1;
          xform_count_next = xform_count - 1;
          xform_phase_next = 1'b0;
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
