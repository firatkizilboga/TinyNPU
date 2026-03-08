`include "defines.sv"

module control_unit (
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
    output logic [ 1:0]                    ppu_in_precision,
    output logic [ 1:0]                    ppu_out_precision,
    output logic [ 1:0]                    ppu_write_offset,

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
  logic [ 1:0] mm_in_precision;
  logic [ 1:0] mm_out_precision;
  logic [ 1:0] mm_write_offset;
  logic [15:0] m_idx, n_idx, k_idx;

  // Flexible counter width for NxN support
  logic [$clog2(`ARRAY_SIZE)-1:0] cycle_cnt, cycle_next;

  logic [15:0] m_next, n_next, k_next;

  // --- Sequential State ---
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state <= CTRL_IDLE;
      pc    <= '0;
      ub_rdata_reg <= '0;
      {latched_cmd, latched_addr, latched_mmvr, latched_arg} <= '0;
      {move_src, move_dest, move_count, move_phase} <= '0;
      {mm_a_base, mm_b_base, mm_c_base, mm_bias_base, mm_m_total, mm_k_total, mm_n_total} <= '0;
      {mm_shift, mm_multiplier, mm_activation, mm_in_precision, mm_out_precision, mm_write_offset} <= '0;
      {m_idx, n_idx, k_idx, cycle_cnt} <= '0;
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
        m_idx <= '0;
        n_idx <= '0;
        k_idx <= '0;
        cycle_cnt <= '0;
        ppu_bias_clear <= 1'b1;
      end else begin
        m_idx <= m_next;
        n_idx <= n_next;
        k_idx <= k_next;
        cycle_cnt <= cycle_next;
      end
    end
  end

    // Tile packing: derive physical address and byte lane from m_idx and precision.
    // This aligns Row-Major Output C with Row-Major Input B for chaining.
    logic [ 1:0] packed_write_offset;
    logic [15:0] m_idx_packed;
    logic [15:0] m_total_packed;

    always_comb begin
        unique case (mm_out_precision)
            2'b00: begin // INT4: 4 tiles per word
                m_idx_packed     = m_idx >> 2;
                packed_write_offset = m_idx[1:0];
                m_total_packed   = (mm_m_total + 16'd3) >> 2;
            end
            2'b01: begin // INT8: 2 tiles per word
                m_idx_packed     = m_idx >> 1;
                packed_write_offset = {1'b0, m_idx[0]};
                m_total_packed   = (mm_m_total + 16'd1) >> 1;
            end
            default: begin // INT16: 1 tile per word
                m_idx_packed     = m_idx;
                packed_write_offset = 2'b0;
                m_total_packed   = mm_m_total;
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
        ppu_cycle_idx = cycle_cnt;
        ppu_wb_en = 1'b0;
        ppu_bias_en = 1'b0;
        ppu_bias_clear = 1'b0;
        ppu_shift = mm_shift;
        ppu_multiplier = mm_multiplier;
        ppu_activation = mm_activation;
        ppu_in_precision = mm_in_precision;
        ppu_out_precision = mm_out_precision;
        ppu_write_offset = packed_write_offset;
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
          ISA_OP_MATMUL: next_state = CTRL_EXEC_MATMUL;
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

        ub_addr   = mm_a_base + (m_idx * mm_k_total * `ARRAY_SIZE) + (k_idx * `ARRAY_SIZE) + cycle_cnt;
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

      CTRL_MM_LOAD_BIAS: begin
        status_out = `STATUS_BUSY;
        cycle_next = '0;
        next_state = CTRL_MM_DRAIN_SA;
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
        ub_wr_en = 1'b1;
        ppu_wb_en = 1'b1;

        // Writeback Address: pack consecutive M-tiles into the same physical word.
        // This makes output compatible with Input B (Top Matrix) packing for Layer 2.
        ub_addr = mm_c_base + (m_idx_packed * mm_n_total * `ARRAY_SIZE) + (n_idx * `ARRAY_SIZE) + cycle_cnt;

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
