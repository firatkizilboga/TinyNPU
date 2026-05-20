`include "defines.sv"

// Synthesis-only integration top for the CPU+NPU system.
//
// The Verilator testbench integrates the NPU through mm_ram.sv, but that file
// also contains simulation peripherals and random-stall logic. This wrapper
// keeps the real CPU and NPU RTL in one synthesis hierarchy while replacing the
// program/data memories with minimal abstract responses.
module cv32e40p_tinynpu_synth_top (
    input  logic clk_i,
    input  logic rst_ni,
    input  logic fetch_enable_i,
    input  logic [31:0] instr_rdata_i,
    output logic core_sleep_o,
    output logic npu_result_valid_o,
    output logic npu_all_done_o,
    output logic [31:0] observe_o
);

  localparam logic [31:0] NPU_BASE_ADDR = 32'h3000_0000;
  localparam logic [31:0] NPU_SHARED_UB_BASE_ADDR = 32'h3100_0000;
  localparam logic [31:0] NPU_SHARED_IM_BASE_ADDR = 32'h3200_0000;
  localparam int unsigned NPU_MMIO_SIZE = 32;
  localparam int unsigned NPU_SHARED_WORDS = 32768;
  localparam int unsigned NPU_SHARED_SIZE = NPU_SHARED_WORDS * 16;
  localparam logic [15:0] NPU_IM_BASE_WORD_ADDR = 16'h9000;

  logic        instr_req;
  logic        instr_rvalid_q;
  logic [31:0] instr_addr;

  logic        data_req;
  logic        data_we;
  logic [ 3:0] data_be;
  logic [31:0] data_addr;
  logic [31:0] data_wdata;
  logic [31:0] data_rdata_q;
  logic        data_rvalid_q;

  logic [31:0] irq_lines;
  logic        irq_ack;
  logic [ 4:0] irq_id;
  logic        debug_havereset;
  logic        debug_running;
  logic        debug_halted;

  logic [ 4:0] npu_host_addr;
  logic [ 7:0] npu_host_wr_data;
  logic        npu_host_wr_en;
  logic [ 7:0] npu_host_rd_data;
  logic [15:0] npu_shared_addr;
  logic [ 1:0] npu_shared_lane;
  logic [31:0] npu_shared_wr_data;
  logic [ 3:0] npu_shared_wr_be;
  logic        npu_shared_wr_en;
  logic        npu_shared_rd_en;
  logic [31:0] npu_shared_rd_data;
  logic        npu_shared_allow;
  logic        npu_result_valid;
  logic        npu_all_done;

  function automatic logic is_npu_addr(input logic [31:0] addr);
    return (addr >= NPU_BASE_ADDR) && (addr < (NPU_BASE_ADDR + NPU_MMIO_SIZE));
  endfunction

  function automatic logic is_npu_shared_ub_addr(input logic [31:0] addr);
    return (addr >= NPU_SHARED_UB_BASE_ADDR) && (addr < (NPU_SHARED_UB_BASE_ADDR + NPU_SHARED_SIZE));
  endfunction

  function automatic logic is_npu_shared_im_addr(input logic [31:0] addr);
    return (addr >= NPU_SHARED_IM_BASE_ADDR) && (addr < (NPU_SHARED_IM_BASE_ADDR + NPU_SHARED_SIZE));
  endfunction

  function automatic logic is_npu_shared_addr(input logic [31:0] addr);
    return is_npu_shared_ub_addr(addr) || is_npu_shared_im_addr(addr);
  endfunction

  function automatic logic [7:0] selected_write_byte(
      input logic [31:0] wdata,
      input logic [31:0] addr
  );
    case (addr[1:0])
      2'd0: selected_write_byte = wdata[7:0];
      2'd1: selected_write_byte = wdata[15:8];
      2'd2: selected_write_byte = wdata[23:16];
      default: selected_write_byte = wdata[31:24];
    endcase
  endfunction

  (* keep_hierarchy = "yes" *) cv32e40p_top #(
      .COREV_PULP      (0),
      .COREV_CLUSTER   (0),
      .FPU             (0),
      .FPU_ADDMUL_LAT  (0),
      .FPU_OTHERS_LAT  (0),
      .ZFINX           (0),
      .NUM_MHPMCOUNTERS(1)
  ) u_cpu (
      .clk_i (clk_i),
      .rst_ni(rst_ni),

      .pulp_clock_en_i(1'b1),
      .scan_cg_en_i   (1'b0),

      .boot_addr_i        (32'h0000_0180),
      .mtvec_addr_i       (32'h0000_0000),
      .dm_halt_addr_i     (32'h1A11_0800),
      .hart_id_i          (32'h0000_0000),
      .dm_exception_addr_i(32'h0000_0000),

      .instr_req_o   (instr_req),
      .instr_gnt_i   (instr_req),
      .instr_rvalid_i(instr_rvalid_q),
      .instr_addr_o  (instr_addr),
      .instr_rdata_i (instr_rdata_i),

      .data_req_o   (data_req),
      .data_gnt_i   (data_req),
      .data_rvalid_i(data_rvalid_q),
      .data_we_o    (data_we),
      .data_be_o    (data_be),
      .data_addr_o  (data_addr),
      .data_wdata_o (data_wdata),
      .data_rdata_i (data_rdata_q),

      .irq_i    (irq_lines),
      .irq_ack_o(irq_ack),
      .irq_id_o (irq_id),

      .debug_req_i      (1'b0),
      .debug_havereset_o(debug_havereset),
      .debug_running_o  (debug_running),
      .debug_halted_o   (debug_halted),

      .fetch_enable_i(fetch_enable_i),
      .core_sleep_o  (core_sleep_o)
  );

  (* keep_hierarchy = "yes", dont_touch = "yes" *) tinynpu_top u_tinynpu (
      .clk                (clk_i),
      .rst_n              (rst_ni),
      .host_addr          (npu_host_addr),
      .host_wr_data       (npu_host_wr_data),
      .host_wr_en         (npu_host_wr_en),
      .host_rd_data       (npu_host_rd_data),
      .host_shared_addr   (npu_shared_addr),
      .host_shared_lane   (npu_shared_lane),
      .host_shared_wr_data(npu_shared_wr_data),
      .host_shared_wr_be  (npu_shared_wr_be),
      .host_shared_wr_en  (npu_shared_wr_en),
      .host_shared_rd_en  (npu_shared_rd_en),
      .host_shared_rd_data(npu_shared_rd_data),
      .host_shared_allow  (npu_shared_allow),
      .result_valid       (npu_result_valid),
      .all_done           (npu_all_done)
  );

  always_ff @(posedge clk_i or negedge rst_ni) begin
    if (!rst_ni) begin
      instr_rvalid_q <= 1'b0;
      data_rvalid_q  <= 1'b0;
      data_rdata_q   <= 32'h0;
    end else begin
      instr_rvalid_q <= instr_req;
      data_rvalid_q  <= data_req;
      if (is_npu_addr(data_addr)) begin
        data_rdata_q <= {4{npu_host_rd_data}};
      end else if (is_npu_shared_addr(data_addr)) begin
        data_rdata_q <= npu_shared_allow ? npu_shared_rd_data : 32'h0;
      end else begin
        data_rdata_q <= 32'h0;
      end
    end
  end

  assign npu_host_addr = data_addr[4:0];
  assign npu_host_wr_data = selected_write_byte(data_wdata, data_addr);
  assign npu_host_wr_en = data_req && data_we && is_npu_addr(data_addr) && data_be[data_addr[1:0]];

  assign npu_shared_addr = is_npu_shared_im_addr(data_addr)
      ? (NPU_IM_BASE_WORD_ADDR + ((data_addr - NPU_SHARED_IM_BASE_ADDR) >> 4))
      : ((data_addr - NPU_SHARED_UB_BASE_ADDR) >> 4);
  assign npu_shared_lane = data_addr[3:2];
  assign npu_shared_wr_data = data_wdata;
  assign npu_shared_wr_be = data_be;
  assign npu_shared_wr_en = data_req && data_we && is_npu_shared_addr(data_addr) && npu_shared_allow;
  assign npu_shared_rd_en = data_req && !data_we && is_npu_shared_addr(data_addr) && npu_shared_allow;

  assign irq_lines = 32'h0;
  assign npu_result_valid_o = npu_result_valid;
  assign npu_all_done_o = npu_all_done;
  assign observe_o = instr_addr ^ data_addr ^ {24'h0, irq_id, irq_ack, debug_running, debug_halted};

endmodule
