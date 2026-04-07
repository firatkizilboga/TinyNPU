`include "defines.sv"

module control_top #(
    parameter IM_INIT_FILE = "",
    parameter PERF_ENABLE = 0
) (
    input  logic clk,
    input  logic rst_n,

    // Interface to Host (MMIO)
    input  logic [`MMIO_ADDR_WIDTH-1:0] host_addr,
    input  logic [`HOST_DATA_WIDTH-1:0] host_wr_data,
    input  logic                        host_wr_en,
    output logic [`HOST_DATA_WIDTH-1:0] host_rd_data,

    // Optional shared SRAM host window into IM.
    input  logic [`ADDR_WIDTH-1:0]      host_shared_addr,
    input  logic [1:0]                  host_shared_lane,
    input  logic [31:0]                 host_shared_wr_data,
    input  logic [3:0]                  host_shared_wr_be,
    input  logic                        host_shared_wr_en,
    input  logic                        host_shared_rd_en,
    output logic [31:0]                 host_shared_rd_data,

    // Interface to Unified Buffer
    output logic                        ub_req,
    output logic                        ub_wr_en,
    output logic [`ADDR_WIDTH-1:0]      ub_addr,
    output logic [`ADDR_WIDTH-1:0]      ub_w_addr, // For Weights
    output logic [`BUFFER_WIDTH-1:0]    ub_wdata,
    input  logic [`BUFFER_WIDTH-1:0]    ub_rdata,
    output logic                        conv_stream_gather_en,
    output logic [`ADDR_WIDTH-1:0]      conv_stream_lane_word_addr[`ARRAY_SIZE-1:0],
    output logic [$clog2(`ARRAY_SIZE)-1:0] conv_stream_lane_word_lane[`ARRAY_SIZE-1:0],
    output logic [1:0]                  conv_stream_lane_subidx[`ARRAY_SIZE-1:0],
    output logic [`ARRAY_SIZE-1:0]      conv_stream_lane_valid,
    output logic [1:0]                  conv_stream_in_precision,

    // Interface to Systolic Array & PPU
    output logic                        acc_clear,
    output logic                        compute_enable,
    output logic                        drain_enable,
    output logic                        ppu_wb_en,
    output logic                        ppu_bias_en,
    output logic                        ppu_bias_clear,
    
    // PPU Control
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

    // Sequencer Markers
    output logic                        sa_input_first,
    output logic                        sa_input_last,
    output logic                        sa_weight_first,
    output logic                        sa_weight_last,

    // Shared SRAM host access gate (high when host direct access is safe)
    output logic                        host_shared_allow,
    input  logic                        all_done_in
);

    logic [`HOST_DATA_WIDTH-1:0] cmd_bus;
    logic [`ADDR_WIDTH-1:0]      addr_bus;
    logic [`ARG_WIDTH-1:0]       arg_bus;
    logic [`BUFFER_WIDTH-1:0]    mmvr_bus;
    logic                        doorbell_pulse;
    logic [`HOST_DATA_WIDTH-1:0] status_bus;
    logic                        mmvr_wr_en;
    logic [`BUFFER_WIDTH-1:0]    mmvr_internal;
    logic                        host_shared_im_wr_fire;
    logic                        host_shared_im_rd_fire;

    // Instruction Memory Internal Signals
    logic                        im_wr_en;
    logic [`ADDR_WIDTH-1:0]      im_addr;
    logic [`BUFFER_WIDTH-1:0]    im_wdata;
    logic [`INST_WIDTH-1:0]      im_rdata;

    // MMIO Interface Instance
    mmio_interface u_mmio (
        .clk            (clk),
        .rst_n          (rst_n),
        .host_addr      (host_addr),
        .host_wr_data   (host_wr_data),
        .host_wr_en     (host_wr_en),
        .host_rd_data   (host_rd_data),
        .cmd_out        (cmd_bus),
        .addr_out       (addr_bus),
        .arg_out        (arg_bus),
        .mmvr_out       (mmvr_bus),
        .mmvr_wr_en     (mmvr_wr_en),
        .mmvr_in        (mmvr_internal),
        .doorbell_pulse (doorbell_pulse),
        .status_in      (status_bus)
    );

    // Instruction Memory Instance
    instruction_memory #(
        .INIT_FILE(IM_INIT_FILE)
    ) u_im (
        .clk     (clk),
        .rst_n   (rst_n),
        .wr_en   (im_wr_en),
        .wr_addr (im_addr),
        .wr_data (im_wdata),
        .host_shared_addr(host_shared_addr),
        .host_shared_lane(host_shared_lane),
        .host_shared_wr_data(host_shared_wr_data),
        .host_shared_wr_be(host_shared_wr_be),
        .host_shared_wr_en(host_shared_im_wr_fire),
        .host_shared_rd_en(host_shared_im_rd_fire),
        .host_shared_rd_data(host_shared_rd_data),
        .rd_addr (im_addr),
        .rd_data (im_rdata)
    );

    // Control Unit Instance
    control_unit #(
        .PERF_ENABLE(PERF_ENABLE)
    ) u_cu (
        .clk            (clk),
        .rst_n          (rst_n),
        .cmd_in         (cmd_bus),
        .addr_in        (addr_bus),
        .arg_in         (arg_bus),
        .mmvr_in        (mmvr_bus),
        .doorbell_pulse (doorbell_pulse),
        .status_out     (status_bus),
        .mmvr_wr_en     (mmvr_wr_en),
        .mmvr_out       (mmvr_internal),
        .im_wr_en       (im_wr_en),
        .im_addr        (im_addr),
        .im_wdata       (im_wdata),
        .im_rdata       (im_rdata),
        .ub_req         (ub_req),
        .ub_wr_en       (ub_wr_en),
        .ub_addr        (ub_addr),
        .ub_w_addr      (ub_w_addr),
        .ub_wdata       (ub_wdata),
        .ub_rdata       (ub_rdata),
        .conv_stream_gather_en(conv_stream_gather_en),
        .conv_stream_lane_word_addr(conv_stream_lane_word_addr),
        .conv_stream_lane_word_lane(conv_stream_lane_word_lane),
        .conv_stream_lane_subidx(conv_stream_lane_subidx),
        .conv_stream_lane_valid(conv_stream_lane_valid),
        .conv_stream_in_precision(conv_stream_in_precision),
        .acc_clear      (acc_clear),
        .compute_enable (compute_enable),
        .drain_enable   (drain_enable),
        .ppu_wb_en      (ppu_wb_en),
        .ppu_bias_en    (ppu_bias_en),
        .ppu_bias_clear  (ppu_bias_clear),
        .sa_input_first (sa_input_first),
        .sa_input_last  (sa_input_last),
        .sa_weight_first(sa_weight_first),
        .sa_weight_last (sa_weight_last),
        .ppu_cycle_idx  (ppu_cycle_idx),
        .ppu_capture_en (ppu_capture_en),
        .ppu_shift      (ppu_shift),
        .ppu_multiplier (ppu_multiplier),
        .ppu_activation (ppu_activation),
        .ppu_h_gelu_x_scale_shift(ppu_h_gelu_x_scale_shift),
        .ppu_in_precision (ppu_in_precision),
        .ppu_out_precision(ppu_out_precision),
        .ppu_write_offset(ppu_write_offset),
        .ppu_output_layout(ppu_output_layout),
        .all_done_in    (all_done_in)
    );

    assign host_shared_allow = (status_bus != `STATUS_BUSY);
    assign host_shared_im_wr_fire = host_shared_allow && host_shared_wr_en;
    assign host_shared_im_rd_fire = host_shared_allow && host_shared_rd_en;

endmodule
