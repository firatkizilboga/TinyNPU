`include "defines.sv"

module control_top (
    input  logic clk,
    input  logic rst_n,

    // Interface to Host (MMIO)
    input  logic [`MMIO_ADDR_WIDTH-1:0] host_addr,
    input  logic [`HOST_DATA_WIDTH-1:0] host_wr_data,
    input  logic                        host_wr_en,
    output logic [`HOST_DATA_WIDTH-1:0] host_rd_data,

    // Interface to Unified Buffer
    output logic                        ub_req,
    output logic                        ub_wr_en,
    output logic [`ADDR_WIDTH-1:0]      ub_addr,
    output logic [`ADDR_WIDTH-1:0]      ub_w_addr, // For Weights
    output logic [`BUFFER_WIDTH-1:0]    ub_wdata,
    input  logic [`BUFFER_WIDTH-1:0]    ub_rdata,

    // Interface to Systolic Array & PPU
    output logic                        acc_clear,
    output logic                        compute_enable,
    output logic                        drain_enable,
    
    // PPU Control
    output logic [$clog2(`ARRAY_SIZE)-1:0] ppu_cycle_idx,
    output logic                           ppu_capture_en,

    // Sequencer Markers
    output logic                        sa_input_first,
    output logic                        sa_input_last,
    output logic                        sa_weight_first,
    output logic                        sa_weight_last,
    input  logic                        all_done_in
);

    logic [`HOST_DATA_WIDTH-1:0] cmd_bus;
    logic [`ADDR_WIDTH-1:0]      addr_bus;
    logic [`ARG_WIDTH-1:0]       arg_bus;
    logic [`BUFFER_WIDTH-1:0]    mmvr_bus;
    logic                        doorbell_pulse;
    logic [`HOST_DATA_WIDTH-1:0] status_bus;

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
        .doorbell_pulse (doorbell_pulse),
        .status_in      (status_bus)
    );

    // Instruction Memory Instance
    instruction_memory u_im (
        .clk     (clk),
        .rst_n   (rst_n),
        .wr_en   (im_wr_en),
        .wr_addr (im_addr),
        .wr_data (im_wdata),
        .rd_addr (im_addr),
        .rd_data (im_rdata)
    );

    // Control Unit Instance
    control_unit u_cu (
        .clk            (clk),
        .rst_n          (rst_n),
        .cmd_in         (cmd_bus),
        .addr_in        (addr_bus),
        .arg_in         (arg_bus),
        .mmvr_in        (mmvr_bus),
        .doorbell_pulse (doorbell_pulse),
        .status_out     (status_bus),
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
        .acc_clear      (acc_clear),
        .compute_enable (compute_enable),
        .drain_enable   (drain_enable),
        .sa_input_first (sa_input_first),
        .sa_input_last  (sa_input_last),
        .sa_weight_first(sa_weight_first),
        .sa_weight_last (sa_weight_last),
        .ppu_cycle_idx  (ppu_cycle_idx),
        .ppu_capture_en (ppu_capture_en),
        .all_done_in    (all_done_in)
    );

endmodule
