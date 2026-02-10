`include "defines.sv"

module tinynpu_top (
    input  logic clk,
    input  logic rst_n,

    input  logic [`MMIO_ADDR_WIDTH-1:0] host_addr,
    input  logic [`HOST_DATA_WIDTH-1:0] host_wr_data,
    input  logic                        host_wr_en,
    output logic [`HOST_DATA_WIDTH-1:0] host_rd_data,

    output logic [(`ARRAY_SIZE * `ARRAY_SIZE * `ACC_WIDTH)-1:0] results_flat,
    output logic result_valid,
    output logic all_done
);

    logic                        ub_req;
    logic                        ub_wr_en;
    logic [`ADDR_WIDTH-1:0]      ub_addr;
    logic [`ADDR_WIDTH-1:0]      ub_w_addr;
    logic [`BUFFER_WIDTH-1:0]    ub_wdata;
    logic [`BUFFER_WIDTH-1:0]    ub_rdata;
    logic                        acc_clear;
    logic                        compute_enable;
    logic                        drain_enable;
    logic                        sa_input_first, sa_input_last;
    logic                        sa_weight_first, sa_weight_last;
    logic [$clog2(`ARRAY_SIZE)-1:0] ppu_cycle_idx;
    logic                           ppu_capture_en;

    control_top u_brain (
        .clk            (clk),
        .rst_n          (rst_n),
        .host_addr      (host_addr),
        .host_wr_data   (host_wr_data),
        .host_wr_en     (host_wr_en),
        .host_rd_data   (host_rd_data),
        .ub_req         (ub_req),
        .ub_wr_en       (ub_wr_en),
        .ub_addr        (ub_addr),
        .ub_w_addr      (ub_w_addr),
        .ub_wdata       (ub_wdata),
        .ub_rdata       (ub_rdata),
        .acc_clear      (acc_clear),
        .compute_enable (compute_enable),
        .drain_enable   (drain_enable),
        .ppu_cycle_idx  (ppu_cycle_idx),
        .ppu_capture_en (ppu_capture_en),
        .sa_input_first (sa_input_first),
        .sa_input_last  (sa_input_last),
        .sa_weight_first(sa_weight_first),
        .sa_weight_last (sa_weight_last),
        .all_done_in    (all_done)
    );

    ubss u_muscle (
        .clk            (clk),
        .rst_n          (rst_n),
        .en             (1'b1),

        .cu_req         (ub_req),
        .cu_wr_en       (ub_wr_en),
        .cu_addr        (ub_addr),
        .cu_wdata       (ub_wdata),
        .cu_rdata       (ub_rdata),

        .sa_input_addr  (ub_addr),
        .sa_input_first (sa_input_first),
        .sa_input_last  (sa_input_last),
        .sa_weight_addr (ub_w_addr),
        .sa_weight_first(sa_weight_first),
        .sa_weight_last (sa_weight_last),

        .precision_mode (2'b10),
        .compute_enable (compute_enable),
        .drain_enable   (drain_enable),
        .acc_clear      (acc_clear),

        .ppu_cycle_idx  (ppu_cycle_idx),
        .ppu_capture_en (ppu_capture_en),

        .results_flat   (results_flat),
        .result_valid   (result_valid),
        .all_done       (all_done)
    );

endmodule