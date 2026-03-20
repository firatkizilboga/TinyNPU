`include "defines.sv"

module tinynpu_top #(
    parameter IM_INIT_FILE = "",
    parameter UB_INIT_FILE = "",
    parameter PERF_ENABLE = 0
) (
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
    logic                        ppu_wb_en;
    logic                        ppu_bias_en;
    logic                        ppu_bias_clear;
    logic [$clog2(`ARRAY_SIZE)-1:0] ppu_cycle_idx;
    logic                        ppu_capture_en;
    logic [ 7:0]                 ppu_shift;
    logic [15:0]                 ppu_multiplier;
    logic [ 7:0]                 ppu_activation;
    logic [ 7:0]                 ppu_h_gelu_x_scale_shift;
    logic [ 1:0]                 ppu_in_precision;
    logic [ 1:0]                 ppu_out_precision;
    logic [ 1:0]                 ppu_write_offset;
    output_layout_t              ppu_output_layout;
    logic                        sa_input_first, sa_input_last;
    logic                        sa_weight_first, sa_weight_last;

    initial begin
        if ($test$plusargs("trace")) begin
            $dumpfile("tinynpu_trace.vcd");
            $dumpvars(0, tinynpu_top);
        end
    end

    control_top #(
        .IM_INIT_FILE(IM_INIT_FILE),
        .PERF_ENABLE(PERF_ENABLE)
    ) u_brain (
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
        .all_done_in    (all_done)
    );

    ubss #(
        .UB_INIT_FILE(UB_INIT_FILE)
    ) u_muscle (
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
        .precision_mode (precision_mode_t'(ppu_in_precision)), 
        .compute_enable (compute_enable),
        .drain_enable   (drain_enable),
        .ppu_wb_en      (ppu_wb_en),
        .ppu_bias_en    (ppu_bias_en),
        .ppu_bias_clear  (ppu_bias_clear),
        .acc_clear      (acc_clear),
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
        .results_flat   (results_flat),
        .result_valid   (result_valid),
        .all_done       (all_done)
    );

endmodule
