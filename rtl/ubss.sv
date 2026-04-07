`include "defines.sv"

module ubss #(
    parameter UB_INIT_FILE = ""
) (
    input  logic clk,
    input  logic rst_n,
    input  logic en, // Top-level enable

    // ------------------------------------------------------------------------
    // Control Unit Interface (Read/Write Requests)
    // ------------------------------------------------------------------------
    input  logic                     cu_req,    // CU requesting access
    input  logic                     cu_wr_en,  // 1=Write, 0=Read
    input  logic [`ADDR_WIDTH-1:0]   cu_addr,
    input  logic [`BUFFER_WIDTH-1:0] cu_wdata,  // Data from CU (MMIO Load)
    output logic [`BUFFER_WIDTH-1:0] cu_rdata,  // Data to CU (MMIO Read / Move)

    // ------------------------------------------------------------------------
    // Host shared SRAM window (32-bit lane access into 128-bit UB word)
    // ------------------------------------------------------------------------
    input  logic [`ADDR_WIDTH-1:0]   host_shared_addr,
    input  logic [1:0]               host_shared_lane,
    input  logic [31:0]              host_shared_wr_data,
    input  logic [3:0]               host_shared_wr_be,
    input  logic                     host_shared_wr_en,
    input  logic                     host_shared_rd_en,
    input  logic                     host_shared_allow,
    output logic [31:0]              host_shared_rd_data,

    // ------------------------------------------------------------------------
    // Systolic Array Interface (Streamer)
    // ------------------------------------------------------------------------
    input  logic [`ADDR_WIDTH-1:0]   sa_input_addr,   // Base address for Input Matrix
    input  logic                     sa_input_first,  // Marker
    input  logic                     sa_input_last,   // Marker
    input  logic                     conv_stream_gather_en,
    input  logic [`ADDR_WIDTH-1:0]   conv_stream_lane_word_addr[`ARRAY_SIZE-1:0],
    input  logic [$clog2(`ARRAY_SIZE)-1:0] conv_stream_lane_word_lane[`ARRAY_SIZE-1:0],
    input  logic [1:0]               conv_stream_lane_subidx[`ARRAY_SIZE-1:0],
    input  logic [`ARRAY_SIZE-1:0]   conv_stream_lane_valid,
    input  logic [1:0]               conv_stream_in_precision,
    
    input  logic [`ADDR_WIDTH-1:0]   sa_weight_addr,  // Base address for Weight Matrix
    input  logic                     sa_weight_first, // Marker
    input  logic                     sa_weight_last,  // Marker

    input  precision_mode_t          precision_mode,
    input  logic                     compute_enable,
    input  logic                     drain_enable,    // From CU: Enable SA Drain
    input  logic                     acc_clear,

    // PPU Control
    input  logic                     ppu_wb_en,
    input  logic                     ppu_bias_en,
    input  logic                     ppu_bias_clear,
    input  logic [$clog2(`ARRAY_SIZE)-1:0] ppu_cycle_idx,
    input  logic                           ppu_capture_en,
    input  logic [ 7:0]                    ppu_shift,
    input  logic [15:0]                    ppu_multiplier,
    input  logic [ 7:0]                    ppu_activation,
    input  logic [ 7:0]                    ppu_h_gelu_x_scale_shift,
    input  logic [ 1:0]                    ppu_in_precision,
    input  logic [ 1:0]                    ppu_out_precision,
    input  logic [ 1:0]                    ppu_write_offset,
    input  output_layout_t                 ppu_output_layout,

    // ------------------------------------------------------------------------
    // Outputs
    // ------------------------------------------------------------------------
    output logic [(`ARRAY_SIZE * `ARRAY_SIZE * `ACC_WIDTH)-1:0] results_flat,
    output logic                     result_valid,
    output logic                     all_done
);

    // Internal Wires
    logic [`BUFFER_WIDTH-1:0] ub_rdata_internal;
    logic [`BUFFER_WIDTH-1:0] ppu_wdata;
    logic [`BUFFER_WIDTH-1:0] ub_wr_mask;
    logic [`BUFFER_WIDTH-1:0] host_shared_mask;
    logic [`BUFFER_WIDTH-1:0] host_shared_wide_wdata;
    logic                     host_shared_wr_fire;
    
    // Mask Generation for Compact Packing
    always_comb begin
        ub_wr_mask = '1; // Default: Write all bits (for MMIO/Load)
        if (ppu_wb_en) begin
            if (ppu_output_layout == OUT_LAYOUT_A) begin
                ub_wr_mask = '1;
            end else begin
                unique case (ppu_out_precision)
                    2'b00: begin // INT4
                        for (int i=0; i<`ARRAY_SIZE; i++) 
                            ub_wr_mask[i*16 +: 16] = 16'h000F << (ppu_write_offset * 4);
                    end
                    2'b01: begin // INT8
                        for (int i=0; i<`ARRAY_SIZE; i++)
                            ub_wr_mask[i*16 +: 16] = 16'h00FF << (ppu_write_offset * 8);
                    end
                    default: begin // INT16 (2'b10)
                        ub_wr_mask = '1;
                    end
                endcase
            end
        end
    end

    // Mux for UB Write Data: PPU (compute result) > host_shared > CU (MMIO load)
    // PPU must win to avoid corrupting writeback.
    logic [`BUFFER_WIDTH-1:0] ub_final_wdata;
    assign ub_final_wdata = ppu_wb_en ? ppu_wdata
                          : host_shared_wr_fire ? host_shared_wide_wdata
                          : cu_wdata;

    // ========================================================================
    // Unified Buffer Instance
    // ========================================================================
    // The Buffer needs to handle requests from both CU and the Streaming Logic.
    
    // Wire up the Skewer ports
    logic [`BUFFER_WIDTH-1:0] skewer_input_data;
    logic [`BUFFER_WIDTH-1:0] skewer_input_data_comb;
    logic [`BUFFER_WIDTH-1:0] skewer_weight_data;
    logic [`BUFFER_WIDTH-1:0] host_data_comb;
    logic                     skewer_input_first, skewer_input_last;
    logic                     skewer_weight_first, skewer_weight_last;
    logic [`ADDR_WIDTH-1:0]   ub_port_a_addr;
    logic                     ub_port_a_first_in, ub_port_a_last_in;
    logic [`ADDR_WIDTH-1:0]   ub_port_b_addr;
    logic                     ub_port_b_first_in, ub_port_b_last_in;

    // Read-port arbiter:
    // - Port A is shared between SA input stream and CU random reads.
    // - Port B serves SA weight stream.
    // This keeps UB read addressing to 2 ports total.
    always_comb begin
        if (compute_enable) begin
            ub_port_a_addr     = sa_input_addr;
            ub_port_a_first_in = sa_input_first;
            ub_port_a_last_in  = sa_input_last;
            ub_port_b_addr     = sa_weight_addr;
            ub_port_b_first_in = sa_weight_first;
            ub_port_b_last_in  = sa_weight_last;
        end else begin
            ub_port_a_addr     = cu_addr;
            ub_port_a_first_in = 1'b0;
            ub_port_a_last_in  = 1'b0;
            ub_port_b_addr     = '0;
            ub_port_b_first_in = 1'b0;
            ub_port_b_last_in  = 1'b0;
        end
    end

    // CU reads observe Port A combinational tap when arbiter routes CU onto Port A.
    assign cu_rdata = cu_req && !cu_wr_en ? skewer_input_data_comb : '0;

    always_comb begin
        host_shared_mask = '0;
        host_shared_wide_wdata = '0;
        for (int b = 0; b < 4; b++) begin
            if (host_shared_wr_be[b]) begin
                host_shared_mask[(host_shared_lane * 32) + (b * 8) +: 8] = 8'hFF;
            end
        end
        host_shared_wide_wdata[(host_shared_lane * 32) +: 32] = host_shared_wr_data;
    end

    assign host_shared_wr_fire = (host_shared_allow === 1'b1) && (host_shared_wr_en === 1'b1);
    // Keep read data address-driven (not rd_en-gated). The core samples on
    // response timing, which can be a cycle after the request pulse.
    assign host_shared_rd_data = host_data_comb[(host_shared_lane * 32) +: 32];

    unified_buffer #(
        .INIT_FILE(UB_INIT_FILE)
    ) u_buffer (
        .clk             (clk),
        .rst_n           (rst_n),
        .wr_en           (cu_wr_en | ppu_wb_en | host_shared_wr_fire),
        .wr_mask         (host_shared_wr_fire && !ppu_wb_en ? host_shared_mask : ub_wr_mask),
        .wr_addr         (host_shared_wr_fire && !ppu_wb_en ? host_shared_addr : cu_addr),
        .wr_data         (ub_final_wdata),
        
        // Port A: SA Input Stream or CU Read (arbiter selected)
        .input_first_in  (ub_port_a_first_in),
        .input_last_in   (ub_port_a_last_in),
        .input_addr      (ub_port_a_addr),
        .conv_stream_gather_en(conv_stream_gather_en),
        .conv_stream_lane_word_addr(conv_stream_lane_word_addr),
        .conv_stream_lane_word_lane(conv_stream_lane_word_lane),
        .conv_stream_lane_subidx(conv_stream_lane_subidx),
        .conv_stream_lane_valid(conv_stream_lane_valid),
        .conv_stream_in_precision(conv_stream_in_precision),
        .input_first_out (skewer_input_first),
        .input_last_out  (skewer_input_last),
        .input_data      (skewer_input_data),
        .input_data_comb (skewer_input_data_comb),
        
        // Port B: Weight Matrix Stream
        .weight_first_in (ub_port_b_first_in),
        .weight_last_in  (ub_port_b_last_in),
        .weight_addr     (ub_port_b_addr),
        .weight_first_out(skewer_weight_first),
        .weight_last_out (skewer_weight_last),
        .weight_data     (skewer_weight_data),
        .host_addr       (host_shared_addr),
        .host_data_comb  (host_data_comb)
    );
    
    // ========================================================================
    // Skewers
    // ========================================================================
    logic [`DATA_WIDTH-1:0] skewed_input  [`ARRAY_SIZE-1:0];
    logic [`DATA_WIDTH-1:0] skewed_weight [`ARRAY_SIZE-1:0];
    logic [`ARRAY_SIZE-1:0] input_valid_bus;
    logic [`ARRAY_SIZE-1:0] weight_valid_bus;
    logic                   array_input_first, array_input_last;
    logic                   array_weight_first, array_weight_last;
    
    // Unpack Buffer Data
    logic [`DATA_WIDTH-1:0] input_vec  [`ARRAY_SIZE-1:0];
    logic [`DATA_WIDTH-1:0] weight_vec [`ARRAY_SIZE-1:0];
    
    generate
        genvar i;
        for (i=0; i<`ARRAY_SIZE; i++) begin
            assign input_vec[i]  = skewer_input_data[i*16 +: 16];
            assign weight_vec[i] = skewer_weight_data[i*16 +: 16];
        end
    endgenerate

    streaming_skewer #(.N(`ARRAY_SIZE), .DATA_WIDTH(`DATA_WIDTH)) u_input_skewer (
        .clk(clk), .rst_n(rst_n), .en(compute_enable),
        .data_in(input_vec), .data_out(skewed_input),
        .first_in(skewer_input_first), .last_in(skewer_input_last),
        .first_out(array_input_first), .last_out(array_input_last),
        .valid_out(input_valid_bus),
        .data_out_flat()
    );

    streaming_skewer #(.N(`ARRAY_SIZE), .DATA_WIDTH(`DATA_WIDTH)) u_weight_skewer (
        .clk(clk), .rst_n(rst_n), .en(compute_enable),
        .data_in(weight_vec), .data_out(skewed_weight),
        .first_in(skewer_weight_first), .last_in(skewer_weight_last),
        .first_out(array_weight_first), .last_out(array_weight_last),
        .valid_out(weight_valid_bus),
        .data_out_flat()
    );

    // ========================================================================
    // Systolic Array
    // ========================================================================
    logic signed [`ACC_WIDTH-1:0] sa_results [`ARRAY_SIZE-1:0][`ARRAY_SIZE-1:0];
    
    systolic_array u_array (
        .clk(clk), .rst_n(rst_n),
        .input_data(skewed_input), .weight_data(skewed_weight),
        .input_first(array_input_first), .input_last(array_input_last),
        .input_valid(input_valid_bus),
        .weight_first(array_weight_first), .weight_last(array_weight_last),
        .weight_valid(weight_valid_bus),
        .precision_mode(precision_mode),
        .compute_enable(compute_enable),
        .drain_enable(drain_enable),
        .acc_clear(acc_clear),
        .results(sa_results),
        .results_flat(results_flat),
        .result_valid(result_valid),
        .computation_started(), .computation_done(), .all_done(all_done)
    );

    // ========================================================================
    // Post-Processing Unit (PPU)
    // ========================================================================
    // Connects bottom of SA to Buffer Write Data
    logic signed [`ACC_WIDTH-1:0] bottom_row_acc [`ARRAY_SIZE-1:0];
    
    // Extract bottom row (Row 3 for 4x4)
    generate
        genvar k;
        for (k=0; k<`ARRAY_SIZE; k++) begin
            assign bottom_row_acc[k] = sa_results[`ARRAY_SIZE-1][k];
        end
    endgenerate

    ppu u_ppu (
        .clk(clk),
        .rst_n(rst_n),
        .capture_en(ppu_capture_en),
              .bias_en       (ppu_bias_en),
              .bias_clear    (ppu_bias_clear),
              .ppu_cycle_idx (ppu_cycle_idx),
        .shift         (ppu_shift),
        .multiplier(ppu_multiplier),
        .activation(ppu_activation),
        .h_gelu_x_scale_shift(ppu_h_gelu_x_scale_shift),
        .precision(ppu_out_precision),
        .write_offset(ppu_write_offset),
        .output_layout(ppu_output_layout),
        .bias_in(cu_rdata),
        .acc_in(bottom_row_acc),
        .ub_wdata(ppu_wdata)
    );

endmodule
