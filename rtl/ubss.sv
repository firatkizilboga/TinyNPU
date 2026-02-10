`include "defines.sv"

// ============================================================================
// UBSS - Unified Buffer + Skewer + Systolic Array (Integrated Arbiter)
// ============================================================================
// Connects:
//   1. Unified Buffer (Shared Memory)
//   2. Input/Weight Skewers (Timing)
//   3. Systolic Array (Compute)
// Features:
//   - Arbiter: Prioritizes Host/DMA (Control Unit) over Systolic Array.
//   - Status: Exposes real-time computation status signals.

module ubss #(
    parameter N = `ARRAY_SIZE,
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter ACC_WIDTH = `ACC_WIDTH,
    parameter ADDR_WIDTH = `ADDR_WIDTH,
    parameter BUFFER_WIDTH = `BUFFER_WIDTH,
    parameter INIT_FILE = ""
) (
    input logic clk,
    input logic rst_n,
    input logic en,

    // --- Control Unit Interface (Priority Port) ---
    input  logic                    cu_req,
    input  logic                    cu_wr_en,
    input  logic [ADDR_WIDTH-1:0]   cu_addr,
    input  logic [BUFFER_WIDTH-1:0] cu_wdata,
    output logic [BUFFER_WIDTH-1:0] cu_rdata,

    // --- Systolic Address Interface (from Brain AGU) ---
    input  logic [ADDR_WIDTH-1:0]   sa_input_addr,
    input  logic                    sa_input_first,
    input  logic                    sa_input_last,
    input  logic [ADDR_WIDTH-1:0]   sa_weight_addr,
    input  logic                    sa_weight_first,
    input  logic                    sa_weight_last,

    // --- Systolic Control ---
    input precision_mode_t precision_mode,
    input logic            compute_enable,
    input logic            drain_enable,
    input logic            acc_clear,

    // --- Status / Results ---
    output logic [(N * N * ACC_WIDTH)-1:0] results_flat,
    output logic                            result_valid,
    output logic                            computation_started,
    output logic                            computation_done,
    output logic                            all_done,

    // --- Debug Outputs ---
    output logic [N*DATA_WIDTH-1:0]         input_skewed_flat,
    output logic [N*DATA_WIDTH-1:0]         weight_skewed_flat
);

    // ========================================================================
    // Internal Wires & Arbiter Mux
    // ========================================================================
    logic [ADDR_WIDTH-1:0]   mux_addr_a;
    logic                    mux_first_a;
    logic                    mux_last_a;
    logic [BUFFER_WIDTH-1:0] raw_rdata_a;
    
    logic [BUFFER_WIDTH-1:0] ub_weight_data;
    logic                    ub_weight_first;
    logic                    ub_weight_last;
    
    // Markers from UB (Muxed output)
    logic                    ub_input_first;
    logic                    ub_input_last;

    // --- ARBITER LOGIC ---
    // If cu_req is HIGH, the Control Unit steals Port A (Input Path).
    // Otherwise, the Systolic Array owns Port A.
    always_comb begin
        if (cu_req) begin
            // CU Access
            mux_addr_a  = cu_addr;
            mux_first_a = 1'b0; // CU ignores markers
            mux_last_a  = 1'b0;
            cu_rdata    = raw_rdata_a;
        end else begin
            // SA Access
            mux_addr_a  = sa_input_addr;
            mux_first_a = sa_input_first;
            mux_last_a  = sa_input_last;
            cu_rdata    = '0;
        end
    end

    // ========================================================================
    // Unified Buffer Instance
    // ========================================================================
    unified_buffer #(
        .INIT_FILE(INIT_FILE)
    ) ub_inst (
        .clk             (clk),
        .rst_n           (rst_n),
        // Write Port (Exclusive to CU)
        .wr_en           (cu_req && cu_wr_en),
        .wr_addr         (cu_addr),
        .wr_data         (cu_wdata),
        
        // Read Port A (Muxed: CU or SA Inputs)
        .input_addr      (mux_addr_a),
        .input_first_in  (mux_first_a),
        .input_last_in   (mux_last_a),
        .input_data      (raw_rdata_a),
        .input_first_out (ub_input_first),
        .input_last_out  (ub_input_last),
        
        // Read Port B (Dedicated to SA Weights)
        .weight_addr     (sa_weight_addr),
        .weight_first_in (sa_weight_first),
        .weight_last_in  (sa_weight_last),
        .weight_data     (ub_weight_data),
        .weight_first_out(ub_weight_first),
        .weight_last_out (ub_weight_last)
    );

    // ========================================================================
    // Unpack & Skew Logic
    // ========================================================================
    logic [DATA_WIDTH-1:0] input_unpacked  [N-1:0];
    logic [DATA_WIDTH-1:0] weight_unpacked [N-1:0];
    logic [DATA_WIDTH-1:0] input_skewed    [N-1:0];
    logic [DATA_WIDTH-1:0] weight_skewed   [N-1:0];
    
    // Skewer markers
    logic skewer_input_first, skewer_input_last;
    logic skewer_weight_first, skewer_weight_last;

    genvar i;
    generate
        for (i = 0; i < N; i++) begin : unpack
            // When CU is active, we force 0 to Skewers to prevent noise
            assign input_unpacked[i]  = cu_req ? {DATA_WIDTH{1'b0}} : raw_rdata_a[(i+1)*DATA_WIDTH-1-:DATA_WIDTH];
            assign weight_unpacked[i] = ub_weight_data[(i+1)*DATA_WIDTH-1-:DATA_WIDTH];
        end
    endgenerate

    streaming_skewer #(.N(N), .DATA_WIDTH(DATA_WIDTH)) i_skew (
        .clk(clk), .rst_n(rst_n), .en(en),
        .data_in(input_unpacked), .data_out(input_skewed),
        .first_in(ub_input_first), .last_in(ub_input_last),
        .first_out(skewer_input_first), .last_out(skewer_input_last),
        .data_out_flat(input_skewed_flat)
    );

    streaming_skewer #(.N(N), .DATA_WIDTH(DATA_WIDTH)) w_skew (
        .clk(clk), .rst_n(rst_n), .en(en),
        .data_in(weight_unpacked), .data_out(weight_skewed),
        .first_in(ub_weight_first), .last_in(ub_weight_last),
        .first_out(skewer_weight_first), .last_out(skewer_weight_last),
        .data_out_flat(weight_skewed_flat)
    );

    // ========================================================================
    // Valid Window Gating (Safety)
    // ========================================================================
    logic input_valid_window, weight_valid_window;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_valid_window <= 1'b0;
            weight_valid_window <= 1'b0;
        end else begin
            if (skewer_input_first)  input_valid_window <= 1'b1;
            else if (skewer_input_last)   input_valid_window <= 1'b0;
            
            if (skewer_weight_first) weight_valid_window <= 1'b1;
            else if (skewer_weight_last)  weight_valid_window <= 1'b0;
        end
    end
    
    // Only compute when data is actively flowing (window + markers)
    wire data_active = (input_valid_window | skewer_input_first | skewer_input_last) & 
                       (weight_valid_window | skewer_weight_first | skewer_weight_last);

    // ========================================================================
    // Systolic Array Instance
    // ========================================================================
    systolic_array sa_inst (
        .clk(clk), .rst_n(rst_n),
        .input_data(input_skewed), .weight_data(weight_skewed),
        .input_first(skewer_input_first), .input_last(skewer_input_last),
        .weight_first(skewer_weight_first), .weight_last(skewer_weight_last),
        .precision_mode(precision_mode), 
        .compute_enable(compute_enable && data_active),
        .drain_enable(drain_enable), 
        .acc_clear(acc_clear),
        .results_flat(results_flat), .results(), .result_valid(result_valid),
        .computation_started(computation_started), .computation_done(computation_done), .all_done(all_done)
    );

endmodule