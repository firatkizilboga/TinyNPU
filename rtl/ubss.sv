`include "defines.sv"

module ubss (
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
    // Systolic Array Interface (Streamer)
    // ------------------------------------------------------------------------
    input  logic [`ADDR_WIDTH-1:0]   sa_input_addr,   // Base address for Input Matrix
    input  logic                     sa_input_first,  // Marker
    input  logic                     sa_input_last,   // Marker
    
    input  logic [`ADDR_WIDTH-1:0]   sa_weight_addr,  // Base address for Weight Matrix
    input  logic                     sa_weight_first, // Marker
    input  logic                     sa_weight_last,  // Marker

    input  precision_mode_t          precision_mode,
    input  logic                     compute_enable,
    input  logic                     drain_enable,    // From CU: Enable SA Drain
    input  logic                     acc_clear,

    // PPU Control
    input  logic [$clog2(`ARRAY_SIZE)-1:0] ppu_cycle_idx,
    input  logic                           ppu_capture_en,

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
    
    // Mux for UB Write Data: Either from CU (MMIO Load) or PPU (Compute Result)
    logic [`BUFFER_WIDTH-1:0] ub_final_wdata;
    assign ub_final_wdata = (drain_enable) ? ppu_wdata : cu_wdata; // Drain implies PPU writing

    // ========================================================================
    // Unified Buffer Instance
    // ========================================================================
    // The Buffer needs to handle requests from both CU and the Streaming Logic.
    // Ideally, we'd have a priority arbiter. 
    // For now, we assume CU and Streaming phases are mutually exclusive 
    // OR that CU access during compute is for non-conflicting addresses (not enforced here).
    
    // Wire up the Skewer ports
    logic [`BUFFER_WIDTH-1:0] skewer_input_data;
    logic [`BUFFER_WIDTH-1:0] skewer_weight_data;
    logic                     skewer_input_first, skewer_input_last;
    logic                     skewer_weight_first, skewer_weight_last;

    unified_buffer u_buffer (
        .clk             (clk),
        .rst_n           (rst_n),
        .wr_en           (cu_wr_en), // Note: control_unit handles pulsing this for PPU too
        .wr_addr         (cu_addr),
        .wr_data         (ub_final_wdata),
        
        // Port A: Input Matrix Stream
        .input_first_in  (sa_input_first),
        .input_last_in   (sa_input_last),
        .input_addr      (sa_input_addr),
        .input_first_out (skewer_input_first),
        .input_last_out  (skewer_input_last),
        .input_data      (skewer_input_data),
        
        // Port B: Weight Matrix Stream
        .weight_first_in (sa_weight_first),
        .weight_last_in  (sa_weight_last),
        .weight_addr     (sa_weight_addr),
        .weight_first_out(skewer_weight_first),
        .weight_last_out (skewer_weight_last),
        .weight_data     (skewer_weight_data)
    );
    
    // Hook up read data for CU (only valid if not streaming?)
    // Actually, unified_buffer doesn't have a dedicated random-access read port yet 
    // beyond the streaming ports.
    // Wait, check unified_buffer.sv definition.
    // It has NO random access read port. It only reads via the 'input' and 'weight' streams.
    // This is a limitation for `CMD_READ_MEM` and `ISA_OP_MOVE`.
    // FOR NOW: We will assume CU reads via the "Input" port logic or we need to add a port.
    // But 'unified_buffer.sv' is custom. 
    // Let's assume for this task we are focusing on the Compute Path.
    assign cu_rdata = '0; // STUB for now as requested task is Drain.

    // ========================================================================
    // Skewers
    // ========================================================================
    logic [`DATA_WIDTH-1:0] skewed_input  [`ARRAY_SIZE-1:0];
    logic [`DATA_WIDTH-1:0] skewed_weight [`ARRAY_SIZE-1:0];
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
        .data_out_flat()
    );

    streaming_skewer #(.N(`ARRAY_SIZE), .DATA_WIDTH(`DATA_WIDTH)) u_weight_skewer (
        .clk(clk), .rst_n(rst_n), .en(compute_enable),
        .data_in(weight_vec), .data_out(skewed_weight),
        .first_in(skewer_weight_first), .last_in(skewer_weight_last),
        .first_out(array_weight_first), .last_out(array_weight_last),
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
        .weight_first(array_weight_first), .weight_last(array_weight_last),
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
        .cycle_idx(ppu_cycle_idx),
        .acc_in(bottom_row_acc),
        .ub_wdata(ppu_wdata)
    );

endmodule
