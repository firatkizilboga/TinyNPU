`include "defines.sv"

// ============================================================================
// Processing Element (PE) Module - Output Stationary with Multi-Precision SWAR
// ============================================================================
// Implements packed SWAR (SIMD Within A Register) dot product with configurable precision
// Supported modes:
//   - INT4:  4×INT4 values → 4 products
//   - INT8:  2×INT8 values → 2 products
//   - INT16: 1×INT16 value → 1 product
// 
// Data Flow (Output-Stationary):
//   - Inputs flow LEFT → RIGHT (horizontal, 16-bit)
//   - Weights/Accumulators flow TOP → BOTTOM (vertical, 64-bit)
//     * Compute mode: 16-bit weights (lower bits of 64-bit bus)
//     * Drain mode: 64-bit accumulators
//   - Accumulators are STATIONARY (stay in PE during compute)
//
// Packed Format (UINT16):
//   INT4:  [15:12][11:8][7:4][3:0] = 4 nibbles
//   INT8:  [15:8][7:0] = 2 bytes
//   INT16: [15:0] = 1 word

module pe (
    input  logic clk,
    input  logic rst_n,
    
    // Horizontal data flow (16-bit): inputs
    input  logic [`DATA_WIDTH-1:0] input_from_left,
    output logic [`DATA_WIDTH-1:0] input_to_right,
    
    // Vertical data flow (64-bit): weights during compute, accumulators during drain
    input  logic signed [`ACC_WIDTH-1:0] data_from_top,
    output logic signed [`ACC_WIDTH-1:0] data_to_bottom,
    
    // Control signals
    input  precision_mode_t precision_mode,   // Bit-width mode selection
    input  logic compute_enable,              // Enable SWAR computation
    input  logic drain_enable,                // Enable drain mode (shift accumulators vertically)
    input  logic acc_clear,                   // Clear accumulator
    
    // Accumulator output (stationary result)
    output logic signed [`ACC_WIDTH-1:0] acc_out
);

    // ------------------------------------------------------------------------
    // Internal Registers (Latches for Incoming Data)
    // ------------------------------------------------------------------------
    logic [`DATA_WIDTH-1:0] input_latch;
    logic [`DATA_WIDTH-1:0] weight_latch;  // Extracted from lower 16 bits of data_from_top
    logic signed [`ACC_WIDTH-1:0] vertical_latch;  // Registered vertical output
    logic signed [`ACC_WIDTH-1:0] accumulator;
    
    // ------------------------------------------------------------------------
    // Extract Weight from Vertical Bus (Compute Mode)
    // ------------------------------------------------------------------------
    // During compute: data_from_top[15:0] = weight (upper bits ignored)
    // During drain: data_from_top[63:0] = accumulator from above
    
    // Register weight to match input_latch timing
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_latch <= '0;
        end else begin
            weight_latch <= data_from_top[`DATA_WIDTH-1:0];
        end
    end
    
    // Register vertical output to delay weights/accumulators by 1 cycle per row
    // This ensures each row's weight timing is aligned with its input timing
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            vertical_latch <= '0;
        end else begin
            vertical_latch <= data_from_top;
        end
    end

    always_ff @(negedge clk) begin : debugBlock
        $display("[PE] Time=%0t weight_latch=0x%04X input_latch=0x%04X", $time, weight_latch, input_latch);
        $display("[PE Acc] Time=%0t accumulator=0x%04X", $time, accumulator);
    end 
    // ------------------------------------------------------------------------
    // Multi-Precision SWAR Dot Product Logic
    // ------------------------------------------------------------------------
    logic signed [31:0] partial_sum;  // Accumulated sum of all products
    
    // INT4 mode: 4 products
    logic signed [3:0] i4_0, i4_1, i4_2, i4_3;
    logic signed [3:0] w4_0, w4_1, w4_2, w4_3;
    logic signed [7:0] p4_0, p4_1, p4_2, p4_3;
    
    // INT8 mode: 2 products
    logic signed [7:0] i8_0, i8_1;
    logic signed [7:0] w8_0, w8_1;
    logic signed [15:0] p8_0, p8_1;
    
    // INT16 mode: 1 product
    logic signed [15:0] i16;
    logic signed [15:0] w16;
    logic signed [31:0] p16;
    
    always_comb begin
        partial_sum = '0;
        
        // Default assignments to avoid latches
        i4_0 = '0; i4_1 = '0; i4_2 = '0; i4_3 = '0;
        w4_0 = '0; w4_1 = '0; w4_2 = '0; w4_3 = '0;
        p4_0 = '0; p4_1 = '0; p4_2 = '0; p4_3 = '0;
        
        i8_0 = '0; i8_1 = '0;
        w8_0 = '0; w8_1 = '0;
        p8_0 = '0; p8_1 = '0;
        
        i16 = '0;
        w16 = '0;
        p16 = '0;
        
        unique case (precision_mode)
            MODE_INT4: begin
                // Unpack 4×INT4 from UINT16
                i4_0 = signed'(input_latch[3:0]);
                i4_1 = signed'(input_latch[7:4]);
                i4_2 = signed'(input_latch[11:8]);
                i4_3 = signed'(input_latch[15:12]);
                
                w4_0 = signed'(weight_latch[3:0]);
                w4_1 = signed'(weight_latch[7:4]);
                w4_2 = signed'(weight_latch[11:8]);
                w4_3 = signed'(weight_latch[15:12]);
                
                // Compute 4 products (INT4 × INT4 = INT8)
                p4_0 = i4_0 * w4_0;
                p4_1 = i4_1 * w4_1;
                p4_2 = i4_2 * w4_2;
                p4_3 = i4_3 * w4_3;
                
                // Sum all 4 products
                partial_sum = $signed(p4_0) + $signed(p4_1) + $signed(p4_2) + $signed(p4_3);
            end
            
            MODE_INT8: begin
                // Unpack 2×INT8 from UINT16
                i8_0 = signed'(input_latch[7:0]);
                i8_1 = signed'(input_latch[15:8]);
                
                w8_0 = signed'(weight_latch[7:0]);
                w8_1 = signed'(weight_latch[15:8]);
                
                // Compute 2 products (INT8 × INT8 = INT16)
                p8_0 = i8_0 * w8_0;
                p8_1 = i8_1 * w8_1;
                
                // Sum both products
                partial_sum = $signed(p8_0) + $signed(p8_1);
            end
            
            MODE_INT16: begin
                // Single INT16 value (no unpacking)
                i16 = signed'(input_latch);
                w16 = signed'(weight_latch);
                
                // Single product (INT16 × INT16 = INT32)
                p16 = i16 * w16;
                
                partial_sum = p16;
            end
            
            MODE_RSVD: begin
                partial_sum = '0;
            end
        endcase
    end
    
    // ------------------------------------------------------------------------
    // Accumulator (Stationary Output)
    // ------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= '0;
        end else if (acc_clear) begin
            accumulator <= '0;
        end else if (drain_enable) begin
            accumulator <= data_from_top;
        end else if (compute_enable) begin
            accumulator <= accumulator + $signed({{(`ACC_WIDTH-32){partial_sum[31]}}, partial_sum});
        end else begin
        end
    end
    
    // ------------------------------------------------------------------------
    // Horizontal Data Flow Latch (16-bit)
    // ------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            input_latch <= '0;
        end else begin
            input_latch <= input_from_left;
        end
    end
    
    // ------------------------------------------------------------------------
    // Output Assignments
    // ------------------------------------------------------------------------
    assign input_to_right = input_latch;    // Propagate input horizontally (16-bit)
    
    // Vertical output: pass WEIGHTS during compute, ACCUMULATORS during drain
    // This is critical for multi-row operation!
    // - Compute mode: Weights must flow down to subsequent rows (with 1-cycle delay)
    // - Drain mode: Accumulators flow down to drain the array
    // Use vertical_latch (registered) to match the 1-cycle delay of weight_latch
    assign data_to_bottom = drain_enable ? accumulator : vertical_latch;
    
    assign acc_out        = accumulator;    // Stationary result

endmodule
