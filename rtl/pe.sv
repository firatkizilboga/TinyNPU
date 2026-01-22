`include "defines.sv"

// ============================================================================
// Processing Element (PE) Module
// ============================================================================
// Basic MAC (Multiply-Accumulate) unit for systolic array
// Implements: acc = acc + (input * weight)
// Integer-only arithmetic with configurable bit widths

module pe (
    input  logic clk,
    input  logic rst_n,
    
    // Data inputs (from previous PE or external)
    input  logic signed [`DATA_WIDTH-1:0] data_in,
    input  logic signed [`DATA_WIDTH-1:0] weight_in,
    
    // Data outputs (to next PE)
    output logic signed [`DATA_WIDTH-1:0] data_out,
    output logic signed [`DATA_WIDTH-1:0] weight_out,
    
    // Control signals
    input  logic weight_load,             // Load weight into register
    input  logic acc_enable,              // Enable accumulation
    input  logic acc_clear,               // Clear accumulator
    
    // Accumulator output
    output logic signed [`ACC_WIDTH-1:0] acc_out
);

    // ------------------------------------------------------------------------
    // Internal Registers
    // ------------------------------------------------------------------------
    logic signed [`DATA_WIDTH-1:0]  weight_reg;
    logic signed [`ACC_WIDTH-1:0]   accumulator;
    logic signed [`ACC_WIDTH-1:0]   mac_result;
    
    // ------------------------------------------------------------------------
    // Weight Register - Stationary in PE
    // ------------------------------------------------------------------------
    // In a weight-stationary systolic array, weights are loaded once and
    // remain in the PE while inputs flow through
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            weight_reg <= '0;
        end else if (weight_load) begin
            weight_reg <= weight_in;
        end
    end
    
    // ------------------------------------------------------------------------
    // Multiply-Accumulate Logic
    // ------------------------------------------------------------------------
    // Compute: acc = acc + (data_in * weight_reg)
    always_comb begin
        mac_result = accumulator + (data_in * weight_reg);
    end
    
    // ------------------------------------------------------------------------
    // Accumulator Register
    // ------------------------------------------------------------------------
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accumulator <= '0;
        end else if (acc_clear) begin
            accumulator <= '0;
        end else if (acc_enable) begin
            accumulator <= mac_result;
        end
    end
    
    // ------------------------------------------------------------------------
    // Data Flow - Pipeline Registers
    // ------------------------------------------------------------------------
    // Data and weights flow to neighboring PEs with one cycle delay
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data_out   <= '0;
            weight_out <= '0;
        end else begin
            data_out   <= data_in;
            weight_out <= weight_in;
        end
    end
    
    // ------------------------------------------------------------------------
    // Output Assignment
    // ------------------------------------------------------------------------
    assign acc_out = accumulator;

endmodule
