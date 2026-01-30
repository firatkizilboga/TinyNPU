`include "defines.sv"

// Test wrapper: Skewer + Systolic Array
module skewer_array_tb (
    input logic clk,
    input logic rst_n,
    input logic start,
    output logic busy,
    output logic done,
    
    // Input matrices
    input logic [`DATA_WIDTH-1:0] A [3:0][3:0],
    input logic [`DATA_WIDTH-1:0] B [3:0][3:0]
);

    // Skewer outputs -> Array inputs
    logic [`DATA_WIDTH-1:0] skewed_inputs [3:0];
    logic [`DATA_WIDTH-1:0] skewed_weights [3:0];
    
    // Systolic array control (hardcoded for this test)
    logic precision_mode;
    logic compute_enable;
    logic drain_enable;
    logic acc_clear;
    
    assign precision_mode = 2'b10;  // MODE_INT16
    assign compute_enable = 1'b1;    // Always computing
    assign drain_enable = 1'b0;
    assign acc_clear = 1'b0;
    
    // Instantiate skewer
    skewer #(
        .N(4),
        .K(4)  // Match input matrix dimensions
    ) skewer_inst (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .busy(busy),
        .done(done),
        .A(A),
        .B(B),
        .input_data(skewed_inputs),
        .weight_data(skewed_weights)
    );
    
    // Instantiate systolic array
    systolic_array array_inst (
        .clk(clk),
        .rst_n(rst_n),
        .input_data(skewed_inputs),
        .weight_data(skewed_weights),
        .precision_mode(precision_mode),
        .compute_enable(compute_enable),
        .drain_enable(drain_enable),
        .acc_clear(acc_clear)
    );

endmodule
