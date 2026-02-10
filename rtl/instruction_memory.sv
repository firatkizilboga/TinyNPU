`include "defines.sv"

module instruction_memory (
    input  logic clk,
    input  logic rst_n,

    // Host Write Interface (64-bit chunks)
    input  logic                     wr_en,
    input  logic [`ADDR_WIDTH-1:0]   wr_addr, // Bits [1:0] select the 64-bit word
    input  logic [`BUFFER_WIDTH-1:0] wr_data,

    // Control Unit Read Interface (Full 256-bit)
    input  logic [`ADDR_WIDTH-1:0]   rd_addr,
    output logic [`INST_WIDTH-1:0]   rd_data
);

    // Memory is 256 bits wide
    logic [`INST_WIDTH-1:0] memory [`IM_SIZE-1:0];

    // Host Write Logic (Splicing 64-bit writes into 256-bit rows)
    always_ff @(posedge clk) begin
        if (wr_en) begin
            case (wr_addr[1:0])
                2'b00: memory[wr_addr[`ADDR_WIDTH-1:2]][63:0]    <= wr_data;
                2'b01: memory[wr_addr[`ADDR_WIDTH-1:2]][127:64]  <= wr_data;
                2'b10: memory[wr_addr[`ADDR_WIDTH-1:2]][191:128] <= wr_data;
                2'b11: memory[wr_addr[`ADDR_WIDTH-1:2]][255:192] <= wr_data;
            endcase
        end
    end

    // Fetch Logic (Single cycle)
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_data <= '0;
        end else begin
            rd_data <= memory[rd_addr[`ADDR_WIDTH-1:2]];
        end
    end

endmodule