`include "defines.sv"

module instruction_memory #(
    parameter INIT_FILE = ""
)(
    input logic clk,
    input logic rst_n,

    // Host Write Interface
    input  logic                     wr_en,
    input  logic [  `ADDR_WIDTH-1:0] wr_addr,
    input  logic [`BUFFER_WIDTH-1:0] wr_data,

    // Sequencer Read Interface
    input  logic [  `ADDR_WIDTH-1:0] rd_addr,
    output logic [  `INST_WIDTH-1:0] rd_data
);

    localparam CHUNK_BITS = (`BUFFER_WIDTH < `INST_WIDTH) ? $clog2(`INST_WIDTH / `BUFFER_WIDTH) : 0;
    
    logic [`INST_WIDTH-1:0] memory [0:`IM_SIZE-1];

    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, memory);
        end else begin
            for (int i=0; i<`IM_SIZE; i++) memory[i] = '0;
        end
    end

    logic [`ADDR_WIDTH-1:0] wr_addr_rel;
    logic [`ADDR_WIDTH-1:0] rd_addr_rel;

    assign wr_addr_rel = wr_addr - `IM_BASE_ADDR;
    assign rd_addr_rel = rd_addr - `IM_BASE_ADDR;

    // Host Write Logic
    always_ff @(posedge clk) begin
        if (wr_en) begin
            automatic int row_idx = wr_addr_rel >> CHUNK_BITS;
            if (row_idx < `IM_SIZE) begin
                if (CHUNK_BITS == 0) begin
                    // One chunk per instruction
                    memory[row_idx] <= wr_data[`INST_WIDTH-1:0];
                end else begin
                    // Multiple chunks per instruction
                    for (int c=0; c < (1 << CHUNK_BITS); c++) begin
                        if (wr_addr_rel[CHUNK_BITS-1:0] == c[CHUNK_BITS-1:0]) begin
                            memory[row_idx][c * `BUFFER_WIDTH +: `BUFFER_WIDTH] <= wr_data;
                        end
                    end
                end
            end
        end
    end

    // Fetch Logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_data <= '0;
        end else begin
            automatic int row_idx = rd_addr_rel >> CHUNK_BITS;
            if (row_idx < `IM_SIZE)
                rd_data <= memory[row_idx];
            else
                rd_data <= '0;
        end
    end

endmodule
