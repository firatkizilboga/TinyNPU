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

    // Host shared SRAM window (32-bit lane access into 128-bit IM chunk)
    input  logic [  `ADDR_WIDTH-1:0] host_shared_addr,
    input  logic [1:0]               host_shared_lane,
    input  logic [31:0]              host_shared_wr_data,
    input  logic [3:0]               host_shared_wr_be,
    input  logic                     host_shared_wr_en,
    input  logic                     host_shared_rd_en,
    output logic [31:0]              host_shared_rd_data,

    // Sequencer Read Interface
    input  logic [  `ADDR_WIDTH-1:0] rd_addr,
    output logic [  `INST_WIDTH-1:0] rd_data
);

    localparam INST_CHUNKS = (`BUFFER_WIDTH < `INST_WIDTH) ? (`INST_WIDTH / `BUFFER_WIDTH) : 1;
    
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
    logic [`ADDR_WIDTH-1:0] host_addr_rel;
    logic [`BUFFER_WIDTH-1:0] host_shared_mask_chunk;
    logic [`BUFFER_WIDTH-1:0] host_shared_wdata_chunk;

    assign wr_addr_rel = wr_addr - `IM_BASE_ADDR;
    assign rd_addr_rel = rd_addr - `IM_BASE_ADDR;
    assign host_addr_rel = host_shared_addr - `IM_BASE_ADDR;

    always_comb begin
        host_shared_mask_chunk = '0;
        host_shared_wdata_chunk = '0;
        for (int b = 0; b < 4; b++) begin
            if (host_shared_wr_be[b]) begin
                host_shared_mask_chunk[(host_shared_lane * 32) + (b * 8) +: 8] = 8'hFF;
            end
        end
        host_shared_wdata_chunk[(host_shared_lane * 32) +: 32] = host_shared_wr_data;
    end

    // Host Write Logic
    always_ff @(posedge clk) begin
        automatic int row_idx;
        automatic int chunk_idx;
        if (wr_en) begin
            row_idx = wr_addr_rel / INST_CHUNKS;
            chunk_idx = wr_addr_rel % INST_CHUNKS;
            if (row_idx < `IM_SIZE) begin
                memory[row_idx][chunk_idx * `BUFFER_WIDTH +: `BUFFER_WIDTH] <= wr_data;
            end
        end else if (host_shared_wr_en) begin
            row_idx = host_addr_rel / INST_CHUNKS;
            chunk_idx = host_addr_rel % INST_CHUNKS;
            if (row_idx < `IM_SIZE) begin
                memory[row_idx][chunk_idx * `BUFFER_WIDTH +: `BUFFER_WIDTH]
                    <= (memory[row_idx][chunk_idx * `BUFFER_WIDTH +: `BUFFER_WIDTH] & ~host_shared_mask_chunk)
                     | (host_shared_wdata_chunk & host_shared_mask_chunk);
            end
        end
    end

    // Fetch Logic
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_data <= '0;
        end else begin
            automatic int row_idx = rd_addr_rel / INST_CHUNKS;
            if (row_idx < `IM_SIZE)
                rd_data <= memory[row_idx];
            else
                rd_data <= '0;
        end
    end

    always_comb begin
        int row_idx;
        int chunk_idx;
        host_shared_rd_data = 32'h0;
        row_idx = 0;
        chunk_idx = 0;
        row_idx = host_addr_rel / INST_CHUNKS;
        chunk_idx = host_addr_rel % INST_CHUNKS;
        if (row_idx < `IM_SIZE) begin
            host_shared_rd_data =
                memory[row_idx][chunk_idx * `BUFFER_WIDTH + (host_shared_lane * 32) +: 32];
        end
    end

endmodule
