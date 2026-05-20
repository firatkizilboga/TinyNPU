`include "defines.sv"

`ifdef TINYNPU_FPGA_BRAM
`ifndef TINYNPU_VIVADO_BRAM
(* blackbox *) module instruction_memory #(
    parameter INIT_FILE = ""
)(
    input logic clk,
    input logic rst_n,

    input  logic                     wr_en,
    input  logic [  `ADDR_WIDTH-1:0] wr_addr,
    input  logic [`BUFFER_WIDTH-1:0] wr_data,

    input  logic [  `ADDR_WIDTH-1:0] host_shared_addr,
    input  logic [1:0]               host_shared_lane,
    input  logic [31:0]              host_shared_wr_data,
    input  logic [3:0]               host_shared_wr_be,
    input  logic                     host_shared_wr_en,
    input  logic                     host_shared_rd_en,
    output logic [31:0]              host_shared_rd_data,

    input  logic [  `ADDR_WIDTH-1:0] rd_addr,
    output logic [  `INST_WIDTH-1:0] rd_data
);
endmodule
`else
module instruction_memory #(
    parameter INIT_FILE = ""
)(
    input logic clk,
    input logic rst_n,

    input  logic                     wr_en,
    input  logic [  `ADDR_WIDTH-1:0] wr_addr,
    input  logic [`BUFFER_WIDTH-1:0] wr_data,

    input  logic [  `ADDR_WIDTH-1:0] host_shared_addr,
    input  logic [1:0]               host_shared_lane,
    input  logic [31:0]              host_shared_wr_data,
    input  logic [3:0]               host_shared_wr_be,
    input  logic                     host_shared_wr_en,
    input  logic                     host_shared_rd_en,
    output logic [31:0]              host_shared_rd_data,

    input  logic [  `ADDR_WIDTH-1:0] rd_addr,
    output logic [  `INST_WIDTH-1:0] rd_data
);
    localparam int INST_CHUNKS = (`BUFFER_WIDTH < `INST_WIDTH) ? (`INST_WIDTH / `BUFFER_WIDTH) : 1;
    localparam int IM_ADDR_BITS = $clog2(`IM_SIZE);
    localparam int INST_BYTES = `INST_WIDTH / 8;
    localparam int CHUNK_BYTES = `BUFFER_WIDTH / 8;

    logic [`ADDR_WIDTH-1:0] wr_addr_rel;
    logic [`ADDR_WIDTH-1:0] rd_addr_rel;
    logic [`ADDR_WIDTH-1:0] host_addr_rel;

    assign wr_addr_rel   = wr_addr - `IM_BASE_ADDR;
    assign rd_addr_rel   = rd_addr - `IM_BASE_ADDR;
    assign host_addr_rel = host_shared_addr - `IM_BASE_ADDR;

    logic [IM_ADDR_BITS-1:0] wr_row;
    logic [IM_ADDR_BITS-1:0] rd_row;
    logic [IM_ADDR_BITS-1:0] host_row;
    logic [IM_ADDR_BITS-1:0] read_row;
    logic [$clog2(INST_CHUNKS)-1:0] wr_chunk;
    logic [$clog2(INST_CHUNKS)-1:0] host_chunk;

    assign wr_row     = wr_addr_rel / INST_CHUNKS;
    assign rd_row     = rd_addr_rel / INST_CHUNKS;
    assign host_row   = host_addr_rel / INST_CHUNKS;
    assign wr_chunk   = wr_addr_rel % INST_CHUNKS;
    assign host_chunk = host_addr_rel % INST_CHUNKS;
    assign read_row   = host_shared_rd_en ? host_row : rd_row;

    logic [INST_BYTES-1:0] byte_wr_en;
    logic [7:0]            byte_wr_data [INST_BYTES-1:0];
    logic [7:0]            byte_rd_data [INST_BYTES-1:0];

    always_comb begin
        for (int b = 0; b < INST_BYTES; b++) begin
            byte_wr_en[b] = 1'b0;
            byte_wr_data[b] = 8'h00;
        end

        if (wr_en) begin
            for (int b = 0; b < CHUNK_BYTES; b++) begin
                byte_wr_en[(wr_chunk * CHUNK_BYTES) + b] = 1'b1;
                byte_wr_data[(wr_chunk * CHUNK_BYTES) + b] = wr_data[b*8 +: 8];
            end
        end else if (host_shared_wr_en) begin
            for (int b = 0; b < 4; b++) begin
                if (host_shared_wr_be[b]) begin
                    byte_wr_en[(host_chunk * CHUNK_BYTES) + (host_shared_lane * 4) + b] = 1'b1;
                    byte_wr_data[(host_chunk * CHUNK_BYTES) + (host_shared_lane * 4) + b] =
                        host_shared_wr_data[b*8 +: 8];
                end
            end
        end
    end

    generate
      genvar byte_idx;
      for (byte_idx = 0; byte_idx < INST_BYTES; byte_idx++) begin : g_im_byte_banks
        tinynpu_byte_ram #(
            .DEPTH(`IM_SIZE),
            .ADDR_BITS(IM_ADDR_BITS)
        ) u_bank (
            .clk    (clk),
            .wr_en  (byte_wr_en[byte_idx]),
            .wr_addr(wr_en ? wr_row : host_row),
            .wr_data(byte_wr_data[byte_idx]),
            .rd_addr(read_row),
            .rd_data(byte_rd_data[byte_idx])
        );
      end
    endgenerate

    logic [1:0] host_shared_lane_q;
    logic [$clog2(INST_CHUNKS)-1:0] host_chunk_q;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            host_shared_lane_q <= 2'b00;
            host_chunk_q <= '0;
        end else begin
            host_shared_lane_q <= host_shared_lane;
            host_chunk_q <= host_chunk;
        end
    end

    always_comb begin
        for (int b = 0; b < INST_BYTES; b++) begin
            rd_data[b*8 +: 8] = byte_rd_data[b];
        end

        host_shared_rd_data = 32'h0;
        for (int b = 0; b < 4; b++) begin
            host_shared_rd_data[b*8 +: 8] =
                byte_rd_data[(host_chunk_q * CHUNK_BYTES) + (host_shared_lane_q * 4) + b];
        end
    end
endmodule
`endif
`else
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

    // Synchronous read path. Normal execution uses rd_addr for instruction
    // fetch; host_shared_rd_en temporarily uses the same read path while the
    // NPU is idle. This avoids an asynchronous host tap into the IM array.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_data <= '0;
            host_shared_rd_data <= 32'h0;
        end else begin
            automatic int row_idx;
            automatic int chunk_idx;
            row_idx = host_shared_rd_en ? (host_addr_rel / INST_CHUNKS)
                                        : (rd_addr_rel / INST_CHUNKS);
            chunk_idx = host_addr_rel % INST_CHUNKS;

            if (row_idx < `IM_SIZE) begin
                rd_data <= memory[row_idx];
                if (host_shared_rd_en) begin
                    host_shared_rd_data <=
                        memory[row_idx][chunk_idx * `BUFFER_WIDTH + (host_shared_lane * 32) +: 32];
                end
            end else begin
                rd_data <= '0;
                if (host_shared_rd_en) begin
                    host_shared_rd_data <= 32'h0;
                end
            end
        end
    end

endmodule
`endif
