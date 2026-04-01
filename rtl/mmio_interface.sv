`include "defines.sv"

module mmio_interface (
    input logic clk,
    input logic rst_n,

    input  logic [`MMIO_ADDR_WIDTH-1:0] host_addr,
    input  logic [`HOST_DATA_WIDTH-1:0] host_wr_data,
    input  logic                        host_wr_en,
    output logic [`HOST_DATA_WIDTH-1:0] host_rd_data,

    input logic [`HOST_DATA_WIDTH-1:0] status_in,

    output logic [`HOST_DATA_WIDTH-1:0] cmd_out,
    output logic [     `ADDR_WIDTH-1:0] addr_out,
    output logic [      `ARG_WIDTH-1:0] arg_out,
    output logic [   `BUFFER_WIDTH-1:0] mmvr_out,

    // Internal Write Interface (from CU)
    input logic                     mmvr_wr_en,
    input logic [`BUFFER_WIDTH-1:0] mmvr_in,

    output logic doorbell_pulse
);

  logic [`HOST_DATA_WIDTH-1:0] cmd_reg;
  logic [     `ADDR_WIDTH-1:0] addr_reg;
  logic [      `ARG_WIDTH-1:0] arg_reg;
  logic [   `BUFFER_WIDTH-1:0] mmvr_reg;
  logic                        doorbell_q;
  logic                        mmvr_host_override;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      cmd_reg    <= '0;
      addr_reg   <= '0;
      arg_reg    <= '0;
      mmvr_reg   <= '0;
      doorbell_q <= 1'b0;
      mmvr_host_override <= 1'b0;
    end else begin
      doorbell_q <= 1'b0;

      // Once host software starts preparing the next command while READ_WAIT
      // is still presenting stale readback data, keep MMVR under host control
      // until the control unit leaves READ_WAIT.
      if (!mmvr_wr_en) begin
        mmvr_host_override <= 1'b0;
      end else if (host_wr_en) begin
        mmvr_host_override <= 1'b1;
      end

      // Host writes
      if (host_wr_en) begin
        if (host_addr == `REG_CMD) begin
          cmd_reg <= host_wr_data;
        end else if (host_addr >= `REG_ADDR && host_addr < `REG_ADDR + 2) begin
          addr_reg[(host_addr - `REG_ADDR)*8 +: 8] <= host_wr_data;
        end else if (host_addr >= `REG_ARG && host_addr < `REG_ARG + 4) begin
          arg_reg[(host_addr - `REG_ARG)*8 +: 8] <= host_wr_data;
        end else if (host_addr >= `REG_MMVR && host_addr < `REG_MMVR + (`BUFFER_WIDTH/8)) begin
          mmvr_reg[(host_addr - `REG_MMVR)*8 +: 8] <= host_wr_data;
          // Trigger doorbell on the very last byte of the current buffer width
          if (host_addr == `REG_MMVR + (`BUFFER_WIDTH/8) - 1) begin
            doorbell_q <= 1'b1;
          end
        end
      end

      // Internal MMVR update (READ_MEM result).
      // Host writes to MMVR must take precedence, otherwise the first WRITE_MEM
      // issued from READ_WAIT can be overwritten by stale readback data.
      if (mmvr_wr_en &&
          !mmvr_host_override &&
          !(host_wr_en &&
            host_addr >= `REG_MMVR &&
            host_addr < (`REG_MMVR + (`BUFFER_WIDTH / 8)))) begin
        mmvr_reg <= mmvr_in;
      end
    end
  end

  assign doorbell_pulse = doorbell_q;

  always_comb begin
    host_rd_data = '0;
    if (host_addr == `REG_STATUS) begin
      host_rd_data = status_in;
    end else if (host_addr == `REG_CMD) begin
      host_rd_data = cmd_reg;
    end else if (host_addr >= `REG_ADDR && host_addr < `REG_ADDR + 2) begin
      host_rd_data = addr_reg[(host_addr - `REG_ADDR)*8 +: 8];
    end else if (host_addr >= `REG_ARG && host_addr < `REG_ARG + 4) begin
      host_rd_data = arg_reg[(host_addr - `REG_ARG)*8 +: 8];
    end else if (host_addr >= `REG_MMVR && host_addr < `REG_MMVR + (`BUFFER_WIDTH/8)) begin
      host_rd_data = mmvr_reg[(host_addr - `REG_MMVR)*8 +: 8];
    end
  end

  assign cmd_out  = cmd_reg;
  assign addr_out = addr_reg;
  assign arg_out  = arg_reg;
  assign mmvr_out = mmvr_reg;

endmodule
