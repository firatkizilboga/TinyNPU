`include "defines.sv"

module mmio_interface (
    input  logic clk,
    input  logic rst_n,

    input  logic [`MMIO_ADDR_WIDTH-1:0] host_addr,
    input  logic [`HOST_DATA_WIDTH-1:0] host_wr_data,
    input  logic                        host_wr_en,
    output logic [`HOST_DATA_WIDTH-1:0] host_rd_data,

    input  logic [`HOST_DATA_WIDTH-1:0] status_in,

    output logic [`HOST_DATA_WIDTH-1:0] cmd_out,
    output logic [`ADDR_WIDTH-1:0]      addr_out,
    output logic [`ARG_WIDTH-1:0]       arg_out,
    output logic [`BUFFER_WIDTH-1:0]    mmvr_out,

    output logic        doorbell_pulse
);

    logic [`HOST_DATA_WIDTH-1:0] cmd_reg;
    logic [`ADDR_WIDTH-1:0]      addr_reg;
    logic [`ARG_WIDTH-1:0]       arg_reg;
    logic [`BUFFER_WIDTH-1:0]    mmvr_reg;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmd_reg        <= '0;
            addr_reg       <= '0;
            arg_reg        <= '0;
            mmvr_reg       <= '0;
            doorbell_pulse <= 1'b0;
        end else begin
            doorbell_pulse <= 1'b0;
            if (host_wr_en) begin
                case (host_addr)
                    `REG_CMD: cmd_reg <= host_wr_data;
                    
                    `REG_ADDR:   addr_reg[7:0]   <= host_wr_data;
                    `REG_ADDR+1: addr_reg[15:8]  <= host_wr_data;
                    
                    `REG_ARG:    arg_reg[7:0]    <= host_wr_data;
                    `REG_ARG+1:  arg_reg[15:8]   <= host_wr_data;
                    `REG_ARG+2:  arg_reg[23:16]  <= host_wr_data;
                    `REG_ARG+3:  arg_reg[31:24]  <= host_wr_data;
                    
                    `REG_MMVR:   mmvr_reg[7:0]   <= host_wr_data;
                    `REG_MMVR+1: mmvr_reg[15:8]  <= host_wr_data;
                    `REG_MMVR+2: mmvr_reg[23:16] <= host_wr_data;
                    `REG_MMVR+3: mmvr_reg[31:24] <= host_wr_data;
                    `REG_MMVR+4: mmvr_reg[39:32] <= host_wr_data;
                    `REG_MMVR+5: mmvr_reg[47:40] <= host_wr_data;
                    `REG_MMVR+6: mmvr_reg[55:48] <= host_wr_data;
                    `REG_MMVR+7: begin
                        mmvr_reg[63:56] <= host_wr_data;
                        doorbell_pulse  <= 1'b1;
                    end
                    default: ;
                endcase
            end
        end
    end

    always_comb begin
        case (host_addr)
            `REG_STATUS: host_rd_data = status_in;
            `REG_CMD:    host_rd_data = cmd_reg;
            
            `REG_ADDR:   host_rd_data = addr_reg[7:0];
            `REG_ADDR+1: host_rd_data = addr_reg[15:8];
            
            `REG_ARG:    host_rd_data = arg_reg[7:0];
            `REG_ARG+1:  host_rd_data = arg_reg[15:8];
            `REG_ARG+2:  host_rd_data = arg_reg[23:16];
            `REG_ARG+3:  host_rd_data = arg_reg[31:24];
            
            `REG_MMVR:   host_rd_data = mmvr_reg[7:0];
            `REG_MMVR+1: host_rd_data = mmvr_reg[15:8];
            `REG_MMVR+2: host_rd_data = mmvr_reg[23:16];
            `REG_MMVR+3: host_rd_data = mmvr_reg[31:24];
            `REG_MMVR+4: host_rd_data = mmvr_reg[39:32];
            `REG_MMVR+5: host_rd_data = mmvr_reg[47:40];
            `REG_MMVR+6: host_rd_data = mmvr_reg[55:48];
            `REG_MMVR+7: host_rd_data = mmvr_reg[63:56];
            default:     host_rd_data = '0;
        endcase
    end

    assign cmd_out  = cmd_reg;
    assign addr_out = addr_reg;
    assign arg_out  = arg_reg;
    assign mmvr_out = mmvr_reg;

endmodule