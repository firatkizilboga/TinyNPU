`include "defines.sv"

module control_unit (
    input  logic clk,
    input  logic rst_n,

    // Interface to MMIO
    input  logic [`HOST_DATA_WIDTH-1:0] cmd_in,
    input  logic [`ADDR_WIDTH-1:0]      addr_in,
    input  logic [`ARG_WIDTH-1:0]       arg_in,
    input  logic [`BUFFER_WIDTH-1:0]    mmvr_in,
    input  logic                        doorbell_pulse,
    output logic [`HOST_DATA_WIDTH-1:0] status_out,

    // Instruction Memory Interface (256-bit wide)
    output logic                        im_wr_en,
    output logic [`ADDR_WIDTH-1:0]      im_addr,
    output logic [`BUFFER_WIDTH-1:0]    im_wdata,
    input  logic [`INST_WIDTH-1:0]      im_rdata,

    // Unified Buffer Interface (Data)
    output logic                        ub_wr_en,
    output logic [`ADDR_WIDTH-1:0]      ub_addr,
    output logic [`BUFFER_WIDTH-1:0]    ub_wdata,
    input  logic [`BUFFER_WIDTH-1:0]    ub_rdata,

    // Interface to Systolic Array
    output logic                        acc_clear,
    output logic                        compute_enable
);

    typedef enum logic [3:0] {
        CTRL_IDLE,
        CTRL_HOST_WRITE,
        CTRL_HOST_READ,
        CTRL_FETCH,
        CTRL_DECODE,
        CTRL_EXEC_MOVE,
        CTRL_EXEC_MATMUL,
        CTRL_HALT
    } ctrl_state_t;

    ctrl_state_t state, next_state;
    logic [`ADDR_WIDTH-1:0] pc, pc_next;

    // Internal Registers for MOVE instruction
    logic [`ADDR_WIDTH-1:0] move_src, move_src_next;
    logic [`ADDR_WIDTH-1:0] move_dest, move_dest_next;
    logic [`ADDR_WIDTH-1:0] move_count, move_count_next;
    logic                   move_phase, move_phase_next; // 0=Read, 1=Write

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state      <= CTRL_IDLE;
            pc         <= '0;
            move_src   <= '0;
            move_dest  <= '0;
            move_count <= '0;
            move_phase <= 1'b0;
        end else begin
            state      <= next_state;
            pc         <= pc_next;
            move_src   <= move_src_next;
            move_dest  <= move_dest_next;
            move_count <= move_count_next;
            move_phase <= move_phase_next;
        end
    end

    always_comb begin
        // Defaults
        next_state      = state;
        pc_next         = pc;
        move_src_next   = move_src;
        move_dest_next  = move_dest;
        move_count_next = move_count;
        move_phase_next = move_phase;
        
        status_out      = `STATUS_IDLE;
        
        im_wr_en        = 1'b0;
        im_addr         = '0;
        im_wdata        = mmvr_in;
        
        ub_wr_en        = 1'b0;
        ub_addr         = '0;
        ub_wdata        = mmvr_in;

        acc_clear       = 1'b0;
        compute_enable  = 1'b0;

        case (state)
            CTRL_IDLE: begin
                status_out = `STATUS_IDLE;
                if (doorbell_pulse) begin
                    if      (cmd_in == `CMD_WRITE_MEM) next_state = CTRL_HOST_WRITE;
                    else if (cmd_in == `CMD_READ_MEM)  next_state = CTRL_HOST_READ;
                    else if (cmd_in == `CMD_RUN) begin
                        pc_next    = arg_in[`ADDR_WIDTH-1:0]; 
                        next_state = CTRL_FETCH;
                    end
                end
            end

            CTRL_HOST_WRITE: begin
                status_out = `STATUS_BUSY;
                if (addr_in >= `IM_BASE_ADDR) begin
                    im_wr_en = 1'b1;
                    im_addr  = addr_in - `IM_BASE_ADDR;
                end else begin
                    ub_wr_en = 1'b1;
                    ub_addr  = addr_in;
                end
                next_state = CTRL_IDLE;
            end

            CTRL_HOST_READ: begin
                status_out = `STATUS_BUSY;
                if (addr_in >= `IM_BASE_ADDR) im_addr = addr_in - `IM_BASE_ADDR;
                else                          ub_addr = addr_in;
                next_state = CTRL_IDLE;
            end

            CTRL_FETCH: begin
                status_out = `STATUS_BUSY;
                im_addr    = {pc[`ADDR_WIDTH-3:0], 2'b00}; 
                next_state = CTRL_DECODE;
            end

            CTRL_DECODE: begin
                status_out = `STATUS_BUSY;
                
                case (im_rdata[255:252])
                    ISA_OP_HALT: begin
                        pc_next    = pc + 1;
                        next_state = CTRL_HALT;
                    end
                    ISA_OP_NOP: begin
                        pc_next    = pc + 1;
                        next_state = CTRL_FETCH;
                    end
                    ISA_OP_MATMUL: begin
                        next_state = CTRL_EXEC_MATMUL;
                    end
                    ISA_OP_MOVE: begin
                        // Latch MOVE parameters from 256-bit instruction
                        move_src_next   = im_rdata[247:232];
                        move_dest_next  = im_rdata[231:216];
                        move_count_next = im_rdata[215:200];
                        move_phase_next = 1'b0; // Start with Read
                        next_state      = CTRL_EXEC_MOVE;
                    end
                    default: next_state = CTRL_HALT;
                endcase
            end

            CTRL_EXEC_MOVE: begin
                status_out = `STATUS_BUSY;
                if (move_count == 0) begin
                    pc_next    = pc + 1;
                    next_state = CTRL_FETCH;
                end else begin
                    if (move_phase == 1'b0) begin
                        // Phase 0: Read from Source
                        ub_addr         = move_src;
                        ub_wr_en        = 1'b0;
                        move_phase_next = 1'b1;
                    end else begin
                        // Phase 1: Write to Destination
                        ub_addr         = move_dest;
                        ub_wdata        = ub_rdata; // Captured from memory
                        ub_wr_en        = 1'b1;
                        
                        // Increment/Decrement
                        move_src_next   = move_src + 1;
                        move_dest_next  = move_dest + 1;
                        move_count_next = move_count - 1;
                        move_phase_next = 1'b0;
                    end
                end
            end

            CTRL_EXEC_MATMUL: begin
                status_out = `STATUS_BUSY;
                // STUB
                pc_next    = pc + 1;
                next_state = CTRL_FETCH; 
            end

            CTRL_HALT: begin
                status_out = `STATUS_HALTED;
                if (doorbell_pulse) next_state = CTRL_IDLE;
            end
            
            default: next_state = CTRL_IDLE;
        endcase
    end
endmodule
