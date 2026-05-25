`include "defines.sv"

module xform_unit (
    input  logic clk,
    input  logic rst_n,
    input  logic start,
    input  xform_mode_t mode,
    input  logic [`BUFFER_WIDTH-1:0] in_word,
    input  logic [`BUFFER_WIDTH-1:0] in_word_hi,
    input  logic [15:0] multiplier,
    input  logic [7:0] shift,
    output logic done,
    output logic [`BUFFER_WIDTH-1:0] out_word,
    output logic [`BUFFER_WIDTH-1:0] out_word_hi
);
    localparam int XFORM_LANES_PER_CYCLE = 2;
    localparam int XFORM_GROUP_COUNT =
        (`ARRAY_SIZE + XFORM_LANES_PER_CYCLE - 1) / XFORM_LANES_PER_CYCLE;
    localparam int XFORM_GROUP_WIDTH =
        (XFORM_GROUP_COUNT <= 1) ? 1 : $clog2(XFORM_GROUP_COUNT);

    logic busy;
    logic [XFORM_GROUP_WIDTH-1:0] group_idx;
    xform_mode_t mode_reg;
    logic [`BUFFER_WIDTH-1:0] in_word_reg;
    logic [`BUFFER_WIDTH-1:0] in_word_hi_reg;
    logic [15:0] multiplier_reg;
    logic [7:0] shift_reg;

    function automatic logic signed [63:0] round_shift_right_signed(
        input logic signed [63:0] value,
        input int unsigned shift_amount
    );
        logic signed [63:0] abs_value;
        logic signed [63:0] rounded;
        begin
            if (shift_amount == 0) begin
                round_shift_right_signed = value;
            end else if (shift_amount >= 63) begin
                round_shift_right_signed = '0;
            end else if (value >= 0) begin
                round_shift_right_signed = (value + (64'sd1 <<< (shift_amount - 1))) >>> shift_amount;
            end else begin
                abs_value = -value;
                rounded = (abs_value + (64'sd1 <<< (shift_amount - 1))) >>> shift_amount;
                round_shift_right_signed = -rounded;
            end
        end
    endfunction

    function automatic logic signed [15:0] clip_i16(input logic signed [63:0] value);
        begin
            if (value > 64'sd32767) begin
                clip_i16 = 16'sh7fff;
            end else if (value < -64'sd32768) begin
                clip_i16 = 16'sh8000;
            end else begin
                clip_i16 = value[15:0];
            end
        end
    endfunction

    function automatic logic signed [15:0] quantize_lane_q_f32_i16(
        input logic [31:0] fp32,
        input logic [15:0] lane_multiplier,
        input logic [7:0] lane_shift
    );
        logic sign;
        logic [7:0] exp_bits;
        logic [22:0] frac_bits;
        logic signed [63:0] mant;
        logic signed [63:0] scaled;
        logic signed [63:0] qvalue;
        int exp2;
        int shift_i;
        int left_shift;
        int right_shift;
        begin
            sign = fp32[31];
            exp_bits = fp32[30:23];
            frac_bits = fp32[22:0];
            shift_i = int'(lane_shift);
            qvalue = 64'sd0;
            if (lane_multiplier == 16'd0) begin
                quantize_lane_q_f32_i16 = 16'sd0;
            end else if (exp_bits == 8'hff) begin
                quantize_lane_q_f32_i16 = sign ? -16'sd32768 : 16'sd32767;
            end else if (exp_bits == 8'd0 && frac_bits == 23'd0) begin
                quantize_lane_q_f32_i16 = 16'sd0;
            end else begin
                if (exp_bits == 8'd0) begin
                    mant = $signed({41'd0, frac_bits});
                    exp2 = -149;
                end else begin
                    mant = $signed({40'd0, 1'b1, frac_bits});
                    exp2 = int'(exp_bits) - 150;
                end

                scaled = mant * $signed({1'b0, lane_multiplier});
                if (exp2 >= shift_i) begin
                    left_shift = exp2 - shift_i;
                    qvalue = (left_shift >= 24) ? 64'sh7fffffffffffffff : (scaled <<< left_shift);
                end else begin
                    right_shift = shift_i - exp2;
                    qvalue = round_shift_right_signed(scaled, right_shift);
                end

                if (sign) begin
                    qvalue = -qvalue;
                end
                quantize_lane_q_f32_i16 = clip_i16(qvalue);
            end
        end
    endfunction

    function automatic int unsigned msb_index_u32(input logic [31:0] value);
        begin
            msb_index_u32 = 0;
            for (int bit_idx = 31; bit_idx >= 0; bit_idx--) begin
                if (value[bit_idx]) begin
                    msb_index_u32 = bit_idx;
                    break;
                end
            end
        end
    endfunction

    function automatic logic [31:0] round_shift_right_unsigned32(
        input logic [31:0] value,
        input int unsigned shift_amount
    );
        begin
            if (shift_amount == 0) begin
                round_shift_right_unsigned32 = value;
            end else if (shift_amount >= 32) begin
                round_shift_right_unsigned32 = '0;
            end else begin
                round_shift_right_unsigned32 = (value + (32'd1 << (shift_amount - 1))) >> shift_amount;
            end
        end
    endfunction

    function automatic logic [31:0] dequantize_lane_i16_f32(
        input logic signed [15:0] value,
        input logic [15:0] lane_multiplier,
        input logic [7:0] lane_shift
    );
        logic sign;
        logic [15:0] magnitude;
        logic [31:0] product;
        logic [31:0] rounded;
        logic [22:0] fraction;
        int unsigned msb;
        int signed exp_unbiased;
        int signed exp_float;
        int signed normal_shift;
        int signed subnormal_shift;
        begin
            if (value == 16'sd0 || lane_multiplier == 16'd0) begin
                dequantize_lane_i16_f32 = 32'h0000_0000;
            end else begin
                sign = value[15];
                magnitude = sign ? ((~value[15:0]) + 16'd1) : value[15:0];
                product = {16'd0, magnitude} * {16'd0, lane_multiplier};
                msb = msb_index_u32(product);
                exp_unbiased = int'(msb) - int'(lane_shift);
                exp_float = exp_unbiased + 127;

                if (exp_float >= 255) begin
                    dequantize_lane_i16_f32 = {sign, 8'hfe, 23'h7fffff};
                end else if (exp_float <= 0) begin
                    subnormal_shift = int'(lane_shift) - 149;
                    if (subnormal_shift <= 0) begin
                        rounded = ((-subnormal_shift) >= 32) ? 32'hffff_ffff : (product << (-subnormal_shift));
                    end else begin
                        rounded = round_shift_right_unsigned32(product, subnormal_shift);
                    end
                    if (rounded == 32'd0) begin
                        dequantize_lane_i16_f32 = 32'h0000_0000;
                    end else if (rounded >= 32'h0080_0000) begin
                        dequantize_lane_i16_f32 = {sign, 8'd1, 23'd0};
                    end else begin
                        dequantize_lane_i16_f32 = {sign, 8'd0, rounded[22:0]};
                    end
                end else begin
                    normal_shift = int'(msb) - 23;
                    if (normal_shift >= 0) begin
                        rounded = round_shift_right_unsigned32(product, normal_shift);
                    end else begin
                        rounded = product << (-normal_shift);
                    end
                    if (rounded >= 32'd16777216) begin
                        exp_float = exp_float + 1;
                        fraction = 23'd0;
                    end else begin
                        fraction = rounded[22:0];
                    end
                    if (exp_float >= 255) begin
                        dequantize_lane_i16_f32 = {sign, 8'hfe, 23'h7fffff};
                    end else begin
                        dequantize_lane_i16_f32 = {sign, exp_float[7:0], fraction};
                    end
                end
            end
        end
    endfunction

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            done <= 1'b0;
            busy <= 1'b0;
            group_idx <= '0;
            mode_reg <= XFORM_MODE_NONE;
            in_word_reg <= '0;
            in_word_hi_reg <= '0;
            multiplier_reg <= '0;
            shift_reg <= '0;
            out_word <= '0;
            out_word_hi <= '0;
        end else begin
            done <= 1'b0;
            if (start && !busy) begin
                busy <= 1'b1;
                group_idx <= '0;
                mode_reg <= mode;
                in_word_reg <= in_word;
                in_word_hi_reg <= in_word_hi;
                multiplier_reg <= multiplier;
                shift_reg <= shift;
                out_word <= in_word;
                out_word_hi <= in_word_hi;
            end else if (busy) begin
                for (int lane_offset = 0; lane_offset < XFORM_LANES_PER_CYCLE; lane_offset++) begin
                    int lane;
                    lane = int'(group_idx) * XFORM_LANES_PER_CYCLE + lane_offset;
                    if (lane < `ARRAY_SIZE) begin
                        unique case (mode_reg)
                            XFORM_MODE_Q_F32_I16: begin
                                out_word[lane*16 +: 16] <= quantize_lane_q_f32_i16(
                                    lane < 4 ? in_word_reg[lane*32 +: 32] : in_word_hi_reg[(lane-4)*32 +: 32],
                                    multiplier_reg,
                                    shift_reg
                                );
                            end
                            XFORM_MODE_DQ_I16_F32: begin
                                if (lane < 4) begin
                                    out_word[lane*32 +: 32] <= dequantize_lane_i16_f32(
                                        $signed(in_word_reg[lane*16 +: 16]),
                                        multiplier_reg,
                                        shift_reg
                                    );
                                end else begin
                                    out_word_hi[(lane-4)*32 +: 32] <= dequantize_lane_i16_f32(
                                        $signed(in_word_reg[lane*16 +: 16]),
                                        multiplier_reg,
                                        shift_reg
                                    );
                                end
                            end
                            default: ;
                        endcase
                    end
                end

                if (group_idx == XFORM_GROUP_WIDTH'(XFORM_GROUP_COUNT - 1)) begin
                    busy <= 1'b0;
                    done <= 1'b1;
                end else begin
                    group_idx <= group_idx + 1'b1;
                end
            end
        end
    end

endmodule
