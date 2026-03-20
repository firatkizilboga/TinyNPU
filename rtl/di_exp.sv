module di_exp #(
    parameter int INPUT_WIDTH = 16,
    parameter int OUTPUT_WIDTH = 16,
    parameter int M_I = 128,
    parameter int K_I = 12
) (
    input  logic signed [INPUT_WIDTH-1:0]  x_in,
    output logic        [OUTPUT_WIDTH-1:0] y_out
);

    localparam int unsigned M_F = M_I + (M_I >> 1) - (M_I >> 4);
    localparam int unsigned T_MAG = (M_F > 0) ? (((1 << K_I) + (M_F >> 1)) / M_F) : 0;
    localparam int unsigned Y_MAX = (1 << OUTPUT_WIDTH) - 1;

    always_comb begin
        int signed t_val;
        int signed q_i;
        int signed r_i;
        int signed unshifted_exp;
        int signed result_i;
        int signed x_ext;

        if (M_F == 0 || T_MAG == 0) begin
            y_out = '0;
        end else begin
            x_ext = int'($signed(x_in));
            t_val = -$signed(T_MAG);
            q_i = x_ext / t_val;
            r_i = x_ext - (q_i * t_val);
            unshifted_exp = (r_i >>> 1) - t_val;
            result_i = unshifted_exp >>> q_i;

            if (result_i <= 0) begin
                y_out = '0;
            end else if (result_i > $signed(Y_MAX)) begin
                y_out = OUTPUT_WIDTH'(Y_MAX);
            end else begin
                y_out = OUTPUT_WIDTH'(result_i);
            end
        end
    end

endmodule
