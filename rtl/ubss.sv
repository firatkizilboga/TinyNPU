`include "defines.sv"

// ============================================================================
// UBSS - Unified Buffer + Skewer + Systolic Array
// ============================================================================
// Full datapath test wrapper: UB → Skewers → Systolic Array
// Connects:
//   1. Unified Buffer (data storage)
//   2. Input Skewer (diagonal timing for input matrix A)
//   3. Weight Skewer (diagonal timing for weight matrix B)
//   4. Systolic Array (matrix multiply-accumulate)

module ubss #(
    parameter N = `ARRAY_SIZE,
    parameter DATA_WIDTH = `DATA_WIDTH,
    parameter ACC_WIDTH = `ACC_WIDTH,
    parameter ADDR_WIDTH = `ADDR_WIDTH,
    parameter BUFFER_WIDTH = `BUFFER_WIDTH,
    parameter INIT_FILE = "buffer_init.hex"
) (
    input logic clk,
    input logic rst_n,
    input logic en,

    // Write interface (for loading UB)
    input logic                    wr_en,
    input logic [ADDR_WIDTH-1:0]   wr_addr,
    input logic [BUFFER_WIDTH-1:0] wr_data,

    // Input path address interface
    input logic [ADDR_WIDTH-1:0] input_addr,
    input logic                  input_first_in,
    input logic                  input_last_in,

    // Weight path address interface
    input logic [ADDR_WIDTH-1:0] weight_addr,
    input logic                  weight_first_in,
    input logic                  weight_last_in,

    // Systolic array control
    input precision_mode_t precision_mode,
    input logic            compute_enable,
    input logic            drain_enable,
    input logic            acc_clear,

    // Skewer outputs (flattened for debug)
    output logic [N*DATA_WIDTH-1:0] input_skewed_flat,
    output logic [N*DATA_WIDTH-1:0] weight_skewed_flat,
    output logic                    skewer_input_first,
    output logic                    skewer_input_last,
    output logic                    skewer_weight_first,
    output logic                    skewer_weight_last,

    // Systolic array outputs
    output logic [(N * N * ACC_WIDTH)-1:0] results_flat,
    output logic                            result_valid,
    output logic                            computation_started,
    output logic                            computation_done,
    output logic                            all_done
);

  // ========================================================================
  // Internal wires
  // ========================================================================

  // UB outputs (before skewing)
  logic [BUFFER_WIDTH-1:0] ub_input_data;
  logic [BUFFER_WIDTH-1:0] ub_weight_data;
  logic                    ub_input_first;
  logic                    ub_input_last;
  logic                    ub_weight_first;
  logic                    ub_weight_last;

  // Unpacked arrays for skewer interface
  logic [DATA_WIDTH-1:0] input_unpacked  [N-1:0];
  logic [DATA_WIDTH-1:0] weight_unpacked [N-1:0];
  logic [DATA_WIDTH-1:0] input_skewed    [N-1:0];
  logic [DATA_WIDTH-1:0] weight_skewed   [N-1:0];

  // Skewer marker outputs
  logic input_first_out, input_last_out;
  logic weight_first_out, weight_last_out;

  // Systolic array results (unpacked)
  logic signed [ACC_WIDTH-1:0] results [N-1:0][N-1:0];

  // ========================================================================
  // Unified Buffer Instance
  // ========================================================================
  unified_buffer #(
      .INIT_FILE(INIT_FILE)
  ) ub_inst (
      .clk  (clk),
      .rst_n(rst_n),

      // Write interface
      .wr_en  (wr_en),
      .wr_addr(wr_addr),
      .wr_data(wr_data),

      // Input read interface
      .input_first_in (input_first_in),
      .input_last_in  (input_last_in),
      .input_addr     (input_addr),
      .input_first_out(ub_input_first),
      .input_last_out (ub_input_last),
      .input_data     (ub_input_data),

      // Weight read interface
      .weight_first_in (weight_first_in),
      .weight_last_in  (weight_last_in),
      .weight_addr     (weight_addr),
      .weight_first_out(ub_weight_first),
      .weight_last_out (ub_weight_last),
      .weight_data     (ub_weight_data)
  );

  // ========================================================================
  // Unpack UB output into arrays
  // ========================================================================
  genvar i;
  generate
    for (i = 0; i < N; i++) begin : unpack
      assign input_unpacked[i]  = ub_input_data[(i+1)*DATA_WIDTH-1-:DATA_WIDTH];
      assign weight_unpacked[i] = ub_weight_data[(i+1)*DATA_WIDTH-1-:DATA_WIDTH];
    end
  endgenerate

  // ========================================================================
  // Input Skewer Instance
  // ========================================================================
  streaming_skewer #(
      .N         (N),
      .DATA_WIDTH(DATA_WIDTH)
  ) input_skewer (
      .clk          (clk),
      .rst_n        (rst_n),
      .en           (en),
      .data_in      (input_unpacked),
      .data_out     (input_skewed),
      .first_in     (ub_input_first),
      .last_in      (ub_input_last),
      .first_out    (input_first_out),
      .last_out     (input_last_out),
      .data_out_flat(input_skewed_flat)
  );

  // ========================================================================
  // Weight Skewer Instance
  // ========================================================================
  streaming_skewer #(
      .N         (N),
      .DATA_WIDTH(DATA_WIDTH)
  ) weight_skewer (
      .clk          (clk),
      .rst_n        (rst_n),
      .en           (en),
      .data_in      (weight_unpacked),
      .data_out     (weight_skewed),
      .first_in     (ub_weight_first),
      .last_in      (ub_weight_last),
      .first_out    (weight_first_out),
      .last_out     (weight_last_out),
      .data_out_flat(weight_skewed_flat)
  );

  // Expose skewer markers
  assign skewer_input_first = input_first_out;
  assign skewer_input_last = input_last_out;
  assign skewer_weight_first = weight_first_out;
  assign skewer_weight_last = weight_last_out;

  // ========================================================================
  // Valid Data Window Tracking
  // ========================================================================
  // Track when valid data (between first and last markers) is flowing
  // to gate compute_enable and avoid computing with stale/invalid data.
  
  logic input_valid_window;
  logic weight_valid_window;
  logic gated_compute_enable;
  
  // Propagation counter: keep compute active for N cycles after last_out
  // This allows data already in the systolic array to fully propagate
  logic [$clog2(N+1):0] propagation_counter;
  logic propagation_active;
  
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      propagation_counter <= '0;
      propagation_active <= 1'b0;
    end else if (input_last_out || weight_last_out) begin
      // Start countdown when last data enters array
      propagation_counter <= N;
      propagation_active <= 1'b1;
    end else if (propagation_counter > 0) begin
      propagation_counter <= propagation_counter - 1;
    end else begin
      propagation_active <= 1'b0;
    end
  end
  
  // Input valid window: starts at first_out, ends after last_out
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      input_valid_window <= 1'b0;
    end else if (input_first_out) begin
      input_valid_window <= 1'b1;
    end else if (input_last_out) begin
      input_valid_window <= 1'b0;  // Turn off after this cycle
    end
  end
  
  // Weight valid window: starts at first_out, ends after last_out
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      weight_valid_window <= 1'b0;
    end else if (weight_first_out) begin
      weight_valid_window <= 1'b1;
    end else if (weight_last_out) begin
      weight_valid_window <= 1'b0;
    end
  end
  
  // Gate compute_enable: compute when valid data OR during propagation phase
  // Include the cycle where first/last asserts
  wire input_active = input_valid_window | input_first_out | input_last_out;
  wire weight_active = weight_valid_window | weight_first_out | weight_last_out;
  wire data_active = input_active & weight_active;
  assign gated_compute_enable = compute_enable & (data_active | propagation_active);

  // ========================================================================
  // Systolic Array Instance
  // ========================================================================
  systolic_array sa_inst (
      .clk  (clk),
      .rst_n(rst_n),

      // Data inputs (from skewers)
      .input_data  (input_skewed),
      .weight_data (weight_skewed),

      // Data flow markers (from skewers)
      .input_first (input_first_out),
      .input_last  (input_last_out),
      .weight_first(weight_first_out),
      .weight_last (weight_last_out),

      // Control signals
      .precision_mode(precision_mode),
      .compute_enable(gated_compute_enable),  // Gated by valid data window
      .drain_enable  (drain_enable),
      .acc_clear     (acc_clear),

      // Outputs
      .results           (results),
      .results_flat      (results_flat),
      .result_valid      (result_valid),
      .computation_started(computation_started),
      .computation_done  (computation_done),
      .all_done          (all_done)
  );

endmodule
