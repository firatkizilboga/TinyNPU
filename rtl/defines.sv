`ifndef DEFINES_SV
`define DEFINES_SV

// ============================================================================
// TinyNPU Global Definitions
// ============================================================================
// Architecture parameters for the systolic array-based NPU
// Integer-only inference with configurable precision

// ----------------------------------------------------------------------------
// Data Type Parameters
// ----------------------------------------------------------------------------
`define DATA_WIDTH 8              // Input/Weight data width (8-bit integers)
`define ACC_WIDTH 32              // Accumulator width (prevent overflow)
`define ADDR_WIDTH 16             // Address bus width

// ----------------------------------------------------------------------------
// Systolic Array Dimensions
// ----------------------------------------------------------------------------
`define ARRAY_HEIGHT 4            // Number of PE rows
`define ARRAY_WIDTH 4             // Number of PE columns
`define NUM_PES (`ARRAY_HEIGHT * `ARRAY_WIDTH)

// ----------------------------------------------------------------------------
// Buffer Parameters
// ----------------------------------------------------------------------------
`define BUFFER_DEPTH 1024         // Unified buffer depth (entries)
`define BUFFER_DATA_WIDTH (`DATA_WIDTH * `ARRAY_WIDTH)

// ----------------------------------------------------------------------------
// Control Signals
// ----------------------------------------------------------------------------
typedef enum logic [2:0] {
    IDLE        = 3'b000,
    LOAD_WEIGHT = 3'b001,
    LOAD_INPUT  = 3'b010,
    COMPUTE     = 3'b011,
    DRAIN       = 3'b100,
    DONE        = 3'b101
} npu_state_t;

// ----------------------------------------------------------------------------
// Operation Types
// ----------------------------------------------------------------------------
typedef enum logic [1:0] {
    OP_MAC      = 2'b00,          // Multiply-Accumulate
    OP_RESET    = 2'b01,          // Reset accumulator
    OP_PASS     = 2'b10,          // Pass-through mode
    OP_IDLE     = 2'b11           // Idle state
} pe_op_t;

`endif // DEFINES_SV
