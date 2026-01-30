`ifndef DEFINES_SV
`define DEFINES_SV

// ============================================================================
// TinyNPU Global Definitions
// ============================================================================
// Output-stationary systolic array with SWAR (SIMD Within A Register)
// Packed INT8 operations: 2×INT8 values in 1×UINT16 for 2× throughput
// Integer-only inference, mimics FP16 performance with packed execution

// ----------------------------------------------------------------------------
// Data Type Parameters
// ----------------------------------------------------------------------------
`define DATA_WIDTH 16             // Packed data width (2×INT8 in UINT16)
`define ACC_WIDTH 64              // Accumulator width (holds sum of products)
`define ADDR_WIDTH 16             // Address bus width

// ----------------------------------------------------------------------------
// Systolic Array Dimensions
// ----------------------------------------------------------------------------
`define ARRAY_SIZE 4              // N×N array (rows = columns)
`define NUM_PES (`ARRAY_SIZE * `ARRAY_SIZE)

// ----------------------------------------------------------------------------
// Buffer Parameters
// ----------------------------------------------------------------------------
`define BUFFER_DEPTH 1024         // Number of vectors in unified buffer
`define BUFFER_WIDTH (`DATA_WIDTH * `ARRAY_SIZE)  // 1024 vectors × N elements

// ----------------------------------------------------------------------------
// Instruction Register Parameters
// ----------------------------------------------------------------------------
// Metadata/config loaded before each computation
`define INSTR_WIDTH 16            // Width per instruction field
`define INSTR_SIZE `ARRAY_SIZE    // Vector of size N

// Bit-width modes for SWAR packing
typedef enum logic [1:0] {
    MODE_INT4  = 2'b00,           // 4×INT4 in UINT16 (4-bit quantization)
    MODE_INT8  = 2'b01,           // 2×INT8 in UINT16 (8-bit quantization)
    MODE_INT16 = 2'b10,           // 1×INT16 in UINT16 (16-bit, no packing)
    MODE_RSVD  = 2'b11            // Reserved
} precision_mode_t;

// Instruction register field definitions
typedef struct packed {
    precision_mode_t mode;        // [15:14] Bit-width mode
    logic [5:0]  scale_shift;     // [13:8]  Quantization scale (shift amount)
    logic [7:0]  zero_point;      // [7:0]   Quantization zero point
} quant_config_t;

// ----------------------------------------------------------------------------
// Control FSM States
// ----------------------------------------------------------------------------
typedef enum logic [2:0] {
    IDLE        = 3'b000,
    LOAD_INSTR  = 3'b001,         // Load instruction register
    LOAD_DATA   = 3'b010,         // Load input/weight data
    COMPUTE     = 3'b011,         // Execute systolic computation
    DRAIN       = 3'b100,         // Drain pipeline
    DONE        = 3'b101
} npu_state_t;

// ----------------------------------------------------------------------------
// PE Operation Modes
// ----------------------------------------------------------------------------
typedef enum logic [1:0] {
    OP_SWAR_MAC = 2'b00,          // Packed SWAR MAC (2×INT8 dot product)
    OP_RESET    = 2'b01,          // Reset accumulator
    OP_PASS     = 2'b10,          // Pass-through (bubble propagation)
    OP_IDLE     = 2'b11           // Idle state
} pe_op_t;

`endif // DEFINES_SV
