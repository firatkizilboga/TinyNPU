# TinyNPU Control Plane & ISA Specification

## 1. Overview
Transforming TinyNPU from a passive compute engine into an autonomous accelerator. The system features a **Memory-Mapped I/O (MMIO)** interface for the host and an internal **Control Unit (CU)** executing a dedicated instruction set.

## 2. Host Interface (Memory Map)
The Host communicates with the TPU via a set of memory-mapped registers.

| Offset | Register Name | R/W | Description |
| :--- | :--- | :--- | :--- |
| `0x00` | **STATUS_REG** | R | Current state of the TPU (see Status Codes). |
| `0x04` | **CMD_REG** | W | Command register to trigger Host-side actions. |
| `0x08` | **ADDR_REG** | RW | Target address in **Unified Memory**. |
| `0x0C` | **ARG_REG** | RW | Optional argument (e.g., Burst Length, PC Start Addr). |
| `0x10` | **MMVR** | RW | **Memory Mapped Vector Register**. Data staging window (64-bit / 8-bytes). |

### 2.1 The "Doorbell" Mechanism (Implicit Trigger)
To minimize Host overhead, the TPU uses an **Implicit Trigger** on the MMVR:
*   The Host writes Bytes 0-6 of the `MMVR` to set up the data or instruction parameters.
*   The write to **Byte 7** (the MSB) acts as the **Doorbell**. 
*   **Hardware Action:** On the clock cycle following a write to `MMVR[Byte 7]`, the TPU captures the full 64-bit value and executes the action currently selected in `CMD_REG` (e.g., `CMD_WRITE_MEM` or `CMD_DECODE_INST`).

### 2.2 Status Codes (STATUS_REG)
*   `0x00` **IDLE (WAITING-FOR-IO)**: TPU is asleep, waiting for the Doorbell.
*   `0x01` **BUSY**: TPU is executing instructions.
*   `0x02` **DATA_VALID**: Read operation complete; data is ready in MMVR.
*   `0x03` **READY_FOR_WRITE**: TPU is ready to accept data into MMVR.
*   `0xFE` **ERROR**: Invalid instruction or illegal state.
*   `0xFF` **HALTED**: Execution finished, waiting for reset/clear.

### 2.3 Host Commands (CMD_REG)
*   `CMD_WRITE_MEM`: Write content of `MMVR` to `Unified_Memory[ADDR_REG]`.
*   `CMD_READ_MEM`: Fetch `Unified_Memory[ADDR_REG]` into `MMVR`. Sets Status to `DATA_VALID` when done.
*   `CMD_RUN`: Start executing instructions from `Unified_Memory[ARG_REG]`.

---

## 3. Instruction Set Architecture (ISA)
The Control Unit fetches 32-bit (or 64-bit) instructions from the **Unified Memory**.
*   **No Branching:** Execution is strictly linear until `HALT`.
*   **Unified Model:** Code and Data share the same address space.

### 3.1 Opcode Table (Draft)

| Opcode | Mnemonic | Operands | Description |
| :--- | :--- | :--- | :--- |
| `0x00` | `NOP` | - | No Operation. |
| `0x01` | `HALT` | - | Stop execution and enter **WAITING-FOR-IO** state. |
| `0x02` | `MATMUL` | *Complex* | **Super Instruction**: Tiled Compute + Bias + ReLU + Quant + Writeback. |
| `0x03` | `MOVE` | `Src`, `Dest`, `Len` | **Internal DMA**: Copy `Len` words from `Src` to `Dest` within UB. |
| `0x04` | `CFG_Q` | `Mode`, `Shift` | Configure Quantization (Precision, Shift). |

### 3.2 The `MATMUL` Instruction (Super-Instruction)
A single instruction orchestrates the entire pipeline: Fetch -> Compute -> PPU -> Writeback.

**Structure (Multi-Word Instruction):**
*   **Word 0:** `OPCODE` | `FLAGS` (Acc_Clear, Drain_En, ReLU_En)
*   **Word 1:** `TILE_A_ADDR` (Input) | `TILE_A_STRIDE`
*   **Word 2:** `TILE_B_ADDR` (Weight) | `TILE_B_STRIDE`
*   **Word 3:** `OUTPUT_ADDR` (Result Writeback)
*   **Word 4:** `BIAS_ADDR` | `QUANT_SCALE`
*   **Word 5:** `LOOP_COUNTS` (M, K, N)

**Execution Flow (Hardware Sequencer):**
1.  **Load:** Fetch A and B tiles from memory.
2.  **Compute:** Run Systolic Array for K cycles.
3.  **Drain:** Flush accumulators.
4.  **PPU:** Add Bias (from `BIAS_ADDR`) -> Apply ReLU (if `FLAGS.ReLU`) -> Quantize (using `QUANT_SCALE`).
5.  **Writeback:** Store final 8-bit/16-bit result to `OUTPUT_ADDR`.

---

## 4. Hardware Modules Required

1.  **Instruction Decoder:** Parses the instruction from Unified Memory.
2.  **Address Generation Unit (AGU):**
    *   Counters for M, K, N loops.
    *   Adder to calculate `Base + (Stride * Counter)`.
    *   **Shared by `MOVE` instruction** for linear address generation.
3.  **Master State Machine:**
    *   **FETCH**: Get instruction from Memory.
    *   **DECODE**: Parse Opcode.
    *   **EXECUTE**:
        *   If `MATMUL`: Hand over control to the *Systolic Sequencer*.
        *   If `MOVE`: Perform word-by-word copy using AGU.
        *   If `HALT`: Update Status Register to IDLE.
4.  **MMVR Interface:** A simpler arbiter to handle Host reads/writes vs. Internal reads/writes.

## 5. Design Considerations & Thoughts

*   **Handling Large Data (Burst Mode):** 
    *   To support 8-bit aligned hosts writing to wide 64-bit/128-bit internal buffers, the `CMD_REG` can include a `BURST_EN` bit. When active, every write to `MMVR` automatically increments `ADDR_REG`.
*   **Internal MOVE Command:** 
    *   This allows "Ping-Pong" buffering. While one MATMUL is running, the `MOVE` engine (or a DMA) could be moving the next layer's weights into the active area. 
    *   *Constraint:* We need to decide if the UB is single-ported or dual-ported. If single-ported, `MOVE` and `MATMUL` cannot happen at the same time.
*   **Synchronization:**
    *   We need a `FENCE` or `BARRIER` instruction if we add parallel data loading while computing. For now, `HALT` is sufficient (Host waits for HALT before reading results).
*   **Interrupts:**
    *   Instead of just polling `STATUS_REG`, add an `IRQ` line to the host CPU for "Done" or "Error".