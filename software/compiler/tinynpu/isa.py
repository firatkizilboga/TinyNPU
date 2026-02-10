from enum import IntEnum


class Opcode(IntEnum):
    NOP = 0x0
    HALT = 0x1
    MATMUL = 0x2
    MOVE = 0x3

# MMIO Registers (from defines.sv)


class MMIOReg(IntEnum):
    STATUS = 0x00
    CMD = 0x04
    ADDR = 0x08
    ARG = 0x0C
    MMVR = 0x10

# Host Commands (from defines.sv)


class HostCmd(IntEnum):
    WRITE_MEM = 0x01
    READ_MEM = 0x02
    RUN = 0x03

# Status Codes (from defines.sv)


class Status(IntEnum):
    IDLE = 0x00
    BUSY = 0x01
    DATA_VALID = 0x02
    READY_WRITE = 0x03
    ERROR = 0xFE
    HALTED = 0xFF


# Instruction Width is 256 bits
INST_WIDTH = 256


def generate_host_messages(cmd_type, addr=0, arg=0, data_64=None):
    """
    Generates a list of (reg_addr, value) tuples representing MMIO writes.
    Matches the Doorbell mechanism (writing Byte 7 of MMVR triggers action).
    """
    messages = []

    # 1. Set Command
    messages.append((MMIOReg.CMD, cmd_type & 0xFF))

    # 2. Set Address
    messages.append((MMIOReg.ADDR, addr & 0xFFFF))

    # 3. Set Argument
    messages.append((MMIOReg.ARG, arg & 0xFFFFFFFF))

    # 4. Set MMVR (Data) - Writing MSB triggers the TPU action
    if data_64 is not None:
        # We simulate the 8-byte write sequence.
        # In a real driver, Byte 7 is written last.
        # For simplicity in this tool, we just record the full 64-bit value.
        messages.append((MMIOReg.MMVR, data_64))
    elif cmd_type == HostCmd.RUN:
        # For RUN, the doorbell might be implicit or tied to a dummy MMVR write
        # depending on RTL specifics. Based on spec 2.1, Doorbell is MSB of MMVR.
        messages.append((MMIOReg.MMVR, 0))  # Trigger RUN

    return messages


def pack_matmul(opcode, a_addr, b_addr, c_addr, m, k, n, bias_addr=0):
    """
    Packs a MATMUL instruction into a 256-bit integer.

    Format:
    [255:252] Opcode
    [247:232] A Base
    [231:216] B Base
    [215:200] C Base
    [183:168] M Total
    [167:152] K Total
    [151:136] N Total
    [135:120] Bias Base (0 if none)
    """
    instr = 0
    instr |= (opcode & 0xF) << 252
    instr |= (a_addr & 0xFFFF) << 232
    instr |= (b_addr & 0xFFFF) << 216
    instr |= (c_addr & 0xFFFF) << 200
    instr |= (m & 0xFFFF) << 168
    instr |= (k & 0xFFFF) << 152
    instr |= (n & 0xFFFF) << 136
    instr |= (bias_addr & 0xFFFF) << 120
    return instr


def pack_move(opcode, src, dest, count):
    """
    Packs a MOVE instruction into a 256-bit integer.

    Format:
    [255:252] Opcode
    [247:232] Source Addr
    [231:216] Dest Addr
    [215:200] Count
    """
    instr = 0
    instr |= (opcode & 0xF) << 252
    instr |= (src & 0xFFFF) << 232
    instr |= (dest & 0xFFFF) << 216
    instr |= (count & 0xFFFF) << 200
    return instr


def pack_simple(opcode):
    """
    Packs a simple instruction (NOP, HALT) into a 256-bit integer.
    """
    instr = 0
    instr |= (opcode & 0xF) << 252
    return instr
