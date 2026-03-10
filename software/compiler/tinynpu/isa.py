from enum import IntEnum

class Opcode(IntEnum):
    NOP = 0x0
    HALT = 0x1
    MATMUL = 0x2
    MOVE = 0x3

class PrecisionMode(IntEnum):
    INT4 = 0
    INT8 = 1
    INT16 = 2

class Instruction:
    def encode(self, symbol_to_addr):
        raise NotImplementedError()

class MatMul(Instruction):
    def __init__(self, a, b, c, bias=None, shift=0, multiplier=1, activation=0, in_prec=PrecisionMode.INT16, out_prec=PrecisionMode.INT16, write_offset=0):
        self.a = a
        self.b = b
        self.c = c
        self.bias = bias
        self.shift = shift
        self.multiplier = multiplier
        self.activation = activation
        self.in_prec = in_prec
        self.out_prec = out_prec
        self.write_offset = write_offset
        
        # Tile dimensions (logical) will be set by the compiler during inference
        self.m = 0
        self.k = 0
        self.n = 0

    def encode(self, symbol_to_addr):
        a_addr = symbol_to_addr[self.a]
        b_addr = symbol_to_addr[self.b]
        c_addr = symbol_to_addr[self.c]
        bias_addr = symbol_to_addr[self.bias] if self.bias else 0xFFFF
        
        instr = 0
        instr |= (Opcode.MATMUL & 0xF) << 252
        instr |= (a_addr & 0xFFFF) << 232
        instr |= (b_addr & 0xFFFF) << 216
        instr |= (c_addr & 0xFFFF) << 200
        instr |= (self.m & 0xFFFF) << 168
        instr |= (self.k & 0xFFFF) << 152
        instr |= (self.n & 0xFFFF) << 136
        instr |= (bias_addr & 0xFFFF) << 120
        instr |= (self.shift & 0xFF) << 112
        instr |= (self.multiplier & 0xFFFF) << 96
        instr |= (self.activation & 0xFF) << 88
        instr |= (self.out_prec & 0x3) << 86
        instr |= (self.write_offset & 0x3) << 84
        instr |= (self.in_prec & 0x3) << 82
        return instr

class Move(Instruction):
    def __init__(self, src, dest):
        self.src = src
        self.dest = dest
        self.count = 0 # Words count, inferred by compiler

    def encode(self, symbol_to_addr):
        src_addr = symbol_to_addr[self.src]
        dest_addr = symbol_to_addr[self.dest]
        
        instr = 0
        instr |= (Opcode.MOVE & 0xF) << 252
        instr |= (src_addr & 0xFFFF) << 232
        instr |= (dest_addr & 0xFFFF) << 216
        instr |= (self.count & 0xFFFF) << 200
        return instr

class Halt(Instruction):
    def encode(self, symbol_to_addr):
        instr = 0
        instr |= (Opcode.HALT & 0xF) << 252
        return instr

# --- Legacy Functions for compatibility if needed ---
def pack_matmul(opcode, a_addr, b_addr, c_addr, m, k, n, bias_addr=0xFFFF, shift=0, multiplier=1, activation=0, in_precision=PrecisionMode.INT16, out_precision=PrecisionMode.INT16, write_offset=0):
    instr = 0
    instr |= (opcode & 0xF) << 252
    instr |= (a_addr & 0xFFFF) << 232
    instr |= (b_addr & 0xFFFF) << 216
    instr |= (c_addr & 0xFFFF) << 200
    instr |= (m & 0xFFFF) << 168
    instr |= (k & 0xFFFF) << 152
    instr |= (n & 0xFFFF) << 136
    instr |= (bias_addr & 0xFFFF) << 120
    instr |= (shift & 0xFF) << 112
    instr |= (multiplier & 0xFFFF) << 96
    instr |= (activation & 0xFF) << 88
    instr |= (out_precision & 0x3) << 86
    instr |= (write_offset & 0x3) << 84
    instr |= (in_precision & 0x3) << 82
    return instr

def pack_move(opcode, src, dest, count):
    instr = 0
    instr |= (opcode & 0xF) << 252
    instr |= (src & 0xFFFF) << 232
    instr |= (dest & 0xFFFF) << 216
    instr |= (count & 0xFFFF) << 200
    return instr

def pack_simple(opcode):
    instr = 0
    instr |= (opcode & 0xF) << 252
    return instr

# --- MMIO Helpers ---
class MMIOReg(IntEnum):
    STATUS = 0x00
    CMD = 0x04
    ADDR = 0x08
    ARG = 0x0C
    MMVR = 0x10

class HostCmd(IntEnum):
    WRITE_MEM = 0x01
    READ_MEM = 0x02
    RUN = 0x03

def generate_host_messages(cmd_type, addr=0, arg=0, data_64=None):
    messages = []
    messages.append((MMIOReg.CMD, cmd_type & 0xFF))
    messages.append((MMIOReg.ADDR, addr & 0xFFFF))
    messages.append((MMIOReg.ARG, arg & 0xFFFFFFFF))
    if data_64 is not None:
        messages.append((MMIOReg.MMVR, data_64))
    elif cmd_type == HostCmd.RUN:
        messages.append((MMIOReg.MMVR, 0))
    return messages
