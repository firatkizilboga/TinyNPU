from enum import IntEnum

class Opcode(IntEnum):
    NOP = 0x0
    HALT = 0x1
    MATMUL = 0x2
    MOVE = 0x3
    XFORM = 0x4

class PrecisionMode(IntEnum):
    INT4 = 0
    INT8 = 1
    INT16 = 2


class ActivationMode(IntEnum):
    NONE = 0
    RELU = 1
    SIGMOID = 2
    H_GELU = 3


class OutputLayout(IntEnum):
    C = 0
    A = 1
    B = 2


class WritebackMode(IntEnum):
    NORMAL = 0
    V_CACHE_APPEND_INT16 = 1
    K_CACHE_APPEND_INT16 = 2

class BReadMode(IntEnum):
    NORMAL = 0
    K_CACHE_INT16 = 1


class XformMode(IntEnum):
    NONE = 0
    Q_F32_I16 = 1
    DQ_I16_F32 = 2


# Matches rtl/ppu.sv PPU_HGELU_SHIFT_WIDTH. Larger values need a wider
# dynamic shifter in the activation path and are not part of the hardware ABI.
MAX_H_GELU_X_SCALE_SHIFT = 15

# Matches rtl/ppu.sv PPU_PRODUCT_WIDTH - 1. Larger values would be clamped by
# the PPU, so reject them at the ABI boundary instead of silently changing math.
MAX_MATMUL_RESCALE_SHIFT = 63


def _validate_h_gelu_x_scale_shift(value):
    value = int(value)
    if value < 0 or value > MAX_H_GELU_X_SCALE_SHIFT:
        raise ValueError(
            f"h_gelu_x_scale_shift={value} is outside the supported hardware range "
            f"0..{MAX_H_GELU_X_SCALE_SHIFT}."
        )
    return value


def _validate_matmul_rescale_shift(value):
    value = int(value)
    if value < 0 or value > MAX_MATMUL_RESCALE_SHIFT:
        raise ValueError(
            f"shift={value} is outside the supported matmul rescale range "
            f"0..{MAX_MATMUL_RESCALE_SHIFT}."
        )
    return value


class Instruction:
    def encode(self, symbol_to_addr):
        raise NotImplementedError()

class MatMul(Instruction):
    def __init__(
        self,
        a,
        b,
        c,
        bias=None,
        shift=0,
        multiplier=1,
        activation=0,
        in_prec=PrecisionMode.INT16,
        out_prec=PrecisionMode.INT16,
        write_offset=0,
        h_gelu_x_scale_shift=7,
        output_layout=OutputLayout.C,
        writeback_mode=WritebackMode.NORMAL,
        output_word_offset=0,
        b_word_offset=0,
        b_read_mode=BReadMode.NORMAL,
    ):
        self.a = a
        self.b = b
        self.c = c
        self.bias = bias
        self.shift = _validate_matmul_rescale_shift(shift)
        self.multiplier = multiplier
        self.activation = activation
        self.in_prec = in_prec
        self.out_prec = out_prec
        self.write_offset = write_offset
        self.h_gelu_x_scale_shift = _validate_h_gelu_x_scale_shift(h_gelu_x_scale_shift)
        self.output_layout = output_layout
        self.writeback_mode = writeback_mode
        self.output_word_offset = output_word_offset
        self.b_word_offset = b_word_offset
        self.b_read_mode = b_read_mode
        
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
        instr |= (self.writeback_mode & 0xF) << 248
        instr |= (a_addr & 0xFFFF) << 232
        instr |= (b_addr & 0xFFFF) << 216
        instr |= (c_addr & 0xFFFF) << 200
        instr |= (self.output_word_offset & 0xFFFF) << 184
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
        instr |= (self.h_gelu_x_scale_shift & 0xFF) << 74
        instr |= (self.output_layout & 0x3) << 72
        instr |= (self.b_word_offset & 0xFFFF) << 56
        instr |= (self.b_read_mode & 0xF) << 52
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


class Xform(Instruction):
    def __init__(self, src, dest, mode=XformMode.Q_F32_I16, multiplier=1, shift=0):
        self.src = src
        self.dest = dest
        self.mode = mode
        self.multiplier = multiplier
        self.shift = shift
        self.count = 0  # Words count, inferred by compiler

    def encode(self, symbol_to_addr):
        src_addr = symbol_to_addr[self.src]
        dest_addr = symbol_to_addr[self.dest]
        instr = 0
        instr |= (Opcode.XFORM & 0xF) << 252
        instr |= (self.mode & 0xF) << 248
        instr |= (src_addr & 0xFFFF) << 232
        instr |= (dest_addr & 0xFFFF) << 216
        instr |= (self.count & 0xFFFF) << 200
        instr |= (self.multiplier & 0xFFFF) << 184
        instr |= (self.shift & 0xFF) << 176
        return instr


class XformRopeK16(Instruction):
    """Retired hardware RoPE transform placeholder.

    Kept only so older serialized/programmatic callers fail with a clear error
    instead of encoding an unsupported XFORM mode.
    """
    def __init__(self, src, dest, cs_addr, row_index=0):
        self.src = src        # symbol name of K tensor
        self.dest = dest      # symbol name of output K (same as src for in-place)
        self.cs_addr = cs_addr  # symbol name of cos/sin table tensor
        self.row_index = row_index
        self.half_count = 0   # filled in by program.compile()
        self.src_word_offset = 0
        self.dest_word_offset = 0

    def encode(self, symbol_to_addr):
        raise NotImplementedError("Hardware RoPE XFORM has been removed; lower RoPE as a CPU host op.")

# --- Legacy Functions for compatibility if needed ---
def pack_matmul(
    opcode,
    a_addr,
    b_addr,
    c_addr,
    m,
    k,
    n,
    bias_addr=0xFFFF,
    shift=0,
    multiplier=1,
    activation=0,
    in_precision=PrecisionMode.INT16,
    out_precision=PrecisionMode.INT16,
    write_offset=0,
    h_gelu_x_scale_shift=7,
    output_layout=OutputLayout.C,
    writeback_mode=WritebackMode.NORMAL,
    output_word_offset=0,
    b_word_offset=0,
    b_read_mode=BReadMode.NORMAL,
):
    h_gelu_x_scale_shift = _validate_h_gelu_x_scale_shift(h_gelu_x_scale_shift)
    shift = _validate_matmul_rescale_shift(shift)
    instr = 0
    instr |= (opcode & 0xF) << 252
    instr |= (writeback_mode & 0xF) << 248
    instr |= (a_addr & 0xFFFF) << 232
    instr |= (b_addr & 0xFFFF) << 216
    instr |= (c_addr & 0xFFFF) << 200
    instr |= (output_word_offset & 0xFFFF) << 184
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
    instr |= (h_gelu_x_scale_shift & 0xFF) << 74
    instr |= (output_layout & 0x3) << 72
    instr |= (b_word_offset & 0xFFFF) << 56
    instr |= (b_read_mode & 0xF) << 52
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


def pack_xform(opcode, mode, src, dest, count, multiplier=1, shift=0):
    instr = 0
    instr |= (opcode & 0xF) << 252
    instr |= (mode & 0xF) << 248
    instr |= (src & 0xFFFF) << 232
    instr |= (dest & 0xFFFF) << 216
    instr |= (count & 0xFFFF) << 200
    instr |= (multiplier & 0xFFFF) << 184
    instr |= (shift & 0xFF) << 176
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
