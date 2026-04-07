import numpy as np
import os
import re
import json
from .isa import Opcode, HostCmd, PrecisionMode, MatMul, Move, Halt, OutputLayout, generate_host_messages
from .packer import Packer
from .memory import MemoryManager

class HardwareConfig:
    """Parses hardware parameters from defines.sv to ensure compiler sync."""
    def __init__(self, defines_path=None):
        if defines_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            defines_path = os.path.join(base_dir, "rtl", "defines.sv")

        self.params = {
            'ARRAY_SIZE': 8,
            'DATA_WIDTH': 16,
            'BUFFER_WIDTH': 128,
            'IM_BASE_ADDR': 0x8000
        }

        if os.path.exists(defines_path):
            with open(defines_path, "r") as f:
                for line in f:
                    match = re.search(r'`define\s+(\w+)\s+([\d\'\wh\+\*\(\)\$]+)', line)
                    if match:
                        name, val = match.groups()
                        val = val.split("//")[0].strip()
                        if "'h" in val:
                            try: val = int(val.split("'h")[1], 16)
                            except: continue
                        else:
                            try: val = int(re.sub(r'[`\(\)]', '', val))
                            except ValueError: continue
                        self.params[name] = val

        if 'BUFFER_WIDTH' not in self.params:
            self.params['BUFFER_WIDTH'] = self.params['DATA_WIDTH'] * self.params['ARRAY_SIZE']

class Symbol:
    def __init__(self, name, shape, precision=PrecisionMode.INT16, role='C', data=None):
        self.name = name
        self.shape = shape
        self.precision = precision
        self.storage_role = role
        self.data = data
        self.addr = None
        self.word_count = 0

class TinyNPUProgram:
    def __init__(self, defines_path=None):
        self.hw = HardwareConfig(defines_path)
        self.array_size = self.hw.params['ARRAY_SIZE']
        self.buffer_width = self.hw.params['BUFFER_WIDTH']
        self.im_base_addr = self.hw.params['IM_BASE_ADDR']

        self.symbols = {} # name -> Symbol
        self.instructions = []
        self.expected_results = {}
        
        self.memory = MemoryManager()
        self.packer = Packer(self.array_size)

    def declare_data(self, name, data, precision=PrecisionMode.INT16, role='A'):
        if name in self.symbols:
            raise ValueError(f"Symbol '{name}' already declared.")
        if role == 'BIAS':
            arr = np.array(data)
            if np.any(arr > np.int64(0x7FFFFFFF)) or np.any(arr < np.int64(-0x80000000)):
                raise ValueError(f"Bias symbol '{name}' contains values outside signed 32-bit range.")
            arr = arr.astype(np.int32)
        else:
            arr = np.array(data, dtype=np.int16)
        if len(arr.shape) != 2:
            raise ValueError(f"Symbol '{name}' must be a 2D matrix.")
        self.symbols[name] = Symbol(name, arr.shape, precision, role=role, data=arr)

    def declare_image(self, name, data, h, w, c, precision=PrecisionMode.INT16):
        """Declares a 3D image and stores its logical dimensions."""
        arr = np.array(data, dtype=np.int16).reshape(h, w, c)
        sym = Symbol(name, (h*w, c), precision, role='B', data=arr.reshape(h*w, c))
        sym.dims = (h, w, c)
        self.symbols[name] = sym

    def declare_kernel(self, name, data, kh, kw, c, oc, precision=PrecisionMode.INT16):
        """Declares a 4D kernel and stores its logical dimensions."""
        arr = np.array(data, dtype=np.int16).reshape(kh, kw, c, oc)
        sym = Symbol(name, (kh*kw*c, oc), precision, role='A', data=arr.reshape(-1, oc))
        sym.dims = (kh, kw, c, oc)
        self.symbols[name] = sym

    def add_expected_result(self, name, data):
        self.expected_results[name] = np.array(data)

    def matmul(
        self,
        a_name,
        b_name,
        out_name,
        bias_name=None,
        shift=0,
        multiplier=1,
        activation=0,
        in_precision=PrecisionMode.INT16,
        out_precision=PrecisionMode.INT16,
        write_offset=0,
        h_gelu_x_scale_shift=7,
        output_layout=OutputLayout.C,
        conv_stream=None,
    ):
        if a_name not in self.symbols: raise ValueError(f"Symbol '{a_name}' not found.")
        if b_name not in self.symbols: raise ValueError(f"Symbol '{b_name}' not found.")
        A, B = self.symbols[a_name], self.symbols[b_name]
        # Keep generated/output tensors in Role C to preserve verification semantics.
        # Only coerce source tensors that are not Role C.
        if A.storage_role != 'C':
            A.storage_role = 'A'
        if B.storage_role != 'C':
            B.storage_role = 'B'
        if conv_stream is None and A.shape[1] != B.shape[0]:
            raise ValueError(f"Dimension mismatch: {A.shape} and {B.shape}")
        if conv_stream is not None:
            in_h = int(conv_stream["input_h"])
            in_w = int(conv_stream["input_w"])
            in_c = int(conv_stream["input_c"])
            kernel = int(conv_stream["kernel_size"])
            stride = int(conv_stream["stride"])
            padding = int(conv_stream["padding"])
            if A.shape != (in_h * in_w, in_c):
                raise ValueError(f"conv_stream expects lhs shape {(in_h * in_w, in_c)}, got {A.shape}.")
            if kernel <= 0 or stride <= 0:
                raise ValueError("conv_stream requires kernel_size > 0 and stride > 0.")
            out_h = ((in_h + 2 * padding - kernel) // stride) + 1
            out_w = ((in_w + 2 * padding - kernel) // stride) + 1
            if out_h <= 0 or out_w <= 0:
                raise ValueError("conv_stream produced non-positive output spatial dimensions.")
            expected_out_rows = out_h * out_w
            if B.shape[0] != (kernel * kernel * in_c):
                raise ValueError(
                    f"conv_stream expects rhs rows {kernel * kernel * in_c}, got {B.shape[0]}."
                )
        else:
            expected_out_rows = A.shape[0]
        if output_layout == OutputLayout.A:
            out_role = 'A'
        elif output_layout == OutputLayout.B:
            out_role = 'B'
        else:
            out_role = 'C'
        if out_name not in self.symbols:
            out_rows = expected_out_rows if conv_stream is not None else A.shape[0]
            out_shape = (out_rows, B.shape[1])
            self.symbols[out_name] = Symbol(out_name, out_shape, out_precision, role=out_role)
        else:
            out_rows = expected_out_rows if conv_stream is not None else A.shape[0]
            self.symbols[out_name].shape = (out_rows, B.shape[1])
            self.symbols[out_name].storage_role = out_role
            self.symbols[out_name].precision = out_precision
        if bias_name:
            if bias_name not in self.symbols:
                bias_shape = (1, B.shape[1])
                self.symbols[bias_name] = Symbol(bias_name, bias_shape, PrecisionMode.INT16, role='BIAS')
            else:
                self.symbols[bias_name].storage_role = 'BIAS'
        instr = MatMul(
            a_name,
            b_name,
            out_name,
            bias_name,
            shift,
            multiplier,
            activation,
            in_precision,
            out_precision,
            write_offset,
            h_gelu_x_scale_shift,
            output_layout,
            conv_stream=conv_stream,
        )
        p_in = 1 << (2 - in_precision)
        instr.m = (expected_out_rows + self.array_size - 1) // self.array_size
        logical_k = B.shape[0] if conv_stream is not None else A.shape[1]
        k_phys = (logical_k + p_in - 1) // p_in
        instr.k = (k_phys + self.array_size - 1) // self.array_size
        instr.n = (B.shape[1] + self.array_size - 1) // self.array_size
        self.instructions.append(instr)

    def conv2d_im2col(self, image_name, kernel_name, out_name, bias_name=None, stride=1, padding=0, shift=0, multiplier=1, activation=0, in_precision=PrecisionMode.INT16, out_precision=PrecisionMode.INT16):
        img_sym, ker_sym = self.symbols[image_name], self.symbols[kernel_name]
        H, W, C = img_sym.dims
        KH, KW, _, OC = ker_sym.dims
        OH = (H + 2*padding - KH) // stride + 1
        OW = (W + 2*padding - KW) // stride + 1
        print(f"Applying im2col: Image({H}x{W}x{C}) * Kernel({KH}x{KW}x{OC}) -> Output({OH}x{OW}x{OC})")
        img_data = img_sym.data.reshape(H, W, C)
        if padding > 0: img_data = np.pad(img_data, ((padding, padding), (padding, padding), (0, 0)), mode='constant')
        col_matrix = []
        for y in range(0, H + 2*padding - KH + 1, stride):
            for x in range(0, W + 2*padding - KW + 1, stride):
                patch = img_data[y:y+KH, x:x+KW, :]
                col_matrix.append(patch.transpose(2, 0, 1).flatten())
        col_matrix = np.array(col_matrix, dtype=np.int16) # (HW, K)
        kernel_matrix = ker_sym.data.reshape(-1, OC) # (K, OC)
        col_sym_name, ker_flat_name = f"_{image_name}_im2col", f"_{kernel_name}_flat"
        self.declare_data(col_sym_name, col_matrix, precision=in_precision, role='A')
        self.declare_data(ker_flat_name, kernel_matrix, precision=in_precision, role='B')
        self.matmul(col_sym_name, ker_flat_name, out_name, bias_name=bias_name, shift=shift, multiplier=multiplier, activation=activation, in_precision=in_precision, out_precision=out_precision)
        self.symbols[out_name].dims = (OH, OW, OC)

    def move(self, src_name, dest_name):
        if src_name not in self.symbols: raise ValueError(f"Source symbol '{src_name}' not found.")
        src = self.symbols[src_name]
        if dest_name not in self.symbols: self.symbols[dest_name] = Symbol(dest_name, src.shape, src.precision, role=src.storage_role)
        self.instructions.append(Move(src_name, dest_name))

    def halt(self): self.instructions.append(Halt())

    def compile(self):
        for name in sorted(self.symbols.keys()):
            sym = self.symbols[name]
            p = 1 << (2 - sym.precision)
            sz = self.array_size
            m = (sym.shape[0] + sz - 1) // sz
            if sym.storage_role == 'A':
                k = (sym.shape[1]//p + sz - 1) // sz
                n = 1
            elif sym.storage_role == 'B':
                k = (sym.shape[0]//p + sz - 1) // sz
                n = (sym.shape[1] + sz - 1) // sz
                m = 1
            elif sym.storage_role == 'BIAS':
                m, k, n = 1, 1, (sym.shape[1] + sz - 1) // sz
            else: # C
                m, n, k = (sym.shape[0] + sz - 1) // sz, (sym.shape[1] + sz - 1) // sz, 1
            sym.word_count = self.packer.get_physical_word_count(sym.storage_role, sym.precision, m, k, n)
            if sym.addr is None:  # respect pre-assigned addresses from the memory planner
                sym.addr = self.memory.allocate(name, sym.word_count)
        for instr in self.instructions:
            if isinstance(instr, Move): instr.count = self.symbols[instr.src].word_count
        symbol_to_addr = {name: sym.addr for name, sym in self.symbols.items()}
        # Build UB image indexed by address so pre-assigned (non-sequential) addresses work.
        total_ub_words = max((sym.addr + sym.word_count for sym in self.symbols.values()), default=0)
        ub_image = [0] * total_ub_words
        for name, sym in self.symbols.items():
            p = 1 << (2 - sym.precision)
            sz = self.array_size
            m = (sym.shape[0] + sz - 1) // sz
            if sym.storage_role == 'A':
                k, n = (sym.shape[1]//p + sz - 1) // sz, 1
            elif sym.storage_role == 'B':
                k, n, m = (sym.shape[0]//p + sz - 1) // sz, (sym.shape[1] + sz - 1) // sz, 1
            elif sym.storage_role == 'BIAS':
                m, k, n = 1, 1, (sym.shape[1] + sz - 1) // sz
            else: # C
                m, n, k = (sym.shape[0] + sz - 1) // sz, (sym.shape[1] + sz - 1) // sz, 1
            packed_words = self.packer.pack(sym.data, sym.storage_role, sym.precision, m, k, n)
            for i, word in enumerate(packed_words):
                ub_image[sym.addr + i] = word
        binary_im = [instr.encode(symbol_to_addr) for instr in self.instructions]
        self.last_compiled = {"im": binary_im, "ub": ub_image}
        return self.last_compiled

    def save_npu(self, filename):
        if not hasattr(self, 'last_compiled'): self.compile()
        binary = self.last_compiled
        npu_data = {
            "config": {"array_size": self.array_size, "buffer_width": self.buffer_width, "im_base": self.im_base_addr},
            "symbols": {name: {"addr": sym.addr, "shape": list(sym.shape), "role": sym.storage_role, "precision": int(sym.precision)} for name, sym in self.symbols.items()},
            "im": [f"0x{inst:064x}" for inst in binary['im']],
            "ub": [f"0x{word:0{self.buffer_width//4}x}" for word in binary['ub']],
            "expected": {name: self.expected_results[name].tolist() for name in self.expected_results if name in self.symbols}
        }
        with open(filename, "w") as f: json.dump(npu_data, f, indent=2)
        print(f"Saved NPU program to {filename}")
