import numpy as np
import os
import re
import json
from .isa import Opcode, HostCmd, PrecisionMode, MatMul, Move, Halt, generate_host_messages
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
        self.role = role
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

    def declare_data(self, name, data, precision=PrecisionMode.INT16):
        if name in self.symbols:
            raise ValueError(f"Symbol '{name}' already declared.")
        arr = np.array(data, dtype=np.uint16)
        if len(arr.shape) != 2:
            raise ValueError(f"Symbol '{name}' must be a 2D matrix.")
        self.symbols[name] = Symbol(name, arr.shape, precision, data=arr)

    def add_expected_result(self, name, data):
        self.expected_results[name] = np.array(data)

    def matmul(self, a_name, b_name, out_name, bias_name=None, shift=0, multiplier=1, activation=0, in_precision=PrecisionMode.INT16, out_precision=PrecisionMode.INT16, write_offset=0):
        # 1. Verification & Type Checking
        if a_name not in self.symbols:
            raise ValueError(f"Symbol '{a_name}' not found.")
        if b_name not in self.symbols:
            raise ValueError(f"Symbol '{b_name}' not found.")
        
        A, B = self.symbols[a_name], self.symbols[b_name]
        A.role, B.role = 'A', 'B'
        A.precision = B.precision = in_precision

        if A.shape[1] != B.shape[0]:
            raise ValueError(f"Dimension mismatch: {A.shape} and {B.shape}")

        # 2. Implicit Symbol Creation (Inference)
        if out_name not in self.symbols:
            out_shape = (A.shape[0], B.shape[1])
            self.symbols[out_name] = Symbol(out_name, out_shape, out_precision, role='C')
        
        if bias_name and bias_name not in self.symbols:
            bias_shape = (1, B.shape[1])
            self.symbols[bias_name] = Symbol(bias_name, bias_shape, PrecisionMode.INT16, role='BIAS')
        elif bias_name:
            self.symbols[bias_name].role = 'BIAS'
            self.symbols[bias_name].precision = PrecisionMode.INT16

        # 3. Create Instruction and calculate tiles
        instr = MatMul(a_name, b_name, out_name, bias_name, shift, multiplier, activation, in_precision, out_precision, write_offset)
        
        p_in = 1 << (2 - in_precision)
        instr.m = (A.shape[0] + self.array_size - 1) // self.array_size
        k_phys = (A.shape[1] + p_in - 1) // p_in
        instr.k = (k_phys + self.array_size - 1) // self.array_size
        instr.n = (B.shape[1] + self.array_size - 1) // self.array_size
        
        self.instructions.append(instr)

    def move(self, src_name, dest_name):
        if src_name not in self.symbols:
            raise ValueError(f"Source symbol '{src_name}' not found.")
        
        src = self.symbols[src_name]
        if dest_name not in self.symbols:
            self.symbols[dest_name] = Symbol(dest_name, src.shape, src.precision, role=src.role)
        
        self.instructions.append(Move(src_name, dest_name))

    def halt(self):
        self.instructions.append(Halt())

    def compile(self):
        # Pass 1: Finalize physical counts and addresses
        for name in sorted(self.symbols.keys()):
            sym = self.symbols[name]
            
            # Infer m, k, n for address allocation if not set
            p = 1 << (2 - sym.precision)
            m = (sym.shape[0] + self.array_size - 1) // self.array_size
            if sym.role == 'A':
                k = (sym.shape[1]//p + self.array_size - 1) // self.array_size
                n = 1
            elif sym.role == 'B':
                k = (sym.shape[0]//p + self.array_size - 1) // self.array_size
                n = (sym.shape[1] + self.array_size - 1) // self.array_size
                m = 1
            elif sym.role == 'BIAS':
                m, k = 1, 1
                n = (sym.shape[1] + self.array_size - 1) // self.array_size
            else: # C
                m = (sym.shape[0] + self.array_size - 1) // self.array_size
                n = (sym.shape[1] + self.array_size - 1) // self.array_size
                k = 1

            sym.word_count = self.packer.get_physical_word_count(sym.role, sym.precision, m, k, n)
            sym.addr = self.memory.allocate(name, sym.word_count)

        # Pass 2: Update instruction counts (for MOVE)
        for instr in self.instructions:
            if isinstance(instr, Move):
                instr.count = self.symbols[instr.src].word_count

        # Pass 3: Generate Memory Image and Instructions
        ub_image = []
        symbol_to_addr = {name: sym.addr for name, sym in self.symbols.items()}
        
        # Build UB image by packing each symbol
        for name in sorted(self.symbols.keys()):
            sym = self.symbols[name]
            
            # Re-calculate tiles for packing
            p = 1 << (2 - sym.precision)
            m = (sym.shape[0] + self.array_size - 1) // self.array_size
            if sym.role == 'A':
                k = (sym.shape[1]//p + self.array_size - 1) // self.array_size
                n = 1
            elif sym.role == 'B':
                k = (sym.shape[0]//p + self.array_size - 1) // self.array_size
                n = (sym.shape[1] + self.array_size - 1) // self.array_size
                m = 1
            elif sym.role == 'BIAS':
                m, k = 1, 1
                n = (sym.shape[1] + self.array_size - 1) // self.array_size
            else: # C
                m = (sym.shape[0] + self.array_size - 1) // self.array_size
                n = (sym.shape[1] + self.array_size - 1) // self.array_size
                k = 1

            packed_words = self.packer.pack(sym.data, sym.role, sym.precision, m, k, n)
            ub_image.extend(packed_words)

        binary_im = [instr.encode(symbol_to_addr) for instr in self.instructions]
        
        self.last_compiled = {"im": binary_im, "ub": ub_image}
        return self.last_compiled

    def save_npu(self, filename):
        if not hasattr(self, 'last_compiled'):
            self.compile()
        
        binary = self.last_compiled
        npu_data = {
            "config": {
                "array_size": self.array_size,
                "buffer_width": self.buffer_width,
                "im_base": self.im_base_addr
            },
            "symbols": {
                name: {
                    "addr": sym.addr,
                    "shape": list(sym.shape),
                    "role": sym.role,
                    "precision": int(sym.precision)
                } for name, sym in self.symbols.items()
            },
            "im": [f"0x{inst:064x}" for inst in binary['im']],
            "ub": [f"0x{word:0{self.buffer_width//4}x}" for word in binary['ub']],
            "expected": {
                name: self.expected_results[name].tolist()
                for name in self.expected_results if name in self.symbols
            }
        }
        with open(filename, "w") as f:
            json.dump(npu_data, f, indent=2)
        print(f"Saved NPU program to {filename}")
