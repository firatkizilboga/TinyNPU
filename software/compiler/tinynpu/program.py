import numpy as np
import os
import re
import json
from .isa import Opcode, HostCmd, PrecisionMode, generate_host_messages, pack_matmul, pack_move, pack_simple


class HardwareConfig:
    """Parses hardware parameters from defines.sv to ensure compiler sync."""

    def __init__(self, defines_path=None):
        if defines_path is None:
            # Try to find defines.sv relative to this file
            base_dir = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            defines_path = os.path.join(base_dir, "rtl", "defines.sv")

        self.params = {
            'ARRAY_SIZE': 8,
            'DATA_WIDTH': 16,
            'BUFFER_WIDTH': 128,
            'IM_BASE_ADDR': 0x8000
        }

        if os.path.exists(defines_path):
            with open(defines_path, "r") as f:
                content = f.read()
                # Match `define NAME VALUE
                matches = re.findall(r'`define\s+(\w+)\s+([\w\'h]+)', content)
                for name, val in matches:
                    # Handle hex values like 16'h8000
                    if "'h" in val:
                        val = int(val.split("'h")[1], 16)
                    else:
                        try:
                            # Handle simple integer defines
                            val = int(val)
                        except ValueError:
                            # Handle expressions like (`DATA_WIDTH * `ARRAY_SIZE)
                            if '*' in val:
                                parts = val.replace('`','').replace('(','').replace(')','').split('*')
                                try:
                                    res = 1
                                    for p in parts:
                                        res *= self.params.get(p.strip(), 1)
                                    val = res
                                except:
                                    continue
                            else:
                                continue
                    self.params[name] = val

        # Derived if not explicitly found
        if 'BUFFER_WIDTH' not in matches:
            self.params['BUFFER_WIDTH'] = self.params['DATA_WIDTH'] * \
                self.params['ARRAY_SIZE']


class TinyNPUProgram:
    def __init__(self, defines_path=None):
        self.hw = HardwareConfig(defines_path)
        self.array_size = self.hw.params['ARRAY_SIZE']
        self.buffer_width = self.hw.params['BUFFER_WIDTH']
        self.im_base_addr = self.hw.params['IM_BASE_ADDR']

        self.instructions = []
        self.raw_symbols = {}
        self.symbol_to_addr = {}
        self.ub_data = []
        self.expected_results = {}  # name -> np.array

    def add_expected_result(self, name, data):
        """Sets the golden reference for an output symbol."""
        self.expected_results[name] = np.array(data, dtype=np.uint16)

    def declare_data(self, name, data):
        """
        Declares a matrix symbol with its initial data.
        Data should be a 2D numpy array or list of lists.
        """
        if name in self.raw_symbols:
            raise ValueError(f"Symbol '{name}' already declared.")

        arr = np.array(data, dtype=np.uint16)
        if len(arr.shape) != 2:
            raise ValueError(f"Symbol '{name}' must be a 2D matrix.")

        self.raw_symbols[name] = {
            'data': arr,
            'shape': arr.shape
        }

    def matmul(self, a_name, b_name, out_name, bias_name=None, shift=0, multiplier=1, activation=0, in_precision=PrecisionMode.INT16, out_precision=PrecisionMode.INT16, write_offset=0):
        """
        Adds a MATMUL instruction: out = a * b + bias
        Precision: 0=INT4, 1=INT8, 2=INT16
        Write Offset: 0-3 (sub-word index for packing)
        """
        if a_name not in self.raw_symbols:
            raise ValueError(f"Symbol '{a_name}' not found.")
        if b_name not in self.raw_symbols:
            raise ValueError(f"Symbol '{b_name}' not found.")
        if bias_name and bias_name not in self.raw_symbols:
            raise ValueError(f"Bias symbol '{bias_name}' not found.")

        shape_a = self.raw_symbols[a_name]['shape']
        shape_b = self.raw_symbols[b_name]['shape']

        if shape_a[1] != shape_b[0]:
            raise ValueError(f"Dimension mismatch: {shape_a} and {shape_b}")

        # Precision elements per word
        p_in = 1 << (2 - in_precision)
        
        # Hardware uses N x N tiles of PACKED words
        m_tiles = (shape_a[0] + self.array_size - 1) // self.array_size
        # K is packed into words, so physical K dimension is reduced
        k_phys = (shape_a[1] + p_in - 1) // p_in
        k_tiles = (k_phys + self.array_size - 1) // self.array_size
        # N remains logical for tiling, as each pass produces sz outputs
        n_tiles = (shape_b[1] + self.array_size - 1) // self.array_size

        # If output symbol doesn't exist, record its expected shape
        if out_name not in self.raw_symbols:
            self.raw_symbols[out_name] = {
                'data': None,  # To be filled by HW
                'shape': (shape_a[0], shape_b[1])
            }

        self.instructions.append({
            'type': 'MATMUL',
            'a_name': a_name,
            'b_name': b_name,
            'c_name': out_name,
            'bias_name': bias_name,
            'm': m_tiles,
            'k': k_tiles,
            'n': n_tiles,
            'shift': shift,
            'multiplier': multiplier,
            'activation': activation,
            'in_precision': in_precision,
            'out_precision': out_precision,
            'write_offset': write_offset
        })

    def move(self, src_name, dest_name):
        """
        Adds a MOVE instruction: copies src matrix to dest.
        """
        if src_name not in self.raw_symbols:
            raise ValueError(f"Source symbol '{src_name}' not found.")

        shape = self.raw_symbols[src_name]['shape']

        if dest_name not in self.raw_symbols:
            self.raw_symbols[dest_name] = {
                'data': None,
                'shape': shape
            }

        self.instructions.append({
            'type': 'MOVE',
            'src_name': src_name,
            'dest_name': dest_name
        })

    def halt(self):
        self.instructions.append({'type': 'HALT'})

    def _pack_tiled(self, data, role, precision, m_tiles=0, k_tiles=0, n_tiles=0):
        """
        Packs matrix data into the format expected by the TPU hardware.
        """
        sz = self.array_size
        p = 1 << (2 - precision)
        bits = 16 // p
        mask = (1 << bits) - 1

        # Determine logical dimensions from tiles if data is None
        if data is None:
            if role == 'A':
                M, K = m_tiles * sz, k_tiles * sz * p
            elif role == 'B':
                K, N = k_tiles * sz * p, n_tiles * sz
            else: # C
                M, N = m_tiles * sz, n_tiles * sz
        else:
            M, K_or_N = data.shape
            if role == 'B':
                K, N = M, K_or_N
            else:
                M, K_or_N = M, K_or_N
                if role == 'A': K = K_or_N
                else: N = K_or_N

        if role == 'A':
            # Matrix A (Left): Each physical word = 8 rows, each lane = P packed K-elements
            kt_words = k_tiles * sz
            padded = np.zeros((m_tiles * sz, kt_words * p), dtype=np.int32)
            if data is not None:
                # Clamp data to fit padded if necessary (shouldn't be if tiles are correct)
                d_m, d_k = data.shape
                padded[:min(d_m, m_tiles*sz), :min(d_k, kt_words*p)] = data[:min(d_m, m_tiles*sz), :min(d_k, kt_words*p)]
            
            packed = []
            for m in range(m_tiles):
                for k in range(k_tiles):
                    tile = padded[m*sz:m*sz+sz, (k*sz*p):(k*sz+sz)*p]
                    for col_idx in range(sz):
                        word = 0
                        for i in range(sz):
                            start_k = col_idx * p
                            subword = 0
                            for bit_idx in range(p):
                                val = int(tile[i, start_k + bit_idx]) & mask
                                subword |= val << (bit_idx * bits)
                            word |= (subword & 0xFFFF) << (i * 16)
                        packed.append(word)
            return packed

        elif role == 'B':
            # Matrix B (Top): Each physical word = 8 cols, each lane = P packed K-elements
            kt_words = k_tiles * sz
            padded = np.zeros((kt_words * p, n_tiles * sz), dtype=np.int32)
            if data is not None:
                d_k, d_n = data.shape
                padded[:min(d_k, kt_words*p), :min(d_n, n_tiles*sz)] = data[:min(d_k, kt_words*p), :min(d_n, n_tiles*sz)]

            packed = []
            for k in range(k_tiles):
                for n in range(n_tiles):
                    tile = padded[(k*sz*p):(k*sz+sz)*p, n*sz:n*sz+sz]
                    for row_idx in range(sz):
                        word = 0
                        for i in range(sz):
                            start_k = row_idx * p
                            subword = 0
                            for bit_idx in range(p):
                                val = int(tile[start_k + bit_idx, i]) & mask
                                subword |= val << (bit_idx * bits)
                            word |= (subword & 0xFFFF) << (i * 16)
                        packed.append(word)
            return packed

        elif role == 'C':
            # Matrix C/Bias: Strided packing for N-tiles
            p_out = p
            nt_phys = (n_tiles + p_out - 1) // p_out
            padded_width = nt_phys * sz * p_out
            padded = np.zeros((m_tiles * sz, padded_width), dtype=np.int32)
            if data is not None:
                d_m, d_n = data.shape
                padded[:min(d_m, m_tiles*sz), :min(d_n, n_tiles*sz)] = data[:min(d_m, m_tiles*sz), :min(d_n, n_tiles*sz)]
            
            packed = []
            for m in range(m_tiles):
                for nt in range(nt_phys):
                    for i in range(sz):
                        row_idx = m * sz + i
                        word = 0
                        for j in range(sz):
                            start_n = nt * (sz * p_out) + j
                            subword = 0
                            for bit_idx in range(p_out):
                                val = int(padded[row_idx, start_n + bit_idx * sz]) & mask
                                subword |= val << (bit_idx * bits)
                            word |= (subword & 0xFFFF) << (j * 16)
                        packed.append(word)
            return packed
        elif role == 'BIAS':
            # Bias is 16-bit, one word per logical N-tile
            # data shape is (1, N) or (N,)
            if data is None:
                return [0] * n_tiles
            
            flat_data = data.flatten()
            N = flat_data.shape[0]
            padded_N = n_tiles * sz
            padded = np.zeros(padded_N, dtype=np.int32)
            padded[:min(N, padded_N)] = flat_data[:min(N, padded_N)]
            
            packed = []
            for n in range(n_tiles):
                word = 0
                for j in range(sz):
                    val = int(padded[n*sz + j]) & 0xFFFF
                    word |= val << (j * 16)
                packed.append(word)
            return packed
        return []
        return []

    def compile(self):
        """
        Compiles the program into Instruction Memory (IM) and Unified Buffer (UB) images.
        """
        # Pass 1: Resolve Roles, Precisions, and allocate addresses
        roles = {}
        self.symbol_precisions = {}
        mkn_info = {}
        
        for instr in self.instructions:
            if instr['type'] == 'MATMUL':
                # Assign roles and record precisions
                if instr['a_name'] not in roles:
                    roles[instr['a_name']] = 'A'
                    self.symbol_precisions[instr['a_name']] = instr['in_precision']
                if instr['b_name'] not in roles:
                    roles[instr['b_name']] = 'B'
                    self.symbol_precisions[instr['b_name']] = instr['in_precision']
                if instr['c_name'] not in roles:
                    roles[instr['c_name']] = 'C'
                    self.symbol_precisions[instr['c_name']] = instr['out_precision']
                
                if instr['bias_name']:
                    roles[instr['bias_name']] = 'BIAS'
                    self.symbol_precisions[instr['bias_name']] = PrecisionMode.INT16 # Bias is always INT16
                    mkn_info[instr['bias_name']] = (1, 1, instr['n'])

                mkn_info[instr['c_name']] = (instr['m'], instr['k'], instr['n'])
                mkn_info[instr['a_name']] = (instr['m'], instr['k'], instr['n'])
                mkn_info[instr['b_name']] = (instr['m'], instr['k'], instr['n'])

            elif instr['type'] == 'MOVE':
                src = instr['src_name']
                dest = instr['dest_name']
                if src not in roles:
                    roles[src] = 'A'
                    self.symbol_precisions[src] = PrecisionMode.INT16
                roles[dest] = roles[src]
                self.symbol_precisions[dest] = self.symbol_precisions[src]
                
                shape = self.raw_symbols[src]['shape']
                p = 1 << (2 - self.symbol_precisions[src])
                role = roles[src]
                
                if role == 'A':
                    mt = (shape[0] + self.array_size - 1) // self.array_size
                    kt_words = (shape[1]//p + self.array_size - 1) // self.array_size
                    mkn_info[src] = (mt, kt_words, 1)
                elif role == 'B':
                    kt_words = (shape[0]//p + self.array_size - 1) // self.array_size
                    nt = (shape[1] + self.array_size - 1) // self.array_size
                    mkn_info[src] = (1, kt_words, nt)
                else: # C
                    mt = (shape[0] + self.array_size - 1) // self.array_size
                    nt = (shape[1] + self.array_size - 1) // self.array_size
                    mkn_info[src] = (mt, 1, nt)
                mkn_info[dest] = mkn_info[src]

        self.ub_data = []
        self.symbol_to_addr = {}
        self.symbol_roles = roles
        next_addr = 0

        # Pass 1.5: Allocate addresses for all symbols
        # Sort symbols for deterministic layout
        for name in sorted(self.raw_symbols.keys()):
            info = self.raw_symbols[name]
            role = roles.get(name, 'C') 
            precision = self.symbol_precisions.get(name, PrecisionMode.INT16)
            sz = self.array_size
            
            # Default m, k, n if not in mkn_info (should only happen for unused symbols)
            m, k, n = mkn_info.get(name, (1, 1, 1))
            if name not in mkn_info:
                # Infer m, k, n from shape for unused/MOVE symbols
                shape = info['shape']
                p = 1 << (2 - precision)
                if role == 'A':
                    m, k, n = (shape[0]+sz-1)//sz, (shape[1]//p+sz-1)//sz, 1
                elif role == 'B':
                    m, k, n = 1, (shape[0]//p+sz-1)//sz, (shape[1]+sz-1)//sz
                elif role == 'BIAS':
                    m, k, n = 1, 1, (shape[1]+sz-1)//sz
                else: # C
                    m, k, n = (shape[0]+sz-1)//sz, 1, (shape[1]+sz-1)//sz

            # Get physical word count by performing a dummy packing
            dummy_packed = self._pack_tiled(info['data'], role, precision, m, k, n)
            
            self.symbol_to_addr[name] = next_addr
            self.ub_data.extend(dummy_packed)
            next_addr += len(dummy_packed)

        # Pass 2: Generate Instructions
        compiled_im = []
        for instr in self.instructions:
            if instr['type'] == 'MATMUL':
                bias_addr = 0xFFFF
                if instr['bias_name']:
                    bias_addr = self.symbol_to_addr[instr['bias_name']]

                raw = pack_matmul(Opcode.MATMUL,
                                  self.symbol_to_addr[instr['a_name']],
                                  self.symbol_to_addr[instr['b_name']],
                                  self.symbol_to_addr[instr['c_name']],
                                  instr['m'], instr['k'], instr['n'],
                                  bias_addr,
                                  instr['shift'],
                                  instr['multiplier'],
                                  instr['activation'],
                                  instr['in_precision'],
                                  instr['out_precision'],
                                  instr['write_offset'])
                compiled_im.append(raw)
            elif instr['type'] == 'MOVE':
                role = roles[instr['src_name']]
                precision = self.symbol_precisions[instr['src_name']]
                m, k, n = mkn_info[instr['src_name']]
                
                # Physical words count
                dummy_packed = self._pack_tiled(None, role, precision, m, k, n)
                count = len(dummy_packed)
                
                raw = pack_move(Opcode.MOVE,
                                self.symbol_to_addr[instr['src_name']],
                                self.symbol_to_addr[instr['dest_name']],
                                count)
                compiled_im.append(raw)
            elif instr['type'] == 'HALT':
                compiled_im.append(pack_simple(Opcode.HALT))

        self.last_compiled = {"im": compiled_im, "ub": self.ub_data}
        return self.last_compiled

    def generate_driver_source(self, filename="generated_driver.py"):
        """
        Exports a complete Python driver script for the compiled program.
        """
        if not hasattr(self, 'last_compiled'):
            self.compile()

        binary = self.last_compiled
        sz_bytes = self.buffer_width // 8

        with open(filename, "w") as f:
            f.write("# Auto-generated TinyNPU Driver\n")
            f.write("import numpy as np\n\n")

            f.write(f"ARRAY_SIZE = {self.array_size}\n")
            f.write(f"BUFFER_WIDTH_BYTES = {sz_bytes}\n")
            f.write(f"IM_BASE = {self.im_base_addr:#x}\n\n")

            f.write("SYMBOLS = {\n")
            for name, addr in self.symbol_to_addr.items():
                f.write(f"    '{name}': {addr},\n")
            f.write("}\n\n")

            f.write("INSTRUCTIONS = [\n")
            for inst in binary['im']:
                f.write(f"    {inst:#066x},\n")
            f.write("]\n\n")

            f.write("UB_DATA = [\n")
            for word in binary['ub']:
                # Dynamic width hex formatting
                hex_str = f"{word:0{self.buffer_width//4}x}"
                f.write(f"    0x{hex_str},\n")
            f.write("]\n\n")

            f.write("DRIVER_MESSAGES = [\n")
            # 1. Load UB (ADDR -> CMD -> MMVR)
            for addr, data in enumerate(binary['ub']):
                f.write(f"    (0x08, {addr}), # ADDR\n")
                f.write(f"    (0x04, 0x01), # CMD_WRITE_MEM\n")
                f.write(f"    (0x10, {data}), # MMVR (Base)\n")

            # 2. Load IM
            inst_width = 256
            for addr, inst in enumerate(binary['im']):
                if self.buffer_width >= inst_width:
                    # Write whole instruction in one go
                    f.write(
                        f"    (0x08, {self.im_base_addr + addr}), # ADDR\n")
                    f.write(f"    (0x04, 0x01), # CMD_WRITE_MEM\n")
                    f.write(f"    (0x10, {inst}), # MMVR (Base)\n")
                else:
                    # Power-of-2 chunking
                    num_chunks = inst_width // self.buffer_width
                    for i in range(num_chunks):
                        chunk_mask = (1 << self.buffer_width) - 1
                        chunk = (inst >> (i * self.buffer_width)) & chunk_mask
                        im_addr = (addr * num_chunks) + i
                        f.write(
                            f"    (0x08, {self.im_base_addr + im_addr}), # ADDR\n")
                        f.write(f"    (0x04, 0x01), # CMD_WRITE_MEM\n")
                        f.write(f"    (0x10, {chunk}), # MMVR (Base)\n")

            # 3. Run
            doorbell_addr = 0x10 + (self.buffer_width // 8) - 1
            f.write(f"    (0x0C, {self.im_base_addr:#x}), # ARG (PC Start)\n")
            f.write("    (0x04, 0x03), # CMD_RUN\n")
            f.write(
                f"    ({doorbell_addr:#x}, 0), # MMVR (Doorbell trigger)\n")
            f.write("]\n")

    def save_npu(self, filename):
        """
        Exports the program to a unified .npu (JSON) format for testing.
        """
        if not hasattr(self, 'last_compiled'):
            self.compile()

        binary = self.last_compiled

        # We need to export UB and IM as lists of hex strings for JSON portability
        # and include symbol table for verification
        npu_data = {
            "config": {
                "array_size": self.array_size,
                "buffer_width": self.buffer_width,
                "im_base": self.im_base_addr
            },
            "symbols": {
                name: {
                    "addr": addr,
                    "shape": list(self.raw_symbols[name]['shape']),
                    "role": self.symbol_roles.get(name, 'A'),
                    "precision": int(self.symbol_precisions.get(name, PrecisionMode.INT16))
                } for name, addr in self.symbol_to_addr.items()
            },
            "im": [f"0x{inst:064x}" for inst in binary['im']],
            "ub": [f"0x{word:0{self.buffer_width//4}x}" for word in binary['ub']],
            "expected": {
                name: self.expected_results[name].tolist()
                for name in self.expected_results if name in self.raw_symbols
            }
        }

        with open(filename, "w") as f:
            json.dump(npu_data, f, indent=2)
        print(f"Saved NPU program to {filename}")

    def to_assembly(self): return "; Tiled Assembly"
