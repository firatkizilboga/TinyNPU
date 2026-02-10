import numpy as np
import os
import re
from .isa import Opcode, HostCmd, generate_host_messages, pack_matmul, pack_move, pack_simple

class HardwareConfig:
    """Parses hardware parameters from defines.sv to ensure compiler sync."""
    def __init__(self, defines_path=None):
        if defines_path is None:
            # Try to find defines.sv relative to this file
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
            defines_path = os.path.join(base_dir, "rtl", "defines.sv")
            
        self.params = {
            'ARRAY_SIZE': 4,
            'DATA_WIDTH': 16,
            'BUFFER_WIDTH': 64,
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
                            val = int(val)
                        except ValueError:
                            continue # Skip non-integer defines
                    self.params[name] = val
        
        # Derived
        self.params['BUFFER_WIDTH'] = self.params['DATA_WIDTH'] * self.params['ARRAY_SIZE']

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

    def matmul(self, a_name, b_name, out_name):
        """
        Adds a MATMUL instruction: out = a * b
        Dimensions are automatically inferred from declared shapes.
        """
        if a_name not in self.raw_symbols:
            raise ValueError(f"Symbol '{a_name}' not found.")
        if b_name not in self.raw_symbols:
            raise ValueError(f"Symbol '{b_name}' not found.")
            
        shape_a = self.raw_symbols[a_name]['shape']
        shape_b = self.raw_symbols[b_name]['shape']
        
        if shape_a[1] != shape_b[0]:
            raise ValueError(f"Dimension mismatch: {shape_a} and {shape_b}")
            
        # Hardware uses N x N tiles
        m_tiles = (shape_a[0] + self.array_size - 1) // self.array_size
        k_tiles = (shape_a[1] + self.array_size - 1) // self.array_size
        n_tiles = (shape_b[1] + self.array_size - 1) // self.array_size
        
        # If output symbol doesn't exist, record its expected shape
        if out_name not in self.raw_symbols:
            self.raw_symbols[out_name] = {
                'data': None, # To be filled by HW
                'shape': (shape_a[0], shape_b[1])
            }
            
        self.instructions.append({
            'type': 'MATMUL',
            'a_name': a_name,
            'b_name': b_name,
            'c_name': out_name,
            'm': m_tiles,
            'k': k_tiles,
            'n': n_tiles
        })

    def halt(self):
        self.instructions.append({'type': 'HALT'})

    def _pack_tiled(self, data, role, m_tiles=0, k_tiles=0, n_tiles=0):
        """
        Packs matrix data into the format expected by the TPU hardware.
        """
        sz = self.array_size
        if role == 'A':
            # Matrix A (Left): Column-Major Tiles
            M, K = data.shape
            mt, kt = (M+sz-1)//sz, (K+sz-1)//sz
            padded = np.zeros((mt*sz, kt*sz), dtype=np.uint16)
            padded[:M, :K] = data
            packed = []
            for m in range(mt):
                for k in range(kt):
                    tile = padded[m*sz:m*sz+sz, k*sz:k*sz+sz]
                    for col_idx in range(sz):
                        col = tile[:, col_idx]
                        word = 0
                        for i in range(sz):
                            word |= int(col[i]) << (i * 16)
                        packed.append(word)
            return packed
        elif role == 'B':
            # Matrix B (Top): Row-Major Tiles
            K, N = data.shape
            kt, nt = (K+sz-1)//sz, (N+sz-1)//sz
            padded = np.zeros((kt*sz, nt*sz), dtype=np.uint16)
            padded[:K, :N] = data
            packed = []
            for k in range(kt):
                for n in range(nt):
                    tile = padded[k*sz:k*sz+sz, n*sz:n*sz+sz]
                    for row_idx in range(sz):
                        row = tile[row_idx, :]
                        word = 0
                        for i in range(sz):
                            word |= int(row[i]) << (i * 16)
                        packed.append(word)
            return packed
        elif role == 'C':
            # Output space allocation (Row-Major Tiles)
            return [0] * (m_tiles * n_tiles * sz)
        return []

    def compile(self):
        """
        Compiles the program into Instruction Memory (IM) and Unified Buffer (UB) images.
        """
        # Pass 1: Resolve Roles and allocate addresses
        roles = {}
        mkn_info = {}
        for instr in self.instructions:
            if instr['type'] == 'MATMUL':
                if instr['a_name'] not in roles: roles[instr['a_name']] = 'A'
                if instr['b_name'] not in roles: roles[instr['b_name']] = 'B'
                if instr['c_name'] not in roles: roles[instr['c_name']] = 'C'
                mkn_info[instr['c_name']] = (instr['m'], instr['k'], instr['n'])

        self.ub_data = []
        self.symbol_to_addr = {}
        next_addr = 0
        
        # Sort symbols for deterministic layout
        for name in sorted(self.raw_symbols.keys()):
            info = self.raw_symbols[name]
            role = roles.get(name, 'A')
            m, k, n = mkn_info.get(name, (0,0,0))
            
            if info['data'] is None:
                packed = self._pack_tiled(np.zeros(info['shape']), 'C', m, 0, n)
            else:
                packed = self._pack_tiled(info['data'], role, m, k, n)
                
            self.symbol_to_addr[name] = next_addr
            self.ub_data.extend(packed)
            next_addr += len(packed)

        # Pass 2: Generate Instructions
        compiled_im = []
        for instr in self.instructions:
            if instr['type'] == 'MATMUL':
                raw = pack_matmul(Opcode.MATMUL, 
                                  self.symbol_to_addr[instr['a_name']], 
                                  self.symbol_to_addr[instr['b_name']], 
                                  self.symbol_to_addr[instr['c_name']],
                                  instr['m'], instr['k'], instr['n'])
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
                    f.write(f"    (0x08, {self.im_base_addr + addr}), # ADDR\n")
                    f.write(f"    (0x04, 0x01), # CMD_WRITE_MEM\n")
                    f.write(f"    (0x10, {inst}), # MMVR (Base)\n")
                else:
                    # Power-of-2 chunking
                    num_chunks = inst_width // self.buffer_width
                    for i in range(num_chunks):
                        chunk_mask = (1 << self.buffer_width) - 1
                        chunk = (inst >> (i * self.buffer_width)) & chunk_mask
                        im_addr = (addr * num_chunks) + i
                        f.write(f"    (0x08, {self.im_base_addr + im_addr}), # ADDR\n")
                        f.write(f"    (0x04, 0x01), # CMD_WRITE_MEM\n")
                        f.write(f"    (0x10, {chunk}), # MMVR (Base)\n")

            # 3. Run
            doorbell_addr = 0x10 + (self.buffer_width // 8) - 1
            f.write(f"    (0x0C, {self.im_base_addr:#x}), # ARG (PC Start)\n")
            f.write("    (0x04, 0x03), # CMD_RUN\n")
            f.write(f"    ({doorbell_addr:#x}, 0), # MMVR (Doorbell trigger)\n")
            f.write("]\n")

    def to_assembly(self): return "; Tiled Assembly"
