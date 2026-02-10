import numpy as np
from .isa import Opcode, HostCmd, generate_host_messages, pack_matmul, pack_move, pack_simple

class TinyNPUProgram:
    def __init__(self):
        self.instructions = []
        self.raw_symbols = {} 
        self.symbol_to_addr = {}
        self.ub_data = []
        self.im_base_addr = 0x8000
        
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
            
        # Hardware uses 4x4 tiles
        m_tiles = (shape_a[0] + 3) // 4
        k_tiles = (shape_a[1] + 3) // 4
        n_tiles = (shape_b[1] + 3) // 4
        
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
        if role == 'A':
            # Matrix A (Left): Column-Major Tiles
            M, K = data.shape
            mt, kt = (M+3)//4, (K+3)//4
            padded = np.zeros((mt*4, kt*4), dtype=np.uint16)
            padded[:M, :K] = data
            packed = []
            for m in range(mt):
                for k in range(kt):
                    tile = padded[m*4:m*4+4, k*4:k*4+4]
                    for col_idx in range(4):
                        col = tile[:, col_idx]
                        # Pack 4x 16-bit values into one 64-bit word
                        packed.append(int(col[0]) | (int(col[1]) << 16) | (int(col[2]) << 32) | (int(col[3]) << 48))
            return packed
        elif role == 'B':
            # Matrix B (Top): Row-Major Tiles
            K, N = data.shape
            kt, nt = (K+3)//4, (N+3)//4
            padded = np.zeros((kt*4, nt*4), dtype=np.uint16)
            padded[:K, :N] = data
            packed = []
            for k in range(kt):
                for n in range(nt):
                    tile = padded[k*4:k*4+4, n*4:n*4+4]
                    for row_idx in range(4):
                        row = tile[row_idx, :]
                        packed.append(int(row[0]) | (int(row[1]) << 16) | (int(row[2]) << 32) | (int(row[3]) << 48))
            return packed
        elif role == 'C':
            # Output space allocation (Row-Major Tiles)
            return [0] * (m_tiles * n_tiles * 4)
        return []

    def compile(self):
        """
        Compiles the program into Instruction Memory (IM) and Unified Buffer (UB) images.
        """
        # Pass 1: Resolve Roles and allocate addresses
        # We need to decide which symbol is A, B, or C to pack it correctly.
        # If a symbol is used in multiple roles (e.g. C of layer 1 is B of layer 2),
        # we prioritize the packing that matches its 'source' (usually C).
        roles = {}
        mkn_info = {}
        for instr in self.instructions:
            if instr['type'] == 'MATMUL':
                # Only set role if not already set (prevents overwriting B with A etc.)
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
            
            # If C-type output has no initial data, allocate zeros
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
        
        with open(filename, "w") as f:
            f.write("# Auto-generated TinyNPU Driver\n")
            f.write("import numpy as np\n\n")
            
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
                f.write(f"    {word:#018x},\n")
            f.write("]\n\n")

            f.write("DRIVER_MESSAGES = [\n")
            # 1. Load UB (ADDR -> CMD -> DOORBELL)
            for addr, data in enumerate(binary['ub']):
                f.write(f"    (0x08, {addr}), # ADDR\n")
                f.write(f"    (0x04, 0x01), # CMD_WRITE_MEM\n")
                f.write(f"    (0x10, {data}), # MMVR (Doorbell)\n")
            
            # 2. Load IM
            for addr, inst in enumerate(binary['im']):
                for i in range(4):
                    chunk = (inst >> (i * 64)) & 0xFFFFFFFFFFFFFFFF
                    im_addr = 0x8000 + (addr * 4) + i
                    f.write(f"    (0x08, {im_addr}), # ADDR\n")
                    f.write(f"    (0x04, 0x01), # CMD_WRITE_MEM\n")
                    f.write(f"    (0x10, {chunk}), # MMVR (Doorbell)\n")

            # 3. Run
            f.write("    (0x0C, 0x8000), # ARG (PC Start)\n")
            f.write("    (0x04, 0x03), # CMD_RUN\n")
            f.write("    (0x10, 0), # MMVR (Doorbell)\n")
            f.write("]\n")

    def to_assembly(self): return "; Tiled Assembly"
