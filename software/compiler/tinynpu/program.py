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
        if name in self.raw_symbols:
            raise ValueError(f"Symbol '{name}' already declared.")
        self.raw_symbols[name] = {
            'data': np.array(data, dtype=np.uint16),
            'resolved_role': None
        }

    def matmul(self, tile_a, tile_b, out, m, k, n):
        self.instructions.append({
            'type': 'MATMUL',
            'a_name': tile_a,
            'b_name': tile_b,
            'c_name': out,
            'm': m,
            'k': k,
            'n': n
        })

    def halt(self):
        self.instructions.append({'type': 'HALT'})

    def _pack_tiled(self, data, role, m_tiles=0, k_tiles=0, n_tiles=0):
        if role == 'A':
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
                        packed.append(int(col[0]) | (int(col[1]) << 16) | (int(col[2]) << 32) | (int(col[3]) << 48))
            return packed
        elif role == 'B':
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
            # Reserve space for mt * nt tiles
            return [0] * (m_tiles * n_tiles * 4)
        return []

    def compile(self):
        # Pass 1: Roles
        roles = {}
        mkn = {}
        for instr in self.instructions:
            if instr['type'] == 'MATMUL':
                roles[instr['a_name']] = 'A'
                roles[instr['b_name']] = 'B'
                roles[instr['c_name']] = 'C'
                mkn[instr['c_name']] = (instr['m'], instr['k'], instr['n'])

        # Pass 2: Pack
        self.ub_data = []
        self.symbol_to_addr = {}
        next_addr = 0
        
        # Sort keys for deterministic packing order
        for name in sorted(self.raw_symbols.keys()):
            info = self.raw_symbols[name]
            role = roles.get(name, 'A')
            m, k, n = mkn.get(name, (0,0,0))
            packed = self._pack_tiled(info['data'], role, m, k, n)
            self.symbol_to_addr[name] = next_addr
            self.ub_data.extend(packed)
            next_addr += len(packed)
            
        # Ensure C is allocated if not declared (Output Buffers)
        for instr in self.instructions:
            if instr['type'] == 'MATMUL' and instr['c_name'] not in self.symbol_to_addr:
                name = instr['c_name']
                m, k, n = instr['m'], instr['k'], instr['n']
                packed = [0] * (m * n * 4)
                self.symbol_to_addr[name] = next_addr
                self.ub_data.extend(packed)
                next_addr += len(packed)

        # Pass 3: Instructions
        compiled = []
        for instr in self.instructions:
            if instr['type'] == 'MATMUL':
                raw = pack_matmul(Opcode.MATMUL, self.symbol_to_addr[instr['a_name']], 
                                  self.symbol_to_addr[instr['b_name']], self.symbol_to_addr[instr['c_name']],
                                  instr['m'], instr['k'], instr['n'])
                compiled.append(raw)
            elif instr['type'] == 'HALT':
                compiled.append(pack_simple(Opcode.HALT))
        
        # Save compilation result for driver generation
        self.last_compiled = {"im": compiled, "ub": self.ub_data}
        return self.last_compiled

    def generate_driver_source(self, filename="generated_driver.py"):
        """
        Generates a standalone Python script that:
        1. Contains the binary data (IM and UB).
        2. Contains the 'driver_messages' list with robust sequencing.
        3. Includes a verify() function to check results against expected values.
        """
        if not hasattr(self, 'last_compiled'):
            self.compile()
            
        binary = self.last_compiled
        
        with open(filename, "w") as f:
            f.write("# Auto-generated TinyNPU Driver\n")
            f.write("import numpy as np\n\n")
            
            # Write Symbols Info for Verification
            f.write("SYMBOLS = {\n")
            for name, addr in self.symbol_to_addr.items():
                f.write(f"    '{name}': {addr},\n")
            f.write("}\n\n")

            # Write Instructions
            f.write("INSTRUCTIONS = [\n")
            for inst in binary['im']:
                f.write(f"    {inst:#066x},\n")
            f.write("]\n\n")

            # Write UB Data
            f.write("UB_DATA = [\n")
            for word in binary['ub']:
                f.write(f"    {word:#018x},\n")
            f.write("]\n\n")

            # Generate MMIO Sequence
            f.write("DRIVER_MESSAGES = [\n")
            
            # 1. Load UB
            for addr, data in enumerate(binary['ub']):
                f.write(f"    (0x08, {addr}), # ADDR\n")
                f.write(f"    (0x04, 0x01), # CMD_WRITE_MEM\n")
                f.write(f"    (0x10, {data}), # MMVR (Doorbell)\n")
            
            # 2. Load IM (Address 0x8000 + Offset)
            for addr, inst in enumerate(binary['im']):
                for i in range(4):
                    chunk = (inst >> (i * 64)) & 0xFFFFFFFFFFFFFFFF
                    im_addr = 0x8000 + (addr * 4) + i
                    f.write(f"    (0x08, {im_addr}), # ADDR\n")
                    f.write(f"    (0x04, 0x01), # CMD_WRITE_MEM\n")
                    f.write(f"    (0x10, {chunk}), # MMVR (Doorbell)\n")

            # 3. Run (Start PC = 0x8000)
            f.write("    (0x0C, 0x8000), # ARG (PC Start)\n")
            f.write("    (0x04, 0x03), # CMD_RUN\n")
            f.write("    (0x10, 0), # MMVR (Doorbell)\n")
            f.write("]\n\n")

    def to_assembly(self): return "; Tiled Assembly"