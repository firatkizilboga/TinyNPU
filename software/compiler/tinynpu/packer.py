import numpy as np
from .isa import PrecisionMode

class Packer:
    """
    Handles hardware-specific data layout and bit-packing.
    Decouples logical matrices from physical Unified Buffer words.
    """
    def __init__(self, array_size):
        self.sz = array_size

    def pack(self, data, role, precision, m_tiles=1, k_tiles=1, n_tiles=1):
        p = 1 << (2 - precision)
        if data is None:
            count = self.get_physical_word_count(role, precision, m_tiles, k_tiles, n_tiles)
            return [0] * count

        if role == 'A':
            return self._pack_role_a(data, precision, m_tiles, k_tiles)
        elif role == 'B':
            return self._pack_role_b(data, precision, k_tiles, n_tiles)
        elif role == 'BIAS':
            return self._pack_bias(data, n_tiles)
        else: # Role C
            return self._pack_role_c(data, precision, m_tiles, n_tiles)

    def get_physical_word_count(self, role, precision, m_tiles, k_tiles, n_tiles):
        p = 1 << (2 - precision)
        if role == 'A':
            return m_tiles * k_tiles * self.sz
        elif role == 'B':
            return k_tiles * n_tiles * self.sz
        elif role == 'BIAS':
            return n_tiles * 2
        else: # Role C
            mt_phys = (m_tiles + p - 1) // p
            return mt_phys * n_tiles * self.sz

    def _pack_role_a(self, data, precision, m_tiles, k_tiles):
        """Matrix A (Left): Each word = 8 rows, each lane = P packed K-elements."""
        p = 1 << (2 - precision)
        bits, mask = 16 // p, (1 << (16 // p)) - 1
        kt_words = k_tiles * self.sz
        padded = np.zeros((m_tiles * self.sz, kt_words * p), dtype=np.int32)
        dm, dk = data.shape
        padded[:min(dm, m_tiles*self.sz), :min(dk, kt_words*p)] = data[:min(dm, m_tiles*self.sz), :min(dk, kt_words*p)]
        
        packed = []
        for m in range(m_tiles):
            for k in range(k_tiles):
                for col_idx in range(self.sz):
                    word = 0
                    for i in range(self.sz):
                        # Row i of the physical tile
                        start_k = (k * self.sz + col_idx) * p
                        subword = 0
                        for bit_idx in range(p):
                            val = int(padded[m*self.sz + i, start_k + bit_idx]) & mask
                            subword |= val << (bit_idx * bits)
                        word |= (subword & 0xFFFF) << (i * 16)
                    packed.append(word)
        return packed

    def _pack_role_b(self, data, precision, k_tiles, n_tiles):
        """Matrix B (Top): Each word = 8 columns, each lane = P packed K-elements."""
        p = 1 << (2 - precision)
        bits, mask = 16 // p, (1 << (16 // p)) - 1
        kt_words = k_tiles * self.sz
        padded = np.zeros((kt_words * p, n_tiles * self.sz), dtype=np.int32)
        dk, dn = data.shape
        padded[:min(dk, kt_words*p), :min(dn, n_tiles*self.sz)] = data[:min(dk, kt_words*p), :min(dn, n_tiles*self.sz)]

        packed = []
        for k in range(k_tiles):
            for n in range(n_tiles):
                for row_idx in range(self.sz):
                    word = 0
                    for i in range(self.sz):
                        # Column i of the physical tile
                        start_k = (k * self.sz + row_idx) * p
                        subword = 0
                        for bit_idx in range(p):
                            val = int(padded[start_k + bit_idx, n*self.sz + i]) & mask
                            subword |= val << (bit_idx * bits)
                        word |= (subword & 0xFFFF) << (i * 16)
                    packed.append(word)
        return packed

    def _pack_role_c(self, data, precision, m_tiles, n_tiles):
        """Matrix C: Each word = 8 columns, each lane = P packed M-elements (Strided)."""
        p = 1 << (2 - precision)
        bits, mask = 16 // p, (1 << (16 // p)) - 1
        mt_phys = (m_tiles + p - 1) // p
        padded_height = mt_phys * self.sz * p
        padded = np.zeros((padded_height, n_tiles * self.sz), dtype=np.int32)
        if data is not None:
            dm, dn = data.shape
            padded[:min(dm, padded_height), :min(dn, n_tiles*self.sz)] = data[:min(dm, padded_height), :min(dn, n_tiles*self.sz)]
        
        packed = []
        for mt in range(mt_phys):
            for nt in range(n_tiles):
                for i in range(self.sz):
                    word = 0
                    for j in range(self.sz):
                        # Lane j contains P elements from different M-tiles, separated by sz
                        start_m = mt * (self.sz * p) + i
                        col_idx = nt * self.sz + j
                        subword = 0
                        for bit_idx in range(p):
                            val = int(padded[start_m + bit_idx * self.sz, col_idx]) & mask
                            subword |= val << (bit_idx * bits)
                        word |= (subword & 0xFFFF) << (j * 16)
                    packed.append(word)
        return packed

    def _pack_bias(self, data, n_tiles):
        flat_data = data.flatten()
        padded_N = n_tiles * self.sz
        padded = np.zeros(padded_N, dtype=np.int64)
        padded[:min(flat_data.shape[0], padded_N)] = flat_data[:min(flat_data.shape[0], padded_N)]
        packed = []
        for n in range(n_tiles):
            word0, word1 = 0, 0
            for j in range(4):
                word0 |= (int(padded[n*self.sz + j]) & 0xFFFFFFFF) << (j * 32)
                word1 |= (int(padded[n*self.sz + 4 + j]) & 0xFFFFFFFF) << (j * 32)
            packed.extend([word0, word1])
        return packed
