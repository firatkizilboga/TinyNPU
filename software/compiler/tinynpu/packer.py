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
        """
        Main entry point for packing.
        Returns a list of 128-bit integers (UB words).
        """
        p = 1 << (2 - precision)
        bits = 16 // p
        mask = (1 << bits) - 1

        if data is None:
            # Return empty buffer of zeros matching the expected physical size
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
        p = 1 << (2 - precision)
        bits = 16 // p
        mask = (1 << bits) - 1
        kt_words = k_tiles * self.sz
        padded = np.zeros((m_tiles * self.sz, kt_words * p), dtype=np.int32)
        d_m, d_k = data.shape
        padded[:min(d_m, m_tiles*self.sz), :min(d_k, kt_words*p)] = data[:min(d_m, m_tiles*self.sz), :min(d_k, kt_words*p)]
        
        packed = []
        for m in range(m_tiles):
            for k in range(k_tiles):
                tile = padded[m*self.sz:m*self.sz+self.sz, (k*self.sz*p):(k*self.sz+self.sz)*p]
                for col_idx in range(self.sz):
                    word = 0
                    for i in range(self.sz):
                        start_k = col_idx * p
                        subword = 0
                        for bit_idx in range(p):
                            val = int(tile[i, start_k + bit_idx]) & mask
                            subword |= val << (bit_idx * bits)
                        word |= (subword & 0xFFFF) << (i * 16)
                    packed.append(word)
        return packed

    def _pack_role_b(self, data, precision, k_tiles, n_tiles):
        p = 1 << (2 - precision)
        bits = 16 // p
        mask = (1 << bits) - 1
        kt_words = k_tiles * self.sz
        padded = np.zeros((kt_words * p, n_tiles * self.sz), dtype=np.int32)
        d_k, d_n = data.shape
        padded[:min(d_k, kt_words*p), :min(d_n, n_tiles*self.sz)] = data[:min(d_k, kt_words*p), :min(d_n, n_tiles*self.sz)]

        packed = []
        for k in range(k_tiles):
            for n in range(n_tiles):
                tile = padded[(k*self.sz*p):(k*self.sz+self.sz)*p, n*self.sz:n*self.sz+self.sz]
                for row_idx in range(self.sz):
                    word = 0
                    for i in range(self.sz):
                        start_k = row_idx * p
                        subword = 0
                        for bit_idx in range(p):
                            val = int(tile[start_k + bit_idx, i]) & mask
                            subword |= val << (bit_idx * bits)
                        word |= (subword & 0xFFFF) << (i * 16)
                    packed.append(word)
        return packed

    def _pack_role_c(self, data, precision, m_tiles, n_tiles):
        p = 1 << (2 - precision)
        bits = 16 // p
        mask = (1 << bits) - 1
        mt_phys = (m_tiles + p - 1) // p
        padded_height = mt_phys * self.sz * p
        padded = np.zeros((padded_height, n_tiles * self.sz), dtype=np.int32)
        if data is not None:
            d_m, d_n = data.shape
            padded[:min(d_m, padded_height), :min(d_n, n_tiles*self.sz)] = data[:min(d_m, padded_height), :min(d_n, n_tiles*self.sz)]
        packed = []
        for mt in range(mt_phys):
            for nt in range(n_tiles):
                for i in range(self.sz):
                    word = 0
                    for j in range(self.sz):
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
            word0 = 0
            for j in range(4):
                val = int(padded[n*self.sz + j]) & 0xFFFFFFFF
                word0 |= val << (j * 32)
            packed.append(word0)
            word1 = 0
            for j in range(4):
                val = int(padded[n*self.sz + 4 + j]) & 0xFFFFFFFF
                word1 |= val << (j * 32)
            packed.append(word1)
        return packed
