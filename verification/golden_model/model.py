import numpy as np

def ppu_quantize(acc, bias, multiplier, shift, activation=False, precision='INT16', write_offset=0):
    """
    Bit-accurate emulation of the NPU's Post-Processing Unit (PPU).
    
    Args:
        acc: 64-bit integer accumulator value
        bias: 32-bit integer bias value
        multiplier: 16-bit unsigned multiplier (M0)
        shift: 8-bit shift amount
        activation: Boolean, if True applies ReLU
        precision: 'INT16', 'INT8', or 'INT4'
        write_offset: 0-3 (for packing sub-word elements)
        
    Returns:
        16-bit integer representing the slot in the Unified Buffer word.
    """
    # 1. Bias Addition (sign-extended)
    biased_acc = np.int64(acc) + np.int64(bias)
    
    # 2. Rescale (Multiplier)
    # multiplier is treated as unsigned 16-bit in RTL, so we ensure it's positive
    rescaled = biased_acc * np.int64(multiplier & 0xFFFF)
    
    # 3. Rounding and Shift
    if shift > 0:
        rounding_offset = np.int64(1) << (shift - 1)
        rounded = rescaled + rounding_offset
        shifted = rounded >> shift
    else:
        shifted = rescaled
        
    # 4. Activation (ReLU)
    if activation:
        shifted = max(0, shifted)
        
    # 5. Precision Saturation & Packing
    if precision == 'INT4':
        # Range [-8, 7]
        sat = np.clip(shifted, -8, 7).astype(np.int8)
        result = (np.uint16(sat & 0xF)) << (write_offset * 4)
    elif precision == 'INT8':
        # Range [-128, 127]
        sat = np.clip(shifted, -128, 127).astype(np.int8)
        result = (np.uint16(sat & 0xFF)) << (write_offset * 8)
    else: # INT16
        # Range [-32768, 32767]
        sat = np.clip(shifted, -32768, 32767).astype(np.int16)
        result = np.uint16(sat)
        
    return result

class GoldenModel:
    """
    High-level functional model of the TinyNPU.
    Replicates the systolic array and PPU logic with bit-accuracy.
    """
    def __init__(self, array_size=8):
        self.sz = array_size

    def matmul(self, A, B, bias=None, multiplier=1, shift=0, activation=False, 
               in_precision='INT16', out_precision='INT16'):
        """
        Performs matrix multiplication A * B + bias with NPU-accurate quantization.
        
        A: (M, K) matrix
        B: (K, N) matrix
        bias: (N,) vector
        """
        # Ensure NumPy arrays
        A = np.array(A, dtype=np.int64)
        B = np.array(B, dtype=np.int64)
        
        if bias is None:
            bias = np.zeros(B.shape[1], dtype=np.int64)
        else:
            bias = np.array(bias, dtype=np.int64).flatten()

        # Core GEMM
        # Note: Hardware uses 64-bit accumulators, NumPy int64 matches this.
        acc_matrix = np.matmul(A, B)
        
        # Apply PPU logic to each element
        M, N = acc_matrix.shape
        output = np.zeros((M, N), dtype=np.int32)
        
        for m in range(M):
            for n in range(N):
                # Logical PPU emulation (ignoring packing for logical output)
                # We reuse the logic but return the clipped value directly
                val = self._logical_ppu(acc_matrix[m, n], bias[m], multiplier, shift, activation, out_precision)
                output[m, n] = val
                
        return output

    def _logical_ppu(self, acc, bias, multiplier, shift, activation, precision):
        # 1. Bias Addition
        # Hardware: 64-bit acc + 32-bit signed bias -> 65-bit biased_acc
        biased_acc = np.int64(acc) + np.int64(np.int32(bias))
        # 2. Rescale
        rescaled = biased_acc * np.int64(multiplier)
        # 3. Rounding & Shift
        if shift > 0:
            rounding_offset = np.int64(1) << (shift - 1)
            shifted = (rescaled + rounding_offset) >> shift
        else:
            shifted = rescaled
        # 4. Activation
        if activation:
            shifted = max(0, shifted)
        # 5. Saturation
        if precision == 'INT4':
            return np.clip(shifted, -8, 7)
        elif precision == 'INT8':
            return np.clip(shifted, -128, 127)
        else:
            return np.clip(shifted, -32768, 32767)

def get_golden_result(a, b, bias, m0, shift, relu, out_prec):
    """Convenience wrapper for test scripts."""
    gm = GoldenModel()
    return gm.matmul(a, b, bias, m0, shift, relu, out_precision=out_prec)
