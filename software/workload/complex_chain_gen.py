import numpy as np
import sys
import os

# Add compiler to path
sys.path.append(os.path.join(os.getcwd(), "../compiler"))
from tinynpu import TinyNPUProgram, PrecisionMode

def clip_and_quantize(data, precision):
    if precision == PrecisionMode.INT4:
        return np.clip(data, -8, 7).astype(np.int8) # Store as int8 for convenience
    elif precision == PrecisionMode.INT8:
        return np.clip(data, -128, 127).astype(np.int8)
    elif precision == PrecisionMode.INT16:
        return np.clip(data, -32768, 32767).astype(np.int16)
    return data

def generate_complex_chain():
    prog = TinyNPUProgram()
    sz = prog.array_size # 8

    print(f"Generating 7-Layer Mixed Precision Chain (32x32 matrices)...")

    # Dimensions: 32x32 (4x4 tiles)
    # This ensures > 6 physical tiles usage.
    M, N = 32, 32
    
    # Initial Input: INT8
    Input = np.random.randint(-10, 10, size=(M, N)).astype(np.int16)
    
    # Declare initial input as Role B (Row-Major) because it feeds from the Top
    prog.declare_data("Input", Input, precision=PrecisionMode.INT8, role='B')

    # Chain Definition
    # (Out Name, Out Precision, In Precision for Next)
    # L1: In INT8 -> Out INT8
    # L2: In INT8 -> Out INT16
    # L3: In INT16 -> Out INT4
    # L4: In INT4 -> Out INT8
    # L5: In INT8 -> Out INT4
    # L6: In INT4 -> Out INT16
    # L7: In INT16 -> Out INT8
    
    layers = [
        ("L1", PrecisionMode.INT8),  # In INT8 -> Out INT8 (M-Packed)
        ("L2", PrecisionMode.INT16), # In INT8 (Packed) -> Out INT16
        ("L3", PrecisionMode.INT4),  # In INT16 -> Out INT4 (M-Packed)
        ("L4", PrecisionMode.INT8),  # In INT4 (Packed) -> Out INT8 (M-Packed)
        ("L5", PrecisionMode.INT4),  # In INT8 (Packed) -> Out INT4 (M-Packed)
        ("L6", PrecisionMode.INT16), # In INT4 (Packed) -> Out INT16
        ("L7", PrecisionMode.INT8),  # In INT16 -> Out INT8
    ]

    current_input = "Input"
    current_in_prec = PrecisionMode.INT8
    
    golden_input = Input.copy()

    for idx, (layer_name, out_prec) in enumerate(layers):
        # Layer computation precision cycling: INT8, INT16, INT4...
        multi_precs = [PrecisionMode.INT8, PrecisionMode.INT16, PrecisionMode.INT4, 
                       PrecisionMode.INT8, PrecisionMode.INT4, PrecisionMode.INT16, PrecisionMode.INT8]
        
        comp_prec = multi_precs[idx]
        
        W = np.random.randint(-2, 2, size=(N, N)).astype(np.int16)
        B = np.random.randint(-3, 3, size=(1, N)).astype(np.int16)
        
        w_name = f"W_{idx}"
        b_name = f"B_{idx}"
        
        if comp_prec == PrecisionMode.INT4:
            W = np.clip(W, -8, 7)
        elif comp_prec == PrecisionMode.INT8:
            W = np.clip(W, -128, 127)
            
        # Weights MUST match the computation precision for the PE to unpack correctly
        # They feed from the Left (Role A)
        prog.declare_data(w_name, W, precision=comp_prec, role='A')
        prog.declare_data(b_name, B, precision=PrecisionMode.INT16) 

        shift = 5
        multiplier = 1
        
        print(f"Adding Layer {layer_name}: {comp_prec.name} Mul -> {out_prec.name} Out")
        
        # KEY FIX: Activation Chaining (Output -> Matrix B)
        # Previous Layer Output (Row-Major, M-Packed) -> Layer 2 Input B (Row-Major, K-Packed)
        prog.matmul(w_name, current_input, layer_name, bias_name=b_name,
                    shift=shift, multiplier=multiplier,
                    in_precision=comp_prec, out_precision=out_prec)
        
        # Update Golden Model: W * Input + B
        acc = np.matmul(W.astype(np.int64), golden_input.astype(np.int64)) + B.astype(np.int64).T
        
        # 2. Scale & Shift with Rounding
        rescaled = acc * multiplier
        if shift > 0:
            rounding_offset = 1 << (shift - 1)
            shifted = (rescaled + rounding_offset) >> shift
        else:
            shifted = rescaled
        
        # 3. Saturation (Output Precision)
        golden_output = clip_and_quantize(shifted, out_prec)
        
        # Add expected result for EVERY layer for debugging
        prog.add_expected_result(layer_name, golden_output)
        
        current_input = layer_name
        current_in_prec = out_prec 
        golden_input = golden_output

    prog.halt()
    
    # Final Result
    prog.add_expected_result("L7", golden_input)

    output_path = "complex_chain.npu"
    prog.save_npu(output_path)
    print(f"Complex chain test generated: {output_path}")

if __name__ == "__main__":
    generate_complex_chain()
