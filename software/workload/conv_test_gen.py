import numpy as np
import sys
import os

# Add compiler to path
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "../compiler"))
from tinynpu import TinyNPUProgram, PrecisionMode

def generate_conv_test():
    prog = TinyNPUProgram()
    sz = prog.array_size

    print(f"Generating 2D Convolution Test using im2col for {sz}x{sz} array...")

    # Image: 14x14x1 (e.g. MNIST-like small)
    # Kernel: 3x3x1, 8 filters
    H, W, C = 14, 14, 1
    KH, KW, OC = 3, 3, 8
    
    img_data = np.random.randint(0, 5, size=(H, W, C)).astype(np.int16)
    ker_data = np.random.randint(-2, 2, size=(KH, KW, C, OC)).astype(np.int16)
    bias_data = np.random.randint(0, 10, size=(1, OC)).astype(np.int16)

    prog.declare_image("InputImage", img_data, H, W, C, precision=PrecisionMode.INT16)
    prog.declare_kernel("ConvKernel", ker_data, KH, KW, C, OC, precision=PrecisionMode.INT16)
    prog.declare_data("Bias", bias_data, precision=PrecisionMode.INT16, role='BIAS')

    # Apply Conv2D
    prog.conv2d_im2col("InputImage", "ConvKernel", "Output", bias_name="Bias", stride=1, padding=0)
    prog.halt()

    # Golden model (Host-side Conv2D)
    OH, OW = (H - KH) + 1, (W - KW) + 1
    expected_output = np.zeros((OH, OW, OC), dtype=np.int64)
    
    for oc in range(OC):
        for y in range(OH):
            for x in range(OW):
                patch = img_data[y:y+KH, x:x+KW, :]
                kernel = ker_data[:, :, :, oc]
                val = np.sum(patch.astype(np.int64) * kernel.astype(np.int64)) + bias_data[0, oc]
                expected_output[y, x, oc] = val

    # Matmul output layout is (OH*OW, OC): each row is one spatial position,
    # each column is one output channel.
    expected_matrix = expected_output.reshape(-1, OC).astype(np.int16)
    prog.add_expected_result("Output", expected_matrix)

    output_path = "conv_test.npu"
    prog.save_npu(output_path)
    print(f"Convolution test generated: {output_path}")

if __name__ == "__main__":
    generate_conv_test()
