from __future__ import annotations

import numpy as np

from software.compiler.tinynpu_jit import VerificationMode, compile_module, run_host_emulation
from software.compiler.tinynpu_quant import LayerQuantConfig, QConv2d, QLinear, convert_qat_model_for_compiler


def _build_qat_model():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TinyQATCompilerReadyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = QConv2d(1, 2, 3, padding=1, config=LayerQuantConfig(w_bits=4, a_bits=4))
            self.fc = QLinear(2, 3, config=LayerQuantConfig(w_bits=16, a_bits=16))

            with torch.no_grad():
                self.conv.conv.weight.copy_(
                    torch.tensor(
                        [
                            [[[0.50, -0.25, 0.125], [0.25, 0.75, -0.5], [0.0, 0.125, -0.25]]],
                            [[[-0.125, 0.375, 0.5], [0.25, -0.625, 0.25], [0.125, 0.0, -0.125]]],
                        ],
                        dtype=torch.float32,
                    )
                )
                self.conv.conv.bias.copy_(torch.tensor([0.125, -0.25], dtype=torch.float32))
                self.fc.linear.weight.copy_(
                    torch.tensor(
                        [
                            [0.75, -0.50],
                            [-0.25, 0.50],
                            [0.125, -0.375],
                        ],
                        dtype=torch.float32,
                    )
                )
                self.fc.linear.bias.copy_(torch.tensor([0.125, -0.25, 0.375], dtype=torch.float32))
                self.conv.w_scale.copy_(torch.tensor(0.125))
                self.conv.a_scale.copy_(torch.tensor(0.125))
                self.fc.w_scale.copy_(torch.tensor(0.125))
                self.fc.a_scale.copy_(torch.tensor(0.125))

        def forward(self, x):
            x = F.relu(self.conv(x))
            x = x.mean(dim=[2, 3])
            x = self.fc(x)
            return x

    return TinyQATCompilerReadyModel().eval()


def build_qat_compiler_ready_artifact():
    import torch

    qat_model = _build_qat_model()
    compiler_ready = convert_qat_model_for_compiler(qat_model, layer_order=["conv", "fc"])
    example = torch.tensor(
        [[[[0.25, -0.5, 0.75], [0.125, -0.25, 0.5], [0.0, 0.375, -0.125]]]],
        dtype=torch.float32,
    )
    artifact = compile_module(compiler_ready, (example,))
    return artifact, example.numpy()


def smoke_run_qat_compiler_ready():
    artifact, example = build_qat_compiler_ready_artifact()
    result = run_host_emulation(
        artifact,
        {artifact.plan.inputs[0]: example},
        VerificationMode.DEBUG,
        debug=True,
    )
    return {
        "output": result.tensors[artifact.plan.outputs[0]],
        "debug_kinds": [event["kind"] for event in result.debug_trace],
    }


if __name__ == "__main__":
    results = smoke_run_qat_compiler_ready()
    print("output", results["output"].reshape(-1).tolist())
    print("debug_kinds", results["debug_kinds"])
