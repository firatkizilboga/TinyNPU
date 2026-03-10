import torch
import torch.nn as nn

from tinynpu_jit import VerificationMode, compile_module, mark_for_verify


class Demo(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))

    def forward(self, x):
        scores = torch.matmul(self.weight, x)
        scores = mark_for_verify(scores, "scores")
        return torch.softmax(scores, dim=0)


if __name__ == "__main__":
    model = Demo()
    example = torch.tensor([[1.0, 3.0], [2.0, 4.0]], dtype=torch.float32)
    artifact = compile_module(model, (example,))
    result = artifact.run_host_emulation({"x": example.numpy()}, verification=VerificationMode.DEBUG)
    print("Verified:", result.verified)
    print("Output:\n", result.tensors["softmax"])
