import numpy as np
import torch
import torch.nn as nn

from software.compiler.tinynpu_jit import compile_module, mark_for_verify


class SimpleChainModule(nn.Module):
    def __init__(self, w1, w2):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(w1.astype(np.float32)), requires_grad=False)
        self.w2 = nn.Parameter(torch.tensor(w2.astype(np.float32)), requires_grad=False)

    def forward(self, x):
        a1 = torch.matmul(self.w1, x)
        a1 = mark_for_verify(a1, "A1")
        a2 = torch.matmul(self.w2, a1)
        return a2


def build_simple_chain_artifact(seed: int = 7, dim: int = 16):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 10, size=(dim, dim), dtype=np.int16)
    w1 = rng.integers(0, 5, size=(dim, dim), dtype=np.int16)
    w2 = rng.integers(0, 5, size=(dim, dim), dtype=np.int16)

    module = SimpleChainModule(w1, w2)
    artifact = compile_module(module, (torch.tensor(a.astype(np.float32)),))
    verify_names = {spec.verify_label: name for name, spec in artifact.plan.tensors.items() if spec.verify_label}
    expected = {
        "A1": np.array(artifact.expected_tensors[verify_names["A1"]], copy=True),
        artifact.plan.outputs[0]: np.array(artifact.expected_tensors[artifact.plan.outputs[0]], copy=True),
    }

    return {
        "artifact": artifact,
        "inputs": {"x": a},
        "expected": expected,
    }
