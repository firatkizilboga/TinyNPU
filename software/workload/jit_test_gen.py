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

    a1_expected = np.matmul(w1.astype(np.int64), a.astype(np.int64))
    a1_expected = np.clip(a1_expected, -32768, 32767).astype(np.int32)
    a2_expected = np.matmul(w2.astype(np.int64), a1_expected.astype(np.int64))
    a2_expected = np.clip(a2_expected, -32768, 32767).astype(np.int32)

    return {
        "artifact": artifact,
        "inputs": {"x": a},
        "expected": {"A1": a1_expected, artifact.plan.outputs[0]: a2_expected},
    }
