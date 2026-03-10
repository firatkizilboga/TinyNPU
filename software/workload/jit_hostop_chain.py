import numpy as np
import torch
import torch.nn as nn

from software.compiler.tinynpu_jit import compile_module, mark_for_verify, quantize_for_npu


class HostOpChainModule(nn.Module):
    def __init__(self, w1, w2, reentry_scale: float):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(w1.astype(np.float32)), requires_grad=False)
        self.w2 = nn.Parameter(torch.tensor(w2.astype(np.float32)), requires_grad=False)
        self.reentry_scale = float(reentry_scale)

    def forward(self, x):
        scores = torch.matmul(self.w1, x)
        scores = mark_for_verify(scores, "scores")
        probs = torch.softmax(scores, dim=0)
        q_probs = quantize_for_npu(probs, self.reentry_scale, 0, "int16")
        q_probs = mark_for_verify(q_probs, "q_probs")
        return torch.matmul(self.w2, q_probs)


def build_hostop_chain_artifact(seed: int = 11, dim: int = 16, reentry_scale: float = 256.0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 8, size=(dim, dim), dtype=np.int16)
    w1 = rng.integers(0, 4, size=(dim, dim), dtype=np.int16)
    w2 = rng.integers(0, 4, size=(dim, dim), dtype=np.int16)

    module = HostOpChainModule(w1, w2, reentry_scale)
    artifact = compile_module(module, (torch.tensor(x.astype(np.float32)),))
    verify_names = {spec.verify_label: name for name, spec in artifact.plan.tensors.items() if spec.verify_label}
    expected = {
        "scores": np.array(artifact.expected_tensors[verify_names["scores"]], copy=True),
        "q_probs": np.array(artifact.expected_tensors[verify_names["q_probs"]], copy=True),
        artifact.plan.outputs[0]: np.array(artifact.expected_tensors[artifact.plan.outputs[0]], copy=True),
    }

    return {
        "artifact": artifact,
        "inputs": {"x": x},
        "expected": expected,
    }
