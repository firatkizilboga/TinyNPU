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

    scores = np.matmul(w1.astype(np.int64), x.astype(np.int64))
    scores = np.clip(scores, -32768, 32767).astype(np.int32)
    shifted = scores.astype(np.float32) - np.max(scores.astype(np.float32), axis=0, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=0, keepdims=True)
    q_probs = np.clip(np.rint(probs * np.float32(reentry_scale)), -32768, 32767).astype(np.int16)
    out = np.matmul(w2.astype(np.int64), q_probs.astype(np.int64))
    out = np.clip(out, -32768, 32767).astype(np.int32)

    return {
        "artifact": artifact,
        "inputs": {"x": x},
        "expected": {
            "scores": scores,
            "q_probs": q_probs,
            artifact.plan.outputs[0]: out,
        },
    }
