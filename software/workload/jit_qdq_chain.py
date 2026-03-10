import numpy as np
import torch
import torch.nn as nn
from torch.ao.nn.quantized import DeQuantize, Quantize

from software.compiler.tinynpu_jit import compile_module, mark_for_verify, npu_matmul


class QDQChainModule(nn.Module):
    def __init__(self, w1, w2, *, input_scale: float, score_scale: float, prob_scale: float):
        super().__init__()
        self.w1 = nn.Parameter(torch.tensor(w1.astype(np.float32)), requires_grad=False)
        self.w2 = nn.Parameter(torch.tensor(w2.astype(np.float32)), requires_grad=False)
        self.q_in = Quantize(input_scale, 0, torch.qint8)
        self.dq_scores = DeQuantize()
        self.q_probs = Quantize(prob_scale, 0, torch.qint8)
        self.score_scale = float(score_scale)

    def forward(self, x):
        x_q = self.q_in(x)
        scores_q = npu_matmul(
            self.w1,
            x_q,
            multiplier=1,
            shift=0,
            activation="none",
            in_dtype="int8",
            out_dtype="int16",
            output_scale=self.score_scale,
            output_zero_point=0,
        )
        scores_q = mark_for_verify(scores_q, "scores_q")
        scores = self.dq_scores(scores_q)
        probs = torch.softmax(scores, dim=0)
        q_probs = self.q_probs(probs)
        q_probs = mark_for_verify(q_probs, "q_probs")
        return npu_matmul(
            self.w2,
            q_probs,
            multiplier=1,
            shift=0,
            activation="none",
            in_dtype="int8",
            out_dtype="int16",
        )


def build_qdq_chain_artifact(
    seed: int = 17,
    dim: int = 16,
    input_scale: float = 0.25,
    score_scale: float = 0.125,
    prob_scale: float = 1.0 / 64.0,
):
    rng = np.random.default_rng(seed)
    x_int = rng.integers(0, 8, size=(dim, dim), dtype=np.int16)
    w1 = rng.integers(-3, 4, size=(dim, dim), dtype=np.int8)
    w2 = rng.integers(-3, 4, size=(dim, dim), dtype=np.int8)
    x_float = x_int.astype(np.float32) * np.float32(input_scale)

    module = QDQChainModule(
        w1,
        w2,
        input_scale=input_scale,
        score_scale=score_scale,
        prob_scale=prob_scale,
    )
    artifact = compile_module(module, (torch.tensor(x_float),))
    verify_names = {spec.verify_label: name for name, spec in artifact.plan.tensors.items() if spec.verify_label}
    expected = {
        "scores_q": np.array(artifact.expected_tensors[verify_names["scores_q"]], copy=True),
        "q_probs": np.array(artifact.expected_tensors[verify_names["q_probs"]], copy=True),
        artifact.plan.outputs[0]: np.array(artifact.expected_tensors[artifact.plan.outputs[0]], copy=True),
    }

    return {
        "artifact": artifact,
        "inputs": {"x": x_float.astype(np.float32)},
        "expected": expected,
    }
