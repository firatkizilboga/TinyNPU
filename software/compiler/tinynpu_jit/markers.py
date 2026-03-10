from __future__ import annotations


def mark_for_verify(tensor, label: str | None = None):
    metadata = getattr(tensor, "_tinynpu_verify", None)
    if metadata is None and hasattr(tensor, "__dict__"):
        tensor._tinynpu_verify = []
        metadata = tensor._tinynpu_verify
    if isinstance(metadata, list):
        metadata.append(label)
    return tensor


def quantize_for_npu(
    tensor,
    scale: float,
    zero_point: int = 0,
    dtype: str = "int16",
):
    return tensor


def npu_matmul(
    lhs,
    rhs,
    *,
    multiplier: int = 1,
    shift: int = 0,
    activation: str = "none",
    in_dtype: str = "int16",
    out_dtype: str = "int16",
):
    import torch

    result = torch.matmul(lhs, rhs)
    if activation == "relu":
        return torch.relu(result)
    return result


try:
    import torch.fx  # type: ignore

    torch.fx.wrap("mark_for_verify")
    torch.fx.wrap("quantize_for_npu")
    torch.fx.wrap("npu_matmul")
except Exception:
    pass
