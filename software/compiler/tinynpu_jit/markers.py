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
    output_scale: float | None = None,
    output_zero_point: int = 0,
):
    import torch

    result = torch.matmul(lhs, rhs)
    if activation == "relu":
        return torch.relu(result)
    return result


def im2col_for_npu(
    image,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
):
    import torch
    import torch.nn.functional as F

    if image.ndim != 3:
        raise ValueError(f"im2col_for_npu expects HWC image input, got shape {tuple(image.shape)}.")
    orig_dtype = image.dtype
    chw = image.permute(2, 0, 1).unsqueeze(0)
    if orig_dtype in (torch.int8, torch.int16, torch.int32):
        chw = chw.to(torch.float32)
    cols = F.unfold(chw, kernel_size=kernel_size, stride=stride, padding=padding)
    cols = cols.squeeze(0).transpose(0, 1).contiguous()
    if orig_dtype in (torch.int8, torch.int16, torch.int32):
        cols = cols.to(orig_dtype)
    return cols


try:
    import torch.fx  # type: ignore

    torch.fx.wrap("mark_for_verify")
    torch.fx.wrap("quantize_for_npu")
    torch.fx.wrap("npu_matmul")
    torch.fx.wrap("im2col_for_npu")
except Exception:
    pass
