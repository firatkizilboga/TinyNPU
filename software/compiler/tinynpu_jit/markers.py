from __future__ import annotations


def mark_for_verify(tensor, label: str | None = None):
    metadata = getattr(tensor, "_tinynpu_verify", None)
    if metadata is None and hasattr(tensor, "__dict__"):
        tensor._tinynpu_verify = []
        metadata = tensor._tinynpu_verify
    if isinstance(metadata, list):
        metadata.append(label)
    return tensor


try:
    import torch.fx  # type: ignore

    torch.fx.wrap("mark_for_verify")
except Exception:
    pass
