"""Helpers for safe tensor truthiness in conditional flows.

PyTorch tensors raise on implicit boolean conversion; ``BoolSafeTensor`` makes
``if tensor`` checks safe by always returning ``False``. Useful in optional
output fields where code may guard on presence.
"""

from __future__ import annotations

import torch


class BoolSafeTensor(torch.Tensor):
    """Tensor subclass whose truthiness is always False."""

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __bool__(self) -> bool:  # type: ignore[override]
        return False


def make_bool_safe(tensor: torch.Tensor) -> torch.Tensor:
    """Cast a tensor to ``BoolSafeTensor`` to avoid accidental bool errors."""
    if isinstance(tensor, BoolSafeTensor):
        return tensor
    return tensor.as_subclass(BoolSafeTensor)
