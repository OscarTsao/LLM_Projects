from __future__ import annotations

import torch


class BoolSafeTensor(torch.Tensor):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls)

    def __bool__(self) -> bool:  # type: ignore[override]
        return False


def make_bool_safe(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, BoolSafeTensor):
        return tensor
    return tensor.as_subclass(BoolSafeTensor)
