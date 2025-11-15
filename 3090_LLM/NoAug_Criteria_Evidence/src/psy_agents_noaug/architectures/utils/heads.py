from __future__ import annotations

from collections.abc import Sequence
from typing import Union

import torch
from torch import nn

ActivationLike = Union[str, nn.Module]


def _activation(name: ActivationLike) -> nn.Module:
    if isinstance(name, nn.Module):
        return name
    key = (name or "gelu").lower()
    if key == "relu":
        return nn.ReLU()
    if key == "silu":
        return nn.SiLU()
    if key in ("gelu", "geglu"):
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'")


def _as_hidden_dims(
    hidden: int | Sequence[int] | None,
    layers: int,
    *,
    fallback: int,
) -> list[int]:
    if layers <= 1:
        return []
    if hidden is None:
        return [fallback] * (layers - 1)
    if isinstance(hidden, Sequence) and not isinstance(hidden, (str, bytes)):
        dims = [int(h) for h in hidden]
    else:
        dims = [int(hidden)] * (layers - 1)
    if len(dims) != layers - 1:
        raise ValueError("Number of hidden dimensions must equal layers - 1.")
    return dims


class SequencePooler(nn.Module):
    def __init__(self, hidden_size: int, pooling: str = "cls") -> None:
        super().__init__()
        pooling = pooling.lower()
        if pooling not in {"cls", "mean", "max", "attn"}:
            raise ValueError(f"Unsupported pooling '{pooling}'")
        self.pooling = pooling
        if self.pooling == "attn":
            self.attention = nn.Linear(hidden_size, 1)
        else:
            self.attention = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pooler_output: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.pooling == "cls":
            if pooler_output is not None:
                return pooler_output
            return hidden_states[:, 0]

        mask = attention_mask
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            mask = mask.to(dtype=hidden_states.dtype)

        if self.pooling == "mean":
            if mask is None:
                return hidden_states.mean(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            summed = (hidden_states * mask).sum(dim=1)
            return summed / denom

        if self.pooling == "max":
            if mask is None:
                return hidden_states.max(dim=1).values
            fill = torch.finfo(hidden_states.dtype).min
            masked = hidden_states.masked_fill(mask == 0, fill)
            return masked.max(dim=1).values

        # Attention pooling
        scores = self.attention(hidden_states).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(
                mask.squeeze(-1) == 0, torch.finfo(scores.dtype).min
            )
        weights = torch.softmax(scores, dim=-1)
        if mask is not None:
            weights = weights * mask.squeeze(-1)
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)
        context = torch.bmm(weights.unsqueeze(1), hidden_states)
        return context.squeeze(1)


class ClassificationHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        *,
        layers: int = 1,
        hidden: int | Sequence[int] | None = None,
        activation: ActivationLike = "gelu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        hidden_dims = _as_hidden_dims(hidden, layers, fallback=input_dim)
        dims = [input_dim] + hidden_dims
        blocks: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            blocks.append(nn.Linear(in_dim, out_dim))
            blocks.append(_activation(activation))
            blocks.append(nn.Dropout(dropout))
        self.ffn = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(dims[-1], num_labels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.ffn(inputs)
        x = self.dropout(x)
        return self.out(x)


class SpanPredictionHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        *,
        layers: int = 1,
        hidden: int | Sequence[int] | None = None,
        activation: ActivationLike = "gelu",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be >= 1")
        hidden_dims = _as_hidden_dims(hidden, layers, fallback=input_dim)
        dims = [input_dim] + hidden_dims
        self.layers = nn.ModuleList()
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(_activation(activation))
            self.layers.append(nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(dims[-1], 2)

    def forward(
        self, sequence_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = sequence_output
        if self.layers:
            for layer in self.layers:
                x = layer(x)
        x = self.dropout(x)
        logits = self.out_proj(x)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)
