from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional

import torch
import torch.nn as nn


class AttentionPooler(nn.Module):
    def __init__(self, hidden_size: int, attn_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(attn_dim))
        self.key = nn.Linear(hidden_size, attn_dim, bias=False)
        self.score = nn.Linear(attn_dim, 1, bias=False)
        nn.init.normal_(self.query, mean=0.0, std=1.0 / math.sqrt(attn_dim))

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        keys = torch.tanh(self.key(hidden_states))  # [B, T, D]
        scores = self.score(keys).squeeze(-1)  # [B, T]
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.bool(), float("-inf"))
        attn_weights = torch.softmax(scores, dim=-1).unsqueeze(-1)  # [B, T, 1]
        return torch.sum(hidden_states * attn_weights, dim=1)


class Pooler(nn.Module):
    def __init__(self, pooler_type: str, hidden_size: int, attn_dim: Optional[int] = None):
        super().__init__()
        pooler_type = pooler_type.lower()
        self.pooler_type = pooler_type
        if pooler_type == "attention":
            if attn_dim is None:
                raise ValueError("attention pooler requires attn_dim")
            self.attention = AttentionPooler(hidden_size, attn_dim)
        else:
            self.attention = None

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.pooler_type == "cls":
            return hidden_states[:, 0]
        if self.pooler_type == "mean":
            if attention_mask is None:
                return hidden_states.mean(dim=1)
            mask = attention_mask.unsqueeze(-1).float()
            summed = torch.sum(hidden_states * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1.0)
            return summed / counts
        if self.pooler_type == "max":
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).bool()
                hidden_states = hidden_states.masked_fill(~mask, float("-inf"))
            return hidden_states.max(dim=1).values
        if self.pooler_type == "attention":
            if attention_mask is None:
                attention_mask = torch.ones(hidden_states.size()[:2], dtype=torch.long, device=hidden_states.device)
            return self.attention(hidden_states, attention_mask)
        raise ValueError(f"Unknown pooler_type: {self.pooler_type}")


def build_pooler(pooler_type: str, hidden_size: int, attn_dim: int | None = None) -> nn.Module:
    """Factory that returns a pooling module."""
    return Pooler(pooler_type=pooler_type, hidden_size=hidden_size, attn_dim=attn_dim)


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


@dataclass
class HeadConfig:
    pooler_type: str
    type: str
    activation: str
    dropout: float
    norm: str
    hidden_dim: Optional[int] = None
    layers: Optional[int] = None
    attn_dim: Optional[int] = None


class ClassificationHead(nn.Module):
    def __init__(self, cfg: HeadConfig, input_dim: int, num_classes: int):
        super().__init__()
        self.pooler = build_pooler(cfg.pooler_type, hidden_size=input_dim, attn_dim=cfg.attn_dim)

        layers: list[nn.Module] = []
        dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

        if cfg.norm == "layernorm":
            layers.append(nn.LayerNorm(input_dim))

        head_type = cfg.type.lower()
        if head_type == "linear":
            layers.extend([dropout, nn.Linear(input_dim, num_classes)])
        elif head_type == "mlp":
            hidden_dim = cfg.hidden_dim or input_dim
            num_layers = cfg.layers or 1
            in_features = input_dim
            for layer_idx in range(num_layers):
                out_features = hidden_dim if layer_idx < num_layers - 1 else num_classes
                layers.append(nn.Linear(in_features, out_features))
                if layer_idx < num_layers - 1:
                    layers.append(_activation(cfg.activation))
                    layers.append(dropout)
                in_features = hidden_dim
        else:
            raise ValueError(f"Unknown head type: {cfg.type}")

        self.head = nn.Sequential(*layers)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        pooled = self.pooler(hidden_states, attention_mask)
        logits = self.head(pooled)
        return logits


def build_head(cfg: dict, in_dim: int, num_classes: int) -> nn.Module:
    head_cfg = HeadConfig(
        pooler_type=cfg.get("pooler_type", "cls"),
        type=cfg.get("type", "linear"),
        activation=cfg.get("activation", "gelu"),
        dropout=float(cfg.get("dropout", 0.1)),
        norm=cfg.get("norm", "none"),
        hidden_dim=cfg.get("hidden"),
        layers=cfg.get("layers"),
        attn_dim=cfg.get("attn_dim"),
    )
    return ClassificationHead(head_cfg, input_dim=in_dim, num_classes=num_classes)
