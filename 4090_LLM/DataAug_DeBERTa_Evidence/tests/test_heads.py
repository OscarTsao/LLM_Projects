from __future__ import annotations

import itertools

import torch

from dataaug_multi_both.model.heads import build_head


POOLERS = ["cls", "mean", "max", "attention"]
HEAD_TYPES = ["linear", "mlp"]


def _head_cfg(pooler: str, head_type: str) -> dict:
    cfg = {
        "pooler_type": pooler,
        "type": head_type,
        "activation": "gelu",
        "dropout": 0.1,
        "norm": "none",
    }
    if pooler == "attention":
        cfg["attn_dim"] = 64
    if head_type == "mlp":
        cfg["layers"] = 2
        cfg["hidden"] = 32
    return cfg


def test_all_heads_forward_shapes():
    hidden_states = torch.randn(3, 7, 16)
    attention_mask = torch.ones(3, 7, dtype=torch.long)
    for pooler, head_type in itertools.product(POOLERS, HEAD_TYPES):
        head = build_head(_head_cfg(pooler, head_type), in_dim=16, num_classes=5)
        logits = head(hidden_states, attention_mask)
        assert logits.shape == (3, 5)
        assert not torch.isnan(logits).any()
