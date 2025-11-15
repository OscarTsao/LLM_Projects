from __future__ import annotations

import optuna

from src.dataaug_multi_both.hpo.search_space import (
    FORBIDDEN_PARAMS,
    narrow_numeric,
    stage_b_space_from_winner,
    suggest,
)


def test_suggest_returns_expected_keys():
    fixed = optuna.trial.FixedTrial(
        {
            "head.type": "mlp",
            "pooling": "cls",
            "head.dropout": 0.2,
            "head.mlp.layers": 2,
            "head.mlp.hidden_dim": 512,
            "head.mlp.activation": "gelu",
            "head.mlp.layernorm": True,
            "loss.name": "focal",
            "loss.focal.gamma": 2.0,
            "loss.focal.alpha_pos": 0.3,
            "pred.threshold.policy": "opt_global",
            "pred.threshold.global": 0.4,
            "optim.name": "adamw",
            "optim.lr_encoder": 3e-5,
            "optim.lr_head": 1e-4,
            "optim.weight_decay": 0.01,
            "sched.name": "linear",
            "sched.warmup_ratio": 0.05,
            "train.grad_clip_norm": 1.0,
        }
    )
    params = suggest(fixed)
    assert params["head.type"] == "mlp"
    assert params["pooling"] in {"cls", "mean"}
    for key in params:
        assert not any(key.startswith(prefix) for prefix in FORBIDDEN_PARAMS)


def test_stage_b_space_freezes_structural_choices():
    winner = {
        "head.type": "linear",
        "pooling": "mean",
        "loss.name": "bce",
        "optim.lr_encoder": 2.5e-5,
        "optim.lr_head": 8e-4,
        "optim.weight_decay": 5e-3,
        "sched.warmup_ratio": 0.08,
        "head.dropout": 0.1,
    }
    space = stage_b_space_from_winner(winner)
    assert space["head.type"] == ("freeze", "linear")
    assert space["pooling"] == ("freeze", "mean")
    assert space["loss.name"] == ("freeze", "bce")
    assert space["optim.lr_encoder"][0] == "float_log"
    assert space["head.dropout"][0] == "float"


def test_narrow_numeric_handles_zero_winner():
    lo, hi = narrow_numeric(0.0, 0.5, 2.0, (1e-6, 1e-2))
    assert lo <= hi
    assert lo >= 0.0
    assert hi <= 0.01
