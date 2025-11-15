"""Optuna search-space construction utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna  # noqa: TCH002 - used at runtime, not just for type hints

from . import utils


@dataclass
class SpaceConstraints:
    """Per-parameter restrictions applied to the base search space."""

    categorical: dict[str, list[Any]] = field(default_factory=dict)
    floats: dict[str, tuple[float, float]] = field(default_factory=dict)
    ints: dict[str, tuple[int, int]] = field(default_factory=dict)

    def merged_choices(self, name: str, default: list[Any]) -> list[Any]:
        choices = self.categorical.get(name)
        if choices:
            return list(dict.fromkeys(choices))
        return list(dict.fromkeys(default))

    def merged_float(self, name: str, low: float, high: float) -> tuple[float, float]:
        bounds = self.floats.get(name)
        if bounds:
            return float(bounds[0]), float(bounds[1])
        return float(low), float(high)

    def merged_int(self, name: str, low: int, high: int) -> tuple[int, int]:
        bounds = self.ints.get(name)
        if bounds:
            return int(bounds[0]), int(bounds[1])
        return int(low), int(high)


class SearchSpace:
    """Agent-aware search space builder."""

    def __init__(self, agent: str, configs_root: Path | None = None) -> None:
        self.agent = agent
        config_dir = (
            configs_root or Path(__file__).resolve().parents[2] / "configs" / "model"
        )
        self.backbones = utils.load_backbone_configs(config_dir)
        if not self.backbones:
            # Fallback to well-supported HF checkpoints (8 architectures for SUPERMAX)
            self.backbones = [
                "bert-base-uncased",
                "roberta-base",
                "microsoft/deberta-v3-base",
                "google/electra-base-discriminator",
                "albert-base-v2",
                "distilbert-base-uncased",
                "YituTech/conv-bert-base",
                "xlnet-base-cased",
            ]

    def sample(
        self,
        trial: optuna.Trial,
        constraints: SpaceConstraints | None = None,
    ) -> dict[str, Any]:
        """Sample hyperparameters for ``trial`` respecting ``constraints``."""

        constraints = constraints or SpaceConstraints()
        params: dict[str, Any] = {"agent": self.agent}

        params["model.name"] = trial.suggest_categorical(
            "model.name", constraints.merged_choices("model.name", self.backbones)
        )

        max_len_low, max_len_high = constraints.merged_int("tok.max_length", 128, 512)
        params["tok.max_length"] = trial.suggest_int(
            "tok.max_length", max_len_low, max_len_high, step=64
        )

        params["model.gradient_checkpointing"] = trial.suggest_categorical(
            "model.gradient_checkpointing",
            constraints.merged_choices("model.gradient_checkpointing", [False, True]),
        )

        params["head.pooling"] = trial.suggest_categorical(
            "head.pooling",
            constraints.merged_choices(
                "head.pooling", ["cls", "mean", "max", "attention"]
            ),
        )

        params["head.hidden_mult"] = trial.suggest_categorical(
            "head.hidden_mult",
            constraints.merged_choices("head.hidden_mult", [0.0, 0.5, 1.0, 2.0]),
        )

        if params["head.hidden_mult"] > 0:
            hidden_low, hidden_high = constraints.merged_int(
                "head.hidden_dim", 128, 1024
            )
            params["head.hidden_dim"] = trial.suggest_int(
                "head.hidden_dim", hidden_low, hidden_high, step=128
            )
        else:
            params["head.hidden_dim"] = 0

        params["head.n_layers"] = trial.suggest_int(
            "head.n_layers", *constraints.merged_int("head.n_layers", 1, 3)
        )
        params["head.activation"] = trial.suggest_categorical(
            "head.activation",
            constraints.merged_choices("head.activation", ["relu", "gelu", "swish"]),
        )

        drop_low, drop_high = constraints.merged_float("head.dropout", 0.0, 0.5)
        params["head.dropout"] = trial.suggest_float(
            "head.dropout", drop_low, drop_high, step=0.05
        )

        params["optim.name"] = trial.suggest_categorical(
            "optim.name",
            constraints.merged_choices(
                "optim.name",
                ["adamw", "adam", "adafactor", "lion", "lamb", "adamw_8bit"],
            ),
        )

        lr_low, lr_high = constraints.merged_float("optim.lr", 1e-6, 3e-3)
        params["optim.lr"] = trial.suggest_float("optim.lr", lr_low, lr_high, log=True)

        wd_low, wd_high = constraints.merged_float("optim.weight_decay", 1e-6, 1e-2)
        params["optim.weight_decay"] = trial.suggest_float(
            "optim.weight_decay", wd_low, wd_high, log=True
        )

        params["sched.name"] = trial.suggest_categorical(
            "sched.name",
            constraints.merged_choices(
                "sched.name",
                ["linear", "cosine", "cosine_restart", "polynomial", "one_cycle"],
            ),
        )

        warm_low, warm_high = constraints.merged_float("sched.warmup_ratio", 0.0, 0.2)
        params["sched.warmup_ratio"] = trial.suggest_float(
            "sched.warmup_ratio", warm_low, warm_high
        )

        params["train.batch_size"] = trial.suggest_categorical(
            "train.batch_size",
            constraints.merged_choices("train.batch_size", [8, 16, 24, 32, 48, 64]),
        )
        params["train.grad_accum"] = trial.suggest_categorical(
            "train.grad_accum",
            constraints.merged_choices("train.grad_accum", [1, 2, 4]),
        )
        params["train.amp"] = trial.suggest_categorical(
            "train.amp", constraints.merged_choices("train.amp", [True, False])
        )

        label_low, label_high = constraints.merged_float(
            "reg.label_smoothing", 0.0, 0.2
        )
        params["reg.label_smoothing"] = trial.suggest_float(
            "reg.label_smoothing", label_low, label_high
        )

        params["reg.max_grad_norm"] = trial.suggest_categorical(
            "reg.max_grad_norm",
            constraints.merged_choices("reg.max_grad_norm", [0.0, 0.5, 1.0, 2.0]),
        )

        params["null.strategy"] = trial.suggest_categorical(
            "null.strategy",
            constraints.merged_choices(
                "null.strategy",
                ["none", "threshold", "ratio", "calibrated"],
            ),
        )

        thr_low, thr_high = constraints.merged_float("null.threshold", 0.0, 0.9)
        params["null.threshold"] = trial.suggest_float(
            "null.threshold", thr_low, thr_high
        )

        ratio_low, ratio_high = constraints.merged_float("null.ratio", 0.0, 0.8)
        params["null.ratio"] = trial.suggest_float("null.ratio", ratio_low, ratio_high)

        temp_low, temp_high = constraints.merged_float("null.temperature", 0.5, 3.0)
        params["null.temperature"] = trial.suggest_float(
            "null.temperature", temp_low, temp_high
        )

        # Augmentation parameters (SUPERMAX Phase 3)
        params["aug.enabled"] = trial.suggest_categorical(
            "aug.enabled", constraints.merged_choices("aug.enabled", [False, True])
        )

        if params["aug.enabled"]:
            # Probability of applying augmentation to a sample
            prob_low, prob_high = constraints.merged_float("aug.p_apply", 0.05, 0.30)
            params["aug.p_apply"] = trial.suggest_float(
                "aug.p_apply", prob_low, prob_high, step=0.05
            )

            # Number of augmentation operations per sample
            ops_low, ops_high = constraints.merged_int("aug.ops_per_sample", 1, 3)
            params["aug.ops_per_sample"] = trial.suggest_int(
                "aug.ops_per_sample", ops_low, ops_high
            )

            # Maximum fraction of tokens to replace
            replace_low, replace_high = constraints.merged_float(
                "aug.max_replace", 0.1, 0.4
            )
            params["aug.max_replace"] = trial.suggest_float(
                "aug.max_replace", replace_low, replace_high, step=0.05
            )

            # Antonym guard strategy
            params["aug.antonym_guard"] = trial.suggest_categorical(
                "aug.antonym_guard",
                constraints.merged_choices(
                    "aug.antonym_guard", ["off", "on_low_weight"]
                ),
            )

            # Method selection strategy
            params["aug.method_strategy"] = trial.suggest_categorical(
                "aug.method_strategy",
                constraints.merged_choices(
                    "aug.method_strategy", ["all", "nlpaug", "textattack", "light"]
                ),
            )
        else:
            # No augmentation - set defaults
            params["aug.p_apply"] = 0.0
            params["aug.ops_per_sample"] = 0
            params["aug.max_replace"] = 0.0
            params["aug.antonym_guard"] = "off"
            params["aug.method_strategy"] = "none"

        return params
