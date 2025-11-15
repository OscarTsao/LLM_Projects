"""Utility helpers for setting up optimizers and schedulers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from transformers import (
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

if TYPE_CHECKING:
    from torch.optim import Optimizer


def _get_parameter_groups(
    model: torch.nn.Module,
    weight_decay: float,
) -> list[dict]:
    """Create parameter groups with and without weight decay."""
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            any(name.endswith(f"{suffix}") for suffix in ("bias", ".bias"))
            or "LayerNorm.weight" in name
            or "layer_norm.weight" in name
        ):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups: list[dict] = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})
    return (
        groups
        if groups
        else [{"params": model.parameters(), "weight_decay": weight_decay}]
    )


def create_optimizer(
    model: torch.nn.Module,
    training_cfg,
) -> Optimizer:
    """Instantiate optimizer based on configuration."""
    lr = float(training_cfg.learning_rate)
    weight_decay = float(training_cfg.weight_decay)
    name = training_cfg.optimizer.name.lower()
    betas = tuple(training_cfg.optimizer.get("betas", [0.9, 0.999]))
    eps = float(training_cfg.optimizer.get("eps", 1e-8))

    param_groups = _get_parameter_groups(model, weight_decay)

    if name == "adamw":
        return torch.optim.AdamW(param_groups, lr=lr, betas=betas, eps=eps)
    if name == "adam":
        return torch.optim.Adam(param_groups, lr=lr, betas=betas, eps=eps)

    raise ValueError(f"Unsupported optimizer: {training_cfg.optimizer.name}")


def compute_total_steps(
    num_batches: int,
    gradient_accumulation_steps: int,
    num_epochs: int,
) -> int:
    """Compute total optimizer update steps."""
    if num_batches == 0:
        return 0
    updates_per_epoch = math.ceil(num_batches / max(gradient_accumulation_steps, 1))
    return updates_per_epoch * max(num_epochs, 1)


def create_scheduler(
    optimizer: Optimizer,
    scheduler_cfg,
    total_training_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR | None:
    """Instantiate learning-rate scheduler."""
    if total_training_steps <= 0:
        return None

    scheduler_type = scheduler_cfg.type.lower()
    warmup_ratio = float(scheduler_cfg.get("warmup_ratio", 0.0))
    warmup_steps = int(total_training_steps * warmup_ratio)

    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
    if scheduler_type == "cosine_with_restarts":
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )
    if scheduler_type in {"none", "constant"}:
        return None

    raise ValueError(f"Unsupported scheduler type: {scheduler_cfg.type}")
