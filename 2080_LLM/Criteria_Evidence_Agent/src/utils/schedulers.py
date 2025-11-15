"""Learning rate scheduler utilities."""

import math

import torch
from omegaconf import DictConfig
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup


def get_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_cfg: DictConfig,
    total_steps: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler based on configuration.

    Args:
        optimizer: PyTorch optimizer
        scheduler_cfg: Scheduler configuration
        total_steps: Total number of training steps

    Returns:
        Learning rate scheduler

    Raises:
        ValueError: If scheduler name is not supported
    """
    warmup_steps = int(scheduler_cfg.get("warmup_ratio", 0.0) * total_steps)
    name = scheduler_cfg.get("name", "linear").lower()

    if name == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

    if name == "cosine":
        num_cycles = scheduler_cfg.get("cosine_cycles", 0.5)
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=num_cycles,
        )

    if name == "onecycle":
        max_lr = scheduler_cfg.get("onecycle_max_lr", 5e-5)
        pct_start = scheduler_cfg.get("onecycle_pct_start", 0.3)
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            anneal_strategy='cos',
        )

    if name == "plateau":
        patience = scheduler_cfg.get("plateau_patience", 2)
        factor = scheduler_cfg.get("plateau_factor", 0.5)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=factor,
            patience=patience,
            verbose=True,
        )

    if name == "polynomial":
        power = scheduler_cfg.get("polynomial_power", 1.0)

        def polynomial_decay(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return (1.0 - progress) ** power

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=polynomial_decay)

    raise ValueError(f"Unsupported scheduler: {name}")
