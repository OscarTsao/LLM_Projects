from __future__ import annotations

from typing import Mapping

import torch

from .lion import Lion


def _parameter_groups(
    model: torch.nn.Module,
    *,
    lr_encoder: float,
    lr_head: float,
    weight_decay: float,
) -> list[dict]:
    encoder_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []
    other_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("encoder"):
            encoder_params.append(param)
        elif name.startswith("head_evidence") or name.startswith("head_criteria"):
            head_params.append(param)
        else:
            other_params.append(param)

    groups: list[dict] = []
    if encoder_params:
        groups.append({"params": encoder_params, "lr": lr_encoder, "weight_decay": weight_decay})
    if head_params:
        groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})
    if other_params:
        groups.append({"params": other_params, "lr": lr_head, "weight_decay": weight_decay})
    return groups


def build_optimizer(cfg: Mapping[str, object], model: torch.nn.Module) -> torch.optim.Optimizer:
    optim_cfg = cfg["optim"]
    optimizer_type = optim_cfg.get("optimizer", "adamw").lower()

    lr_encoder = float(optim_cfg["lr_encoder"])
    lr_head = float(optim_cfg["lr_head"])
    weight_decay = float(optim_cfg.get("weight_decay", 0.0))
    betas = tuple(optim_cfg.get("betas", (0.9, 0.999)))
    eps = float(optim_cfg.get("eps", 1e-8))

    param_groups = _parameter_groups(
        model,
        lr_encoder=lr_encoder,
        lr_head=lr_head,
        weight_decay=weight_decay,
    )

    if optimizer_type == "adamw":
        return torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    if optimizer_type == "lion":
        return Lion(param_groups, lr=lr_head, betas=betas, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer type: {optimizer_type}")
