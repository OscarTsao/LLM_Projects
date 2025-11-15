from __future__ import annotations

import torch
import torch.nn as nn


def build_criterion(loss_cfg: dict, pos_weight: torch.Tensor | None = None) -> nn.Module:
    name = loss_cfg.get("loss.name", "bce")
    if name == "bce":
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "weighted_bce":
        assert pos_weight is not None, "weighted_bce requires pos_weight tensor"
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if name == "focal":
        gamma = float(loss_cfg.get("loss.focal.gamma", 2.0))
        alpha_pos = float(loss_cfg.get("loss.focal.alpha_pos", 0.25))
        return FocalLoss(gamma=gamma, alpha_pos=alpha_pos, reduction="mean")
    if name == "asymmetric":
        gpos = float(loss_cfg.get("loss.asym.gamma_pos", 0.0))
        gneg = float(loss_cfg.get("loss.asym.gamma_neg", 4.0))
        return AsymmetricLoss(gamma_pos=gpos, gamma_neg=gneg, reduction="mean")
    raise ValueError(f"Unknown loss.name: {name}")

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha_pos: float = 0.25, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha_pos = alpha_pos
        self.reduction = reduction
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        alpha = self.alpha_pos * targets + (1 - self.alpha_pos) * (1 - targets)
        loss = -alpha * (1 - pt).pow(self.gamma) * torch.log(pt.clamp_min(1e-8))
        return _reduce(loss, self.reduction)

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_pos: float = 0.0, gamma_neg: float = 4.0, reduction: str = "mean"):
        super().__init__()
        self.gp = gamma_pos
        self.gn = gamma_neg
        self.reduction = reduction
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        xs = torch.sigmoid(logits)
        xs_pos = xs
        xs_neg = 1.0 - xs
        log_pos = torch.log(xs_pos.clamp_min(1e-8))
        log_neg = torch.log(xs_neg.clamp_min(1e-8))
        loss_pos = (1 - xs_pos) ** self.gp * -targets * log_pos
        loss_neg = (1 - xs_neg) ** self.gn * -(1 - targets) * log_neg
        loss = loss_pos + loss_neg
        return _reduce(loss, self.reduction)

def _reduce(x: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return x.mean()
    if reduction == "sum":
        return x.sum()
    return x
