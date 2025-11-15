from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _apply_label_smoothing(targets: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0.0:
        return targets
    return targets * (1.0 - smoothing) + 0.5 * smoothing


class BCEWithLogitsLossMulti(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = _apply_label_smoothing(targets, self.label_smoothing)
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        if self.class_weights is not None:
            loss = loss * self.class_weights
        return loss.mean()


class FocalLossMulti(nn.Module):
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.gamma = float(gamma)
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None
        self.label_smoothing = float(label_smoothing)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = _apply_label_smoothing(targets, self.label_smoothing)
        probas = torch.sigmoid(logits)
        probas = probas.clamp(min=self.eps, max=1 - self.eps)
        ce_loss = F.binary_cross_entropy(probas, targets, reduction="none")
        pt = targets * probas + (1 - targets) * (1 - probas)
        focal_factor = (1 - pt) ** self.gamma
        loss = focal_factor * ce_loss
        if self.alpha is not None:
            loss = loss * self.alpha
        return loss.mean()


def build_loss_fn(
    loss_type: str,
    class_weights: Optional[torch.Tensor],
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
) -> nn.Module:
    if loss_type == "bce":
        return BCEWithLogitsLossMulti(class_weights=class_weights, label_smoothing=label_smoothing)
    if loss_type == "focal":
        return FocalLossMulti(gamma=focal_gamma, alpha=class_weights, label_smoothing=label_smoothing)
    raise ValueError(f"Unsupported loss_type: {loss_type}")


__all__ = ["build_loss_fn", "BCEWithLogitsLossMulti", "FocalLossMulti"]
