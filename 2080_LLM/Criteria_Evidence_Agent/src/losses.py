from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def adaptive_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    initial_gamma: float = 2.0,
    target_positive_rate: float = 0.25,
    alpha: float = 0.25,
    min_gamma: float = 1.0,
    max_gamma: float = 5.0,
) -> torch.Tensor:
    """Adaptive focal loss adjusting gamma based on observed positive rate."""
    probs = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    positive_rate = probs.detach().mean().clamp(min=1e-6)
    ratio = positive_rate / max(target_positive_rate, 1e-6)
    gamma = torch.clamp(initial_gamma * ratio, min=min_gamma, max=max_gamma)

    alpha_factor = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    pt = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal_weight = (1.0 - pt).clamp(min=1e-6).pow(gamma)
    loss = alpha_factor * focal_weight * bce
    return loss.mean()


def multi_label_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Binary cross entropy with logits for multi-label classification.

    Args:
        logits: Model predictions (before sigmoid)
        targets: Ground truth labels (0 or 1)
        pos_weight: Positive class weights
        label_smoothing: Label smoothing factor (0.0 = no smoothing)

    Returns:
        Scalar loss value
    """
    # Apply label smoothing: targets become (1-ε) for 1 and ε for 0
    if label_smoothing > 0:
        targets = targets * (1.0 - label_smoothing) + 0.5 * label_smoothing

    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction="mean")


def token_classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    class_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Cross entropy loss for token classification."""
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
    return loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))


def span_classification_loss(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    start_positions: torch.Tensor,
    end_positions: torch.Tensor,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Cross entropy loss for span prediction (start and end indices)."""
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    start_loss = loss_fn(start_logits, start_positions)
    end_loss = loss_fn(end_logits, end_positions)
    return start_loss, end_loss
