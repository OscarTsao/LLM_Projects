"""Loss functions for multi-task training.

This module provides various loss functions for criteria matching and
evidence binding tasks, including BCE, focal loss, and hybrid losses.
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """Combined loss for criteria matching and evidence binding.

    Combines:
    - Criteria matching loss (multi-label classification)
    - Evidence binding loss (span extraction)
    """

    def __init__(
        self,
        criteria_loss_weight: float = 1.0,
        evidence_loss_weight: float = 1.0,
        criteria_loss_type: str = "bce",
        label_smoothing: float = 0.0,
    ):
        """Initialize multi-task loss.

        Args:
            criteria_loss_weight: Weight for criteria matching loss
            evidence_loss_weight: Weight for evidence binding loss
            criteria_loss_type: Type of criteria loss (bce, focal, weighted_bce)
            label_smoothing: Label smoothing factor (0.0-0.2)
        """
        super().__init__()

        self.criteria_loss_weight = criteria_loss_weight
        self.evidence_loss_weight = evidence_loss_weight
        self.label_smoothing = label_smoothing

        # Criteria matching loss
        if criteria_loss_type == "bce":
            self.criteria_loss_fn = BCELoss(label_smoothing=label_smoothing)
        elif criteria_loss_type == "focal":
            self.criteria_loss_fn = FocalLoss(gamma=2.0)
        elif criteria_loss_type == "weighted_bce":
            self.criteria_loss_fn = WeightedBCELoss(label_smoothing=label_smoothing)
        else:
            raise ValueError(f"Unknown criteria loss type: {criteria_loss_type}")

        # Evidence binding loss (cross-entropy for start/end positions)
        self.evidence_loss_fn = nn.CrossEntropyLoss()

        logger.info(
            f"Initialized MultiTaskLoss: "
            f"criteria_weight={criteria_loss_weight}, "
            f"evidence_weight={evidence_loss_weight}, "
            f"criteria_type={criteria_loss_type}"
        )

    def forward(
        self,
        criteria_logits: torch.Tensor,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        criteria_labels: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss.

        Args:
            criteria_logits: Criteria predictions [batch_size, num_labels]
            start_logits: Start position logits [batch_size, seq_len]
            end_logits: End position logits [batch_size, seq_len]
            criteria_labels: Criteria ground truth [batch_size, num_labels]
            start_positions: Start position labels [batch_size]
            end_positions: End position labels [batch_size]

        Returns:
            Combined loss
        """
        # Criteria matching loss
        criteria_loss = self.criteria_loss_fn(criteria_logits, criteria_labels.float())

        # Evidence binding loss
        start_loss = self.evidence_loss_fn(start_logits, start_positions)
        end_loss = self.evidence_loss_fn(end_logits, end_positions)
        evidence_loss = (start_loss + end_loss) / 2.0

        # Combined loss
        total_loss = (
            self.criteria_loss_weight * criteria_loss + self.evidence_loss_weight * evidence_loss
        )

        return total_loss


class BCELoss(nn.Module):
    """Binary cross-entropy loss with optional label smoothing."""

    def __init__(self, label_smoothing: float = 0.0):
        """Initialize BCE loss.

        Args:
            label_smoothing: Label smoothing factor (0.0-0.2)
        """
        super().__init__()

        if not 0.0 <= label_smoothing <= 0.2:
            raise ValueError(f"label_smoothing must be in [0, 0.2], got {label_smoothing}")

        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute BCE loss.

        Args:
            logits: Predicted logits [batch_size, num_labels]
            labels: Ground truth labels [batch_size, num_labels]

        Returns:
            BCE loss
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        return loss


class WeightedBCELoss(nn.Module):
    """Weighted BCE loss for handling class imbalance."""

    def __init__(self, pos_weight: torch.Tensor | None = None, label_smoothing: float = 0.0):
        """Initialize weighted BCE loss.

        Args:
            pos_weight: Weight for positive class [num_labels]
            label_smoothing: Label smoothing factor
        """
        super().__init__()

        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute weighted BCE loss.

        Args:
            logits: Predicted logits [batch_size, num_labels]
            labels: Ground truth labels [batch_size, num_labels]

        Returns:
            Weighted BCE loss
        """
        # Apply label smoothing
        if self.label_smoothing > 0:
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        loss = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=self.pos_weight)
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance.

    Focal Loss = -alpha * (1 - p_t)^gamma * log(p_t)

    Reference: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """Initialize focal loss.

        Args:
            alpha: Weighting factor (0.0-1.0)
            gamma: Focusing parameter (1.0-5.0)
        """
        super().__init__()

        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        if not 1.0 <= gamma <= 5.0:
            raise ValueError(f"gamma must be in [1, 5], got {gamma}")

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            logits: Predicted logits [batch_size, num_labels]
            labels: Ground truth labels [batch_size, num_labels]

        Returns:
            Focal loss
        """
        # Compute probabilities
        probs = torch.sigmoid(logits)

        # Compute focal weight
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_weight = (1 - p_t) ** self.gamma

        # Compute BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")

        # Apply focal weight and alpha
        focal_loss = self.alpha * focal_weight * bce_loss

        return focal_loss.mean()


class AdaptiveFocalLoss(nn.Module):
    """Adaptive focal loss with learnable gamma parameter."""

    def __init__(self, alpha: float = 0.25, gamma_init: float = 2.0):
        """Initialize adaptive focal loss.

        Args:
            alpha: Weighting factor
            gamma_init: Initial gamma value
        """
        super().__init__()

        self.alpha = alpha
        self.gamma = nn.Parameter(torch.tensor(gamma_init))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute adaptive focal loss.

        Args:
            logits: Predicted logits
            labels: Ground truth labels

        Returns:
            Adaptive focal loss
        """
        # Clamp gamma to valid range
        gamma = torch.clamp(self.gamma, 1.0, 5.0)

        probs = torch.sigmoid(logits)
        p_t = probs * labels + (1 - probs) * (1 - labels)
        focal_weight = (1 - p_t) ** gamma

        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        focal_loss = self.alpha * focal_weight * bce_loss

        return focal_loss.mean()


class HybridLoss(nn.Module):
    """Hybrid loss combining BCE and focal loss."""

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0, label_smoothing: float = 0.0):
        """Initialize hybrid loss.

        Args:
            alpha: Weight for BCE vs focal (0.1-0.9)
            gamma: Focal loss gamma parameter
            label_smoothing: Label smoothing factor
        """
        super().__init__()

        if not 0.1 <= alpha <= 0.9:
            raise ValueError(f"alpha must be in [0.1, 0.9], got {alpha}")

        self.alpha = alpha
        self.bce_loss = BCELoss(label_smoothing=label_smoothing)
        self.focal_loss = FocalLoss(alpha=0.25, gamma=gamma)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute hybrid loss.

        Args:
            logits: Predicted logits
            labels: Ground truth labels

        Returns:
            Hybrid loss
        """
        bce = self.bce_loss(logits, labels)
        focal = self.focal_loss(logits, labels)

        return self.alpha * bce + (1 - self.alpha) * focal
