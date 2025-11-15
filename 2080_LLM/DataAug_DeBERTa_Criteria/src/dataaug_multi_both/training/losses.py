"""Loss functions for criteria matching training.

This module provides various loss functions for multi-label criteria
classification, including BCE, focal loss, and hybrid losses.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

logger = logging.getLogger(__name__)


class CriteriaLoss(nn.Module):
    """Loss function for criteria matching (multi-label classification)."""

    def __init__(
        self,
        loss_type: str = "bce",
        label_smoothing: float = 0.0,
        focal_gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None
    ):
        """Initialize criteria loss.

        Args:
            loss_type: Type of loss (bce, focal, weighted_bce, hybrid)
            label_smoothing: Label smoothing factor (0.0-0.2)
            focal_gamma: Gamma parameter for focal loss (1.0-5.0)
            pos_weight: Positive class weight for weighted BCE
        """
        super().__init__()

        self.loss_type = loss_type
        self.label_smoothing = label_smoothing

        # Select loss function
        if loss_type == "bce":
            self.loss_fn = BCELoss(label_smoothing=label_smoothing)
        elif loss_type == "focal":
            self.loss_fn = FocalLoss(gamma=focal_gamma)
        elif loss_type == "weighted_bce":
            self.loss_fn = WeightedBCELoss(
                pos_weight=pos_weight,
                label_smoothing=label_smoothing
            )
        elif loss_type == "hybrid":
            self.loss_fn = HybridLoss(
                gamma=focal_gamma,
                label_smoothing=label_smoothing
            )
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        logger.info(
            f"Initialized CriteriaLoss: "
            f"type={loss_type}, "
            f"label_smoothing={label_smoothing}"
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Compute criteria loss.

        Args:
            logits: Criteria predictions [batch_size, num_labels]
            labels: Criteria ground truth [batch_size, num_labels]

        Returns:
            Loss value
        """
        return self.loss_fn(logits, labels.float())


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

    def __init__(
        self,
        pos_weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
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

        loss = F.binary_cross_entropy_with_logits(
            logits, labels, pos_weight=self.pos_weight
        )
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
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')

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

        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
        focal_loss = self.alpha * focal_weight * bce_loss

        return focal_loss.mean()


class HybridLoss(nn.Module):
    """Hybrid loss combining BCE and focal loss."""

    def __init__(
        self,
        alpha: float = 0.5,
        gamma: float = 2.0,
        label_smoothing: float = 0.0
    ):
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


# Factory function for easy loss creation
def create_loss(
    loss_type: str = "bce",
    label_smoothing: float = 0.0,
    focal_gamma: float = 2.0,
    pos_weight: Optional[torch.Tensor] = None
) -> CriteriaLoss:
    """Factory function to create loss function.

    Args:
        loss_type: Type of loss (bce, focal, weighted_bce, hybrid)
        label_smoothing: Label smoothing factor
        focal_gamma: Gamma parameter for focal loss
        pos_weight: Positive class weight

    Returns:
        Initialized CriteriaLoss

    Example:
        loss_fn = create_loss(loss_type="focal", focal_gamma=2.5)
    """
    return CriteriaLoss(
        loss_type=loss_type,
        label_smoothing=label_smoothing,
        focal_gamma=focal_gamma,
        pos_weight=pos_weight
    )
