"""Loss functions for multi-task training.

This module provides various loss functions for criteria matching and
evidence binding tasks, including BCE, focal loss, and hybrid losses.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

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
        focal_gamma: float = 2.0,
        hybrid_alpha: float = 0.5,
        adaptive_focal_gamma_init: float = 2.0,
    ):
        """Initialize multi-task loss.

        Args:
            criteria_loss_weight: Weight for criteria matching loss
            evidence_loss_weight: Weight for evidence binding loss
            criteria_loss_type: Type of criteria loss
            label_smoothing: Label smoothing factor (0.0-0.2)
            focal_gamma: Gamma value for focal-based losses
            hybrid_alpha: Weight for the first component in hybrid losses
            adaptive_focal_gamma_init: Initial gamma for adaptive focal loss
        """
        super().__init__()

        self.criteria_loss_weight = criteria_loss_weight
        self.evidence_loss_weight = evidence_loss_weight
        self.label_smoothing = label_smoothing

        # Criteria matching loss
        self.criteria_loss_type = criteria_loss_type
        self.criteria_loss_fn = self._build_criteria_loss(
            criteria_loss_type=criteria_loss_type,
            label_smoothing=label_smoothing,
            focal_gamma=focal_gamma,
            hybrid_alpha=hybrid_alpha,
            adaptive_focal_gamma_init=adaptive_focal_gamma_init,
        )

        # Evidence binding loss (cross-entropy for start/end positions)
        self.evidence_loss_fn = nn.CrossEntropyLoss()

        logger.info(
            "Initialized MultiTaskLoss: criteria_weight=%s, evidence_weight=%s, criteria_type=%s",
            criteria_loss_weight,
            evidence_loss_weight,
            criteria_loss_type,
        )

    def forward(
        self,
        criteria_logits: torch.Tensor,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        criteria_labels: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor
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
            self.criteria_loss_weight * criteria_loss +
            self.evidence_loss_weight * evidence_loss
        )

        return total_loss

    @staticmethod
    def _build_criteria_loss(
        criteria_loss_type: str,
        label_smoothing: float,
        focal_gamma: float,
        hybrid_alpha: float,
        adaptive_focal_gamma_init: float,
    ) -> nn.Module:
        """Instantiate the requested criteria loss."""

        def _make_loss(loss_name: str) -> nn.Module:
            if loss_name == "bce":
                return BCELoss(label_smoothing=label_smoothing)
            if loss_name == "weighted_bce":
                return WeightedBCELoss(label_smoothing=label_smoothing)
            if loss_name == "focal":
                return FocalLoss(gamma=focal_gamma)
            if loss_name == "adaptive_focal":
                return AdaptiveFocalLoss(gamma_init=adaptive_focal_gamma_init)
            raise ValueError(f"Unknown criteria loss component: {loss_name}")

        hybrid_pairs: Dict[str, Tuple[str, str]] = {
            "hybrid_bce_focal": ("bce", "focal"),
            "hybrid_bce_adaptive_focal": ("bce", "adaptive_focal"),
            "hybrid_weighted_bce_focal": ("weighted_bce", "focal"),
            "hybrid_weighted_bce_adaptive_focal": ("weighted_bce", "adaptive_focal"),
        }

        base_losses = {"bce", "weighted_bce", "focal", "adaptive_focal"}
        if criteria_loss_type in base_losses:
            return _make_loss(criteria_loss_type)

        if criteria_loss_type in hybrid_pairs:
            primary_key, secondary_key = hybrid_pairs[criteria_loss_type]
            primary_loss = _make_loss(primary_key)
            secondary_loss = _make_loss(secondary_key)
            return HybridLoss(
                primary_loss=primary_loss,
                secondary_loss=secondary_loss,
                alpha=hybrid_alpha,
            )

        raise ValueError(f"Unknown criteria loss type: {criteria_loss_type}")


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
    """Hybrid loss combining two criteria losses."""

    def __init__(
        self,
        primary_loss: Optional[nn.Module] = None,
        secondary_loss: Optional[nn.Module] = None,
        *,
        alpha: float = 0.5,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
    ):
        """Initialize hybrid loss.

        Args:
            primary_loss: First loss component (weight alpha). Defaults to :class:`BCELoss`.
            secondary_loss: Second loss component (weight 1 - alpha). Defaults to :class:`FocalLoss`.
            alpha: Mixing factor between the two losses.
            gamma: Gamma parameter passed to the default focal loss.
            label_smoothing: Label smoothing applied to the default BCE component.
        """
        super().__init__()

        if not 0.05 < alpha < 0.95:
            raise ValueError(f"alpha must be in (0.05, 0.95), got {alpha}")

        self.alpha = alpha
        self.primary_loss = primary_loss or BCELoss(label_smoothing=label_smoothing)
        self.secondary_loss = secondary_loss or FocalLoss(gamma=gamma)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute hybrid loss.

        Args:
            logits: Predicted logits
            labels: Ground truth labels

        Returns:
            Hybrid loss
        """
        primary = self.primary_loss(logits, labels)
        secondary = self.secondary_loss(logits, labels)

        return self.alpha * primary + (1 - self.alpha) * secondary
