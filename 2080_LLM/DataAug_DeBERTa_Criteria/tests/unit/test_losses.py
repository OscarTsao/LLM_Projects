"""Unit tests for loss functions."""

import pytest
import torch
from src.dataaug_multi_both.training.losses import (
    CriteriaLoss,
    BCELoss,
    WeightedBCELoss,
    FocalLoss,
    AdaptiveFocalLoss,
    HybridLoss,
    create_loss
)


class TestBCELoss:
    """Test suite for BCELoss."""

    def test_bce_loss_computation(self):
        """Test that BCE loss computes correctly."""
        loss_fn = BCELoss()

        logits = torch.randn(2, 9)
        labels = torch.randint(0, 2, (2, 9)).float()

        loss = loss_fn(logits, labels)

        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_bce_loss_with_label_smoothing(self):
        """Test BCE loss with label smoothing."""
        loss_fn = BCELoss(label_smoothing=0.1)

        logits = torch.randn(2, 9)
        labels = torch.ones(2, 9)  # All ones

        loss = loss_fn(logits, labels)

        assert loss.item() >= 0.0

    def test_bce_loss_invalid_label_smoothing(self):
        """Test that invalid label smoothing raises error."""
        with pytest.raises(ValueError, match="label_smoothing must be in"):
            BCELoss(label_smoothing=0.5)


class TestWeightedBCELoss:
    """Test suite for WeightedBCELoss."""

    def test_weighted_bce_loss_computation(self):
        """Test that weighted BCE loss computes correctly."""
        loss_fn = WeightedBCELoss()

        logits = torch.randn(2, 9)
        labels = torch.randint(0, 2, (2, 9)).float()

        loss = loss_fn(logits, labels)

        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_weighted_bce_loss_with_pos_weight(self):
        """Test weighted BCE loss with positive class weights."""
        pos_weight = torch.ones(9) * 2.0  # Weight positive class 2x
        loss_fn = WeightedBCELoss(pos_weight=pos_weight)

        logits = torch.randn(2, 9)
        labels = torch.randint(0, 2, (2, 9)).float()

        loss = loss_fn(logits, labels)

        assert loss.item() >= 0.0


class TestFocalLoss:
    """Test suite for FocalLoss."""

    def test_focal_loss_computation(self):
        """Test that focal loss computes correctly."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        logits = torch.randn(2, 9)
        labels = torch.randint(0, 2, (2, 9)).float()

        loss = loss_fn(logits, labels)

        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_focal_loss_invalid_alpha(self):
        """Test that invalid alpha raises error."""
        with pytest.raises(ValueError, match="alpha must be in"):
            FocalLoss(alpha=1.5)

    def test_focal_loss_invalid_gamma(self):
        """Test that invalid gamma raises error."""
        with pytest.raises(ValueError, match="gamma must be in"):
            FocalLoss(gamma=10.0)

    def test_focal_loss_focuses_on_hard_examples(self):
        """Test that focal loss down-weights easy examples."""
        loss_fn = FocalLoss(alpha=0.25, gamma=2.0)

        # Easy example (high confidence, correct)
        easy_logits = torch.tensor([[10.0, -10.0, 10.0]])
        easy_labels = torch.tensor([[1.0, 0.0, 1.0]])
        easy_loss = loss_fn(easy_logits, easy_labels)

        # Hard example (low confidence)
        hard_logits = torch.tensor([[0.5, -0.5, 0.5]])
        hard_labels = torch.tensor([[1.0, 0.0, 1.0]])
        hard_loss = loss_fn(hard_logits, hard_labels)

        # Hard examples should have higher loss
        assert hard_loss > easy_loss


class TestAdaptiveFocalLoss:
    """Test suite for AdaptiveFocalLoss."""

    def test_adaptive_focal_loss_computation(self):
        """Test that adaptive focal loss computes correctly."""
        loss_fn = AdaptiveFocalLoss(alpha=0.25, gamma_init=2.0)

        logits = torch.randn(2, 9)
        labels = torch.randint(0, 2, (2, 9)).float()

        loss = loss_fn(logits, labels)

        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_adaptive_focal_loss_gamma_is_learnable(self):
        """Test that gamma parameter is learnable."""
        loss_fn = AdaptiveFocalLoss(gamma_init=2.0)

        # Check that gamma is a parameter
        assert hasattr(loss_fn, 'gamma')
        assert loss_fn.gamma.requires_grad

    def test_adaptive_focal_loss_gamma_clamped(self):
        """Test that gamma is clamped to valid range."""
        loss_fn = AdaptiveFocalLoss(gamma_init=2.0)

        # Manually set gamma to invalid value
        loss_fn.gamma.data = torch.tensor(10.0)

        logits = torch.randn(2, 9)
        labels = torch.randint(0, 2, (2, 9)).float()

        # Should not raise error (gamma is clamped internally)
        loss = loss_fn(logits, labels)
        assert loss.item() >= 0.0


class TestHybridLoss:
    """Test suite for HybridLoss."""

    def test_hybrid_loss_computation(self):
        """Test that hybrid loss computes correctly."""
        loss_fn = HybridLoss(alpha=0.5, gamma=2.0)

        logits = torch.randn(2, 9)
        labels = torch.randint(0, 2, (2, 9)).float()

        loss = loss_fn(logits, labels)

        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_hybrid_loss_invalid_alpha(self):
        """Test that invalid alpha raises error."""
        with pytest.raises(ValueError, match="alpha must be in"):
            HybridLoss(alpha=0.05)

        with pytest.raises(ValueError, match="alpha must be in"):
            HybridLoss(alpha=0.95)

    def test_hybrid_loss_combines_bce_and_focal(self):
        """Test that hybrid loss is combination of BCE and focal."""
        alpha = 0.7
        loss_fn = HybridLoss(alpha=alpha, gamma=2.0)

        logits = torch.randn(2, 9)
        labels = torch.randint(0, 2, (2, 9)).float()

        hybrid_loss = loss_fn(logits, labels)

        # Hybrid loss should be positive
        assert hybrid_loss.item() > 0.0


class TestCriteriaLoss:
    """Test suite for CriteriaLoss wrapper."""

    def test_criteria_loss_bce(self):
        """Test criteria loss with BCE."""
        loss_fn = CriteriaLoss(loss_type="bce")

        logits = torch.randn(2, 9)
        labels = torch.randint(0, 2, (2, 9))

        loss = loss_fn(logits, labels)

        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_criteria_loss_focal(self):
        """Test criteria loss with focal loss."""
        loss_fn = CriteriaLoss(loss_type="focal", focal_gamma=2.5)

        logits = torch.randn(2, 9)
        labels = torch.randint(0, 2, (2, 9))

        loss = loss_fn(logits, labels)

        assert loss.item() >= 0.0

    def test_criteria_loss_weighted_bce(self):
        """Test criteria loss with weighted BCE."""
        pos_weight = torch.ones(9) * 2.0
        loss_fn = CriteriaLoss(loss_type="weighted_bce", pos_weight=pos_weight)

        logits = torch.randn(2, 9)
        labels = torch.randint(0, 2, (2, 9))

        loss = loss_fn(logits, labels)

        assert loss.item() >= 0.0

    def test_criteria_loss_hybrid(self):
        """Test criteria loss with hybrid loss."""
        loss_fn = CriteriaLoss(loss_type="hybrid", focal_gamma=2.0)

        logits = torch.randn(2, 9)
        labels = torch.randint(0, 2, (2, 9))

        loss = loss_fn(logits, labels)

        assert loss.item() >= 0.0

    def test_criteria_loss_invalid_type(self):
        """Test that invalid loss type raises error."""
        with pytest.raises(ValueError, match="Unknown loss type"):
            CriteriaLoss(loss_type="invalid")


class TestCreateLoss:
    """Test suite for create_loss factory function."""

    def test_create_loss_default(self):
        """Test creating loss with defaults."""
        loss_fn = create_loss()

        assert isinstance(loss_fn, CriteriaLoss)
        assert loss_fn.loss_type == "bce"

    def test_create_loss_focal(self):
        """Test creating focal loss."""
        loss_fn = create_loss(loss_type="focal", focal_gamma=2.5)

        assert isinstance(loss_fn, CriteriaLoss)
        assert loss_fn.loss_type == "focal"

    def test_create_loss_with_label_smoothing(self):
        """Test creating loss with label smoothing."""
        loss_fn = create_loss(loss_type="bce", label_smoothing=0.1)

        assert isinstance(loss_fn, CriteriaLoss)
        assert loss_fn.label_smoothing == 0.1
