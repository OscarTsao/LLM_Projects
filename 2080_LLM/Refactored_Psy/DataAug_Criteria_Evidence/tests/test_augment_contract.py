"""Test augmentation contract guarantees.

This module tests that augmentation guarantees:
1. Deterministic results with same seed
2. Only applies to training data
3. Produces expected number of augmented samples
"""

import pytest

from psy_agents_aug.augment import (
    AugmentationConfig,
    NLPAugPipeline,
)


@pytest.fixture
def config():
    """Create test augmentation config."""
    return AugmentationConfig(
        enabled=True,
        ratio=0.5,
        max_aug_per_sample=1,
        seed=42,
        train_only=True,
    )


@pytest.fixture
def nlpaug_pipeline(config):
    """Create NLPAug pipeline for testing."""
    try:
        return NLPAugPipeline(config, aug_method="synonym")
    except ImportError:
        pytest.skip("nlpaug not available")


def test_deterministic_augmentation(nlpaug_pipeline):
    """Test that augmentation is deterministic with same seed."""
    text = "The patient reports feeling anxious and stressed."
    
    # Run augmentation multiple times
    result1 = nlpaug_pipeline.augment_text(text, num_variants=1)
    result2 = nlpaug_pipeline.augment_text(text, num_variants=1)
    result3 = nlpaug_pipeline.augment_text(text, num_variants=1)
    
    # All results should be identical
    assert result1 == result2 == result3, "Augmentation should be deterministic"


def test_train_only_constraint(nlpaug_pipeline):
    """Test that augmentation ONLY applies to training data."""
    texts = ["Test sentence one.", "Test sentence two."]
    
    # Augment training data
    train_aug, _ = nlpaug_pipeline.augment_batch(texts, split="train")
    assert len(train_aug) > len(texts), "Training data should be augmented"
    
    # Augment validation data (should not augment)
    val_aug, _ = nlpaug_pipeline.augment_batch(texts, split="val")
    assert len(val_aug) == len(texts), "Validation data should NOT be augmented"
    
    # Augment test data (should not augment)
    test_aug, _ = nlpaug_pipeline.augment_batch(texts, split="test")
    assert len(test_aug) == len(texts), "Test data should NOT be augmented"


def test_augmentation_count(nlpaug_pipeline):
    """Test that augmentation produces expected number of samples."""
    texts = ["Test sentence."] * 10
    
    # With max_aug_per_sample=1, we expect at most 2x original size
    aug_texts, _ = nlpaug_pipeline.augment_batch(texts, split="train")
    assert len(aug_texts) <= len(texts) * 2, "Should not exceed max augmentation"
    assert len(aug_texts) >= len(texts), "Should at least keep originals"


def test_disabled_augmentation():
    """Test that disabled augmentation returns original data."""
    config = AugmentationConfig(enabled=False)
    
    try:
        pipeline = NLPAugPipeline(config, aug_method="synonym")
    except ImportError:
        pytest.skip("nlpaug not available")
    
    texts = ["Test sentence."]
    aug_texts, _ = pipeline.augment_batch(texts, split="train")
    
    assert len(aug_texts) == len(texts), "Disabled augmentation should not change size"
    assert aug_texts == texts, "Disabled augmentation should return originals"


def test_train_only_config_enforcement():
    """Test that train_only=False raises error."""
    with pytest.raises(ValueError, match="CRITICAL"):
        config = AugmentationConfig(enabled=True, train_only=False)
        # This should raise an error in __post_init__
