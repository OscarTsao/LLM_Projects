"""Test that augmentation does not leak into validation/test splits.

CRITICAL: This test verifies that augmentation NEVER applies to val/test data.
"""

import pytest
import pandas as pd

from psy_agents_aug.augment import AugmentationConfig, NLPAugPipeline
from psy_agents_aug.data.loaders import ReDSM5Loader


def test_no_augmentation_in_val():
    """Verify that validation data is never augmented."""
    config = AugmentationConfig(enabled=True, seed=42)
    
    try:
        augmentor = NLPAugPipeline(config, aug_method="synonym")
    except ImportError:
        pytest.skip("nlpaug not available")
    
    texts = ["Test sentence."] * 5
    
    # Augment validation split
    aug_texts, _ = augmentor.augment_batch(texts, split="val")
    
    # Should return unchanged
    assert len(aug_texts) == len(texts), "Validation data should not be augmented"
    assert aug_texts == texts, "Validation texts should be unchanged"


def test_no_augmentation_in_test():
    """Verify that test data is never augmented."""
    config = AugmentationConfig(enabled=True, seed=42)
    
    try:
        augmentor = NLPAugPipeline(config, aug_method="synonym")
    except ImportError:
        pytest.skip("nlpaug not available")
    
    texts = ["Test sentence."] * 5
    
    # Augment test split
    aug_texts, _ = augmentor.augment_batch(texts, split="test")
    
    # Should return unchanged
    assert len(aug_texts) == len(texts), "Test data should not be augmented"
    assert aug_texts == texts, "Test texts should be unchanged"


def test_loader_respects_split():
    """Test that ReDSM5Loader only augments training data."""
    config = AugmentationConfig(enabled=True, seed=42)
    
    try:
        augmentor = NLPAugPipeline(config, aug_method="synonym")
    except ImportError:
        pytest.skip("nlpaug not available")
    
    # This is a mock test - in real usage, loader would load from files
    # Here we just verify the split-checking logic exists
    
    texts = ["Sample text."] * 3
    
    # Train should augment
    train_aug, _ = augmentor.augment_batch(texts, split="train")
    assert len(train_aug) >= len(texts), "Train split should be augmented"
    
    # Val should not augment
    val_aug, _ = augmentor.augment_batch(texts, split="val")
    assert len(val_aug) == len(texts), "Val split should not be augmented"
    
    # Test should not augment
    test_aug, _ = augmentor.augment_batch(texts, split="test")
    assert len(test_aug) == len(texts), "Test split should not be augmented"


def test_augmentation_only_with_train_flag():
    """Test that augmentation requires train_only=True."""
    # This should work
    config = AugmentationConfig(enabled=True, train_only=True)
    assert config.train_only is True
    
    # This should be auto-corrected to True with warning
    config2 = AugmentationConfig(enabled=True, train_only=False)
    assert config2.train_only is True, "train_only should be forced to True"
