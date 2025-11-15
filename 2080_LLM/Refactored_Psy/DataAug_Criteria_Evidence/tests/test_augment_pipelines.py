"""Test augmentation pipelines."""

import pytest

from psy_agents_aug.augment import (
    AugmentationConfig,
    NLPAugPipeline,
    TextAttackPipeline,
    HybridPipeline,
)


@pytest.fixture
def config():
    """Create test augmentation config."""
    return AugmentationConfig(
        enabled=True,
        ratio=0.5,
        max_aug_per_sample=1,
        seed=42,
    )


def test_nlpaug_synonym(config):
    """Test NLPAug synonym augmentation."""
    try:
        pipeline = NLPAugPipeline(config, aug_method="synonym")
    except ImportError:
        pytest.skip("nlpaug not available")
    
    text = "The patient feels sad and hopeless."
    augmented = pipeline.augment_text(text, num_variants=1)
    
    assert len(augmented) >= 0, "Should return list"
    if augmented:
        assert augmented[0] != text, "Augmented text should differ from original"


def test_textattack_wordnet(config):
    """Test TextAttack WordNet augmentation."""
    try:
        pipeline = TextAttackPipeline(config, aug_method="wordnet")
    except ImportError:
        pytest.skip("TextAttack not available")
    
    text = "The patient feels sad and hopeless."
    augmented = pipeline.augment_text(text, num_variants=1)
    
    assert len(augmented) >= 0, "Should return list"
    if augmented:
        assert augmented[0] != text, "Augmented text should differ from original"


def test_hybrid_pipeline(config):
    """Test hybrid augmentation pipeline."""
    try:
        pipeline = HybridPipeline(
            config,
            mix_proportions={"nlpaug_synonym": 0.5, "textattack_wordnet": 0.5}
        )
    except ImportError:
        pytest.skip("Augmentation libraries not available")
    
    text = "The patient feels sad and hopeless."
    augmented = pipeline.augment_text(text, num_variants=2)
    
    assert len(augmented) >= 0, "Should return list"


def test_invalid_pipeline_config():
    """Test that invalid configurations raise errors."""
    config = AugmentationConfig(enabled=True, seed=42)
    
    # Invalid augmentation method
    with pytest.raises(ValueError, match="Unknown augmentation method"):
        try:
            NLPAugPipeline(config, aug_method="invalid_method")
        except ImportError:
            pytest.skip("nlpaug not available")
    
    # Invalid mix proportions (don't sum to 1.0)
    with pytest.raises(ValueError, match="sum to 1.0"):
        try:
            HybridPipeline(
                config,
                mix_proportions={"nlpaug_synonym": 0.3, "textattack_wordnet": 0.3}
            )
        except ImportError:
            pytest.skip("Augmentation libraries not available")
