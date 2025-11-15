"""Unit tests for text augmentation."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.dataaug_multi_both.data.augmentation import (
    TEXTATTACK_AVAILABLE,
    AugmentationConfig,
    TextAugmenter,
    create_augmenter,
)


class TestAugmentationConfig:
    """Test suite for AugmentationConfig."""

    def test_config_valid_defaults(self):
        """Test that default config is valid."""
        config = AugmentationConfig()
        assert config.method == "synonym"
        assert config.probability == 0.3
        assert config.num_augmentations == 1
        assert config.preserve_non_evidence is True

    def test_config_valid_probability_range(self):
        """Test that valid probability range is accepted."""
        config = AugmentationConfig(probability=0.0)
        assert config.probability == 0.0

        config = AugmentationConfig(probability=0.5)
        assert config.probability == 0.5

    def test_config_invalid_probability_too_low(self):
        """Test that probability < 0 raises error."""
        with pytest.raises(ValueError, match="probability must be in"):
            AugmentationConfig(probability=-0.1)

    def test_config_invalid_probability_too_high(self):
        """Test that probability > 0.5 raises error."""
        with pytest.raises(ValueError, match="probability must be in"):
            AugmentationConfig(probability=0.6)

    def test_config_invalid_method(self):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Unknown augmentation method"):
            AugmentationConfig(method="invalid_method")

    def test_config_valid_methods(self):
        """Test that all valid methods are accepted."""
        valid_methods = ["synonym", "insert", "swap", "char_perturb", "eda", "none"]
        for method in valid_methods:
            config = AugmentationConfig(method=method)
            assert config.method == method


class TestTextAugmenter:
    """Test suite for TextAugmenter."""

    def test_augmenter_initialization_none(self):
        """Test that 'none' method creates disabled augmenter."""
        config = AugmentationConfig(method="none")
        augmenter = TextAugmenter(config)
        assert augmenter.augmenter is None

    def test_augmenter_initialization_valid_method(self):
        """Test that valid methods initialize augmenter."""
        config = AugmentationConfig(method="synonym")
        augmenter = TextAugmenter(config)
        # Augmenter may be None if TextAttack not available
        if TEXTATTACK_AVAILABLE:
            assert augmenter.augmenter is not None

    def test_augment_text_empty(self):
        """Test that empty text returns empty list."""
        config = AugmentationConfig(method="synonym")
        augmenter = TextAugmenter(config)
        result = augmenter.augment_text("")
        assert result == []

    def test_augment_text_disabled(self):
        """Test that disabled augmenter returns empty list."""
        config = AugmentationConfig(method="none")
        augmenter = TextAugmenter(config)
        result = augmenter.augment_text("This is a test")
        assert result == []

    @pytest.mark.skipif(not TEXTATTACK_AVAILABLE, reason="TextAttack not installed")
    def test_augment_text_returns_list(self):
        """Test that augment_text returns a list."""
        config = AugmentationConfig(method="synonym", probability=1.0)
        augmenter = TextAugmenter(config)
        result = augmenter.augment_text("This is a test sentence")
        assert isinstance(result, list)

    def test_augment_evidence_zero_probability(self):
        """Test that zero probability never augments."""
        config = AugmentationConfig(method="synonym", probability=0.0)
        augmenter = TextAugmenter(config)

        text = "This is a test with evidence"
        augmented, metadata = augmenter.augment_evidence(text)

        assert augmented == text
        assert metadata["augmentation_applied"] is False

    def test_augment_evidence_preserves_structure(self):
        """Test that augmentation preserves text structure."""
        config = AugmentationConfig(method="none", probability=0.5)
        augmenter = TextAugmenter(config)

        text = "This is a test with evidence"
        evidence_spans = [(10, 14)]  # "test"

        augmented, metadata = augmenter.augment_evidence(text, evidence_spans)

        # With 'none' method, text should be unchanged
        assert augmented == text

    def test_augment_batch_preserves_originals(self):
        """Test that batch augmentation preserves original examples."""
        config = AugmentationConfig(method="none", probability=0.5)
        augmenter = TextAugmenter(config)

        examples = [
            {"text": "Example 1"},
            {"text": "Example 2"},
        ]

        augmented = augmenter.augment_batch(examples)

        # Should at least contain originals
        assert len(augmented) >= len(examples)
        assert augmented[0]["text"] == "Example 1"
        assert augmented[1]["text"] == "Example 2"


class TestPropertyBasedAugmentation:
    """Property-based tests for augmentation."""

    @given(st.text(min_size=1, max_size=100))
    def test_augmentation_preserves_non_evidence_with_none_method(self, text):
        """Property: 'none' method never modifies text."""
        config = AugmentationConfig(method="none", probability=0.5)
        augmenter = TextAugmenter(config)

        augmented, metadata = augmenter.augment_evidence(text)

        assert augmented == text
        assert metadata["augmentation_applied"] is False

    @given(
        st.text(min_size=10, max_size=100),
        st.lists(st.tuples(st.integers(0, 50), st.integers(0, 50)), min_size=0, max_size=3)
    )
    def test_augmentation_preserves_text_length_order(self, text, spans):
        """Property: Augmentation preserves relative text structure."""
        # Filter valid spans
        valid_spans = [(s, e) for s, e in spans if 0 <= s < e <= len(text)]

        config = AugmentationConfig(method="none", probability=0.5)
        augmenter = TextAugmenter(config)

        augmented, metadata = augmenter.augment_evidence(text, valid_spans)

        # With 'none' method, should be unchanged
        assert augmented == text


class TestCreateAugmenter:
    """Test suite for create_augmenter factory function."""

    def test_create_augmenter_defaults(self):
        """Test that create_augmenter uses correct defaults."""
        augmenter = create_augmenter()
        assert augmenter.config.method == "synonym"
        assert augmenter.config.probability == 0.3

    def test_create_augmenter_custom_params(self):
        """Test that create_augmenter accepts custom parameters."""
        augmenter = create_augmenter(
            method="swap",
            probability=0.4,
            num_augmentations=2
        )
        assert augmenter.config.method == "swap"
        assert augmenter.config.probability == 0.4
        assert augmenter.config.num_augmentations == 2

