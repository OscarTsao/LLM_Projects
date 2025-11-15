"""
Tests for augmenter registry and instantiation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import string

from aug.registry import AugmenterRegistry


class TestAugmenterRegistry:
    """Test suite for AugmenterRegistry."""
    
    @pytest.fixture
    def registry(self):
        """Create registry instance."""
        return AugmenterRegistry()
    
    def test_registry_initialization(self, registry):
        """Test that registry loads 28 augmenters."""
        assert len(registry.augmenters) == 28
    
    def test_list_augmenters(self, registry):
        """Test listing augmenters."""
        all_augs = registry.list_augmenters()
        assert len(all_augs) == 28
        
        # Test filtering by stage
        char_augs = registry.list_augmenters(stage="char")
        assert len(char_augs) == 6
        
        word_augs = registry.list_augmenters(stage="word")
        assert len(word_augs) == 5
        
        contextual_augs = registry.list_augmenters(stage="contextual")
        assert len(contextual_augs) == 4
        
        backtrans_augs = registry.list_augmenters(stage="backtranslation")
        assert len(backtrans_augs) == 5
        
        format_augs = registry.list_augmenters(stage="format")
        assert len(format_augs) == 8
    
    def test_get_augmenter_config(self, registry):
        """Test getting augmenter configuration."""
        config = registry.get_augmenter_config("random_delete")
        
        assert config["name"] == "random_delete"
        assert config["lib"] == "nlpaug"
        assert config["stage"] == "char"
        assert "defaults" in config
    
    def test_get_augmenter_stage(self, registry):
        """Test getting augmenter stage."""
        assert registry.get_augmenter_stage("random_delete") == "char"
        assert registry.get_augmenter_stage("word_dropout") == "word"
        assert registry.get_augmenter_stage("mlm_infill_bert") == "contextual"
    
    def test_get_default_params(self, registry):
        """Test getting default parameters."""
        params = registry.get_default_params("random_delete")
        assert "aug_char_p" in params
    
    def test_get_param_space(self, registry):
        """Test getting parameter space."""
        space = registry.get_param_space("random_delete")
        assert "aug_char_p" in space
    
    def test_stage_distribution(self, registry):
        """Test stage distribution."""
        dist = registry.get_stage_distribution()
        
        assert dist["char"] == 6
        assert dist["word"] == 5
        assert dist["contextual"] == 4
        assert dist["backtranslation"] == 5
        assert dist["format"] == 8
    
    def test_library_distribution(self, registry):
        """Test library distribution."""
        dist = registry.get_library_distribution()
        
        assert dist["nlpaug"] == 23
        assert dist["textattack"] == 5
    
    def test_invalid_augmenter(self, registry):
        """Test error handling for invalid augmenter."""
        with pytest.raises(ValueError):
            registry.get_augmenter_config("nonexistent")

    @pytest.mark.parametrize(
        "name",
        [
            "whitespace_jitter",
            "casing_jitter",
            "remove_punctuation",
            "add_typos",
        ],
    )
    def test_textattack_augmenters_return_strings(self, name, registry):
        """Ensure textattack augmenters instantiate and return strings."""
        pytest.importorskip("textattack")
        augmenter = registry.instantiate_augmenter(name, seed=123)
        output = augmenter.augment("Sample text!", n=1)
        assert isinstance(output, list)
        assert isinstance(output[0], str)
        assert output[0]

    def test_punctuation_noise_behaviour(self, registry):
        """Punctuation noise should alter text when probability is high."""
        augmenter = registry.instantiate_augmenter(
            "punctuation_noise",
            params={"aug_p": 1.0},
            seed=7,
        )
        original = "Hello world"
        augmented = augmenter.augment(original, n=1)[0]
        assert augmented != original
        assert any(ch in string.punctuation for ch in augmented)

    def test_unicode_normalization(self, registry):
        """Unicode normalization should decompose ligatures."""
        augmenter = registry.instantiate_augmenter("normalize_unicode")
        augmented = augmenter.augment("ﬁancé", n=1)[0]
        assert augmented == "fiancé"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
