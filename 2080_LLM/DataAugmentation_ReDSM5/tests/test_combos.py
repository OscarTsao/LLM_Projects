"""
Tests for combination generation.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aug.combos import ComboGenerator
from aug.registry import AugmenterRegistry


class TestComboGenerator:
    """Test suite for ComboGenerator."""
    
    @pytest.fixture
    def generator(self):
        """Create generator instance."""
        return ComboGenerator()
    
    @pytest.fixture
    def registry(self):
        """Create registry instance."""
        return AugmenterRegistry()
    
    def test_exclusion_detection(self, generator):
        """Test that exclusions are detected."""
        # Back-translation augmenters should be mutually exclusive
        combo1 = ["en_de_en", "en_fr_en"]
        assert not generator.is_valid_combo(combo1)
        
        # MLM models should be mutually exclusive
        combo2 = ["mlm_infill_bert", "mlm_infill_roberta"]
        assert not generator.is_valid_combo(combo2)
    
    def test_stage_diversity(self, generator):
        """Test stage diversity requirement."""
        # For k >= 2, should require at least 2 different stages
        combo = ["random_delete", "random_insert"]  # Both char stage
        
        if generator.min_stage_diversity >= 2:
            assert not generator.is_valid_combo(combo)
    
    def test_valid_combo(self, generator):
        """Test that valid combos are accepted."""
        combo = ["random_delete", "word_dropout"]  # Different stages
        assert generator.is_valid_combo(combo)
    
    def test_generate_k1_combos(self, generator):
        """Test generating k=1 combinations."""
        combos = generator.generate_k_combos(k=1, ordered=False)
        
        # Should have exactly 28 single augmenters
        assert len(combos) == 28
    
    def test_generate_k2_combos(self, generator):
        """Test generating k=2 combinations."""
        combos = generator.generate_k_combos(k=2, ordered=True)
        
        # Should respect safety cap
        max_pairs = generator.safety_caps.get("max_pairs", float("inf"))
        assert len(combos) <= max_pairs
    
    def test_generate_all_combos(self, generator):
        """Test generating all combinations."""
        all_combos = generator.generate_all_combos(ordered=True, verbose=False)
        
        # Should have combinations for k=1, 2, 3
        assert 1 in all_combos
        assert 2 in all_combos
        assert 3 in all_combos
        
        # Check total count
        total = sum(len(combos) for combos in all_combos.values())
        assert total > 0
    
    def test_combo_statistics(self, generator):
        """Test computing statistics."""
        all_combos = generator.generate_all_combos(ordered=True, verbose=False)
        stats = generator.get_combo_statistics(all_combos)
        
        assert "total_combos" in stats
        assert "by_k" in stats
        assert stats["total_combos"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
