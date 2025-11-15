"""
Tests for deterministic augmentation.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aug.seeds import SeedManager
from utils.hashing import compute_combo_hash, compute_text_hash


class TestDeterminism:
    """Test suite for deterministic behavior."""
    
    @pytest.fixture
    def seed_manager(self):
        """Create seed manager."""
        return SeedManager(global_seed=13)
    
    def test_seed_consistency(self, seed_manager):
        """Test that same position gives same seed."""
        seed1 = seed_manager.get_augmenter_seed(0)
        seed2 = seed_manager.get_augmenter_seed(0)
        
        assert seed1 == seed2
    
    def test_seed_variation(self, seed_manager):
        """Test that different positions give different seeds."""
        seed1 = seed_manager.get_augmenter_seed(0)
        seed2 = seed_manager.get_augmenter_seed(1)
        
        assert seed1 != seed2
    
    def test_example_seed_consistency(self, seed_manager):
        """Test example seed consistency."""
        seed1 = seed_manager.get_example_seed(0, 0)
        seed2 = seed_manager.get_example_seed(0, 0)
        
        assert seed1 == seed2
    
    def test_example_seed_variation(self, seed_manager):
        """Test example seed variation."""
        # Different examples
        seed1 = seed_manager.get_example_seed(0, 0)
        seed2 = seed_manager.get_example_seed(1, 0)
        assert seed1 != seed2
        
        # Different positions
        seed1 = seed_manager.get_example_seed(0, 0)
        seed2 = seed_manager.get_example_seed(0, 1)
        assert seed1 != seed2
    
    def test_combo_hash_consistency(self):
        """Test that combo hash is deterministic."""
        combo = ["random_delete", "word_dropout"]
        params = {"random_delete": {"aug_char_p": 0.1}}
        seed = 13
        
        hash1 = compute_combo_hash(combo, params, seed)
        hash2 = compute_combo_hash(combo, params, seed)
        
        assert hash1 == hash2
    
    def test_combo_hash_sensitivity(self):
        """Test that combo hash changes with inputs."""
        combo1 = ["random_delete", "word_dropout"]
        combo2 = ["word_dropout", "random_delete"]  # Different order
        params = {}
        seed = 13
        
        hash1 = compute_combo_hash(combo1, params, seed)
        hash2 = compute_combo_hash(combo2, params, seed)
        
        assert hash1 != hash2
    
    def test_text_hash_consistency(self):
        """Test text hash consistency."""
        text = "This is a test sentence."
        
        hash1 = compute_text_hash(text)
        hash2 = compute_text_hash(text)
        
        assert hash1 == hash2
    
    def test_text_hash_variation(self):
        """Test that different texts have different hashes."""
        text1 = "This is a test sentence."
        text2 = "This is another test sentence."
        
        hash1 = compute_text_hash(text1)
        hash2 = compute_text_hash(text2)
        
        assert hash1 != hash2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
