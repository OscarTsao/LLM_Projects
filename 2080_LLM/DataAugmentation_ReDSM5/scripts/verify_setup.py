#!/usr/bin/env python3
"""
Verify that the project setup is correct.

Usage:
    python scripts/verify_setup.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from dataio.loader import REDSM5Loader
        from dataio.parquet_io import ParquetIO
        print("  ✓ dataio module")
    except Exception as e:
        print(f"  ✗ dataio module: {e}")
        return False
    
    try:
        from aug.registry import AugmenterRegistry
        from aug.compose import AugmentationPipeline
        from aug.combos import ComboGenerator
        from aug.seeds import SeedManager
        print("  ✓ aug module")
    except Exception as e:
        print(f"  ✗ aug module: {e}")
        return False
    
    try:
        from hpo.search import HPOSearch
        from hpo.trainer import AugmentationTrainer
        print("  ✓ hpo module")
    except Exception as e:
        print(f"  ✗ hpo module: {e}")
        return False
    
    try:
        from utils.hashing import compute_combo_hash, compute_text_hash
        from utils.logging import setup_logger
        from utils.stats import compute_dataset_statistics
        from utils.estimate import estimate_cache_size
        print("  ✓ utils module")
    except Exception as e:
        print(f"  ✗ utils module: {e}")
        return False
    
    return True


def test_config_files():
    """Test that configuration files exist and are valid."""
    print("\nTesting configuration files...")
    
    import yaml
    
    # Test run.yaml
    run_config_path = Path("configs/run.yaml")
    if not run_config_path.exists():
        print(f"  ✗ {run_config_path} not found")
        return False
    
    try:
        with open(run_config_path) as f:
            run_config = yaml.safe_load(f)
        
        required_keys = ["dataset", "combinations", "io", "hpo", "global"]
        for key in required_keys:
            if key not in run_config:
                print(f"  ✗ run.yaml missing key: {key}")
                return False
        
        print(f"  ✓ {run_config_path}")
    except Exception as e:
        print(f"  ✗ Error loading {run_config_path}: {e}")
        return False
    
    # Test augmenters_28.yaml
    aug_config_path = Path("configs/augmenters_28.yaml")
    if not aug_config_path.exists():
        print(f"  ✗ {aug_config_path} not found")
        return False
    
    try:
        with open(aug_config_path) as f:
            aug_config = yaml.safe_load(f)
        
        if "augmenters" not in aug_config:
            print("  ✗ augmenters_28.yaml missing 'augmenters' key")
            return False
        
        num_augmenters = len(aug_config["augmenters"])
        if num_augmenters != 28:
            print(f"  ✗ Expected 28 augmenters, found {num_augmenters}")
            return False
        
        print(f"  ✓ {aug_config_path} (28 augmenters)")
    except Exception as e:
        print(f"  ✗ Error loading {aug_config_path}: {e}")
        return False
    
    return True


def test_augmenter_registry():
    """Test that augmenter registry works."""
    print("\nTesting augmenter registry...")
    
    try:
        from aug.registry import AugmenterRegistry
        
        registry = AugmenterRegistry(config_path="configs/augmenters_28.yaml")
        
        # Check total count
        all_augs = registry.list_augmenters()
        if len(all_augs) != 28:
            print(f"  ✗ Expected 28 augmenters, got {len(all_augs)}")
            return False
        
        # Check stage distribution
        dist = registry.get_stage_distribution()
        expected = {"char": 6, "word": 5, "contextual": 4, "backtranslation": 5, "format": 8}
        
        for stage, expected_count in expected.items():
            if dist.get(stage, 0) != expected_count:
                print(f"  ✗ Stage '{stage}': expected {expected_count}, got {dist.get(stage, 0)}")
                return False
        
        print("  ✓ AugmenterRegistry (28 augmenters, correct distribution)")
        
    except Exception as e:
        print(f"  ✗ Error testing registry: {e}")
        return False
    
    return True


def test_combo_generator():
    """Test that combo generator works."""
    print("\nTesting combo generator...")
    
    try:
        from aug.combos import ComboGenerator
        
        generator = ComboGenerator(config_path="configs/run.yaml")
        
        # Test k=1 generation
        combos_k1 = generator.generate_k_combos(k=1, ordered=False)
        if len(combos_k1) != 28:
            print(f"  ✗ Expected 28 k=1 combos, got {len(combos_k1)}")
            return False
        
        # Test exclusion
        invalid_combo = ["en_de_en", "en_fr_en"]  # Back-translations are mutually exclusive
        if generator.is_valid_combo(invalid_combo):
            print("  ✗ Exclusion constraint not working")
            return False
        
        print(f"  ✓ ComboGenerator (28 k=1 combos, exclusions working)")
        
    except Exception as e:
        print(f"  ✗ Error testing combo generator: {e}")
        return False
    
    return True


def test_seed_manager():
    """Test deterministic seeding."""
    print("\nTesting seed manager...")
    
    try:
        from aug.seeds import SeedManager
        
        sm = SeedManager(global_seed=13)
        
        # Test consistency
        seed1 = sm.get_augmenter_seed(0)
        seed2 = sm.get_augmenter_seed(0)
        
        if seed1 != seed2:
            print("  ✗ Seeds not consistent")
            return False
        
        # Test variation
        seed3 = sm.get_augmenter_seed(1)
        if seed1 == seed3:
            print("  ✗ Seeds not varying")
            return False
        
        print("  ✓ SeedManager (deterministic and varying)")
        
    except Exception as e:
        print(f"  ✗ Error testing seed manager: {e}")
        return False
    
    return True


def test_hashing():
    """Test hashing utilities."""
    print("\nTesting hashing utilities...")
    
    try:
        from utils.hashing import compute_combo_hash, compute_text_hash
        
        # Test combo hash consistency
        combo = ["random_delete", "word_dropout"]
        params = {"random_delete": {"aug_char_p": 0.1}}
        seed = 13
        
        hash1 = compute_combo_hash(combo, params, seed)
        hash2 = compute_combo_hash(combo, params, seed)
        
        if hash1 != hash2:
            print("  ✗ Combo hash not consistent")
            return False
        
        # Test text hash
        text = "This is a test."
        text_hash1 = compute_text_hash(text)
        text_hash2 = compute_text_hash(text)
        
        if text_hash1 != text_hash2:
            print("  ✗ Text hash not consistent")
            return False
        
        print("  ✓ Hashing (deterministic)")
        
    except Exception as e:
        print(f"  ✗ Error testing hashing: {e}")
        return False
    
    return True


def main():
    """Main verification function."""
    print("=" * 60)
    print("REDSM5 Augmentation Pipeline Setup Verification")
    print("=" * 60)
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_config_files()
    all_passed &= test_augmenter_registry()
    all_passed &= test_combo_generator()
    all_passed &= test_seed_manager()
    all_passed &= test_hashing()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Setup is correct.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
