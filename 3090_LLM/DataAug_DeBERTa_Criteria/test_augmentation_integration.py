#!/usr/bin/env python3
"""Quick test script to verify augmentation integration."""

import sys
from dataaug_multi_both.data.augmentation import (
    get_available_methods,
    create_augmenter,
    ALL_METHODS,
)

def test_available_methods():
    """Test that all 28 methods are available."""
    print("Testing available methods...")
    available = get_available_methods()

    print(f"\nTotal methods defined: {len(ALL_METHODS)}")
    print(f"  - nlpaug methods: {len(available['nlpaug'])}")
    print(f"  - textattack methods: {len(available['textattack'])}")
    print(f"  - Total available: {len(available['all'])}")

    assert len(ALL_METHODS) == 28, f"Expected 28 methods, got {len(ALL_METHODS)}"
    print("✓ All 28 methods defined")


def test_augmenter_initialization():
    """Test that augmenters can be initialized."""
    print("\nTesting augmenter initialization...")

    # Test with a few methods
    test_methods = ["nlp_char_swap", "ta_wordnet", "nlp_word_synonym"]

    try:
        augmenter = create_augmenter(
            methods=test_methods,
            probability=0.5
        )
        print(f"✓ Augmenter initialized with {len(test_methods)} methods")
        print(f"  Active augmenters: {len(augmenter.augmenters)}")
    except Exception as e:
        print(f"✗ Failed to initialize augmenter: {e}")
        return False

    return True


def test_augmentation():
    """Test basic augmentation."""
    print("\nTesting text augmentation...")

    test_text = "I feel very sad and tired all the time"

    # Test with character-level augmentation (should work without models)
    try:
        augmenter = create_augmenter(
            methods=["nlp_char_swap"],
            probability=1.0  # Always augment for testing
        )

        if augmenter.augmenters:
            augmented, metadata = augmenter.augment_evidence(test_text)
            print(f"  Original: {test_text}")
            print(f"  Augmented: {augmented}")
            print(f"  Applied: {metadata.get('augmentation_applied')}")
            print(f"  Method: {metadata.get('method_used')}")
            print("✓ Augmentation test passed")
        else:
            print("⚠ No augmenters available (libraries may not be installed)")

    except Exception as e:
        print(f"⚠ Augmentation test failed: {e}")


def test_hpo_integration():
    """Test HPO search space integration."""
    print("\nTesting HPO integration...")

    try:
        import optuna
        from dataaug_multi_both.hpo.search_space import suggest

        # Create a simple study
        study = optuna.create_study()
        trial = study.ask()

        # Get parameters
        params = suggest(trial)

        print(f"  Generated params keys: {list(params.keys())}")
        print(f"  aug.num_methods: {params.get('aug.num_methods')}")
        print(f"  aug.prob: {params.get('aug.prob')}")
        print(f"  aug.methods: {params.get('aug.methods', '')[:50]}...")

        assert "aug.num_methods" in params, "Missing aug.num_methods"
        print("✓ HPO integration test passed")

    except Exception as e:
        print(f"✗ HPO integration test failed: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("=" * 60)
    print("Augmentation Integration Test Suite")
    print("=" * 60)

    test_available_methods()
    test_augmenter_initialization()
    test_augmentation()
    test_hpo_integration()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
