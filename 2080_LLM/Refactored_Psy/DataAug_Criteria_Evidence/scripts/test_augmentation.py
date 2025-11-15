#!/usr/bin/env python3
"""Test augmentation determinism and verify train-only constraint.

This script verifies that:
1. Augmentation produces deterministic results with same seed
2. Augmentation ONLY applies to training data (NEVER val/test)
3. Augmentation counts match expected values
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psy_agents_aug.augment import (
    AugmentationConfig,
    NLPAugPipeline,
    TextAttackPipeline,
    HybridPipeline,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_determinism(augmentor, text: str, num_runs: int = 3):
    """Test that augmentation is deterministic with same seed."""
    logger.info("Testing augmentation determinism...")
    
    results = []
    for run in range(num_runs):
        augmented = augmentor.augment_text(text, num_variants=2)
        results.append(augmented)
        logger.info(f"Run {run + 1}: {augmented}")
    
    # Check if all runs produce same results
    for i in range(1, num_runs):
        if results[i] != results[0]:
            logger.error(f"FAILED: Run {i+1} differs from Run 1!")
            logger.error(f"  Run 1: {results[0]}")
            logger.error(f"  Run {i+1}: {results[i]}")
            return False
    
    logger.info("PASSED: Augmentation is deterministic")
    return True


def test_train_only_constraint(augmentor):
    """Test that augmentation only applies to training data."""
    logger.info("Testing train-only constraint...")
    
    texts = ["This is a test sentence.", "Another test sentence."]
    
    # Test on training split
    train_aug, _ = augmentor.augment_batch(texts, split="train")
    logger.info(f"Train split: {len(texts)} -> {len(train_aug)} samples")
    
    # Test on validation split
    val_aug, _ = augmentor.augment_batch(texts, split="val")
    logger.info(f"Val split: {len(texts)} -> {len(val_aug)} samples")
    
    # Test on test split
    test_aug, _ = augmentor.augment_batch(texts, split="test")
    logger.info(f"Test split: {len(texts)} -> {len(test_aug)} samples")
    
    # Verify
    if len(train_aug) <= len(texts):
        logger.error(f"FAILED: Train augmentation did not increase sample count!")
        return False
    
    if len(val_aug) != len(texts):
        logger.error(f"FAILED: Val split was augmented (should be unchanged)!")
        return False
    
    if len(test_aug) != len(texts):
        logger.error(f"FAILED: Test split was augmented (should be unchanged)!")
        return False
    
    logger.info("PASSED: Augmentation only applied to training data")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Test augmentation determinism and constraints"
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["nlpaug", "textattack", "hybrid"],
        default="nlpaug",
        help="Augmentation pipeline to test",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    
    args = parser.parse_args()
    
    # Create augmentation config
    config = AugmentationConfig(
        enabled=True,
        ratio=0.5,
        max_aug_per_sample=1,
        seed=args.seed,
        train_only=True,
    )
    
    # Create augmentor
    logger.info(f"Creating {args.pipeline} augmentor...")
    if args.pipeline == "nlpaug":
        augmentor = NLPAugPipeline(config, aug_method="synonym")
    elif args.pipeline == "textattack":
        augmentor = TextAttackPipeline(config, aug_method="wordnet")
    elif args.pipeline == "hybrid":
        augmentor = HybridPipeline(config)
    
    # Test text
    test_text = (
        "The patient reports feeling depressed and has difficulty sleeping. "
        "They have lost interest in activities they previously enjoyed."
    )
    
    # Run tests
    logger.info("=" * 60)
    logger.info("Starting augmentation tests")
    logger.info("=" * 60)
    
    success = True
    
    # Test 1: Determinism
    if not test_determinism(augmentor, test_text):
        success = False
    
    logger.info("")
    
    # Test 2: Train-only constraint
    if not test_train_only_constraint(augmentor):
        success = False
    
    logger.info("=" * 60)
    if success:
        logger.info("ALL TESTS PASSED")
    else:
        logger.error("SOME TESTS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
