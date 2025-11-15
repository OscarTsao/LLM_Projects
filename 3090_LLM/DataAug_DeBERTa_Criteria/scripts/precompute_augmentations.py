#!/usr/bin/env python3
"""Pre-compute all augmentations and save to persistent cache.

This script:
1. Loads the training dataset
2. Finds all unique texts
3. Applies all 25 augmentation methods to each text
4. Saves to a persistent cache file

Run once (takes ~6 hours), then all HPO trials use cached augmentations instantly.
"""

import json
import logging
import pickle
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataaug_multi_both.data.augmentation import ALL_METHODS, AugmentationConfig, TextAugmenter
from dataaug_multi_both.data.dataset import load_hf_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def precompute_augmentations(
    output_file: str = "experiments/augmentation_cache.pkl",
    dataset_id: str = "redsm5",
):
    """Pre-compute augmentations for all texts and methods.

    Args:
        output_file: Path to save the cache
        dataset_id: Dataset identifier
    """
    logger.info("Loading dataset...")
    dataset, metadata = load_hf_dataset(dataset_id)
    train_data = dataset["train"]

    # Find all unique texts
    unique_texts = {}
    for idx, item in enumerate(train_data):
        text = item.get("post_text", "")
        if text and text not in unique_texts:
            unique_texts[text] = idx

    logger.info(f"Found {len(unique_texts)} unique texts to augment")
    logger.info(f"Will generate {len(unique_texts)} × {len(ALL_METHODS)} = {len(unique_texts) * len(ALL_METHODS)} cached augmentations")

    # Initialize cache
    cache = {}
    stats = {"success": 0, "failed": 0, "failed_methods": {}}

    # Pre-compute augmentations
    for text_idx, (text, original_idx) in enumerate(unique_texts.items()):
        if text_idx % 50 == 0:
            logger.info(
                f"Progress: {text_idx}/{len(unique_texts)} texts "
                f"({100*text_idx/len(unique_texts):.1f}%) - "
                f"Success: {stats['success']}, Failed: {stats['failed']}"
            )

        for method in ALL_METHODS:
            try:
                # Create augmenter for this specific method
                config = AugmentationConfig(
                    methods=[method], probability=1.0, num_augmentations=1, preserve_non_evidence=False
                )
                augmenter = TextAugmenter(config)

                # Apply augmentation
                if augmenter.augmenters:
                    augmented_text, metadata = augmenter.augment_evidence(text)
                    if metadata.get("augmentation_applied", False):
                        cache[(original_idx, method)] = augmented_text
                        stats["success"] += 1
                    else:
                        # Augmentation didn't apply, use original
                        cache[(original_idx, method)] = text
                        stats["failed"] += 1
                        stats["failed_methods"][method] = stats["failed_methods"].get(method, 0) + 1
                else:
                    # No augmenter initialized, use original
                    cache[(original_idx, method)] = text
                    stats["failed"] += 1
                    stats["failed_methods"][method] = stats["failed_methods"].get(method, 0) + 1

            except Exception as e:
                logger.warning(f"Failed to augment text {original_idx} with {method}: {e}")
                cache[(original_idx, method)] = text  # Fallback to original
                stats["failed"] += 1
                stats["failed_methods"][method] = stats["failed_methods"].get(method, 0) + 1

    # Save cache
    logger.info("Saving cache to disk...")
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save metadata
    meta = {
        "num_texts": len(unique_texts),
        "num_methods": len(ALL_METHODS),
        "total_cached": len(cache),
        "stats": stats,
        "methods": ALL_METHODS,
        "dataset_id": dataset_id,
    }

    meta_path = output_path.with_suffix(".json")
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"✓ Cache saved to {output_path}")
    logger.info(f"✓ Metadata saved to {meta_path}")
    logger.info(f"✓ Total cached entries: {len(cache)}")
    logger.info(f"✓ Success: {stats['success']}, Failed: {stats['failed']}")
    logger.info(f"✓ Cache size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    if stats["failed_methods"]:
        logger.info("\nMethods with failures:")
        for method, count in sorted(stats["failed_methods"].items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {method}: {count} failures")


if __name__ == "__main__":
    precompute_augmentations()
