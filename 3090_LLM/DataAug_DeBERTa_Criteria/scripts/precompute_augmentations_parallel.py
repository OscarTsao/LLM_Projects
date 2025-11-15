#!/usr/bin/env python3
"""Parallelized pre-computation of all augmentations with persistent cache.

Optimizations:
- Multiprocessing: Process multiple texts in parallel
- Progress tracking: Real-time progress with tqdm
- Checkpointing: Save intermediate results every 100 texts
- Resource-aware: Uses 75% of CPU cores
"""

import json
import logging
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path

# Force CPU-only for augmentation (avoids CUDA multiprocessing issues)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataaug_multi_both.data.augmentation import ALL_METHODS, AugmentationConfig, TextAugmenter
from dataaug_multi_both.data.dataset import load_hf_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def augment_single_text(args):
    """Augment a single text with a single method (worker function).

    Args:
        args: Tuple of (text_idx, text, method)

    Returns:
        Tuple of (text_idx, method, augmented_text, success)
    """
    text_idx, text, method = args

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
                return (text_idx, method, augmented_text, True)

        # Fallback to original
        return (text_idx, method, text, False)

    except Exception as e:
        # Fallback to original on error
        return (text_idx, method, text, False)


def precompute_augmentations_parallel(
    output_file: str = "experiments/augmentation_cache.pkl",
    dataset_id: str = "redsm5",
    num_workers: int = None,
    checkpoint_every: int = 100,
):
    """Pre-compute augmentations using parallel processing.

    Args:
        output_file: Path to save the cache
        dataset_id: Dataset identifier
        num_workers: Number of parallel workers (None = use 75% of CPUs)
        checkpoint_every: Save checkpoint every N texts
    """
    # Determine number of workers
    if num_workers is None:
        num_workers = max(1, int(mp.cpu_count() * 0.75))

    logger.info(f"Using {num_workers} parallel workers")

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

    # Prepare all work items
    work_items = []
    for text, text_idx in unique_texts.items():
        for method in ALL_METHODS:
            work_items.append((text_idx, text, method))

    logger.info(f"Created {len(work_items)} work items")

    # Initialize cache and stats
    cache = {}
    stats = {"success": 0, "failed": 0, "failed_methods": {}}

    # Process in parallel
    logger.info("Starting parallel processing...")
    completed = 0

    with mp.Pool(processes=num_workers) as pool:
        # Process work items
        for result in pool.imap_unordered(augment_single_text, work_items, chunksize=10):
            text_idx, method, augmented_text, success = result

            # Store in cache
            cache[(text_idx, method)] = augmented_text

            # Update stats
            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1
                stats["failed_methods"][method] = stats["failed_methods"].get(method, 0) + 1

            completed += 1

            # Progress reporting
            if completed % 100 == 0:
                progress_pct = 100 * completed / len(work_items)
                logger.info(
                    f"Progress: {completed}/{len(work_items)} ({progress_pct:.1f}%) - "
                    f"Success: {stats['success']}, Failed: {stats['failed']}"
                )

            # Checkpoint
            if completed % (checkpoint_every * len(ALL_METHODS)) == 0:
                logger.info(f"Saving checkpoint at {completed} items...")
                checkpoint_path = Path(output_file).with_suffix(".checkpoint.pkl")
                with checkpoint_path.open("wb") as f:
                    pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Save final cache
    logger.info("Saving final cache to disk...")
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
        "num_workers": num_workers,
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
        for method, count in sorted(stats["failed_methods"].items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {method}: {count} failures")


if __name__ == "__main__":
    # Use spawn mode for multiprocessing (safer for ML libraries)
    mp.set_start_method('spawn', force=True)
    precompute_augmentations_parallel()
