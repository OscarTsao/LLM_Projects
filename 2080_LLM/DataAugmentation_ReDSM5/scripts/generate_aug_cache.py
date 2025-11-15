#!/usr/bin/env python3
"""
Generate augmentation cache for all valid combinations.

Usage:
    python scripts/generate_aug_cache.py [--config CONFIG] [--combos COMBOS] [--num-workers N]
"""

import argparse
import sys
from pathlib import Path
import json
from multiprocessing import Pool
from functools import partial

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataio.loader import REDSM5Loader
from dataio.parquet_io import ParquetIO
from aug.compose import AugmentationPipeline
from aug.registry import AugmenterRegistry
from utils.logging import setup_logger
from utils.hashing import generate_cache_filename
import yaml


def augment_split(combo, split, df, config, logger):
    """
    Augment a single split for a given combo.
    
    Args:
        combo: List of augmenter names
        split: Split name
        df: DataFrame to augment
        config: Configuration dictionary
        logger: Logger instance
        
    Returns:
        Tuple of (combo_hash, split, success)
    """
    try:
        # Create pipeline
        pipeline = AugmentationPipeline(
            combo=combo,
            seed=config["global"]["seed"],
        )
        
        # Augment data
        logger.info(f"Augmenting {split} with {' -> '.join(combo)}...")
        df_aug = pipeline.augment_dataframe(
            df,
            text_field=config["dataset"]["text_field"],
            verbose=False,
        )
        
        # Get combo hash
        combo_hash = pipeline.get_combo_hash()
        
        # Save to cache
        cache_dir = Path(config["io"]["cache"]["combos_dir"])
        cache_file = cache_dir / generate_cache_filename(combo_hash, split)
        
        parquet_io = ParquetIO(
            compression=config["io"]["parquet"]["compression"],
            compression_level=config["io"]["parquet"]["compression_level"],
        )
        
        metadata = {
            "combo": combo,
            "combo_hash": combo_hash,
            "split": split,
            "seed": config["global"]["seed"],
        }
        
        parquet_io.write_dataframe(df_aug, cache_file, metadata=metadata)
        
        logger.info(f"Saved {cache_file}")
        
        return combo_hash, split, True
    
    except Exception as e:
        logger.error(f"Error augmenting {split} with {combo}: {e}")
        return None, split, False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate augmentation cache"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/run.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--combos",
        type=str,
        default="data/redsm5/combos/valid_combos.json",
        help="Path to valid combinations JSON",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Splits to augment (default: train val)",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger("generate_cache", level="INFO")
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load valid combinations
    logger.info(f"Loading combinations from {args.combos}...")
    with open(args.combos) as f:
        combos_by_k = json.load(f)
    
    # Flatten to single list
    all_combos = []
    for k, combo_list in combos_by_k.items():
        all_combos.extend(combo_list)
    
    logger.info(f"Found {len(all_combos)} combinations")
    
    # Load base data
    logger.info("Loading base data...")
    loader = REDSM5Loader(
        base_path=config["dataset"]["base_path"],
        text_field=config["dataset"]["text_field"],
        label_fields=config["dataset"]["label_fields"],
    )
    
    splits = args.splits or ["train", "val"]
    data = loader.load_from_parquet(splits=splits)
    
    for split, df in data.items():
        logger.info(f"  {split}: {len(df)} examples")
    
    # Create cache directory
    cache_dir = Path(config["io"]["cache"]["combos_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate cache for each combo and split
    logger.info(f"\nGenerating cache with {args.num_workers} workers...")
    
    total_tasks = len(all_combos) * len(splits)
    completed = 0
    
    for combo in all_combos:
        for split in splits:
            df = data[split]
            result = augment_split(combo, split, df, config, logger)
            completed += 1
            
            if completed % 10 == 0:
                logger.info(f"Progress: {completed}/{total_tasks} ({100*completed/total_tasks:.1f}%)")
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
