#!/usr/bin/env python3
"""
List and save all valid augmentation combinations.

Usage:
    python scripts/list_combos.py [--config CONFIG] [--k-max K_MAX] [--output OUTPUT]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from aug.combos import ComboGenerator
from utils.logging import setup_logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="List valid augmentation combinations"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/run.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--k-max",
        type=int,
        default=None,
        help="Maximum combination length (overrides config)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/redsm5/combos/valid_combos.json",
        help="Output JSON file",
    )
    parser.add_argument(
        "--ordered",
        action="store_true",
        help="Treat combinations as ordered (permutations)",
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger("list_combos", level="INFO")
    
    # Initialize generator
    logger.info("Initializing combo generator...")
    generator = ComboGenerator(config_path=args.config)
    
    # Override k_max if specified
    if args.k_max:
        generator.k_max = args.k_max
    
    logger.info(f"Generating combinations up to k={generator.k_max}...")
    logger.info(f"Ordered: {args.ordered}")
    logger.info(f"Exclusions: {len(generator.exclusions)}")
    logger.info(f"Min stage diversity: {generator.min_stage_diversity}")
    
    # Generate all combinations
    all_combos = generator.generate_all_combos(
        ordered=args.ordered,
        verbose=True,
    )
    
    # Compute statistics
    logger.info("\nComputing statistics...")
    stats = generator.get_combo_statistics(all_combos)
    
    logger.info(f"\nTotal combinations: {stats['total_combos']}")
    logger.info("By k:")
    for k, count in stats['by_k'].items():
        logger.info(f"  k={k}: {count}")
    
    logger.info("\nTop augmenters by frequency:")
    for aug, count in list(stats['augmenter_frequency'].items())[:10]:
        logger.info(f"  {aug}: {count}")
    
    # Save to file
    logger.info(f"\nSaving to {args.output}...")
    generator.save_combos(all_combos, args.output)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
