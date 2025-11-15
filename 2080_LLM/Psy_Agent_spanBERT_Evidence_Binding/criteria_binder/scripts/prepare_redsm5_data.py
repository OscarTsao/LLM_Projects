#!/usr/bin/env python3
# File: scripts/prepare_redsm5_data.py
"""Script to prepare RedSM5 data for training."""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.redsm5_loader import RedSM5DataLoader


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main function to prepare RedSM5 data."""
    parser = argparse.ArgumentParser(description="Prepare RedSM5 data for training")

    parser.add_argument(
        "--dsm-criteria",
        type=str,
        default="Data/DSM-5/DSM_Criteria_Array_Fixed_Major_Depressive.json",
        help="Path to DSM-5 criteria JSON file"
    )
    parser.add_argument(
        "--posts",
        type=str,
        default="Data/redsm5/redsm5_posts.csv",
        help="Path to posts CSV file"
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="Data/redsm5/redsm5_annotations.csv",
        help="Path to annotations CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/redsm5",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Training set ratio"
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.15,
        help="Development set ratio"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test set ratio"
    )
    parser.add_argument(
        "--split-by-post",
        action="store_true",
        default=True,
        help="Split by post ID to avoid data leakage"
    )
    parser.add_argument(
        "--include-negatives",
        action="store_true",
        default=True,
        help="Include negative examples (status=0)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting RedSM5 data preparation")

    try:
        # Initialize data loader
        loader = RedSM5DataLoader(
            dsm_criteria_path=args.dsm_criteria,
            posts_path=args.posts,
            annotations_path=args.annotations
        )

        # Print statistics
        logger.info(f"Unique posts: {len(loader.get_unique_posts())}")
        logger.info(f"Unique symptoms: {len(loader.get_unique_symptoms())}")
        logger.info(f"Positive annotations: {len(loader.get_positive_annotations())}")
        logger.info(f"Negative annotations: {len(loader.get_negative_annotations())}")

        # Split data
        split_data = loader.split_data(
            train_ratio=args.train_ratio,
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio,
            split_by_post=args.split_by_post,
            random_seed=args.random_seed
        )

        # Print split statistics
        for split_name, examples in split_data.items():
            positive_count = sum(1 for ex in examples if ex['label'] == 1)
            negative_count = sum(1 for ex in examples if ex['label'] == 0)
            logger.info(f"{split_name}: {len(examples)} total, {positive_count} positive, {negative_count} negative")

        # Save to JSONL files
        loader.save_split_to_jsonl(split_data, args.output_dir)

        logger.info("Data preparation completed successfully")

    except Exception as e:
        logger.error(f"Error during data preparation: {e}")
        raise


if __name__ == "__main__":
    main()