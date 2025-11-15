#!/usr/bin/env python3
"""
Prepare REDSM5 dataset by loading from source and saving as Parquet files.

Usage:
    python scripts/prepare_redsm5.py [--config CONFIG] [--source SOURCE]
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataio.loader import REDSM5Loader
from utils.logging import setup_logger
import yaml
import json


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prepare REDSM5 dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/run.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="local",
        choices=["local", "hub", "csv", "parquet"],
        help="Data source (local, hub, csv, or parquet)",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="redsm5",
        help="Dataset name (for hub source)",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.7,
        help="Training set proportion (default: 0.7)",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Validation set proportion (default: 0.15)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Test set proportion (default: 0.15)",
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Setup logger
    logger = setup_logger("prepare_redsm5", level="INFO")

    logger.info("=" * 80)
    logger.info("REDSM5 Dataset Preparation")
    logger.info("=" * 80)

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_config = config["dataset"]
    io_config = config["io"]

    # Initialize loader
    logger.info("Initializing REDSM5 loader...")
    loader = REDSM5Loader(
        base_path=dataset_config["base_path"],
        text_field=dataset_config["text_field"],
        label_fields=dataset_config["label_fields"],
        random_seed=config.get("global", {}).get("seed", 42),
    )

    logger.info(f"Data source: {args.source}")
    logger.info(f"Output directory: {dataset_config['base_path']}")

    # Load data based on source
    try:
        if args.source == "local":
            # Load from local CSV/JSON files and create splits
            logger.info("\n" + "=" * 80)
            logger.info("Loading data from local files...")
            logger.info("=" * 80)

            data = loader.prepare_base_dataset(
                train_size=args.train_size,
                val_size=args.val_size,
                test_size=args.test_size,
            )

        elif args.source == "hub":
            logger.info(f"\nLoading from Hugging Face Hub: {args.dataset_name}")
            dataset_dict = loader.load_from_hub(
                dataset_name=args.dataset_name,
                splits=dataset_config["splits"],
            )

            # Convert to DataFrames
            data = {
                split: ds.to_pandas()
                for split, ds in dataset_dict.items()
            }

        elif args.source == "csv":
            logger.info("\nLoading from CSV files...")
            data = loader.load_from_csv(splits=dataset_config["splits"])

        elif args.source == "parquet":
            logger.info("\nLoading from Parquet files...")
            data = loader.load_from_parquet(splits=dataset_config["splits"])

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Please ensure all source files are in the correct location.")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Data validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error while loading data: {e}")
        raise

    # Validate data
    logger.info("\n" + "=" * 80)
    logger.info("Validating data...")
    logger.info("=" * 80)
    for split, df in data.items():
        try:
            loader.validate_data(df)
            logger.info(f"  {split}: {len(df)} examples - VALID")
        except ValueError as e:
            logger.error(f"  {split}: VALIDATION FAILED - {e}")
            sys.exit(1)

    # Compute and display statistics
    logger.info("\n" + "=" * 80)
    logger.info("Dataset Statistics")
    logger.info("=" * 80)
    stats = loader.get_statistics(data)

    for split, split_stats in stats.items():
        logger.info(f"\n{split.upper()}:")
        logger.info(f"  Number of examples: {split_stats['num_examples']}")
        logger.info(f"  Text field: {split_stats['text_field']}")
        logger.info(f"  Avg text length: {split_stats['avg_text_length']:.1f} chars")
        logger.info(f"  Min text length: {split_stats['min_text_length']} chars")
        logger.info(f"  Max text length: {split_stats['max_text_length']} chars")

        # Display label distributions
        for label_field in loader.label_fields:
            dist_key = f"{label_field}_distribution"
            if dist_key in split_stats:
                logger.info(f"\n  {label_field} distribution:")
                for label, count in split_stats[dist_key].items():
                    pct = count / split_stats['num_examples'] * 100
                    logger.info(f"    {label}: {count} ({pct:.1f}%)")

    # Save to Parquet
    logger.info("\n" + "=" * 80)
    logger.info("Saving to Parquet format...")
    logger.info("=" * 80)
    logger.info(f"Output directory: {dataset_config['base_path']}")
    logger.info(f"Compression: {io_config['parquet']['compression']} (level {io_config['parquet']['compression_level']})")

    try:
        loader.save_to_parquet(
            data,
            compression=io_config["parquet"]["compression"],
            compression_level=io_config["parquet"]["compression_level"],
        )
    except Exception as e:
        logger.error(f"Error saving Parquet files: {e}")
        raise

    # Save metadata
    metadata = {
        "source": args.source,
        "splits": list(data.keys()),
        "split_sizes": {split: len(df) for split, df in data.items()},
        "text_field": dataset_config["text_field"],
        "label_fields": dataset_config["label_fields"],
        "statistics": stats,
        "config": config,
    }

    metadata_path = Path(dataset_config["base_path"]) / "metadata.json"
    logger.info(f"\nSaving metadata to {metadata_path}")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("\n" + "=" * 80)
    logger.info("Dataset preparation completed successfully!")
    logger.info("=" * 80)
    logger.info(f"\nOutput files:")
    for split in data.keys():
        parquet_file = Path(dataset_config["base_path"]) / f"{split}.parquet"
        if parquet_file.exists():
            size_mb = parquet_file.stat().st_size / (1024 * 1024)
            logger.info(f"  {parquet_file} ({size_mb:.2f} MB)")
    logger.info(f"  {metadata_path}")


if __name__ == "__main__":
    main()
