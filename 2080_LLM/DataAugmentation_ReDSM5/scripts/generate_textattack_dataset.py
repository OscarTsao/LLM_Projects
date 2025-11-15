#!/usr/bin/env python3
"""Generate TextAttack-only augmented dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

from src.augmentation.base import AugmentationConfig
from src.augmentation.textattack_pipeline import TextAttackAugmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TextAttack augmented dataset")
    parser.add_argument("--output-dir", type=Path, default=AugmentationConfig().output_dir)
    parser.add_argument("--posts", type=Path, default=AugmentationConfig().posts_path)
    parser.add_argument("--annotations", type=Path, default=AugmentationConfig().annotations_path)
    parser.add_argument("--seed", type=int, default=AugmentationConfig().random_seed)
    parser.add_argument("--variants", type=int, default=AugmentationConfig().variants_per_example)
    parser.add_argument("--exclude-original", action="store_true", help="Do not include original evidence sentence")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AugmentationConfig(
        output_dir=args.output_dir,
        posts_path=args.posts,
        annotations_path=args.annotations,
        random_seed=args.seed,
        variants_per_example=args.variants,
        include_original=not args.exclude_original,
    )
    augmenter = TextAttackAugmenter(config)
    output = augmenter.run()
    print(f"TextAttack dataset written to {output}")


if __name__ == "__main__":
    main()
