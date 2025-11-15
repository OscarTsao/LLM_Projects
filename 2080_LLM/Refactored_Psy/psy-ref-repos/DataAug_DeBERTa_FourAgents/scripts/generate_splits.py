#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from src.utils.data import load_dataset
from src.utils.io import persist_splits, try_generate_groupkfold_splits


def simple_split(num_items: int, seed: int = 42):
    indices = list(range(num_items))
    random.Random(seed).shuffle(indices)
    if num_items <= 2:
        train = indices[:1]
        rest = indices[1:]
    else:
        pivot = int(max(1, num_items * 0.6))
        train = indices[:pivot]
        rest = indices[pivot:]
    half = len(rest) // 2
    dev = rest[:half] or train
    test = rest[half:] or train
    return train, dev, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic GroupKFold splits")
    parser.add_argument("--data", type=Path, default=Path("data/redsm5_sample.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("configs/data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    dataset = list(load_dataset(args.data))
    posts = [item["post_id"] for item in dataset]
    groups = posts

    try:
        train_idx, dev_idx, test_idx = try_generate_groupkfold_splits(posts, groups, n_splits=5, seed=args.seed)
    except Exception:
        train_idx, dev_idx, test_idx = simple_split(len(posts), seed=args.seed)

    train_ids = [posts[i] for i in train_idx]
    dev_ids = [posts[i] for i in dev_idx]
    test_ids = [posts[i] for i in test_idx]

    persist_splits(train_ids, dev_ids, test_ids, str(args.out_dir))
    print(f"Splits written to {args.out_dir}")
if __name__ == "__main__":
    main()
