from __future__ import annotations

import json
from pathlib import Path

from src.training.dataset_builder import assemble_dataset, build_splits


def _write_ground_truth(path: Path) -> None:
    rows = []
    for idx in range(6):
        post_id = f"post_{idx // 2}"
        criteria = {
            "DEPRESSED_MOOD": {"groundtruth": int(idx % 2 == 0)},
            "SLEEP_ISSUES": {"groundtruth": int(idx % 3 == 0)},
        }
        rows.append({"post_id": post_id, "post": f"Post text {idx}", "criteria": criteria})
    path.write_text(json.dumps(rows))


def test_assemble_dataset(tmp_path: Path) -> None:
    ground_truth = tmp_path / "ground_truth.json"
    _write_ground_truth(ground_truth)
    config = {
        "ground_truth_path": str(ground_truth),
        "include_original": True,
        "use_augmented": [],
        "augmented_sources": {},
        "splits": {"train": 0.7, "val": 0.15, "test": 0.15},
        "shuffle_seed": 42,
    }
    df = assemble_dataset(config)
    assert {"post_id", "criterion", "text_a", "text_b", "label", "source"}.issubset(df.columns)
    assert set(df["source"].unique()) == {"original"}
    assert (df["text_a"].str.startswith("Post text")).all()


def test_group_split_respects_posts(tmp_path: Path) -> None:
    ground_truth = tmp_path / "ground_truth.json"
    _write_ground_truth(ground_truth)
    config = {
        "ground_truth_path": str(ground_truth),
        "include_original": True,
        "use_augmented": [],
        "augmented_sources": {},
        "splits": {"train": 0.6, "val": 0.2, "test": 0.2},
        "shuffle_seed": 13,
    }
    splits = build_splits(config)
    train_posts = set(splits.train["post_id"])
    val_posts = set(splits.val["post_id"])
    test_posts = set(splits.test["post_id"])
    assert train_posts.isdisjoint(val_posts)
    assert train_posts.isdisjoint(test_posts)
    assert val_posts.isdisjoint(test_posts)
