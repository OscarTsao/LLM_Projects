"""Dataset assembly utilities for training and evaluation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.criteria_descriptions import CRITERIA
from src.data.redsm5_loader import load_ground_truth_frame


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def _latest_match(pattern: str) -> Path | None:
    paths = sorted(Path().glob(pattern))
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _build_base_records(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["criterion_text"] = df["criterion"].map(CRITERIA)
    df["text_a"] = df["post_text"].astype(str)
    df["text_b"] = df["criterion_text"].astype(str)
    df["source"] = "original"
    df["is_augmented"] = False
    return df[["post_id", "criterion", "text_a", "text_b", "label", "source", "is_augmented"]]


def _build_augmented_records(df: pd.DataFrame, source: str) -> pd.DataFrame:
    df = df.copy()
    if "augmented_post" not in df.columns:
        raise ValueError(f"Augmented dataset {source} missing 'augmented_post' column")
    df["criterion_text"] = df["criterion"].map(CRITERIA)
    df["text_a"] = df["augmented_post"].fillna(df.get("post_text", "")).astype(str)
    df["text_b"] = df["criterion_text"].astype(str)
    df["label"] = 1
    df["source"] = source
    df["is_augmented"] = True
    columns = ["post_id", "criterion", "text_a", "text_b", "label", "source", "is_augmented"]
    return df[columns]


def assemble_dataset(config: Mapping[str, object]) -> pd.DataFrame:
    ground_truth_path = Path(str(config.get("ground_truth_path")))
    include_original = bool(config.get("include_original", True))
    use_augmented = list(config.get("use_augmented", []))
    augmented_sources = dict(config.get("augmented_sources", {}))

    frames: list[pd.DataFrame] = []
    if include_original:
        base_df = load_ground_truth_frame(ground_truth_path)
        frames.append(_build_base_records(base_df))

    for key in use_augmented:
        pattern = augmented_sources.get(key)
        if not pattern:
            raise ValueError(f"No augmented source pattern configured for key '{key}'")
        path = _latest_match(pattern)
        if path is None:
            raise FileNotFoundError(f"Could not find augmented dataset for pattern {pattern}")
        aug_df = pd.read_csv(path)
        frames.append(_build_augmented_records(aug_df, source=key))

    if not frames:
        raise ValueError("No data sources selected for dataset assembly")

    combined = pd.concat(frames, ignore_index=True)
    combined.dropna(subset=["text_a", "text_b"], inplace=True)
    combined["post_id"] = combined["post_id"].astype(str)
    combined["criterion"] = combined["criterion"].astype(str)
    combined["label"] = combined["label"].astype(int)
    return combined


def _group_split(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetSplit:
    total = train_ratio + val_ratio + test_ratio
    if not math.isclose(total, 1.0, rel_tol=1e-2):
        raise ValueError("train/val/test ratios must sum to 1.0")

    groups = df["post_id"].values
    labels = df["label"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio + test_ratio, random_state=seed)
    train_idx, rest_idx = next(gss.split(df, labels, groups))
    train_df = df.iloc[train_idx]
    rest_df = df.iloc[rest_idx]

    if rest_df.empty:
        raise ValueError("Insufficient samples for validation/test splits")

    rest_groups = rest_df["post_id"].values
    rest_labels = rest_df["label"].values
    # Avoid zero division when val_ratio or test_ratio is zero.
    remaining = val_ratio + test_ratio
    if remaining == 0.0:
        return DatasetSplit(train=train_df.reset_index(drop=True), val=pd.DataFrame(), test=pd.DataFrame())

    test_fraction = test_ratio / remaining
    gss_rest = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed + 1)
    val_idx, test_idx = next(gss_rest.split(rest_df, rest_labels, rest_groups))
    val_df = rest_df.iloc[val_idx]
    test_df = rest_df.iloc[test_idx]

    return DatasetSplit(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def build_splits(config: Mapping[str, object]) -> DatasetSplit:
    assembled = assemble_dataset(config)
    splits = config.get("splits", {})
    train_ratio = float(splits.get("train", 0.7))
    val_ratio = float(splits.get("val", 0.15))
    test_ratio = float(splits.get("test", 0.15))
    seed = int(config.get("shuffle_seed", 42))
    return _group_split(assembled, train_ratio, val_ratio, test_ratio, seed)

