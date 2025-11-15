"""Dataset assembly utilities for the criteria baseline rebuild."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import json
import os
import warnings

import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch.utils.data import Dataset

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

# Minimal DSM-5 criteria descriptions (same as original project)
CRITERIA: Dict[str, str] = {
    "DEPRESSED_MOOD": "Persistent depressed mood most of the day nearly every day.",
    "ANHEDONIA": "Markedly diminished interest or pleasure in nearly all activities.",
    "APPETITE_CHANGE": "Significant weight loss or gain, or decrease or increase in appetite.",
    "SLEEP_ISSUES": "Insomnia or hypersomnia nearly every day.",
    "PSYCHOMOTOR": "Observable psychomotor agitation or retardation.",
    "FATIGUE": "Fatigue or loss of energy nearly every day.",
    "WORTHLESSNESS": "Feelings of worthlessness or excessive guilt.",
    "COGNITIVE_ISSUES": "Diminished ability to think, concentrate, or indecisiveness.",
    "SUICIDAL_THOUGHTS": "Recurrent thoughts of death or suicidal ideation.",
}


def _load_ground_truth_rows(path: Path) -> Iterable[Dict[str, object]]:
    """Flatten the nested ground truth JSON."""
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError("Ground truth file must contain a list of examples")

    for entry in data:
        post_id = str(entry["post_id"])
        post_text = str(entry["post"])
        for criterion, payload in entry.get("criteria", {}).items():
            if criterion not in CRITERIA:
                continue
            label = int(payload.get("groundtruth", 0))
            yield {
                "post_id": post_id,
                "criterion": criterion,
                "post_text": post_text,
                "label": label,
            }


def load_ground_truth_frame(path: Path) -> pd.DataFrame:
    """Return ground truth as a flat dataframe."""
    rows = list(_load_ground_truth_rows(path))
    return pd.DataFrame(rows)


def _load_augmented_records(path: Path, source: str) -> pd.DataFrame:
    """Load augmentation CSV and mark metadata."""
    df = pd.read_csv(path)
    if "augmented_post" not in df.columns:
        raise ValueError(f"{path.name} missing 'augmented_post' column")

    out = pd.DataFrame(
        {
            "post_id": df["post_id"].astype(str),
            "criterion": df["criterion"].astype(str),
            "text_a": df["criterion"].map(CRITERIA).astype(str),
            "text_b": df["augmented_post"].fillna(df.get("post_text", "")).astype(str),
            "label": 1,
            "source": source,
            "is_augmented": True,
        }
    )
    return out


def assemble_dataset(cfg: Mapping[str, object]) -> pd.DataFrame:
    """Combine ground truth pairs with selected augmentation sources."""
    ground_truth_path = Path(str(cfg["ground_truth_path"])).resolve()
    include_original = bool(cfg.get("include_original", True))
    use_augmented: List[str] = list(cfg.get("use_augmented", []))
    augmentation_dir = Path(str(cfg.get("augmentation_dir", ground_truth_path.parent))).resolve()
    augmentation_patterns: Mapping[str, str] = cfg.get("augmentation_patterns", {})

    frames: List[pd.DataFrame] = []

    if include_original:
        base = load_ground_truth_frame(ground_truth_path)
        base["text_a"] = base["criterion"].map(CRITERIA).astype(str)
        base["text_b"] = base["post_text"].astype(str)
        base["source"] = "original"
        base["is_augmented"] = False
        frames.append(
            base[["post_id", "criterion", "text_a", "text_b", "label", "source", "is_augmented"]]
        )

    for key in use_augmented:
        pattern = augmentation_patterns.get(key)
        if not pattern:
            raise ValueError(f"No augmentation pattern configured for '{key}'")
        candidates = sorted(augmentation_dir.glob(pattern))
        if not candidates:
            raise FileNotFoundError(f"Could not find files for pattern '{pattern}' in {augmentation_dir}")
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        frames.append(_load_augmented_records(latest, source=key))

    if not frames:
        raise ValueError("No dataset sources were assembled.")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=["text_a", "text_b"])
    combined["label"] = combined["label"].astype(int)
    return combined.reset_index(drop=True)


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_dataset(df: pd.DataFrame, cfg: Mapping[str, object]) -> DatasetSplit:
    """Perform group-aware splits so post_ids stay in one split."""
    splits = cfg.get("splits", {})
    train_ratio = float(splits.get("train", 0.7))
    val_ratio = float(splits.get("val", 0.15))
    test_ratio = float(splits.get("test", 0.15))
    seed = int(cfg.get("shuffle_seed", 42))
    train_aug_only = bool(cfg.get("augmented_to_train_only", False))

    if train_ratio <= 0 or train_ratio > 1:
        raise ValueError("Train split ratio must be within (0, 1].")
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("Validation and test ratios must be non-negative.")

    base_df = df
    aug_df = df.iloc[0:0]
    if train_aug_only:
        aug_df = df[df["is_augmented"] == True]  # noqa: E712
        base_df = df[df["is_augmented"] == False]  # noqa: E712
        if base_df.empty:
            raise ValueError("augmented_to_train_only is set but no original examples are available.")

    if base_df.empty:
        raise ValueError("No data available to split. Ensure dataset assembly produced examples.")

    groups = base_df["post_id"].to_numpy()
    labels = base_df["label"].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio + test_ratio, random_state=seed)
    train_idx, rest_idx = next(gss.split(base_df, labels, groups))
    train_df = base_df.iloc[train_idx]
    rest_df = base_df.iloc[rest_idx]

    remaining = val_ratio + test_ratio
    if remaining == 0:
        final_train = train_df
        if train_aug_only and not aug_df.empty:
            final_train = pd.concat([final_train, aug_df], ignore_index=True)
        return DatasetSplit(
            train=final_train.reset_index(drop=True),
            val=pd.DataFrame(columns=df.columns),
            test=pd.DataFrame(columns=df.columns),
        )

    test_fraction = test_ratio / remaining if remaining > 0 else 0.0
    gss_rest = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed + 1)
    val_idx, test_idx = next(
        gss_rest.split(rest_df, rest_df["label"].to_numpy(), rest_df["post_id"].to_numpy())
    )
    val_df = rest_df.iloc[val_idx]
    test_df = rest_df.iloc[test_idx]

    if train_aug_only and not aug_df.empty:
        train_df = pd.concat([train_df, aug_df], ignore_index=True)
        train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return DatasetSplit(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )


def create_cross_validation_splits(df: pd.DataFrame, cfg: Mapping[str, object]) -> List[DatasetSplit]:
    """Generate stratified group k-fold splits with optional augmentation injection."""
    cv_cfg = cfg.get("cross_validation", {})
    num_folds = int(cv_cfg.get("num_folds", 5))
    if num_folds < 2:
        raise ValueError("cross_validation.num_folds must be at least 2.")
    seed = int(cv_cfg.get("shuffle_seed", cfg.get("shuffle_seed", 42)))
    train_aug_only = bool(cfg.get("augmented_to_train_only", False))

    base_df = df
    aug_df = df.iloc[0:0]
    if train_aug_only:
        aug_df = df[df["is_augmented"] == True]  # noqa: E712
        base_df = df[df["is_augmented"] == False]  # noqa: E712
        if base_df.empty:
            raise ValueError("cross-validation requires original (non-augmented) examples for folds.")

    if base_df.empty:
        raise ValueError("No data available to split for cross-validation.")

    groups = base_df["post_id"].to_numpy()
    labels = base_df["label"].to_numpy()

    splitter = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    folds: List[DatasetSplit] = []

    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(base_df, labels, groups), start=1):
        train_df = base_df.iloc[train_idx]
        val_df = base_df.iloc[val_idx]

        if train_aug_only and not aug_df.empty:
            train_df = pd.concat([train_df, aug_df], ignore_index=True)
            train_df = train_df.sample(frac=1.0, random_state=seed + fold_index).reset_index(drop=True)
        else:
            train_df = train_df.reset_index(drop=True)

        val_df = val_df.reset_index(drop=True)
        test_df = val_df.copy(deep=True)

        folds.append(
            DatasetSplit(
                train=train_df,
                val=val_df,
                test=test_df,
            )
        )

    return folds


class CriteriaDataset(Dataset):
    """Torch dataset that tokenizes criterion-post pairs on the fly."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
        max_length: int,
        cache_in_memory: bool = False,
        cache_max_ram_fraction: float = 0.9,
    ):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_in_memory = cache_in_memory
        self.cache_max_ram_fraction = max(0.0, min(cache_max_ram_fraction, 1.0))
        self._cached_samples: Dict[int, Dict[str, torch.Tensor]] = {}
        self._cache_stats = {"cached_samples": 0, "cached_bytes": 0, "limit_bytes": None, "fully_cached": False}
        if self.cache_in_memory and not self.frame.empty:
            self._warm_cache()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx in self._cached_samples:
            cached = self._cached_samples[idx]
            return {k: v for k, v in cached.items()}
        row = self.frame.iloc[idx]
        return self._encode_row(row)

    def _encode_row(self, row: pd.Series) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            row["text_a"],
            row["text_b"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(row["label"], dtype=torch.float32)
        return item

    def _warm_cache(self) -> None:
        limit_bytes = _resolve_memory_limit_bytes(self.cache_max_ram_fraction)
        self._cache_stats["limit_bytes"] = limit_bytes
        if limit_bytes is None or limit_bytes <= 0:
            warnings.warn(
                "Skipping in-memory dataset cache because system memory limits could not be determined.",
                RuntimeWarning,
            )
            return

        cached: Dict[int, Dict[str, torch.Tensor]] = {}
        consumed = 0
        sharing_enabled = True
        for idx, (_, row) in enumerate(self.frame.iterrows()):
            sample = self._encode_row(row)
            sample_bytes = _tensor_dict_num_bytes(sample)
            if sample_bytes <= 0:
                continue
            if consumed + sample_bytes > limit_bytes:
                warnings.warn(
                    "Stopped caching dataset after reaching the configured RAM budget "
                    f"({consumed / (1024 ** 2):.1f} MiB used of {limit_bytes / (1024 ** 2):.1f} MiB limit).",
                    RuntimeWarning,
                )
                break

            if sharing_enabled:
                try:
                    for value in sample.values():
                        if isinstance(value, torch.Tensor):
                            value.share_memory_()
                except (RuntimeError, OSError) as share_err:
                    warnings.warn(
                        "Disabling in-memory dataset cache after shared-memory setup failure; "
                        f"falling back to on-the-fly tokenization. Details: {share_err}",
                        RuntimeWarning,
                    )
                    self._cached_samples = {}
                    return
            cached[idx] = sample
            consumed += sample_bytes

        if cached:
            self._cached_samples = cached
            self._cache_stats.update(
                {
                    "cached_samples": len(cached),
                    "cached_bytes": consumed,
                    "fully_cached": len(cached) == len(self.frame),
                }
            )


def _resolve_memory_limit_bytes(fraction: float) -> int | None:
    fraction = max(0.0, min(fraction, 1.0))
    total_bytes: int | None = None
    available_bytes: int | None = None

    if psutil is not None:
        stats = psutil.virtual_memory()
        total_bytes = int(stats.total)
        available_bytes = int(stats.available)
    else:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
            if isinstance(page_size, int):
                if isinstance(phys_pages, int):
                    total_bytes = int(page_size * phys_pages)
                if isinstance(avail_pages, int):
                    available_bytes = int(page_size * avail_pages)
        except (AttributeError, ValueError, OSError):
            pass

    reference = total_bytes if total_bytes is not None else available_bytes
    if reference is None:
        return None

    limit = int(reference * fraction)
    if available_bytes is not None:
        limit = min(limit, int(available_bytes * 0.95))
    return max(limit, 0)


def _tensor_dict_num_bytes(sample: Mapping[str, torch.Tensor]) -> int:
    total = 0
    for value in sample.values():
        if isinstance(value, torch.Tensor):
            total += int(value.element_size() * value.nelement())
    return total
