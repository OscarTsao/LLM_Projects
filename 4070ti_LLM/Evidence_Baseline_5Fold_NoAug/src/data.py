"""Dataset assembly utilities for the criteria baseline rebuild."""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset

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


def _load_ground_truth_rows_json(path: Path) -> Iterable[Dict[str, object]]:
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


def _load_redsm5_rows(directory: Path) -> Iterable[Dict[str, object]]:
    """Yield ground-truth rows from the original ReDSM5 CSV bundle."""
    posts_path = directory / "redsm5_posts.csv"
    annotations_path = directory / "redsm5_annotations.csv"

    if not posts_path.exists():
        raise FileNotFoundError(f"Missing posts file at {posts_path}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Missing annotations file at {annotations_path}")

    posts_df = pd.read_csv(posts_path)
    annotations_df = pd.read_csv(annotations_path)

    posts_df["post_id"] = posts_df["post_id"].astype(str)
    posts_df["text"] = posts_df["text"].fillna("").astype(str)

    annotations_df["post_id"] = annotations_df["post_id"].astype(str)
    annotations_df["DSM5_symptom"] = annotations_df["DSM5_symptom"].astype(str).str.strip().str.upper()
    annotations_df["DSM5_symptom"] = annotations_df["DSM5_symptom"].replace({"LEEP_ISSUES": "SLEEP_ISSUES"})
    annotations_df["status"] = annotations_df["status"].fillna(0).astype(int)

    valid_mask = annotations_df["DSM5_symptom"].isin(CRITERIA)
    dropped = annotations_df[~valid_mask]
    if not dropped.empty:
        extras = ", ".join(sorted(set(dropped["DSM5_symptom"])))
        warnings.warn(f"Dropping annotations with unsupported DSM-5 symptoms: {extras}")
    annotations_df = annotations_df[valid_mask]

    if annotations_df.empty:
        raise ValueError("No ReDSM5 annotations remain after filtering supported symptoms.")

    status_lookup = annotations_df.groupby(["post_id", "DSM5_symptom"])["status"].max().to_dict()

    annotated_posts = {pid for pid, _ in status_lookup}
    missing_posts = sorted(annotated_posts - set(posts_df["post_id"]))
    if missing_posts:
        warnings.warn(
            f"{len(missing_posts)} annotations reference posts missing from redsm5_posts.csv; they will be ignored."
        )

    post_map = dict(zip(posts_df["post_id"], posts_df["text"]))

    if not post_map:
        raise ValueError("redsm5_posts.csv did not contain any rows.")

    for post_id, post_text in post_map.items():
        for criterion in CRITERIA:
            label = int(status_lookup.get((post_id, criterion), 0) > 0)
            yield {
                "post_id": post_id,
                "criterion": criterion,
                "post_text": post_text,
                "label": label,
            }


def _resolve_ground_truth_format(path: Path, format_hint: str | None) -> str:
    """Determine which loader to use for the specified ground-truth path."""
    if format_hint:
        return str(format_hint).lower()
    if path.is_dir():
        return "redsm5"
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    raise ValueError(
        f"Unable to infer ground truth format for '{path}'. "
        "Specify dataset.ground_truth_format in the Hydra configuration."
    )


def load_ground_truth_frame(path: Path, format_hint: str | None = None) -> pd.DataFrame:
    """Return ground truth as a flat dataframe."""
    resolved_format = _resolve_ground_truth_format(path, format_hint)
    if resolved_format == "json":
        rows = list(_load_ground_truth_rows_json(path))
    elif resolved_format == "redsm5":
        rows = list(_load_redsm5_rows(path))
    else:
        raise ValueError(f"Unsupported ground truth format '{resolved_format}'.")
    return pd.DataFrame(rows)


def _load_augmented_records(path: Path, source: str) -> pd.DataFrame:
    """Load augmentation CSV and mark metadata."""
    df = pd.read_csv(path)
    if "augmented_post" not in df.columns:
        raise ValueError(f"{path.name} missing 'augmented_post' column")

    criteria_map = df["criterion"].map(CRITERIA)
    invalid_mask = criteria_map.isna()
    if invalid_mask.any():
        extras = ", ".join(sorted(set(df.loc[invalid_mask, "criterion"])))
        warnings.warn(f"Dropping augmentation rows with unsupported criteria: {extras}")
        df = df.loc[~invalid_mask].copy()
        criteria_map = criteria_map.loc[~invalid_mask]
    if df.empty:
        raise ValueError(f"No usable augmentation rows found in {path.name}")

    out = pd.DataFrame(
        {
            "post_id": df["post_id"].astype(str),
            "criterion": df["criterion"].astype(str),
            "text_a": criteria_map.astype(str),
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
    format_hint = cfg.get("ground_truth_format")
    include_original = bool(cfg.get("include_original", True))
    use_augmented: List[str] = list(cfg.get("use_augmented", []))
    augmentation_dir = Path(str(cfg.get("augmentation_dir", ground_truth_path.parent))).resolve()
    augmentation_patterns: Mapping[str, str] = cfg.get("augmentation_patterns", {})

    frames: List[pd.DataFrame] = []

    if include_original:
        base = load_ground_truth_frame(ground_truth_path, format_hint)
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


def compute_label_counts(frame: pd.DataFrame) -> Dict[int, int]:
    """Return the number of positives/negatives in a dataframe."""
    counts = frame["label"].value_counts()
    return {int(label): int(count) for label, count in counts.items()}


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
    """Torch dataset that tokenizes criterion-post pairs with optional sliding windows."""

    def __init__(
        self,
        frame: pd.DataFrame,
        tokenizer,
        max_length: int,
        cache_in_memory: bool = False,
        cache_max_ram_fraction: float = 0.9,
        doc_stride: int = 0,
    ):
        self.frame = frame.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max(1, int(max_length))
        self.doc_stride = max(0, int(doc_stride))
        self.cache_in_memory = bool(cache_in_memory)
        self.labels = torch.tensor(self.frame["label"].to_numpy(), dtype=torch.float32)
        self._sample_index: List[Tuple[int, int]] = []
        self._row_cache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._build_index()

    def __len__(self) -> int:
        return len(self._sample_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self._sample_index):
            raise IndexError("Dataset index out of range")
        row_idx, window_idx = self._sample_index[idx]
        encoding = self._tokenize_row(row_idx)
        sample: Dict[str, torch.Tensor] = {}
        for key, tensor in encoding.items():
            if key == "overflow_to_sample_mapping":
                continue
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.ndim == 0:
                continue
            value = tensor[window_idx]
            if not self.cache_in_memory:
                value = value.clone()
            sample[key] = value
        sample["labels"] = self.labels[row_idx].clone()
        return sample

    def _tokenize_row(self, row_idx: int) -> Dict[str, torch.Tensor]:
        cached = self._row_cache.get(row_idx)
        if cached is not None:
            return cached
        row = self.frame.iloc[row_idx]
        overflow = self.doc_stride > 0
        encoded = self.tokenizer(
            row["text_a"],
            row["text_b"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            stride=self.doc_stride if overflow else 0,
            return_overflowing_tokens=overflow,
            return_tensors="pt",
        )
        sanitized: Dict[str, torch.Tensor] = {}
        for key, value in encoded.items():
            if not isinstance(value, torch.Tensor):
                continue
            sanitized[key] = value.detach()
        if self.cache_in_memory:
            self._row_cache[row_idx] = sanitized
        return sanitized

    def _build_index(self) -> None:
        self._sample_index.clear()
        if self.frame.empty:
            return
        for row_idx in range(len(self.frame)):
            encoding = self._tokenize_row(row_idx)
            input_ids = encoding.get("input_ids")
            if input_ids is None:
                raise ValueError("Tokenizer did not return input_ids tensor.")
            window_count = int(input_ids.size(0))
            if window_count <= 0:
                continue
            for window_idx in range(window_count):
                self._sample_index.append((row_idx, window_idx))
