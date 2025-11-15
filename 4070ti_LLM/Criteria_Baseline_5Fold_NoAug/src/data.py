"""Dataset assembly utilities for the criteria baseline rebuild."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import json
import os
import warnings

import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
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
        doc_stride: int = 0,
    ):
        self.frame = frame.reset_index(drop=True).copy()
        self.frame["_group_id"] = self.frame["post_id"].astype(str) + "::" + self.frame["criterion"].astype(str)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_in_memory = cache_in_memory
        self.cache_max_ram_fraction = max(0.0, min(cache_max_ram_fraction, 1.0))
        self.doc_stride = max(0, int(doc_stride))
        self._cached_samples: Dict[int, Dict[str, torch.Tensor]] = {}
        self._cache_stats = {"cached_samples": 0, "cached_bytes": 0, "limit_bytes": None, "fully_cached": False}
        self._precomputed_samples: Optional[List[Dict[str, object]]] = None
        if self.doc_stride > 0:
            self._precompute_with_stride()
        elif self.cache_in_memory and not self.frame.empty:
            self._warm_cache()

    def __len__(self) -> int:
        if self._precomputed_samples is not None:
            return len(self._precomputed_samples)
        return len(self.frame)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        if self._precomputed_samples is not None:
            sample = self._precomputed_samples[idx]
            # Return a shallow copy so DataLoader collate does not mutate the stored sample.
            return {k: v for k, v in sample.items()}
        if idx in self._cached_samples:
            cached = self._cached_samples[idx]
            return {k: v for k, v in cached.items()}
        row = self.frame.iloc[idx]
        return self._encode_row(row)

    def _encode_row(self, row: pd.Series) -> Dict[str, object]:
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
        item["group_id"] = row["_group_id"]
        item["chunk_index"] = torch.tensor(0, dtype=torch.int64)
        return item

    def _precompute_with_stride(self) -> None:
        samples: List[Dict[str, object]] = []
        share_memory = self.cache_in_memory
        total_bytes = 0
        for _, row in self.frame.iterrows():
            batch = self.tokenizer(
                row["text_a"],
                row["text_b"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                stride=self.doc_stride,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )
            input_ids = batch["input_ids"]
            if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
            attention_mask = batch["attention_mask"]
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None and token_type_ids.dim() == 1:
                token_type_ids = token_type_ids.unsqueeze(0)

            num_chunks = input_ids.size(0)
            for chunk_idx in range(num_chunks):
                sample: Dict[str, object] = {
                    "input_ids": input_ids[chunk_idx].clone(),
                    "attention_mask": attention_mask[chunk_idx].clone(),
                    "labels": torch.tensor(row["label"], dtype=torch.float32),
                    "group_id": row["_group_id"],
                    "chunk_index": torch.tensor(chunk_idx, dtype=torch.int64),
                }
                if token_type_ids is not None:
                    sample["token_type_ids"] = token_type_ids[chunk_idx].clone()
                if share_memory:
                    for value in sample.values():
                        if isinstance(value, torch.Tensor):
                            value.share_memory_()
                samples.append(sample)
                total_bytes += _tensor_dict_num_bytes(sample)

        self._precomputed_samples = samples
        self._cache_stats.update(
            {
                "cached_samples": len(samples),
                "cached_bytes": total_bytes,
                "limit_bytes": None,
                "fully_cached": True,
            }
        )

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
