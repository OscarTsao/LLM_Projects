"""Dataset builders for training and evaluation splits.

Key ideas
---------
1) Two tokenisation modes are supported:
   - eager: tokenise once during dataset construction (fewer CPU cycles)
   - lazy:  defer tokenisation to the collate_fn (needed for on‑the‑fly
            augmentation before tokenisation)

2) When augmentation is enabled, we pass raw strings through the pipeline in
   the collate_fn, then call the tokenizer for the whole batch to leverage
   fast tokenisers and vectorised padding/truncation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    import pandas as pd
    from transformers import PreTrainedTokenizerBase

    from psy_agents_noaug.augmentation import AugmenterPipeline


@dataclass
class DatasetSplits:
    """Container for train/val/test datasets."""

    train: Dataset
    val: Dataset
    test: Dataset


class ClassificationDataset(Dataset):
    """Tokenized dataset for sequence classification tasks."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        text_column: str,
        text_pair_column: str | None = None,
        label_column: str = "label",
        label_dtype: str = "int",
        augmenter: AugmenterPipeline | None = None,
        lazy_encode: bool = False,
    ) -> None:
        if dataframe.empty:
            raise ValueError("Received empty dataframe for dataset construction")

        texts = dataframe[text_column].astype(str).tolist()
        text_pairs = (
            dataframe[text_pair_column].astype(str).tolist()
            if text_pair_column
            else None
        )

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.text_pair_column = text_pair_column
        self.lazy_encode = bool(lazy_encode)
        self.augmenter = augmenter
        self._texts = texts
        self._text_pairs = text_pairs

        if self.lazy_encode:
            # Defer tokenisation to collate_fn; store raw strings only
            self.input_ids = None
            self.attention_mask = None
        else:
            # Eager path: pre-tokenise texts at construction time
            # Format: [CLS] criterion [SEP] post [SEP] (criterion first for better attention)
            encoded = tokenizer(
                (
                    text_pairs if text_pairs else texts
                ),  # Criterion first (or single text if no pair)
                text_pair=texts if text_pairs else None,  # Post second
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            self.input_ids = encoded["input_ids"]
            self.attention_mask = encoded["attention_mask"]

        if label_dtype == "float":
            labels = torch.tensor(
                dataframe[label_column].astype(float).values, dtype=torch.float32
            )
        else:
            labels = torch.tensor(
                dataframe[label_column].astype(int).values, dtype=torch.long
            )

        self.labels = labels

        if "criterion_index" in dataframe.columns:
            self.criterion_indices = torch.tensor(
                dataframe["criterion_index"].astype(int).values, dtype=torch.long
            )
        else:
            self.criterion_indices = None

        self.post_ids = (
            dataframe["post_id"].astype(str).tolist()
            if "post_id" in dataframe.columns
            else None
        )
        self.criterion_ids = (
            dataframe["criterion_id"].astype(str).tolist()
            if "criterion_id" in dataframe.columns
            else None
        )

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if self.lazy_encode:
            # Return raw texts for collate_fn to augment and tokenise
            item: dict[str, object] = {
                "text": self._texts[idx],
                "text_pair": self._text_pairs[idx] if self._text_pairs else None,
                "labels": self.labels[idx],
            }
        else:
            # Return ready‑to‑use tensors for the model
            # In eager mode, input_ids and attention_mask are pre-computed
            assert self.input_ids is not None, "input_ids should be set in eager mode"
            assert self.attention_mask is not None, "attention_mask should be set in eager mode"
            item = {
                "input_ids": self.input_ids[idx],
                "attention_mask": self.attention_mask[idx],
                "labels": self.labels[idx],
            }

        if self.criterion_indices is not None:
            item["criterion_index"] = self.criterion_indices[idx]

        if self.post_ids is not None:
            item["post_id"] = self.post_ids[idx]

        if self.criterion_ids is not None:
            item["criterion_id"] = self.criterion_ids[idx]

        return item  # type: ignore[return-value]


def create_classification_collate(
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    *,
    has_text_pair: bool,
    augmenter: AugmenterPipeline | None = None,
):
    """Create collate_fn that handles optional augmentation before tokenization."""

    def _collate(batch: list[dict[str, object]]) -> dict[str, torch.Tensor]:
        """Batch builder with optional augmentation and on‑the‑fly tokenisation.

        Workflow when not pre‑tokenised:
          1) optionally augment each raw sample (text and optional pair)
          2) call the HF tokenizer once for the whole batch for efficiency
        """
        if not batch:
            raise ValueError("Received empty batch in collate function")

        # If samples are already tokenized, fall back to default behaviour.
        first = batch[0]
        if "input_ids" in first and "attention_mask" in first:
            return default_collate(batch)  # type: ignore[return-value]

        texts: list[str] = []
        pairs: list[str] | None = [] if has_text_pair else None
        labels: list[torch.Tensor] = []

        criterion_indices: list[torch.Tensor] = []
        post_ids: list[str] = []
        criterion_ids: list[str] = []

        for sample in batch:
            text = str(sample["text"])
            if augmenter is not None:
                text = augmenter(text)
            texts.append(text)

            if has_text_pair and pairs is not None:
                pair_val = sample.get("text_pair")
                pairs.append("" if pair_val is None else str(pair_val))

            labels.append(sample["labels"])

            if "criterion_index" in sample:
                criterion_indices.append(sample["criterion_index"])
            if "post_id" in sample:
                post_ids.append(str(sample["post_id"]))
            if "criterion_id" in sample:
                criterion_ids.append(str(sample["criterion_id"]))

        encoding_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": max_length,
            "return_tensors": "pt",
        }

        # Format: [CLS] criterion [SEP] post [SEP] (criterion first for better attention)
        if has_text_pair and pairs:
            encoded = tokenizer(
                pairs, text_pair=texts, **encoding_kwargs
            )  # Swapped: pairs (criterion) first
        else:
            encoded = tokenizer(texts, **encoding_kwargs)

        batch_dict: dict[str, torch.Tensor | list[str]] = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.stack(labels),
        }

        if criterion_indices:
            batch_dict["criterion_index"] = torch.stack(criterion_indices)
        if post_ids:
            batch_dict["post_id"] = post_ids
        if criterion_ids:
            batch_dict["criterion_id"] = criterion_ids

        return batch_dict  # type: ignore[return-value]

    return _collate


def load_splits(splits_path: Path) -> dict[str, list[str]]:
    """Load train/val/test splits from JSON file."""
    with open(splits_path, encoding="utf-8") as f:
        data = json.load(f)

    return {
        "train": data.get("train", []),
        "val": data.get("val", data.get("validation", [])),
        "test": data.get("test", []),
    }


def build_criterion_mappings(
    dsm_entries: Sequence[Mapping[str, str]],
) -> tuple[dict[str, str], dict[str, int], dict[int, str]]:
    """Construct mapping dictionaries for DSM criteria."""
    criterion_text_map: dict[str, str] = {}
    for entry in dsm_entries:
        identifier = str(entry["id"])
        criterion_text_map[identifier] = entry.get("text", "")

    sorted_ids = sorted(criterion_text_map.keys())
    criterion_to_index = {cid: idx for idx, cid in enumerate(sorted_ids)}
    index_to_criterion = {idx: cid for cid, idx in criterion_to_index.items()}

    return criterion_text_map, criterion_to_index, index_to_criterion


def rename_post_columns(
    posts: pd.DataFrame, field_map: Mapping[str, str]
) -> pd.DataFrame:
    """Rename post dataframe columns to canonical names."""
    renamed = posts.rename(
        columns={
            field_map["post_id"]: "post_id",
            field_map["text"]: "post_text",
        }
    )
    return renamed[["post_id", "post_text"]].drop_duplicates()


def prepare_criteria_dataframe(
    groundtruth: pd.DataFrame,
    posts: Mapping[str, str],
    criterion_text: Mapping[str, str],
    criterion_to_index: Mapping[str, int],
    split_post_ids: Iterable[str],
) -> pd.DataFrame:
    """Create dataframe for criteria classification."""
    df = groundtruth[groundtruth["post_id"].isin(split_post_ids)].copy()
    if df.empty:
        raise ValueError("Criteria groundtruth yielded no rows for selected split")

    df["post_text"] = df["post_id"].map(posts)
    df["criterion_text"] = df["criterion_id"].map(criterion_text)
    df["criterion_index"] = df["criterion_id"].map(criterion_to_index)
    df["source"] = "original"

    df = df.dropna(subset=["post_text", "criterion_text", "criterion_index"])
    return df.reset_index(drop=True)


def prepare_evidence_dataframe(
    criteria_groundtruth: pd.DataFrame,
    evidence_groundtruth: pd.DataFrame,
    posts: Mapping[str, str],
    criterion_text: Mapping[str, str],
    criterion_to_index: Mapping[str, int],
    split_post_ids: Iterable[str],
) -> pd.DataFrame:
    """Create dataframe for evidence presence classification."""
    base_pairs = (
        criteria_groundtruth[["post_id", "criterion_id"]].drop_duplicates().copy()
    )

    positive_pairs = (
        evidence_groundtruth[["post_id", "criterion_id"]]
        .drop_duplicates()
        .assign(label=1)
    )

    aggregated_evidence = {}
    if "evidence_text" in evidence_groundtruth.columns:
        grouped = evidence_groundtruth.dropna(subset=["evidence_text"]).groupby(
            ["post_id", "criterion_id"]
        )["evidence_text"]
        aggregated_evidence = grouped.apply(
            lambda texts: " ".join(str(t) for t in texts if isinstance(t, str))
        ).to_dict()

    df = base_pairs.merge(positive_pairs, on=["post_id", "criterion_id"], how="left")
    df["label"] = df["label"].fillna(0).astype(int)
    df = df[df["post_id"].isin(split_post_ids)].copy()

    df["post_text"] = df["post_id"].map(posts)
    df["criterion_text"] = df["criterion_id"].map(criterion_text)
    df["criterion_index"] = df["criterion_id"].map(criterion_to_index)
    df["evidence_text"] = df.apply(
        lambda row: aggregated_evidence.get((row["post_id"], row["criterion_id"]), ""),
        axis=1,
    )
    df["input_text"] = df["evidence_text"]
    empty_mask = df["input_text"].str.strip() == ""
    df.loc[empty_mask, "input_text"] = df.loc[empty_mask, "post_text"]
    df["source"] = "original"

    df = df.dropna(subset=["post_text", "criterion_text", "criterion_index"])
    return df.reset_index(drop=True)


def build_datasets(
    task_name: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    field_map: Mapping[str, Mapping[str, str]],
    posts_df: pd.DataFrame,
    dsm_entries: Sequence[Mapping[str, str]],
    criteria_groundtruth: pd.DataFrame,
    evidence_groundtruth: pd.DataFrame,
    splits: Mapping[str, Sequence[str]],
    evidence_augmenter: AugmenterPipeline | None = None,
) -> tuple[DatasetSplits, dict[str, int], dict[int, str]]:
    """Build tokenized datasets for the requested task."""
    posts = rename_post_columns(posts_df, field_map["posts"])
    post_text_map = posts.set_index("post_id")["post_text"].to_dict()

    criterion_text_map, criterion_to_index, index_to_criterion = (
        build_criterion_mappings(dsm_entries)
    )

    if task_name == "criteria":
        train_df = prepare_criteria_dataframe(
            criteria_groundtruth,
            post_text_map,
            criterion_text_map,
            criterion_to_index,
            splits["train"],
        )
        val_df = prepare_criteria_dataframe(
            criteria_groundtruth,
            post_text_map,
            criterion_text_map,
            criterion_to_index,
            splits["val"],
        )
        test_df = prepare_criteria_dataframe(
            criteria_groundtruth,
            post_text_map,
            criterion_text_map,
            criterion_to_index,
            splits["test"],
        )

        # Format: [CLS] criterion [SEP] post [SEP] (criterion first for better attention)
        datasets = DatasetSplits(
            train=ClassificationDataset(
                train_df,
                tokenizer=tokenizer,
                max_length=max_length,
                text_column="criterion_text",  # First sequence
                text_pair_column="post_text",  # Second sequence
                label_column="label",
                label_dtype="int",
            ),
            val=ClassificationDataset(
                val_df,
                tokenizer=tokenizer,
                max_length=max_length,
                text_column="criterion_text",  # First sequence
                text_pair_column="post_text",  # Second sequence
                label_column="label",
                label_dtype="int",
            ),
            test=ClassificationDataset(
                test_df,
                tokenizer=tokenizer,
                max_length=max_length,
                text_column="criterion_text",  # First sequence
                text_pair_column="post_text",  # Second sequence
                label_column="label",
                label_dtype="int",
            ),
        )
    elif task_name == "evidence":
        train_df = prepare_evidence_dataframe(
            criteria_groundtruth,
            evidence_groundtruth,
            post_text_map,
            criterion_text_map,
            criterion_to_index,
            splits["train"],
        )
        val_df = prepare_evidence_dataframe(
            criteria_groundtruth,
            evidence_groundtruth,
            post_text_map,
            criterion_text_map,
            criterion_to_index,
            splits["val"],
        )
        test_df = prepare_evidence_dataframe(
            criteria_groundtruth,
            evidence_groundtruth,
            post_text_map,
            criterion_text_map,
            criterion_to_index,
            splits["test"],
        )

        datasets = DatasetSplits(
            train=ClassificationDataset(
                train_df,
                tokenizer=tokenizer,
                max_length=max_length,
                text_column="input_text",
                text_pair_column="criterion_text",
                label_column="label",
                label_dtype="int",
                augmenter=evidence_augmenter,
                lazy_encode=evidence_augmenter is not None,
            ),
            val=ClassificationDataset(
                val_df,
                tokenizer=tokenizer,
                max_length=max_length,
                text_column="input_text",
                text_pair_column="criterion_text",
                label_column="label",
                label_dtype="int",
            ),
            test=ClassificationDataset(
                test_df,
                tokenizer=tokenizer,
                max_length=max_length,
                text_column="input_text",
                text_pair_column="criterion_text",
                label_column="label",
                label_dtype="int",
            ),
        )
    else:
        raise ValueError(f"Unsupported task: {task_name}")

    return datasets, criterion_to_index, index_to_criterion
