"""Dataset builders for training and evaluation splits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


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
        text_pair_column: Optional[str] = None,
        label_column: str = "label",
        label_dtype: str = "int",
    ) -> None:
        if dataframe.empty:
            raise ValueError("Received empty dataframe for dataset construction")

        texts = dataframe[text_column].astype(str).tolist()
        text_pairs = (
            dataframe[text_pair_column].astype(str).tolist()
            if text_pair_column
            else None
        )

        encoded = tokenizer(
            texts,
            text_pair=text_pairs,
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, object] = {
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


def load_splits(splits_path: Path) -> Dict[str, List[str]]:
    """Load train/val/test splits from JSON file."""
    with open(splits_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return {
        "train": data.get("train", []),
        "val": data.get("val", data.get("validation", [])),
        "test": data.get("test", []),
    }


def build_criterion_mappings(
    dsm_entries: Sequence[Mapping[str, str]],
) -> Tuple[Dict[str, str], Dict[str, int], Dict[int, str]]:
    """Construct mapping dictionaries for DSM criteria."""
    criterion_text_map: Dict[str, str] = {}
    for entry in dsm_entries:
        identifier = str(entry["id"])
        criterion_text_map[identifier] = entry.get("text", "")

    sorted_ids = sorted(criterion_text_map.keys())
    criterion_to_index = {cid: idx for idx, cid in enumerate(sorted_ids)}
    index_to_criterion = {idx: cid for cid, idx in criterion_to_index.items()}

    return criterion_text_map, criterion_to_index, index_to_criterion


def rename_post_columns(posts: pd.DataFrame, field_map: Mapping[str, str]) -> pd.DataFrame:
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
        criteria_groundtruth[["post_id", "criterion_id"]]
        .drop_duplicates()
        .copy()
    )

    positive_pairs = (
        evidence_groundtruth[["post_id", "criterion_id"]]
        .drop_duplicates()
        .assign(label=1)
    )

    aggregated_evidence = {}
    if "evidence_text" in evidence_groundtruth.columns:
        grouped = (
            evidence_groundtruth.dropna(subset=["evidence_text"])
            .groupby(["post_id", "criterion_id"])["evidence_text"]
        )
        aggregated_evidence = grouped.apply(
            lambda texts: " ".join(str(t) for t in texts if isinstance(t, str))
        ).to_dict()

    df = base_pairs.merge(
        positive_pairs, on=["post_id", "criterion_id"], how="left"
    )
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
) -> Tuple[DatasetSplits, Dict[str, int], Dict[int, str]]:
    """Build tokenized datasets for the requested task."""
    posts = rename_post_columns(posts_df, field_map["posts"])
    post_text_map = posts.set_index("post_id")["post_text"].to_dict()

    criterion_text_map, criterion_to_index, index_to_criterion = (
        build_criterion_mappings(dsm_entries)
    )

    if task_name == "criteria":
        train_df = prepare_criteria_dataframe(
            criteria_groundtruth, post_text_map, criterion_text_map, criterion_to_index, splits["train"]
        )
        val_df = prepare_criteria_dataframe(
            criteria_groundtruth, post_text_map, criterion_text_map, criterion_to_index, splits["val"]
        )
        test_df = prepare_criteria_dataframe(
            criteria_groundtruth, post_text_map, criterion_text_map, criterion_to_index, splits["test"]
        )

        datasets = DatasetSplits(
            train=ClassificationDataset(
                train_df,
                tokenizer=tokenizer,
                max_length=max_length,
                text_column="post_text",
                text_pair_column="criterion_text",
                label_column="label",
                label_dtype="int",
            ),
            val=ClassificationDataset(
                val_df,
                tokenizer=tokenizer,
                max_length=max_length,
                text_column="post_text",
                text_pair_column="criterion_text",
                label_column="label",
                label_dtype="int",
            ),
            test=ClassificationDataset(
                test_df,
                tokenizer=tokenizer,
                max_length=max_length,
                text_column="post_text",
                text_pair_column="criterion_text",
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
