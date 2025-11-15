from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase

from .utils import ensure_dir, load_yaml, resolve_samples_limit, setup_logger


@dataclass
class DatasetBundle:
    dataset: Dataset
    doc_ids: List[str]
    doc_targets: Dict[str, np.ndarray]
    doc_texts: Dict[str, str]
    split: str

    @property
    def num_labels(self) -> int:
        if not self.doc_targets:
            return 0
        return len(next(iter(self.doc_targets.values())))


class MultiLabelDataCollator(DataCollatorWithPadding):
    """Pad inputs to the longest sequence and keep labels as float tensors."""

    label_key: str = "labels"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:  # type: ignore[override]
        labels = [feature[self.label_key] for feature in features]
        for feature in features:
            feature.pop("doc_id", None)
        batch = super().__call__(features)
        batch[self.label_key] = torch.tensor(labels, dtype=torch.float32)
        return batch


def load_label_list(path: str | Path) -> List[str]:
    cfg = load_yaml(path)
    labels = cfg.get("labels", [])
    drop_labels = set(cfg.get("drop_labels", []))
    return [label for label in labels if label not in drop_labels]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        return pd.read_json(path, orient="records", lines=True)
    if path.suffix == ".json":
        return pd.read_json(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported file extension for {path}")


def _detect_split_file(data_dir: Path, split: str) -> Path:
    for ext in (".jsonl", ".json", ".csv", ".tsv"):
        candidate = data_dir / f"{split}{ext}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find a file for split '{split}' in {data_dir}")


def _auto_detect_label_columns(df: pd.DataFrame, label_list: Sequence[str]) -> List[str]:
    present = [label for label in label_list if label in df.columns]
    if len(present) != len(label_list):
        missing = set(label_list) - set(present)
        raise ValueError(f"Missing label columns: {sorted(missing)}")
    return present


def _make_doc_ids(size: int, split: str) -> List[str]:
    return [f"{split}_{idx}" for idx in range(size)]


def _stratified_split_df(
    df: pd.DataFrame,
    label_list: Sequence[str],
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    labels = df[list(label_list)].values.astype(int)
    signatures = ["".join(map(str, row.tolist())) for row in labels]
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=signatures,
        )
    except ValueError:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_local_dataset(
    data_dir: str | Path,
    label_list: Sequence[str],
    splits: Tuple[str, str, str],
    seed: int,
) -> DatasetDict:
    data_dir = Path(data_dir)
    train_split, dev_split, test_split = splits
    train_path = _detect_split_file(data_dir, train_split)
    train_df = _read_table(train_path)

    label_cols = _auto_detect_label_columns(train_df, label_list)

    split_frames: Dict[str, pd.DataFrame] = {train_split: train_df}
    missing_splits = []
    for split_name in (dev_split, test_split):
        try:
            split_path = _detect_split_file(data_dir, split_name)
            split_df = _read_table(split_path)
            _auto_detect_label_columns(split_df, label_list)
            split_frames[split_name] = split_df
        except FileNotFoundError:
            missing_splits.append(split_name)

    temp_train = split_frames[train_split]
    if dev_split in missing_splits:
        temp_train, dev_df = _stratified_split_df(temp_train, label_list, test_size=0.1, seed=seed)
        split_frames[dev_split] = dev_df
    if test_split in missing_splits:
        temp_train, test_df = _stratified_split_df(temp_train, label_list, test_size=0.1, seed=seed)
        split_frames[test_split] = test_df
    split_frames[train_split] = temp_train

    dataset_dict = {}
    for split_name, df in split_frames.items():
        if "text" not in df.columns:
            raise ValueError(f"Split '{split_name}' must contain a 'text' column")
        df = df.reset_index(drop=True).copy()
        df["doc_id"] = _make_doc_ids(len(df), split_name)
        dataset_dict[split_name] = Dataset.from_pandas(df, preserve_index=False)
    return DatasetDict(dataset_dict)


def _stratified_split(
    dataset: Dataset,
    label_list: Sequence[str],
    test_size: float,
    seed: int,
) -> Tuple[Dataset, Dataset]:
    if len(dataset) == 0:
        return dataset, dataset
    df = dataset.to_pandas()
    labels = df[list(label_list)].values.astype(int)
    signatures = ["".join(map(str, row.tolist())) for row in labels]
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=signatures,
        )
    except ValueError:
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True), preserve_index=False)
    test_ds = Dataset.from_pandas(test_df.reset_index(drop=True), preserve_index=False)
    return train_ds, test_ds


def load_hf_dataset(
    hf_id: str,
    hf_config: Optional[str],
    splits: Tuple[str, str, str],
    label_list: Sequence[str],
    seed: int,
) -> DatasetDict:
    dataset_dict = load_dataset(hf_id, hf_config or None)
    train_split, dev_split, test_split = splits
    if train_split not in dataset_dict:
        raise ValueError(f"HF dataset must include a '{train_split}' split")

    if dev_split not in dataset_dict:
        train_ds, dev_ds = _stratified_split(dataset_dict[train_split], label_list, 0.1, seed)
        dataset_dict[train_split] = train_ds
        dataset_dict[dev_split] = dev_ds
    if test_split not in dataset_dict:
        train_ds, test_ds = _stratified_split(dataset_dict[train_split], label_list, 0.1, seed)
        dataset_dict[train_split] = train_ds
        dataset_dict[test_split] = test_ds

    for split_name in dataset_dict.keys():
        if "text" not in dataset_dict[split_name].column_names:
            raise ValueError(f"HF dataset split '{split_name}' must include a 'text' column")

    return dataset_dict


def compute_label_stats(doc_targets: Mapping[str, np.ndarray]) -> Dict[str, float]:
    if not doc_targets:
        return {}
    stacked = np.stack(list(doc_targets.values()), axis=0)
    return {
        "pos_rate": stacked.mean(axis=0).tolist(),
        "support": stacked.sum(axis=0).tolist(),
        "total": float(stacked.shape[0]),
    }


def derive_class_weights(
    stats: Dict[str, float],
    mode: str,
    epsilon: float = 1e-6,
) -> Optional[np.ndarray]:
    if mode == "none" or not stats:
        return None
    pos_rate = np.asarray(stats["pos_rate"], dtype=np.float32)
    pos_rate = np.clip(pos_rate, epsilon, 1.0)
    if mode == "inv":
        weights = 1.0 / pos_rate
    elif mode == "sqrt_inv":
        weights = 1.0 / np.sqrt(pos_rate)
    else:
        raise ValueError(f"Unknown class_weighting='{mode}'")
    return weights.astype(np.float32)


def _prepare_split_metadata(
    dataset: Dataset,
    label_list: Sequence[str],
    split_name: str,
) -> Tuple[Dataset, Dict[str, np.ndarray], Dict[str, str]]:
    doc_targets: Dict[str, np.ndarray] = {}
    doc_texts: Dict[str, str] = {}

    def _process(example: Dict[str, Any]) -> Dict[str, Any]:
        labels = [float(example[label]) for label in label_list]
        doc_id = example.get("doc_id")
        if doc_id is None:
            raise KeyError("doc_id missing after preprocessing")
        doc_targets.setdefault(doc_id, np.asarray(labels, dtype=np.float32))
        doc_texts.setdefault(doc_id, example["text"])
        return {"labels": labels}

    dataset = dataset.map(_process)
    dataset = dataset.remove_columns([col for col in label_list if col in dataset.column_names])
    return dataset, doc_targets, doc_texts


def _assign_doc_ids(dataset: Dataset, split_name: str) -> Dataset:
    def _add_doc_id(example: Dict[str, Any], idx: int) -> Dict[str, Any]:
        doc_id = example.get("id")
        if doc_id is None or doc_id == "":
            doc_id = f"{split_name}_{idx}"
        return {"doc_id": str(doc_id)}

    dataset = dataset.map(_add_doc_id, with_indices=True)
    return dataset


def _tokenize_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    doc_stride: int,
    truncation_strategy: str,
) -> Tuple[Dataset, List[str]]:
    use_windows = truncation_strategy == "window_pool"

    def _tokenize(batch: Dict[str, List[Any]]) -> Dict[str, Any]:
        texts = batch["text"]
        outputs = tokenizer(
            texts,
            max_length=max_length,
            padding=False,
            truncation=True,
            stride=doc_stride if use_windows else 0,
            return_overflowing_tokens=use_windows,
            return_attention_mask=True,
        )
        sample_mapping = outputs.pop("overflow_to_sample_mapping", None)
        if sample_mapping is None:
            doc_ids = batch["doc_id"]
            labels = batch["labels"]
        else:
            doc_ids = [batch["doc_id"][i] for i in sample_mapping]
            labels = [batch["labels"][i] for i in sample_mapping]
        outputs["doc_id"] = doc_ids
        outputs["labels"] = labels
        return outputs

    tokenized = dataset.map(
        _tokenize,
        batched=True,
        remove_columns=[col for col in dataset.column_names if col not in {"text", "labels", "doc_id"}],
    )

    doc_ids = tokenized["doc_id"]
    tokenized = tokenized.remove_columns(["text"])
    return tokenized, doc_ids


def build_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    return tokenizer


def prepare_datasets(
    cfg: Mapping[str, Any],
    labels_path: str | Path,
    tokenizer: Optional[PreTrainedTokenizerBase] = None,
    logger=None,
) -> Tuple[Dict[str, DatasetBundle], np.ndarray, List[str], PreTrainedTokenizerBase]:
    logger = logger or setup_logger()
    label_list = load_label_list(labels_path)
    splits = (
        cfg.get("train_split", "train"),
        cfg.get("dev_split", "validation"),
        cfg.get("test_split", "test"),
    )
    seed = int(cfg.get("seed", 42))

    if cfg.get("hf_id"):
        dataset_dict = load_hf_dataset(cfg["hf_id"], cfg.get("hf_config"), splits, label_list, seed)
    elif cfg.get("data_dir"):
        dataset_dict = load_local_dataset(cfg["data_dir"], label_list, splits, seed)
    else:
        raise ValueError("Either hf_id or data_dir must be specified in the configuration")

    bundles: Dict[str, DatasetBundle] = {}
    doc_stats: Dict[str, np.ndarray] = {}

    if tokenizer is None:
        tokenizer = build_tokenizer(cfg["model_id"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    split_limit_map = {
        splits[0]: cfg.get("max_train_samples"),
        splits[1]: cfg.get("max_eval_samples"),
        splits[2]: cfg.get("max_test_samples"),
    }

    for split_name, dataset in dataset_dict.items():
        dataset = _assign_doc_ids(dataset, split_name)
        limit = split_limit_map.get(split_name)
        if limit:
            limit = resolve_samples_limit(limit, len(dataset))
            dataset = dataset.select(range(limit))
        dataset, doc_targets, doc_texts = _prepare_split_metadata(dataset, label_list, split_name)
        tokenized, doc_ids = _tokenize_dataset(
            dataset,
            tokenizer,
            max_length=int(cfg.get("max_length", 4096)),
            doc_stride=int(cfg.get("doc_stride", 512)),
            truncation_strategy=str(cfg.get("truncation_strategy", "window_pool")),
        )
        doc_stats[split_name] = np.stack(list(doc_targets.values())) if doc_targets else np.zeros((0, len(label_list)))
        bundles[split_name] = DatasetBundle(
            dataset=tokenized,
            doc_ids=doc_ids,
            doc_targets=doc_targets,
            doc_texts=doc_texts,
            split=split_name,
        )

    train_targets = bundles[splits[0]].doc_targets
    stats = compute_label_stats(train_targets)
    class_weights = derive_class_weights(stats, cfg.get("class_weighting", "none"))

    # Log class imbalance information
    if stats:
        pos_rates = stats["pos_rate"]
        message = "Class positive rates: " + ", ".join(
            f"{label}:{rate:.3f}" for label, rate in zip(label_list, pos_rates)
        )
        logger.info(message)

    for bundle in bundles.values():
        selected_cols = [col for col in bundle.dataset.column_names if col in {"input_ids", "attention_mask", "labels"}]
        bundle.dataset.set_format(type="torch", columns=selected_cols)

    return bundles, class_weights, label_list, tokenizer


__all__ = [
    "DatasetBundle",
    "MultiLabelDataCollator",
    "prepare_datasets",
    "build_tokenizer",
    "load_label_list",
    "compute_label_stats",
]
