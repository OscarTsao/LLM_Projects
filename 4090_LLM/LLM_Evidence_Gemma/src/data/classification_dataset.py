"""
Dataset helpers for LLM-based text classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from datasets import DatasetDict, load_dataset


def load_classification_dataset(
    dataset_name: Optional[str],
    dataset_config: Optional[str],
    train_file: Optional[str],
    validation_file: Optional[str],
    text_column: str,
    label_column: str,
    second_text_column: Optional[str] = None,
) -> Tuple[DatasetDict, List[str]]:
    """
    Load a Hugging Face dataset (hub or local files) and infer label names.
    """
    if dataset_name:
        dataset = load_dataset(dataset_name, dataset_config)
    else:
        data_files = {}
        if train_file:
            data_files["train"] = train_file
        if validation_file:
            data_files["validation"] = validation_file
        if not data_files:
            raise ValueError("Provide --dataset_name or train/validation files.")
        extension = train_file.split(".")[-1] if train_file else validation_file.split(".")[-1]
        dataset = load_dataset(extension, data_files=data_files)

    label_list = infer_label_list(dataset["train"], label_column)
    dataset = dataset.filter(lambda example: example[label_column] is not None)
    label2id = {name: idx for idx, name in enumerate(label_list)}

    def encode_labels(example):
        return {"labels": label2id[str(example[label_column])]}

    dataset = dataset.map(encode_labels)

    columns_to_keep = {"labels", text_column}
    if second_text_column:
        columns_to_keep.add(second_text_column)

    for split in dataset.keys():
        remove_cols = [col for col in dataset[split].column_names if col not in columns_to_keep]
        if remove_cols:
            dataset[split] = dataset[split].remove_columns(remove_cols)
    return dataset, label_list


def infer_label_list(dataset, label_column: str) -> List[str]:
    """
    Infer label names from Dataset features or unique values.
    """
    feature = dataset.features.get(label_column)
    if feature and feature.dtype == "string":
        values = sorted(set(dataset[label_column]))
    elif feature and feature.dtype == "int64" and hasattr(feature, "names"):
        values = list(feature.names)
    else:
        values = sorted({str(v) for v in dataset[label_column]})
    return [str(v) for v in values]


def compute_class_weights(labels: Iterable[int], num_labels: int) -> List[float]:
    """
    Compute inverse frequency class weights for unbalanced datasets.
    """
    counts = [0] * num_labels
    for label in labels:
        counts[label] += 1
    total = sum(counts)
    weights = [0.0] * num_labels
    for idx, count in enumerate(counts):
        weights[idx] = total / (num_labels * max(count, 1))
    return weights
