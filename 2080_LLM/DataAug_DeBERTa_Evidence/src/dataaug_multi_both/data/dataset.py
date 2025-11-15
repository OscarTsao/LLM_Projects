from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, Features, Value, load_dataset
from transformers import PreTrainedTokenizerBase

from dataaug_multi_both.data.token_cache import (
    build_cache_path,
    load_cached_dataset,
    save_cached_dataset,
)

logger = logging.getLogger(__name__)


@dataclass
class DatasetMetadata:
    dataset_id: str
    revision: str | None
    num_examples: dict[str, int]
    tokenizer_name: str
    max_length: int


def _build_smoke_dataset(size: int) -> DatasetDict:
    size = max(4, size)
    rng = np.random.default_rng(123)
    texts = [f"Sample text {idx}" for idx in range(size)]
    evidence_sentences = [f"Evidence sentence {idx}" for idx in range(size)]
    evidence_labels = rng.integers(0, 3, size=size).tolist()
    criteria_labels = rng.integers(0, 5, size=size).tolist()

    records = []
    for text, evidence, ev_label, cr_label in zip(
        texts, evidence_sentences, evidence_labels, criteria_labels
    ):
        records.append(
            {
                "text": text,
                "evidence": evidence,
                "evidence_label": int(ev_label),
                "criteria_label": int(cr_label),
            }
        )

    train = Dataset.from_list(records[: int(size * 0.6)])
    validation = Dataset.from_list(records[int(size * 0.6) : int(size * 0.8)])
    test = Dataset.from_list(records[int(size * 0.8) :])
    features = Features(
        {
            "text": Value("string"),
            "evidence": Value("string"),
            "evidence_label": Value("int64"),
            "criteria_label": Value("int64"),
        }
    )
    train = train.cast(features)
    validation = validation.cast(features)
    test = test.cast(features)
    return DatasetDict(train=train, validation=validation, test=test)


def load_raw_datasets(cfg: dict) -> Tuple[DatasetDict, DatasetMetadata]:
    dataset_cfg = cfg["data"]
    dataset_id = dataset_cfg["dataset_id"]
    revision = dataset_cfg.get("revision")
    cache_dir = dataset_cfg.get("cache_dir")
    splits = dataset_cfg.get("splits", {})
    max_examples = dataset_cfg.get("max_examples")

    if dataset_id == "local/smoke":
        size = int(dataset_cfg.get("smoke_dataset_size", 32))
        dataset = _build_smoke_dataset(size)
        metadata = DatasetMetadata(
            dataset_id=dataset_id,
            revision=revision,
            num_examples={split: dataset[split].num_rows for split in dataset},
            tokenizer_name=cfg["encoder"]["tokenizer_name"],
            max_length=int(cfg["tokenizer"]["max_length"]),
        )
        return dataset, metadata

    logger.info("Loading dataset %s (revision=%s)", dataset_id, revision or "latest")
    dataset = load_dataset(
        dataset_id,
        split=None,
        revision=revision,
        cache_dir=cache_dir,
    )

    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})

    mapped_splits = {}
    for logical_split, hf_split in splits.items():
        if hf_split not in dataset:
            raise KeyError(f"Split '{hf_split}' not found in dataset {dataset_id}")
        mapped_splits[logical_split] = dataset[hf_split]
    dataset = DatasetDict(mapped_splits)

    if max_examples:
        for split_name, ds in dataset.items():
            if ds.num_rows > max_examples:
                dataset[split_name] = ds.select(range(max_examples))

    metadata = DatasetMetadata(
        dataset_id=dataset_id,
        revision=revision,
        num_examples={split: dataset[split].num_rows for split in dataset},
        tokenizer_name=cfg["encoder"]["tokenizer_name"],
        max_length=int(cfg["tokenizer"]["max_length"]),
    )

    return dataset, metadata


def _prepare_class_weights(labels: Iterable[int], num_classes: int) -> np.ndarray:
    counts = np.bincount(list(labels), minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    inv_freq = 1.0 / counts
    inv_freq /= inv_freq.sum()
    return inv_freq


def tokenize_datasets(
    dataset: DatasetDict,
    cfg: dict,
    tokenizer: PreTrainedTokenizerBase,
    augmentation_signature: str | None = None,
) -> Tuple[DatasetDict, Dict[str, Any]]:
    data_cfg = cfg["data"]
    token_cfg = cfg["tokenizer"]
    text_field = data_cfg["fields"]["text"]
    evidence_field = data_cfg["fields"]["evidence"]
    evidence_label_field = data_cfg["fields"]["evidence_label"]
    criteria_label_field = data_cfg["fields"]["criteria_label"]
    max_length = int(token_cfg["max_length"])

    cache_dir = data_cfg.get("token_cache_dir", "./Data/token_cache")
    cache_path = build_cache_path(
        cache_dir,
        dataset_id=data_cfg["dataset_id"],
        revision=data_cfg.get("revision"),
        tokenizer_name=tokenizer.name_or_path,
        max_length=max_length,
        extra=augmentation_signature,
    )

    cached = load_cached_dataset(cache_path)
    if cached is not None:
        logger.info("Loaded tokenized dataset from cache %s", cache_path)
        return cached, {
            "class_weights": {
                "evidence": _prepare_class_weights(
                    cached["train"]["evidence_label"], cfg["heads"]["evidence"]["num_classes"]
                ),
                "criteria": _prepare_class_weights(
                    cached["train"]["criteria_label"], cfg["heads"]["criteria"]["num_classes"]
                ),
            }
        }

    remove_columns = set()
    for split in dataset:
        remove_columns.update(dataset[split].column_names)
    remove_columns -= {
        text_field,
        evidence_field,
        evidence_label_field,
        criteria_label_field,
    }

    def preprocess(batch: dict) -> dict:
        combined_inputs = [
            f"{text} [SEP] Evidence: {evidence}"
            for text, evidence in zip(batch[text_field], batch[evidence_field])
        ]
        encoded = tokenizer(
            combined_inputs,
            max_length=max_length,
            truncation=True,
            padding="max_length",
        )
        encoded["criteria_label"] = batch[criteria_label_field]
        encoded["evidence_label"] = batch[evidence_label_field]
        return encoded

    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=list(remove_columns),
    )
    tokenized.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "criteria_label", "evidence_label"],
    )

    save_cached_dataset(tokenized, cache_path)
    logger.info("Saved tokenized dataset cache to %s", cache_path)

    class_weights = {
        "evidence": _prepare_class_weights(
            dataset["train"][evidence_label_field], cfg["heads"]["evidence"]["num_classes"]
        ),
        "criteria": _prepare_class_weights(
            dataset["train"][criteria_label_field], cfg["heads"]["criteria"]["num_classes"]
        ),
    }

    return tokenized, {"class_weights": class_weights}
