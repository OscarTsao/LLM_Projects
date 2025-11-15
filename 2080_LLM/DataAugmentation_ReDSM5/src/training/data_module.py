"""PyTorch dataset and dataloader utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .dataset_builder import DatasetSplit


class PairDataset(Dataset):
    def __init__(self, records):
        self._records = records

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int):
        row = self._records.iloc[index]
        return {
            "text_a": row["text_a"],
            "text_b": row["text_b"],
            "label": int(row["label"]),
        }


@dataclass
class DataModuleConfig:
    tokenizer_name: str
    max_seq_length: int
    batch_size: int
    num_workers: int
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    eval_batch_size: int | None = None  # If None, uses batch_size for eval too


class PairCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[dict]):
        texts_a = [example["text_a"] for example in batch]
        texts_b = [example["text_b"] for example in batch]
        labels = torch.tensor([example["label"] for example in batch], dtype=torch.long)
        encodings = self.tokenizer(
            texts_a,
            texts_b,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encodings["labels"] = labels
        return encodings


class DataModule:
    def __init__(self, split: DatasetSplit, config: DataModuleConfig) -> None:
        self.split = split
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name, use_fast=True)
        self.collator = PairCollator(self.tokenizer, config.max_seq_length)

    def _loader(self, dataset: Dataset, shuffle: bool, batch_size: int | None = None) -> DataLoader:
        if batch_size is None:
            batch_size = self.config.batch_size
        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            collate_fn=self.collator,
        )
        if self.config.num_workers > 0:
            loader_kwargs["persistent_workers"] = self.config.persistent_workers
            loader_kwargs["prefetch_factor"] = self.config.prefetch_factor
        return DataLoader(dataset, **loader_kwargs)

    def train_dataloader(self) -> DataLoader:
        return self._loader(PairDataset(self.split.train), shuffle=True)

    def val_dataloader(self) -> DataLoader:
        eval_bs = self.config.eval_batch_size or self.config.batch_size
        return self._loader(PairDataset(self.split.val), shuffle=False, batch_size=eval_bs)

    def test_dataloader(self) -> DataLoader:
        eval_bs = self.config.eval_batch_size or self.config.batch_size
        return self._loader(PairDataset(self.split.test), shuffle=False, batch_size=eval_bs)
