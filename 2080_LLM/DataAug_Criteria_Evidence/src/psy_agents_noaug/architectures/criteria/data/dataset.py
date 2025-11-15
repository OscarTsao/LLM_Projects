from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from psy_agents_noaug.architectures.utils import (
    build_criterion_text_map,
    resolve_criterion_text,
)
from psy_agents_noaug.architectures.utils.dsm_criteria import DEFAULT_DSM_CRITERIA_PATH

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[5] / "data" / "redsm5" / "redsm5_annotations.csv"
)


class CriteriaDataset(Dataset):
    """Dataset reading RED-SM5 sentence annotations for binary criteria classification."""

    def __init__(
        self,
        csv_path: Path | str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        tokenizer_name: str = "bert-base-uncased",
        text_column: str = "sentence_text",
        label_column: str = "status",
        criterion_column: str | None = None,
        max_length: int = 256,
        padding: str = "max_length",
        truncation: bool = True,
        dsm_criteria_path: Path | str | None = None,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path or DEFAULT_DATASET_PATH)
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {self.csv_path}")

        with self.csv_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            if reader.fieldnames is None:
                raise ValueError(
                    f"The dataset file {self.csv_path} is missing a header row."
                )
            self.examples: list[dict[str, str]] = list(reader)

        if not self.examples:
            raise ValueError(f"No rows found in dataset file {self.csv_path}.")

        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.criterion_columns: list[str] = []
        if criterion_column:
            self.criterion_columns.append(criterion_column)
        self.criterion_columns.extend(["criterion_id", "DSM5_symptom"])
        # Preserve order but drop duplicates
        seen: dict[str, None] = {}
        self.criterion_columns = [
            column
            for column in self.criterion_columns
            if not (column in seen or seen.setdefault(column, None))
        ]

        missing_columns = [
            column
            for column in (self.text_column, self.label_column)
            if column not in self.examples[0]
        ]
        if missing_columns:
            raise KeyError(
                f"Columns {missing_columns} not found in dataset file {self.csv_path}. "
                "Available columns: "
                f"{sorted(self.examples[0].keys())}"
            )

        # Users may provide a pre-configured tokenizer (e.g., shared between train/val).
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name)
        criteria_path = dsm_criteria_path or DEFAULT_DSM_CRITERIA_PATH
        self.criteria_map = build_criterion_text_map(criteria_path)

        self.available_criterion_columns = [
            column for column in self.criterion_columns if column in self.examples[0]
        ]
        self.has_direct_text_column = "criterion_text" in self.examples[0]
        if not self.has_direct_text_column and not self.available_criterion_columns:
            raise KeyError(
                "No criterion identifier column found. "
                f"Expected one of {self.criterion_columns}, found {sorted(self.examples[0].keys())}"
            )

    @staticmethod
    def _format_identifier(identifier: str | int) -> str:
        text = str(identifier).replace("_", " ").replace("-", " ").strip()
        return text if text else str(identifier)

    def _get_criterion_text(self, example: dict[str, str]) -> str:
        if self.has_direct_text_column:
            raw_text = str(example.get("criterion_text", "")).strip()
            if raw_text:
                return raw_text

        for column in self.available_criterion_columns:
            value = example.get(column)
            if value is None:
                continue
            fallback = self._format_identifier(value)
            if isinstance(value, str) and not value.strip():
                continue
            return resolve_criterion_text(value, self.criteria_map, fallback=fallback)

        raise KeyError(
            "Could not resolve criterion identifier. "
            f"Tried columns: {self.criterion_columns}. "
            f"Available columns: {sorted(example.keys())}"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        text = example[self.text_column]
        label = int(example[self.label_column])
        criterion_text = self._get_criterion_text(example)

        # Format: [CLS] criterion [SEP] text [SEP] (criterion first for better attention)
        encoded = self.tokenizer(
            criterion_text,  # First sequence
            text,  # Second sequence
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item: dict[str, torch.Tensor] = {
            key: value.squeeze(0) for key, value in encoded.items()
        }
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def load_criteria_dataset(
    splits: Sequence[float] | None = None,
    **dataset_kwargs,
) -> CriteriaDataset | tuple[Dataset, ...]:
    """Utility to construct CriteriaDataset objects, optionally split into subsets.

    Args:
        splits: Fractions summing to 1.0 to split dataset sequentially (e.g., [0.8, 0.2]).
        **dataset_kwargs: Parameters forwarded to ``CriteriaDataset``.
    """
    dataset: CriteriaDataset = CriteriaDataset(**dataset_kwargs)
    if not splits:
        return dataset

    if not torch.isclose(torch.tensor(sum(splits)), torch.tensor(1.0)).item():
        raise ValueError("Splits must sum to 1.0.")

    lengths = [int(len(dataset) * fraction) for fraction in splits]
    remainder = len(dataset) - sum(lengths)
    for i in range(remainder):
        lengths[i % len(lengths)] += 1

    subsets = torch.utils.data.random_split(dataset, lengths)
    return tuple(subsets)


__all__ = ["CriteriaDataset", "load_criteria_dataset", "DEFAULT_DATASET_PATH"]
