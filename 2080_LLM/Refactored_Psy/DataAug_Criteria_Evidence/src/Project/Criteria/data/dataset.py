from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[4] / "data" / "redsm5" / "redsm5_annotations.csv"
)


class CriteriaDataset(Dataset):
    """Dataset reading RED-SM5 sentence annotations for binary criteria classification."""

    def __init__(
        self,
        csv_path: Optional[Union[Path, str]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        tokenizer_name: str = "bert-base-uncased",
        text_column: str = "sentence_text",
        label_column: str = "status",
        max_length: int = 256,
        padding: str = "max_length",
        truncation: bool = True,
    ) -> None:
        super().__init__()
        self.csv_path = Path(csv_path or DEFAULT_DATASET_PATH)
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {self.csv_path}")

        with self.csv_path.open("r", encoding="utf-8", newline="") as fp:
            reader = csv.DictReader(fp)
            if reader.fieldnames is None:
                raise ValueError(f"The dataset file {self.csv_path} is missing a header row.")
            self.examples: List[Dict[str, str]] = [row for row in reader]

        if not self.examples:
            raise ValueError(f"No rows found in dataset file {self.csv_path}.")

        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

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

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        example = self.examples[index]
        text = example[self.text_column]
        label = int(example[self.label_column])

        encoded = self.tokenizer(
            text,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt",
        )

        item: Dict[str, torch.Tensor] = {
            key: value.squeeze(0) for key, value in encoded.items()
        }
        item["labels"] = torch.tensor(label, dtype=torch.long)
        return item


def load_criteria_dataset(
    splits: Optional[Sequence[float]] = None,
    **dataset_kwargs,
) -> Union[CriteriaDataset, Tuple[Dataset, ...]]:
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
