from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[4]
    / "data"
    / "processed"
    / "redsm5_matched_evidence.csv"
)


def _find_answer_span(context: str, answer: str) -> tuple[int, int]:
    answer = answer.strip()
    start_idx = context.find(answer)
    if start_idx == -1:
        lower_context = context.lower()
        lower_answer = answer.lower()
        start_idx = lower_context.find(lower_answer)
        if start_idx == -1:
            raise ValueError("Answer span not found in context.")
        end_idx = start_idx + len(lower_answer)
        answer = context[start_idx:end_idx]
    end_idx = start_idx + len(answer)
    return start_idx, end_idx


def _token_span_from_char(
    offsets: torch.Tensor,
    start_char: int,
    end_char: int,
) -> tuple[int, int]:
    start_token = None
    end_token = None
    for idx, (token_start, token_end) in enumerate(offsets.tolist()):
        if token_start == token_end == 0:
            continue
        if start_token is None and token_start <= start_char < token_end:
            start_token = idx
        if token_start < end_char <= token_end:
            end_token = idx
            break
    if start_token is None or end_token is None:
        raise ValueError("Unable to align character span with token offsets.")
    return start_token, end_token


class ShareDataset(Dataset):
    """Dataset for multi-task learning with shared encoder (classification + span)."""

    def __init__(
        self,
        csv_path: Path | str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        tokenizer_name: str = "bert-base-uncased",
        context_column: str = "post_text",
        answer_column: str = "sentence_text",
        label_column: str = "status",
        max_length: int = 512,
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
                raise ValueError(
                    f"The dataset file {self.csv_path} is missing a header row."
                )
            self.examples: list[dict[str, str]] = [row for row in reader]

        if not self.examples:
            raise ValueError(f"No rows found in dataset file {self.csv_path}.")

        self.context_column = context_column
        self.answer_column = answer_column
        self.label_column = label_column
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        missing_columns = [
            column
            for column in (self.context_column, self.answer_column, self.label_column)
            if column not in self.examples[0]
        ]
        if missing_columns:
            raise KeyError(
                f"Columns {missing_columns} not found in dataset file {self.csv_path}. "
                f"Available columns: {sorted(self.examples[0].keys())}"
            )

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(tokenizer_name)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]

        context = example[self.context_column]
        answer = example[self.answer_column]
        label = int(example[self.label_column])

        start_char, end_char = _find_answer_span(context, answer)

        encoded = self.tokenizer(
            context,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        offsets = encoded.pop("offset_mapping").squeeze(0)
        start_token, end_token = _token_span_from_char(offsets, start_char, end_char)

        item: dict[str, torch.Tensor] = {
            key: value.squeeze(0) for key, value in encoded.items()
        }
        item["labels"] = torch.tensor(label, dtype=torch.long)
        item["start_positions"] = torch.tensor(start_token, dtype=torch.long)
        item["end_positions"] = torch.tensor(end_token, dtype=torch.long)
        return item


def load_share_dataset(
    splits: Sequence[float] | None = None,
    **dataset_kwargs,
) -> ShareDataset | tuple[Dataset, ...]:
    dataset: ShareDataset = ShareDataset(**dataset_kwargs)
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


__all__ = ["ShareDataset", "load_share_dataset", "DEFAULT_DATASET_PATH"]
