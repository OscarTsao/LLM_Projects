"""Evidence dataset for span extraction over RED-SM5 annotations.

Builds paired inputs (context + criterion text) and aligns character-level
evidence spans to token-level start/end positions using tokenizer offsets.
"""

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
    Path(__file__).resolve().parents[5]
    / "data"
    / "processed"
    / "redsm5_matched_evidence.csv"
)


def _find_answer_span(context: str, answer: str) -> tuple[int, int]:
    """Locate the answer span within the context and return character indices."""
    answer = answer.strip()
    start_idx = context.find(answer)
    if start_idx == -1:
        # Fallback to case-insensitive search
        lower_context = context.lower()
        lower_answer = answer.lower()
        start_idx = lower_context.find(lower_answer)
        if start_idx == -1:
            raise ValueError("Answer span not found in context.")
        end_idx = start_idx + len(lower_answer)
        # Map back to original casing to keep alignment
        answer = context[start_idx:end_idx]
    end_idx = start_idx + len(answer)
    return start_idx, end_idx


def _token_span_from_char(
    offsets: torch.Tensor,
    start_char: int,
    end_char: int,
    sequence_ids: Sequence[int | None] | None = None,
    target_sequence_id: int = 0,
) -> tuple[int, int]:
    start_token = None
    end_token = None
    offsets_list = offsets.tolist()
    for idx, (token_start, token_end) in enumerate(offsets_list):
        if sequence_ids is not None:
            segment_id = sequence_ids[idx]
            if segment_id != target_sequence_id:
                continue
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


class EvidenceDataset(Dataset):
    """Span extraction dataset mapping context+criterion to start/end tokens."""

    def __init__(
        self,
        csv_path: Path | str | None = None,
        tokenizer: PreTrainedTokenizerBase | None = None,
        tokenizer_name: str = "bert-base-uncased",
        context_column: str = "post_text",
        answer_column: str = "sentence_text",
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
        criterion_column: str | None = None,
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

        self.context_column = context_column
        self.answer_column = answer_column
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.criterion_columns: list[str] = []
        if criterion_column:
            self.criterion_columns.append(criterion_column)
        self.criterion_columns.extend(["criterion_id", "DSM5_symptom"])
        seen: dict[str, None] = {}
        self.criterion_columns = [
            column
            for column in self.criterion_columns
            if not (column in seen or seen.setdefault(column, None))
        ]

        missing_columns = [
            column
            for column in (self.context_column, self.answer_column)
            if column not in self.examples[0]
        ]
        if missing_columns:
            raise KeyError(
                f"Columns {missing_columns} not found in dataset file {self.csv_path}. "
                f"Available columns: {sorted(self.examples[0].keys())}"
            )

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
            if isinstance(value, str) and not value.strip():
                continue
            fallback = self._format_identifier(value)
            return resolve_criterion_text(value, self.criteria_map, fallback=fallback)

        raise KeyError(
            "Could not resolve criterion identifier. "
            f"Tried columns: {self.criterion_columns}. "
            f"Available columns: {sorted(example.keys())}"
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return a tokenised sample with start/end positions for the answer."""
        example = self.examples[index]

        context = example[self.context_column]
        answer = example[self.answer_column]
        criterion_text = self._get_criterion_text(example)

        start_char, end_char = _find_answer_span(context, answer)

        encoded = self.tokenizer(
            context,
            criterion_text,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        sequence_ids = encoded.sequence_ids(0)
        offsets = encoded.pop("offset_mapping").squeeze(0)
        start_token, end_token = _token_span_from_char(
            offsets,
            start_char,
            end_char,
            sequence_ids=sequence_ids,
            target_sequence_id=0,
        )

        item: dict[str, torch.Tensor] = {
            key: value.squeeze(0) for key, value in encoded.items()
        }
        item["start_positions"] = torch.tensor(start_token, dtype=torch.long)
        item["end_positions"] = torch.tensor(end_token, dtype=torch.long)
        return item


def load_evidence_dataset(
    splits: Sequence[float] | None = None,
    **dataset_kwargs,
) -> EvidenceDataset | tuple[Dataset, ...]:
    """Load a single dataset or split into subsets according to ``splits``."""
    dataset: EvidenceDataset = EvidenceDataset(**dataset_kwargs)
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


__all__ = ["EvidenceDataset", "load_evidence_dataset", "DEFAULT_DATASET_PATH"]
