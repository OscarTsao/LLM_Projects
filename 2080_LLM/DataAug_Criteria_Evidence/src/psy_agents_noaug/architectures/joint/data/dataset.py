"""Joint dataset that feeds dual encoders (criteria + evidence).

Builds two tokenised views of each sample and aligns evidence spans to token
offsets, returning both classification labels and start/end positions.
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


class JointDataset(Dataset):
    """Provide inputs for joint criteria and evidence models in one sample."""

    def __init__(
        self,
        csv_path: Path | str | None = None,
        criteria_tokenizer: PreTrainedTokenizerBase | None = None,
        evidence_tokenizer: PreTrainedTokenizerBase | None = None,
        criteria_tokenizer_name: str = "bert-base-uncased",
        evidence_tokenizer_name: str = "bert-base-uncased",
        sentence_column: str = "sentence_text",
        context_column: str = "post_text",
        label_column: str = "status",
        criteria_max_length: int = 256,
        evidence_max_length: int = 512,
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

        self.sentence_column = sentence_column
        self.context_column = context_column
        self.label_column = label_column
        self.criteria_max_length = criteria_max_length
        self.evidence_max_length = evidence_max_length
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
            for column in (self.sentence_column, self.context_column, self.label_column)
            if column not in self.examples[0]
        ]
        if missing_columns:
            raise KeyError(
                f"Columns {missing_columns} not found in dataset file {self.csv_path}. "
                f"Available columns: {sorted(self.examples[0].keys())}"
            )

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

        self.criteria_tokenizer = criteria_tokenizer or AutoTokenizer.from_pretrained(
            criteria_tokenizer_name
        )
        if (
            evidence_tokenizer is None
            and criteria_tokenizer_name == evidence_tokenizer_name
        ):
            # Reuse tokenizer instance when weights are expected to be shared.
            self.evidence_tokenizer = self.criteria_tokenizer
        else:
            self.evidence_tokenizer = (
                evidence_tokenizer
                or AutoTokenizer.from_pretrained(evidence_tokenizer_name)
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
        example = self.examples[index]
        sentence = example[self.sentence_column]
        context = example[self.context_column]
        label = int(example[self.label_column])
        criterion_text = self._get_criterion_text(example)

        # Format: [CLS] criterion [SEP] sentence [SEP] (criterion first for better attention)
        criteria_encoded = self.criteria_tokenizer(
            criterion_text,  # First sequence
            sentence,  # Second sequence
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.criteria_max_length,
            return_tensors="pt",
        )

        start_char, end_char = _find_answer_span(context, sentence)
        # Format: [CLS] criterion [SEP] context [SEP] (criterion first for better attention)
        evidence_encoded = self.evidence_tokenizer(
            criterion_text,  # First sequence
            context,  # Second sequence
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.evidence_max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )

        sequence_ids = evidence_encoded.sequence_ids(0)
        offsets = evidence_encoded.pop("offset_mapping").squeeze(0)
        start_token, end_token = _token_span_from_char(
            offsets,
            start_char,
            end_char,
            sequence_ids=sequence_ids,
            target_sequence_id=0,
        )

        item: dict[str, torch.Tensor] = {}

        for key, value in criteria_encoded.items():
            item[f"criteria_{key}"] = value.squeeze(0)

        for key, value in evidence_encoded.items():
            item[f"evidence_{key}"] = value.squeeze(0)

        item["labels"] = torch.tensor(label, dtype=torch.long)
        item["start_positions"] = torch.tensor(start_token, dtype=torch.long)
        item["end_positions"] = torch.tensor(end_token, dtype=torch.long)
        return item


def load_joint_dataset(
    splits: Sequence[float] | None = None,
    **dataset_kwargs,
) -> JointDataset | tuple[Dataset, ...]:
    dataset: JointDataset = JointDataset(**dataset_kwargs)
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


__all__ = ["JointDataset", "load_joint_dataset", "DEFAULT_DATASET_PATH"]
