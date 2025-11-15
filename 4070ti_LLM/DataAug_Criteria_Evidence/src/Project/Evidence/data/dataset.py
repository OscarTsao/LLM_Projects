from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from collections.abc import Sequence

DEFAULT_DATASET_PATH = (
    Path(__file__).resolve().parents[4]
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
            # Fallback: Try to find first word of answer
            answer_words = lower_answer.split()
            if answer_words:
                first_word = answer_words[0]
                start_idx = lower_context.find(first_word)
                if start_idx != -1:
                    # Use first word position, extend to full answer length
                    end_idx = start_idx + len(lower_answer)
                    # Clamp to context length
                    end_idx = min(end_idx, len(context))
                    return start_idx, end_idx
            # Final fallback: use beginning of context
            # This handles cases where answer truly isn't in context
            # Model will learn these are "unanswerable" via loss
            start_idx = 0
            end_idx = min(len(answer), len(context))
            return start_idx, end_idx
        end_idx = start_idx + len(lower_answer)
        # Map back to original casing to keep alignment
        answer = context[start_idx:end_idx]
    end_idx = start_idx + len(answer)
    return start_idx, end_idx


def _token_span_from_char(
    offsets: torch.Tensor,
    start_char: int,
    end_char: int,
) -> tuple[int, int]:
    """Find token span from character positions with robust fallback.

    Args:
        offsets: Token offset mapping from tokenizer
        start_char: Character start position
        end_char: Character end position

    Returns:
        Tuple of (start_token_idx, end_token_idx)

    Raises:
        ValueError: If span cannot be aligned even with fallbacks
    """
    start_token = None
    end_token = None

    # First pass: exact match
    for idx, (token_start, token_end) in enumerate(offsets.tolist()):
        if token_start == token_end == 0:  # Skip special tokens
            continue
        if start_token is None and token_start <= start_char < token_end:
            start_token = idx
        if token_start < end_char <= token_end:
            end_token = idx
            break

    # Second pass: find closest tokens if exact match failed
    if start_token is None or end_token is None:
        min_start_dist = float("inf")
        min_end_dist = float("inf")

        for idx, (token_start, token_end) in enumerate(offsets.tolist()):
            if token_start == token_end == 0:
                continue

            # Find closest token to start_char
            if start_token is None:
                dist = min(abs(token_start - start_char), abs(token_end - start_char))
                if dist < min_start_dist:
                    min_start_dist = dist
                    start_token = idx

            # Find closest token to end_char
            if end_token is None:
                dist = min(abs(token_start - end_char), abs(token_end - end_char))
                if dist < min_end_dist and idx >= (start_token or 0):
                    min_end_dist = dist
                    end_token = idx

    # Validate we found something reasonable
    if start_token is None or end_token is None:
        raise ValueError(
            f"Unable to align character span [{start_char}, {end_char}] "
            f"with token offsets. start_token={start_token}, end_token={end_token}"
        )

    # Ensure end >= start
    end_token = max(end_token, start_token)

    return start_token, end_token


class EvidenceDataset(Dataset):
    """Dataset for span extraction on RED-SM5 evidence annotations.

    Supports optional data augmentation during training via augmentation_pipeline.
    """

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
        augmentation_pipeline: Any | None = None,
        is_training: bool = False,
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

        # Augmentation support
        self.augmentation_pipeline = augmentation_pipeline
        self.is_training = is_training

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]

        context = example[self.context_column]
        answer = example[self.answer_column]

        # NOTE: Augmentation disabled for Evidence/QA tasks
        # Augmenting context breaks answer span alignment (character positions change)
        # Future work: Implement span-aware augmentation that tracks position changes
        # For now, augmentation pipeline is accepted but not applied
        # if self.augmentation_pipeline and self.is_training:
        #     context = self.augmentation_pipeline.augment(context)

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
        item["start_positions"] = torch.tensor(start_token, dtype=torch.long)
        item["end_positions"] = torch.tensor(end_token, dtype=torch.long)
        return item


def load_evidence_dataset(
    splits: Sequence[float] | None = None,
    **dataset_kwargs,
) -> EvidenceDataset | tuple[Dataset, ...]:
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
