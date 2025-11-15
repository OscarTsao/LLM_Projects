from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from .data import QAExample


@dataclass
class TokenizedFeature:
    """Container describing one tokenized span for QA fine-tuning."""

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]]
    start_positions: Optional[int]
    end_positions: Optional[int]
    offset_mapping: List[List[int]]
    example_index: int


def prepare_train_features(
    examples: List[QAExample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    doc_stride: int,
) -> List[TokenizedFeature]:
    """Tokenize training examples into SpanBERT-compatible QA features."""

    questions = ["" for _ in examples]
    contexts = [ex.context for ex in examples]
    answer_starts = [ex.answer_start for ex in examples]
    answer_texts = [ex.answer_text for ex in examples]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    features: List[TokenizedFeature] = []
    for feature_idx, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][feature_idx]
        attention_mask = tokenized["attention_mask"][feature_idx]
        token_type_ids = tokenized.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[feature_idx]
        sequence_ids = tokenized.sequence_ids(feature_idx)

        sample_idx = sample_mapping[feature_idx]
        answer_start_char = answer_starts[sample_idx]
        answer_text = answer_texts[sample_idx]
        answer_end_char = answer_start_char + len(answer_text)

        cls_index = input_ids.index(tokenizer.cls_token_id)

        context_start = 0
        while context_start < len(sequence_ids) and sequence_ids[context_start] != 1:
            context_start += 1
        context_end = len(sequence_ids) - 1
        while context_end >= 0 and sequence_ids[context_end] != 1:
            context_end -= 1

        if context_start > context_end:
            start_pos = cls_index
            end_pos = cls_index
        else:
            # Default to CLS if the gold span falls outside this stride chunk.
            if not (
                offsets[context_start][0] <= answer_start_char
                and offsets[context_end][1] >= answer_end_char
            ):
                start_pos = cls_index
                end_pos = cls_index
            else:
                start_pos = end_pos = cls_index
                for idx in range(context_start, context_end + 1):
                    if (
                        start_pos == cls_index
                        and offsets[idx][0] <= answer_start_char
                        and offsets[idx][1] > answer_start_char
                    ):
                        start_pos = idx
                    if (
                        offsets[idx][0] < answer_end_char
                        and offsets[idx][1] >= answer_end_char
                    ):
                        end_pos = idx
                        break
                if start_pos == cls_index or end_pos == cls_index:
                    start_pos = cls_index
                    end_pos = cls_index

        features.append(
            TokenizedFeature(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_pos,
                end_positions=end_pos,
                offset_mapping=offsets,
                example_index=sample_idx,
            )
        )

    return features


def prepare_eval_features(
    examples: List[QAExample],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    doc_stride: int,
) -> List[TokenizedFeature]:
    """Tokenize evaluation examples while keeping alignment metadata."""

    questions = ["" for _ in examples]
    contexts = [ex.context for ex in examples]

    tokenized = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    features: List[TokenizedFeature] = []
    for feature_idx, offsets in enumerate(offset_mapping):
        input_ids = tokenized["input_ids"][feature_idx]
        attention_mask = tokenized["attention_mask"][feature_idx]
        token_type_ids = tokenized.get("token_type_ids", None)
        if token_type_ids is not None:
            token_type_ids = token_type_ids[feature_idx]

        features.append(
            TokenizedFeature(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=None,
                end_positions=None,
                offset_mapping=offsets,
                example_index=sample_mapping[feature_idx],
            )
        )

    return features


class TrainQADataset(Dataset):
    """Tensor-ready dataset for QA fine-tuning."""

    def __init__(self, features: List[TokenizedFeature]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        feat = self.features[idx]
        item = {
            "input_ids": torch.tensor(feat.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(feat.attention_mask, dtype=torch.long),
            "start_positions": torch.tensor(feat.start_positions, dtype=torch.long),
            "end_positions": torch.tensor(feat.end_positions, dtype=torch.long),
        }
        if feat.token_type_ids is not None:
            item["token_type_ids"] = torch.tensor(feat.token_type_ids, dtype=torch.long)
        return item


class EvalQADataset(Dataset):
    """Dataset preserving offset mappings for evaluation."""

    def __init__(self, features: List[TokenizedFeature]):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        feat = self.features[idx]
        item: Dict[str, object] = {
            "input_ids": torch.tensor(feat.input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(feat.attention_mask, dtype=torch.long),
            "offset_mapping": torch.tensor(feat.offset_mapping, dtype=torch.long),
            "example_index": torch.tensor(feat.example_index, dtype=torch.long),
        }
        if feat.token_type_ids is not None:
            item["token_type_ids"] = torch.tensor(feat.token_type_ids, dtype=torch.long)
        return item


def eval_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Custom collation that keeps alignment metadata as python lists."""

    tensor_keys = {"input_ids", "attention_mask", "token_type_ids", "example_index"}
    output: Dict[str, object] = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch if key in item]
        if key in tensor_keys and values:
            output[key] = torch.stack(values)
        elif key == "offset_mapping":
            output[key] = [v.tolist() for v in values]
    return output


__all__ = [
    "TokenizedFeature",
    "prepare_train_features",
    "prepare_eval_features",
    "TrainQADataset",
    "EvalQADataset",
    "eval_collate_fn",
]
