import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer


def _load_groundtruth(path: str) -> pd.DataFrame:
    """Load groundtruth JSON lines into a dataframe."""
    records: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return pd.DataFrame(records)


def load_dataset(cfg: DictConfig) -> pd.DataFrame:
    """Load and merge posts with groundtruth annotations."""
    posts_df = pd.read_csv(cfg.posts_path)
    gt_df = _load_groundtruth(cfg.groundtruth_path)

    merged = gt_df.merge(posts_df, on=cfg.id_field, how="left", suffixes=("_gt", "_post"))
    text_field_gt = f"{cfg.text_field}_gt"
    text_field_post = f"{cfg.text_field}_post"
    if text_field_gt in merged.columns:
        merged[cfg.text_field] = merged[text_field_gt].fillna(merged[text_field_post])
        merged.drop(columns=[text_field_gt, text_field_post], inplace=True)
    merged[cfg.text_field] = merged[cfg.text_field].fillna("")
    merged.dropna(subset=[cfg.text_field], inplace=True)
    return merged.reset_index(drop=True)


def train_val_test_split(
    df: pd.DataFrame,
    val_size: float,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train/val/test partitions."""
    from sklearn.model_selection import train_test_split

    remaining_size = 1.0 - test_size
    if remaining_size <= 0:
        raise ValueError("test_size must be less than 1.")

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    val_ratio = val_size / remaining_size
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_ratio,
        random_state=seed,
        shuffle=True,
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def _parse_json_field(value) -> Optional[List]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return None


def _normalise_spans(spans: Optional[List]) -> Optional[List[Tuple[int, int]]]:
    if spans is None:
        return None
    normalised: List[Tuple[int, int]] = []
    for span in spans:
        if isinstance(span, (list, tuple)) and len(span) >= 2:
            start, end = int(span[0]), int(span[1])
            if end > start:
                normalised.append((start, end))
    return normalised if normalised else None


def _normalise_labels(sequence: Optional[List]) -> Optional[List[int]]:
    if sequence is None:
        return None
    labels: List[int] = []
    for value in sequence:
        try:
            labels.append(int(value))
        except (TypeError, ValueError):
            labels.append(0)
    return labels if labels else None


class PostDataset(Dataset):
    """Dataset returning raw text, labels, and optional evidence annotations."""

    def __init__(
        self,
        df: pd.DataFrame,
        id_field: str,
        text_field: str,
        multi_label_fields: Sequence[str],
        evidence_char_spans_field: Optional[str] = None,
        evidence_token_labels_field: Optional[str] = None,
    ) -> None:
        self.ids = df[id_field].tolist()
        self.texts = df[text_field].tolist()
        self.multi_labels = df[multi_label_fields].values.astype(np.float32)
        self.label_fields = list(multi_label_fields)

        self.char_spans: List[Optional[List[Tuple[int, int]]]] = []
        self.token_label_sequences: List[Optional[List[int]]] = []

        for _, row in df.iterrows():
            char_spans = None
            if (
                evidence_char_spans_field
                and evidence_char_spans_field in row
                and not pd.isna(row[evidence_char_spans_field])
            ):
                char_spans = _normalise_spans(_parse_json_field(row[evidence_char_spans_field]))
            self.char_spans.append(char_spans)

            token_labels = None
            if (
                evidence_token_labels_field
                and evidence_token_labels_field in row
                and not pd.isna(row[evidence_token_labels_field])
            ):
                token_labels = _normalise_labels(
                    _parse_json_field(row[evidence_token_labels_field])
                )
            self.token_label_sequences.append(token_labels)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return {
            "post_id": self.ids[idx],
            "text": self.texts[idx],
            "multi_labels": self.multi_labels[idx],
            "char_spans": self.char_spans[idx],
            "token_label_ids": self.token_label_sequences[idx],
        }


class TokenizedDataCollator:
    """Collate function that tokenizes on the fly and returns tensors."""

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        token_head_cfg: Optional[Dict] = None,
        span_head_cfg: Optional[Dict] = None,
        return_tensors: str = "pt",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.return_tensors = return_tensors

        self.token_head_enabled = bool(token_head_cfg and token_head_cfg.get("enabled", True))
        self.token_ignore_index = (
            token_head_cfg.get("ignore_index", -100) if token_head_cfg else -100
        )

        self.span_head_enabled = bool(span_head_cfg and span_head_cfg.get("enabled", True))
        self.span_ignore_index = span_head_cfg.get("ignore_index", -100) if span_head_cfg else -100

        self.need_offsets = self.token_head_enabled or self.span_head_enabled

    @staticmethod
    def _labels_from_spans(
        offsets: List[Tuple[int, int]],
        spans: List[Tuple[int, int]],
        ignore_index: int,
    ) -> torch.Tensor:
        labels = torch.full((len(offsets),), ignore_index, dtype=torch.long)
        if not spans:
            return labels

        for idx, (start, end) in enumerate(offsets):
            if start == end:
                continue
            for span_start, span_end in spans:
                if span_start < end and span_end > start:
                    labels[idx] = 1
                    break
            if labels[idx] == ignore_index:
                labels[idx] = 0
        return labels

    @staticmethod
    def _span_positions_from_offsets(
        offsets: List[Tuple[int, int]],
        spans: List[Tuple[int, int]],
        ignore_index: int,
    ) -> Tuple[int, int]:
        if not spans:
            return ignore_index, ignore_index

        span_start, span_end = spans[0]
        start_index = ignore_index
        end_index = ignore_index
        for idx, (start, end) in enumerate(offsets):
            if start == end:
                continue
            if start_index == ignore_index and span_start >= start and span_start < end:
                start_index = idx
            if span_end > start and span_end <= end:
                end_index = idx
            elif span_end >= end and span_start < end:
                end_index = idx
            if start_index != ignore_index and end_index != ignore_index:
                break
        return start_index, end_index

    def __call__(self, batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
        texts = [sample["text"] for sample in batch]
        post_ids = [sample["post_id"] for sample in batch]
        multi_labels = torch.tensor(
            np.stack([sample["multi_labels"] for sample in batch], axis=0), dtype=torch.float32
        )

        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors=self.return_tensors,
            return_offsets_mapping=self.need_offsets,
        )

        offset_mapping = encodings.pop("offset_mapping", None)
        batch_size, seq_len = encodings["input_ids"].shape

        if self.token_head_enabled:
            token_labels = torch.full(
                (batch_size, seq_len), self.token_ignore_index, dtype=torch.long
            )
        else:
            token_labels = None

        if self.span_head_enabled:
            start_positions = torch.full((batch_size,), self.span_ignore_index, dtype=torch.long)
            end_positions = torch.full((batch_size,), self.span_ignore_index, dtype=torch.long)
        else:
            start_positions = end_positions = None

        if self.need_offsets and offset_mapping is not None:
            for i in range(batch_size):
                spans = batch[i].get("char_spans")
                token_sequence = batch[i].get("token_label_ids")
                offsets = [tuple(pair.tolist()) for pair in offset_mapping[i]]

                if self.token_head_enabled:
                    if token_sequence:
                        seq_tensor = torch.tensor(token_sequence[:seq_len], dtype=torch.long)
                        token_labels[i, : seq_tensor.size(0)] = seq_tensor
                        if seq_tensor.size(0) < seq_len:
                            token_labels[i, seq_tensor.size(0) :] = self.token_ignore_index
                    elif spans:
                        derived = self._labels_from_spans(offsets, spans, self.token_ignore_index)
                        token_labels[i] = derived[:seq_len]

                if self.span_head_enabled:
                    start_idx, end_idx = self._span_positions_from_offsets(
                        offsets, spans or [], self.span_ignore_index
                    )
                    start_positions[i] = start_idx
                    end_positions[i] = end_idx

        collated = dict(encodings)
        collated["multi_labels"] = multi_labels
        collated["post_ids"] = post_ids
        collated["texts"] = texts

        if token_labels is not None:
            collated["token_labels"] = token_labels
        if start_positions is not None and end_positions is not None:
            collated["start_positions"] = start_positions
            collated["end_positions"] = end_positions

        return collated


@dataclass
class DataModule:
    """Simple data module encapsulating dataloader creation."""

    data_cfg: DictConfig
    model_cfg: DictConfig

    def setup(self) -> None:
        tokenizer_kwargs = {"use_fast": True, "padding_side": "right"}
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_cfg.encoder.pretrained_model_name_or_path,
            **tokenizer_kwargs,
        )
        df = load_dataset(self.data_cfg)

        train_df, val_df, test_df = train_val_test_split(
            df,
            val_size=self.data_cfg.val_size,
            test_size=self.data_cfg.test_size,
            seed=self.data_cfg.seed,
        )

        evidence_char_field = self.data_cfg.get("evidence_char_spans_field")
        evidence_token_field = self.data_cfg.get("evidence_token_labels_field")

        self.train_dataset = PostDataset(
            train_df,
            self.data_cfg.id_field,
            self.data_cfg.text_field,
            self.data_cfg.multi_label_fields,
            evidence_char_spans_field=evidence_char_field,
            evidence_token_labels_field=evidence_token_field,
        )
        self.val_dataset = PostDataset(
            val_df,
            self.data_cfg.id_field,
            self.data_cfg.text_field,
            self.data_cfg.multi_label_fields,
            evidence_char_spans_field=evidence_char_field,
            evidence_token_labels_field=evidence_token_field,
        )
        self.test_dataset = PostDataset(
            test_df,
            self.data_cfg.id_field,
            self.data_cfg.text_field,
            self.data_cfg.multi_label_fields,
            evidence_char_spans_field=evidence_char_field,
            evidence_token_labels_field=evidence_token_field,
        )

        token_head_cfg = self.model_cfg.heads.get("evidence_token", {})
        span_head_cfg = self.model_cfg.heads.get("evidence_span", {})

        self.collator = TokenizedDataCollator(
            self.tokenizer,
            max_length=self.data_cfg.max_length,
            token_head_cfg=token_head_cfg if token_head_cfg.get("enabled", False) else None,
            span_head_cfg=span_head_cfg if span_head_cfg.get("enabled", False) else None,
        )

    def dataloaders(
        self,
        batch_size: int,
        val_batch_size: Optional[int] = None,
        test_batch_size: Optional[int] = None,
        num_workers: int = 0,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        if not hasattr(self, "train_dataset"):
            self.setup()

        val_batch_size = val_batch_size or batch_size
        test_batch_size = test_batch_size or val_batch_size

        common_args = {
            "num_workers": num_workers,
            "collate_fn": self.collator,
            "pin_memory": True,
        }
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            **common_args,
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            **common_args,
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            **common_args,
        )
        return train_loader, val_loader, test_loader
