from __future__ import annotations

from typing import Iterable, List

import torch


class DynamicPaddingCollator:
    def __init__(self, pad_token_id: int, max_length: int):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def _pad(self, sequences: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = min(
            max(seq.size(0) for seq in sequences),
            self.max_length,
        )
        padded = torch.full((len(sequences), max_len), pad_value, dtype=sequences[0].dtype)
        for idx, seq in enumerate(sequences):
            length = min(seq.size(0), max_len)
            padded[idx, :length] = seq[:length]
        return padded

    def __call__(self, batch: Iterable[dict]) -> dict:
        batch = list(batch)
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        criteria = torch.tensor([item["criteria_label"] for item in batch], dtype=torch.long)
        evidence = torch.tensor([item["evidence_label"] for item in batch], dtype=torch.long)

        input_ids = self._pad(input_ids, self.pad_token_id)
        attention_mask = self._pad(attention_mask, 0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "criteria_label": criteria,
            "evidence_label": evidence,
        }
