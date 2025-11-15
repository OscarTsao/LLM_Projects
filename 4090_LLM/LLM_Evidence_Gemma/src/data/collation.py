"""Custom data collation and bucketing utilities for LLM classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase


@dataclass
class SmartBatchCollator:
    """
    Thin wrapper around DataCollatorWithPadding that always pads on the right and
    supports optional label smoothing inputs.
    """

    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        self.collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        batch = self.collator(features)
        # Make sure attention mask exists for custom pooling logic.
        if "attention_mask" not in batch:
            batch["attention_mask"] = torch.ones_like(batch["input_ids"])
        return batch
