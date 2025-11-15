"""
Tokenizer utilities for classification tasks.

Enforces right-padding, guarantees pad tokens, and exposes helpers for
pair-input formatting along with optional padding multiples for
FlashAttention-friendly shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def build_tokenizer(model_id: str, right_pad: bool = True) -> PreTrainedTokenizerBase:
    """
    Create a tokenizer configured for right padding with a guaranteed pad token.
    """
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if right_pad:
        tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def apply_pair_format(
    examples: dict,
    tokenizer: PreTrainedTokenizerBase,
    text_column: str,
    second_text_column: Optional[str],
    max_length: int,
    pad_to_multiple_of: Optional[int] = None,
) -> dict:
    """
    Tokenize single or pair inputs with right padding enforced.
    """
    texts: List[str] = examples[text_column]
    second_texts: Optional[List[str]] = None
    if second_text_column:
        second_texts = examples[second_text_column]

    tokenized = tokenizer(
        texts,
        second_texts,
        truncation=True,
        max_length=max_length,
        padding=False,  # handled downstream by smart collator
    )

    if pad_to_multiple_of is not None:
        tokenized = tokenizer.pad(
            tokenized,
            padding=False,
            pad_to_multiple_of=pad_to_multiple_of,
        )

    return tokenized


@dataclass
class LengthBucket:
    """
    Simple helper to place lengths into coarse buckets for logging/sanity.
    """

    boundaries: Tuple[int, ...] = (64, 128, 256, 512, 1024, 2048)

    def __call__(self, length: int) -> str:
        for boundary in self.boundaries:
            if length <= boundary:
                return f"<= {boundary}"
        return f">{self.boundaries[-1]}"
