# File: src/data/alignment.py
"""Utilities for aligning character spans with tokenized sequences."""

import torch
from transformers import PreTrainedTokenizer
from typing import List, Tuple, Optional, Dict, Any
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


_TOKENIZER_REGISTRY: Dict[str, PreTrainedTokenizer] = {}


def register_tokenizer_for_alignment(tokenizer: PreTrainedTokenizer) -> str:
    """Register a tokenizer instance for alignment caching and return key."""
    cache_key = getattr(tokenizer, "name_or_path", None) or f"tokenizer_{id(tokenizer)}"
    _TOKENIZER_REGISTRY.setdefault(cache_key, tokenizer)
    return cache_key


@lru_cache(maxsize=10000)
def get_alignment_info_cached(
    tokenizer_key: str,
    criterion_text: str,
    document_text: str,
    max_length: int,
    doc_stride: int,
) -> List[Dict[str, Any]]:
    """Cached wrapper around :func:`get_alignment_info`."""
    try:
        tokenizer = _TOKENIZER_REGISTRY[tokenizer_key]
    except KeyError as exc:
        raise KeyError(
            "Tokenizer key not registered. Call register_tokenizer_for_alignment() "
            "before invoking get_alignment_info_cached."
        ) from exc

    return get_alignment_info(tokenizer, criterion_text, document_text, max_length, doc_stride)


def get_alignment_info(
    tokenizer: PreTrainedTokenizer,
    criterion_text: str,
    document_text: str,
    max_length: int,
    doc_stride: int = 128,
) -> List[Dict[str, Any]]:
    """Get alignment information for criterion-document pairs with sliding windows.

    Args:
        tokenizer: HuggingFace tokenizer
        criterion_text: Criterion text
        document_text: Document text
        max_length: Maximum sequence length
        doc_stride: Stride for sliding windows over document

    Returns:
        List of window dictionaries with alignment info
    """
    # Tokenize criterion separately to get its length
    criterion_tokens = tokenizer(
        criterion_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )
    criterion_length = len(criterion_tokens["input_ids"])

    # Account for [CLS], criterion, [SEP], and final [SEP]
    available_doc_length = max_length - criterion_length - 3

    if available_doc_length <= 0:
        raise ValueError(
            f"Criterion text too long ({criterion_length} tokens). "
            f"Max length {max_length} leaves no room for document."
        )

    # Tokenize document to plan windows
    doc_tokens = tokenizer(
        document_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        return_attention_mask=False,
    )

    doc_input_ids = doc_tokens["input_ids"]
    doc_offset_mapping = doc_tokens["offset_mapping"]

    windows = []
    start_idx = 0

    while start_idx < len(doc_input_ids):
        # Determine end index for this window
        end_idx = min(start_idx + available_doc_length, len(doc_input_ids))

        # Get document slice for this window
        window_doc_ids = doc_input_ids[start_idx:end_idx]
        window_doc_offsets = doc_offset_mapping[start_idx:end_idx]

        # Create full sequence: [CLS] criterion [SEP] document_window [SEP]
        input_ids = (
            [tokenizer.cls_token_id] +
            criterion_tokens["input_ids"] +
            [tokenizer.sep_token_id] +
            window_doc_ids +
            [tokenizer.sep_token_id]
        )

        # Create token type IDs (0 for criterion, 1 for document)
        token_type_ids = (
            [0] * (1 + criterion_length + 1) +  # [CLS] + criterion + [SEP]
            [1] * (len(window_doc_ids) + 1)     # document + [SEP]
        )

        # Create attention mask
        attention_mask = [1] * len(input_ids)

        # Create masks for different token types
        criterion_mask = [False] * len(input_ids)
        text_mask = [False] * len(input_ids)

        # Mark criterion tokens (excluding [CLS] and first [SEP])
        for i in range(1, 1 + criterion_length):
            criterion_mask[i] = True

        # Mark document tokens (excluding final [SEP])
        doc_start_pos = 1 + criterion_length + 1
        for i in range(doc_start_pos, doc_start_pos + len(window_doc_ids)):
            text_mask[i] = True

        # Create offset mapping for the full sequence
        offset_mapping = []

        # [CLS] token
        offset_mapping.append((0, 0))

        # Criterion tokens
        for start, end in criterion_tokens["offset_mapping"]:
            offset_mapping.append((start, end))

        # First [SEP]
        offset_mapping.append((len(criterion_text), len(criterion_text)))

        # Document tokens (adjust offsets to be relative to document_text)
        for start, end in window_doc_offsets:
            offset_mapping.append((start, end))

        # Final [SEP]
        doc_end = doc_offset_mapping[-1][1] if doc_offset_mapping else 0
        offset_mapping.append((doc_end, doc_end))

        # Store window information
        window_info = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "criterion_mask": criterion_mask,
            "text_mask": text_mask,
            "offset_mapping": offset_mapping,
            "doc_start_char": doc_offset_mapping[0][0] if doc_offset_mapping else 0,
            "doc_end_char": doc_offset_mapping[-1][1] if doc_offset_mapping else 0,
            "doc_start_token": start_idx,
            "doc_end_token": end_idx,
            "window_doc_token_to_char": window_doc_offsets,
        }

        windows.append(window_info)

        # Move to next window
        if end_idx >= len(doc_input_ids):
            break
        start_idx = max(start_idx + doc_stride, end_idx - available_doc_length // 2)

    return windows


def char_spans_to_token_spans(
    char_spans: List[Tuple[int, int]],
    window_info: Dict[str, Any],
) -> List[Tuple[int, int]]:
    """Convert character spans to token spans within a window.

    Args:
        char_spans: List of (start_char, end_char) in document text
        window_info: Window alignment information

    Returns:
        List of (start_token, end_token) positions in the tokenized sequence.
        Returns (-1, -1) for spans not in this window.
    """
    token_spans = []
    offset_mapping = window_info["offset_mapping"]
    text_mask = window_info["text_mask"]

    for char_start, char_end in char_spans:
        # Find token positions that correspond to character span
        start_token = -1
        end_token = -1

        for i, (token_char_start, token_char_end) in enumerate(offset_mapping):
            if not text_mask[i]:  # Skip non-document tokens
                continue

            # Check if this token overlaps with the character span
            if token_char_start <= char_start < token_char_end:
                start_token = i
            if token_char_start < char_end <= token_char_end:
                end_token = i

        # Handle edge cases
        if start_token == -1 or end_token == -1:
            # Span not found in this window
            token_spans.append((-1, -1))
        else:
            token_spans.append((start_token, end_token))

    return token_spans


def token_spans_to_char_spans(
    token_spans: List[Tuple[int, int]],
    window_info: Dict[str, Any],
) -> List[Tuple[int, int]]:
    """Convert token spans back to character spans.

    Args:
        token_spans: List of (start_token, end_token) positions
        window_info: Window alignment information

    Returns:
        List of (start_char, end_char) in document text
    """
    char_spans = []
    offset_mapping = window_info["offset_mapping"]

    for start_token, end_token in token_spans:
        if start_token == -1 or end_token == -1:
            char_spans.append((-1, -1))
            continue

        if (start_token >= len(offset_mapping) or
            end_token >= len(offset_mapping)):
            char_spans.append((-1, -1))
            continue

        start_char = offset_mapping[start_token][0]
        end_char = offset_mapping[end_token][1]
        char_spans.append((start_char, end_char))

    return char_spans


def find_best_span_in_window(
    char_spans: List[Tuple[int, int]],
    window_info: Dict[str, Any],
) -> Tuple[int, int]:
    """Find the best character span that fits in this window.

    Args:
        char_spans: List of character spans
        window_info: Window alignment information

    Returns:
        (start_token, end_token) of best span, or (-1, -1) if none fit
    """
    if not char_spans:
        return (-1, -1)

    token_spans = char_spans_to_token_spans(char_spans, window_info)
    valid_spans = [(s, e) for s, e in token_spans if s != -1 and e != -1]

    if not valid_spans:
        return (-1, -1)

    # Return the shortest valid span (prefer precision)
    best_span = min(valid_spans, key=lambda x: x[1] - x[0])
    return best_span


def validate_alignment(
    original_text: str,
    char_span: Tuple[int, int],
    window_info: Dict[str, Any],
    tokenizer: PreTrainedTokenizer,
) -> bool:
    """Validate that character span alignment is correct.

    Args:
        original_text: Original document text
        char_span: Character span to validate
        window_info: Window alignment information
        tokenizer: Tokenizer used

    Returns:
        True if alignment is valid
    """
    start_char, end_char = char_span

    if start_char < 0 or end_char < 0 or start_char >= end_char:
        return False

    if end_char > len(original_text):
        return False

    # Convert to token span and back
    token_spans = char_spans_to_token_spans([char_span], window_info)
    if not token_spans or token_spans[0] == (-1, -1):
        return False

    recovered_char_spans = token_spans_to_char_spans(token_spans, window_info)
    if not recovered_char_spans:
        return False

    recovered_start, recovered_end = recovered_char_spans[0]

    # Allow some tolerance for subword tokenization
    return (abs(recovered_start - start_char) <= 2 and
            abs(recovered_end - end_char) <= 2)