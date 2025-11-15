# File: src/utils/decode.py
"""Span decoding utilities with NMS and multi-window aggregation."""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict, Any, Optional
import logging
import numpy as np

from ..data.alignment import token_spans_to_char_spans

logger = logging.getLogger(__name__)


def decode_spans_from_logits(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    text_mask: torch.Tensor,
    max_answer_len: int = 64,
    top_k: int = 5,
    min_score_threshold: float = -float('inf'),
) -> List[List[Dict[str, Any]]]:
    """Decode spans from start/end logits for a batch.

    Args:
        start_logits: Start position logits [B, T]
        end_logits: End position logits [B, T]
        text_mask: Document token mask [B, T]
        max_answer_len: Maximum span length
        top_k: Number of top spans to return per example
        min_score_threshold: Minimum score threshold for spans

    Returns:
        List of lists of span dictionaries, one list per batch item
    """
    batch_size = start_logits.shape[0]
    batch_spans = []

    for b in range(batch_size):
        spans = decode_single_example_spans(
            start_logits[b],
            end_logits[b],
            text_mask[b],
            max_answer_len=max_answer_len,
            top_k=top_k,
            min_score_threshold=min_score_threshold,
        )
        batch_spans.append(spans)

    return batch_spans


def decode_single_example_spans(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    text_mask: torch.Tensor,
    max_answer_len: int = 64,
    top_k: int = 5,
    min_score_threshold: float = -float('inf'),
) -> List[Dict[str, Any]]:
    """Decode spans for a single example.

    Args:
        start_logits: Start position logits [T]
        end_logits: End position logits [T]
        text_mask: Document token mask [T]
        max_answer_len: Maximum span length
        top_k: Number of top spans to return
        min_score_threshold: Minimum score threshold

    Returns:
        List of span dictionaries with token positions and scores
    """
    # Get valid document positions
    valid_positions = text_mask.nonzero(as_tuple=False).squeeze(-1)

    if len(valid_positions) == 0:
        return []

    # Convert logits to scores (can use logits directly or softmax)
    start_scores = start_logits
    end_scores = end_logits

    # Find all valid spans
    candidate_spans = []

    for start_idx in valid_positions:
        for end_idx in valid_positions:
            if (start_idx <= end_idx and
                end_idx - start_idx + 1 <= max_answer_len):

                score = start_scores[start_idx] + end_scores[end_idx]
                if score >= min_score_threshold:
                    candidate_spans.append({
                        'start_token': start_idx.item(),
                        'end_token': end_idx.item(),
                        'score': score.item(),
                        'length': (end_idx - start_idx + 1).item(),
                    })

    # Sort by score and take top-k
    candidate_spans.sort(key=lambda x: x['score'], reverse=True)
    return candidate_spans[:top_k]


def aggregate_spans_across_windows(
    window_spans: List[List[Dict[str, Any]]],
    window_metadata: List[Dict[str, Any]],
    document_text: str,
    top_k: int = 5,
    nms_iou_thresh: float = 0.5,
    allow_overlap: bool = False,
) -> List[Dict[str, Any]]:
    """Aggregate spans across multiple windows with NMS.

    Args:
        window_spans: List of span lists from each window
        window_metadata: Metadata for each window (offset mapping, etc.)
        document_text: Original document text for validation
        top_k: Final number of spans to return
        nms_iou_thresh: IoU threshold for NMS
        allow_overlap: Whether to allow overlapping spans

    Returns:
        List of final span predictions with character positions
    """
    if not window_spans or not window_metadata:
        return []

    # Convert all spans to character coordinates
    all_char_spans = []

    for window_idx, (spans, metadata) in enumerate(zip(window_spans, window_metadata)):
        for span in spans:
            # Convert token positions to character positions
            token_spans = [(span['start_token'], span['end_token'])]
            char_spans = token_spans_to_char_spans(token_spans, metadata)

            if char_spans and char_spans[0] != (-1, -1):
                start_char, end_char = char_spans[0]

                # Validate span
                if (start_char >= 0 and end_char <= len(document_text) and
                    start_char < end_char):

                    all_char_spans.append({
                        'start_char': start_char,
                        'end_char': end_char,
                        'score': span['score'],
                        'window_idx': window_idx,
                        'text': document_text[start_char:end_char],
                    })

    if not all_char_spans:
        return []

    # Sort by score
    all_char_spans.sort(key=lambda x: x['score'], reverse=True)

    # Apply NMS if not allowing overlap
    if not allow_overlap:
        all_char_spans = apply_nms_to_spans(all_char_spans, nms_iou_thresh)

    # Return top-k
    return all_char_spans[:top_k]


def apply_nms_to_spans(
    spans: List[Dict[str, Any]],
    iou_thresh: float = 0.5,
) -> List[Dict[str, Any]]:
    """Apply Non-Maximum Suppression to remove overlapping spans.

    Args:
        spans: List of span dictionaries with char positions and scores
        iou_thresh: IoU threshold for suppression

    Returns:
        Filtered list of spans
    """
    if not spans:
        return []

    # Sort by score (descending)
    spans = sorted(spans, key=lambda x: x['score'], reverse=True)

    kept_spans = []

    for span in spans:
        # Check overlap with already kept spans
        should_keep = True

        for kept_span in kept_spans:
            iou = compute_span_iou(
                (span['start_char'], span['end_char']),
                (kept_span['start_char'], kept_span['end_char'])
            )

            if iou > iou_thresh:
                should_keep = False
                break

        if should_keep:
            kept_spans.append(span)

    return kept_spans


def compute_span_iou(span1: Tuple[int, int], span2: Tuple[int, int]) -> float:
    """Compute IoU between two character spans.

    Args:
        span1: (start, end) of first span
        span2: (start, end) of second span

    Returns:
        IoU value between 0 and 1
    """
    start1, end1 = span1
    start2, end2 = span2

    # Compute intersection
    intersection_start = max(start1, start2)
    intersection_end = min(end1, end2)
    intersection_len = max(0, intersection_end - intersection_start)

    # Compute union
    union_len = (end1 - start1) + (end2 - start2) - intersection_len

    if union_len == 0:
        return 1.0 if intersection_len == 0 else 0.0

    return intersection_len / union_len


class SpanDecoder:
    """Span decoder that handles the full pipeline from logits to final spans."""

    def __init__(
        self,
        max_answer_len: int = 64,
        top_k: int = 5,
        nms_iou_thresh: float = 0.5,
        allow_overlap: bool = False,
        min_score_threshold: float = -float('inf'),
    ) -> None:
        """Initialize the span decoder.

        Args:
            max_answer_len: Maximum span length in tokens
            top_k: Number of top spans to return
            nms_iou_thresh: IoU threshold for NMS
            allow_overlap: Whether to allow overlapping spans
            min_score_threshold: Minimum score threshold
        """
        self.max_answer_len = max_answer_len
        self.top_k = top_k
        self.nms_iou_thresh = nms_iou_thresh
        self.allow_overlap = allow_overlap
        self.min_score_threshold = min_score_threshold

    def decode_batch(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        text_masks: torch.Tensor,
        metadata: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Decode spans for a batch, aggregating across windows.

        Args:
            start_logits: Start logits [B, T]
            end_logits: End logits [B, T]
            text_masks: Text masks [B, T]
            metadata: List of metadata for each window

        Returns:
            Dictionary mapping example IDs to final span predictions
        """
        # Decode spans for each window
        window_spans = decode_spans_from_logits(
            start_logits,
            end_logits,
            text_masks,
            max_answer_len=self.max_answer_len,
            top_k=self.top_k * 2,  # Get more candidates per window
            min_score_threshold=self.min_score_threshold,
        )

        # Group windows by example ID
        example_windows = {}
        for window_idx, meta in enumerate(metadata):
            example_id = meta['example_id']
            if example_id not in example_windows:
                example_windows[example_id] = {
                    'spans': [],
                    'metadata': [],
                    'document_text': meta['document_text'],
                }

            example_windows[example_id]['spans'].append(window_spans[window_idx])
            example_windows[example_id]['metadata'].append(meta)

        # Aggregate spans for each example
        final_predictions = {}
        for example_id, data in example_windows.items():
            final_spans = aggregate_spans_across_windows(
                data['spans'],
                data['metadata'],
                data['document_text'],
                top_k=self.top_k,
                nms_iou_thresh=self.nms_iou_thresh,
                allow_overlap=self.allow_overlap,
            )
            final_predictions[example_id] = final_spans

        return final_predictions

    def decode_single_example(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        text_mask: torch.Tensor,
        metadata: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Decode spans for a single window/example.

        Args:
            start_logits: Start logits [T]
            end_logits: End logits [T]
            text_mask: Text mask [T]
            metadata: Window metadata

        Returns:
            List of span predictions
        """
        # Decode token spans
        token_spans = decode_single_example_spans(
            start_logits,
            end_logits,
            text_mask,
            max_answer_len=self.max_answer_len,
            top_k=self.top_k,
            min_score_threshold=self.min_score_threshold,
        )

        # Convert to character spans
        char_spans = []
        document_text = metadata['document_text']

        for span in token_spans:
            token_span_list = [(span['start_token'], span['end_token'])]
            char_span_list = token_spans_to_char_spans(token_span_list, metadata)

            if char_span_list and char_span_list[0] != (-1, -1):
                start_char, end_char = char_span_list[0]

                if (start_char >= 0 and end_char <= len(document_text) and
                    start_char < end_char):

                    char_spans.append({
                        'start_char': start_char,
                        'end_char': end_char,
                        'score': span['score'],
                        'text': document_text[start_char:end_char],
                    })

        return char_spans