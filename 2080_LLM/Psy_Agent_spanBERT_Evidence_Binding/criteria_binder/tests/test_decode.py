# File: tests/test_decode.py
"""Tests for span decoding utilities."""

import pytest
import torch
import numpy as np

from src.utils.decode import (
    decode_spans_from_logits,
    decode_single_example_spans,
    aggregate_spans_across_windows,
    apply_nms_to_spans,
    compute_span_iou,
    SpanDecoder,
)


class TestSpanDecoding:
    """Test span decoding functionality."""

    def test_compute_span_iou(self):
        """Test IoU computation between spans."""
        # Identical spans
        assert compute_span_iou((5, 10), (5, 10)) == 1.0

        # No overlap
        assert compute_span_iou((0, 5), (10, 15)) == 0.0

        # Partial overlap
        iou = compute_span_iou((0, 10), (5, 15))
        expected = 5 / 15  # intersection=5, union=15
        assert abs(iou - expected) < 1e-6

        # One span contains another
        iou = compute_span_iou((0, 10), (2, 8))
        expected = 6 / 10  # intersection=6, union=10
        assert abs(iou - expected) < 1e-6

        # Adjacent spans
        assert compute_span_iou((0, 5), (5, 10)) == 0.0

        # Single point spans
        assert compute_span_iou((5, 5), (5, 5)) == 1.0

    def test_apply_nms_to_spans(self):
        """Test Non-Maximum Suppression on spans."""
        spans = [
            {"start_char": 0, "end_char": 10, "score": 0.9},
            {"start_char": 5, "end_char": 15, "score": 0.8},  # Overlaps with first
            {"start_char": 20, "end_char": 30, "score": 0.7},  # No overlap
            {"start_char": 25, "end_char": 35, "score": 0.6},  # Overlaps with third
        ]

        # High IoU threshold - should keep all
        filtered = apply_nms_to_spans(spans, iou_thresh=0.9)
        assert len(filtered) == 4

        # Low IoU threshold - should remove overlapping spans
        filtered = apply_nms_to_spans(spans, iou_thresh=0.1)
        assert len(filtered) == 2  # Should keep highest scoring from each group

        # Check that highest scoring spans are kept
        scores = [span["score"] for span in filtered]
        assert 0.9 in scores  # Highest overall
        assert 0.7 in scores  # Highest in second group

    def test_decode_single_example_spans(self):
        """Test span decoding for a single example."""
        seq_len = 20
        start_logits = torch.randn(seq_len)
        end_logits = torch.randn(seq_len)

        # Create text mask (positions 5-15 are document tokens)
        text_mask = torch.zeros(seq_len, dtype=torch.bool)
        text_mask[5:16] = True

        spans = decode_single_example_spans(
            start_logits, end_logits, text_mask, max_answer_len=5, top_k=3
        )

        # Should return at most top_k spans
        assert len(spans) <= 3

        # All spans should be within text mask
        for span in spans:
            start = span["start_token"]
            end = span["end_token"]
            assert text_mask[start].item()
            assert text_mask[end].item()
            assert start <= end

        # All spans should respect max_answer_len
        for span in spans:
            length = span["end_token"] - span["start_token"] + 1
            assert length <= 5

    def test_decode_spans_from_logits_batch(self):
        """Test batch span decoding."""
        batch_size = 3
        seq_len = 15

        start_logits = torch.randn(batch_size, seq_len)
        end_logits = torch.randn(batch_size, seq_len)

        # Create text masks
        text_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        text_mask[:, 5:12] = True  # Document tokens

        batch_spans = decode_spans_from_logits(
            start_logits, end_logits, text_mask, max_answer_len=4, top_k=2
        )

        assert len(batch_spans) == batch_size

        for spans in batch_spans:
            assert len(spans) <= 2  # Respects top_k
            for span in spans:
                assert "start_token" in span
                assert "end_token" in span
                assert "score" in span

    def test_span_decoder_class(self):
        """Test SpanDecoder class functionality."""
        decoder = SpanDecoder(
            max_answer_len=6,
            top_k=3,
            nms_iou_thresh=0.5,
            allow_overlap=False,
        )

        # Test single example decoding
        seq_len = 20
        start_logits = torch.randn(seq_len)
        end_logits = torch.randn(seq_len)
        text_mask = torch.zeros(seq_len, dtype=torch.bool)
        text_mask[8:18] = True

        metadata = {
            "offset_mapping": [(i, i+1) for i in range(seq_len)],
            "document_text": "This is a test document with some words.",
        }

        spans = decoder.decode_single_example(
            start_logits, end_logits, text_mask, metadata
        )

        # Should return character-level spans
        for span in spans:
            assert "start_char" in span
            assert "end_char" in span
            assert "score" in span
            assert "text" in span

    def test_aggregate_spans_across_windows(self):
        """Test span aggregation across multiple windows."""
        # Simulate spans from 2 windows
        window_spans = [
            [  # Window 1
                {"start_token": 5, "end_token": 8, "score": 0.9},
                {"start_token": 10, "end_token": 12, "score": 0.7},
            ],
            [  # Window 2
                {"start_token": 3, "end_token": 5, "score": 0.8},  # Different position but same text
                {"start_token": 15, "end_token": 18, "score": 0.6},
            ],
        ]

        # Metadata for windows
        window_metadata = [
            {  # Window 1
                "offset_mapping": [(i, i+1) for i in range(20)],
                "document_text": "This is a test document",
            },
            {  # Window 2 (overlapping)
                "offset_mapping": [(i+10, i+11) for i in range(20)],
                "document_text": "This is a test document",
            },
        ]

        document_text = "This is a test document"

        aggregated = aggregate_spans_across_windows(
            window_spans,
            window_metadata,
            document_text,
            top_k=3,
            nms_iou_thresh=0.5,
        )

        # Should have aggregated and filtered spans
        assert len(aggregated) <= 3
        assert len(aggregated) > 0

        # All spans should be valid
        for span in aggregated:
            assert span["start_char"] >= 0
            assert span["end_char"] <= len(document_text)
            assert span["start_char"] < span["end_char"]

    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        # Empty logits
        empty_logits = torch.empty(0)
        empty_mask = torch.empty(0, dtype=torch.bool)

        spans = decode_single_example_spans(
            empty_logits, empty_logits, empty_mask
        )
        assert spans == []

        # No valid document tokens
        seq_len = 10
        start_logits = torch.randn(seq_len)
        end_logits = torch.randn(seq_len)
        text_mask = torch.zeros(seq_len, dtype=torch.bool)  # All False

        spans = decode_single_example_spans(
            start_logits, end_logits, text_mask
        )
        assert spans == []

    def test_score_threshold(self):
        """Test minimum score threshold filtering."""
        seq_len = 10
        start_logits = torch.ones(seq_len) * -10  # Very low scores
        end_logits = torch.ones(seq_len) * -10

        text_mask = torch.ones(seq_len, dtype=torch.bool)

        # With very high threshold, should get no spans
        spans = decode_single_example_spans(
            start_logits, end_logits, text_mask, min_score_threshold=0.0
        )
        assert len(spans) == 0

        # With very low threshold, should get some spans
        spans = decode_single_example_spans(
            start_logits, end_logits, text_mask, min_score_threshold=-100.0
        )
        assert len(spans) > 0

    def test_max_answer_length_constraint(self):
        """Test maximum answer length constraint."""
        seq_len = 20
        # Create high scores for long span
        start_logits = torch.full((seq_len,), -10.0)
        end_logits = torch.full((seq_len,), -10.0)

        # High scores for position 5 (start) and position 15 (end)
        start_logits[5] = 10.0
        end_logits[15] = 10.0

        text_mask = torch.ones(seq_len, dtype=torch.bool)

        # With max_answer_len=5, should not find the long span
        spans = decode_single_example_spans(
            start_logits, end_logits, text_mask, max_answer_len=5
        )

        # Should either find no spans or shorter spans
        for span in spans:
            length = span["end_token"] - span["start_token"] + 1
            assert length <= 5

    def test_span_sorting(self):
        """Test that spans are sorted by score."""
        seq_len = 15
        start_logits = torch.randn(seq_len)
        end_logits = torch.randn(seq_len)
        text_mask = torch.ones(seq_len, dtype=torch.bool)

        spans = decode_single_example_spans(
            start_logits, end_logits, text_mask, top_k=5
        )

        # Should be sorted by score (descending)
        if len(spans) > 1:
            scores = [span["score"] for span in spans]
            assert scores == sorted(scores, reverse=True)

    def test_valid_span_constraints(self):
        """Test that only valid spans (start <= end) are returned."""
        seq_len = 10
        start_logits = torch.randn(seq_len)
        end_logits = torch.randn(seq_len)
        text_mask = torch.ones(seq_len, dtype=torch.bool)

        spans = decode_single_example_spans(
            start_logits, end_logits, text_mask
        )

        for span in spans:
            assert span["start_token"] <= span["end_token"]