# File: tests/test_alignment.py
"""Tests for character-token alignment utilities."""

import pytest
from transformers import AutoTokenizer

from src.data.alignment import (
    get_alignment_info,
    char_spans_to_token_spans,
    token_spans_to_char_spans,
    find_best_span_in_window,
    validate_alignment,
)


@pytest.fixture
def tokenizer():
    """Get a test tokenizer."""
    return AutoTokenizer.from_pretrained("bert-base-uncased")


class TestAlignment:
    """Test character-token alignment functionality."""

    def test_basic_alignment(self, tokenizer):
        """Test basic alignment with simple text."""
        criterion = "test criterion"
        document = "This is a test document with some words."

        windows = get_alignment_info(
            tokenizer, criterion, document, max_length=64, doc_stride=32
        )

        assert len(windows) >= 1
        window = windows[0]

        # Check required fields
        assert "input_ids" in window
        assert "attention_mask" in window
        assert "token_type_ids" in window
        assert "text_mask" in window
        assert "offset_mapping" in window

        # Check sequence structure
        input_ids = window["input_ids"]
        token_type_ids = window["token_type_ids"]
        text_mask = window["text_mask"]

        # Should start with [CLS]
        assert input_ids[0] == tokenizer.cls_token_id

        # Check token type IDs (0 for criterion, 1 for document)
        criterion_tokens = sum(1 for i, tid in enumerate(token_type_ids) if tid == 0 and i > 0)
        doc_tokens = sum(1 for tid in token_type_ids if tid == 1)

        assert criterion_tokens > 0  # Should have criterion tokens
        assert doc_tokens > 0  # Should have document tokens

        # Text mask should only be True for document tokens
        text_token_count = sum(text_mask)
        assert text_token_count > 0

    def test_char_to_token_spans(self, tokenizer):
        """Test character span to token span conversion."""
        criterion = "test"
        document = "Hello world test document"
        char_spans = [(6, 11)]  # "world"

        windows = get_alignment_info(tokenizer, criterion, document, max_length=64)
        window = windows[0]

        token_spans = char_spans_to_token_spans(char_spans, window)
        assert len(token_spans) == 1

        start_token, end_token = token_spans[0]
        assert start_token != -1
        assert end_token != -1
        assert start_token <= end_token

    def test_token_to_char_spans(self, tokenizer):
        """Test token span to character span conversion."""
        criterion = "test"
        document = "Hello world"

        windows = get_alignment_info(tokenizer, criterion, document, max_length=64)
        window = windows[0]

        # Find document token positions
        text_mask = window["text_mask"]
        doc_positions = [i for i, mask in enumerate(text_mask) if mask]

        if len(doc_positions) >= 2:
            token_spans = [(doc_positions[0], doc_positions[1])]
            char_spans = token_spans_to_char_spans(token_spans, window)

            assert len(char_spans) == 1
            start_char, end_char = char_spans[0]
            assert start_char != -1
            assert end_char != -1
            assert start_char < end_char
            assert end_char <= len(document)

    def test_round_trip_alignment(self, tokenizer):
        """Test round-trip conversion: char -> token -> char."""
        criterion = "criterion text"
        document = "The patient shows symptoms of depression and anxiety."
        char_spans = [(4, 11), (18, 26)]  # "patient", "symptoms"

        windows = get_alignment_info(tokenizer, criterion, document, max_length=64)
        window = windows[0]

        # Convert char to token
        token_spans = char_spans_to_token_spans(char_spans, window)

        # Convert back to char
        recovered_char_spans = token_spans_to_char_spans(token_spans, window)

        # Check that we get reasonable results
        for original, recovered in zip(char_spans, recovered_char_spans):
            if recovered != (-1, -1):
                # Allow some tolerance for subword tokenization
                assert abs(original[0] - recovered[0]) <= 5
                assert abs(original[1] - recovered[1]) <= 5

    def test_find_best_span_in_window(self, tokenizer):
        """Test finding best span that fits in window."""
        criterion = "test criterion"
        document = "This is a test document."
        char_spans = [(10, 14)]  # "test"

        windows = get_alignment_info(tokenizer, criterion, document, max_length=64)
        window = windows[0]

        best_span = find_best_span_in_window(char_spans, window)

        # Should find a valid span
        assert best_span != (-1, -1)
        start_token, end_token = best_span
        assert start_token <= end_token

    def test_multiple_windows(self, tokenizer):
        """Test sliding windows with long document."""
        criterion = "short criterion"
        document = " ".join(["word"] * 100)  # Long document

        windows = get_alignment_info(
            tokenizer, criterion, document, max_length=64, doc_stride=16
        )

        # Should create multiple windows
        assert len(windows) > 1

        # Each window should have the same criterion part
        for window in windows:
            token_type_ids = window["token_type_ids"]
            criterion_part = [i for i, tid in enumerate(token_type_ids) if tid == 0]
            assert len(criterion_part) > 1  # [CLS] + criterion tokens

    def test_empty_spans(self, tokenizer):
        """Test handling of empty span lists."""
        criterion = "test"
        document = "test document"
        char_spans = []

        windows = get_alignment_info(tokenizer, criterion, document, max_length=64)
        window = windows[0]

        token_spans = char_spans_to_token_spans(char_spans, window)
        assert token_spans == []

        best_span = find_best_span_in_window(char_spans, window)
        assert best_span == (-1, -1)

    def test_out_of_bounds_spans(self, tokenizer):
        """Test handling of spans that are out of bounds."""
        criterion = "test"
        document = "short doc"
        char_spans = [(100, 110)]  # Out of bounds

        windows = get_alignment_info(tokenizer, criterion, document, max_length=64)
        window = windows[0]

        token_spans = char_spans_to_token_spans(char_spans, window)
        assert len(token_spans) == 1
        assert token_spans[0] == (-1, -1)

    def test_unicode_text(self, tokenizer):
        """Test alignment with Unicode characters."""
        criterion = "test with émojis 😀"
        document = "This tëxt has ünïcödë characters and émojis 🎉"
        char_spans = [(5, 9)]  # "tëxt"

        windows = get_alignment_info(tokenizer, criterion, document, max_length=64)
        window = windows[0]

        token_spans = char_spans_to_token_spans(char_spans, window)
        assert len(token_spans) == 1

        # Should handle Unicode properly
        if token_spans[0] != (-1, -1):
            recovered_spans = token_spans_to_char_spans(token_spans, window)
            assert recovered_spans[0] != (-1, -1)

    def test_validate_alignment(self, tokenizer):
        """Test alignment validation."""
        criterion = "test"
        document = "This is a test document"

        windows = get_alignment_info(tokenizer, criterion, document, max_length=64)
        window = windows[0]

        # Valid span
        valid_span = (10, 14)  # "test"
        assert validate_alignment(document, valid_span, window, tokenizer)

        # Invalid spans
        invalid_spans = [
            (-1, 5),  # Negative start
            (5, 100),  # End beyond document
            (10, 5),  # Start > end
        ]

        for span in invalid_spans:
            assert not validate_alignment(document, span, window, tokenizer)

    def test_long_criterion(self, tokenizer):
        """Test handling of very long criterion text."""
        criterion = " ".join(["criterion"] * 50)  # Very long criterion
        document = "Short document"

        # Should raise error if criterion is too long
        with pytest.raises(ValueError, match="Criterion text too long"):
            get_alignment_info(tokenizer, criterion, document, max_length=64)

    def test_punctuation_handling(self, tokenizer):
        """Test alignment with punctuation."""
        criterion = "test criterion"
        document = "Hello, world! This is a test. How are you?"
        char_spans = [(26, 30)]  # "test"

        windows = get_alignment_info(tokenizer, criterion, document, max_length=64)
        window = windows[0]

        token_spans = char_spans_to_token_spans(char_spans, window)

        # Should handle punctuation correctly
        assert len(token_spans) == 1
        if token_spans[0] != (-1, -1):
            recovered_spans = token_spans_to_char_spans(token_spans, window)
            assert recovered_spans[0] != (-1, -1)