# File: tests/test_collator.py
"""Tests for data collator functionality."""

import pytest
import torch
from transformers import AutoTokenizer

from src.training.collator import CriteriaBindingCollator, InferenceCollator


@pytest.fixture
def tokenizer():
    """Get a test tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@pytest.fixture
def sample_examples():
    """Create sample examples for testing."""
    return [
        {
            "id": "test_001",
            "criterion_text": "Test criterion text",
            "document_text": "This is a test document with some content.",
            "label": 1,
            "evidence_char_spans": [(10, 14)],  # "test"
        },
        {
            "id": "test_002",
            "criterion_text": "Another criterion",
            "document_text": "Short doc.",
            "label": 0,
            "evidence_char_spans": [],
        },
    ]


class TestCriteriaBindingCollator:
    """Test the main training collator."""

    def test_basic_collation(self, tokenizer, sample_examples):
        """Test basic collation functionality."""
        collator = CriteriaBindingCollator(
            tokenizer=tokenizer,
            max_length=128,
            doc_stride=64,
        )

        batch = collator(sample_examples)

        # Check batch structure
        assert isinstance(batch, dict)
        required_keys = [
            "input_ids", "attention_mask", "token_type_ids", "text_mask"
        ]
        for key in required_keys:
            assert key in batch
            assert isinstance(batch[key], torch.Tensor)

        # Check batch dimensions
        batch_size = batch["input_ids"].shape[0]
        seq_len = batch["input_ids"].shape[1]

        assert batch_size >= len(sample_examples)  # May have more due to windows
        assert seq_len <= 128  # Respects max_length

        # Check tensor shapes
        for key in required_keys:
            assert batch[key].shape == (batch_size, seq_len)

        # Check that text_mask is boolean
        assert batch["text_mask"].dtype == torch.bool

    def test_sliding_windows(self, tokenizer):
        """Test sliding window creation for long documents."""
        long_document = " ".join(["word"] * 200)  # Very long document
        examples = [
            {
                "id": "long_test",
                "criterion_text": "Short criterion",
                "document_text": long_document,
                "label": 1,
                "evidence_char_spans": [],
            }
        ]

        collator = CriteriaBindingCollator(
            tokenizer=tokenizer,
            max_length=128,
            doc_stride=32,
        )

        batch = collator(examples)

        # Should create multiple windows
        batch_size = batch["input_ids"].shape[0]
        assert batch_size > 1  # Multiple windows for long document

    def test_padding(self, tokenizer, sample_examples):
        """Test dynamic padding within batch."""
        collator = CriteriaBindingCollator(
            tokenizer=tokenizer,
            max_length=128,
        )

        batch = collator(sample_examples)

        # All sequences should have same length (padded)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        batch_size, seq_len = input_ids.shape

        # Check padding
        for i in range(batch_size):
            # Find actual sequence length
            actual_len = attention_mask[i].sum().item()
            assert actual_len <= seq_len

            # Check that padding tokens are used correctly
            if actual_len < seq_len:
                pad_token_id = tokenizer.pad_token_id
                assert all(input_ids[i, actual_len:] == pad_token_id)

    def test_token_type_ids(self, tokenizer, sample_examples):
        """Test token type ID assignment."""
        collator = CriteriaBindingCollator(tokenizer=tokenizer, max_length=128)
        batch = collator(sample_examples)

        token_type_ids = batch["token_type_ids"]
        batch_size = token_type_ids.shape[0]

        for i in range(batch_size):
            # Should have both segment 0 (criterion) and segment 1 (document)
            unique_segments = torch.unique(token_type_ids[i])
            assert 0 in unique_segments  # Criterion segment
            # May or may not have segment 1 depending on window content

    def test_text_mask(self, tokenizer, sample_examples):
        """Test text mask creation."""
        collator = CriteriaBindingCollator(tokenizer=tokenizer, max_length=128)
        batch = collator(sample_examples)

        text_mask = batch["text_mask"]
        token_type_ids = batch["token_type_ids"]
        attention_mask = batch["attention_mask"]

        batch_size = text_mask.shape[0]

        for i in range(batch_size):
            # Text mask should only be True for document tokens
            # Document tokens have token_type_ids == 1
            valid_positions = attention_mask[i].bool()
            doc_positions = (token_type_ids[i] == 1) & valid_positions

            # Text mask should be subset of document positions
            text_positions = text_mask[i]
            assert torch.all(text_positions <= doc_positions)

    def test_span_positions(self, tokenizer, sample_examples):
        """Test span position assignment."""
        collator = CriteriaBindingCollator(tokenizer=tokenizer, max_length=128)
        batch = collator(sample_examples)

        if "start_positions" in batch and "end_positions" in batch:
            start_pos = batch["start_positions"]
            end_pos = batch["end_positions"]

            # Check that positions are valid
            for i in range(len(start_pos)):
                if start_pos[i] != -1 and end_pos[i] != -1:
                    assert start_pos[i] <= end_pos[i]
                    # Positions should be within sequence length
                    seq_len = batch["input_ids"].shape[1]
                    assert 0 <= start_pos[i] < seq_len
                    assert 0 <= end_pos[i] < seq_len

    def test_empty_batch(self, tokenizer):
        """Test handling of empty example list."""
        collator = CriteriaBindingCollator(tokenizer=tokenizer, max_length=128)
        batch = collator([])

        # Should return empty batch
        assert "input_ids" in batch
        assert batch["input_ids"].shape[0] == 0

    def test_malformed_examples(self, tokenizer):
        """Test handling of malformed examples."""
        malformed_examples = [
            {
                "id": "bad_001",
                # Missing required fields
            },
            {
                "id": "bad_002",
                "criterion_text": "",  # Empty criterion
                "document_text": "Some text",
                "label": 1,
            },
        ]

        collator = CriteriaBindingCollator(tokenizer=tokenizer, max_length=128)

        # Should handle gracefully (may skip bad examples)
        batch = collator(malformed_examples)
        # Exact behavior depends on implementation, but shouldn't crash

    def test_window_count_estimation(self, tokenizer):
        """Test window count estimation."""
        collator = CriteriaBindingCollator(
            tokenizer=tokenizer,
            max_length=128,
            doc_stride=32,
        )

        # Short document - should be 1 window
        count = collator.get_window_count_for_example(
            "Short criterion",
            "Short document"
        )
        assert count == 1

        # Long document - should be multiple windows
        long_doc = " ".join(["word"] * 100)
        count = collator.get_window_count_for_example(
            "Short criterion",
            long_doc
        )
        assert count > 1


class TestInferenceCollator:
    """Test the inference collator."""

    def test_inference_collation(self, tokenizer, sample_examples):
        """Test inference collator functionality."""
        collator = InferenceCollator(
            tokenizer=tokenizer,
            max_length=128,
            doc_stride=64,
        )

        result = collator(sample_examples)

        # Should return batch and metadata
        assert "batch" in result
        assert "metadata" in result

        batch = result["batch"]
        metadata = result["metadata"]

        # Batch should have standard structure
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "token_type_ids" in batch
        assert "text_mask" in batch

        # Metadata should have window information
        assert isinstance(metadata, list)
        for meta in metadata:
            assert "offset_mapping" in meta
            assert "example_id" in meta
            assert "criterion_text" in meta
            assert "document_text" in meta

    def test_metadata_preservation(self, tokenizer, sample_examples):
        """Test that metadata is properly preserved."""
        collator = InferenceCollator(tokenizer=tokenizer, max_length=128)
        result = collator(sample_examples)

        metadata = result["metadata"]
        batch = result["batch"]

        # Should have metadata for each window in batch
        batch_size = batch["input_ids"].shape[0]
        assert len(metadata) == batch_size

        # Each metadata should correspond to original examples
        example_ids = {ex["id"] for ex in sample_examples}
        metadata_ids = {meta["example_id"] for meta in metadata}
        assert metadata_ids.issubset(example_ids)

    def test_offset_mapping(self, tokenizer, sample_examples):
        """Test offset mapping in metadata."""
        collator = InferenceCollator(tokenizer=tokenizer, max_length=128)
        result = collator(sample_examples)

        metadata = result["metadata"]

        for meta in metadata:
            offset_mapping = meta["offset_mapping"]
            batch_seq_len = result["batch"]["input_ids"].shape[1]

            # Offset mapping should match sequence length
            assert len(offset_mapping) <= batch_seq_len

            # Each offset should be a tuple of (start, end)
            for offset in offset_mapping:
                assert isinstance(offset, tuple)
                assert len(offset) == 2
                assert offset[0] <= offset[1]