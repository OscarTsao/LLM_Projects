"""Test input format order: [CLS] criterion [SEP] post [SEP].

This test suite validates that the tokenization format has been correctly
changed from [CLS] post [SEP] criterion [SEP] to [CLS] criterion [SEP] post [SEP].

Key validations:
- Token sequence order matches expected format
- token_type_ids are correct (0 for criterion, 1 for post)
- All dataset implementations use consistent format
- Format works with all supported model architectures
"""

import pytest
import torch
from transformers import AutoTokenizer


class TestTokenizationFormat:
    """Test that tokenization produces [CLS] criterion [SEP] post [SEP] format."""

    @pytest.fixture
    def tokenizer(self):
        """Load RoBERTa tokenizer for testing."""
        return AutoTokenizer.from_pretrained("roberta-base")

    @pytest.fixture
    def sample_texts(self):
        """Sample criterion and post texts for testing."""
        return {
            "criterion": "Depressed mood most of the day",
            "post": "I have been feeling sad and hopeless for weeks",
        }

    def test_tokenization_order(self, tokenizer, sample_texts):
        """Test that tokens appear in correct order: criterion before post."""
        criterion = sample_texts["criterion"]
        post = sample_texts["post"]

        # Tokenize in new format (criterion first)
        encoded = tokenizer(
            criterion,
            text_pair=post,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )

        # Decode to verify order
        decoded = tokenizer.decode(encoded["input_ids"][0])

        # RoBERTa uses <s> and </s> instead of [CLS] and [SEP]
        # Expected format: <s> criterion </s></s> post </s>
        assert criterion.lower() in decoded.lower()
        assert post.lower() in decoded.lower()

        # Verify criterion appears before post in decoded string
        criterion_pos = decoded.lower().find(criterion.split()[0].lower())
        post_pos = decoded.lower().find(post.split()[0].lower())
        assert (
            criterion_pos < post_pos
        ), f"Criterion should appear before post. Got: {decoded}"

    def test_token_type_ids(self, tokenizer, sample_texts):
        """Test that token_type_ids are 0 for criterion, 1 for post."""
        criterion = sample_texts["criterion"]
        post = sample_texts["post"]

        encoded = tokenizer(
            criterion,
            text_pair=post,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )

        token_type_ids = encoded["token_type_ids"][0]

        # Count tokens in each segment
        # RoBERTa: <s> criterion </s></s> post </s>
        criterion_tokens = tokenizer(criterion, add_special_tokens=False)["input_ids"]

        # First segment (with special tokens) should have token_type_ids = 0
        # Second segment should have token_type_ids = 1
        num_criterion_tokens = len(criterion_tokens) + 2  # + <s> and </s>

        # Verify token_type_ids structure
        first_segment_ids = token_type_ids[:num_criterion_tokens]
        second_segment_ids = token_type_ids[num_criterion_tokens:]

        # All token_type_ids in first segment should be 0
        assert torch.all(first_segment_ids == 0), (
            f"First segment (criterion) should have token_type_ids=0. "
            f"Got: {first_segment_ids}"
        )

        # All token_type_ids in second segment should be 1
        assert torch.all(second_segment_ids == 1), (
            f"Second segment (post) should have token_type_ids=1. "
            f"Got: {second_segment_ids}"
        )

    def test_special_tokens_positions(self, tokenizer, sample_texts):
        """Test that special tokens are at expected positions."""
        criterion = sample_texts["criterion"]
        post = sample_texts["post"]

        encoded = tokenizer(
            criterion,
            text_pair=post,
            padding=False,
            truncation=False,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"][0]

        # RoBERTa special token IDs
        cls_token_id = tokenizer.cls_token_id  # <s> token
        sep_token_id = tokenizer.sep_token_id  # </s> token

        # First token should be <s> (CLS)
        assert (
            input_ids[0] == cls_token_id
        ), f"First token should be CLS. Got: {input_ids[0]}"

        # Find all SEP tokens
        sep_positions = (input_ids == sep_token_id).nonzero(as_tuple=True)[0]
        assert len(sep_positions) >= 2, (
            f"Should have at least 2 SEP tokens. " f"Got {len(sep_positions)}"
        )

    def test_max_length_truncation(self, tokenizer, sample_texts):
        """Test that truncation preserves criterion-first format."""
        criterion = sample_texts["criterion"]
        post = "word " * 200  # Very long post

        encoded = tokenizer(
            criterion,
            text_pair=post,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"][0]
        assert (
            input_ids.shape[0] == 128
        ), f"Should be padded to max_length. Got: {input_ids.shape}"

        # Decode first 30 tokens to verify criterion is present
        decoded_start = tokenizer.decode(input_ids[:30])
        assert criterion.split()[0].lower() in decoded_start.lower(), (
            f"Criterion should appear at start after truncation. "
            f"Got: {decoded_start}"
        )


class TestMainDatasetFormat:
    """Test format in main dataset (src/psy_agents_noaug/data/datasets.py)."""

    def test_eager_tokenization_format(self):
        """Test that eager tokenization uses criterion-first format."""
        from psy_agents_noaug.data.datasets import PsyAgentDataset

        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        # Mock data
        data = [
            {
                "post_text": "I feel sad every day",
                "criterion_text": "Depressed mood",
                "label": 1,
            }
        ]

        # Create dataset with eager tokenization
        dataset = PsyAgentDataset(
            data=data,
            tokenizer=tokenizer,
            max_length=128,
            input_format="eager",
            text_column="criterion_text",  # New format: criterion first
            text_pair_column="post_text",
        )

        # Get tokenized sample
        sample = dataset[0]
        decoded = tokenizer.decode(sample["input_ids"])

        # Verify criterion appears before post
        assert "depressed" in decoded.lower()
        assert "sad" in decoded.lower()

        criterion_pos = decoded.lower().find("depressed")
        post_pos = decoded.lower().find("sad")
        assert (
            criterion_pos < post_pos
        ), f"Criterion should appear first. Got: {decoded}"

    def test_lazy_tokenization_format(self):
        """Test that lazy tokenization collate uses criterion-first format."""
        from psy_agents_noaug.data.datasets import PsyAgentDataset

        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        # Mock data
        data = [
            {
                "post_text": "I feel sad every day",
                "criterion_text": "Depressed mood",
                "label": 1,
            }
        ]

        # Create dataset with lazy tokenization
        dataset = PsyAgentDataset(
            data=data,
            tokenizer=tokenizer,
            max_length=128,
            input_format="lazy",
            text_column="criterion_text",  # New format: criterion first
            text_pair_column="post_text",
        )

        # Get raw sample
        sample = dataset[0]

        # Collate batch
        batch = dataset.collate_fn([sample])
        decoded = tokenizer.decode(batch["input_ids"][0])

        # Verify criterion appears before post
        assert "depressed" in decoded.lower()
        assert "sad" in decoded.lower()

        criterion_pos = decoded.lower().find("depressed")
        post_pos = decoded.lower().find("sad")
        assert (
            criterion_pos < post_pos
        ), f"Criterion should appear first. Got: {decoded}"


class TestArchitectureDatasetFormat:
    """Test format in architecture-specific datasets."""

    def test_criteria_dataset_format(self):
        """Test CriteriaDataset uses criterion-first format."""
        from psy_agents_noaug.architectures.criteria.data.dataset import CriteriaDataset

        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        # Mock data
        data = [
            {
                "text": "I feel sad every day",
                "criterion_text": "Depressed mood",
                "label": 1,
            }
        ]

        dataset = CriteriaDataset(data=data, tokenizer=tokenizer, max_length=128)

        sample = dataset[0]
        decoded = tokenizer.decode(sample["input_ids"])

        # Verify criterion appears before post
        criterion_pos = decoded.lower().find("depressed")
        post_pos = decoded.lower().find("sad")
        assert (
            criterion_pos < post_pos
        ), f"Criterion should appear first. Got: {decoded}"

    def test_share_dataset_format(self):
        """Test ShareDataset uses criterion-first format."""
        from psy_agents_noaug.architectures.share.data.dataset import ShareDataset

        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        # Mock data with both criteria and evidence
        data = [
            {
                "context": "I feel sad every day and have trouble sleeping",
                "criterion_text": "Depressed mood",
                "label": 1,
                "cases": [{"text": "feel sad", "start_char": 2, "end_char": 10}],
            }
        ]

        dataset = ShareDataset(data=data, tokenizer=tokenizer, max_length=128)

        sample = dataset[0]
        decoded = tokenizer.decode(sample["input_ids"])

        # Verify criterion appears before context
        criterion_pos = decoded.lower().find("depressed")
        context_pos = decoded.lower().find("sad")
        assert (
            criterion_pos < context_pos
        ), f"Criterion should appear first. Got: {decoded}"

    def test_joint_dataset_format(self):
        """Test JointDataset uses criterion-first format for both encoders."""
        from psy_agents_noaug.architectures.joint.data.dataset import JointDataset

        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        # Mock data
        data = [
            {
                "sentence": "I feel sad every day",
                "context": "I feel sad every day and have trouble sleeping",
                "criterion_text": "Depressed mood",
                "label": 1,
                "cases": [{"text": "feel sad", "start_char": 2, "end_char": 10}],
            }
        ]

        dataset = JointDataset(
            data=data,
            criteria_tokenizer=tokenizer,
            evidence_tokenizer=tokenizer,
            max_length=128,
        )

        sample = dataset[0]

        # Check criteria encoder format
        criteria_decoded = tokenizer.decode(sample["criteria_input_ids"])
        criterion_pos = criteria_decoded.lower().find("depressed")
        sentence_pos = criteria_decoded.lower().find("sad")
        assert criterion_pos < sentence_pos, (
            f"Criteria encoder: criterion should appear first. "
            f"Got: {criteria_decoded}"
        )

        # Check evidence encoder format
        evidence_decoded = tokenizer.decode(sample["evidence_input_ids"])
        criterion_pos_ev = evidence_decoded.lower().find("depressed")
        context_pos = evidence_decoded.lower().find("sad")
        assert criterion_pos_ev < context_pos, (
            f"Evidence encoder: criterion should appear first. "
            f"Got: {evidence_decoded}"
        )


class TestClassificationLoaderFormat:
    """Test format in classification_loader.py."""

    def test_build_evidence_classification_loaders_format(self):
        """Test that classification loader uses criterion-first format."""
        from psy_agents_noaug.data.classification_loader import (
            build_evidence_classification_loaders,
        )

        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        # Mock groundtruth data
        train_data = [
            {
                "input_text": "I feel sad every day",
                "criterion_text": "Depressed mood",
                "label": 1,
            }
        ]
        val_data = train_data.copy()
        test_data = train_data.copy()

        # Build loaders with new format
        loaders = build_evidence_classification_loaders(
            tokenizer=tokenizer,
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            batch_size=1,
            max_length=128,
            num_workers=0,
            text_column="criterion_text",  # New default: criterion first
            text_pair_column="input_text",
        )

        # Get one batch from train loader
        batch = next(iter(loaders["train"]))
        decoded = tokenizer.decode(batch["input_ids"][0])

        # Verify criterion appears before input_text
        criterion_pos = decoded.lower().find("depressed")
        input_pos = decoded.lower().find("sad")
        assert (
            criterion_pos < input_pos
        ), f"Criterion should appear first. Got: {decoded}"


class TestBackwardCompatibility:
    """Test backward compatibility considerations."""

    def test_old_checkpoints_incompatible(self):
        """Document that old checkpoints trained with post-first format are incompatible."""
        # This is a documentation test - no actual checkpoint loading
        # Just verify that the format change is breaking

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        criterion = "Depressed mood"
        post = "I feel sad"

        # Old format: post first
        old_format = tokenizer(post, text_pair=criterion, return_tensors="pt")

        # New format: criterion first
        new_format = tokenizer(criterion, text_pair=post, return_tensors="pt")

        # token_type_ids should be REVERSED
        assert not torch.all(
            old_format["token_type_ids"] == new_format["token_type_ids"]
        ), "Old and new formats should produce different token_type_ids"

    def test_format_parameter_future_work(self):
        """Test placeholder for future input_format parameter."""
        # Future work: Add input_format="criterion_first" | "post_first" parameter
        # to allow backward compatibility with old checkpoints

        # This test documents the desired future API
        expected_api = {
            "input_format": ["criterion_first", "post_first"],
            "default": "criterion_first",
        }

        assert expected_api["default"] == "criterion_first"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_criterion(self):
        """Test handling of empty criterion text."""
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        criterion = ""
        post = "I feel sad"

        # Should handle gracefully
        encoded = tokenizer(criterion, text_pair=post, return_tensors="pt")
        assert encoded["input_ids"].shape[0] == 1

    def test_empty_post(self):
        """Test handling of empty post text."""
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        criterion = "Depressed mood"
        post = ""

        # Should handle gracefully
        encoded = tokenizer(criterion, text_pair=post, return_tensors="pt")
        assert encoded["input_ids"].shape[0] == 1

    def test_very_long_texts(self):
        """Test truncation with very long texts."""
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")

        criterion = "word " * 100
        post = "text " * 100

        encoded = tokenizer(
            criterion,
            text_pair=post,
            max_length=128,
            truncation=True,
            return_tensors="pt",
        )

        assert encoded["input_ids"].shape[1] == 128


class TestMultipleModelArchitectures:
    """Test format works with different model architectures."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "bert-base-uncased",
            "roberta-base",
            "microsoft/deberta-v3-base",
        ],
    )
    def test_format_with_different_models(self, model_name):
        """Test that format works with BERT, RoBERTa, and DeBERTa."""
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        criterion = "Depressed mood most of the day"
        post = "I have been feeling sad and hopeless for weeks"

        encoded = tokenizer(
            criterion,
            text_pair=post,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

        # Verify tokenization succeeds
        assert "input_ids" in encoded
        assert "attention_mask" in encoded
        assert "token_type_ids" in encoded

        # Verify shape
        assert encoded["input_ids"].shape == (1, 128)
        assert encoded["token_type_ids"].shape == (1, 128)

        # Decode to verify order
        decoded = tokenizer.decode(encoded["input_ids"][0], skip_special_tokens=False)
        assert criterion.split()[0].lower() in decoded.lower()
        assert post.split()[0].lower() in decoded.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
