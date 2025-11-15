"""
Data preprocessing and collation for evidence extraction.

Handles tokenization and conversion of raw text to model inputs.
"""

from __future__ import annotations

import torch
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


class EvidenceCollator:
    """
    Collate function for evidence extraction datasets.

    Tokenizes sentence_text and creates span labels based on status field.
    For status=1 (has evidence), treats the entire sentence as the evidence span.
    For status=0 (no evidence), uses -1 to indicate no answer.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True,
    ):
        """
        Initialize collator.

        Args:
            tokenizer: Hugging Face tokenizer
            max_length: Maximum sequence length
            padding: Padding strategy ('max_length', 'longest', etc.)
            truncation: Whether to truncate long sequences
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        # Build set of special token IDs (excluding None)
        self.special_token_ids = {
            tokenizer.pad_token_id,
            tokenizer.sep_token_id,
            tokenizer.eos_token_id,
            tokenizer.cls_token_id,
            tokenizer.bos_token_id,
        }
        # Remove None values
        self.special_token_ids = {tid for tid in self.special_token_ids if tid is not None}

    def __call__(self, examples: list[dict]) -> dict[str, torch.Tensor]:
        """
        Collate a batch of examples.

        Args:
            examples: List of dataset examples with 'sentence_text' and 'status' fields

        Returns:
            Dictionary with input_ids, attention_mask, start_positions, end_positions
        """
        # Validate schema
        required_fields = {"sentence_text", "status"}
        for idx, ex in enumerate(examples):
            missing = required_fields - ex.keys()
            if missing:
                raise ValueError(
                    f"Example {idx} missing required fields: {missing}. "
                    f"Available fields: {list(ex.keys())}"
                )

        # Extract texts and status
        texts = [ex["sentence_text"] for ex in examples]
        statuses = [ex["status"] for ex in examples]

        # Tokenize all texts
        encodings = self.tokenizer(
            texts,
            padding=self.padding,
            truncation=self.truncation,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Create span positions
        start_positions = []
        end_positions = []

        for status, encoding in zip(statuses, encodings["input_ids"], strict=True):
            if status == 1:
                # Evidence present: span is the entire text (excluding special tokens)
                # Find first non-special token
                token_ids = encoding.tolist()

                # Skip [CLS] or <s> token at the beginning
                start_pos = 1

                # Find last non-special token (before [SEP], </s>, or padding)
                # Iterate backwards from the end
                end_pos = len(token_ids) - 1
                while end_pos > start_pos:
                    token_id = token_ids[end_pos]
                    # Check if it's a special token
                    if token_id in self.special_token_ids:
                        end_pos -= 1
                    else:
                        break

                # Ensure valid span
                if end_pos < start_pos:
                    end_pos = start_pos

                start_positions.append(start_pos)
                end_positions.append(end_pos)
            else:
                # No evidence: use -1 to indicate no answer
                start_positions.append(-1)
                end_positions.append(-1)

        # Convert to tensors
        batch = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "start_positions": torch.tensor(start_positions, dtype=torch.long),
            "end_positions": torch.tensor(end_positions, dtype=torch.long),
        }

        return batch


def create_collator(
    model_name_or_path: str,
    max_length: int = 512,
    *,
    local_files_only: bool = False,
) -> EvidenceCollator:
    """
    Factory function to create a collator with appropriate tokenizer.

    Args:
        model_name_or_path: Model identifier or path for tokenizer
        max_length: Maximum sequence length
        local_files_only: Whether to restrict loading to local Hugging Face cache

    Returns:
        EvidenceCollator instance
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
    )

    return EvidenceCollator(
        tokenizer=tokenizer,
        max_length=max_length,
    )
