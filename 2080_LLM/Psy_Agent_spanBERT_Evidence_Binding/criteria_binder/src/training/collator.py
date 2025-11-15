# File: src/training/collator.py
"""Data collator for criteria binding with sliding windows and dynamic padding."""

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer
from typing import List, Dict, Any, Optional
import logging

from ..data.alignment import (
    get_alignment_info,
    get_alignment_info_cached,
    register_tokenizer_for_alignment,
    find_best_span_in_window,
)

logger = logging.getLogger(__name__)


class CriteriaBindingCollator:
    """Data collator for criteria binding with sliding window support.

    Handles:
    - Sliding windows over long documents
    - Dynamic padding to batch max length
    - Token alignment for span positions
    - Proper masking for criterion vs document tokens
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        doc_stride: int = 128,
        pad_to_multiple_of: Optional[int] = 8,  # Pad to multiples of 8 for tensor cores
        return_tensors: str = "pt",
    ) -> None:
        """Initialize the collator.

        Args:
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            doc_stride: Stride for sliding windows
            pad_to_multiple_of: Pad to multiple of this value
            return_tensors: Format for returned tensors
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self._tokenizer_key = register_tokenizer_for_alignment(tokenizer)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate a batch of examples into tensors.

        Args:
            examples: List of examples from the dataset

        Returns:
            Batch dictionary with padded tensors
        """
        # Process each example into windows
        all_windows = []
        example_ids = []

        for example in examples:
            try:
                windows = self._process_example(example)
                all_windows.extend(windows)
                example_ids.extend([example["id"]] * len(windows))
            except Exception as e:
                logger.warning(f"Failed to process example {example.get('id', 'unknown')}: {e}")
                continue

        if not all_windows:
            # Return empty batch
            return self._create_empty_batch()

        # Convert to tensors and pad
        return self._collate_windows(all_windows, example_ids)

    def _process_example(self, example: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single example into sliding windows.

        Args:
            example: Single example dictionary

        Returns:
            List of window dictionaries
        """
        criterion_text = example["criterion_text"]
        document_text = example["document_text"]
        evidence_spans = example.get("evidence_char_spans", [])
        label = example.get("label")

        # Get alignment info for all windows (cached)
        windows = get_alignment_info_cached(
            self._tokenizer_key,
            criterion_text,
            document_text,
            self.max_length,
            self.doc_stride,
        )

        # Process each window
        processed_windows = []
        for window_info in windows:
            window_data = {
                "input_ids": window_info["input_ids"],
                "attention_mask": window_info["attention_mask"],
                "token_type_ids": window_info["token_type_ids"],
                "text_mask": window_info["text_mask"],
            }

            # Add label if present
            if label is not None:
                window_data["labels"] = label

            # Find best evidence span for this window
            if evidence_spans:
                start_pos, end_pos = find_best_span_in_window(evidence_spans, window_info)
                window_data["start_positions"] = start_pos
                window_data["end_positions"] = end_pos
            else:
                # No evidence spans - use -1 to ignore in loss
                window_data["start_positions"] = -1
                window_data["end_positions"] = -1

            processed_windows.append(window_data)

        return processed_windows

    def _collate_windows(
        self,
        windows: List[Dict[str, Any]],
        example_ids: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Collate windows into a batch with padding.

        Args:
            windows: List of window dictionaries
            example_ids: Corresponding example IDs

        Returns:
            Batched and padded tensors
        """
        if not windows:
            return self._create_empty_batch()

        # Determine batch max length with efficient padding
        max_len = max(len(window["input_ids"]) for window in windows)
        if self.pad_to_multiple_of is not None:
            max_len = ((max_len + self.pad_to_multiple_of - 1) //
                      self.pad_to_multiple_of * self.pad_to_multiple_of)

        # Cap max length to avoid excessive memory usage
        max_len = min(max_len, self.max_length)

        # Pad all sequences
        batch = {}

        # Handle sequence tensors that need padding with memory efficiency
        for key in ["input_ids", "attention_mask", "token_type_ids", "text_mask"]:
            if key in windows[0]:
                # Pre-allocate tensor for better memory efficiency
                if key == "text_mask":
                    batch_tensor = torch.zeros(len(windows), max_len, dtype=torch.bool)
                else:
                    batch_tensor = torch.zeros(len(windows), max_len, dtype=torch.long)

                for i, window in enumerate(windows):
                    seq = window[key]
                    seq_len = len(seq)

                    if key == "attention_mask" or key == "text_mask":
                        # Pad with False/0
                        pad_value = 0
                    else:
                        # Use tokenizer's pad token ID
                        pad_value = self.tokenizer.pad_token_id

                    # Fill tensor directly to avoid intermediate lists
                    if seq_len > 0:
                        if key == "text_mask":
                            batch_tensor[i, :seq_len] = torch.tensor(seq, dtype=torch.bool)
                        else:
                            batch_tensor[i, :seq_len] = torch.tensor(seq, dtype=torch.long)
                            if seq_len < max_len:
                                batch_tensor[i, seq_len:] = pad_value

                batch[key] = batch_tensor

        # Handle scalar targets that don't need padding
        for key in ["labels", "start_positions", "end_positions"]:
            if key in windows[0]:
                values = [window[key] for window in windows]
                batch[key] = torch.tensor(values, dtype=torch.long)

        # Add example IDs for tracking
        batch["example_ids"] = example_ids

        return batch

    def _create_empty_batch(self) -> Dict[str, torch.Tensor]:
        """Create an empty batch for when no valid examples are found."""
        return {
            "input_ids": torch.empty(0, 0, dtype=torch.long),
            "attention_mask": torch.empty(0, 0, dtype=torch.long),
            "token_type_ids": torch.empty(0, 0, dtype=torch.long),
            "text_mask": torch.empty(0, 0, dtype=torch.bool),
            "example_ids": [],
        }

    def get_window_count_for_example(
        self,
        criterion_text: str,
        document_text: str,
    ) -> int:
        """Get the number of windows that would be created for an example.

        Args:
            criterion_text: Criterion text
            document_text: Document text

        Returns:
            Number of windows
        """
        try:
            windows = get_alignment_info(
                self.tokenizer,
                criterion_text,
                document_text,
                self.max_length,
                self.doc_stride,
            )
            return len(windows)
        except Exception:
            return 0


class InferenceCollator(CriteriaBindingCollator):
    """Collator for inference that preserves window information for decoding."""

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate examples and preserve window metadata.

        Args:
            examples: List of examples

        Returns:
            Batch with additional window metadata
        """
        # Process examples but keep window metadata
        all_windows = []
        all_metadata = []
        example_ids = []

        for example in examples:
            try:
                windows = get_alignment_info(
                    self.tokenizer,
                    example["criterion_text"],
                    example["document_text"],
                    self.max_length,
                    self.doc_stride,
                )

                for window_info in windows:
                    window_data = {
                        "input_ids": window_info["input_ids"],
                        "attention_mask": window_info["attention_mask"],
                        "token_type_ids": window_info["token_type_ids"],
                        "text_mask": window_info["text_mask"],
                    }

                    # Store metadata for decoding
                    metadata = {
                        "offset_mapping": window_info["offset_mapping"],
                        "doc_start_char": window_info["doc_start_char"],
                        "doc_end_char": window_info["doc_end_char"],
                        "example_id": example["id"],
                        "criterion_text": example["criterion_text"],
                        "document_text": example["document_text"],
                    }

                    all_windows.append(window_data)
                    all_metadata.append(metadata)
                    example_ids.append(example["id"])

            except Exception as e:
                logger.warning(f"Failed to process example {example.get('id', 'unknown')}: {e}")
                continue

        if not all_windows:
            return {
                "batch": self._create_empty_batch(),
                "metadata": [],
            }

        # Collate windows
        batch = self._collate_windows(all_windows, example_ids)

        return {
            "batch": batch,
            "metadata": all_metadata,
        }