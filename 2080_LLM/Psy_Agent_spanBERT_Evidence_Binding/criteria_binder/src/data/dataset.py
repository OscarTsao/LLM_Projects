# File: src/data/dataset.py
"""Dataset implementation for criteria binding."""

import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import logging

from ..utils.io import read_jsonl

logger = logging.getLogger(__name__)


class CriteriaBindingDataset(Dataset):
    """Dataset for criteria binding task.

    Loads JSONL data with criterion text, document text, labels, and evidence spans.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        label_to_id: Optional[Dict[str, int]] = None,
        max_examples: Optional[int] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            data_path: Path to JSONL data file
            label_to_id: Mapping from label strings to integers
            max_examples: Maximum number of examples to load (for debugging)
        """
        self.data_path = Path(data_path)
        self.label_to_id = label_to_id or {}
        self.max_examples = max_examples

        # Load data
        self.examples = self._load_data()

        logger.info(f"Loaded {len(self.examples)} examples from {data_path}")

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load and validate examples from JSONL file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        examples = read_jsonl(self.data_path)

        if self.max_examples is not None:
            examples = examples[:self.max_examples]

        # Validate and process examples
        processed_examples = []
        for i, example in enumerate(examples):
            try:
                processed_example = self._process_example(example)
                processed_examples.append(processed_example)
            except Exception as e:
                logger.warning(f"Skipping invalid example {i}: {e}")

        return processed_examples

    def _process_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Process and validate a single example."""
        # Required fields
        if "id" not in example:
            raise ValueError("Missing 'id' field")
        if "criterion_text" not in example:
            raise ValueError("Missing 'criterion_text' field")
        if "document_text" not in example:
            raise ValueError("Missing 'document_text' field")

        processed = {
            "id": example["id"],
            "criterion_text": example["criterion_text"].strip(),
            "document_text": example["document_text"].strip(),
        }

        # Validate texts are not empty
        if not processed["criterion_text"]:
            raise ValueError("Empty criterion_text")
        if not processed["document_text"]:
            raise ValueError("Empty document_text")

        # Process label if present
        if "label" in example:
            label = example["label"]
            if isinstance(label, str):
                if label not in self.label_to_id:
                    # Auto-assign label IDs
                    self.label_to_id[label] = len(self.label_to_id)
                processed["label"] = self.label_to_id[label]
            elif isinstance(label, int):
                # For integer labels, ensure we track them in label_to_id
                label_str = str(label)
                if label_str not in self.label_to_id:
                    self.label_to_id[label_str] = label
                processed["label"] = label
            else:
                raise ValueError(f"Invalid label type: {type(label)}")

        # Process evidence spans if present
        if "evidence_char_spans" in example:
            spans = example["evidence_char_spans"]
            if not isinstance(spans, list):
                raise ValueError("evidence_char_spans must be a list")

            # Validate span format
            processed_spans = []
            for span in spans:
                if not isinstance(span, (list, tuple)) or len(span) != 2:
                    raise ValueError("Each span must be [start, end]")
                start, end = span
                if not isinstance(start, int) or not isinstance(end, int):
                    raise ValueError("Span positions must be integers")
                if start < 0 or end < start:
                    raise ValueError(f"Invalid span: [{start}, {end}]")
                if end > len(processed["document_text"]):
                    raise ValueError(
                        f"Span end {end} exceeds document length "
                        f"{len(processed['document_text'])}"
                    )
                processed_spans.append((start, end))

            processed["evidence_char_spans"] = processed_spans
        else:
            processed["evidence_char_spans"] = []

        return processed

    def __len__(self) -> int:
        """Return number of examples."""
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get example by index."""
        if idx < 0 or idx >= len(self.examples):
            raise IndexError(f"Index {idx} out of range [0, {len(self.examples)})")

        return self.examples[idx].copy()

    def get_label_mapping(self) -> Dict[str, int]:
        """Get the label to ID mapping."""
        return self.label_to_id.copy()

    def get_id_to_label_mapping(self) -> Dict[int, str]:
        """Get the ID to label mapping."""
        return {v: k for k, v in self.label_to_id.items()}

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        stats = {
            "num_examples": len(self.examples),
            "num_labels": len(self.label_to_id),
            "label_distribution": {},
            "avg_criterion_length": 0,
            "avg_document_length": 0,
            "avg_evidence_spans": 0,
            "examples_with_evidence": 0,
        }

        if not self.examples:
            return stats

        # Compute statistics
        criterion_lengths = []
        document_lengths = []
        evidence_counts = []
        label_counts = {}

        for example in self.examples:
            criterion_lengths.append(len(example["criterion_text"]))
            document_lengths.append(len(example["document_text"]))

            evidence_spans = example.get("evidence_char_spans", [])
            evidence_counts.append(len(evidence_spans))

            if evidence_spans:
                stats["examples_with_evidence"] += 1

            if "label" in example:
                label = example["label"]
                label_counts[label] = label_counts.get(label, 0) + 1

        stats["avg_criterion_length"] = sum(criterion_lengths) / len(criterion_lengths)
        stats["avg_document_length"] = sum(document_lengths) / len(document_lengths)
        stats["avg_evidence_spans"] = sum(evidence_counts) / len(evidence_counts)
        stats["label_distribution"] = label_counts

        return stats


def create_label_mappings(train_dataset: CriteriaBindingDataset) -> Dict[str, Dict]:
    """Create label mappings from training dataset.

    Args:
        train_dataset: Training dataset

    Returns:
        Dictionary with label2id and id2label mappings
    """
    label2id = train_dataset.get_label_mapping()
    id2label = train_dataset.get_id_to_label_mapping()

    return {
        "label2id": label2id,
        "id2label": id2label,
        "num_labels": len(label2id),
    }