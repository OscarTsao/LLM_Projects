"""Joint dataset for multi-task training of criteria matching and evidence binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from transformers import AutoTokenizer

from .evidence_loader import EvidenceExample, load_evidence_annotations
from .redsm5_loader import load_ground_truth_frame


@dataclass
class JointExample:
    """Training example for joint criteria matching and evidence binding."""
    
    post_id: str
    post_text: str
    criterion: str
    criterion_text: str
    
    # Criteria matching labels
    criteria_label: int  # 0 or 1
    
    # Evidence binding labels (only meaningful if criteria_label == 1)
    evidence_spans: List[Tuple[int, int]]  # Token positions
    has_evidence: bool


def create_joint_dataset(
    ground_truth_path: str,
    posts_path: str,
    annotations_path: str,
    criteria_path: Optional[str] = None
) -> List[JointExample]:
    """Create joint dataset combining criteria matching and evidence binding data."""
    
    # Load ground truth for criteria matching
    ground_truth_df = load_ground_truth_frame(ground_truth_path)
    
    # Load evidence annotations
    evidence_examples = load_evidence_annotations(posts_path, annotations_path, criteria_path)
    
    # Create mapping from (post_id, criterion) to evidence
    evidence_map = {}
    for example in evidence_examples:
        key = (example.post_id, example.criterion)
        evidence_map[key] = example
    
    # Create joint examples
    joint_examples = []
    
    for _, row in ground_truth_df.iterrows():
        post_id = row['post_id']
        post_text = row['text_a']  # Post text
        criterion_text = row['text_b']  # Criterion text
        criteria_label = row['label']
        
        # Extract criterion ID from text (this might need adjustment based on data format)
        criterion_id = extract_criterion_id(criterion_text)
        
        # Get evidence information
        evidence_key = (post_id, criterion_id)
        evidence_example = evidence_map.get(evidence_key)
        
        if evidence_example:
            evidence_spans = [(span.start_token, span.end_token) for span in evidence_example.evidence_spans]
            has_evidence = evidence_example.has_evidence
        else:
            evidence_spans = []
            has_evidence = False
        
        joint_example = JointExample(
            post_id=post_id,
            post_text=post_text,
            criterion=criterion_id,
            criterion_text=criterion_text,
            criteria_label=criteria_label,
            evidence_spans=evidence_spans,
            has_evidence=has_evidence
        )
        
        joint_examples.append(joint_example)
    
    return joint_examples


def extract_criterion_id(criterion_text: str) -> str:
    """Extract criterion ID from criterion text."""
    # This is a simple heuristic - might need adjustment based on actual data format
    from .criteria_descriptions import CRITERIA
    
    for criterion_id, text in CRITERIA.items():
        if text.lower() in criterion_text.lower() or criterion_text.lower() in text.lower():
            return criterion_id
    
    # Fallback: try to extract A.X pattern
    import re
    match = re.search(r'A\.(\d+)', criterion_text)
    if match:
        return f"A.{match.group(1)}"
    
    return "A.1"  # Default fallback


class JointDataset(torch.utils.data.Dataset):
    """Dataset for joint training of criteria matching and evidence binding."""
    
    def __init__(
        self,
        examples: List[JointExample],
        tokenizer_name: str,
        max_length: int = 512
    ):
        self.examples = examples
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Process examples
        self.processed_examples = []
        self._process_examples()
    
    def _process_examples(self):
        """Process examples to create tokenized inputs."""
        for example in self.examples:
            # Tokenize post and criterion
            encoding = self.tokenizer(
                example.post_text,
                example.criterion_text,
                return_offsets_mapping=True,
                max_length=self.max_length,
                truncation=True,
                padding=False
            )
            
            # Create start/end position arrays for evidence
            start_positions = [0] * len(encoding['input_ids'])
            end_positions = [0] * len(encoding['input_ids'])
            
            # Mark evidence spans (only for positive criteria)
            if example.criteria_label == 1 and example.has_evidence:
                for start_token, end_token in example.evidence_spans:
                    if 0 <= start_token < len(start_positions):
                        start_positions[start_token] = 1
                    if 0 <= end_token < len(end_positions):
                        end_positions[end_token] = 1
            
            processed_example = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'criteria_label': example.criteria_label,
                'start_positions': start_positions,
                'end_positions': end_positions,
                'has_evidence': example.has_evidence,
                'post_id': example.post_id,
                'criterion': example.criterion
            }
            
            self.processed_examples.append(processed_example)
    
    def __len__(self):
        return len(self.processed_examples)
    
    def __getitem__(self, idx):
        return self.processed_examples[idx]


class JointCollator:
    """Collator for joint training datasets."""
    
    def __init__(self, tokenizer_name: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
    
    def __call__(self, batch):
        # Pad sequences
        max_len = min(max(len(ex['input_ids']) for ex in batch), self.max_length)
        
        input_ids = []
        attention_mask = []
        criteria_labels = []
        start_positions = []
        end_positions = []
        has_evidence = []
        
        for example in batch:
            # Pad or truncate
            ids = example['input_ids'][:max_len]
            mask = example['attention_mask'][:max_len]
            start_pos = example['start_positions'][:max_len]
            end_pos = example['end_positions'][:max_len]
            
            # Pad to max_len
            padding_length = max_len - len(ids)
            ids.extend([self.tokenizer.pad_token_id] * padding_length)
            mask.extend([0] * padding_length)
            start_pos.extend([0] * padding_length)
            end_pos.extend([0] * padding_length)
            
            input_ids.append(ids)
            attention_mask.append(mask)
            criteria_labels.append(example['criteria_label'])
            start_positions.append(start_pos)
            end_positions.append(end_pos)
            has_evidence.append(example['has_evidence'])
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'criteria': torch.tensor(criteria_labels, dtype=torch.long),
            'start_positions': torch.tensor(start_positions, dtype=torch.float),
            'end_positions': torch.tensor(end_positions, dtype=torch.float),
            'has_evidence': torch.tensor(has_evidence, dtype=torch.bool)
        }


@dataclass
class JointDataSplit:
    """Data splits for joint training."""
    
    train: List[JointExample]
    val: List[JointExample]
    test: List[JointExample]


def create_joint_splits(
    examples: List[JointExample],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> JointDataSplit:
    """Create train/val/test splits for joint training using group-based splitting."""
    
    # Extract post IDs for group-based splitting
    post_ids = [example.post_id for example in examples]
    unique_post_ids = list(set(post_ids))
    
    # First split: train vs (val + test)
    splitter1 = GroupShuffleSplit(
        n_splits=1,
        train_size=train_ratio,
        random_state=random_state
    )
    
    train_indices, temp_indices = next(splitter1.split(examples, groups=post_ids))
    
    # Second split: val vs test
    temp_examples = [examples[i] for i in temp_indices]
    temp_post_ids = [post_ids[i] for i in temp_indices]
    
    val_size = val_ratio / (val_ratio + test_ratio)
    splitter2 = GroupShuffleSplit(
        n_splits=1,
        train_size=val_size,
        random_state=random_state
    )
    
    val_temp_indices, test_temp_indices = next(splitter2.split(temp_examples, groups=temp_post_ids))
    
    # Map back to original indices
    val_indices = [temp_indices[i] for i in val_temp_indices]
    test_indices = [temp_indices[i] for i in test_temp_indices]
    
    # Create splits
    train_examples = [examples[i] for i in train_indices]
    val_examples = [examples[i] for i in val_indices]
    test_examples = [examples[i] for i in test_indices]
    
    return JointDataSplit(
        train=train_examples,
        val=val_examples,
        test=test_examples
    )


def build_joint_datasets(
    ground_truth_path: str,
    posts_path: str,
    annotations_path: str,
    tokenizer_name: str,
    criteria_path: Optional[str] = None,
    max_length: int = 512,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[JointDataset, JointDataset, JointDataset]:
    """Build train/val/test datasets for joint training."""
    
    # Create joint examples
    examples = create_joint_dataset(
        ground_truth_path=ground_truth_path,
        posts_path=posts_path,
        annotations_path=annotations_path,
        criteria_path=criteria_path
    )
    
    # Create splits
    splits = create_joint_splits(
        examples=examples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_state=random_state
    )
    
    # Create datasets
    train_dataset = JointDataset(splits.train, tokenizer_name, max_length)
    val_dataset = JointDataset(splits.val, tokenizer_name, max_length)
    test_dataset = JointDataset(splits.test, tokenizer_name, max_length)
    
    return train_dataset, val_dataset, test_dataset
