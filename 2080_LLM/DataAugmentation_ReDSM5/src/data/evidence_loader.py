"""Data loading utilities for evidence span annotations."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer

from .criteria_descriptions import CRITERIA


@dataclass
class EvidenceSpan:
    """Represents an evidence span in a post."""
    
    post_id: str
    criterion: str
    sentence_text: str
    start_char: int
    end_char: int
    start_token: int
    end_token: int
    status: int  # 1 for positive evidence, 0 for negative


@dataclass
class EvidenceExample:
    """Training example for evidence binding."""
    
    post_id: str
    post_text: str
    criterion: str
    criterion_text: str
    evidence_spans: List[EvidenceSpan]
    has_evidence: bool


def load_evidence_annotations(
    posts_path: str,
    annotations_path: str,
    criteria_path: Optional[str] = None
) -> List[EvidenceExample]:
    """Load evidence annotations from ReDSM5 dataset."""
    
    # Load posts
    posts_df = pd.read_csv(posts_path)
    posts_dict = dict(zip(posts_df['post_id'], posts_df['text']))
    
    # Load annotations
    annotations_df = pd.read_csv(annotations_path)
    
    # Load criteria descriptions
    if criteria_path:
        with open(criteria_path, 'r') as f:
            criteria_data = json.load(f)
        criteria_map = {}
        for item in criteria_data:
            for criterion in item.get("criteria", []):
                criteria_map[criterion["id"]] = criterion["text"]
    else:
        criteria_map = CRITERIA
    
    # Create symptom to criterion mapping
    symptom_to_criterion = {
        "DEPRESSED_MOOD": "A.1",
        "ANHEDONIA": "A.2", 
        "APPETITE_CHANGE": "A.3",
        "SLEEP_ISSUES": "A.4",
        "PSYCHOMOTOR": "A.5",
        "FATIGUE": "A.6",
        "WORTHLESSNESS": "A.7",
        "COGNITIVE_ISSUES": "A.8",
        "SUICIDAL_THOUGHTS": "A.9",
    }
    
    # Group annotations by post and criterion
    examples = []
    grouped = annotations_df.groupby(['post_id', 'DSM5_symptom'])
    
    for (post_id, symptom), group in grouped:
        if post_id not in posts_dict:
            continue
            
        post_text = posts_dict[post_id]
        criterion_id = symptom_to_criterion.get(symptom)
        if not criterion_id or criterion_id not in criteria_map:
            continue
            
        criterion_text = criteria_map[criterion_id]
        
        # Extract evidence spans
        evidence_spans = []
        has_evidence = False
        
        for _, row in group.iterrows():
            if row['status'] == 1:  # Positive evidence
                sentence_text = row['sentence_text']
                
                # Find sentence in post text
                start_char, end_char = find_sentence_in_post(post_text, sentence_text)
                if start_char != -1:
                    evidence_span = EvidenceSpan(
                        post_id=post_id,
                        criterion=criterion_id,
                        sentence_text=sentence_text,
                        start_char=start_char,
                        end_char=end_char,
                        start_token=-1,  # Will be filled during tokenization
                        end_token=-1,
                        status=1
                    )
                    evidence_spans.append(evidence_span)
                    has_evidence = True
        
        example = EvidenceExample(
            post_id=post_id,
            post_text=post_text,
            criterion=criterion_id,
            criterion_text=criterion_text,
            evidence_spans=evidence_spans,
            has_evidence=has_evidence
        )
        examples.append(example)
    
    return examples


def find_sentence_in_post(post_text: str, sentence_text: str) -> Tuple[int, int]:
    """Find the character positions of a sentence in the post text."""
    # Clean and normalize text for matching
    post_clean = re.sub(r'\s+', ' ', post_text.strip())
    sentence_clean = re.sub(r'\s+', ' ', sentence_text.strip())
    
    # Try exact match first
    start = post_clean.find(sentence_clean)
    if start != -1:
        return start, start + len(sentence_clean)
    
    # Try fuzzy matching with different approaches
    # 1. Remove punctuation
    import string
    post_no_punct = post_clean.translate(str.maketrans('', '', string.punctuation))
    sentence_no_punct = sentence_clean.translate(str.maketrans('', '', string.punctuation))
    
    start = post_no_punct.find(sentence_no_punct)
    if start != -1:
        # Map back to original text (approximate)
        return start, start + len(sentence_no_punct)
    
    # 2. Try word-level matching
    post_words = post_clean.split()
    sentence_words = sentence_clean.split()
    
    if len(sentence_words) > 0:
        for i in range(len(post_words) - len(sentence_words) + 1):
            if post_words[i:i+len(sentence_words)] == sentence_words:
                # Calculate character positions
                start_char = len(' '.join(post_words[:i]))
                if i > 0:
                    start_char += 1  # Add space
                end_char = start_char + len(' '.join(sentence_words))
                return start_char, end_char
    
    return -1, -1  # Not found


def char_to_token_positions(
    text: str,
    char_start: int,
    char_end: int,
    tokenizer: AutoTokenizer,
    max_length: int = 512
) -> Tuple[int, int]:
    """Convert character positions to token positions."""
    # Tokenize with return_offsets_mapping
    encoding = tokenizer(
        text,
        return_offsets_mapping=True,
        max_length=max_length,
        truncation=True,
        padding=False
    )
    
    offset_mapping = encoding['offset_mapping']
    
    start_token = -1
    end_token = -1
    
    for i, (token_start, token_end) in enumerate(offset_mapping):
        # Skip special tokens
        if token_start == 0 and token_end == 0:
            continue
            
        # Find start token
        if start_token == -1 and token_start <= char_start < token_end:
            start_token = i
        
        # Find end token
        if token_start < char_end <= token_end:
            end_token = i
            break
    
    # If exact match not found, find closest tokens
    if start_token == -1:
        for i, (token_start, token_end) in enumerate(offset_mapping):
            if token_start == 0 and token_end == 0:
                continue
            if token_start >= char_start:
                start_token = i
                break
    
    if end_token == -1:
        for i in range(len(offset_mapping) - 1, -1, -1):
            token_start, token_end = offset_mapping[i]
            if token_start == 0 and token_end == 0:
                continue
            if token_end <= char_end:
                end_token = i
                break
    
    return start_token, end_token


class EvidenceDataset(torch.utils.data.Dataset):
    """Dataset for evidence binding training."""
    
    def __init__(
        self,
        examples: List[EvidenceExample],
        tokenizer_name: str,
        max_length: int = 512,
        include_negative: bool = True
    ):
        self.examples = examples
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.include_negative = include_negative
        
        # Process examples to include token positions
        self.processed_examples = []
        self._process_examples()
    
    def _process_examples(self):
        """Process examples to compute token positions."""
        for example in self.examples:
            # Create input text: [CLS] post [SEP] criterion [SEP]
            input_text = f"{example.post_text} [SEP] {example.criterion_text}"
            
            # Tokenize to get offset mapping
            encoding = self.tokenizer(
                example.post_text,
                example.criterion_text,
                return_offsets_mapping=True,
                max_length=self.max_length,
                truncation=True,
                padding=False
            )
            
            # Create start/end position arrays
            start_positions = [0] * len(encoding['input_ids'])
            end_positions = [0] * len(encoding['input_ids'])
            
            # Mark evidence spans
            for span in example.evidence_spans:
                start_token, end_token = char_to_token_positions(
                    example.post_text,
                    span.start_char,
                    span.end_char,
                    self.tokenizer,
                    self.max_length
                )
                
                if start_token != -1 and end_token != -1:
                    # Adjust for [CLS] token and potential truncation
                    if start_token < len(start_positions):
                        start_positions[start_token] = 1
                    if end_token < len(end_positions):
                        end_positions[end_token] = 1
            
            processed_example = {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
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


class EvidenceCollator:
    """Collator for evidence binding datasets."""
    
    def __init__(self, tokenizer_name: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
    
    def __call__(self, batch):
        # Pad sequences
        max_len = min(max(len(ex['input_ids']) for ex in batch), self.max_length)
        
        input_ids = []
        attention_mask = []
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
            start_positions.append(start_pos)
            end_positions.append(end_pos)
            has_evidence.append(example['has_evidence'])
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'start_positions': torch.tensor(start_positions, dtype=torch.float),
            'end_positions': torch.tensor(end_positions, dtype=torch.float),
            'has_evidence': torch.tensor(has_evidence, dtype=torch.bool)
        }
