"""Evidence binding agent for predicting evidence spans in posts."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .base import AgentOutput, BaseAgent, EvidenceBindingConfig


class EvidenceBindingAgent(BaseAgent):
    """Agent for predicting start and end tokens of evidence sentences."""
    
    def __init__(self, config: EvidenceBindingConfig):
        super().__init__(config)
        self.config = config
        
        # Load BERT model
        self.bert_config = AutoConfig.from_pretrained(config.model_name)
        self.encoder = AutoModel.from_pretrained(config.model_name, config=self.bert_config)
        
        # Token classification heads for start and end positions
        hidden_size = self.bert_config.hidden_size
        self.start_classifier = nn.Linear(hidden_size, 1)
        self.end_classifier = nn.Linear(hidden_size, 1)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Loss function
        self.loss_fn = nn.BCEWithLogitsLoss()
        if config.label_smoothing > 0:
            # Use label smoothing for better generalization
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            self.label_smoothing = config.label_smoothing
        else:
            self.label_smoothing = 0.0
        
        # Tokenizer for preprocessing
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
    def forward(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        **kwargs
    ) -> AgentOutput:
        """Forward pass through the evidence binding agent."""
        # Encode input
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = encoder_outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        
        # Predict start and end positions
        start_logits = self.start_classifier(sequence_output).squeeze(-1)
        end_logits = self.end_classifier(sequence_output).squeeze(-1)
        
        # Apply attention mask to logits
        start_logits = start_logits.masked_fill(~attention_mask.bool(), -1e9)
        end_logits = end_logits.masked_fill(~attention_mask.bool(), -1e9)
        
        # Get probabilities
        start_probs = torch.sigmoid(start_logits)
        end_probs = torch.sigmoid(end_logits)
        
        # Extract spans based on threshold
        spans = self._extract_spans(start_probs, end_probs, attention_mask)
        
        # Calculate confidence as average of max start and end probabilities
        max_start_prob = torch.max(start_probs * attention_mask.float(), dim=1)[0]
        max_end_prob = torch.max(end_probs * attention_mask.float(), dim=1)[0]
        confidence = (max_start_prob + max_end_prob) / 2
        
        return AgentOutput(
            predictions=spans,
            confidence=confidence,
            logits={"start": start_logits, "end": end_logits},
            probabilities={"start": start_probs, "end": end_probs},
            metadata={"agent_type": "evidence_binding"}
        )
    
    def predict(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor,
        **kwargs
    ) -> AgentOutput:
        """Make predictions without computing gradients."""
        self.eval()
        with torch.no_grad():
            return self.forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    def get_loss(self, outputs: AgentOutput, targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for training."""
        start_targets = targets["start_positions"].float()
        end_targets = targets["end_positions"].float()
        
        start_logits = outputs.logits["start"]
        end_logits = outputs.logits["end"]
        
        if self.label_smoothing > 0:
            # Apply label smoothing
            start_targets = start_targets * (1 - self.label_smoothing) + self.label_smoothing / 2
            end_targets = end_targets * (1 - self.label_smoothing) + self.label_smoothing / 2
            
            start_loss = self.loss_fn(start_logits, start_targets).mean()
            end_loss = self.loss_fn(end_logits, end_targets).mean()
        else:
            start_loss = self.loss_fn(start_logits, start_targets)
            end_loss = self.loss_fn(end_logits, end_targets)
        
        return (start_loss + end_loss) / 2
    
    def _extract_spans(
        self, 
        start_probs: torch.Tensor, 
        end_probs: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[List[Tuple[int, int]]]:
        """Extract evidence spans from start and end probabilities."""
        batch_size = start_probs.size(0)
        spans = []
        
        for i in range(batch_size):
            seq_len = attention_mask[i].sum().item()
            start_prob = start_probs[i][:seq_len]
            end_prob = end_probs[i][:seq_len]
            
            # Find positions above threshold
            start_positions = (start_prob > self.config.span_threshold).nonzero(as_tuple=True)[0]
            end_positions = (end_prob > self.config.span_threshold).nonzero(as_tuple=True)[0]
            
            # Match start and end positions
            sequence_spans = []
            for start_pos in start_positions:
                # Find the nearest end position after this start
                valid_ends = end_positions[end_positions > start_pos]
                if len(valid_ends) > 0:
                    end_pos = valid_ends[0]
                    # Check span length constraint
                    if end_pos - start_pos <= self.config.max_span_length:
                        sequence_spans.append((start_pos.item(), end_pos.item()))
            
            spans.append(sequence_spans)
        
        return spans
    
    def tokenize_inputs(self, posts: List[str], criteria: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize post-criteria pairs for input to the model."""
        encodings = self.tokenizer(
            posts,
            criteria,
            padding=True,
            truncation=True,
            max_length=self.config.max_seq_length,
            return_tensors="pt"
        )
        return self.to_device(encodings)
    
    def predict_batch(self, posts: List[str], criteria: List[str]) -> AgentOutput:
        """Predict on a batch of post-criteria pairs."""
        inputs = self.tokenize_inputs(posts, criteria)
        return self.predict(**inputs)
    
    def decode_spans(
        self, 
        spans: List[List[Tuple[int, int]]], 
        input_ids: torch.Tensor
    ) -> List[List[str]]:
        """Decode token spans back to text."""
        decoded_spans = []
        
        for i, sequence_spans in enumerate(spans):
            sequence_decoded = []
            for start, end in sequence_spans:
                # Extract tokens and decode
                span_tokens = input_ids[i][start:end+1]
                span_text = self.tokenizer.decode(span_tokens, skip_special_tokens=True)
                sequence_decoded.append(span_text)
            decoded_spans.append(sequence_decoded)
        
        return decoded_spans
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()


def create_evidence_binding_agent(config: Optional[EvidenceBindingConfig] = None) -> EvidenceBindingAgent:
    """Factory function to create an evidence binding agent."""
    if config is None:
        config = EvidenceBindingConfig()
    
    agent = EvidenceBindingAgent(config)
    
    # Apply hardware optimizations
    if config.use_gradient_checkpointing:
        agent.enable_gradient_checkpointing()
    
    if config.use_compile:
        agent = agent.compile_model()
    
    return agent
