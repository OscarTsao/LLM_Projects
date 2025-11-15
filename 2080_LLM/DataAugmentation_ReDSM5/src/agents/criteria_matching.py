"""Criteria matching agent for DSM-5 criteria classification."""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .base import AgentOutput, BaseAgent, CriteriaMatchingConfig


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce_with_logits(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class AdaptiveFocalLoss(nn.Module):
    """Adaptive focal loss that scales the focusing parameter per example."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        delta: float = 1.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce_with_logits(inputs, targets)
        probs = torch.sigmoid(inputs).clamp(min=1e-6, max=1 - 1e-6)
        pt = targets * probs + (1 - targets) * (1 - probs)

        adaptive_gamma = self.gamma + self.delta * (1 - pt)
        modulating_factor = (1 - pt) ** adaptive_gamma
        loss = self.alpha * modulating_factor * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss


class CriteriaMatchingAgent(BaseAgent):
    """Agent for determining if DSM-5 criteria match a given post."""
    
    def __init__(self, config: CriteriaMatchingConfig):
        super().__init__(config)
        self.config = config
        
        # Load BERT model
        self.bert_config = AutoConfig.from_pretrained(config.model_name)
        self.encoder = AutoModel.from_pretrained(config.model_name, config=self.bert_config)
        
        # Classification head
        hidden_size = self.bert_config.hidden_size
        layers = []
        in_features = hidden_size
        
        for hidden in config.classifier_hidden_sizes:
            layers.extend([
                nn.Linear(in_features, hidden),
                nn.GELU(),
                nn.Dropout(config.dropout)
            ])
            in_features = hidden
            
        layers.append(nn.Linear(in_features, 1))  # Binary classification
        self.classifier = nn.Sequential(*layers)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Loss function
        self.loss_fn = self._get_loss_function()
        
        # Tokenizer for preprocessing
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
    def _get_loss_function(self) -> nn.Module:
        """Get the appropriate loss function based on config."""
        if self.config.loss_type == "bce":
            return nn.BCEWithLogitsLoss()
        elif self.config.loss_type == "focal":
            return FocalLoss(alpha=self.config.alpha, gamma=self.config.gamma)
        elif self.config.loss_type == "adaptive_focal":
            return AdaptiveFocalLoss(
                alpha=self.config.alpha,
                gamma=self.config.gamma,
                delta=self.config.delta
            )
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> AgentOutput:
        """Forward pass through the criteria matching agent."""
        # Encode input
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Get pooled representation
        if hasattr(encoder_outputs, 'pooler_output') and encoder_outputs.pooler_output is not None:
            pooled = encoder_outputs.pooler_output
        else:
            pooled = encoder_outputs.last_hidden_state[:, 0]  # Use [CLS] token
            
        pooled = self.dropout(pooled)
        
        # Classification
        logits = self.classifier(pooled).squeeze(-1)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities > 0.5).float()
        
        # Calculate confidence as max probability
        confidence = torch.max(probabilities, 1 - probabilities)
        
        return AgentOutput(
            predictions=predictions,
            confidence=confidence,
            logits=logits,
            probabilities=probabilities,
            metadata={"agent_type": "criteria_matching"}
        )
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> AgentOutput:
        """Make predictions without computing gradients."""
        self.eval()
        with torch.no_grad():
            return self.forward(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
    
    def get_loss(self, outputs: AgentOutput, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss for training."""
        return self.loss_fn(outputs.logits, targets.float())
    
    def tokenize_inputs(self, posts: list[str], criteria: list[str]) -> Dict[str, torch.Tensor]:
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
    
    def predict_batch(self, posts: list[str], criteria: list[str]) -> AgentOutput:
        """Predict on a batch of post-criteria pairs."""
        inputs = self.tokenize_inputs(posts, criteria)
        return self.predict(**inputs)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable()


def create_criteria_matching_agent(config: Optional[CriteriaMatchingConfig] = None) -> CriteriaMatchingAgent:
    """Factory function to create a criteria matching agent."""
    if config is None:
        config = CriteriaMatchingConfig()
    
    agent = CriteriaMatchingAgent(config)
    
    # Apply hardware optimizations
    if config.use_gradient_checkpointing:
        agent.enable_gradient_checkpointing()
    
    if config.use_compile:
        agent = agent.compile_model()
    
    return agent
