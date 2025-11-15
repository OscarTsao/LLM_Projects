"""Classification head for evidence extraction task."""

import torch
import torch.nn as nn
from typing import Optional


class EvidenceClassificationHead(nn.Module):
    """
    Classification head for evidence extraction.
    
    Maps encoded text to evidence labels (binary or multi-class).
    """
    
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
    ):
        """
        Initialize evidence head.
        
        Args:
            input_dim: Dimension of input encodings
            num_classes: Number of evidence classes
            dropout: Dropout probability
            hidden_dim: Optional hidden layer dimension
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        if hidden_dim:
            # Two-layer classification head
            self.classifier = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            # Single-layer classification head
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(input_dim, num_classes),
            )
    
    def forward(self, encodings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            encodings: Input encodings [batch_size, input_dim]
            
        Returns:
            Class logits [batch_size, num_classes]
        """
        return self.classifier(encodings)


class EvidenceModel(nn.Module):
    """Complete model for evidence extraction (encoder + head)."""
    
    def __init__(
        self,
        encoder: nn.Module,
        num_classes: int,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None,
    ):
        """
        Initialize evidence model.
        
        Args:
            encoder: Text encoder module
            num_classes: Number of evidence classes
            dropout: Dropout probability
            hidden_dim: Optional hidden layer dimension
        """
        super().__init__()
        
        self.encoder = encoder
        self.head = EvidenceClassificationHead(
            input_dim=encoder.hidden_size,
            num_classes=num_classes,
            dropout=dropout,
            hidden_dim=hidden_dim,
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through encoder and classification head.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Class logits [batch_size, num_classes]
        """
        # Encode
        encodings = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        # Classify
        logits = self.head(encodings)
        
        return logits
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get predictions (argmax of logits).
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Predicted class indices [batch_size]
        """
        logits = self.forward(input_ids, attention_mask)
        return torch.argmax(logits, dim=-1)
