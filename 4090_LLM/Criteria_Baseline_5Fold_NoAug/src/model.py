"""Model definitions and loss functions for the criteria baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


class FocalLoss(nn.Module):
    """Standard focal loss."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self._bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self._bce(inputs, targets)
        pt = torch.exp(-bce)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class AdaptiveFocalLoss(nn.Module):
    """Adaptive focal loss with per-example gamma adjustment."""

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        delta: float = 1.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.delta = delta
        self.reduction = reduction
        self._bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = self._bce(inputs, targets)
        probs = torch.sigmoid(inputs).clamp(min=1e-6, max=1 - 1e-6)
        pt = targets * probs + (1 - targets) * (1 - probs)
        adaptive_gamma = self.gamma + self.delta * (1 - pt)
        modulating = (1 - pt) ** adaptive_gamma
        loss = self.alpha * modulating * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


@dataclass
class CriteriaModelConfig:
    """Configuration for CriteriaClassifier."""

    model_name: str
    hidden_sizes: List[int]
    dropout: float
    loss_type: str = "adaptive_focal"
    alpha: float = 0.25
    gamma: float = 2.0
    delta: float = 1.0
    use_gradient_checkpointing: bool = True
    base_model_dropout: Optional[float] = None
    base_model_attention_dropout: Optional[float] = None


class CriteriaClassifier(nn.Module):
    """Transformer encoder with MLP head for binary classification."""

    def __init__(self, config: CriteriaModelConfig):
        super().__init__()
        self.config = config
        self.base_config = AutoConfig.from_pretrained(config.model_name)

        if config.base_model_dropout is not None:
            dropout_value = float(config.base_model_dropout)
            for attr in (
                "hidden_dropout_prob",
                "hidden_dropout",
                "dropout",
                "classifier_dropout",
                "classifier_dropout_prob",
            ):
                if hasattr(self.base_config, attr):
                    setattr(self.base_config, attr, dropout_value)
        if config.base_model_attention_dropout is not None:
            attn_dropout_value = float(config.base_model_attention_dropout)
            for attr in ("attention_probs_dropout_prob", "attention_dropout", "attn_pdrop"):
                if hasattr(self.base_config, attr):
                    setattr(self.base_config, attr, attn_dropout_value)

        self.encoder = AutoModel.from_pretrained(config.model_name, config=self.base_config)

        if config.use_gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        hidden = self.base_config.hidden_size
        layers: List[nn.Module] = []
        for size in config.hidden_sizes:
            layers.extend([nn.Linear(hidden, size), nn.GELU(), nn.Dropout(config.dropout)])
            hidden = size
        layers.append(nn.Linear(hidden, 1))
        self.classifier = nn.Sequential(*layers)
        self.dropout = nn.Dropout(config.dropout)
        self.loss_fn = self._init_loss()

    def _init_loss(self) -> nn.Module:
        if self.config.loss_type == "bce":
            return nn.BCEWithLogitsLoss()
        if self.config.loss_type == "focal":
            return FocalLoss(alpha=self.config.alpha, gamma=self.config.gamma)
        if self.config.loss_type == "adaptive_focal":
            return AdaptiveFocalLoss(
                alpha=self.config.alpha,
                gamma=self.config.gamma,
                delta=self.config.delta,
            )
        raise ValueError(f"Unknown loss type: {self.config.loss_type}")

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).squeeze(-1)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())
        return logits, loss

    @torch.no_grad()
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        logits, _ = self.forward(input_ids=input_ids, attention_mask=attention_mask)
        return torch.sigmoid(logits)
