"""Model definitions and loss functions for the criteria baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

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
    encoder_dropout: Optional[float] = None
    attention_dropout: Optional[float] = None
    loss_type: str = "adaptive_focal"
    alpha: float = 0.25
    gamma: float = 2.0
    delta: float = 1.0
    use_gradient_checkpointing: bool = True
    freeze_encoder_layers: int = 0


class CriteriaClassifier(nn.Module):
    """DeBERTa encoder with MLP head for binary classification."""

    def __init__(self, config: CriteriaModelConfig):
        super().__init__()
        self.config = config
        self.base_config = AutoConfig.from_pretrained(config.model_name)
        if config.encoder_dropout is not None:
            encoder_dropout = float(config.encoder_dropout)
            if hasattr(self.base_config, "hidden_dropout_prob"):
                self.base_config.hidden_dropout_prob = encoder_dropout
            if hasattr(self.base_config, "hidden_dropout"):
                self.base_config.hidden_dropout = encoder_dropout
        if config.attention_dropout is not None:
            attention_dropout = float(config.attention_dropout)
            if hasattr(self.base_config, "attention_probs_dropout_prob"):
                self.base_config.attention_probs_dropout_prob = attention_dropout
            if hasattr(self.base_config, "attention_dropout"):
                self.base_config.attention_dropout = attention_dropout
        if hasattr(self.base_config, "classifier_dropout"):
            self.base_config.classifier_dropout = float(config.dropout)
        self.encoder = AutoModel.from_pretrained(config.model_name, config=self.base_config)

        if config.use_gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            try:
                self.encoder.gradient_checkpointing_enable(use_reentrant=False)
            except TypeError:
                self.encoder.gradient_checkpointing_enable()

        freeze_layers = max(0, int(config.freeze_encoder_layers))
        if freeze_layers > 0:
            encoder_module = getattr(self.encoder, "encoder", None)
            layer_stack = getattr(encoder_module, "layer", None) if encoder_module is not None else None
            if layer_stack is not None:
                for layer in list(layer_stack)[:freeze_layers]:
                    for param in layer.parameters():
                        param.requires_grad = False

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
