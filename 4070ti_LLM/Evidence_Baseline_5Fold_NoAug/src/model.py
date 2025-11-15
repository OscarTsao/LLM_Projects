"""Model definitions and loss functions for the evidence binding baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel


class FocalLoss(nn.Module):
    """Standard binary focal loss with optional alpha weighting."""

    def __init__(self, alpha: Optional[float] = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = None if alpha is None else float(alpha)
        self.gamma = float(gamma)
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        pt = probs * targets + (1 - targets) * (1 - probs)
        pt = pt.clamp(min=1e-6, max=1 - 1e-6)
        modulating = (1 - pt) ** self.gamma
        if self.alpha is not None:
            alpha_value = inputs.new_tensor(self.alpha)
            alpha_factor = targets * alpha_value + (1 - targets) * (1 - alpha_value)
        else:
            alpha_factor = torch.ones_like(targets)
        loss = alpha_factor * modulating * bce
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
        targets = targets.float()
        bce = self._bce(inputs, targets)
        probs = torch.sigmoid(inputs).clamp(min=1e-6, max=1 - 1e-6)
        pt = targets * probs + (1 - targets) * (1 - probs)
        adaptive_gamma = self.gamma + self.delta * (1 - pt)
        modulating = (1 - pt) ** adaptive_gamma
        alpha_value = inputs.new_tensor(self.alpha)
        alpha_factor = targets * alpha_value + (1 - targets) * (1 - alpha_value)
        loss = alpha_factor * modulating * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class BalancedFocalLoss(nn.Module):
    """Focal loss variant with effective-number class balancing."""

    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        beta: Optional[float] = None,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = float(gamma)
        self.reduction = reduction
        self.beta = None if beta is None else float(beta)
        self._alpha_override = float(alpha) if alpha is not None else None
        self._resolved_alpha: Optional[float] = self._alpha_override
        self.register_buffer("_class_weights", torch.ones(2, dtype=torch.float32), persistent=False)
        self._class_weights_initialized = False

    def configure_from_counts(self, positive: int, negative: int) -> None:
        pos = max(int(positive), 0)
        neg = max(int(negative), 0)
        counts = [neg if neg > 0 else 1, pos if pos > 0 else 1]
        if self.beta is not None:
            beta = min(max(self.beta, 0.0), 0.999999)
            weights: List[float] = []
            for count in counts:
                if count <= 0:
                    weights.append(0.0)
                    continue
                if beta >= 1.0:
                    weights.append(1.0)
                else:
                    weight = (1 - beta) / (1 - beta ** count)
                    weights.append(weight)
        else:
            weights = [float(value) for value in counts]
        total = sum(weights)
        if total > 0:
            scale = len(weights) / total
            weights = [w * scale for w in weights]
        else:
            weights = [1.0 for _ in weights]
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self._class_weights.device)
        self._class_weights.copy_(weights_tensor)
        self._class_weights_initialized = True
        if self._alpha_override is None:
            denom = float(weights_tensor.sum().item())
            if denom > 0:
                self._resolved_alpha = float(weights_tensor[1].item() / denom)
            else:
                self._resolved_alpha = 0.5
        else:
            self._resolved_alpha = self._alpha_override

    def set_alpha(self, alpha: Optional[float]) -> None:
        self._alpha_override = float(alpha) if alpha is not None else None
        self._resolved_alpha = self._alpha_override

    @property
    def resolved_alpha(self) -> float:
        return float(self._resolved_alpha if self._resolved_alpha is not None else 0.5)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        pt = probs * targets + (1 - targets) * (1 - probs)
        pt = pt.clamp(min=1e-6, max=1 - 1e-6)
        modulating = (1 - pt) ** self.gamma
        alpha_value = inputs.new_tensor(self.resolved_alpha)
        alpha_factor = targets * alpha_value + (1 - targets) * (1 - alpha_value)
        loss = alpha_factor * modulating * bce
        if self._class_weights_initialized:
            class_weights = self._class_weights.to(inputs.device)
            sample_weights = targets * class_weights[1] + (1 - targets) * class_weights[0]
            loss = loss * sample_weights
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
    loss_type: str = "balanced_focal"
    alpha: Optional[float] = None
    gamma: float = 2.0
    delta: float = 1.0
    effective_beta: Optional[float] = None
    use_gradient_checkpointing: bool = True
    freeze_encoder_layers: int = 0
    pooling: str = "cls"
    classifier_activation: str = "gelu"


class CriteriaClassifier(nn.Module):
    """Transformer encoder with pooled MLP head for evidence binding."""

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
        activation_name = (config.classifier_activation or "gelu").lower()
        if activation_name == "relu":
            activation_factory = nn.ReLU
        elif activation_name == "tanh":
            activation_factory = nn.Tanh
        else:
            activation_factory = nn.GELU
        for size in config.hidden_sizes:
            layers.extend([nn.Linear(hidden, size), activation_factory(), nn.Dropout(config.dropout)])
            hidden = size
        layers.append(nn.Linear(hidden, 1))
        self.classifier = nn.Sequential(*layers)
        self.dropout = nn.Dropout(config.dropout)
        self.pooling = (config.pooling or "cls").lower()
        self.loss_fn = self._init_loss()

    def _init_loss(self) -> nn.Module:
        loss_type = str(self.config.loss_type).lower()
        if loss_type == "bce":
            return nn.BCEWithLogitsLoss()
        if loss_type == "focal":
            return FocalLoss(alpha=self.config.alpha, gamma=self.config.gamma)
        if loss_type == "adaptive_focal":
            return AdaptiveFocalLoss(
                alpha=self.config.alpha,
                gamma=self.config.gamma,
                delta=self.config.delta,
            )
        if loss_type == "balanced_focal":
            return BalancedFocalLoss(
                alpha=self.config.alpha,
                gamma=self.config.gamma,
                beta=self.config.effective_beta,
            )
        raise ValueError(f"Unknown loss type: {self.config.loss_type}")

    def configure_loss_balancing(self, counts: Mapping[int, int]) -> None:
        """Configure loss weights based on class counts."""
        if not isinstance(counts, Mapping):
            return
        positive = int(counts.get(1, 0))
        negative = int(counts.get(0, 0))
        if hasattr(self.loss_fn, "configure_from_counts"):
            self.loss_fn.configure_from_counts(positive=positive, negative=negative)

    def _pool_hidden(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        pooling = self.pooling
        if pooling == "max":
            mask = attention_mask.unsqueeze(-1).bool()
            masked = hidden_states.masked_fill(~mask, float("-inf"))
            pooled = masked.max(dim=1).values
            pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
            return pooled
        if pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = (hidden_states * mask).sum(dim=1)
            denom = mask.sum(dim=1).clamp(min=1e-6)
            return summed / denom
        return hidden_states[:, 0]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool_hidden(outputs.last_hidden_state, attention_mask)
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
