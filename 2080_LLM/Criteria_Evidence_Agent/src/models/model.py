from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn

from .encoder_factory import build_encoder


def _get_activation(name: str) -> nn.Module:
    name = (name or "gelu").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "leakyrelu":
        return nn.LeakyReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "mish":
        return nn.Mish()
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation: {name}")


def _build_feedforward(
    input_dim: int,
    hidden_dims: List[int],
    activation: str,
    dropout: float,
) -> Tuple[nn.Sequential, int]:
    layers: List[nn.Module] = []
    prev_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(_get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = hidden_dim
    return nn.Sequential(*layers) if layers else nn.Identity(), prev_dim


class BaseClassificationHead(nn.Module):
    """Base head wrapping common configuration utilities."""

    head_type: str

    def __init__(self, input_dim: int, cfg: DictConfig) -> None:
        super().__init__()
        layers_cfg = cfg.get("layers", {})
        hidden_dims = layers_cfg.get("hidden_dims", [])
        activation = layers_cfg.get("activation", "gelu")
        dropout = layers_cfg.get("dropout", 0.0)
        self.feedforward, self.output_dim = _build_feedforward(input_dim, hidden_dims, activation, dropout)

    def forward(  # type: ignore[override]
        self,
        sequence_output: torch.Tensor,
        pooled_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class MultiLabelClassificationHead(BaseClassificationHead):
    head_type = "multi_label"

    def __init__(self, input_dim: int, cfg: DictConfig) -> None:
        super().__init__(input_dim, cfg)
        num_labels = len(cfg.labels)
        # Add dropout before final classifier
        classifier_dropout = cfg.get("classifier_dropout", cfg.get("layers", {}).get("dropout", 0.1))
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.output_dim, num_labels)

    def forward(  # type: ignore[override]
        self,
        sequence_output: torch.Tensor,
        pooled_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.feedforward(pooled_output)
        features = self.classifier_dropout(features)
        logits = self.classifier(features)
        return {"logits": logits}


class TokenClassificationHead(BaseClassificationHead):
    head_type = "token_classification"

    def __init__(self, input_dim: int, cfg: DictConfig) -> None:
        super().__init__(input_dim, cfg)
        num_labels = cfg.get("num_labels", 2)
        # Add dropout before final classifier
        classifier_dropout = cfg.get("classifier_dropout", cfg.get("layers", {}).get("dropout", 0.1))
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.output_dim, num_labels)

    def forward(  # type: ignore[override]
        self,
        sequence_output: torch.Tensor,
        pooled_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.feedforward(sequence_output)
        features = self.classifier_dropout(features)
        logits = self.classifier(features)
        return {"logits": logits}


class SpanClassificationHead(BaseClassificationHead):
    head_type = "span_classification"

    def __init__(self, input_dim: int, cfg: DictConfig) -> None:
        super().__init__(input_dim, cfg)
        # Add dropout before final classifier
        classifier_dropout = cfg.get("classifier_dropout", cfg.get("layers", {}).get("dropout", 0.1))
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.start_classifier = nn.Linear(self.output_dim, 1)
        self.end_classifier = nn.Linear(self.output_dim, 1)

    def forward(  # type: ignore[override]
        self,
        sequence_output: torch.Tensor,
        pooled_output: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        features = self.feedforward(sequence_output)
        features = self.classifier_dropout(features)
        start_logits = self.start_classifier(features).squeeze(-1)
        end_logits = self.end_classifier(features).squeeze(-1)
        return {"start_logits": start_logits, "end_logits": end_logits}


HEAD_TYPE_MAP = {
    "multi_label": MultiLabelClassificationHead,
    "token_classification": TokenClassificationHead,
    "span_classification": SpanClassificationHead,
}


class EvidenceModel(nn.Module):
    """Full model encapsulating encoder and multiple classification heads."""

    def __init__(self, model_cfg: DictConfig) -> None:
        super().__init__()
        encoder_cfg = model_cfg.encoder
        self.encoder, self.hidden_size = build_encoder(encoder_cfg)
        self.pooling = encoder_cfg.get("pooling", "cls")
        self.dropout = nn.Dropout(encoder_cfg.get("output_dropout", 0.0))

        self.head_configs: Dict[str, DictConfig] = {}
        self.classification_heads = nn.ModuleDict()
        for head_name, head_cfg in model_cfg.heads.items():
            if head_cfg.get("enabled", True) or head_cfg.get("type") == "multi_label":
                head = self._build_head(head_cfg)
                self.classification_heads[head_name] = head
                self.head_configs[head_name] = head_cfg

        self._maybe_apply_lora(encoder_cfg)
        self._maybe_freeze_base(encoder_cfg)

    def _build_head(self, head_cfg: DictConfig) -> BaseClassificationHead:
        head_type = head_cfg.get("type")
        if head_type not in HEAD_TYPE_MAP:
            raise ValueError(f"Unsupported head type: {head_type}")
        head_cls = HEAD_TYPE_MAP[head_type]
        return head_cls(self.hidden_size, head_cfg)

    def _maybe_apply_lora(self, encoder_cfg: DictConfig) -> None:
        lora_cfg = encoder_cfg.get("lora")
        if not lora_cfg or not lora_cfg.get("enabled", False):
            return
        target_modules = lora_cfg.get("target_modules") or ["query", "key", "value"]
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_cfg.get("r", 16),
            lora_alpha=lora_cfg.get("alpha", 32),
            lora_dropout=lora_cfg.get("dropout", 0.05),
            target_modules=target_modules,
        )
        self.encoder = get_peft_model(self.encoder, config)

    def _maybe_freeze_base(self, encoder_cfg: DictConfig) -> None:
        if not encoder_cfg.get("freeze_encoder", False):
            return
        for name, param in self.encoder.named_parameters():
            if "lora_" in name:
                continue
            param.requires_grad = False

    def pooled_output(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            summed = torch.sum(hidden_states * mask, dim=1)
            counts = mask.sum(dim=1).clamp(min=1e-6)
            return summed / counts
        return hidden_states[:, 0]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        sequence_output = encoder_outputs.last_hidden_state
        pooled_output = self.pooled_output(sequence_output, attention_mask)
        pooled_output = self.dropout(pooled_output)

        head_outputs: Dict[str, Dict[str, torch.Tensor]] = {}
        for head_name, head in self.classification_heads.items():
            head_outputs[head_name] = head(
                sequence_output=sequence_output,
                pooled_output=pooled_output,
                attention_mask=attention_mask,
            )
        return {
            "sequence_output": sequence_output,
            "pooled_output": pooled_output,
            "head_outputs": head_outputs,
        }

