"""Model definitions for BERT-based pair classification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


@dataclass
class ModelConfig:
    pretrained_model_name: str
    classifier_hidden_sizes: Sequence[int]
    dropout: float
    num_labels: int = 2


class BertPairClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.model_config = AutoConfig.from_pretrained(config.pretrained_model_name)
        self.model_config.num_labels = config.num_labels
        self.bert = AutoModel.from_pretrained(config.pretrained_model_name, config=self.model_config)
        hidden_size = self.model_config.hidden_size

        layers: list[nn.Module] = []
        in_features = hidden_size
        for hidden in config.classifier_hidden_sizes:
            layers.append(nn.Linear(in_features, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            in_features = hidden
        layers.append(nn.Linear(in_features, config.num_labels))
        self.classifier = nn.Sequential(*layers)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, **inputs):
        labels = inputs.pop("labels", None)
        outputs = self.bert(**inputs)
        # DeBERTa models don't have pooler_output, use CLS token instead
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]  # CLS token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits.view(-1, self.model_config.num_labels), labels.view(-1))
        return {
            "loss": loss,
            "logits": logits,
        }

