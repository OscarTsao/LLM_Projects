from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput

LOGGER = logging.getLogger(__name__)


def _get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unsupported activation '{name}'.")


class FeedForwardSpanHead(nn.Module):
    """Configurable classification head for span prediction."""

    def __init__(self, input_dim: int, head_cfg: DictConfig):
        super().__init__()
        num_layers = max(1, int(head_cfg.num_layers))
        hidden_size = int(head_cfg.hidden_size)
        dropout_prob = float(head_cfg.dropout)
        self.activation = _get_activation(head_cfg.activation)
        self.layer_norm = head_cfg.layer_norm

        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_size))
            if self.layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout_prob))
            in_dim = hidden_size

        self.hidden = nn.Sequential(*layers) if layers else None
        self.qa_outputs = nn.Linear(in_dim, 2)
        self.dropout = nn.Dropout(dropout_prob)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.qa_outputs.weight, std=0.02)
        if self.qa_outputs.bias is not None:
            nn.init.zeros_(self.qa_outputs.bias)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        x = self.dropout(sequence_output)
        if self.hidden is not None:
            x = self.hidden(x)
        logits = self.qa_outputs(x)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


class SpanClassificationModel(PreTrainedModel):
    """Wrapper around a transformer backbone with a configurable span head."""

    config_class = AutoConfig

    def __init__(self, config, head_cfg: DictConfig):
        super().__init__(config)
        self.transformer = AutoModel.from_config(config)
        self.span_head = FeedForwardSpanHead(config.hidden_size, head_cfg)
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, head_cfg: DictConfig, **kwargs):
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = cls(config, head_cfg=head_cfg)
        base = AutoModel.from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)
        model.transformer = base
        return model

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        start_positions=None,
        end_positions=None,
        **kwargs,
    ):
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        sequence_output = outputs[0]
        start_logits, end_logits = self.span_head(sequence_output)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def freeze_base_model(self):
        for param in self.transformer.parameters():
            param.requires_grad = False
        LOGGER.info("Base transformer parameters frozen.")
