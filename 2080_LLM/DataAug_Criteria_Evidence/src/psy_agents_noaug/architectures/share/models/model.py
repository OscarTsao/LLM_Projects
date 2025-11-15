"""Share architecture model (shared encoder with dual heads).

Supports a single encoder with separate heads and returns logits consistently
for downstream consumers and HPO logging.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import ModelOutput

from psy_agents_noaug.architectures.utils import (
    ClassificationHead,
    SequencePooler,
    SpanPredictionHead,
    make_bool_safe,
)


@dataclass
class ShareOutput(ModelOutput):
    logits: torch.Tensor | None = None
    start_logits: torch.Tensor | None = None
    end_logits: torch.Tensor | None = None
    hidden_states: tuple[torch.Tensor, ...] | None = None
    attentions: tuple[torch.Tensor, ...] | None = None


class Model(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        *,
        head_cfg: dict[str, Any] | None = None,
        task_cfg: dict[str, Any] | None = None,
        criteria_num_labels: int = 2,
        criteria_dropout: float = 0.1,
        criteria_layer_num: int = 1,
        criteria_hidden_dims: tuple[int, ...] | None = None,
        evidence_dropout: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        head_cfg = dict(head_cfg or {})
        task_cfg = dict(task_cfg or {})

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        criteria_cfg = dict(head_cfg.get("criteria", head_cfg))
        evidence_cfg = dict(head_cfg.get("evidence", head_cfg))

        num_labels = (
            task_cfg.get("criteria_num_labels")
            or task_cfg.get("num_labels")
            or criteria_cfg.get("num_labels")
            or criteria_num_labels
        )

        pooling = criteria_cfg.get("pooling") or task_cfg.get("pooling") or "cls"
        if criteria_hidden_dims is not None and "layers" not in criteria_cfg:
            criteria_layer_num = len(tuple(criteria_hidden_dims)) + 1
        criteria_layers = criteria_cfg.get("layers", criteria_layer_num)
        criteria_hidden = criteria_cfg.get("hidden", criteria_hidden_dims)
        criteria_activation = criteria_cfg.get("activation", "gelu")
        criteria_drop = criteria_cfg.get("dropout", criteria_dropout)

        evidence_hidden = evidence_cfg.get("hidden")
        if isinstance(evidence_hidden, list | tuple):
            default_evidence_layers = len(evidence_hidden) + 1
        elif evidence_hidden is not None and "layers" not in evidence_cfg:
            default_evidence_layers = 2
        else:
            default_evidence_layers = 1
        evidence_layers = evidence_cfg.get("layers", default_evidence_layers)
        evidence_activation = evidence_cfg.get("activation", "gelu")
        evidence_drop = evidence_cfg.get("dropout", evidence_dropout)

        self.pooler = SequencePooler(hidden_size, pooling=pooling)
        self.criteria_head = ClassificationHead(
            hidden_size,
            num_labels=num_labels,
            layers=criteria_layers,
            hidden=criteria_hidden,
            activation=criteria_activation,
            dropout=criteria_drop,
        )
        self.evidence_head = SpanPredictionHead(
            hidden_size,
            layers=evidence_layers,
            hidden=evidence_hidden,
            activation=evidence_activation,
            dropout=evidence_drop,
        )

    def forward(
        self, *args: Any, **kwargs: Any
    ) -> ShareOutput | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if args:
            if isinstance(args[0], Mapping):
                forwarded = dict(args[0])
                forwarded.update(kwargs)
                kwargs = forwarded
            else:
                raise TypeError(
                    "Unexpected positional arguments passed to Model.forward"
                )

        return_dict = kwargs.pop("return_dict", True)
        input_ids = kwargs.pop("input_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        token_type_ids = kwargs.pop("token_type_ids", None)

        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
            **kwargs,
        )

        pooled = self.pooler(
            encoder_outputs.last_hidden_state,
            attention_mask=attention_mask,
            pooler_output=encoder_outputs.pooler_output,
        )
        criteria_logits = self.criteria_head(pooled)
        start_logits, end_logits = self.evidence_head(encoder_outputs.last_hidden_state)

        if return_dict:
            return ShareOutput(
                logits=criteria_logits,
                start_logits=make_bool_safe(start_logits),
                end_logits=make_bool_safe(end_logits),
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )
        return criteria_logits, start_logits, end_logits
