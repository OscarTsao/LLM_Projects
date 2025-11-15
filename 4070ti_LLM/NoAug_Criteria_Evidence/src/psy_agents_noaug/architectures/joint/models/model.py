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
class JointOutput(ModelOutput):
    logits: torch.Tensor | None = None
    start_logits: torch.Tensor | None = None
    end_logits: torch.Tensor | None = None
    criteria_hidden_states: tuple[torch.Tensor, ...] | None = None
    evidence_hidden_states: tuple[torch.Tensor, ...] | None = None
    criteria_attentions: tuple[torch.Tensor, ...] | None = None
    evidence_attentions: tuple[torch.Tensor, ...] | None = None


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
        fusion_dropout: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        head_cfg = dict(head_cfg or {})
        task_cfg = dict(task_cfg or {})

        criteria_cfg = dict(head_cfg.get("criteria", head_cfg))
        evidence_cfg = dict(head_cfg.get("evidence", head_cfg))
        shared_cfg = dict(head_cfg.get("shared", {}))

        criteria_model_name = (
            task_cfg.get("criteria_model_name")
            or shared_cfg.get("criteria_model_name")
            or model_name
        )
        evidence_model_name = (
            task_cfg.get("evidence_model_name")
            or shared_cfg.get("evidence_model_name")
            or model_name
        )

        self.criteria_encoder = AutoModel.from_pretrained(criteria_model_name)
        self.evidence_encoder = AutoModel.from_pretrained(evidence_model_name)

        criteria_hidden = self.criteria_encoder.config.hidden_size
        evidence_hidden = self.evidence_encoder.config.hidden_size

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
        criteria_hidden_setting = criteria_cfg.get("hidden", criteria_hidden_dims)
        criteria_activation = criteria_cfg.get("activation", "gelu")
        criteria_drop = criteria_cfg.get("dropout", criteria_dropout)

        evidence_hidden_setting = evidence_cfg.get("hidden")
        if isinstance(evidence_hidden_setting, (list, tuple)):
            default_evidence_layers = len(evidence_hidden_setting) + 1
        elif evidence_hidden_setting is not None and "layers" not in evidence_cfg:
            default_evidence_layers = 2
        else:
            default_evidence_layers = 1
        evidence_layers = evidence_cfg.get("layers", default_evidence_layers)
        evidence_activation = evidence_cfg.get("activation", "gelu")
        evidence_drop = evidence_cfg.get("dropout", evidence_dropout)

        fusion_drop = shared_cfg.get("dropout", fusion_dropout)

        self.pooler = SequencePooler(criteria_hidden, pooling=pooling)
        self.criteria_head = ClassificationHead(
            criteria_hidden,
            num_labels=num_labels,
            layers=criteria_layers,
            hidden=criteria_hidden_setting,
            activation=criteria_activation,
            dropout=criteria_drop,
        )
        self.align = (
            nn.Linear(criteria_hidden, evidence_hidden)
            if criteria_hidden != evidence_hidden
            else nn.Identity()
        )
        self.fusion_dropout = nn.Dropout(fusion_drop)
        self.evidence_head = SpanPredictionHead(
            evidence_hidden,
            layers=evidence_layers,
            hidden=evidence_hidden_setting,
            activation=evidence_activation,
            dropout=evidence_drop,
        )

    def forward(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> JointOutput | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        criteria_input_ids = kwargs.pop("criteria_input_ids", None)
        criteria_attention_mask = kwargs.pop("criteria_attention_mask", None)
        criteria_token_type_ids = kwargs.pop("criteria_token_type_ids", None)

        evidence_input_ids = kwargs.pop("evidence_input_ids", None)
        evidence_attention_mask = kwargs.pop("evidence_attention_mask", None)
        evidence_token_type_ids = kwargs.pop("evidence_token_type_ids", None)

        fallback_input = kwargs.pop("input_ids", None)
        fallback_mask = kwargs.pop("attention_mask", None)
        fallback_type_ids = kwargs.pop("token_type_ids", None)

        if criteria_input_ids is None:
            criteria_input_ids = fallback_input
            criteria_attention_mask = fallback_mask
            criteria_token_type_ids = fallback_type_ids

        if evidence_input_ids is None:
            evidence_input_ids = (
                fallback_input if fallback_input is not None else criteria_input_ids
            )
            evidence_attention_mask = (
                fallback_mask if fallback_mask is not None else criteria_attention_mask
            )
            evidence_token_type_ids = (
                fallback_type_ids
                if fallback_type_ids is not None
                else criteria_token_type_ids
            )

        criteria_outputs = self.criteria_encoder(
            input_ids=criteria_input_ids,
            attention_mask=criteria_attention_mask,
            token_type_ids=criteria_token_type_ids,
            return_dict=True,
            **kwargs,
        )
        # Handle models without pooler_output (DeBERTa, ELECTRA, etc.)
        criteria_pooler_output = getattr(criteria_outputs, 'pooler_output', None)
        criteria_pooled = self.pooler(
            criteria_outputs.last_hidden_state,
            attention_mask=criteria_attention_mask,
            pooler_output=criteria_pooler_output,
        )
        criteria_logits = self.criteria_head(criteria_pooled)

        evidence_outputs = self.evidence_encoder(
            input_ids=evidence_input_ids,
            attention_mask=evidence_attention_mask,
            token_type_ids=evidence_token_type_ids,
            return_dict=True,
            **kwargs,
        )
        fusion_vector = self.align(criteria_pooled)
        fusion_vector = self.fusion_dropout(fusion_vector).unsqueeze(1)
        fused_sequence = evidence_outputs.last_hidden_state + fusion_vector
        start_logits, end_logits = self.evidence_head(fused_sequence)

        if return_dict:
            return JointOutput(
                logits=criteria_logits,
                start_logits=make_bool_safe(start_logits),
                end_logits=make_bool_safe(end_logits),
                criteria_hidden_states=criteria_outputs.hidden_states,
                evidence_hidden_states=evidence_outputs.hidden_states,
                criteria_attentions=criteria_outputs.attentions,
                evidence_attentions=evidence_outputs.attentions,
            )
        return criteria_logits, start_logits, end_logits
