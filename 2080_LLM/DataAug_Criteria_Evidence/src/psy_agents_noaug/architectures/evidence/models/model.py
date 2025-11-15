"""Evidence architecture model: encoder + span prediction head.

Produces start/end logits for answer spans (evidence) with a configurable
MLP stack on top of token hidden states.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from psy_agents_noaug.architectures.utils import SpanPredictionHead, make_bool_safe


class Model(nn.Module):
    """Pretrained encoder with configurable span head (layers/hidden/act/do)."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        *,
        head_cfg: dict[str, Any] | None = None,
        task_cfg: dict[str, Any] | None = None,
        dropout_prob: float = 0.1,
        **_: Any,
    ) -> None:
        super().__init__()
        head_cfg = dict(head_cfg or {})
        task_cfg = dict(task_cfg or {})

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        layers = head_cfg.get("layers", task_cfg.get("layers", 1))
        hidden = head_cfg.get("hidden")
        activation = head_cfg.get("activation", "gelu")
        dropout = head_cfg.get("dropout", dropout_prob)

        self.span_head = SpanPredictionHead(
            hidden_size,
            layers=layers,
            hidden=hidden,
            activation=activation,
            dropout=dropout,
        )

    def forward(
        self, *args: Any, **kwargs: Any
    ) -> QuestionAnsweringModelOutput | tuple[torch.Tensor, torch.Tensor]:
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

        start_logits, end_logits = self.span_head(encoder_outputs.last_hidden_state)
        if return_dict:
            return QuestionAnsweringModelOutput(
                start_logits=make_bool_safe(start_logits),
                end_logits=make_bool_safe(end_logits),
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )
        return start_logits, end_logits
