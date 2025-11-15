from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput

from psy_agents_noaug.architectures.utils import ClassificationHead, SequencePooler


class Model(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        *,
        head_cfg: dict[str, Any] | None = None,
        task_cfg: dict[str, Any] | None = None,
        num_labels: int = 2,
        classifier_dropout: float = 0.1,
        classifier_layer_num: int = 1,
        classifier_hidden_dims: Sequence[int] | None = None,
        **_: Any,
    ) -> None:
        super().__init__()
        head_cfg = dict(head_cfg or {})
        task_cfg = dict(task_cfg or {})

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        if classifier_hidden_dims is not None and "layers" not in head_cfg:
            classifier_layer_num = len(tuple(classifier_hidden_dims)) + 1

        resolved_labels = task_cfg.get(
            "num_labels",
            head_cfg.get("num_labels", num_labels),
        )
        layers = head_cfg.get("layers", classifier_layer_num)
        hidden = head_cfg.get("hidden", classifier_hidden_dims)
        activation = head_cfg.get("activation", "gelu")
        dropout = head_cfg.get("dropout", classifier_dropout)
        pooling = head_cfg.get("pooling") or task_cfg.get("pooling") or "cls"

        self.pooler = SequencePooler(hidden_size, pooling=pooling)
        self.classifier = ClassificationHead(
            hidden_size,
            num_labels=resolved_labels,
            layers=layers,
            hidden=hidden,
            activation=activation,
            dropout=dropout,
        )

    def forward(
        self, *args: Any, **kwargs: Any
    ) -> SequenceClassifierOutput | tuple[torch.Tensor]:
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

        # Handle models without pooler_output (DeBERTa, ELECTRA, etc.)
        pooler_output = getattr(encoder_outputs, 'pooler_output', None)
        pooled = self.pooler(
            encoder_outputs.last_hidden_state,
            attention_mask=attention_mask,
            pooler_output=pooler_output,
        )
        logits = self.classifier(pooled)

        if return_dict:
            return SequenceClassifierOutput(
                logits=logits,
                hidden_states=encoder_outputs.hidden_states,
                attentions=encoder_outputs.attentions,
            )
        return (logits,)
