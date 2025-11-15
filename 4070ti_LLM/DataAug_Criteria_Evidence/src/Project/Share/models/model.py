from collections.abc import Sequence
from typing import Any

import torch
import transformers


class ClassificationHead(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        dropout_prob: float = 0.1,
        layer_num: int = 1,
        hidden_dims: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if layer_num < 1:
            raise ValueError("layer_num must be at least 1.")

        if hidden_dims is not None:
            hidden_dims = tuple(hidden_dims)
            if len(hidden_dims) != layer_num - 1:
                raise ValueError(
                    "hidden_dims length must be equal to layer_num - 1 when provided."
                )
        else:
            hidden_dims = tuple([input_dim] * (layer_num - 1))

        dims = (input_dim,) + hidden_dims
        layers = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:], strict=False):
            layers.append(torch.nn.Linear(in_dim, out_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(dropout_prob))

        self.hidden_layers = torch.nn.Sequential(*layers)
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.output_layer = torch.nn.Linear(dims[-1], num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        if len(self.hidden_layers) > 0:
            x = self.hidden_layers(x)
        return self.output_layer(x)


class SpanPredictionHead(torch.nn.Module):
    def __init__(self, hidden_size: int, dropout_prob: float = 0.1) -> None:
        super().__init__()
        self.dropout = torch.nn.Dropout(dropout_prob)
        self.linear = torch.nn.Linear(hidden_size, 2)

    def forward(
        self, sequence_output: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_output = self.dropout(sequence_output)
        logits = self.linear(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


class Model(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        *,
        head_cfg: dict[str, Any] | None = None,
        task_cfg: dict[str, Any] | None = None,
        criteria_num_labels: int = 2,
        criteria_dropout: float = 0.1,
        criteria_layer_num: int = 1,
        criteria_hidden_dims: Sequence[int] | None = None,
        evidence_dropout: float = 0.1,
    ) -> None:
        """
        Initialize Share model with shared encoder and dual heads.

        Args:
            model_name: HuggingFace model identifier
            head_cfg: Head configuration dict (optional, for HPO compatibility)
                Expected keys: layers, hidden, activation, dropout
            task_cfg: Task configuration dict (optional, for HPO compatibility)
                Expected keys: num_labels
            criteria_num_labels: Number of criteria classes (overridden by task_cfg)
            criteria_dropout: Criteria head dropout (overridden by head_cfg)
            criteria_layer_num: Criteria head layers (overridden by head_cfg)
            criteria_hidden_dims: Criteria hidden dims (overridden by head_cfg)
            evidence_dropout: Evidence head dropout (overridden by head_cfg)
        """
        super().__init__()

        # Extract from head_cfg if provided (for HPO compatibility)
        if head_cfg:
            criteria_dropout = head_cfg.get("dropout", criteria_dropout)
            evidence_dropout = head_cfg.get("dropout", evidence_dropout)
            criteria_layer_num = head_cfg.get("layers", criteria_layer_num)
            hidden = head_cfg.get("hidden", None)
            if hidden is not None:
                if isinstance(hidden, int):
                    criteria_hidden_dims = (hidden,) * (criteria_layer_num - 1)
                elif isinstance(hidden, list | tuple):
                    criteria_hidden_dims = tuple(hidden)

        # Extract from task_cfg if provided
        if task_cfg:
            criteria_num_labels = task_cfg.get("num_labels", criteria_num_labels)

        self.model_name = model_name  # Store for later use
        self.encoder = transformers.AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        if criteria_hidden_dims is not None:
            criteria_layer_num = len(criteria_hidden_dims) + 1

        self.criteria_head = ClassificationHead(
            input_dim=hidden_size,
            num_labels=criteria_num_labels,
            dropout_prob=criteria_dropout,
            layer_num=criteria_layer_num,
            hidden_dims=criteria_hidden_dims,
        )
        self.evidence_head = SpanPredictionHead(
            hidden_size, dropout_prob=evidence_dropout
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        return_dict: bool = False,
    ):
        # DistilBERT doesn't support token_type_ids
        if "distilbert" in self.model_name.lower():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        else:
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
        # DistilBERT doesn't have pooler_output
        pooled_output = getattr(outputs, "pooler_output", None)
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]

        criteria_logits = self.criteria_head(pooled_output)
        start_logits, end_logits = self.evidence_head(outputs.last_hidden_state)

        if return_dict:
            return {
                "logits": criteria_logits,  # Fixed: was "criteria_logits"
                "start_logits": start_logits,
                "end_logits": end_logits,
            }
        return criteria_logits, start_logits, end_logits
