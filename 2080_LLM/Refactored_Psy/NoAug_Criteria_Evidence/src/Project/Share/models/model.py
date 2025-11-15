from typing import Optional, Sequence, Tuple

import torch
import transformers


class ClassificationHead(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        dropout_prob: float = 0.1,
        layer_num: int = 1,
        hidden_dims: Optional[Sequence[int]] = None,
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
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
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

    def forward(self, sequence_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence_output = self.dropout(sequence_output)
        logits = self.linear(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)


class Model(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        criteria_num_labels: int = 2,
        criteria_dropout: float = 0.1,
        criteria_layer_num: int = 1,
        criteria_hidden_dims: Optional[Sequence[int]] = None,
        evidence_dropout: float = 0.1,
    ) -> None:
        super().__init__()
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
        self.evidence_head = SpanPredictionHead(hidden_size, dropout_prob=evidence_dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        pooled_output = outputs.pooler_output
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0, :]

        criteria_logits = self.criteria_head(pooled_output)
        start_logits, end_logits = self.evidence_head(outputs.last_hidden_state)

        if return_dict:
            return {
                "criteria_logits": criteria_logits,
                "start_logits": start_logits,
                "end_logits": end_logits,
            }
        return criteria_logits, start_logits, end_logits
