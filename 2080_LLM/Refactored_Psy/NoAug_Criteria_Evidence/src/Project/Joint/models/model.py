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
        criteria_model_name: str = "bert-base-uncased",
        evidence_model_name: str = "bert-base-uncased",
        criteria_num_labels: int = 2,
        criteria_dropout: float = 0.1,
        criteria_layer_num: int = 1,
        criteria_hidden_dims: Optional[Sequence[int]] = None,
        evidence_dropout: float = 0.1,
        fusion_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.criteria_encoder = transformers.AutoModel.from_pretrained(criteria_model_name)
        self.evidence_encoder = transformers.AutoModel.from_pretrained(evidence_model_name)

        criteria_hidden_size = self.criteria_encoder.config.hidden_size
        evidence_hidden_size = self.evidence_encoder.config.hidden_size

        if criteria_hidden_dims is not None:
            criteria_layer_num = len(criteria_hidden_dims) + 1

        self.criteria_head = ClassificationHead(
            input_dim=criteria_hidden_size,
            num_labels=criteria_num_labels,
            dropout_prob=criteria_dropout,
            layer_num=criteria_layer_num,
            hidden_dims=criteria_hidden_dims,
        )
        self.fusion = torch.nn.Sequential(
            torch.nn.Dropout(fusion_dropout),
            torch.nn.Linear(criteria_hidden_size, evidence_hidden_size),
            torch.nn.GELU(),
        )
        self.evidence_head = SpanPredictionHead(evidence_hidden_size, dropout_prob=evidence_dropout)

    def forward(
        self,
        criteria_input_ids: torch.Tensor,
        criteria_attention_mask: Optional[torch.Tensor] = None,
        criteria_token_type_ids: Optional[torch.Tensor] = None,
        evidence_input_ids: Optional[torch.Tensor] = None,
        evidence_attention_mask: Optional[torch.Tensor] = None,
        evidence_token_type_ids: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ):
        criteria_outputs = self.criteria_encoder(
            input_ids=criteria_input_ids,
            attention_mask=criteria_attention_mask,
            token_type_ids=criteria_token_type_ids,
            return_dict=True,
        )
        criteria_pooled = criteria_outputs.pooler_output
        if criteria_pooled is None:
            criteria_pooled = criteria_outputs.last_hidden_state[:, 0, :]

        criteria_logits = self.criteria_head(criteria_pooled)

        if evidence_input_ids is None:
            evidence_input_ids = criteria_input_ids
            evidence_attention_mask = criteria_attention_mask
            evidence_token_type_ids = criteria_token_type_ids

        evidence_outputs = self.evidence_encoder(
            input_ids=evidence_input_ids,
            attention_mask=evidence_attention_mask,
            token_type_ids=evidence_token_type_ids,
            return_dict=True,
        )
        evidence_sequence = evidence_outputs.last_hidden_state

        fusion_vector = self.fusion(criteria_pooled).unsqueeze(1)
        fused_sequence = evidence_sequence + fusion_vector

        start_logits, end_logits = self.evidence_head(fused_sequence)

        if return_dict:
            return {
                "criteria_logits": criteria_logits,
                "start_logits": start_logits,
                "end_logits": end_logits,
            }
        return criteria_logits, start_logits, end_logits
