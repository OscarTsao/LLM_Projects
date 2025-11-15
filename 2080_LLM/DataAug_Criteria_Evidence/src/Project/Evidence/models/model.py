from typing import Any

import torch
import transformers


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
        dropout_prob: float = 0.1,
    ) -> None:
        """
        Initialize Evidence span prediction model.

        Args:
            model_name: HuggingFace model identifier
            head_cfg: Head configuration dict (optional, for HPO compatibility)
                Expected keys: layers, hidden, activation, dropout
            task_cfg: Task configuration dict (optional, for HPO compatibility)
            dropout_prob: Dropout probability (overridden by head_cfg)
        """
        super().__init__()

        # Extract from head_cfg if provided (for HPO compatibility)
        if head_cfg:
            dropout_prob = head_cfg.get("dropout", dropout_prob)
            # Note: This simple implementation only supports single-layer span head
            # For multi-layer support, SpanPredictionHead would need to be enhanced

        self.encoder = transformers.AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        self.span_head = SpanPredictionHead(hidden_size, dropout_prob)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state
        return self.span_head(sequence_output)
