from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import AutoConfig, AutoModel


class SpanBertForQuestionAnswering(nn.Module):
    """Minimal QA head on top of SpanBERT encoder with gradient checkpointing support.

    Args:
        pretrained_model_name_or_path: Path or name of pretrained model
        dropout: Dropout rate for the QA head
        local_files_only: Whether to only use local files (no HuggingFace Hub)
        gradient_checkpointing: Whether to use gradient checkpointing to save memory
    """

    def __init__(
        self,
        pretrained_model_name_or_path: str,
        dropout: float = 0.1,
        local_files_only: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, local_files_only=local_files_only
        )
        self.encoder = AutoModel.from_pretrained(
            pretrained_model_name_or_path, config=self.config, local_files_only=local_files_only
        )

        # Enable gradient checkpointing if requested
        if gradient_checkpointing and hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

        hidden_size = self.config.hidden_size
        self.dropout = nn.Dropout(dropout if dropout is not None else self.config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(hidden_size, 2)

        # Initialize QA head weights
        self._init_weights(self.qa_outputs)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights for the QA head."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
    ) -> dict:
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            start_positions = start_positions.clamp(0, start_logits.shape[1] - 1)
            end_positions = end_positions.clamp(0, end_logits.shape[1] - 1)
            loss_fct = nn.CrossEntropyLoss()
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return {
            "loss": total_loss,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }

    def load_encoder_state(self, state_dict: dict) -> None:
        self.encoder.load_state_dict(state_dict, strict=False)


__all__ = ["SpanBertForQuestionAnswering"]
