# File: src/models/binder.py
"""SpanBERT-based evidence binding model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, Any, Optional, Tuple


class SpanBertEvidenceBinder(nn.Module):
    """SpanBERT-based model for evidence binding with span and classification heads.

    This model takes criterion and document text as input and predicts:
    1. Start and end positions for evidence spans in the document
    2. Optional classification label for the criterion-document pair
    """

    def __init__(
        self,
        model_name: str = "SpanBERT/spanbert-base-cased",
        num_labels: int = 2,
        use_label_head: bool = True,
        dropout: float = 0.1,
        lambda_span: float = 0.5,
        gradient_checkpointing: bool = False,
    ) -> None:
        """Initialize the SpanBERT evidence binder.

        Args:
            model_name: HuggingFace model name
            num_labels: Number of classification labels
            use_label_head: Whether to include classification head
            dropout: Dropout rate for heads
            lambda_span: Weight for span loss in multi-task learning
            gradient_checkpointing: Enable gradient checkpointing to save memory
        """
        super().__init__()

        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.config.hidden_size

        self.use_label_head = use_label_head
        self.lambda_span = lambda_span
        self.dropout = nn.Dropout(dropout)

        # Enable gradient checkpointing if requested
        if gradient_checkpointing:
            self.backbone.gradient_checkpointing_enable()

        # Span prediction heads
        self.start_head = nn.Linear(self.hidden_size, 1)
        self.end_head = nn.Linear(self.hidden_size, 1)

        # Optional classification head
        if use_label_head:
            self.label_head = nn.Linear(self.hidden_size, num_labels)

        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        text_mask: torch.Tensor,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through the model.

        Args:
            input_ids: Token IDs [B, T]
            attention_mask: Attention mask [B, T]
            token_type_ids: Segment IDs [B, T]
            text_mask: Mask indicating document tokens (True) vs criterion tokens (False) [B, T]
            start_positions: Gold start positions for training [B] or None
            end_positions: Gold end positions for training [B] or None
            labels: Classification labels for training [B] or None

        Returns:
            Dictionary containing logits and loss if training
        """
        # Get contextualized representations
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs.last_hidden_state  # [B, T, H]
        pooled_output = outputs.pooler_output  # [B, H]

        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        pooled_output = self.dropout(pooled_output)

        # Span prediction logits
        start_logits = self.start_head(sequence_output).squeeze(-1)  # [B, T]
        end_logits = self.end_head(sequence_output).squeeze(-1)  # [B, T]

        # Mask out non-document tokens (criterion and special tokens)
        start_logits = start_logits.masked_fill(~text_mask, float('-inf'))
        end_logits = end_logits.masked_fill(~text_mask, float('-inf'))

        result = {
            "start_logits": start_logits,
            "end_logits": end_logits,
        }

        # Classification logits
        if self.use_label_head:
            cls_logits = self.label_head(pooled_output)  # [B, C]
            result["cls_logits"] = cls_logits

        # Compute loss if training
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # Span loss
            start_loss = self.loss_fct(start_logits, start_positions)
            end_loss = self.loss_fct(end_logits, end_positions)
            span_loss = (start_loss + end_loss) / 2
            total_loss = self.lambda_span * span_loss

            result["span_loss"] = span_loss

        if labels is not None and self.use_label_head:
            # Classification loss
            cls_loss = self.loss_fct(result["cls_logits"], labels)
            if total_loss is not None:
                total_loss = total_loss + (1 - self.lambda_span) * cls_loss
            else:
                total_loss = cls_loss

            result["cls_loss"] = cls_loss

        if total_loss is not None:
            result["loss"] = total_loss

        return result

    def get_span_predictions(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        text_mask: torch.Tensor,
        max_answer_len: int = 64,
        top_k: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract top-k span predictions from logits.

        Args:
            start_logits: Start position logits [B, T]
            end_logits: End position logits [B, T]
            text_mask: Document token mask [B, T]
            max_answer_len: Maximum allowed span length
            top_k: Number of top spans to return

        Returns:
            Tuple of (start_indices, end_indices, scores) each [B, top_k]
        """
        batch_size, seq_len = start_logits.shape
        device = start_logits.device

        # Process in chunks to save memory for large batches
        chunk_size = min(batch_size, 4)  # Process 4 examples at a time

        batch_start_indices = []
        batch_end_indices = []
        batch_scores = []

        for chunk_start in range(0, batch_size, chunk_size):
            chunk_end = min(chunk_start + chunk_size, batch_size)

            # Convert logits to probabilities for current chunk
            chunk_start_logits = start_logits[chunk_start:chunk_end]
            chunk_end_logits = end_logits[chunk_start:chunk_end]
            chunk_text_mask = text_mask[chunk_start:chunk_end]

            start_probs = F.softmax(chunk_start_logits, dim=-1)
            end_probs = F.softmax(chunk_end_logits, dim=-1)

            for b in range(chunk_end - chunk_start):
                # Get valid document positions
                valid_positions = chunk_text_mask[b].nonzero(as_tuple=False).squeeze(-1)

                if len(valid_positions) == 0:
                    # No valid positions, return empty spans
                    start_indices = torch.full((top_k,), -1, device=device)
                    end_indices = torch.full((top_k,), -1, device=device)
                    scores = torch.zeros(top_k, device=device)
                else:
                    # Compute all valid span scores more efficiently
                    span_scores = []
                    span_starts = []
                    span_ends = []

                    # Vectorized computation where possible
                    for i, start_pos in enumerate(valid_positions):
                        valid_end_positions = valid_positions[i:]  # Only consider ends >= start
                        valid_end_positions = valid_end_positions[
                            valid_end_positions - start_pos < max_answer_len
                        ]

                        for end_pos in valid_end_positions:
                            score = start_probs[b, start_pos] + end_probs[b, end_pos]
                            span_scores.append(score.item())
                            span_starts.append(start_pos.item())
                            span_ends.append(end_pos.item())

                    if not span_scores:
                        # No valid spans
                        start_indices = torch.full((top_k,), -1, device=device)
                        end_indices = torch.full((top_k,), -1, device=device)
                        scores = torch.zeros(top_k, device=device)
                    else:
                        # Get top-k spans using torch.topk for efficiency
                        if len(span_scores) > top_k:
                            scores_tensor = torch.tensor(span_scores, device=device)
                            top_scores, top_indices = torch.topk(scores_tensor, top_k)
                            top_indices = top_indices.tolist()
                        else:
                            top_indices = list(range(len(span_scores)))

                        start_indices = torch.tensor(
                            [span_starts[i] for i in top_indices] +
                            [-1] * (top_k - len(top_indices)),
                            device=device
                        )
                        end_indices = torch.tensor(
                            [span_ends[i] for i in top_indices] +
                            [-1] * (top_k - len(top_indices)),
                            device=device
                        )
                        scores = torch.tensor(
                            [span_scores[i] for i in top_indices] +
                            [0.0] * (top_k - len(top_indices)),
                            device=device
                        )

                batch_start_indices.append(start_indices)
                batch_end_indices.append(end_indices)
                batch_scores.append(scores)

        return (
            torch.stack(batch_start_indices),
            torch.stack(batch_end_indices),
            torch.stack(batch_scores)
        )