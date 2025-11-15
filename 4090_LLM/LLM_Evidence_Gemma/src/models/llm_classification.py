"""
LLM + classification head wrappers supporting causal (decoder) and
encoderized (bidirectional) modes.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import SequenceClassifierOutput

logger = logging.getLogger(__name__)


class LLMClassificationModel(nn.Module):
    """
    Decoder-only LLM with classification head supporting two modes:

    - mode="causal": keep causal masking, apply last-token pooling, feed into
      a linear classification head.
    - mode="encoderized": convert attention to bidirectional, then pool tokens
      (mean/last/attn) before an MLP head with dropout≈0.10.
    """

    def __init__(
        self,
        backbone: AutoModelForCausalLM,
        num_labels: int,
        mode: str = "causal",
        pooler: str = "mean",
        mlp_dropout: float = 0.10,
    ):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        self.num_labels = num_labels
        self.mode = mode
        self.pooler = pooler
        self.mlp_dropout = mlp_dropout
        self.backbone.config.use_cache = False
        self._current_attention_mask: Optional[torch.Tensor] = None

        hidden_size = getattr(self.backbone.config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Backbone config missing hidden_size.")

        if mode not in {"causal", "encoderized"}:
            raise ValueError("mode must be 'causal' or 'encoderized'")

        if self.mode == "encoderized":
            self._enable_bidirectional_attention()
            self.pooler_fn = self._build_pooler(pooler)
            mlp_hidden = hidden_size
            self.classifier = nn.Sequential(
                nn.Dropout(mlp_dropout),
                nn.Linear(hidden_size, mlp_hidden),
                nn.GELU(),
                nn.Dropout(mlp_dropout),
                nn.Linear(mlp_hidden, num_labels),
            )
        else:
            self.pooler_fn = self._last_token_pool
            self.classifier = nn.Linear(hidden_size, num_labels)

        if not hasattr(self.backbone, "gradient_checkpointing"):
            # Some remote-code models don't expose helper, but we still want to
            # disable cache.
            self.backbone.config.use_cache = False

    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self):
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        self._current_attention_mask = attention_mask if self.mode == "encoderized" else None
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states[-1]
        pooled = self.pooler_fn(hidden_states, attention_mask)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _build_pooler(self, pooler: str):
        pooler = pooler or "mean"
        pooler = pooler.lower()
        if pooler == "last":
            return self._last_token_pool
        if pooler == "mean":
            return self._mean_pool
        if pooler == "attn":
            self.attn_vector = nn.Linear(self.backbone.config.hidden_size, 1, bias=False)
            return self._attn_pool
        raise ValueError("Unsupported pooler: %s" % pooler)

    @staticmethod
    def _last_token_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # attention_mask is right-padded
        idx = attention_mask.to(dtype=torch.long).sum(dim=1) - 1
        idx = idx.clamp(min=0)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_indices, idx]

    @staticmethod
    def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1.0)
        return summed / lengths

    def _attn_pool(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        scores = self.attn_vector(hidden_states).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, -1e4)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), hidden_states).squeeze(1)

    def _enable_bidirectional_attention(self):
        """
        Patch the decoder attention layers to behave bidirectionally.
        """
        model = getattr(self.backbone, "model", None) or getattr(self.backbone, "transformer", None)
        layers = None
        if model is not None:
            layers = getattr(model, "layers", None) or getattr(model, "h", None)
        if layers is None:
            logger.warning("Backbone does not expose decoder layers; cannot enable encoderized mode.")
            return

        for layer in layers:
            attn_layer = getattr(layer, "self_attn", None) or getattr(layer, "attn", None)
            if attn_layer is None:
                continue
            original_forward = attn_layer.forward

            if hasattr(attn_layer, "is_causal"):
                attn_layer.is_causal = False
            if hasattr(attn_layer, "sliding_window"):
                attn_layer.sliding_window = None

            def make_forward(fn):
                def bidirectional_forward(*args, **kwargs):
                    attention_mask = kwargs.get("attention_mask")
                    hidden_states = args[0] if args else kwargs.get("hidden_states")
                    mask = self._current_attention_mask
                    if mask is None and attention_mask is not None:
                        mask = (attention_mask > 0).to(hidden_states.device)
                    if mask is None:
                        mask = torch.ones(
                            hidden_states.size()[:-1],
                            dtype=torch.bool,
                            device=hidden_states.device,
                        )
                    if mask.dtype != torch.bool:
                        mask = mask.ne(0)
                    if mask.dim() == 2:
                        valid_queries = mask.unsqueeze(1).unsqueeze(2)
                        valid_keys = mask.unsqueeze(1).unsqueeze(-1)
                        bidir = valid_queries & valid_keys
                        kwargs["attention_mask"] = (~bidir).to(dtype=torch.bool)
                    kwargs["use_cache"] = False
                    kwargs["past_key_values"] = None
                    return fn(*args, **kwargs)

                return bidirectional_forward

            attn_layer.forward = make_forward(original_forward)
