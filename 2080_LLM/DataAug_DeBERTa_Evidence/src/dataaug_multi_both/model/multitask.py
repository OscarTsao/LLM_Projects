from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
from transformers import AutoModel

from dataaug_multi_both.model.heads import build_head

logger = logging.getLogger(__name__)


@dataclass
class EncoderConfig:
    model_name: str
    revision: str | None
    gradient_checkpointing: bool


class MultiTaskModel(nn.Module):
    def __init__(self, encoder: nn.Module, hidden_size: int, head_cfg: Dict[str, dict]):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.head_evidence = build_head(head_cfg["evidence"], in_dim=hidden_size, num_classes=head_cfg["evidence"]["num_classes"])
        self.head_criteria = build_head(head_cfg["criteria"], in_dim=hidden_size, num_classes=head_cfg["criteria"]["num_classes"])

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        evidence_logits = self.head_evidence(hidden_states, attention_mask)
        criteria_logits = self.head_criteria(hidden_states, attention_mask)
        return {
            "evidence_logits": evidence_logits,
            "criteria_logits": criteria_logits,
        }

def build_multitask_model(cfg: dict) -> MultiTaskModel:
    encoder_cfg = cfg["encoder"]
    model_name = encoder_cfg["model_name"]
    revision = encoder_cfg.get("revision")

    logger.info("Loading encoder %s (revision=%s)", model_name, revision or "main")
    try:
        encoder = AutoModel.from_pretrained(model_name, revision=revision)
    except Exception as exc:  # pragma: no cover - propagate failure
        raise RuntimeError(f"Failed to load encoder {model_name}: {exc}") from exc
    hidden_size = encoder.config.hidden_size

    if encoder_cfg.get("gradient_checkpointing", False) and hasattr(encoder, "gradient_checkpointing_enable"):
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("Enabled gradient checkpointing for encoder (use_reentrant=False)")

    heads_cfg = {
        "evidence": dict(cfg["heads"]["evidence"], num_classes=cfg["heads"]["evidence"]["num_classes"]),
        "criteria": dict(cfg["heads"]["criteria"], num_classes=cfg["heads"]["criteria"]["num_classes"]),
    }

    return MultiTaskModel(encoder=encoder, hidden_size=hidden_size, head_cfg=heads_cfg)
