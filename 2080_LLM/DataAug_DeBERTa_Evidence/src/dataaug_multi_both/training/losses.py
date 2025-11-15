from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F


def _prepare_weights(
    mode: str,
    weights: Optional[Dict[str, torch.Tensor]],
    key: str,
) -> Optional[torch.Tensor]:
    if mode == "none" or weights is None:
        return None
    raw = weights.get(key)
    if raw is None:
        return None
    if mode == "inverse_freq":
        return raw
    if mode == "sqrt_inverse":
        return torch.sqrt(raw)
    raise ValueError(f"Unknown class weighting mode: {mode}")


class MultiTaskLoss(torch.nn.Module):
    def __init__(
        self,
        label_smoothing: float,
        class_weighting: str,
        class_weights: Optional[Dict[str, torch.Tensor]] = None,
    ):
        super().__init__()
        self.label_smoothing = label_smoothing
        self.class_weighting = class_weighting
        if class_weights:
            self.register_buffer(
                "evidence_weight",
                class_weights["evidence"],
                persistent=False,
            )
            self.register_buffer(
                "criteria_weight",
                class_weights["criteria"],
                persistent=False,
            )
        else:
            self.evidence_weight = None
            self.criteria_weight = None

    def forward(
        self,
        logits_evidence: torch.Tensor,
        logits_criteria: torch.Tensor,
        labels_evidence: torch.Tensor,
        labels_criteria: torch.Tensor,
        weight_evidence: float,
    ) -> Dict[str, torch.Tensor]:
        weights = {
            "evidence": getattr(self, "evidence_weight", None),
            "criteria": getattr(self, "criteria_weight", None),
        }
        ev_weight = _prepare_weights(self.class_weighting, weights, "evidence")
        cr_weight = _prepare_weights(self.class_weighting, weights, "criteria")

        loss_ev = F.cross_entropy(
            logits_evidence,
            labels_evidence,
            label_smoothing=self.label_smoothing,
            weight=ev_weight.to(logits_evidence.device) if ev_weight is not None else None,
        )
        loss_cr = F.cross_entropy(
            logits_criteria,
            labels_criteria,
            label_smoothing=self.label_smoothing,
            weight=cr_weight.to(logits_criteria.device) if cr_weight is not None else None,
        )
        total = weight_evidence * loss_ev + (1.0 - weight_evidence) * loss_cr
        return {"loss": total, "loss_evidence": loss_ev, "loss_criteria": loss_cr}
