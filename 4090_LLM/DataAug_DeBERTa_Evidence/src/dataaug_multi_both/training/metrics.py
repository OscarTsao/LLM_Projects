from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(
    logits_evidence,
    logits_criteria,
    labels_evidence,
    labels_criteria,
    split: str,
) -> Dict[str, float]:
    ev_preds = np.asarray(logits_evidence).argmax(axis=-1)
    cr_preds = np.asarray(logits_criteria).argmax(axis=-1)
    y_ev = np.asarray(labels_evidence)
    y_cr = np.asarray(labels_criteria)

    metrics = {
        f"{split}_ev_macro_f1": f1_score(y_ev, ev_preds, average="macro"),
        f"{split}_ev_accuracy": accuracy_score(y_ev, ev_preds),
        f"{split}_cri_macro_f1": f1_score(y_cr, cr_preds, average="macro"),
        f"{split}_cri_accuracy": accuracy_score(y_cr, cr_preds),
    }
    metrics[f"{split}_macro_f1_mean"] = 0.5 * (
        metrics[f"{split}_ev_macro_f1"] + metrics[f"{split}_cri_macro_f1"]
    )
    return metrics
