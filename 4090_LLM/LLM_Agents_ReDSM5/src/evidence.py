from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class EvidenceItem:
    doc_id: str
    label: str
    sentence: str
    score: float


def select_top_k_sentences(
    sentences: Sequence[str],
    label_probs: np.ndarray,
    label_names: Sequence[str],
    top_k: int = 2,
) -> List[EvidenceItem]:
    """Placeholder heuristic scorer that selects the highest-probability labels and returns the first sentences."""
    if not sentences:
        return []
    ranked_labels = np.argsort(-label_probs)
    items: List[EvidenceItem] = []
    for idx in ranked_labels[:top_k]:
        sentence = sentences[min(idx, len(sentences) - 1)]
        items.append(EvidenceItem(doc_id="", label=label_names[idx], sentence=sentence, score=float(label_probs[idx])))
    return items


__all__ = ["EvidenceItem", "select_top_k_sentences"]
