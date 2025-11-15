from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

from src.suggestion.voi import attach_suggestions, suggest_for_post


class SuggestionAgent:
    def __init__(self, top_k: int = 3, uncertain_band: Tuple[float, float] = (0.4, 0.6)) -> None:
        self.top_k = top_k
        self.uncertain_band = uncertain_band

    def enrich(self, criteria_results: List[Dict], grouped_predictions: Dict[str, List[Dict]]) -> None:
        attach_suggestions(criteria_results, grouped_predictions, self.top_k, self.uncertain_band)


__all__ = ["SuggestionAgent", "suggest_for_post"]

