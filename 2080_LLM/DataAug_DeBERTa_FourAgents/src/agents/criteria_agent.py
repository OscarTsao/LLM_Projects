from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from src.criteria.aggregate import build_criteria_results, group_by_post
from src.suggestion.voi import attach_suggestions


class CriteriaAgent:
    def __init__(self, config_path: Path = Path("configs/criteria/aggregator.yaml")) -> None:
        self.config_path = Path(config_path)
        if yaml is not None and self.config_path.is_file():
            with self.config_path.open("r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
        else:
            cfg = {}
        self.symptoms: List[str] = cfg.get("symptoms", [])
        self.min_present_ratio: float = cfg.get("min_present_ratio", 0.5)

    def aggregate(self, predictions: List[Dict], top_k: int = 3, uncertain_band=(0.4, 0.6)) -> List[Dict]:
        symptoms = self.symptoms or sorted({row["symptom"] for row in predictions})
        results = build_criteria_results(predictions, symptoms, self.min_present_ratio)
        grouped = group_by_post(predictions)
        attach_suggestions(results, grouped, top_k, uncertain_band)
        return results


__all__ = ["CriteriaAgent"]
