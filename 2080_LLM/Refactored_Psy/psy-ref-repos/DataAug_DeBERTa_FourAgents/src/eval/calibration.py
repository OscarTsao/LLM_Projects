from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def fit_temperature_scaling(probabilities: List[float]) -> float:
    # Stub that returns neutral temperature; real implementation would optimise.
    return 1.0


def save_calibration(temperature: float, thresholds: Dict[str, float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "temperature": temperature,
        "thresholds": thresholds,
    }
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

