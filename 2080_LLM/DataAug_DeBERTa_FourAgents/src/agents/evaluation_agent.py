from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

from src.eval.metrics import compute_evidence_metrics, compute_criteria_metrics
from src.eval.calibration import fit_temperature_scaling, save_calibration
from src.eval.report import write_metrics


class EvaluationAgent:
    def __init__(self, calibration_path: Path = Path("artifacts/calibration.json")) -> None:
        self.calibration_path = Path(calibration_path)

    def evaluate(
        self,
        predictions: List[Dict],
        criteria_results: List[Dict],
        dataset: Iterable[Dict],
        evaluation_dir: Path,
    ) -> Dict[str, Path]:
        evaluation_dir = Path(evaluation_dir)
        evaluation_dir.mkdir(parents=True, exist_ok=True)

        evidence_metrics = compute_evidence_metrics(predictions)
        criteria_metrics = compute_criteria_metrics(criteria_results, dataset)
        combined = {**evidence_metrics, **criteria_metrics}

        val_path = evaluation_dir / "val_metrics.json"
        test_path = evaluation_dir / "test_metrics.json"
        write_metrics(combined, val_path)
        write_metrics(combined, test_path)

        temperature = fit_temperature_scaling([p["score"] for p in predictions])
        thresholds = {row["symptom"]: 0.5 for row in predictions}
        save_calibration(temperature, thresholds, self.calibration_path)

        return {
            "val_metrics": val_path,
            "test_metrics": test_path,
            "calibration": self.calibration_path,
        }

    def run_gate_check(
        self,
        metrics_path: Path,
        neg_precision_min: float = 0.90,
        criteria_auroc_min: float = 0.80,
        ece_max: float = 0.05,
    ) -> None:
        script = Path("scripts/check_gates.py")
        if not script.is_file():
            return
        cmd = [
            sys.executable,
            str(script),
            "--metrics",
            str(metrics_path),
            "--neg-precision-min",
            str(neg_precision_min),
            "--criteria-auroc-min",
            str(criteria_auroc_min),
            "--ece-max",
            str(ece_max),
        ]
        subprocess.run(cmd, check=True)


__all__ = ["EvaluationAgent"]

