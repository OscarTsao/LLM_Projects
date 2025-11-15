"""Utilities to export Optuna study artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import optuna
import optuna.visualization

from dataaug_multi_both.mlflow_buffer import MlflowBuffer

logger = logging.getLogger(__name__)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)


def _write_text(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(str(value))


def _safe_plot(plot_fn, study: optuna.Study, output: Path) -> None:
    try:
        fig = plot_fn(study)
    except Exception as exc:  # pragma: no cover - visualization dependency missing
        logger.debug("Failed to create plot %s: %s", plot_fn.__name__, exc)
        return
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output))
    except Exception as exc:  # pragma: no cover
        logger.debug("Failed to write plot %s: %s", output, exc)


def export_stage_artifacts(
    stage_result,
    output_dir: Path,
    buffer: MlflowBuffer | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    best_params = stage_result.best_trial.params
    _write_json(output_dir / "best_params.json", best_params)
    _write_text(output_dir / "best_value.txt", stage_result.best_trial.value)

    resolved_config = dict(best_params)
    if stage_result.settings.frozen_params:
        resolved_config.update(stage_result.settings.frozen_params)
    _write_json(output_dir / "resolved_config.json", resolved_config)

    study = stage_result.study
    plots_dir = output_dir / "plots"
    _safe_plot(optuna.visualization.plot_optimization_history, study, plots_dir / "optimization_history.html")
    _safe_plot(optuna.visualization.plot_parallel_coordinate, study, plots_dir / "parallel_coordinates.html")
    _safe_plot(optuna.visualization.plot_slice, study, plots_dir / "slice.html")
    _safe_plot(optuna.visualization.plot_param_importances, study, plots_dir / "param_importances.html")

    if buffer is not None:
        try:
            for artifact in [
                output_dir / "best_params.json",
                output_dir / "best_value.txt",
                output_dir / "resolved_config.json",
            ]:
                buffer.log_artifact(artifact)
            for html_plot in (plots_dir).glob("*.html"):
                buffer.log_artifact(html_plot)
        except Exception as exc:  # pragma: no cover
            logger.debug("Failed to buffer MLflow artifacts: %s", exc)


__all__ = ["export_stage_artifacts"]
