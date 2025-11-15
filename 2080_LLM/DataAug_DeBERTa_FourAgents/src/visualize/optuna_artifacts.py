from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json


def _safe_plot(callable_plot, study, path: Path) -> None:
    try:
        fig = callable_plot(study)
        try:
            import matplotlib

            matplotlib.use("Agg")
        except Exception:
            pass
        fig.write_image(str(path)) if hasattr(fig, "write_image") else fig.figure.savefig(path)  # type: ignore[attr-defined]
    except Exception:
        # Fall back to study statistics dump
        path.with_suffix(".txt").write_text("Plot unavailable for this backend.")


def export_study_artifacts(study, out_dir: Path, best_params: Dict[str, Any], resolved_config: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Plots (matplotlib backend variants)
    try:
        from optuna.visualization.matplotlib import plot_optimization_history, plot_parallel_coordinate, plot_slice, plot_param_importances, plot_contour

        _safe_plot(plot_optimization_history, study, out_dir / "optimization_history.png")
        _safe_plot(plot_parallel_coordinate, study, out_dir / "parallel_coordinate.png")
        _safe_plot(plot_slice, study, out_dir / "slice.png")
        _safe_plot(plot_param_importances, study, out_dir / "parameter_importances.png")
        _safe_plot(plot_contour, study, out_dir / "hyperparameter_contour_or_heatmap.png")
    except Exception:
        pass

    # Best params and value
    (out_dir / "best_params.json").write_text(json.dumps(best_params, indent=2))
    (out_dir / "best_value.txt").write_text(str(study.best_value) if study.best_value is not None else "")

    # Resolved config yaml/json
    try:
        import yaml  # type: ignore

        with (out_dir / "resolved_best_config.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump(resolved_config, fh)
    except Exception:
        (out_dir / "resolved_best_config.json").write_text(json.dumps(resolved_config, indent=2))

    # README.txt
    summary = f"study={study.study_name}, direction={study.direction.name}, n_trials={len(study.trials)}\n"
    if study.best_trial is not None:
        summary += f"best_value={study.best_value}, best_trial_id={study.best_trial.number}\n"
    (out_dir / "README.txt").write_text(summary)

