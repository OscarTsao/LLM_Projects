from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import optuna
import yaml

from dataaug_multi_both.config import load_project_config
from dataaug_multi_both.hpo.run_study import (
    build_pruner,
    build_sampler,
    prepare_stage_b_controls,
    run_optuna_stage,
)

logger = logging.getLogger(__name__)


def _ensure_artifact_dir(stage_label: str) -> Path:
    path = Path("artifacts") / "hpo" / stage_label
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _render_matplotlib_plot(fig, path: Path) -> None:
    try:
        figure = None
        if hasattr(fig, "savefig"):
            figure = fig
        elif hasattr(fig, "figure"):
            figure = fig.figure
        elif isinstance(fig, (list, tuple)) and fig:
            first = fig[0]
            figure = first.figure if hasattr(first, "figure") else None
        elif hasattr(fig, "flat"):
            flat = list(fig.flat)
            if flat:
                first = flat[0]
                figure = first.figure if hasattr(first, "figure") else None
        if figure is None:
            logger.debug("Skipping plot export for unsupported object type: %s", type(fig))
            return
        figure.savefig(path, bbox_inches="tight")
    finally:
        if "figure" in locals() and figure is not None:
            plt.close(figure)


def _materialize_stage_artifacts(stage_label: str, study: optuna.study.Study) -> None:
    artifact_dir = _ensure_artifact_dir(stage_label)
    try:
        from optuna.visualization import matplotlib as vis
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Optuna visualization not available: %s", exc)
        return

    plotting_specs = [
        (vis.plot_optimization_history, "optimization_history.png"),
        (vis.plot_parallel_coordinate, "parallel_coordinate.png"),
        (vis.plot_slice, "slice.png"),
        (vis.plot_param_importances, "parameter_importances.png"),
        (vis.plot_contour, "hyperparameter_heatmap.png"),
    ]

    for plot_fn, filename in plotting_specs:
        try:
            fig = plot_fn(study)
        except Exception as exc:  # pragma: no cover - plotting failures
            logger.debug("Skipping %s: %s", filename, exc)
            continue
        _render_matplotlib_plot(fig, artifact_dir / filename)

    best_trial = study.best_trial
    _save_json(artifact_dir / "best_params.json", best_trial.params)
    _save_text(artifact_dir / "best_value.txt", str(best_trial.value))
    resolved_cfg = best_trial.user_attrs.get("config", {})
    with (artifact_dir / "resolved_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(resolved_cfg, f, sort_keys=False)


def _collect_top_trials(study: optuna.study.Study, k_top: int) -> List[optuna.trial.FrozenTrial]:
    trials = [t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(trials, key=lambda t: t.value or float("-inf"), reverse=True)
    return sorted_trials[: max(1, min(k_top, len(sorted_trials)))]


def run_two_stage_hpo(
    *,
    model: Optional[str],
    trials_a: int,
    epochs_a: int,
    trials_b: int,
    epochs_b: int,
    k_top: int,
    global_seed: int,
    timeout: Optional[int],
    config_files: Optional[Sequence[str | Path]] = None,
    config_overrides: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    cfg = load_project_config(extra_files=config_files, overrides=config_overrides)
    if model:
        cfg["encoder"]["model_name"] = model
        cfg["encoder"]["tokenizer_name"] = model

    hpo_cfg = cfg["hpo"]
    total_cap = int(hpo_cfg.get("total_trial_cap", 500))
    if trials_a + trials_b > total_cap:
        raise ValueError(
            f"Requested trials ({trials_a + trials_b}) exceed total cap ({total_cap})."
        )

    stage_a_cfg = hpo_cfg["stage_a"]
    stage_b_cfg = hpo_cfg["stage_b"]
    direction = hpo_cfg.get("direction", "maximize")

    timeout_a = int(timeout * stage_a_cfg.get("timeout_ratio", 0.6)) if timeout else None
    timeout_b = int(timeout * stage_b_cfg.get("timeout_ratio", 0.4)) if timeout else None

    sampler_a = build_sampler(stage_a_cfg)
    pruner_a = build_pruner(stage_a_cfg)

    stage_a_name = f"{hpo_cfg['study_base_name']}_{stage_a_cfg.get('name_suffix', 'stage_a')}"
    study_a = run_optuna_stage(
        cfg,
        study_name=stage_a_name,
        storage=hpo_cfg["storage"],
        sampler=sampler_a,
        pruner=pruner_a,
        n_trials=trials_a,
        timeout=timeout_a,
        stage_label="stage_a",
        stage_epochs=epochs_a,
        global_seed=global_seed,
        objective_metric=hpo_cfg["objective_metric"],
        plateau_patience=hpo_cfg.get("plateau_patience"),
        study_direction=direction,
    )

    _materialize_stage_artifacts("stage_a", study_a)

    top_trials = _collect_top_trials(study_a, k_top)
    if not top_trials:
        raise RuntimeError("Stage A completed without any successful trials.")

    structure_pool: List[tuple[Mapping[str, Any], Mapping[str, Mapping[str, Any]]]] = []
    enqueue_params: List[Mapping[str, Any]] = []
    for trial in top_trials:
        trial_cfg = trial.user_attrs.get("config")
        if not trial_cfg:
            logger.warning("Trial %d missing config metadata; skipping for Stage B.", trial.number)
            continue
        frozen, narrowed = prepare_stage_b_controls(trial_cfg)
        structure_idx = len(structure_pool)
        structure_pool.append((frozen, narrowed))
        params = dict(trial.params)
        params["structure.id"] = structure_idx
        enqueue_params.append(params)

    if not structure_pool:
        raise RuntimeError("Unable to derive Stage B structural pool from Stage A results.")

    if len(structure_pool) > trials_b:
        structure_pool = structure_pool[:trials_b]
        enqueue_params = enqueue_params[:trials_b]

    sampler_b = build_sampler(stage_b_cfg)
    pruner_b = build_pruner(stage_b_cfg)

    stage_b_name = f"{hpo_cfg['study_base_name']}_{stage_b_cfg.get('name_suffix', 'stage_b')}"
    study_b = run_optuna_stage(
        cfg,
        study_name=stage_b_name,
        storage=hpo_cfg["storage"],
        sampler=sampler_b,
        pruner=pruner_b,
        n_trials=trials_b,
        timeout=timeout_b,
        stage_label="stage_b",
        stage_epochs=epochs_b,
        global_seed=global_seed + 1,
        objective_metric=hpo_cfg["objective_metric"],
        structure_pool=structure_pool,
        plateau_patience=hpo_cfg.get("plateau_patience"),
        study_direction=direction,
        enqueue_params=enqueue_params,
    )

    _materialize_stage_artifacts("stage_b", study_b)

    summary = {
        "stage_a_best": {
            "value": study_a.best_value if study_a.best_trial else None,
            "trial_number": study_a.best_trial.number if study_a.best_trial else None,
            "config": study_a.best_trial.user_attrs.get("config", {}) if study_a.best_trial else {},
        },
        "stage_b_best": {
            "value": study_b.best_value if study_b.best_trial else None,
            "trial_number": study_b.best_trial.number if study_b.best_trial else None,
            "config": study_b.best_trial.user_attrs.get("config", {}) if study_b.best_trial else {},
        },
    }
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Two-stage Optuna HPO driver")
    parser.add_argument("--model", default=None, help="Override model name (default DeBERTa-v3-base)")
    parser.add_argument("--trials-a", type=int, default=380)
    parser.add_argument("--epochs-a", type=int, default=100)
    parser.add_argument("--trials-b", type=int, default=120)
    parser.add_argument("--epochs-b", type=int, default=100)
    parser.add_argument("--k-top", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=604800)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    args = build_parser().parse_args(argv)
    summary = run_two_stage_hpo(
        model=args.model,
        trials_a=args.trials_a,
        epochs_a=args.epochs_a,
        trials_b=args.trials_b,
        epochs_b=args.epochs_b,
        k_top=args.k_top,
        global_seed=args.seed,
        timeout=args.timeout,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
