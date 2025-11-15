"""Command-line interface for two-stage Optuna HPO."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click
import optuna
from optuna.importance import get_param_importances

from dataaug_multi_both.hpo.two_stage import (
    STAGE1_CONFIG,
    STAGE2_CONFIG,
    run_stage1,
    run_stage2,
)

LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


@click.group()
@click.option("--verbose/--no-verbose", default=False, help="Toggle verbose logging output.")
def cli(verbose: bool) -> None:
    """DataAug Multi Both - two-stage HPO orchestrator."""

    _configure_logging(verbose)


@cli.command()
@click.option("--storage", required=True, help="Optuna storage URI (e.g., sqlite:///experiments/optuna.db).")
@click.option("--trials", default=350, show_default=True, type=int, help="Number of trials to execute.")
@click.option("--jobs", default=1, show_default=True, type=int, help="Number of parallel Optuna workers.")
@click.option("--study-name", default=None, help="Optional custom study name.")
@click.option(
    "--experiments-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("experiments"),
    show_default=True,
    help="Directory to store per-trial artifacts.",
)
def stage1(storage: str, trials: int, jobs: int, study_name: Optional[str], experiments_dir: Path) -> None:
    """Run Stage-1 broad / cheap search."""

    study = run_stage1(
        storage=storage,
        n_trials=trials,
        n_jobs=jobs,
        experiments_dir=experiments_dir,
        study_name=study_name,
    )
    best_value = None if study.best_trial is None else study.best_value
    click.echo(f"Stage-1 complete. Study: {study.study_name}. Best macro-F1: {best_value}")


@cli.command()
@click.option("--storage", required=True, help="Optuna storage URI.")
@click.option(
    "--stage1-study",
    default=STAGE1_CONFIG.study_name,
    show_default=True,
    help="Name of the Stage-1 study to consume.",
)
@click.option("--trials", default=150, show_default=True, type=int, help="Number of Stage-2 trials.")
@click.option("--jobs", default=1, show_default=True, type=int, help="Parallel Optuna workers.")
@click.option("--study-name", default=None, help="Optional Stage-2 study name.")
@click.option("--top-k", default=50, show_default=True, type=int, help="Top-K trials used for narrowing.")
@click.option(
    "--experiments-dir",
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    default=Path("experiments"),
    show_default=True,
    help="Directory to store per-trial artifacts.",
)
def stage2(
    storage: str,
    stage1_study: str,
    trials: int,
    jobs: int,
    study_name: Optional[str],
    top_k: int,
    experiments_dir: Path,
) -> None:
    """Run Stage-2 narrow / full-data search."""

    study = run_stage2(
        storage=storage,
        stage1_study=stage1_study,
        n_trials=trials,
        n_jobs=jobs,
        experiments_dir=experiments_dir,
        study_name=study_name,
        top_k=top_k,
    )
    best_value = None if study.best_trial is None else study.best_value
    click.echo(f"Stage-2 complete. Study: {study.study_name}. Best macro-F1: {best_value}")


@cli.command()
@click.option("--storage", required=True, help="Optuna storage URI.")
@click.option("--study-name", required=True, help="Study to inspect.")
@click.option("--top", default=5, show_default=True, type=int, help="Show top-N trials.")
def report(storage: str, study_name: str, top: int) -> None:
    """Summarise an Optuna study (top trials + parameter importances)."""

    study = optuna.load_study(study_name=study_name, storage=storage)
    trials = [
        t for t in study.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    ranked = sorted(trials, key=lambda t: t.value, reverse=True)[:top]

    click.echo(f"Study '{study_name}': {len(study.trials)} total trials, {len(trials)} completed.")
    for trial in ranked:
        trial_id = trial.user_attrs.get("trial_id", f"trial_{trial.number}")
        report_path = trial.user_attrs.get("evaluation_report")
        click.echo(
            f"- Trial #{trial.number} ({trial_id}) -> value={trial.value:.4f}, report={report_path}"
        )

    if trials:
        importances = get_param_importances(study, evaluator=None, target=None)
        click.echo("Top parameter importances:")
        for name, score in list(importances.items())[:10]:
            click.echo(f"  {name}: {score:.4f}")


def main() -> None:
    try:
        cli(standalone_mode=False)
    except SystemExit as exc:
        if exc.code not in (0, None):
            LOGGER.error("CLI exited with status %s", exc.code)
            raise
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.error("Failed to execute CLI: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
