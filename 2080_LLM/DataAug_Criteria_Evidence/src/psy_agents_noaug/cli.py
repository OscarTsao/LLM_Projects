"""CLI entrypoints for the PSY Agents NO‑AUG package.

The CLI offers a thin, dependency-light interface for:
  - training (stubbed here; wire to your project trainer)
  - hyperparameter optimisation (delegates to scripts/tune_max.py)
  - printing top‑K HPO results exported by the tuner script

Design goals:
  - Keep the surface simple and self-documenting
  - Avoid importing heavy frameworks until subcommands run
  - Be explicit about side effects (e.g., MLflow env vars)
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import optuna
import typer

# Top-level app: subcommands are declared below.
app = typer.Typer(
    help="NoAug Criteria/Evidence: train, tune (HPO), eval, and show-best."
)


def _default_outdir(outdir: str | None) -> str:
    """Return the provided output directory or a sensible default.

    We keep default runtime artifacts under ./_runs to avoid polluting the
    repository tree and to make it easy to clean up experiments locally.
    """
    return outdir or "./_runs"


def _ensure_mlflow(outdir: str) -> None:
    """Ensure MLflow writes to an isolated file store under ``outdir``.

    The CLI uses a file URI to keep experiments local by default. Users can
    still override via environment variables when needed.
    """
    os.makedirs(outdir, exist_ok=True)
    mlruns = Path(outdir) / "mlruns"
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"file:{mlruns.as_posix()}")


@app.command()
def train(
    agent: str = typer.Option(..., help="criteria|evidence|share|joint"),
    model_name: str = typer.Option("bert-base-uncased"),
    outdir: str | None = typer.Option(None),
    epochs: int = typer.Option(3),
    seed: int = typer.Option(42),
    batch_size: int = typer.Option(16),
    grad_accum: int = typer.Option(1),
    config: str | None = typer.Option(
        None, help="Optional JSON config to override defaults"
    ),
    aug_enabled: bool = typer.Option(
        False,
        "--aug-enabled/--no-aug-enabled",
        help="Enable on-the-fly augmentation",
    ),
    aug_methods: str = typer.Option(
        "all",
        "--aug-methods",
        help="Comma separated augmenter IDs (use 'all' for full allowlist)",
    ),
    aug_p_apply: float = typer.Option(0.15, "--aug-p-apply"),
    aug_ops_per_sample: int = typer.Option(1, "--aug-ops-per-sample"),
    aug_max_replace: float = typer.Option(0.3, "--aug-max-replace"),
    aug_tfidf_model: str | None = typer.Option(None, "--aug-tfidf-model"),
    aug_reserved_map: str | None = typer.Option(None, "--aug-reserved-map"),
    antonym_guard: str = typer.Option(
        "off", "--antonym-guard", help="off|on_low_weight"
    ),
    loader_workers: int | None = typer.Option(None, "--loader-workers"),
    prefetch_factor: int | None = typer.Option(None, "--prefetch-factor"),
):
    """Run a training job.

    Notes
    -----
    This command is intentionally thin to keep the CLI fast to import and
    test. It prints the parsed configuration and environment so you can wire
    it to your training entrypoint as needed for your environment.
    """
    outdir = _default_outdir(outdir)
    _ensure_mlflow(outdir)

    methods = [m.strip() for m in aug_methods.split(",") if m.strip()]
    if not methods:
        methods = ["all"]
    ops = max(1, min(3, aug_ops_per_sample))

    typer.echo(
        f"[train] agent={agent} model={model_name} epochs={epochs} seed={seed} outdir={outdir}"
    )
    typer.echo(
        f"[train] augmentation enabled={aug_enabled} methods={methods} p_apply={aug_p_apply:.2f} ops={ops} max_replace={aug_max_replace:.2f} antonym_guard={antonym_guard}"
    )
    if loader_workers is not None or prefetch_factor is not None:
        typer.echo(
            f"[train] dataloader workers={loader_workers} prefetch={prefetch_factor}"
        )
    if aug_tfidf_model:
        typer.echo(f"[train] aug tfidf model={aug_tfidf_model}")
    if aug_reserved_map:
        typer.echo(f"[train] aug reserved map={aug_reserved_map}")
    if config:
        cfg = json.loads(Path(config).read_text())
        typer.echo(f"[train] loaded config keys: {list(cfg.keys())}")


@app.command()
def tune(
    agent: str = typer.Option(..., help="criteria|evidence|share|joint"),
    study: str = typer.Option(...),
    n_trials: int = typer.Option(200),
    timeout: int | None = typer.Option(None),
    parallel: int = typer.Option(1),
    outdir: str | None = typer.Option(None),
    storage: str | None = typer.Option(
        None, help="Optuna storage URL (e.g., sqlite:///path/to.db)"
    ),
    stage: str = typer.Option("A", "--stage", help="A | B | C"),
    from_study: str | None = typer.Option(
        None,
        "--from-study",
        help="Stage-B: previous Stage-A study | Stage-C: previous Stage-B study",
    ),
    pareto_limit: int = typer.Option(
        5, "--pareto-limit", help="Stage-C candidate count from Stage-B Pareto front"
    ),
):
    """Launch maximal HPO via ``scripts/tune_max.py``.

    The CLI merely marshals arguments and environment. Search spaces and the
    training bridge live inside the script to keep imports isolated.
    """
    outdir = _default_outdir(outdir)
    _ensure_mlflow(outdir)
    storage = storage or f"sqlite:///{Path('./_optuna/noaug.db').absolute()}"
    typer.echo(
        f"[tune] agent={agent} study={study} stage={stage} "
        f"n_trials={n_trials} parallel={parallel}"
    )
    if from_study:
        typer.echo(f"[tune] staging from study={from_study}")

    cmd = [
        "python",
        "scripts/tune_max.py",
        "--agent",
        agent,
        "--study",
        study,
        "--n-trials",
        str(n_trials),
        "--parallel",
        str(parallel),
        "--outdir",
        outdir,
        "--storage",
        storage,
        "--stage",
        stage,
    ]
    if timeout is not None:
        cmd += ["--timeout", str(timeout)]
    if from_study:
        cmd += ["--from-study", from_study]
    if stage.upper() == "C" and pareto_limit:
        cmd += ["--pareto-limit", str(pareto_limit)]
    subprocess.run(cmd, check=True)


@app.command("show-best")
def show_best(
    agent: str = typer.Option(...),
    study: str = typer.Option(...),
    storage: str | None = typer.Option(None),
    topk: int = typer.Option(5),
):
    """Print top-K trials directly from the Optuna study."""

    storage = storage or "sqlite:///./_optuna/noaug.db"
    study_obj = optuna.load_study(study_name=study, storage=storage)
    trials = sorted(
        [t for t in study_obj.get_trials(deepcopy=False) if t.values is not None],
        key=lambda t: -(t.user_attrs.get("primary", t.value or 0.0)),
    )
    if not trials:
        typer.echo("No completed trials found.")
        raise typer.Exit(0)
    for rank, trial in enumerate(trials[:topk], 1):
        value = trial.values if isinstance(trial.values, tuple) else trial.value
        params_json = trial.user_attrs.get("config_json")
        params = json.loads(params_json) if params_json else trial.params
        typer.echo(
            f"[{rank}] value={value} params={json.dumps(params, sort_keys=True)[:400]}"
        )


@app.command("hpo-max")
def hpo_max(
    agent: str = typer.Option(..., help="criteria|evidence|share|joint"),
    trials: int = typer.Option(100),
    epochs: int = typer.Option(6),
    multi_objective: bool = typer.Option(False),
    timeout_min: int | None = typer.Option(None),
    seeds: str = typer.Option("1"),
    sampler: str = typer.Option("auto"),
    pruner: str = typer.Option("asha"),
):
    """Run maximal HPO via scripts/tune_max.py."""

    cmd = [
        "python",
        "scripts/tune_max.py",
        "--agent",
        agent,
        "--trials",
        str(trials),
        "--epochs",
        str(epochs),
        "--seeds",
        seeds,
        "--sampler",
        sampler,
        "--pruner",
        pruner,
    ]
    if multi_objective:
        cmd.append("--multi-objective")
    if timeout_min is not None:
        cmd.extend(["--timeout-min", str(timeout_min)])
    subprocess.run(cmd, check=True)


@app.command("hpo-stage")
def hpo_stage(
    agent: str = typer.Option(..., help="criteria|evidence|share|joint"),
    stage0_trials: int = typer.Option(64),
    stage1_trials: int = typer.Option(32),
    stage2_trials: int = typer.Option(16),
    stage0_epochs: int = typer.Option(3),
    stage1_epochs: int = typer.Option(6),
    stage2_epochs: int = typer.Option(10),
    refit_epochs: int = typer.Option(12),
    seeds: str = typer.Option("1"),
):
    """Run staged HPO pipeline (S0→S1→S2→Refit)."""

    cmd = [
        "python",
        "scripts/run_hpo_stage.py",
        "--agent",
        agent,
        "--stage0-trials",
        str(stage0_trials),
        "--stage1-trials",
        str(stage1_trials),
        "--stage2-trials",
        str(stage2_trials),
        "--stage0-epochs",
        str(stage0_epochs),
        "--stage1-epochs",
        str(stage1_epochs),
        "--stage2-epochs",
        str(stage2_epochs),
        "--refit-epochs",
        str(refit_epochs),
        "--seeds",
        seeds,
    ]
    subprocess.run(cmd, check=True)


@app.command("tune-supermax")
def tune_supermax(
    agent: str = typer.Option(..., help="criteria|evidence|share|joint"),
    study: str = typer.Option(...),
    n_trials: int = typer.Option(5000, help="Very large default; override as needed"),
    parallel: int = typer.Option(4),
    outdir: str | None = typer.Option(None),
    storage: str | None = typer.Option(None),
):
    """Run very large HPO trials suitable for long-running servers.

    Configures 100-epoch trials and patience-based early stopping via env vars
    so downstream code does not need to be modified.
    """
    outdir = _default_outdir(outdir)
    _ensure_mlflow(outdir)
    storage = storage or f"sqlite:///{Path('./_optuna/noaug.db').absolute()}"
    env = os.environ.copy()
    env["HPO_EPOCHS"] = "100"
    env["HPO_PATIENCE"] = "20"
    cmd = [
        "python",
        "scripts/tune_max.py",
        "--agent",
        agent,
        "--study",
        study,
        "--n-trials",
        str(n_trials),
        "--parallel",
        str(parallel),
        "--outdir",
        outdir,
        "--storage",
        storage,
    ]
    subprocess.run(cmd, check=True, env=env)


def main():
    app()


if __name__ == "__main__":
    main()
