from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import typer

app = typer.Typer(
    help="NoAug Criteria/Evidence: train, tune (HPO), eval, and show-best."
)


def _default_outdir(outdir: str | None) -> str:
    return outdir or "./_runs"


def _ensure_mlflow(outdir: str):
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
):
    """Thin wrapper expected to call the project's trainer; implement as needed."""
    outdir = _default_outdir(outdir)
    _ensure_mlflow(outdir)
    # If you already have a training entrypoint, call it here; otherwise print a stub:
    typer.echo(
        f"[train] agent={agent} model={model_name} epochs={epochs} seed={seed} outdir={outdir}"
    )
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
    multi_objective: bool = typer.Option(False),
):
    """Invoke the maximal HPO driver (scripts/tune_max.py)."""
    outdir = _default_outdir(outdir)
    _ensure_mlflow(outdir)
    storage = storage or f"sqlite:///{Path('./_optuna/noaug.db').absolute()}"
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
    if timeout is not None:
        cmd += ["--timeout", str(timeout)]
    if multi_objective:
        cmd += ["--multi-objective"]
    subprocess.run(cmd, check=True)


@app.command("show-best")
def show_best(
    agent: str = typer.Option(...),
    study: str = typer.Option(...),
    outdir: str | None = typer.Option(None),
    topk: int = typer.Option(5),
):
    """Pretty-print top-K trials exported by tune_max.py."""
    outdir = _default_outdir(outdir)
    path = Path(outdir) / f"{agent}_{study}_topk.json"
    if not path.exists():
        typer.echo(f"Not found: {path}")
        raise typer.Exit(1)
    data = json.loads(path.read_text())
    for i, t in enumerate(data[:topk], 1):
        val = t.get("value")
        params = t.get("params", {})
        typer.echo(f"[{i}] value={val:.4f}  params={json.dumps(params)[:500]}...")


@app.command("tune-supermax")
def tune_supermax(
    agent: str = typer.Option(..., help="criteria|evidence|share|joint"),
    study: str = typer.Option(...),
    n_trials: int = typer.Option(5000, help="Very large default; override as needed"),
    parallel: int = typer.Option(4),
    outdir: str | None = typer.Option(None),
    storage: str | None = typer.Option(None),
):
    """100-epoch trials with EarlyStopping(patience=20). Big n_trials by default."""
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
