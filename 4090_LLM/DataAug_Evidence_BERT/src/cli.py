from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from .config_utils import resolve_config_paths
from .hpo import run_hpo
from .train import run_training

app = typer.Typer(add_completion=False, help="Thin CLI for Evidence QA training and HPO.")


def _load_config(overrides: Optional[List[str]] = None):
    overrides = overrides or []
    config_dir = str(Path(__file__).resolve().parents[1] / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=list(overrides))
    cfg = resolve_config_paths(cfg)
    return cfg


@app.command()
def train(overrides: List[str] = typer.Argument(None, help="Hydra-style overrides")) -> None:
    """Run a single training job."""
    cfg = _load_config(overrides)
    run_training(cfg)


@app.command()
def hpo(overrides: List[str] = typer.Argument(None, help="Hydra-style overrides")) -> None:
    """Launch Optuna-based hyperparameter search."""
    cfg = _load_config(overrides)
    run_hpo(cfg)


def main():
    app()


if __name__ == "__main__":
    main()
