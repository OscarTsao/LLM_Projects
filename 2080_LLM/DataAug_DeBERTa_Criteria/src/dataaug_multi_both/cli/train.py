"""CLI entry point for training.

Implements FR-010: CLI entry point for training/HPO.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import click

from dataaug_multi_both.training.trainer import TrainerConfig, seed_everything
from dataaug_multi_both.training.checkpoint_manager import (
    CheckpointManager,
    CheckpointRetentionPolicy,
    CheckpointMetadata
)

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """DataAug Multi Both - Mental Health Criteria Detection."""
    pass


@cli.command()
@click.option(
    "--trial-id",
    required=True,
    help="Unique trial identifier"
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to configuration file (YAML/JSON)"
)
@click.option(
    "--model-name",
    default="microsoft/deberta-v3-base",
    help="Model name (fixed to microsoft/deberta-v3-base)"
)
@click.option(
    "--learning-rate",
    type=float,
    default=2e-5,
    help="Learning rate"
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    help="Batch size"
)
@click.option(
    "--max-epochs",
    type=int,
    default=10,
    help="Maximum number of epochs"
)
@click.option(
    "--seed",
    type=int,
    default=1337,
    help="Random seed for reproducibility"
)
@click.option(
    "--resume/--no-resume",
    default=True,
    help="Resume from latest checkpoint if available"
)
@click.option(
    "--experiments-dir",
    type=click.Path(),
    default="experiments",
    help="Base directory for experiments"
)
@click.option(
    "--keep-last-n",
    type=int,
    default=1,
    help="Number of last checkpoints to keep"
)
@click.option(
    "--keep-best-k",
    type=int,
    default=1,
    help="Number of best checkpoints to keep"
)
def train(
    trial_id: str,
    config: Optional[str],
    model_name: str,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    seed: int,
    resume: bool,
    experiments_dir: str,
    keep_last_n: int,
    keep_best_k: int
):
    """Train a single model.
    
    Example:
        python -m dataaug_multi_both.cli.train train --trial-id trial_001
    """
    click.echo(f"Starting training: {trial_id}")
    
    # Set seed
    seeds = seed_everything(seed)
    click.echo(f"Seeds: {seeds}")
    
    # Create trial directory
    trial_dir = Path(experiments_dir) / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer config
    trainer_config = TrainerConfig(
        trial_id=trial_id,
        optimization_metric="validation_f1_macro",
        seed=seed,
        max_epochs=max_epochs,
        resume_if_available=resume
    )
    
    # Create checkpoint manager
    policy = CheckpointRetentionPolicy(
        keep_last_n=keep_last_n,
        keep_best_k=keep_best_k
    )
    
    metadata = CheckpointMetadata(
        code_version="1.0.0",
        model_signature=model_name,
        head_configuration="dual-agent"
    )
    
    checkpoint_manager = CheckpointManager(
        trial_dir=trial_dir,
        policy=policy,
        compatibility=metadata
    )
    
    click.echo(f"Trial directory: {trial_dir}")
    click.echo(f"Retention policy: keep_last_n={keep_last_n}, keep_best_k={keep_best_k}")
    
    # TODO: Implement actual training loop
    click.echo("Training not yet implemented. This is a placeholder.")
    
    return 0


@cli.command()
@click.option(
    "--study-name",
    required=True,
    help="Name of the HPO study"
)
@click.option(
    "--n-trials",
    type=int,
    default=10,
    help="Number of trials to run"
)
@click.option(
    "--experiments-dir",
    type=click.Path(),
    default="experiments",
    help="Base directory for experiments"
)
@click.option(
    "--keep-last-n",
    type=int,
    default=1,
    help="Number of last checkpoints to keep per trial"
)
@click.option(
    "--keep-best-k",
    type=int,
    default=1,
    help="Number of best checkpoints to keep per trial"
)
def hpo(
    study_name: str,
    n_trials: int,
    experiments_dir: str,
    keep_last_n: int,
    keep_best_k: int
):
    """Run hyperparameter optimization.
    
    Example:
        python -m dataaug_multi_both.cli.train hpo --study-name study_001 --n-trials 10
    """
    click.echo(f"Starting HPO: {study_name}")
    click.echo(f"Number of trials: {n_trials}")
    click.echo(f"Experiments directory: {experiments_dir}")
    
    # Create experiments directory
    exp_dir = Path(experiments_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # TODO: Implement HPO loop
    click.echo("HPO not yet implemented. This is a placeholder.")
    
    return 0


@cli.command()
@click.option(
    "--trial-dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to trial directory"
)
def resume(trial_dir: str):
    """Resume training from checkpoint.
    
    Example:
        python -m dataaug_multi_both.cli.train resume --trial-dir experiments/trial_001
    """
    click.echo(f"Resuming from: {trial_dir}")
    
    # TODO: Implement resume logic
    click.echo("Resume not yet implemented. This is a placeholder.")
    
    return 0


@cli.command()
@click.option(
    "--experiments-dir",
    type=click.Path(exists=True),
    default="experiments",
    help="Base directory for experiments"
)
def status(experiments_dir: str):
    """Show status of experiments.
    
    Example:
        python -m dataaug_multi_both.cli.train status
    """
    exp_dir = Path(experiments_dir)
    
    if not exp_dir.exists():
        click.echo(f"Experiments directory not found: {exp_dir}")
        return 1
    
    # List trials
    trials = list(exp_dir.glob("trial_*"))
    
    click.echo(f"Experiments directory: {exp_dir}")
    click.echo(f"Number of trials: {len(trials)}")
    
    for trial_dir in sorted(trials):
        click.echo(f"  - {trial_dir.name}")
    
    return 0


def main():
    """Main entry point."""
    try:
        cli()
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

