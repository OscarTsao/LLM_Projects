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

    # Import required modules
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
        from dataaug_multi_both.data.dataset_loader import DatasetLoader, DatasetConfig
    except ImportError as e:
        click.echo(f"Error importing required modules: {e}", err=True)
        click.echo("Please ensure torch, transformers, and datasets are installed.", err=True)
        return 1

    # Load dataset
    click.echo("\nLoading dataset...")
    try:
        dataset_config = DatasetConfig(
            id="irlab-udc/redsm5",
            revision=None,
            splits={"train": "train", "validation": "validation", "test": "test"},
            streaming=False,
            cache_dir=None
        )
        dataset_loader = DatasetLoader()
        dataset_dict = dataset_loader.load(dataset_config)

        click.echo(f"Train samples: {len(dataset_dict['train'])}")
        click.echo(f"Validation samples: {len(dataset_dict['validation'])}")
        click.echo(f"Test samples: {len(dataset_dict['test'])}")
    except Exception as e:
        click.echo(f"Error loading dataset: {e}", err=True)
        return 1

    # Initialize tokenizer
    click.echo(f"\nLoading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    tokenized_datasets = {
        split: ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)
        for split, ds in dataset_dict.items()
    }

    # Convert to PyTorch format
    for split in tokenized_datasets:
        tokenized_datasets[split].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Create data loaders
    train_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=batch_size,
        shuffle=False
    )

    # Initialize model
    click.echo(f"\nInitializing model: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model = model.to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Initialize scheduler
    num_training_steps = len(train_loader) * max_epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    # Create trainer
    trainer = Trainer(
        config=trainer_config,
        checkpoint_manager=checkpoint_manager
    )

    # Prepare training state (handles resume if enabled)
    state = trainer.prepare()
    click.echo(f"Starting from epoch {state.epoch}, step {state.global_step}")

    # Training loop
    click.echo(f"\nTraining for {max_epochs} epochs...")
    criterion = nn.CrossEntropyLoss()
    best_val_f1 = state.best_metric or 0.0

    for epoch in range(state.epoch, max_epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            state.global_step += 1

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

                total_val_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)

        # Calculate F1 score
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        click.echo(
            f"Epoch {epoch + 1}/{max_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val F1: {val_f1:.4f}"
        )

        # Update state
        state.epoch = epoch + 1
        state.metrics = {
            "train_loss": avg_train_loss,
            "validation_loss": avg_val_loss,
            "validation_f1_macro": val_f1
        }

        # Save checkpoint if improved
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            state.best_metric = best_val_f1
            click.echo(f"  New best F1: {best_val_f1:.4f} - Saving checkpoint")
            trainer.save_state(state, metric_value=val_f1)

            # Save model
            model.save_pretrained(trial_dir / "best_model")
            tokenizer.save_pretrained(trial_dir / "best_model")

    click.echo(f"\nTraining complete! Best validation F1: {best_val_f1:.4f}")
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

    # Import required modules
    try:
        import optuna
        from dataaug_multi_both.hpo.search_space import create_search_space
        from dataaug_multi_both.hpo.trial_executor import TrialExecutor, TrialSpec, TrialResult
    except ImportError as e:
        click.echo(f"Error importing HPO modules: {e}", err=True)
        click.echo("Please ensure optuna is installed.", err=True)
        return 1

    # Define objective function for HPO
    def run_trial_fn(spec: TrialSpec) -> TrialResult:
        """Run a single trial with given hyperparameters."""
        trial_id = spec.ensure_id()
        config = spec.config

        click.echo(f"\nTrial {trial_id}:")
        click.echo(f"  Learning rate: {config['learning_rate']}")
        click.echo(f"  Batch size: {config['batch_size']}")
        click.echo(f"  Max epochs: {config.get('max_epochs', 10)}")

        # Import training modules
        import torch
        from torch.utils.data import DataLoader
        from torch import nn
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
        from dataaug_multi_both.data.dataset_loader import DatasetLoader, DatasetConfig
        from sklearn.metrics import f1_score
        import time

        start_time = time.time()

        try:
            # Load dataset
            dataset_config = DatasetConfig(
                id="irlab-udc/redsm5",
                revision=None,
                splits={"train": "train", "validation": "validation", "test": "test"},
                streaming=False,
                cache_dir=None
            )
            dataset_loader = DatasetLoader()
            dataset_dict = dataset_loader.load(dataset_config)

            # Initialize tokenizer
            model_name = config.get('model_name', 'microsoft/deberta-v3-base')
            tokenizer = AutoTokenizer.from_pretrained(model_name)

            # Tokenize datasets
            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    padding="max_length",
                    truncation=True,
                    max_length=512
                )

            tokenized_datasets = {
                split: ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)
                for split, ds in dataset_dict.items()
            }

            # Convert to PyTorch format
            for split in tokenized_datasets:
                tokenized_datasets[split].set_format(
                    type="torch",
                    columns=["input_ids", "attention_mask", "label"]
                )

            # Create data loaders
            train_loader = DataLoader(
                tokenized_datasets["train"],
                batch_size=config['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                tokenized_datasets["validation"],
                batch_size=config['batch_size'],
                shuffle=False
            )

            # Initialize model
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
            model = model.to(device)

            # Initialize optimizer
            optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

            # Initialize scheduler
            max_epochs = config.get('max_epochs', 10)
            num_training_steps = len(train_loader) * max_epochs
            scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=int(0.1 * num_training_steps),
                num_training_steps=num_training_steps
            )

            # Training loop
            criterion = nn.CrossEntropyLoss()
            best_val_f1 = 0.0

            for epoch in range(max_epochs):
                # Training phase
                model.train()
                for batch in train_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)

                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # Validation phase
                model.eval()
                all_preds = []
                all_labels = []

                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch["input_ids"].to(device)
                        attention_mask = batch["attention_mask"].to(device)
                        labels = batch["label"].to(device)

                        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                        preds = torch.argmax(outputs.logits, dim=-1)
                        all_preds.extend(preds.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                val_f1 = f1_score(all_labels, all_preds, average="macro")

                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1

            duration = time.time() - start_time
            click.echo(f"  Best validation F1: {best_val_f1:.4f}")

            return TrialResult(
                trial_id=trial_id,
                metric=best_val_f1,
                status="completed",
                duration_seconds=duration
            )

        except Exception as e:
            duration = time.time() - start_time
            click.echo(f"  Trial failed: {e}", err=True)
            return TrialResult(
                trial_id=trial_id,
                metric=None,
                status=f"failed:{e.__class__.__name__}",
                duration_seconds=duration
            )

    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=None  # In-memory
    )

    # Define search space
    def objective(trial):
        config = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-6, 1e-4, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
            'max_epochs': trial.suggest_int('max_epochs', 3, 10),
            'model_name': 'microsoft/deberta-v3-base'
        }

        spec = TrialSpec(config=config, trial_id=f"trial_{trial.number:03d}")
        result = run_trial_fn(spec)

        if result.metric is None:
            raise optuna.exceptions.TrialPruned()

        return result.metric

    # Run optimization
    click.echo("\nStarting hyperparameter optimization...")
    study.optimize(objective, n_trials=n_trials)

    # Print results
    click.echo(f"\nBest trial:")
    click.echo(f"  Value (F1): {study.best_trial.value:.4f}")
    click.echo(f"  Params:")
    for key, value in study.best_trial.params.items():
        click.echo(f"    {key}: {value}")

    # Save best parameters
    best_params_file = exp_dir / f"{study_name}_best_params.json"
    import json
    with open(best_params_file, 'w') as f:
        json.dump(study.best_trial.params, f, indent=2)

    click.echo(f"\nBest parameters saved to: {best_params_file}")

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

    trial_path = Path(trial_dir)

    if not trial_path.exists():
        click.echo(f"Error: Trial directory not found: {trial_path}", err=True)
        return 1

    # Import required modules
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
        from dataaug_multi_both.data.dataset_loader import DatasetLoader, DatasetConfig
        from dataaug_multi_both.training.trainer import Trainer
        from dataaug_multi_both.training.checkpoint_manager import CheckpointManager
    except ImportError as e:
        click.echo(f"Error importing required modules: {e}", err=True)
        return 1

    # Create checkpoint manager
    checkpoint_manager = CheckpointManager(
        trial_dir=trial_path,
        policy=None,  # Will use defaults
        compatibility=None
    )

    # Try to load the latest checkpoint
    try:
        checkpoint_record, checkpoint_payload = checkpoint_manager.load_latest_checkpoint()
        click.echo(f"Found checkpoint at epoch {checkpoint_record.epoch}")
        click.echo(f"Best metric: {checkpoint_payload.get('best_metric', 'N/A')}")
    except FileNotFoundError:
        click.echo("No checkpoint found in trial directory.", err=True)
        return 1

    # Check if model directory exists
    model_dir = trial_path / "best_model"
    if not model_dir.exists():
        click.echo("No saved model found in trial directory.", err=True)
        return 1

    # Load dataset
    click.echo("\nLoading dataset...")
    try:
        dataset_config = DatasetConfig(
            id="irlab-udc/redsm5",
            revision=None,
            splits={"train": "train", "validation": "validation", "test": "test"},
            streaming=False,
            cache_dir=None
        )
        dataset_loader = DatasetLoader()
        dataset_dict = dataset_loader.load(dataset_config)
    except Exception as e:
        click.echo(f"Error loading dataset: {e}", err=True)
        return 1

    # Load tokenizer and model
    click.echo(f"\nLoading model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model = model.to(device)

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )

    tokenized_datasets = {
        split: ds.map(tokenize_function, batched=True, remove_columns=ds.column_names)
        for split, ds in dataset_dict.items()
    }

    # Convert to PyTorch format
    for split in tokenized_datasets:
        tokenized_datasets[split].set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Get training parameters from checkpoint
    start_epoch = checkpoint_payload.get('epoch', 0)
    global_step = checkpoint_payload.get('global_step', 0)
    best_metric = checkpoint_payload.get('best_metric', 0.0)

    # Create data loaders with same batch size as before (default to 16)
    batch_size = 16
    train_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=batch_size,
        shuffle=True
    )
    val_loader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=batch_size,
        shuffle=False
    )

    # Initialize optimizer and scheduler
    learning_rate = 2e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    max_epochs = 10  # Continue with remaining epochs
    remaining_epochs = max_epochs - start_epoch
    num_training_steps = len(train_loader) * remaining_epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps
    )

    click.echo(f"\nResuming training from epoch {start_epoch}")
    click.echo(f"Best metric so far: {best_metric:.4f}")
    click.echo(f"Remaining epochs: {remaining_epochs}")

    # Continue training
    criterion = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, max_epochs):
        # Training phase
        model.train()
        total_train_loss = 0

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_train_loss += loss.item()
            global_step += 1

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation phase
        model.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

                total_val_loss += loss.item()
                preds = torch.argmax(outputs.logits, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_loader)

        # Calculate F1 score
        from sklearn.metrics import f1_score
        val_f1 = f1_score(all_labels, all_preds, average="macro")

        click.echo(
            f"Epoch {epoch + 1}/{max_epochs} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"Val F1: {val_f1:.4f}"
        )

        # Save checkpoint if improved
        if val_f1 > best_metric:
            best_metric = val_f1
            click.echo(f"  New best F1: {best_metric:.4f} - Saving checkpoint")

            # Save checkpoint
            checkpoint_payload = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "best_metric": best_metric,
                "metrics": {
                    "train_loss": avg_train_loss,
                    "validation_loss": avg_val_loss,
                    "validation_f1_macro": val_f1
                }
            }
            checkpoint_manager.save_checkpoint(
                state=checkpoint_payload,
                epoch=epoch + 1,
                metric_value=val_f1
            )

            # Save model
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)

    click.echo(f"\nTraining complete! Best validation F1: {best_metric:.4f}")
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

