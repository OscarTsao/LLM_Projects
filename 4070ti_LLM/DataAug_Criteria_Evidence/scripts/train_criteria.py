"""Standalone training script for Criteria architecture.

This script trains the Criteria model from scratch or from a checkpoint.
It supports:
- Full reproducibility with seed control
- Mixed precision training (AMP with Float16/BFloat16)
- Hardware optimization (optimal DataLoader settings)
- Early stopping and checkpointing
- MLflow tracking
- Loading best HPO configuration

Usage:
    # Train with default config
    python scripts/train_criteria.py

    # Train with custom config
    python scripts/train_criteria.py task=criteria model=roberta_base

    # Train with best HPO config
    python scripts/train_criteria.py best_config=outputs/hpo_stage2/best_config.yaml

    # Train with custom settings
    python scripts/train_criteria.py training.num_epochs=20 training.batch_size=32
"""

import sys
from pathlib import Path

import hydra
import mlflow
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Project.Criteria.data.dataset import CriteriaDataset
from Project.Criteria.models.model import Model
from psy_agents_noaug.training.train_loop import Trainer
from psy_agents_noaug.utils.mlflow_utils import (
    configure_mlflow,
    log_config,
    resolve_artifact_location,
    resolve_tracking_uri,
)
from psy_agents_noaug.utils.reproducibility import (
    get_device,
    get_optimal_dataloader_kwargs,
    print_system_info,
    set_seed,
)


def create_dataloaders(
    cfg: DictConfig,
    tokenizer,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test data loaders."""
    print("\n" + "=" * 70)
    print("Loading Dataset".center(70))
    print("=" * 70)

    # Load full dataset
    dataset = CriteriaDataset(
        csv_path=cfg.dataset.path,
        tokenizer=tokenizer,
        text_column=cfg.dataset.text_column,
        label_column=cfg.dataset.label_column,
        max_length=cfg.dataset.max_length,
    )

    print(f"Total samples: {len(dataset)}")

    # Split dataset
    train_ratio = 0.8
    val_ratio = 0.1

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    generator = torch.Generator().manual_seed(cfg.training.seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    print(f"Train: {len(train_dataset)} samples")
    print(f"Validation: {len(val_dataset)} samples")
    print(f"Test: {len(test_dataset)} samples")

    # Get optimal DataLoader kwargs
    dataloader_kwargs = get_optimal_dataloader_kwargs(
        device=device,
        num_workers=cfg.dataset.get("num_workers"),
        pin_memory=cfg.dataset.get("pin_memory"),
        persistent_workers=cfg.dataset.get("persistent_workers"),
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        **dataloader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        **dataloader_kwargs,
    )

    return train_loader, val_loader, test_loader


def create_model(cfg: DictConfig) -> Model:
    """Create Criteria model."""
    print("\n" + "=" * 70)
    print("Creating Model".center(70))
    print("=" * 70)

    model = Model(
        model_name=cfg.model.pretrained_model,
        num_labels=2,  # Binary classification
        classifier_dropout=cfg.model.classifier_dropout,
        classifier_layer_num=cfg.model.classifier_layer_num,
        classifier_hidden_dims=cfg.model.get("classifier_hidden_dims"),
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: {cfg.model.pretrained_model}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    return model


def create_optimizer(cfg: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    """Create optimizer with weight decay applied correctly."""
    # Separate parameters for weight decay
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": cfg.training.optimizer.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    return torch.optim.AdamW(
        optimizer_grouped_parameters,
        lr=cfg.training.learning_rate,
    )


def create_scheduler(
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
):
    """Create learning rate scheduler."""
    from transformers import get_scheduler

    scheduler_name = cfg.training.scheduler.name
    warmup_steps = cfg.training.scheduler.warmup_steps

    if scheduler_name and scheduler_name != "none":
        return get_scheduler(
            name=scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )
    return None


@hydra.main(version_base=None, config_path="../configs/criteria", config_name="train")
def main(cfg: DictConfig):
    """
    Train Criteria model.

    This script provides:
    - Full reproducibility (deterministic training)
    - Hardware optimization (AMP, DataLoader settings)
    - Early stopping and checkpointing
    - MLflow experiment tracking
    """
    print("\n" + "=" * 70)
    print("CRITERIA ARCHITECTURE - TRAINING".center(70))
    print("=" * 70)

    # Load best HPO config if provided
    if hasattr(cfg, "best_config") and cfg.best_config:
        print(f"\nLoading best HPO config from: {cfg.best_config}")
        best_config = OmegaConf.load(cfg.best_config)
        cfg = OmegaConf.merge(cfg, best_config)

    project_root = Path(get_original_cwd())

    # Print system information
    print_system_info()

    # Set seed for reproducibility
    set_seed(
        cfg.training.seed,
        deterministic=cfg.training.get("deterministic", True),
        cudnn_benchmark=cfg.training.get("cudnn_benchmark", False),
    )

    # Get device
    device = get_device(prefer_cuda=True)

    # Create tokenizer
    print("\n" + "=" * 70)
    print("Loading Tokenizer".center(70))
    print("=" * 70)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model)
    print(f"Tokenizer: {cfg.model.pretrained_model}")

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(cfg, tokenizer, device)

    # Create model
    model = create_model(cfg)
    model = model.to(device)

    # Create optimizer
    optimizer = create_optimizer(cfg, model)

    # Create scheduler
    num_training_steps = len(train_loader) * cfg.training.epochs
    scheduler = create_scheduler(cfg, optimizer, num_training_steps)

    # Create loss function
    criterion = nn.CrossEntropyLoss()

    # Configure MLflow
    experiment_name = f"{cfg.project}-train"
    run_name = (
        f"{cfg.model.pretrained_model.split('/')[-1]}_epochs{cfg.training.epochs}"
    )
    tracking_uri = resolve_tracking_uri(cfg.mlflow.tracking_uri, project_root)
    artifact_location = resolve_artifact_location(
        cfg.mlflow.get("artifact_location"), project_root
    )

    run_id = configure_mlflow(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        artifact_location=artifact_location,
        config=cfg,
    )

    # Log configuration
    log_config(cfg)

    # Create output directory for checkpoints
    if (
        hasattr(cfg, "hydra")
        and hasattr(cfg.hydra, "run")
        and hasattr(cfg.hydra.run, "dir")
    ):
        output_dir = Path(cfg.hydra.run.dir) / "checkpoints"
    else:
        # Fallback when hydra config is not available
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = Path("outputs") / "criteria" / timestamp / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer
    print("\n" + "=" * 70)
    print("Training Configuration".center(70))
    print("=" * 70)
    print(f"Epochs: {cfg.training.epochs}")
    print(f"Batch size: {cfg.training.train_batch_size}")
    print(f"Learning rate: {cfg.training.learning_rate}")
    print(f"Optimizer: {cfg.training.optimizer.name}")
    print(f"Scheduler: {cfg.training.scheduler.name}")
    print(f"Gradient accumulation: {cfg.training.gradient_accumulation}")
    print("Mixed precision: enabled")
    print(f"Early stopping: {cfg.training.monitor_metric} (patience=1000 - effectively disabled)")

    patience = cfg.training.get("patience", 20)
    min_delta = cfg.training.get("min_delta", 0.0001)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=cfg.training.epochs,
        patience=patience,
        gradient_clip=cfg.training.max_grad_norm,
        gradient_accumulation_steps=cfg.training.gradient_accumulation,
        scheduler=scheduler,
        save_dir=output_dir,
        use_amp=True,
        amp_dtype="float16",
        early_stopping_metric=cfg.training.monitor_metric,
        early_stopping_mode=cfg.training.monitor_mode,
        min_delta=min_delta,
        logging_steps=cfg.training.logging_steps,
    )

    # Train model
    print("\n" + "=" * 70)
    print("Starting Training".center(70))
    print("=" * 70)

    final_metrics = trainer.train()

    # Load best model for test evaluation
    print("\n" + "=" * 70)
    print("Evaluating on Test Set".center(70))
    print("=" * 70)

    best_checkpoint_path = output_dir / "best_checkpoint.pt"
    if best_checkpoint_path.exists():
        checkpoint = torch.load(
            best_checkpoint_path, map_location=device, weights_only=True
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded best checkpoint from epoch {checkpoint['epoch'] + 1}")
    else:
        print("Warning: Best checkpoint not found, using final model")

    # Evaluate on test set
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate test metrics
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1_macro = f1_score(all_labels, all_preds, average="macro")
    test_f1_micro = f1_score(all_labels, all_preds, average="micro")
    test_precision = precision_score(all_labels, all_preds, average="macro")
    test_recall = recall_score(all_labels, all_preds, average="macro")

    test_metrics = {
        "test_accuracy": test_accuracy,
        "test_f1_macro": test_f1_macro,
        "test_f1_micro": test_f1_micro,
        "test_precision": test_precision,
        "test_recall": test_recall,
    }

    # Log test metrics to MLflow
    mlflow.log_metrics(test_metrics)

    # Print results
    print("\n" + "=" * 70)
    print("Test Results".center(70))
    print("=" * 70)
    print(f"Accuracy:  {test_accuracy:.4f}")
    print(f"F1 Macro:  {test_f1_macro:.4f}")
    print(f"F1 Micro:  {test_f1_micro:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")

    # End MLflow run
    mlflow.end_run()

    print("\n" + "=" * 70)
    print("Training Completed Successfully".center(70))
    print("=" * 70)
    print(f"Checkpoints saved to: {output_dir}")
    print(f"MLflow run ID: {run_id}")
    print(
        f"Best {cfg.training.monitor_metric}: {final_metrics[f'best_{cfg.training.monitor_metric}']:.4f}"
    )


if __name__ == "__main__":
    main()
