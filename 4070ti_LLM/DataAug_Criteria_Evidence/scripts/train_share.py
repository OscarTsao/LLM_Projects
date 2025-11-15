"""Standalone training script for Share architecture.

This script trains the Share model (shared encoder with dual heads) from scratch.
It supports:
- Multi-task learning (criteria classification + evidence span extraction)
- Full reproducibility with seed control
- Mixed precision training (AMP)
- Early stopping and checkpointing
- MLflow tracking
- Loading best HPO configuration

Usage:
    # Train with custom config
    python scripts/train_share.py training.epochs=100 training.batch_size=24

    # Quick test
    python scripts/train_share.py training.epochs=10
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

from Project.Share.data.dataset import ShareDataset
from Project.Share.models.model import Model
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
    dataset = ShareDataset(
        csv_path=cfg.dataset.path,
        tokenizer=tokenizer,
        context_column=cfg.dataset.get("context_column", "post_text"),
        answer_column=cfg.dataset.get("answer_column", "sentence_text"),
        label_column=cfg.dataset.get("label_column", "status"),
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
    """Create Share model with dual heads."""
    print("\n" + "=" * 70)
    print("Creating Model".center(70))
    print("=" * 70)

    model = Model(
        model_name=cfg.model.pretrained_model,
        criteria_num_labels=2,  # Binary classification for criteria
        criteria_dropout=cfg.model.get("classifier_dropout", 0.1),
        criteria_layer_num=cfg.model.get("classifier_layer_num", 1),
        criteria_hidden_dims=cfg.model.get("classifier_hidden_dims"),
        evidence_dropout=cfg.model.get("classifier_dropout", 0.1),
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Model: {cfg.model.pretrained_model}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print("Architecture: Shared encoder + Dual heads (classification + span)")

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


def train_one_epoch(model, train_loader, optimizer, criteria_criterion, span_criterion, device, scheduler=None, use_amp=True):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        criteria_labels = batch["labels"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast() if use_amp else torch.no_grad():
            # Forward pass
            outputs = model(input_ids, attention_mask, return_dict=True)
            criteria_logits = outputs["logits"]
            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]

            # Calculate losses
            criteria_loss = criteria_criterion(criteria_logits, criteria_labels)
            start_loss = span_criterion(start_logits, start_positions)
            end_loss = span_criterion(end_logits, end_positions)

            # Combined loss
            loss = criteria_loss + start_loss + end_loss

        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model, val_loader, criteria_criterion, span_criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            criteria_labels = batch["labels"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask, return_dict=True)
            criteria_logits = outputs["logits"]
            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]

            # Calculate losses
            criteria_loss = criteria_criterion(criteria_logits, criteria_labels)
            start_loss = span_criterion(start_logits, start_positions)
            end_loss = span_criterion(end_logits, end_positions)
            loss = criteria_loss + start_loss + end_loss

            total_loss += loss.item()

            # For criteria task
            preds = torch.argmax(criteria_logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(criteria_labels.cpu().numpy())

    # Calculate metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_micro = f1_score(all_labels, all_preds, average="micro")

    return {
        "val_loss": avg_loss,
        "val_accuracy": accuracy,
        "val_f1_macro": f1_macro,
        "val_f1_micro": f1_micro,
    }


@hydra.main(version_base=None, config_path="../configs/criteria", config_name="train")
def main(cfg: DictConfig):
    """Train Share model."""
    print("\n" + "=" * 70)
    print("SHARE ARCHITECTURE - TRAINING".center(70))
    print("=" * 70)

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

    # Create loss functions
    criteria_criterion = nn.CrossEntropyLoss()
    span_criterion = nn.CrossEntropyLoss()

    # Configure MLflow
    experiment_name = f"{cfg.project}-share-train"
    run_name = f"{cfg.model.pretrained_model.split('/')[-1]}_epochs{cfg.training.epochs}"
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

    log_config(cfg)

    # Create output directory
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path("outputs") / "share" / timestamp / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("Training Configuration".center(70))
    print("=" * 70)
    print(f"Epochs: {cfg.training.epochs}")
    print(f"Batch size: {cfg.training.train_batch_size}")
    print(f"Learning rate: {cfg.training.learning_rate}")
    print(f"Optimizer: {cfg.training.optimizer.name}")
    print(f"Scheduler: {cfg.training.scheduler.name}")
    print("Mixed precision: enabled")

    # Training loop
    best_f1 = 0.0
    patience_counter = 0
    patience = 3

    for epoch in range(cfg.training.epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.epochs}")

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criteria_criterion, span_criterion,
            device, scheduler, use_amp=True
        )

        # Evaluate
        val_metrics = evaluate(model, val_loader, criteria_criterion, span_criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['val_loss']:.4f}")
        print(f"Val F1 Macro: {val_metrics['val_f1_macro']:.4f}")

        # Log to MLflow
        mlflow.log_metrics({
            "train_loss": train_loss,
            **val_metrics
        }, step=epoch)

        # Save best model
        if val_metrics["val_f1_macro"] > best_f1:
            best_f1 = val_metrics["val_f1_macro"]
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1_macro": best_f1,
            }, output_dir / "best_checkpoint.pt")
            print(f"✓ Saved best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping after {epoch + 1} epochs")
            break

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("Evaluating on Test Set".center(70))
    print("=" * 70)

    checkpoint = torch.load(output_dir / "best_checkpoint.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criteria_criterion, span_criterion, device)

    print(f"\nTest Results:")
    print(f"Accuracy:  {test_metrics['val_accuracy']:.4f}")
    print(f"F1 Macro:  {test_metrics['val_f1_macro']:.4f}")
    print(f"F1 Micro:  {test_metrics['val_f1_micro']:.4f}")

    # Log test metrics
    mlflow.log_metrics({
        "test_accuracy": test_metrics["val_accuracy"],
        "test_f1_macro": test_metrics["val_f1_macro"],
        "test_f1_micro": test_metrics["val_f1_micro"],
    })

    mlflow.end_run()

    print("\n" + "=" * 70)
    print("Training Completed Successfully".center(70))
    print("=" * 70)
    print(f"Checkpoints saved to: {output_dir}")
    print(f"MLflow run ID: {run_id}")
    print(f"Best Val F1: {best_f1:.4f}")
    print(f"Test F1: {test_metrics['val_f1_macro']:.4f}")


if __name__ == "__main__":
    main()
