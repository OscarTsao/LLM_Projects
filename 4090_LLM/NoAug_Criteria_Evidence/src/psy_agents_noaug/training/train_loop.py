"""Comprehensive training loop with MLflow, AMP, and early stopping."""

import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class Trainer:
    """
    Production-ready training loop with comprehensive features.

    Features:
    - Mixed precision training (AMP)
    - Gradient accumulation
    - Gradient clipping
    - Early stopping on validation F1 macro
    - Learning rate scheduling
    - MLflow metric logging
    - Checkpoint management (best and last)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        num_epochs: int = 10,
        patience: int = 3,
        gradient_clip: float | None = 1.0,
        gradient_accumulation_steps: int = 1,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        save_dir: Path | None = None,
        use_amp: bool = True,
        amp_dtype: str = "float16",
        early_stopping_metric: str = "val_f1_macro",
        early_stopping_mode: str = "max",
        min_delta: float = 0.0001,
        logging_steps: int = 100,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer
            criterion: Loss function
            device: Device to train on
            num_epochs: Maximum number of epochs
            patience: Early stopping patience
            gradient_clip: Gradient clipping value
            gradient_accumulation_steps: Steps for gradient accumulation
            scheduler: Optional learning rate scheduler
            save_dir: Directory to save checkpoints
            use_amp: Enable automatic mixed precision
            amp_dtype: AMP dtype (float16 or bfloat16)
            early_stopping_metric: Metric for early stopping
            early_stopping_mode: Mode for early stopping (max or min)
            min_delta: Minimum change to qualify as improvement
            logging_steps: Log metrics every N steps
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience
        self.gradient_clip = gradient_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.scheduler = scheduler
        self.save_dir = Path(save_dir) if save_dir else None
        self.use_amp = use_amp and torch.cuda.is_available()
        self.amp_dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16
        self.early_stopping_metric = early_stopping_metric
        self.early_stopping_mode = early_stopping_mode
        self.min_delta = min_delta
        self.logging_steps = logging_steps
        self.non_blocking = device.type == "cuda"

        use_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = GradScaler("cuda", enabled=use_scaler)

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.best_metric_value = (
            float("-inf") if early_stopping_mode == "max" else float("inf")
        )
        self.epochs_without_improvement = 0
        self.training_history = []
        self.global_step = 0

    def train_epoch(self, epoch: int) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            epoch: Current epoch number

        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}", leave=False
        )

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            input_ids = batch["input_ids"].to(
                self.device, non_blocking=self.non_blocking
            )
            attention_mask = batch["attention_mask"].to(
                self.device, non_blocking=self.non_blocking
            )
            labels = batch["labels"].to(
                self.device, non_blocking=self.non_blocking
            )

            # Forward pass with AMP
            with autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)

                # Scale loss for gradient accumulation
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.scaler is not None and self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.gradient_clip:
                    if self.scaler is not None and self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.gradient_clip
                    )

                # Optimizer step
                if self.scaler is not None and self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                # Log to MLflow
                if self.global_step % self.logging_steps == 0:
                    mlflow.log_metrics(
                        {
                            "train_loss_step": loss.item()
                            * self.gradient_accumulation_steps,
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                        },
                        step=self.global_step,
                    )

            # Track metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss.item() * self.gradient_accumulation_steps:.4f}",
                    "acc": f"{correct / total:.4f}",
                }
            )

        avg_loss = total_loss / max(len(self.train_loader), 1)
        accuracy = correct / total

        return {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
        }

    def validate(self) -> dict[str, float]:
        """
        Validate on validation set with comprehensive metrics.

        Returns:
            Dictionary with validation metrics including F1 scores
        """
        self.model.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move batch to device
                input_ids = batch["input_ids"].to(
                    self.device, non_blocking=self.non_blocking
                )
                attention_mask = batch["attention_mask"].to(
                    self.device, non_blocking=self.non_blocking
                )
                labels = batch["labels"].to(
                    self.device, non_blocking=self.non_blocking
                )

                # Forward pass
                with autocast("cuda", enabled=self.use_amp, dtype=self.amp_dtype):
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)

                # Track metrics
                total_loss += loss.item()
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        probabilities = np.array(all_probabilities)

        avg_loss = total_loss / max(len(self.val_loader), 1)
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
        precision_macro = precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        )
        recall_macro = recall_score(
            all_labels, all_preds, average="macro", zero_division=0
        )

        auroc_macro = None
        try:
            if probabilities.size == 0:
                auroc_macro = None
            elif probabilities.shape[1] == 2:
                auroc_macro = roc_auc_score(all_labels, probabilities[:, 1])
            else:
                auroc_macro = roc_auc_score(
                    all_labels, probabilities, average="macro", multi_class="ovr"
                )
        except Exception:
            auroc_macro = None

        return {
            "val_loss": avg_loss,
            "val_accuracy": accuracy,
            "val_f1_macro": f1_macro,
            "val_f1_micro": f1_micro,
            "val_precision_macro": precision_macro,
            "val_recall_macro": recall_macro,
            "val_auroc_macro": auroc_macro,
        }

    def save_checkpoint(
        self, epoch: int, metrics: dict[str, float], is_best: bool = False
    ):
        """Save model checkpoint with full training state."""
        if not self.save_dir:
            return

        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_metric_value": self.best_metric_value,
        }

        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        if self.scaler:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()

        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / "latest_checkpoint.pt")

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / "best_checkpoint.pt")
            print(
                f"  Saved best checkpoint with {self.early_stopping_metric}: {metrics[self.early_stopping_metric]:.4f}"
            )

    def check_improvement(self, current_value: float) -> bool:
        """Check if current metric value is an improvement."""
        if self.early_stopping_mode == "max":
            return current_value > (self.best_metric_value + self.min_delta)
        else:
            return current_value < (self.best_metric_value - self.min_delta)

    def train(self) -> dict[str, float]:
        """
        Run full training loop with early stopping.

        Returns:
            Dictionary with final metrics
        """
        print(f"Starting training for up to {self.num_epochs} epochs...")
        print(
            f"Early stopping: {self.early_stopping_metric} ({self.early_stopping_mode}), patience={self.patience}"
        )

        for epoch in range(self.num_epochs):
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate()

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            metrics["epoch"] = epoch + 1
            metrics["lr"] = self.optimizer.param_groups[0]["lr"]
            metrics["epoch_time"] = time.time() - epoch_start

            # Log to MLflow
            mlflow.log_metrics(metrics, step=epoch)

            # Save history
            self.training_history.append(metrics)

            # Print progress
            print(
                f"Epoch {epoch + 1}/{self.num_epochs} - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_accuracy']:.4f}, "
                f"Val F1 Macro: {val_metrics['val_f1_macro']:.4f}"
            )

            # Check for improvement
            current_metric = metrics[self.early_stopping_metric]
            improved = self.check_improvement(current_metric)

            if improved:
                self.best_metric_value = current_metric
                self.epochs_without_improvement = 0
                self.save_checkpoint(epoch, metrics, is_best=True)
            else:
                self.epochs_without_improvement += 1
                self.save_checkpoint(epoch, metrics, is_best=False)

            # Early stopping
            if self.epochs_without_improvement >= self.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

        # Log best metrics
        mlflow.log_metrics(
            {
                f"best_{self.early_stopping_metric}": self.best_metric_value,
                "total_epochs": len(self.training_history),
            }
        )

        return {
            f"best_{self.early_stopping_metric}": self.best_metric_value,
            "total_epochs": len(self.training_history),
        }
