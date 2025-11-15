"""
Training loop with checkpointing and metric tracking.

Implements a complete training pipeline with:
- Forward/backward passes with gradient accumulation
- Validation loop with metrics
- Checkpoint saving with retention policies
- MLflow metric logging
- Early stopping
- Resume from checkpoint capability
- Learning rate scheduling
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import torch
import torch.nn as nn
from dataaug_multi_both.checkpoints.manager import CheckpointManager
from dataaug_multi_both.checkpoints.retention import RetentionPolicy
from dataaug_multi_both.utils.resource_monitor import ResourceMonitor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


class EvidenceExtractionTrainer:
    """
    Trainer for evidence extraction models.

    Handles complete training lifecycle including:
    - Training and validation loops
    - Checkpoint management with retention
    - Metric tracking and logging
    - Early stopping
    - Resume capability
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        output_dir: str | Path,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
        scheduler: _LRScheduler | None = None,
        retention_policy: RetentionPolicy | None = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        early_stopping_patience: int | None = None,
        metric_for_best_model: str = "val_f1",
        mlflow_tracking: bool = True,
        use_amp: bool = True,
        allow_tf32: bool = True,
        non_blocking: bool = True,
        amp_dtype: torch.dtype | None = None,
        scheduler_step_per_batch: bool = False,
        resource_monitor: ResourceMonitor | None = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Model to train
            optimizer: Optimizer for training
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            output_dir: Directory for checkpoints and logs
            device: Device to train on
            scheduler: Learning rate scheduler (optional)
            retention_policy: Checkpoint retention policy
            gradient_accumulation_steps: Steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            early_stopping_patience: Epochs without improvement before stopping
            metric_for_best_model: Metric to track for best model
            mlflow_tracking: Enable MLflow logging
            use_amp: Enable automatic mixed precision when CUDA is available
            allow_tf32: Enable TF32 tensor cores on Ampere+/Ada GPUs
            non_blocking: Use non-blocking CUDA copies when transferring tensors
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.scheduler = scheduler
        self.scheduler_step_per_batch = scheduler_step_per_batch

        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.early_stopping_patience = early_stopping_patience
        self.metric_for_best_model = metric_for_best_model
        self.mlflow_tracking = mlflow_tracking
        self.use_amp = bool(use_amp) and torch.cuda.is_available()
        self.non_blocking = bool(non_blocking) and torch.cuda.is_available()
        self.amp_dtype = amp_dtype if self.use_amp else None
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.resource_monitor = resource_monitor or ResourceMonitor()

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            output_dir=output_dir,
            retention_policy=retention_policy,
        )

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float("-inf")
        self.epochs_without_improvement = 0

        if allow_tf32 and torch.cuda.is_available():
            try:
                # Use new API for PyTorch 2.1+
                torch.backends.cudnn.conv.fp32_precision = 'tf32'
                torch.backends.cuda.matmul.fp32_precision = 'tf32'
            except AttributeError:
                # Fallback for older PyTorch versions (<2.1)
                try:
                    torch.set_float32_matmul_precision("high")
                except AttributeError:
                    pass
                # Use legacy API for older versions
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

        logger.info(
            f"Trainer initialized: device={device}, "
            f"grad_accum={gradient_accumulation_steps}, "
            f"early_stopping={early_stopping_patience}, "
            f"amp={'enabled' if self.use_amp else 'disabled'}, "
            f"tf32={'enabled' if allow_tf32 and torch.cuda.is_available() else 'disabled'}"
        )

    def _resolve_metric_alias(self, name: str) -> str:
        """Map alias names to actual metric keys logged/returned by this trainer."""
        aliases = {
            "span_f1": "f1",
            "val_span_f1": "val_f1",
        }
        return aliases.get(name, name)

    def _get_metric_value(
        self, metrics: dict[str, float], metric_name: str, allow_missing: bool = False
    ) -> float:
        """
        Get metric value with clear error handling.

        Args:
            metrics: Dictionary of available metrics
            metric_name: Name of metric to retrieve (can include 'val_' prefix)
            allow_missing: If True, return 0.0 for missing metrics; if False, raise ValueError

        Returns:
            Metric value

        Raises:
            ValueError: If metric not found and allow_missing=False
        """
        resolved_name = self._resolve_metric_alias(metric_name)
        # Remove 'val_' prefix if present, since validation metrics are stored without it
        clean_name = resolved_name.replace("val_", "")

        if clean_name in metrics:
            return metrics[clean_name]

        if allow_missing:
            logger.warning(
                f"Metric '{metric_name}' (resolved to '{clean_name}') not found in metrics. "
                f"Available: {list(metrics.keys())}. Using 0.0 as default."
            )
            return 0.0

        available = ", ".join(metrics.keys())
        raise ValueError(
            f"Metric '{metric_name}' (resolved to '{clean_name}') not found in metrics. "
            f"Available metrics: {available}"
        )

    def train(
        self,
        num_epochs: int,
        resume_from_checkpoint: str | Path | None = None,
        trial: Any | None = None,
        report_metric: str = "f1",
    ) -> dict[str, Any]:
        """
        Run complete training loop.

        Args:
            num_epochs: Number of epochs to train
            resume_from_checkpoint: Path to checkpoint to resume from
            trial: Optional Optuna trial for pruning/reporting (None when not using HPO)
            report_metric: Validation metric name to report to Optuna (e.g., 'f1')

        Returns:
            Training summary with best metrics
        """
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            self._resume_from_checkpoint(resume_from_checkpoint)

        logger.info(f"Starting training for {num_epochs} epochs")

        try:
            for epoch in range(self.current_epoch, num_epochs):
                self.current_epoch = epoch

                # Begin epoch resource snapshot
                try:
                    if self.resource_monitor:
                        self.resource_monitor.begin_epoch()
                except Exception:
                    pass

                # Training epoch
                train_metrics = self._train_epoch()
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"train_loss: {train_metrics['loss']:.4f}, "
                    f"train_f1: {train_metrics.get('f1', 0.0):.4f}"
                )

                # Validation epoch
                val_metrics = self._validate_epoch()
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs} - "
                    f"val_loss: {val_metrics['loss']:.4f}, "
                    f"val_f1: {val_metrics.get('f1', 0.0):.4f}"
                )

                # Log to MLflow
                if self.mlflow_tracking:
                    self._log_metrics(epoch, train_metrics, val_metrics)
                    # Also log resource utilization per epoch
                    try:
                        if self.resource_monitor:
                            self.resource_monitor.log_epoch_end(epoch)
                    except Exception:
                        pass

                # Report intermediate result to Optuna (for pruning), if trial provided
                if trial is not None:
                    try:
                        # Get metric value for reporting (allow missing for HPO compatibility)
                        value = self._get_metric_value(val_metrics, report_metric, allow_missing=True)
                        import optuna  # type: ignore

                        trial.report(value, step=epoch)
                        if trial.should_prune():
                            raise optuna.TrialPruned(
                                f"Pruned at epoch {epoch + 1} with {report_metric}={value:.4f}"
                            )
                    except Exception as e:
                        logger.debug(f"Optuna reporting/pruning skipped due to: {e}")

                # Get metric for best model tracking (fail fast if missing)
                current_metric = self._get_metric_value(
                    val_metrics, self.metric_for_best_model, allow_missing=False
                )

                # Save checkpoint
                self.checkpoint_manager.save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    step=self.global_step,
                    metric_value=current_metric,
                    metadata={
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "lr": self.optimizer.param_groups[0]["lr"],
                    },
                )

                # Check for improvement
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.epochs_without_improvement = 0
                    logger.info(f"New best {self.metric_for_best_model}: {current_metric:.4f}")
                else:
                    self.epochs_without_improvement += 1
                    logger.info(f"No improvement for {self.epochs_without_improvement} epochs")

                # Early stopping check
                if (
                    self.early_stopping_patience
                    and self.epochs_without_improvement >= self.early_stopping_patience
                ):
                    logger.info(
                        f"Early stopping triggered after {self.early_stopping_patience} epochs without improvement"
                    )
                    break

                # Learning rate scheduling
                if self.scheduler and not self.scheduler_step_per_batch:
                    self.scheduler.step()

        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}")
            raise

        # Load best model for final evaluation
        best_checkpoint = self.checkpoint_manager.load_best_checkpoint()
        if best_checkpoint:
            self.model.load_state_dict(best_checkpoint["model_state_dict"])
            logger.info(
                f"Loaded best model with {self.metric_for_best_model}={self.best_metric:.4f}"
            )

        return {
            "best_metric": self.best_metric,
            "total_epochs": self.current_epoch + 1,
            "global_step": self.global_step,
        }

    def _train_epoch(self) -> dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Training Epoch {self.current_epoch + 1}",
            leave=False,
        )

        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device, non_blocking=self.non_blocking)
            attention_mask = batch["attention_mask"].to(self.device, non_blocking=self.non_blocking)
            start_positions = batch["start_positions"].to(
                self.device, non_blocking=self.non_blocking
            )
            end_positions = batch["end_positions"].to(self.device, non_blocking=self.non_blocking)

            with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                # Forward pass
                start_logits, end_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Compute loss
                loss = self._compute_loss(start_logits, end_logits, start_positions, end_positions)
                loss = loss / self.gradient_accumulation_steps

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                if self.scheduler and self.scheduler_step_per_batch:
                    self.scheduler.step()

                self.optimizer.zero_grad(set_to_none=True)

                self.global_step += 1

            total_loss += loss.detach().item() * self.gradient_accumulation_steps
            num_batches += 1

            # Update progress bar
            progress_bar.set_postfix({"loss": total_loss / num_batches})

        return {"loss": total_loss / num_batches}

    def _validate_epoch(self) -> dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_start_preds = []
        all_end_preds = []
        all_start_labels = []
        all_end_labels = []

        progress_bar = tqdm(
            self.val_dataloader,
            desc=f"Validation Epoch {self.current_epoch + 1}",
            leave=False,
        )

        with torch.no_grad():
            for batch in progress_bar:
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device, non_blocking=self.non_blocking)
                attention_mask = batch["attention_mask"].to(
                    self.device, non_blocking=self.non_blocking
                )
                start_positions = batch["start_positions"].to(
                    self.device, non_blocking=self.non_blocking
                )
                end_positions = batch["end_positions"].to(
                    self.device, non_blocking=self.non_blocking
                )

                with torch.amp.autocast('cuda', enabled=self.use_amp, dtype=self.amp_dtype):
                    # Forward pass
                    start_logits, end_logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )

                # Compute loss
                loss = self._compute_loss(start_logits, end_logits, start_positions, end_positions)

                total_loss += loss.item()
                num_batches += 1

                # Collect predictions for metrics
                start_preds = torch.argmax(start_logits, dim=1)
                end_preds = torch.argmax(end_logits, dim=1)

                all_start_preds.extend(start_preds.cpu().tolist())
                all_end_preds.extend(end_preds.cpu().tolist())
                all_start_labels.extend(start_positions.cpu().tolist())
                all_end_labels.extend(end_positions.cpu().tolist())

                # Update progress bar
                progress_bar.set_postfix({"loss": total_loss / num_batches})

        # Compute metrics
        metrics = self._compute_metrics(
            all_start_preds, all_end_preds, all_start_labels, all_end_labels
        )
        metrics["loss"] = total_loss / num_batches

        return metrics

    def _compute_loss(
        self,
        start_logits: torch.Tensor,
        end_logits: torch.Tensor,
        start_positions: torch.Tensor,
        end_positions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute span extraction loss.

        Args:
            start_logits: Start position predictions (batch_size, seq_len)
            end_logits: End position predictions (batch_size, seq_len)
            start_positions: True start positions (batch_size,)
            end_positions: True end positions (batch_size,)

        Returns:
            Combined loss for start and end positions
        """
        loss_fct = nn.CrossEntropyLoss(ignore_index=-1)

        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)

        total_loss: torch.Tensor = (start_loss + end_loss) / 2
        return total_loss

    def _compute_metrics(
        self,
        start_preds: list[int],
        end_preds: list[int],
        start_labels: list[int],
        end_labels: list[int],
    ) -> dict[str, float]:
        """
        Compute evaluation metrics.

        Args:
            start_preds: Predicted start positions
            end_preds: Predicted end positions
            start_labels: True start positions
            end_labels: True end positions

        Returns:
            Dictionary of metrics (exact_match, f1, etc.)
        """
        exact_matches = 0
        total = len(start_preds)

        for s_pred, e_pred, s_label, e_label in zip(
            start_preds, end_preds, start_labels, end_labels, strict=False
        ):
            if s_pred == s_label and e_pred == e_label:
                exact_matches += 1

        exact_match = exact_matches / total if total > 0 else 0.0

        # Compute F1 (span overlap)
        f1_scores = []
        for s_pred, e_pred, s_label, e_label in zip(
            start_preds, end_preds, start_labels, end_labels, strict=False
        ):
            pred_span = set(range(s_pred, e_pred + 1))
            label_span = set(range(s_label, e_label + 1))

            if len(pred_span) == 0 and len(label_span) == 0:
                f1_scores.append(1.0)
            elif len(pred_span) == 0 or len(label_span) == 0:
                f1_scores.append(0.0)
            else:
                intersection = len(pred_span & label_span)
                precision = intersection / len(pred_span)
                recall = intersection / len(label_span)

                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0

                f1_scores.append(f1)

        avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

        return {
            "exact_match": exact_match,
            "f1": avg_f1,
        }

    def _log_metrics(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
    ) -> None:
        """Log metrics to MLflow."""
        try:
            # Log training metrics
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value, step=epoch)

            # Log validation metrics
            for metric_name, value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", value, step=epoch)

            # Log learning rate
            if self.scheduler:
                mlflow.log_metric("learning_rate", self.scheduler.get_last_lr()[0], step=epoch)
            else:
                mlflow.log_metric("learning_rate", self.optimizer.param_groups[0]["lr"], step=epoch)

        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")

    def _resume_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Resume training from checkpoint."""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")

        checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.current_epoch = checkpoint["epoch"] + 1
        self.global_step = checkpoint["step"]

        if "best_metric" in checkpoint.get("metadata", {}):
            self.best_metric = checkpoint["metadata"]["best_metric"]

        logger.info(f"Resumed from epoch {self.current_epoch}, step {self.global_step}")
