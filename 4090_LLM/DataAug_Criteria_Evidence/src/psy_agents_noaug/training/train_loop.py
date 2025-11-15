"""Comprehensive training loop with MLflow, AMP, and early stopping."""

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

if TYPE_CHECKING:  # pragma: no cover
    from psy_agents_noaug.augmentation import AugmenterPipeline


LOGGER = logging.getLogger(__name__)


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
        augmenter_pipeline: "AugmenterPipeline | None" = None,
        augmenter_config: dict[str, Any] | None = None,
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
        self.augmenter_pipeline = augmenter_pipeline
        self.augmenter_config = augmenter_config
        self._aug_examples_logged = False

        use_scaler = self.use_amp and self.amp_dtype == torch.float16
        # AMP gradient scaler (enabled only for float16 on CUDA)
        self.scaler = GradScaler("cuda", enabled=use_scaler)

        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.best_metric_value = (
            float("-inf") if early_stopping_mode == "max" else float("inf")
        )
        self.epochs_without_improvement = 0
        self.training_history: list[dict[str, float]] = []
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

        # Lightweight throughput telemetry using an EMA of timings
        ema_window = 10
        ema_alpha = 2.0 / (ema_window + 1)
        data_time_ema: float | None = None
        step_time_ema: float | None = None
        batch_start_time = time.time()

        pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch + 1}/{self.num_epochs}", leave=False
        )

        for batch_idx, batch in enumerate(pbar):
            data_ready_time = time.time()
            data_time = data_ready_time - batch_start_time

            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

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

            compute_done_time = time.time()
            step_time = compute_done_time - data_ready_time

            if data_time_ema is None:
                data_time_ema = data_time
            else:
                data_time_ema += ema_alpha * (data_time - data_time_ema)

            if step_time_ema is None:
                step_time_ema = step_time
            else:
                step_time_ema += ema_alpha * (step_time - step_time_ema)

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
                    perf_ratio = 0.0
                    if step_time_ema is not None and step_time_ema > 0:
                        perf_ratio = float(data_time_ema or 0.0) / step_time_ema
                        if perf_ratio > 0.40:
                            LOGGER.warning(
                                "Data/step ratio %.3f exceeds 0.40 threshold; consider tuning augmentation or dataloader.",
                                perf_ratio,
                            )
                            mlflow.log_metric(
                                "perf.data_to_step_ratio_alert",
                                1.0,
                                step=self.global_step,
                            )
                    mlflow.log_metrics(
                        {
                            "train_loss_step": loss.item()
                            * self.gradient_accumulation_steps,
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                            "perf.data_time_ms_ema": ((data_time_ema or 0.0) * 1000.0),
                            "perf.step_time_ms_ema": ((step_time_ema or 0.0) * 1000.0),
                            "perf.data_to_step_ratio": perf_ratio,
                        },
                        step=self.global_step,
                    )

            # Track metrics for the whole epoch
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

            batch_start_time = time.time()

        avg_loss = total_loss / max(len(self.train_loader), 1)
        accuracy = correct / total

        perf_ratio_epoch = 0.0
        if step_time_ema is not None and step_time_ema > 0:
            perf_ratio_epoch = float(data_time_ema or 0.0) / step_time_ema

        return {
            "train_loss": avg_loss,
            "train_accuracy": accuracy,
            "perf.data_time_ms_ema": (data_time_ema or 0.0) * 1000.0,
            "perf.step_time_ms_ema": (step_time_ema or 0.0) * 1000.0,
            "perf.data_to_step_ratio": perf_ratio_epoch,
            "perf.data_to_step_ratio_alert": float(perf_ratio_epoch > 0.40),
        }

    @staticmethod
    def _sanitize_method_name(name: str) -> str:
        """Normalise augmenter names for MLflow metric keys.

        MLflow metric keys must be simple strings without path separators or
        special characters; we replace a few common ones here.
        """
        return (
            name.replace("/", "_").replace(".", "_").replace(" ", "_").replace("-", "_")
        )

    def _log_augmentation_metrics(self, epoch: int) -> None:
        """Log augmentation usage statistics and examples to MLflow.

        Called once per epoch (after validation). Example pairs are written as
        a JSONL artifact only once per run to keep IO bounded.
        """
        if not self.augmenter_pipeline:
            return

        stats = self.augmenter_pipeline.stats()
        metrics = {
            "aug.applied_count": float(stats.get("applied", 0)),
            "aug.skipped_count": float(stats.get("skipped", 0)),
            "aug.total_count": float(stats.get("total", 0)),
        }
        mlflow.log_metrics(metrics, step=epoch)

        method_counts = stats.get("method_counts", {})
        if method_counts:
            mlflow.log_metrics(
                {
                    f"aug.method_count.{self._sanitize_method_name(method)}": float(
                        count
                    )
                    for method, count in method_counts.items()
                },
                step=epoch,
            )

        if not self._aug_examples_logged:
            examples = self.augmenter_pipeline.drain_examples()
            if examples:
                artifact_dir = self.save_dir or Path(".")
                artifact_dir.mkdir(parents=True, exist_ok=True)
                artifact_path = artifact_dir / "aug_examples.jsonl"
                with artifact_path.open("w", encoding="utf-8") as fp:
                    for example in examples:
                        fp.write(json.dumps(example, ensure_ascii=False) + "\n")
                mlflow.log_artifact(str(artifact_path), artifact_path="augmentation")
                self._aug_examples_logged = True

    def _log_augmentation_params(self) -> None:
        """Log augmentation configuration once per training run.

        Accepts both plain dicts and dataclass objects to remain flexible with
        different call sites (Hydra configs vs. CLI dictionaries).
        """
        if mlflow.active_run() is None:
            return

        cfg_obj = self.augmenter_config
        has_pipeline = self.augmenter_pipeline is not None

        if cfg_obj is None:
            mlflow.log_params({"aug.enabled": False})
            return

        def _get(field: str, default: Any = None) -> Any:
            if isinstance(cfg_obj, dict):
                return cfg_obj.get(field, default)
            return getattr(cfg_obj, field, default)

        if has_pipeline and self.augmenter_pipeline is not None:
            methods = sorted(self.augmenter_pipeline.methods)
        else:
            raw_methods = _get("methods", [])
            if isinstance(raw_methods, str):
                raw_methods = [raw_methods]
            methods = sorted(str(m) for m in raw_methods)

        enabled_flag = bool(self.augmenter_pipeline) or bool(_get("enabled", False))
        max_replace = float(_get("max_replace", _get("max_replace_ratio", 0.0)))
        method_weights = _get("method_weights", {}) or {}
        params: dict[str, Any] = {
            "aug.enabled": bool(enabled_flag),
            "aug.methods": ";".join(methods) if methods else "",
            "aug.p_apply": float(_get("p_apply", 0.0)),
            "aug.ops_per_sample": int(_get("ops_per_sample", 1)),
            "aug.max_replace": max_replace,
            "aug.tfidf_model_path": _get("tfidf_model_path"),
            "aug.reserved_map_path": _get("reserved_map_path"),
        }

        if method_weights:
            params["aug.method_weights"] = json.dumps(
                {str(k): float(v) for k, v in method_weights.items()},
                ensure_ascii=False,
                sort_keys=True,
            )

        method_kwargs = _get("method_kwargs", {})
        if method_kwargs:
            params["aug.method_kwargs"] = json.dumps(
                method_kwargs, ensure_ascii=False, sort_keys=True
            )

        mlflow.log_params(params)

    def validate(self) -> dict[str, float]:
        """
        Validate on validation set with comprehensive metrics.

        Returns:
            Dictionary with validation metrics including F1 scores
        """
        self.model.eval()

        total_loss = 0.0
        all_preds: list[int] = []
        all_labels: list[int] = []
        all_probabilities: list[float] = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation", leave=False):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

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
        all_preds = np.array(all_preds)  # type: ignore[assignment]
        all_labels = np.array(all_labels)  # type: ignore[assignment]
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

        # AUROC calculation is guarded because it can fail in corner cases
        auroc_macro: float | None = None
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
            "val_auroc_macro": auroc_macro if auroc_macro is not None else 0.0,
        }

    def save_checkpoint(
        self, epoch: int, metrics: dict[str, float], is_best: bool = False
    ) -> None:
        """Save model checkpoint with optimizer/scheduler/scaler state.

        Two files are maintained in ``save_dir``:
          - ``latest_checkpoint.pt``: always updated
          - ``best_checkpoint.pt``: updated only when metric improves
        """
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
        """Check whether the early‑stopping metric improved by ``min_delta``."""
        if self.early_stopping_mode == "max":
            return current_value > (self.best_metric_value + self.min_delta)
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

        self._log_augmentation_params()

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

            self._log_augmentation_metrics(epoch)

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
