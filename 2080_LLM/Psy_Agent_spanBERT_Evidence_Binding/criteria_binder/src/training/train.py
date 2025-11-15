# File: src/training/train.py
"""Training loop implementation for criteria binding model."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
import math

from ..models.binder import SpanBertEvidenceBinder
from ..data.dataset import CriteriaBindingDataset, create_label_mappings
from ..training.collator import CriteriaBindingCollator
from ..training.callbacks import EarlyStopping, ModelCheckpoint
from ..utils.logging import Timer, ProgressTracker
from ..utils.seed import set_seed, get_device
from ..utils.io import save_yaml, save_json

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for criteria binding model."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: SpanBertEvidenceBinder,
        tokenizer: AutoTokenizer,
        train_dataset: CriteriaBindingDataset,
        eval_dataset: Optional[CriteriaBindingDataset] = None,
        collator: Optional[CriteriaBindingCollator] = None,
    ) -> None:
        """Initialize trainer.

        Args:
            config: Training configuration
            model: Model to train
            tokenizer: Tokenizer
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            collator: Data collator
        """
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        # Set up device and move model
        self.device = get_device()
        self.model.to(self.device)

        # Create collator if not provided
        if collator is None:
            collator = CriteriaBindingCollator(
                tokenizer=tokenizer,
                max_length=config["model"]["max_length"],
                doc_stride=config["model"]["doc_stride"],
            )
        self.collator = collator

        # Create data loaders with optimized settings
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["train"]["batch_size"],
            shuffle=config["train"].get("shuffle", True),
            collate_fn=collator,
            num_workers=config["data"]["num_workers"],
            pin_memory=config["train"].get("dataloader_pin_memory", True),
            prefetch_factor=config["data"].get("prefetch_factor", 2),
            persistent_workers=config["data"].get("persistent_workers", True),
        )

        if eval_dataset is not None:
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=config["train"]["batch_size"],
                shuffle=False,
                collate_fn=collator,
                num_workers=config["data"]["num_workers"],
                pin_memory=config["train"].get("dataloader_pin_memory", True),
                prefetch_factor=config["data"].get("prefetch_factor", 2),
                persistent_workers=config["data"].get("persistent_workers", True),
            )
        else:
            self.eval_loader = None

        # Set up optimization
        self._setup_optimization()

        # Set up callbacks
        self._setup_callbacks()

        # Training state
        self.global_step = 0
        self.current_epoch = 0

    def _setup_optimization(self) -> None:
        """Set up optimizer and scheduler."""
        # Calculate total steps
        num_training_steps = (
            len(self.train_loader) * self.config["train"]["epochs"] //
            self.config["train"]["grad_accum"]
        )

        # Create optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config["train"]["weight_decay"],
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config["train"]["lr"],
        )

        # Create scheduler
        num_warmup_steps = int(num_training_steps * self.config["train"]["warmup_ratio"])
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Mixed precision scaler with optimized settings
        if self.config["train"]["fp16"]:
            self.scaler = torch.amp.GradScaler('cuda')
        elif self.config["train"].get("bf16", False):
            self.scaler = torch.amp.GradScaler('cuda', enabled=False)  # BF16 doesn't need scaling
        else:
            self.scaler = None

        logger.info(f"Total training steps: {num_training_steps}")
        logger.info(f"Warmup steps: {num_warmup_steps}")

    def _setup_callbacks(self) -> None:
        """Set up training callbacks."""
        output_dir = Path(self.config["logging"]["output_dir"])

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config["train"]["early_stop_patience"],
            monitor="combined_score",
            mode="max",
        )

        # Model checkpointing
        self.checkpoint_callback = ModelCheckpoint(
            checkpoint_dir=output_dir,
            monitor="combined_score",
            mode="max",
            save_top_k=self.config["train"]["save_k_best"],
            save_last=True,
        )

    def train(self) -> Dict[str, Any]:
        """Run full training loop.

        Returns:
            Training history and final metrics
        """
        logger.info("Starting training...")
        logger.info(f"Device: {self.device}")
        logger.info(f"Train examples: {len(self.train_dataset)}")
        if self.eval_dataset:
            logger.info(f"Eval examples: {len(self.eval_dataset)}")

        # Save initial config and label mappings
        self._save_training_setup()

        training_history = []
        timer = Timer()
        timer.start()

        for epoch in range(self.config["train"]["epochs"]):
            self.current_epoch = epoch

            # Training epoch
            train_metrics = self._train_epoch()

            # Evaluation
            eval_metrics = {}
            if self.eval_loader and (
                epoch % (self.config["train"]["eval_every_steps"] // len(self.train_loader)) == 0
                or epoch == self.config["train"]["epochs"] - 1
            ):
                eval_metrics = self._eval_epoch()

            # Combine metrics
            epoch_metrics = {
                "epoch": epoch,
                "global_step": self.global_step,
                **train_metrics,
                **eval_metrics,
            }

            training_history.append(epoch_metrics)

            # Log progress
            self._log_epoch_metrics(epoch, epoch_metrics, timer.elapsed())

            # Callbacks
            if eval_metrics:
                # Model checkpointing
                label_mappings = create_label_mappings(self.train_dataset)
                self.checkpoint_callback(
                    epoch,
                    eval_metrics,
                    self.model,
                    self.tokenizer,
                    self.config,
                    label_mappings,
                )

                # Early stopping
                if self.early_stopping(epoch, eval_metrics):
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        total_time = timer.stop()
        logger.info(f"Training completed in {total_time:.2f} seconds")

        # Save training history
        history_path = Path(self.config["logging"]["output_dir"]) / "training_history.json"
        save_json(training_history, history_path)

        # Get best checkpoint path
        best_checkpoint = self.checkpoint_callback.get_best_checkpoint_path()

        return {
            "training_history": training_history,
            "total_time": total_time,
            "best_checkpoint": str(best_checkpoint) if best_checkpoint else None,
            "final_metrics": training_history[-1] if training_history else {},
        }

    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Training metrics for the epoch
        """
        self.model.train()

        total_loss = 0.0
        total_span_loss = 0.0
        total_cls_loss = 0.0
        num_batches = 0

        progress = ProgressTracker(len(self.train_loader), f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}

            # Forward pass with mixed precision
            dtype = torch.float16 if self.config["train"]["fp16"] else (
                torch.bfloat16 if self.config["train"].get("bf16", False) else torch.float32
            )
            with torch.amp.autocast('cuda', enabled=self.config["train"]["fp16"] or self.config["train"].get("bf16", False), dtype=dtype):
                # Only pass model inputs, not metadata
                model_inputs = {k: v for k, v in batch.items()
                              if k in ['input_ids', 'attention_mask', 'token_type_ids',
                                     'text_mask', 'start_positions', 'end_positions', 'labels']}
                outputs = self.model(**model_inputs)
                loss = outputs.get("loss", torch.tensor(0.0))

            # Backward pass
            if self.scaler:
                self.scaler.scale(loss / self.config["train"]["grad_accum"]).backward()
            else:
                (loss / self.config["train"]["grad_accum"]).backward()

            # Accumulate gradients
            if (batch_idx + 1) % self.config["train"]["grad_accum"] == 0:
                # Gradient clipping with configurable max norm
                max_grad_norm = self.config["train"].get("max_grad_norm", 1.0)
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Accumulate losses
            total_loss += loss.item()
            if "span_loss" in outputs:
                total_span_loss += outputs["span_loss"].item()
            if "cls_loss" in outputs:
                total_cls_loss += outputs["cls_loss"].item()
            num_batches += 1

            progress.update()

            # Periodic evaluation during training
            if (self.global_step % self.config["train"]["eval_every_steps"] == 0 and
                self.eval_loader):
                eval_metrics = self._eval_epoch()
                self._log_step_metrics(self.global_step, eval_metrics)

                # Check for early stopping
                if self.early_stopping(self.current_epoch, eval_metrics):
                    logger.info(f"Early stopping triggered at step {self.global_step}")
                    break

                self.model.train()  # Return to training mode

        # Calculate average losses
        metrics = {
            "train_loss": total_loss / num_batches if num_batches > 0 else 0.0,
            "train_span_loss": total_span_loss / num_batches if num_batches > 0 else 0.0,
            "train_cls_loss": total_cls_loss / num_batches if num_batches > 0 else 0.0,
            "learning_rate": self.scheduler.get_last_lr()[0],
        }

        return metrics

    def _eval_epoch(self) -> Dict[str, float]:
        """Evaluate for one epoch.

        Returns:
            Evaluation metrics
        """
        from .eval import evaluate_model

        return evaluate_model(
            self.model,
            self.eval_loader,
            self.device,
            self.config,
        )

    def _save_training_setup(self) -> None:
        """Save initial training configuration and label mappings."""
        output_dir = Path(self.config["logging"]["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        save_yaml(self.config, output_dir / "config.yaml")

        # Save label mappings
        label_mappings = create_label_mappings(self.train_dataset)
        save_json(label_mappings, output_dir / "label_mappings.json")

        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir / "tokenizer")

    def _log_epoch_metrics(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        elapsed_time: float,
    ) -> None:
        """Log metrics for an epoch."""
        logger.info(f"Epoch {epoch} completed in {elapsed_time:.2f}s")

        for key, value in metrics.items():
            if isinstance(value, (int, float)) and key != "epoch" and key != "global_step":
                logger.info(f"  {key}: {value:.4f}")

    def _log_step_metrics(self, step: int, metrics: Dict[str, float]) -> None:
        """Log metrics for a step."""
        logger.info(f"Step {step} evaluation:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {key}: {value:.4f}")


def create_trainer_from_config(
    config: Dict[str, Any],
    train_dataset: CriteriaBindingDataset,
    eval_dataset: Optional[CriteriaBindingDataset] = None,
) -> Trainer:
    """Create trainer from configuration.

    Args:
        config: Configuration dictionary
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset

    Returns:
        Configured trainer
    """
    # Set random seed
    set_seed(config["logging"]["seed"])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create model
    label_mappings = create_label_mappings(train_dataset)
    model = SpanBertEvidenceBinder(
        model_name=config["model"]["name"],
        num_labels=label_mappings["num_labels"],
        use_label_head=config["model"]["use_label_head"],
        dropout=config["model"]["dropout"],
        lambda_span=config["model"]["lambda_span"],
        gradient_checkpointing=config["train"].get("gradient_checkpointing", False),
    )

    # Create collator
    collator = CriteriaBindingCollator(
        tokenizer=tokenizer,
        max_length=config["model"]["max_length"],
        doc_stride=config["model"]["doc_stride"],
    )

    return Trainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        collator=collator,
    )