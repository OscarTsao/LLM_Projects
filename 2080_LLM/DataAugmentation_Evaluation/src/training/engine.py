"""Reusable training engine for BERT pair classification."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from src.utils import mlflow_utils, wandb_utils
from .data_module import DataModule, DataModuleConfig
from .dataset_builder import build_splits
from .modeling import BertPairClassifier, ModelConfig


METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested dictionary for MLflow logging.

    Converts nested dictionaries like {'model': {'lr': 1e-3}} to {'model.lr': 1e-3}.
    This is needed because MLflow parameters must be flat key-value pairs.

    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys (used in recursion)
        sep: Separator between nested keys

    Returns:
        Flattened dictionary with dot-separated keys
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        # Build the full key path (e.g., "model.learning_rate")
        new_key = f"{parent_key}{sep}{k}" if parent_key else k

        if isinstance(v, dict):
            # Recursively flatten nested dicts
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)) and v and not isinstance(v[0], (dict, list, tuple)):
            # Convert simple lists/tuples to strings (e.g., [256, 128] -> "[256, 128]")
            items.append((new_key, str(v)))
        elif not isinstance(v, (list, tuple, dict)):
            # Keep primitive values as-is
            items.append((new_key, v))
        # Note: Nested lists/dicts are skipped as they can't be logged to MLflow
    return dict(items)


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility across PyTorch, NumPy, and CUDA.

    Args:
        seed: Random seed value

    Note:
        This doesn't guarantee 100% reproducibility due to non-deterministic CUDA operations,
        but it makes results more consistent across runs.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: torch.nn.Module, cfg: DictConfig) -> Optimizer:
    """Build optimizer from configuration.

    Args:
        model: PyTorch model to optimize
        cfg: Configuration containing optimizer settings

    Returns:
        Configured optimizer instance

    Raises:
        ValueError: If optimizer name is not supported

    Supported optimizers:
        - adamw_torch: PyTorch's AdamW (recommended)
        - adamw_hf: Legacy HuggingFace AdamW (now uses PyTorch implementation)
        - sgd: Stochastic Gradient Descent with momentum
    """
    optimizer_name = cfg.model.optimizer.lower()
    weight_decay = cfg.model.weight_decay
    learning_rate = cfg.model.learning_rate
    adam_eps = cfg.model.get("adam_eps", 1e-8)

    # Group parameters for weight decay
    # Note: Could be extended to exclude bias and LayerNorm from weight decay
    optimizer_grouped_parameters = [
        {
            "params": [p for _, p in model.named_parameters() if p.requires_grad],
            "weight_decay": weight_decay,
        }
    ]

    if optimizer_name == "adamw_torch":
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    if optimizer_name == "adamw_hf":
        # AdamW removed from transformers in v5+, use torch.optim.AdamW instead
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    if optimizer_name == "sgd":
        return torch.optim.SGD(optimizer_grouped_parameters, lr=learning_rate, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer: Optimizer, cfg: DictConfig, num_warmup_steps: int, num_training_steps: int) -> LambdaLR:
    """Build learning rate scheduler from configuration.

    Args:
        optimizer: Optimizer to schedule
        cfg: Configuration containing scheduler settings
        num_warmup_steps: Number of warmup steps (linear increase from 0 to lr)
        num_training_steps: Total number of training steps

    Returns:
        Learning rate scheduler

    Supported schedulers:
        - linear: Linear decay from lr to 0 after warmup
        - cosine: Cosine annealing from lr to 0 after warmup
        - polynomial: Polynomial decay (power=1.0 is linear) after warmup
        - none: Constant learning rate (no scheduling)
    """
    scheduler_name = cfg.model.scheduler.lower()
    if scheduler_name == "linear":
        from transformers.optimization import get_linear_schedule_with_warmup  # type: ignore

        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    if scheduler_name == "cosine":
        from transformers.optimization import get_cosine_schedule_with_warmup  # type: ignore

        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    if scheduler_name == "polynomial":
        from transformers.optimization import get_polynomial_decay_schedule_with_warmup  # type: ignore

        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=1.0,
        )
    # Default: constant learning rate (no scheduling)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def evaluate(model: torch.nn.Module, dataloader, device: torch.device) -> dict[str, float]:
    """Evaluate model on a dataset and compute classification metrics.

    Args:
        model: Model to evaluate
        dataloader: DataLoader providing evaluation batches
        device: Device to run evaluation on (cpu/cuda)

    Returns:
        Dictionary of metrics: accuracy, precision, recall, f1, roc_auc

    Note:
        - Uses model.eval() mode (disables dropout, batchnorm training)
        - Computes predictions without gradient tracking for efficiency
        - Returns NaN for metrics if no samples are present
        - ROC-AUC may be NaN if only one class is present in the data
    """
    model.eval()
    all_labels: list[int] = []
    all_probs: list[float] = []  # Probabilities for positive class (for ROC-AUC)
    all_preds: list[int] = []     # Predicted class labels

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")

            # Forward pass
            outputs = model(**batch)
            logits = outputs["logits"].detach()

            # Get probabilities and predictions
            probs = torch.softmax(logits, dim=-1)[:, 1]  # Probability of positive class
            preds = torch.argmax(logits, dim=-1)         # Predicted class (0 or 1)

            # Collect all predictions and labels
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())

    # Handle empty evaluation set
    if not all_labels:
        return {metric: math.nan for metric in METRIC_KEYS}

    # Compute classification metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }

    # ROC-AUC requires probabilities and may fail if only one class is present
    try:
        metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # Can fail if all labels are the same class
        metrics["roc_auc"] = float("nan")

    return metrics


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_config(path: Path, cfg: DictConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(path))


def get_training_steps(num_samples: int, batch_size: int, epochs: int, grad_accum: int) -> int:
    steps_per_epoch = math.ceil(num_samples / max(batch_size, 1) / max(grad_accum, 1))
    return steps_per_epoch * epochs


def train_model(
    cfg: DictConfig,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Main training function for BERT pair classification.

    This function handles the complete training pipeline including:
    - Data loading and preprocessing
    - Model initialization and GPU optimization
    - Training loop with gradient accumulation
    - Validation and early stopping
    - Checkpoint management
    - Experiment tracking (MLflow, W&B)
    - Final evaluation on test set

    Args:
        cfg: Hydra configuration containing all training parameters
        output_dir: Directory to save checkpoints and results. If None, uses cfg.output_dir

    Returns:
        Dictionary containing:
            - best_metric: Best validation metric achieved
            - best_metrics: All metrics from best epoch
            - test_metrics: Final test set metrics
            - best_model_path: Path to saved best model
            - output_dir: Output directory path

    Training Features:
        - Automatic mixed precision (FP16/BF16) for faster training
        - Gradient accumulation for large effective batch sizes
        - Early stopping to prevent overfitting
        - Learning rate warmup and scheduling
        - Checkpoint resumption for interrupted training
        - GPU optimizations (TF32, cuDNN benchmark)
        - Optional model compilation (torch.compile for 20-50% speedup)
    """
    set_global_seed(cfg.seed)

    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    flat_params = _flatten_dict(resolved_cfg) if isinstance(resolved_cfg, dict) else {}

    dataset_name = "dataset"
    if hasattr(cfg, "dataset") and cfg.dataset is not None:
        try:
            dataset_name = cfg.dataset.get("name", dataset_name)  # type: ignore[attr-defined]
        except AttributeError:
            dataset_name = getattr(cfg.dataset, "name", dataset_name)

    logging_cfg = getattr(cfg, "logging", None)
    wandb_section = getattr(logging_cfg, "wandb", None) if logging_cfg is not None else None
    wandb_cfg = OmegaConf.to_container(wandb_section, resolve=True) if wandb_section is not None else {}
    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    wandb_project = wandb_cfg.get("project", "redsm5-classification")
    wandb_entity = wandb_cfg.get("entity")
    wandb_group = wandb_cfg.get("group") or dataset_name
    wandb_tags = list(wandb_cfg.get("tags", []))
    run_name_override = wandb_cfg.get("run_name")
    if run_name_override:
        wandb_run_name = str(run_name_override)
    else:
        wandb_run_name = f"train-{dataset_name}"
    wandb_mode = wandb_cfg.get("mode")
    job_type = "training"

    with wandb_utils.start_run(
        enabled=wandb_enabled,
        project=wandb_project,
        entity=wandb_entity,
        name=wandb_run_name,
        config=resolved_cfg if isinstance(resolved_cfg, dict) else None,
        tags=wandb_tags,
        group=wandb_group,
        job_type=job_type,
        mode=wandb_mode,
    ):
        # Setup MLflow tracking
        mlflow_section = getattr(cfg, "mlflow", None)
        experiment_name = "redsm5-classification"
        if mlflow_section is not None:
            experiments_cfg = mlflow_section.get("experiments")
            if experiments_cfg is not None:
                experiment_name = experiments_cfg.get("training", experiment_name)
        mlflow_utils.setup_mlflow(cfg, experiment_name=experiment_name)
        mlflow_run = mlflow_utils.start_run(run_name=f"train_{dataset_name}")

        with mlflow_run:
            # Log configuration parameters
            if flat_params:
                mlflow_utils.log_params(flat_params)
                mlflow_utils.set_tag("model_type", "bert_pair_classifier")
                mlflow_utils.set_tag("framework", "pytorch")

            splits = build_splits(cfg.dataset)
            data_module = DataModule(
                split=splits,
                config=DataModuleConfig(
                    tokenizer_name=cfg.model.pretrained_model_name,
                    max_seq_length=cfg.model.max_seq_length,
                    batch_size=cfg.model.batch_size,
                    eval_batch_size=cfg.model.get("eval_batch_size"),
                    num_workers=cfg.dataloader.num_workers,
                    pin_memory=cfg.dataloader.pin_memory,
                    persistent_workers=cfg.dataloader.persistent_workers,
                    prefetch_factor=cfg.dataloader.prefetch_factor,
                ),
            )

            model = BertPairClassifier(
                ModelConfig(
                    pretrained_model_name=cfg.model.pretrained_model_name,
                    classifier_hidden_sizes=cfg.model.classifier_hidden_sizes,
                    dropout=cfg.model.classifier_dropout,
                    num_labels=2,
                )
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                # Enable TF32 for faster training on Ampere+ GPUs (RTX 30xx/40xx/50xx)
                torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
                torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
                torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
            model.to(device)

            # Optional: Compile model for faster training (PyTorch 2.0+)
            if cfg.model.get("compile_model", False) and hasattr(torch, "compile"):
                model = torch.compile(model)  # type: ignore[assignment]

            optimizer = build_optimizer(model, cfg)
            grad_accum = cfg.model.gradient_accumulation_steps
            total_steps = get_training_steps(len(splits.train), cfg.model.batch_size, cfg.model.num_epochs, grad_accum)
            warmup_steps = int(total_steps * cfg.model.warmup_ratio)
            scheduler = build_scheduler(optimizer, cfg, warmup_steps, total_steps)
            # DeBERTa models have overflow issues with mixed precision due to torch.finfo().min in attention
            model_name = cfg.model.pretrained_model_name.lower()
            disable_amp = "deberta" in model_name

            # Use bfloat16 if requested (better for modern GPUs like RTX 5090)
            use_bfloat16 = cfg.model.get("use_bfloat16", False) and not disable_amp
            dtype = torch.bfloat16 if use_bfloat16 else torch.float16
            # GradScaler not needed for bfloat16 or when AMP is disabled
            scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available() and not use_bfloat16 and not disable_amp)

            if output_dir is None:
                output_dir = Path(cfg.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            checkpoints_dir = output_dir / "checkpoints"
            best_dir = output_dir / "best"
            checkpoints_dir.mkdir(exist_ok=True)
            best_metric = -float("inf")
            best_metrics: dict[str, float] | None = None
            best_model_path = best_dir / "model.pt"

            last_ckpt_path = checkpoints_dir / "last.pt"
            start_epoch = 0
            global_step = 0

            # Resume training from checkpoint if requested
            if cfg.resume and last_ckpt_path.exists():
                state = torch.load(last_ckpt_path, map_location=device)

                # Handle loading checkpoints between compiled and non-compiled models
                # torch.compile wraps models with OptimizedModule and adds "_orig_mod." prefix
                # to all state dict keys. We need to adjust keys when loading across compilation modes.
                model_state = state["model_state"]
                checkpoint_is_compiled = any(k.startswith("_orig_mod.") for k in model_state.keys())
                model_is_compiled = hasattr(model, "_orig_mod")

                if checkpoint_is_compiled and not model_is_compiled:
                    # Loading compiled checkpoint into non-compiled model: strip "_orig_mod." prefix
                    model_state = {k.replace("_orig_mod.", ""): v for k, v in model_state.items()}
                elif not checkpoint_is_compiled and model_is_compiled:
                    # Loading non-compiled checkpoint into compiled model: add "_orig_mod." prefix
                    model_state = {f"_orig_mod.{k}": v for k, v in model_state.items()}

                # Load all checkpoint states
                model.load_state_dict(model_state)
                optimizer.load_state_dict(state["optimizer_state"])
                scheduler.load_state_dict(state["scheduler_state"])
                scaler.load_state_dict(state["scaler_state"])
                start_epoch = state.get("epoch", 0)
                global_step = state.get("global_step", 0)

            train_loader = data_module.train_dataloader()
            val_loader = data_module.val_dataloader()

            # Early stopping configuration
            # Stops training if validation metric doesn't improve for N consecutive epochs
            early_stopping_patience = cfg.get("early_stopping_patience", 20)
            patience_counter = 0

            # ============ TRAINING LOOP ============
            for epoch in range(start_epoch, cfg.model.num_epochs):
                model.train()
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
                epoch_loss = 0.0
                num_batches = 0
                progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.model.num_epochs}", leave=False)

                # --- Training Step ---
                for step, batch in enumerate(progress):
                    # Move batch to device
                    batch = {k: v.to(device) for k, v in batch.items()}
                    labels = batch.pop("labels")

                    # Forward pass with automatic mixed precision
                    # Disable autocast for DeBERTa to avoid overflow in attention masking
                    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available() and not disable_amp, dtype=dtype):
                        outputs = model(**batch, labels=labels)
                        # Scale loss by gradient accumulation steps for correct averaging
                        loss = outputs["loss"] / grad_accum

                    # Backward pass with gradient scaling (for mixed precision)
                    scaler.scale(loss).backward()
                    epoch_loss += loss.item() * grad_accum
                    num_batches += 1

                    # Gradient accumulation: only update weights every N steps
                    if (step + 1) % grad_accum == 0:
                        scaler.unscale_(optimizer)  # Unscale before gradient clipping
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.model.max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()  # Update learning rate
                        global_step += 1

                # --- Validation Step ---
                avg_train_loss = epoch_loss / max(num_batches, 1)
                mlflow_utils.log_metrics({"train_loss": avg_train_loss}, step=epoch)
                wandb_utils.log_metrics({"train/loss": avg_train_loss}, step=epoch)

                # Evaluate on validation set
                metrics = evaluate(model, val_loader, device)
                monitor_metric = metrics.get(cfg.metric_for_best_model, float("nan"))

                # Log validation metrics
                val_metrics_prefixed = {f"val_{k}": v for k, v in metrics.items()}
                mlflow_utils.log_metrics(val_metrics_prefixed, step=epoch)
                wandb_utils.log_metrics({f"val/{k}": v for k, v in metrics.items()}, step=epoch)

                # --- Model Checkpointing ---
                if not math.isnan(monitor_metric) and monitor_metric > best_metric:
                    # New best model found!
                    best_metric = monitor_metric
                    best_metrics = metrics
                    best_dir.mkdir(exist_ok=True)
                    torch.save(model.state_dict(), best_model_path)
                    save_config(best_dir / "config.yaml", cfg)
                    save_json(best_dir / "val_metrics.json", metrics)
                    patience_counter = 0  # Reset patience counter on improvement
                else:
                    # No improvement
                    patience_counter += 1

                # --- Early Stopping Check ---
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs (patience: {early_stopping_patience})")
                    mlflow_utils.log_metrics({"early_stopped": 1, "stopped_epoch": epoch + 1})
                    wandb_utils.log_metrics({"early_stopped": 1, "stopped_epoch": epoch + 1})

                    # Save final checkpoint before stopping
                    torch.save(
                        {
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "scaler_state": scaler.state_dict(),
                            "epoch": epoch + 1,
                            "global_step": global_step,
                        },
                        last_ckpt_path,
                    )
                    break

                # Save checkpoint after each epoch (for resumption)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": scheduler.state_dict(),
                        "scaler_state": scaler.state_dict(),
                        "epoch": epoch + 1,
                        "global_step": global_step,
                    },
                    last_ckpt_path,
                )

            # ============ FINAL EVALUATION ON TEST SET ============
            test_loader = data_module.test_dataloader()

            # Load best model for final evaluation (not the last epoch model)
            if best_model_path.exists():
                best_model_state = torch.load(best_model_path, map_location=device)

                # Handle loading checkpoints between compiled and non-compiled models
                # (same logic as resumption above)
                checkpoint_is_compiled = any(k.startswith("_orig_mod.") for k in best_model_state.keys())
                model_is_compiled = hasattr(model, "_orig_mod")

                if checkpoint_is_compiled and not model_is_compiled:
                    # Loading compiled checkpoint into non-compiled model: strip "_orig_mod." prefix
                    best_model_state = {k.replace("_orig_mod.", ""): v for k, v in best_model_state.items()}
                elif not checkpoint_is_compiled and model_is_compiled:
                    # Loading non-compiled checkpoint into compiled model: add "_orig_mod." prefix
                    best_model_state = {f"_orig_mod.{k}": v for k, v in best_model_state.items()}

                model.load_state_dict(best_model_state)

            # Evaluate on test set
            test_metrics = evaluate(model, test_loader, device)
            save_json(output_dir / "test_metrics.json", test_metrics)

            test_metrics_prefixed = {f"test_{k}": v for k, v in test_metrics.items()}
            mlflow_utils.log_metrics(test_metrics_prefixed)
            if best_model_path.exists():
                mlflow_utils.log_model(best_model_path, artifact_path="model")
            mlflow_utils.log_artifact(best_dir / "config.yaml")
            mlflow_utils.log_artifact(output_dir / "test_metrics.json")

            wandb_utils.log_metrics({f"test/{k}": v for k, v in test_metrics.items()})
            summary_payload: dict[str, Any] = {"output_dir": str(output_dir)}
            if math.isfinite(best_metric):
                summary_payload["best_metric"] = best_metric
            if best_metrics:
                summary_payload.update({f"best/{k}": v for k, v in best_metrics.items()})
            summary_payload.update({f"test/{k}": v for k, v in test_metrics.items()})
            wandb_utils.log_summary(summary_payload)

            for path in checkpoints_dir.glob("*.pt"):
                if path.name != "last.pt":
                    path.unlink(missing_ok=True)

            return {
                "best_metric": best_metric,
                "best_metrics": best_metrics,
                "test_metrics": test_metrics,
                "best_model_path": best_model_path if best_model_path.exists() else None,
                "output_dir": output_dir,
            }
