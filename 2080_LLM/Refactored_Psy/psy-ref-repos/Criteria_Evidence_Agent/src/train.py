"""Training script for multi-label classification model."""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torch.amp import GradScaler, autocast
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None  # type: ignore[assignment]

from src.data.dataset import DataModule
from src.models.model import EvidenceModel
from src.utils import (
    configure_mlflow,
    evaluate,
    flatten_dict,
    get_optimizer,
    get_scheduler,
    prepare_thresholds,
    set_seed,
)
from src.utils.ema import EMA
from src.utils.training import compute_loss

# Register env resolver for OmegaConf
OmegaConf.register_new_resolver("env", lambda var, default=None: os.getenv(var, default))


def _to_bool(value: Any) -> bool:
    """Coerce common truthy inputs to boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    if value is None:
        return False
    return bool(value)


def start_wandb_run(
    cfg: DictConfig, run_name: str, job_type: Optional[str] = None
) -> Optional[Any]:
    """Initialize a Weights & Biases run if enabled in the configuration."""
    wandb_cfg = OmegaConf.select(cfg, "wandb")
    if wandb_cfg is None or not _to_bool(wandb_cfg.get("enabled", False)):
        return None

    if wandb is None:
        raise ImportError(
            "wandb package is required when wandb.enabled is true. Install with `pip install wandb`."
        )

    mode = wandb_cfg.get("mode")
    if mode:
        os.environ.setdefault("WANDB_MODE", str(mode))

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    flattened = flatten_dict(config_dict) if isinstance(config_dict, dict) else {}

    init_kwargs: Dict[str, Any] = {
        "project": wandb_cfg.get("project"),
        "name": run_name,
        "config": flattened,
        "reinit": True,
    }

    applied_job_type = job_type or wandb_cfg.get("job_type")
    if applied_job_type:
        init_kwargs["job_type"] = applied_job_type

    entity = wandb_cfg.get("entity")
    if entity:
        init_kwargs["entity"] = entity

    group = wandb_cfg.get("group")
    if group:
        init_kwargs["group"] = group

    tags = wandb_cfg.get("tags")
    if tags:
        init_kwargs["tags"] = list(tags)

    return wandb.init(**init_kwargs)


def train_loop(cfg: DictConfig) -> Dict[str, float]:
    """Main training loop.

    Args:
        cfg: Hydra configuration

    Returns:
        Dictionary containing training results
    """
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log device information
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    print(f"{'='*60}\n")

    # Setup data
    data_module = DataModule(cfg.data, cfg.model)
    train_loader, val_loader, test_loader = data_module.dataloaders(
        batch_size=cfg.training.batch_size,
        val_batch_size=cfg.training.val_batch_size,
        test_batch_size=cfg.training.test_batch_size,
        num_workers=cfg.training.num_workers,
    )

    # Setup model
    model = EvidenceModel(cfg.model)
    model.to(device)

    wandb_cfg = OmegaConf.select(cfg, "wandb")

    def log_wandb(metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Safely log metrics to Weights & Biases if a run is active."""
        if wandb is None or wandb.run is None:
            return
        wandb.log(metrics, step=step)

    if (
        wandb_cfg
        and wandb is not None
        and wandb.run is not None
        and _to_bool(wandb_cfg.get("watch", False))
    ):
        log_freq = cfg.training.get("logging_interval", 100)
        log_freq = 100 if log_freq is None else max(1, int(log_freq))
        wandb.watch(model, log="all", log_freq=log_freq)

    head_cfgs = model.head_configs
    head_thresholds = {
        head_name: prepare_thresholds(head_cfg)
        for head_name, head_cfg in head_cfgs.items()
        if head_cfg.get("type") == "multi_label"
    }

    # Setup optimizer and scheduler
    optimizer = get_optimizer(model, cfg.training.optimizer)

    num_train_batches = len(train_loader)
    updates_per_epoch = math.ceil(num_train_batches / cfg.training.gradient_accumulation_steps)
    total_steps = updates_per_epoch * cfg.training.max_epochs
    scheduler = get_scheduler(optimizer, cfg.training.scheduler, total_steps)

    # Setup EMA if enabled
    ema = None
    ema_decay = cfg.training.get("ema_decay", 0.0)
    if ema_decay > 0:
        ema = EMA(model, decay=ema_decay)

    # Setup mixed precision training
    torch.set_float32_matmul_precision("medium")

    encoder_config = getattr(getattr(model, "encoder", None), "config", None)
    encoder_model_type = getattr(encoder_config, "model_type", "") or ""
    encoder_model_type = encoder_model_type.lower()

    amp_enabled = cfg.training.amp and device.type == "cuda"
    amp_restricted_types = {"deberta", "deberta-v2", "deberta-v3"}
    if amp_enabled and encoder_model_type in amp_restricted_types:
        print(
            "AMP disabled for DeBERTa-based encoders due to half precision overflow; "
            "running in float32 for stability."
        )
        amp_enabled = False

    is_bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    use_bf16 = amp_enabled and cfg.training.bf16 and is_bf16_supported
    if use_bf16 and encoder_model_type in amp_restricted_types:
        print(
            "BF16 autocast disabled for DeBERTa-based encoders due to known overflow; "
            "falling back to FP16."
        )
        use_bf16 = False

    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(device.type, enabled=amp_enabled and not use_bf16)

    loss_weights = cfg.training.loss_weights
    focal_cfg = cfg.training.focal

    # Training loop
    best_metric = -float("inf")
    best_epoch = -1
    epochs_without_improve = 0
    checkpoint_dir = Path("artifacts")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_dir / "best_model.pt"

    # Outer progress bar for epochs
    epoch_progress = tqdm(
        range(1, cfg.training.max_epochs + 1), desc="Training", position=0, leave=True
    )

    last_epoch = 0

    grad_accum_steps = cfg.training.gradient_accumulation_steps

    def perform_optimizer_step() -> None:
        if scaler.is_enabled():
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update()

    for epoch in epoch_progress:
        last_epoch = epoch
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, position=1)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress, start=1):
            global_step = (epoch - 1) * num_train_batches + step
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            with autocast(
                device.type,
                enabled=amp_enabled,
                dtype=amp_dtype,
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                head_outputs = outputs["head_outputs"]
                loss = compute_loss(head_outputs, batch, head_cfgs, loss_weights, focal_cfg, device)
            loss_value = loss.detach().item()
            loss = loss / grad_accum_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % grad_accum_steps == 0:
                perform_optimizer_step()

            running_loss += loss_value
            avg_loss = running_loss / step
            # Get learning rate (handle ReduceLROnPlateau differently)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                current_lr = optimizer.param_groups[0]["lr"]
            else:
                current_lr = scheduler.get_last_lr()[0]
            progress.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"})

            if step % cfg.training.logging_interval == 0:
                mlflow.log_metric(
                    "train_loss",
                    avg_loss,
                    step=(epoch - 1) * num_train_batches + step,
                )
                log_wandb(
                    {
                        "train/loss": avg_loss,
                        "train/lr": current_lr,
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        if num_train_batches % grad_accum_steps != 0:
            perform_optimizer_step()

        # Validation (use EMA parameters if enabled)
        if ema is not None:
            ema.apply_shadow()

        val_loss, val_metrics = evaluate(model, val_loader, device, cfg, head_thresholds)

        if ema is not None:
            ema.restore()
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        val_payload: Dict[str, float] = {"val/loss": val_loss, "epoch": float(epoch)}
        for name, value in val_metrics.items():
            if name != "val_loss":
                mlflow.log_metric(name, value, step=epoch)
            metric_name = name
            if metric_name.startswith("val_"):
                metric_name = metric_name[4:]
            val_payload[f"val/{metric_name}"] = value
        validation_step = epoch * num_train_batches
        log_wandb(val_payload, step=validation_step if num_train_batches else None)

        # Step ReduceLROnPlateau scheduler if used
        monitor_metric = val_metrics.get(cfg.training.early_stopping.monitor, float("-inf"))
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(monitor_metric)

        # Early stopping
        improved = monitor_metric > best_metric + cfg.training.early_stopping.min_delta

        if improved:
            best_metric = monitor_metric
            best_epoch = epoch
            epochs_without_improve = 0
            torch.save(model.state_dict(), best_checkpoint)
            mlflow.log_artifact(str(best_checkpoint), artifact_path="checkpoints")
            if wandb is not None and wandb.run is not None:
                wandb.run.summary["best_metric"] = best_metric
                wandb.run.summary["best_epoch"] = best_epoch
            status = "✓ Improved"
        else:
            epochs_without_improve += 1
            status = (
                f"No improvement ({epochs_without_improve}/{cfg.training.early_stopping.patience})"
            )

        # Update epoch progress bar
        epoch_progress.set_postfix(
            {
                "val_loss": f"{val_loss:.4f}",
                cfg.training.early_stopping.monitor: f"{monitor_metric:.4f}",
                "best": f"{best_metric:.4f}",
                "status": status,
            }
        )

        if epochs_without_improve >= cfg.training.early_stopping.patience:
            tqdm.write(f"\nEarly stopping triggered at epoch {epoch}")
            tqdm.write(
                f"Best {cfg.training.early_stopping.monitor}: {best_metric:.4f} at epoch {best_epoch}"
            )
            break

    epoch_progress.close()

    # Print training summary
    if epochs_without_improve < cfg.training.early_stopping.patience:
        tqdm.write(f"\nTraining completed - reached max epochs ({cfg.training.max_epochs})")
        tqdm.write(
            f"Best {cfg.training.early_stopping.monitor}: {best_metric:.4f} at epoch {best_epoch}"
        )

    # Load best model and evaluate on test set
    if best_checkpoint.exists():
        model.load_state_dict(torch.load(best_checkpoint, map_location=device, weights_only=True))
        tqdm.write(f"Loaded best model from epoch {best_epoch} for testing")

    # Use EMA for final evaluation if enabled
    if ema is not None:
        ema.apply_shadow()

    test_loss, test_metrics = evaluate(model, test_loader, device, cfg, head_thresholds)

    if ema is not None:
        ema.restore()
    mlflow.log_metric("test_loss", test_loss)
    test_payload: Dict[str, float] = {"test/loss": test_loss}
    for name, value in test_metrics.items():
        if name != "val_loss":
            mlflow.log_metric(name.replace("val_", "test_"), value)
        metric_name = name.replace("val_", "")
        test_payload[f"test/{metric_name}"] = value

    final_step = last_epoch * num_train_batches if num_train_batches else None
    log_wandb(test_payload, step=final_step)
    if wandb is not None and wandb.run is not None:
        wandb.run.summary["test_loss"] = test_loss
        for name, value in test_metrics.items():
            summary_key = name.replace("val_", "test_")
            wandb.run.summary[summary_key] = value

    results = {
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "test_loss": test_loss,
    }

    if (
        wandb_cfg
        and wandb is not None
        and wandb.run is not None
        and _to_bool(wandb_cfg.get("log_artifacts", True))
        and best_checkpoint.exists()
    ):
        artifact = wandb.Artifact(
            name=f"{cfg.model.encoder.type}-best-model",
            type="model",
            metadata={"best_metric": best_metric, "best_epoch": best_epoch},
        )
        artifact.add_file(str(best_checkpoint))
        wandb.log_artifact(artifact)

    log_dir = Path.cwd()
    (log_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    return results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training.

    Args:
        cfg: Hydra configuration
    """
    tracking_uri, artifact_uri = configure_mlflow(cfg)
    print(f"Using MLflow tracking URI: {tracking_uri}")
    if artifact_uri:
        print(f"Using MLflow artifact URI: {artifact_uri}")

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    flattened = flatten_dict(config_dict)

    wandb_run = start_wandb_run(cfg, run_name=cfg.model.encoder.type)

    try:
        with mlflow.start_run(run_name=cfg.model.encoder.type):
            mlflow.log_params(flattened)
            if cfg.mlflow.autolog:
                mlflow.pytorch.autolog(log_models=False)
            results = train_loop(cfg)
            mlflow.log_metric("best_metric", results["best_metric"])
            mlflow.log_metric("best_epoch", results["best_epoch"])
    finally:
        if wandb_run is not None and wandb is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
