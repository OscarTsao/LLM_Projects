"""Training script for multi-label classification model."""

import math
import os
from pathlib import Path
from typing import Dict

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

from src.data.dataset import DataModule
from src.models.model import EvidenceModel
from src.utils import (
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


def train_loop(cfg: DictConfig) -> Dict[str, float]:
    """Main training loop.

    Args:
        cfg: Hydra configuration

    Returns:
        Dictionary containing training results
    """
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    head_cfgs = model.head_configs
    head_thresholds = {
        head_name: prepare_thresholds(head_cfg)
        for head_name, head_cfg in head_cfgs.items()
        if head_cfg.get("type") == "multi_label"
    }

    # Setup optimizer and scheduler
    optimizer = get_optimizer(model, cfg.training.optimizer)

    updates_per_epoch = math.ceil(
        len(train_loader) / cfg.training.gradient_accumulation_steps
    )
    total_steps = updates_per_epoch * cfg.training.max_epochs
    scheduler = get_scheduler(optimizer, cfg.training.scheduler, total_steps)

    # Setup EMA if enabled
    ema = None
    ema_decay = cfg.training.get("ema_decay", 0.0)
    if ema_decay > 0:
        ema = EMA(model, decay=ema_decay)

    # Setup mixed precision training
    torch.set_float32_matmul_precision("medium")
    amp_enabled = cfg.training.amp and device.type == "cuda"
    is_bf16_supported = bool(
        getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    use_bf16 = amp_enabled and cfg.training.bf16 and is_bf16_supported
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = GradScaler(enabled=amp_enabled and not use_bf16)

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
        range(1, cfg.training.max_epochs + 1),
        desc="Training",
        position=0,
        leave=True
    )

    for epoch in epoch_progress:
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, position=1)
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(progress, start=1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            with autocast(
                enabled=amp_enabled,
                dtype=amp_dtype,
            ):
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                head_outputs = outputs["head_outputs"]
                loss = compute_loss(
                    head_outputs, batch, head_cfgs, loss_weights, focal_cfg, device
                )
                loss = loss / cfg.training.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % cfg.training.gradient_accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.training.max_grad_norm
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.training.max_grad_norm
                    )
                    optimizer.step()
                # Step scheduler (except ReduceLROnPlateau which steps after validation)
                if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # Update EMA
                if ema is not None:
                    ema.update()

            running_loss += loss.item()
            avg_loss = running_loss / step
            # Get learning rate (handle ReduceLROnPlateau differently)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                current_lr = optimizer.param_groups[0]['lr']
            else:
                current_lr = scheduler.get_last_lr()[0]
            progress.set_postfix({"loss": f"{avg_loss:.4f}", "lr": f"{current_lr:.2e}"})

            if step % cfg.training.logging_interval == 0:
                mlflow.log_metric(
                    "train_loss",
                    avg_loss,
                    step=(epoch - 1) * len(train_loader) + step,
                )

        # Validation (use EMA parameters if enabled)
        if ema is not None:
            ema.apply_shadow()

        val_loss, val_metrics = evaluate(
            model, val_loader, device, cfg, head_thresholds
        )

        if ema is not None:
            ema.restore()
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        for name, value in val_metrics.items():
            if name != "val_loss":
                mlflow.log_metric(name, value, step=epoch)

        # Step ReduceLROnPlateau scheduler if used
        monitor_metric = val_metrics.get(
            cfg.training.early_stopping.monitor, float("-inf")
        )
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
            status = "✓ Improved"
        else:
            epochs_without_improve += 1
            status = f"No improvement ({epochs_without_improve}/{cfg.training.early_stopping.patience})"

        # Update epoch progress bar
        epoch_progress.set_postfix({
            "val_loss": f"{val_loss:.4f}",
            cfg.training.early_stopping.monitor: f"{monitor_metric:.4f}",
            "best": f"{best_metric:.4f}",
            "status": status
        })

        if epochs_without_improve >= cfg.training.early_stopping.patience:
            tqdm.write(f"\nEarly stopping triggered at epoch {epoch}")
            tqdm.write(f"Best {cfg.training.early_stopping.monitor}: {best_metric:.4f} at epoch {best_epoch}")
            break

    epoch_progress.close()

    # Print training summary
    if epochs_without_improve < cfg.training.early_stopping.patience:
        tqdm.write(f"\nTraining completed - reached max epochs ({cfg.training.max_epochs})")
        tqdm.write(f"Best {cfg.training.early_stopping.monitor}: {best_metric:.4f} at epoch {best_epoch}")

    # Load best model and evaluate on test set
    if best_checkpoint.exists():
        model.load_state_dict(torch.load(best_checkpoint, map_location=device))
        tqdm.write(f"Loaded best model from epoch {best_epoch} for testing")

    # Use EMA for final evaluation if enabled
    if ema is not None:
        ema.apply_shadow()

    test_loss, test_metrics = evaluate(model, test_loader, device, cfg, head_thresholds)

    if ema is not None:
        ema.restore()
    mlflow.log_metric("test_loss", test_loss)
    for name, value in test_metrics.items():
        if name != "val_loss":
            mlflow.log_metric(name.replace("val_", "test_"), value)

    results = {
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "test_metrics": test_metrics,
        "test_loss": test_loss,
    }

    log_dir = Path.cwd()
    (log_dir / "metrics.json").write_text(OmegaConf.to_yaml(OmegaConf.create(results)))

    return results


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training.

    Args:
        cfg: Hydra configuration
    """
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    config_dict = OmegaConf.to_container(cfg, resolve=True)
    flattened = flatten_dict(config_dict)

    with mlflow.start_run(run_name=cfg.model.encoder.type):
        mlflow.log_params(flattened)
        if cfg.mlflow.autolog:
            mlflow.pytorch.autolog(log_models=False)
        results = train_loop(cfg)
        mlflow.log_metric("best_metric", results["best_metric"])
        mlflow.log_metric("best_epoch", results["best_epoch"])


if __name__ == "__main__":
    main()
