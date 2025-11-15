from __future__ import annotations

import logging
import sys
import math
import random
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import optuna
import torch
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from ..data.dataset import create_pytorch_dataset, load_hf_dataset
from ..hpo.search_space import suggest
from ..hpo.trial_executor import DatasetConfig, HardwareConfig, TrialResult
from ..models.criteria_model import CriteriaModel
from ..training.early_stop import EarlyStopping
from ..training.losses import build_criterion
from ..training.metrics import optimize_global_threshold
from ..utils.mlflow_setup import BufferedMLflowLogger, log_system_fingerprint

LOGGER = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "head.type": "linear",
    "head.dropout": 0.1,
    "head.bias": True,
    "pooling": "cls",
    "loss.name": "bce",
    "optim.name": "adamw",
    "optim.lr_encoder": 3e-5,
    "optim.lr_head": 1e-4,
    "optim.weight_decay": 0.01,
    "sched.name": "linear",
    "sched.warmup_ratio": 0.1,
    "train.grad_clip_norm": 1.0,
    "pred.threshold.policy": "fixed",
    "pred.threshold.global": 0.5,
}

AMP_DTYPES: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass(slots=True)
class TrainerSettings:
    dataset_cfg: DatasetConfig
    hardware_cfg: HardwareConfig
    experiments_dir: str
    experiment_name: str
    model_name: str = "microsoft/deberta-v3-base"
    deterministic: bool = False
    threshold_grid: tuple[float, float, int] = (0.2, 0.6, 101)


@dataclass(slots=True)
class StageBManager:
    spaces: list[dict[str, tuple]]
    candidates: list[dict[str, Any]]

    def sample(self, trial: optuna.Trial) -> dict[str, Any]:
        if not self.candidates:
            raise RuntimeError("Stage-B manager requires at least one candidate configuration.")
        if "space_index" in trial.user_attrs:
            index = int(trial.user_attrs["space_index"])
        else:
            index = trial.number % len(self.candidates)
            trial.set_user_attr("space_index", index)
        if index >= len(self.candidates):
            index = 0
        base = dict(self.candidates[index])
        spec = self.spaces[index] if index < len(self.spaces) else {}
        params = base.copy()
        for key, definition in spec.items():
            kind = definition[0]
            if kind == "freeze":
                params[key] = trial.suggest_categorical(key, [definition[1]])
            elif kind == "float":
                _, lo, hi = definition
                params[key] = trial.suggest_float(key, float(lo), float(hi))
            elif kind == "float_log":
                _, lo, hi = definition
                params[key] = trial.suggest_float(key, float(lo), float(hi), log=True)
            else:  # pragma: no cover - defensive
                raise ValueError(f"Unsupported Stage-B spec type: {kind}")
        return params


def merge_defaults(params: Mapping[str, Any]) -> dict[str, Any]:
    merged = DEFAULT_PARAMS.copy()
    merged.update({k: v for k, v in params.items() if v is not None})
    return merged


@lru_cache(maxsize=8)
def _load_tokenizer(model_name: str):
    return AutoTokenizer.from_pretrained(model_name)


@lru_cache(maxsize=4)
def _load_dataset(dataset_id: str, revision: str | None, cache_dir: str | None):
    dataset, metadata = load_hf_dataset(dataset_id, revision=revision, cache_dir=cache_dir, required_splits=("train", "validation"))
    return dataset, metadata


def _prepare_dataloaders(
    dataset_cfg: DatasetConfig,
    hardware_cfg: HardwareConfig,
    tokenizer,
    aug_cfg: Mapping[str, Any] | None = None,
) -> tuple[DataLoader, DataLoader, np.ndarray]:
    dataset, _ = _load_dataset(dataset_cfg.dataset_id, dataset_cfg.revision, dataset_cfg.cache_dir)

    train_split = dataset[dataset_cfg.train_split]
    val_split = dataset[dataset_cfg.validation_split]

    aug_prob = float(aug_cfg.get("aug.prob", 0.0)) if aug_cfg else 0.0
    aug_methods = []

    train_dataset = create_pytorch_dataset(
        train_split,
        tokenizer=tokenizer,
        input_format="multi_label",
        max_length=hardware_cfg.max_length,
        augmentation_prob=aug_prob,
        augmentation_methods=aug_methods,
    )
    val_dataset = create_pytorch_dataset(
        val_split,
        tokenizer=tokenizer,
        input_format="multi_label",
        max_length=hardware_cfg.max_length,
    )

    # Only use pin_memory when CUDA is available
    use_pin_memory = hardware_cfg.pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=hardware_cfg.per_device_batch_size,
        shuffle=True,
        num_workers=hardware_cfg.num_workers,
        pin_memory=use_pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=hardware_cfg.per_device_batch_size,
        shuffle=False,
        num_workers=hardware_cfg.num_workers,
        pin_memory=use_pin_memory,
    )

    labels = np.array(train_split["criteria_labels"], dtype=np.float32)
    return train_loader, val_loader, labels


def _compute_pos_weight(labels: np.ndarray) -> torch.Tensor | None:
    if labels.size == 0:
        return None
    positives = labels.sum(axis=0)
    totals = labels.shape[0]
    negatives = totals - positives
    weight = (negatives + 1e-6) / (positives + 1e-6)
    return torch.tensor(weight, dtype=torch.float32)


def _build_scheduler(name: str, optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float):
    warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = min(warmup_steps, max(total_steps - 1, 0))
    if name == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    raise ValueError(f"Unsupported scheduler: {name}")


def _move_to_device(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def trainer_entrypoint(
    *,
    trial: optuna.Trial | None,
    stage: str,
    max_epochs: int,
    seed: int,
    mlflow_logger: BufferedMLflowLogger | None,
    run_dir: Path | None,
    settings: TrainerSettings,
    stage_b_manager: StageBManager | None = None,
    manual_params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(settings.deterministic, warn_only=True)
    except Exception:  # pragma: no cover - fallback for incompatible ops
        if settings.deterministic:
            LOGGER.warning("Deterministic mode requested but not fully supported by this build.")
    torch.backends.cudnn.deterministic = settings.deterministic
    torch.backends.cudnn.benchmark = not settings.deterministic

    hardware = replace(settings.hardware_cfg)
    hardware.validate()

    if settings.model_name == "microsoft/deberta-v3-base" and hardware.max_length > 512:
        hardware.max_length = 512

    tokenizer = _load_tokenizer(settings.model_name)
    LOGGER.info(f"Tokenizer loaded: {settings.model_name}")

    if trial is None and manual_params is None:
        params = DEFAULT_PARAMS.copy()
    elif trial is not None:
        if stage == "A":
            params = suggest(trial)
        elif stage == "B" and stage_b_manager is not None:
            params = stage_b_manager.sample(trial)
        else:
            params = suggest(trial)
    else:
        params = dict(manual_params or DEFAULT_PARAMS)

    params = merge_defaults(params)
    LOGGER.info("Parameters merged, preparing dataloaders...")

    train_loader, val_loader, labels = _prepare_dataloaders(
        settings.dataset_cfg,
        hardware,
        tokenizer,
        params,
    )
    LOGGER.info(f"Dataloaders prepared: train={len(train_loader)} batches, val={len(val_loader)} batches")

    if labels.ndim == 2 and labels.shape[1] > 0:
        num_labels = int(labels.shape[1])
    else:
        sample = train_loader.dataset[0]
        crit = sample.get("criteria_labels") if isinstance(sample, dict) else sample["criteria_labels"]
        num_labels = int(len(crit)) if crit is not None else 9

    LOGGER.info(f"Initializing model with {num_labels} labels...")
    model = CriteriaModel(
        num_labels=num_labels,
        pooling=params["pooling"],
        head_cfg=params,
        model_name=settings.model_name,
        gradient_checkpointing=hardware.gradient_checkpointing,
    )
    LOGGER.info("Model initialized successfully")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Moving model to device: {device}")
    model.to(device)
    LOGGER.info("Model moved to device")

    pos_weight = _compute_pos_weight(labels) if params.get("loss.name") == "weighted_bce" else None
    criterion = build_criterion(params, pos_weight=pos_weight.to(device) if pos_weight is not None else None)

    encoder_params = [
        p for name, p in model.named_parameters() if name.startswith("encoder") and p.requires_grad
    ]
    head_params = [
        p for name, p in model.named_parameters() if name.startswith("head") and p.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": params["optim.lr_encoder"]},
            {"params": head_params, "lr": params["optim.lr_head"]},
        ],
        weight_decay=params["optim.weight_decay"],
    )

    steps_per_epoch = math.ceil(len(train_loader) / max(1, hardware.grad_accumulation_steps))
    scheduler = _build_scheduler(
        params["sched.name"],
        optimizer,
        total_steps=max(1, steps_per_epoch * max_epochs),
        warmup_ratio=params["sched.warmup_ratio"],
    )

    amp_dtype = AMP_DTYPES.get(hardware.amp_dtype, None)
    # Determine device type for AMP operations
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = GradScaler(device_type, enabled=amp_dtype is not None and torch.cuda.is_available())

    run_name = f"{settings.experiment_name}_{stage}_seed{seed}"
    start_time = time.time()

    if mlflow.active_run() is not None:
        mlflow.end_run()

    run_id = ""
    with mlflow.start_run(run_name=run_name):
        active_run = mlflow.active_run()
        if active_run is not None:
            run_id = active_run.info.run_id
        if mlflow_logger is not None:
            mlflow_logger.replay_buffer()
            mlflow_logger.log_params(params)
            mlflow_logger.log_params(
                {
                    "dataset_id": settings.dataset_cfg.dataset_id,
                    "dataset_revision": settings.dataset_cfg.revision or "latest",
                    "hardware.max_length": hardware.max_length,
                    "hardware.batch_size": hardware.per_device_batch_size,
                    "hardware.grad_accumulation": hardware.grad_accumulation_steps,
                }
            )
            mlflow_logger.set_tags(
                {
                    "stage": stage,
                    "seed": str(seed),
                    "deterministic": str(settings.deterministic),
                }
            )
        log_system_fingerprint()

        best_metric = float("-inf")
        best_threshold = params.get("pred.threshold.global", 0.5)
        best_epoch = -1
        early_stop = EarlyStopping(patience=20, mode="max", min_delta=1e-6)
        epochs_trained = 0

        LOGGER.info(f"Starting training loop for {max_epochs} epochs...")
        try:
            epoch_pbar = tqdm(range(max_epochs), desc="Training", unit="epoch", position=0, leave=True)
            for epoch in epoch_pbar:
                model.train()
                optimizer.zero_grad()
                total_loss = 0.0

                train_pbar = tqdm(
                    enumerate(train_loader),
                    total=len(train_loader),
                    desc=f"Epoch {epoch + 1}/{max_epochs} [Train]",
                    unit="batch",
                    position=1,
                    leave=False,
                )

                for step, batch in train_pbar:
                    batch = _move_to_device(batch, device)
                    targets = batch.pop("criteria_labels")

                    with autocast(device_type, enabled=amp_dtype is not None and torch.cuda.is_available(), dtype=amp_dtype):
                        logits = model(**batch)
                        loss = criterion(logits, targets) / max(1, hardware.grad_accumulation_steps)

                    total_loss += loss.item()

                    if scaler.is_enabled():
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    if (step + 1) % hardware.grad_accumulation_steps == 0:
                        if params["train.grad_clip_norm"] > 0:
                            if scaler.is_enabled():
                                scaler.unscale_(optimizer)
                            clip_grad_norm_(model.parameters(), params["train.grad_clip_norm"])
                        if scaler.is_enabled():
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                    avg_loss = total_loss / float(step + 1)
                    train_pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

                if len(train_loader) > 0 and len(train_loader) % hardware.grad_accumulation_steps != 0:
                    if params["train.grad_clip_norm"] > 0:
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        clip_grad_norm_(model.parameters(), params["train.grad_clip_norm"])
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                model.eval()
                val_losses: list[float] = []
                y_true: list[np.ndarray] = []
                y_prob: list[np.ndarray] = []

                val_pbar = tqdm(
                    enumerate(val_loader),
                    total=len(val_loader),
                    desc=f"Epoch {epoch + 1}/{max_epochs} [Val]",
                    unit="batch",
                    position=1,
                    leave=False,
                )

                with torch.no_grad():
                    for v_step, batch in val_pbar:
                        batch = _move_to_device(batch, device)
                        targets = batch.pop("criteria_labels")
                        with autocast(device_type, enabled=amp_dtype is not None and torch.cuda.is_available(), dtype=amp_dtype):
                            logits = model(**batch)
                            val_loss = criterion(logits, targets)
                        val_losses.append(val_loss.item())
                        y_prob.append(torch.sigmoid(logits).cpu().float().numpy())
                        y_true.append(targets.cpu().float().numpy())

                        avg_vloss = float(np.mean(val_losses)) if val_losses else 0.0
                        val_pbar.set_postfix({"val_loss": f"{avg_vloss:.4f}"})

                y_true_arr = np.concatenate(y_true, axis=0)
                y_prob_arr = np.concatenate(y_prob, axis=0)
                val_loss_mean = float(np.mean(val_losses)) if val_losses else 0.0

                if y_true_arr.size == 0 or y_prob_arr.size == 0:
                    thr = float(params.get("pred.threshold.global", 0.5))
                    macro_f1 = 0.0
                elif params["pred.threshold.policy"] == "opt_global":
                    thr, macro_f1 = optimize_global_threshold(
                        y_true_arr,
                        y_prob_arr,
                        lo=settings.threshold_grid[0],
                        hi=settings.threshold_grid[1],
                        steps=settings.threshold_grid[2],
                    )
                else:
                    thr = float(params.get("pred.threshold.global", 0.5))
                    preds = (y_prob_arr >= thr).astype(int)
                    macro_f1 = float(f1_score(y_true_arr, preds, average="macro", zero_division=0))

                if mlflow_logger is not None:
                    mlflow_logger.log_metrics(
                        {
                            "train_loss": total_loss / max(1, len(train_loader)),
                            "val_loss": val_loss_mean,
                            "macro_f1": macro_f1,
                            "threshold": thr,
                        },
                        step=epoch + 1,
                    )

                # Update epoch progress bar with summary
                epoch_pbar.set_postfix({
                    "train_loss": f"{(total_loss / max(1, len(train_loader))):.4f}",
                    "val_loss": f"{val_loss_mean:.4f}",
                    "f1": f"{macro_f1:.4f}",
                    "best_f1": f"{best_metric if best_metric != float('-inf') else 0.0:.4f}",
                })

                if trial is not None:
                    trial.report(macro_f1, step=epoch + 1)
                    if trial.should_prune():
                        raise optuna.TrialPruned(f"Pruned at epoch {epoch + 1}")

                if macro_f1 > best_metric + 1e-12:
                    best_metric = macro_f1
                    best_threshold = thr
                    best_epoch = epoch

                early_stop.update(macro_f1)
                epochs_trained = epoch + 1
                if early_stop.stop:
                    break

            epoch_pbar.close()

        except torch.cuda.OutOfMemoryError as exc:
            if 'epoch_pbar' in locals():
                epoch_pbar.close()
            torch.cuda.empty_cache()
            if mlflow_logger is not None:
                mlflow_logger.set_tags({"trial_pruned": "1", "prune_reason": "CUDA OOM"})
            raise optuna.TrialPruned("CUDA OOM") from exc
        except RuntimeError as exc:
            if 'epoch_pbar' in locals():
                epoch_pbar.close()
            if "out of memory" in str(exc).lower():
                torch.cuda.empty_cache()
                if mlflow_logger is not None:
                    mlflow_logger.set_tags({"trial_pruned": "1", "prune_reason": "CUDA OOM"})
                raise optuna.TrialPruned("CUDA OOM") from exc
            raise

    duration = time.time() - start_time
    if best_metric == float("-inf"):
        best_metric = 0.0
        best_threshold = float(params.get("pred.threshold.global", 0.5))

    result = TrialResult(
        metric=best_metric,
        threshold=best_threshold,
        run_id=run_id,
        params=dict(params),
        best_epoch=best_epoch,
        epochs_trained=epochs_trained,
        duration_seconds=duration,
    )
    if trial is not None:
        trial.set_user_attr("threshold", best_threshold)
        trial.set_user_attr("best_epoch", best_epoch)
    return {
        "macro_f1": result.metric,
        "threshold": result.threshold,
        "best_epoch": result.best_epoch,
        "epochs_trained": result.epochs_trained,
        "run_id": result.run_id,
        "params": result.params,
        "duration_seconds": result.duration_seconds,
    }


__all__ = [
    "DEFAULT_PARAMS",
    "TrainerSettings",
    "StageBManager",
    "merge_defaults",
    "trainer_entrypoint",
]
