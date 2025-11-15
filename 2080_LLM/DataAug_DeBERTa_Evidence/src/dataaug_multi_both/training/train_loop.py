from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Mapping
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import mlflow
import numpy as np
import torch
from dataaug_multi_both.augment.textattack_factory import (
    build_augmenter,
    log_augmentation_samples,
)
from dataaug_multi_both.data import load_raw_datasets, tokenize_datasets
from dataaug_multi_both.mlflow_buffer import log_params_safe
from dataaug_multi_both.mlflow_init import (
    configure_mlflow,
    log_evaluation_artifact,
    log_metrics,
    log_run_metadata,
    register_run_end,
    register_run_start,
)
from dataaug_multi_both.model.multitask import build_multitask_model
from dataaug_multi_both.optim import build_optimizer
from dataaug_multi_both.training.checkpointing import CheckpointManager
from dataaug_multi_both.training.collator import DynamicPaddingCollator
from dataaug_multi_both.training.losses import MultiTaskLoss
from dataaug_multi_both.training.metrics import compute_metrics
from optuna.exceptions import TrialPruned
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

MAX_TOKEN_LENGTH = 512


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_dataloader(
    dataset,
    cfg: Mapping[str, Any],
    tokenizer,
    split: str,
    shuffle: bool,
) -> DataLoader:
    collator = DynamicPaddingCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=int(cfg["tokenizer"]["max_length"]),
    )

    # Use separate batch size for evaluation to maximize throughput
    # Training uses cfg["train"]["per_device_batch_size"]
    # Evaluation can use larger batches since no gradients are computed
    is_train = split == "train"
    if is_train:
        batch_size = int(cfg["train"]["per_device_batch_size"])
        num_workers = int(cfg["train"].get("num_workers", 4))
    else:
        # Evaluation can use 2x batch size since no gradients/optimizer states
        train_bs = int(cfg["train"]["per_device_batch_size"])
        eval_bs_multiplier = float(cfg["train"].get("eval_batch_size_multiplier", 2.0))
        batch_size = int(train_bs * eval_bs_multiplier)
        # Evaluation can use more workers since it's faster
        num_workers = int(cfg["train"].get("eval_num_workers",
                                          cfg["train"].get("num_workers", 4)))

    # Optimization: persistent_workers keeps worker processes alive between epochs
    # This avoids the overhead of spawning/destroying workers each epoch
    # Only beneficial when num_workers > 0
    persistent_workers = num_workers > 0

    return DataLoader(
        dataset[split],
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=True,  # Faster CPU->GPU transfer via page-locked memory
        persistent_workers=persistent_workers,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch 2 batches per worker
    )


def _prepare_scheduler(
    cfg: Mapping[str, Any],
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
) -> torch.optim.lr_scheduler._LRScheduler:
    epochs = int(cfg["train"]["num_epochs"])
    grad_accum = max(1, int(cfg["train"]["grad_accum_steps"]))
    total_steps = max(1, (len(train_loader) * epochs) // grad_accum)
    warmup_ratio = float(cfg["sched"].get("warmup_ratio", 0.1))
    warmup_steps = int(total_steps * warmup_ratio)
    sched_type = cfg["sched"]["type"]

    # OneCycleLR requires total_steps >= 50 to avoid division by zero issues
    # This is because with warmup_ratio and pct_start calculations, we need enough steps
    if sched_type == "one_cycle" and total_steps < 50:
        logger.warning(
            "OneCycleLR requires at least 50 total_steps, got %d. Falling back to cosine scheduler.",
            total_steps,
        )
        sched_type = "cosine"

    if sched_type == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if sched_type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if sched_type == "one_cycle":
        max_lr = max(group["lr"] for group in optimizer.param_groups if group["lr"] > 0)
        pct_start = warmup_steps / total_steps if total_steps > 0 else 0.1
        pct_start = max(0.001, min(0.9, pct_start))
        try:
            return OneCycleLR(
                optimizer,
                max_lr=max_lr,
                total_steps=total_steps,
                pct_start=pct_start,
                anneal_strategy="cos",
            )
        except (ZeroDivisionError, ValueError) as e:
            logger.warning(
                "OneCycleLR initialization failed with error: %s. Falling back to cosine scheduler.",
                e,
            )
            sched_type = "cosine"
            return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    raise ValueError(f"Unknown scheduler type: {sched_type}")


def _objective_metric(cfg: Mapping[str, Any], metrics: Mapping[str, float]) -> float:
    objective_cfg = cfg["objective"]
    composite_cfg = objective_cfg.get("composite", {})
    if composite_cfg.get("enabled"):
        score = 0.0
        for metric_name, weight in composite_cfg.get("weights", {}).items():
            if metric_name not in metrics:
                raise KeyError(
                    f"Composite metric expects {metric_name} but available metrics are {list(metrics)}"
                )
            score += float(weight) * float(metrics[metric_name])
        return score

    primary_metric = objective_cfg.get("primary_metric") or objective_cfg.get("metric")
    if primary_metric not in metrics:
        raise KeyError(
            f"Objective metric {primary_metric} not found in metrics {list(metrics)}"
        )
    return float(metrics[primary_metric])


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.best_epoch = -1
        self.wait = 0

    def update(self, value: float, epoch: int) -> bool:
        if value > self.best + self.min_delta:
            self.best = value
            self.best_epoch = epoch
            self.wait = 0
            return False
        self.wait += 1
        return self.wait >= self.patience


def _has_active_augmentation(cfg: Mapping[str, Any]) -> bool:
    simple = cfg.get("simple", {}).get("enabled_mask", {})
    ta = cfg.get("ta", {}).get("enabled_mask", {})
    return any(simple.values()) or any(ta.values())


def _ensure_token_length(cfg: Mapping[str, Any]) -> None:
    train_max = min(int(cfg["train"]["max_length"]), MAX_TOKEN_LENGTH)
    if train_max < int(cfg["train"]["max_length"]):
        logger.warning(
            "Capping train.max_length to %d (from %s) to respect model constraints.",
            MAX_TOKEN_LENGTH,
            cfg["train"]["max_length"],
        )
    cfg["train"]["max_length"] = train_max
    tokenizer_max = int(cfg["tokenizer"].get("max_length", train_max))
    if tokenizer_max > MAX_TOKEN_LENGTH:
        logger.warning(
            "Capping tokenizer.max_length to %d (from %s).",
            MAX_TOKEN_LENGTH,
            tokenizer_max,
        )
        tokenizer_max = MAX_TOKEN_LENGTH
    cfg["tokenizer"]["max_length"] = tokenizer_max


def run_training_job(
    cfg: Mapping[str, Any],
    *,
    trial: Optional[Any] = None,
    resume: bool = False,
) -> dict[str, Any]:
    cfg = deepcopy(cfg)
    _ensure_token_length(cfg)
    seed_everything(int(cfg["seed"]))

    configure_mlflow(cfg)
    run_name = f"trial_{trial.number:04d}" if trial else "train"
    start_time = time.time()

    with mlflow.start_run(run_name=run_name) as run:
        register_run_start()
        if trial:
            telemetry = trial.user_attrs.get("telemetry", {})
            mlflow.set_tag("trial.number", trial.number)
            mlflow.set_tag("trial.stage", telemetry.get("stage", "unknown"))
            if telemetry:
                mlflow.set_tags({f"telemetry.{k}": v for k, v in telemetry.items()})
                logger.info("Trial telemetry: %s", telemetry)
        try:
            run_id = run.info.run_id
            tokenizer = _load_tokenizer(cfg["encoder"]["tokenizer_name"], cfg["tokenizer"]["max_length"])

            raw_dataset, dataset_meta = load_raw_datasets(cfg)

            augmentation_cfg = cfg.get("augmentation", {})
            augment_fn = build_augmenter(augmentation_cfg)
            apply_aug = _has_active_augmentation(augmentation_cfg)
            evidence_field = augmentation_cfg.get("apply_to", cfg["data"]["fields"]["evidence"])
            sample_limit = int(augmentation_cfg.get("sample_limit", 10))
            augmentation_samples: list[tuple[str, str]] = []

            if apply_aug:
                logger.info("Applying augmentation to training split (evidence field=%s)", evidence_field)

                def apply_aug_fn(example: Mapping[str, Any]) -> Mapping[str, Any]:
                    original = example[evidence_field]
                    augmented = augment_fn(original)
                    if len(augmentation_samples) < sample_limit:
                        augmentation_samples.append((original, augmented))
                    example = dict(example)
                    example[evidence_field] = augmented
                    return example

                raw_dataset["train"] = raw_dataset["train"].map(apply_aug_fn)
                if augmentation_samples:
                    artifact_base = Path(cfg["checkpoint"]["dir"]) / run_name
                    artifact_dir = artifact_base / augmentation_cfg.get("artifact_subdir", "augmentation")
                    log_augmentation_samples(
                        augmentation_samples,
                        artifact_dir,
                        limit=sample_limit,
                    )

            augmentation_signature = json.dumps(
                {
                    "simple": augmentation_cfg.get("simple", {}),
                    "ta": augmentation_cfg.get("ta", {}),
                    "compose_cross_family": augmentation_cfg.get("compose_cross_family"),
                },
                sort_keys=True,
            )

            tokenized_dataset, extra_meta = tokenize_datasets(
                raw_dataset,
                cfg,
                tokenizer,
                augmentation_signature=augmentation_signature,
            )
            class_weights = {
                key: torch.tensor(value, dtype=torch.float32)
                for key, value in extra_meta["class_weights"].items()
            }

            log_run_metadata(
                cfg,
                {
                    "dataset_id": dataset_meta.dataset_id,
                    "revision": dataset_meta.revision,
                    "num_examples": dataset_meta.num_examples,
                    "tokenizer_name": tokenizer.name_or_path,
                },
            )

            train_loader = _build_dataloader(tokenized_dataset, cfg, tokenizer, "train", shuffle=True)
            val_loader = _build_dataloader(tokenized_dataset, cfg, tokenizer, "validation", shuffle=False)

            model = build_multitask_model(cfg)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            optimizer = build_optimizer(cfg, model)
            scheduler = _prepare_scheduler(cfg, optimizer, train_loader)

            loss_module = MultiTaskLoss(
                label_smoothing=float(cfg["loss"]["label_smoothing"]),
                class_weighting=cfg["loss"]["class_weighting"],
                class_weights=class_weights,
            )

            checkpoint_cfg = cfg["checkpoint"]
            ckpt_manager = CheckpointManager(checkpoint_cfg["dir"], checkpoint_cfg["filename"])
            grad_accum = max(1, int(cfg["train"]["grad_accum_steps"]))
            task_weight = float(cfg["mtl"]["task_weight_evidence"])
            num_epochs = int(cfg["train"]["num_epochs"])

            amp_mode = cfg["train"].get("amp", "bf16")
            bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
            if amp_mode == "bf16" and not bf16_supported:
                logger.warning("bf16 requested but not supported. Falling back to fp16.")
                amp_mode = "fp16"
            use_amp = torch.cuda.is_available() and amp_mode in {"fp16", "bf16"}
            amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
            scaler = GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)

            start_epoch = 0
            best_metric = float("-inf")
            best_metrics: dict[str, float] = {}
            best_checkpoint_path: Optional[Path] = None

            if resume and ckpt_manager.latest():
                state = ckpt_manager.load()
                model.load_state_dict(state["model_state"])
                optimizer.load_state_dict(state["optimizer_state"])
                scheduler.load_state_dict(state["scheduler_state"])
                if scaler.is_enabled() and state.get("scaler_state"):
                    scaler.load_state_dict(state["scaler_state"])
                start_epoch = state["epoch"] + 1
                best_metric = state.get("best_metric", best_metric)
                best_metrics = state.get("best_metrics", best_metrics)
                best_path = state.get("best_checkpoint_path")
                if best_path:
                    best_checkpoint_path = Path(best_path)

            early_cfg = cfg["train"].get("early_stopping", {})
            patience = int(early_cfg.get("patience", 20))
            min_delta = float(early_cfg.get("min_delta", 1e-6))
            early_stopper = EarlyStopping(patience=patience, min_delta=min_delta)

            def evaluate(loader: DataLoader, split: str) -> dict[str, float]:
                model.eval()
                logits_ev, logits_cr, labels_ev, labels_cr = [], [], [], []
                with torch.no_grad():
                    for batch in loader:
                        batch = {
                            k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()
                        }
                        outputs = model(batch["input_ids"], batch["attention_mask"])
                        logits_ev.append(outputs["evidence_logits"].cpu())
                        logits_cr.append(outputs["criteria_logits"].cpu())
                        labels_ev.append(batch["evidence_label"].cpu())
                        labels_cr.append(batch["criteria_label"].cpu())
                logits_ev_tensor = torch.cat(logits_ev).numpy()
                logits_cr_tensor = torch.cat(logits_cr).numpy()
                labels_ev_tensor = torch.cat(labels_ev).numpy()
                labels_cr_tensor = torch.cat(labels_cr).numpy()
                return compute_metrics(
                    logits_ev_tensor,
                    logits_cr_tensor,
                    labels_ev_tensor,
                    labels_cr_tensor,
                    split=split,
                )

            pruned_epoch: Optional[int] = None

            # Setup progress bars for visualization
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
            ) as progress:
                epoch_task = progress.add_task(
                    "[cyan]Training epochs",
                    total=num_epochs - start_epoch,
                )

                for epoch in range(start_epoch, num_epochs):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    epoch_loss = 0.0

                    # Add batch progress bar for current epoch
                    batch_task = progress.add_task(
                        f"[green]Epoch {epoch + 1}/{num_epochs}",
                        total=len(train_loader),
                    )

                    for step, batch in enumerate(train_loader):
                        batch = {
                            k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()
                        }

                        autocast_ctx = (
                            autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp)
                            if torch.cuda.is_available()
                            else nullcontext()
                        )
                        with autocast_ctx:
                            outputs = model(batch["input_ids"], batch["attention_mask"])
                            loss_dict = loss_module(
                                outputs["evidence_logits"],
                                outputs["criteria_logits"],
                                batch["evidence_label"],
                                batch["criteria_label"],
                                task_weight,
                            )
                            loss = loss_dict["loss"] / grad_accum

                        if scaler.is_enabled():
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        epoch_loss += loss_dict["loss"].item()

                        if (step + 1) % grad_accum == 0:
                            if scaler.is_enabled():
                                scaler.step(optimizer)
                                scaler.update()
                            else:
                                optimizer.step()
                            scheduler.step()
                            optimizer.zero_grad(set_to_none=True)

                        # Update batch progress
                        progress.update(batch_task, advance=1)

                    train_metrics = evaluate(train_loader, "train")
                    train_metrics["train_loss"] = epoch_loss / max(1, len(train_loader))
                    log_metrics(train_metrics, step=epoch + 1)

                    val_metrics = evaluate(val_loader, "val")
                    log_metrics(val_metrics, step=epoch + 1)
                    objective_value = _objective_metric(cfg, val_metrics)

                    # Remove batch progress bar after epoch completes
                    progress.remove_task(batch_task)

                    # Update epoch description with metrics
                    progress.update(
                        epoch_task,
                        advance=1,
                        description=f"[cyan]Training epochs (val_loss: {val_metrics.get('val_loss', 0):.4f})",
                    )

                    if trial:
                        trial.report(objective_value, step=epoch + 1)
                        if trial.should_prune():
                            pruned_epoch = epoch + 1
                            trial.set_user_attr("pruned_at_epoch", pruned_epoch)
                            log_metrics({"pruned_objective": objective_value}, step=epoch + 1)
                            mlflow.set_tag("status", "pruned")
                            raise TrialPruned(f"Pruned at epoch {pruned_epoch}")

                    if objective_value > best_metric:
                        best_metric = objective_value
                        best_metrics = dict(val_metrics)
                        state = {
                            "epoch": epoch,
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "scheduler_state": scheduler.state_dict(),
                            "scaler_state": scaler.state_dict() if scaler.is_enabled() else None,
                            "best_metric": best_metric,
                            "best_metrics": best_metrics,
                            "best_checkpoint_path": str(ckpt_manager.path),
                        }
                        best_checkpoint_path = ckpt_manager.save(state)
                        mlflow.set_tag("best_epoch", epoch + 1)

                    if early_stopper.update(objective_value, epoch + 1):
                        logger.info(
                            "Early stopping triggered at epoch %d (best epoch=%d, best=%.6f)",
                            epoch + 1,
                            early_stopper.best_epoch,
                            early_stopper.best,
                        )
                        break

            epochs_run = start_epoch
            if "epoch" in locals():
                epochs_run = epoch + 1

            test_metrics: dict[str, float] = {}
            if "test" in tokenized_dataset:
                test_loader = _build_dataloader(
                    tokenized_dataset, cfg, tokenizer, "test", shuffle=False
                )
                test_metrics = evaluate(test_loader, "test")
                log_metrics(test_metrics, step=epochs_run + 1)

            evaluation_dir = Path("experiments") / run_name
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            evaluation_report = {
                "run_id": run_id,
                "training_metrics": best_metrics,
                "test_metrics": test_metrics,
            }
            report_path = log_evaluation_artifact(evaluation_report, evaluation_dir, run_id)

            if best_checkpoint_path:
                log_params_safe({"checkpoint.best": str(best_checkpoint_path)})

            log_metrics({"objective": best_metric}, step=epochs_run + 1)

            duration = time.time() - start_time
            mlflow.set_tag("duration_seconds", f"{duration:.2f}")
            mlflow.set_tag("status", "completed")
            result = {
                "objective": best_metric,
                "metrics": {**best_metrics, **test_metrics},
                "checkpoint_path": str(best_checkpoint_path) if best_checkpoint_path else "",
                "evaluation_report_path": str(report_path),
                "epochs_run": epochs_run,
                "pruned_at_epoch": pruned_epoch or 0,
            }
            return result

        except TrialPruned:
            raise
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc):
                context = {
                    "oom.batch_size": cfg["train"]["per_device_batch_size"],
                    "oom.grad_accum": cfg["train"]["grad_accum_steps"],
                    "oom.max_length": cfg["train"]["max_length"],
                    "oom.model_name": cfg["encoder"]["model_name"],
                }
                logger.error("CUDA OOM encountered: %s", context)
                log_params_safe(context)
                if trial:
                    trial.set_user_attr("oom_context", context)
                    raise TrialPruned("Pruned due to CUDA OOM") from exc
            raise
        finally:
            register_run_end()
def _load_tokenizer(model_name: str, max_length: int) -> Any:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as exc:  # pragma: no cover - propagate failure
        hint = ""
        if "SentencePiece" in str(exc):
            hint = (
                " (install the 'sentencepiece' package or ensure the tokenizer files are available locally)"
            )
        raise RuntimeError(f"Failed to load tokenizer {model_name}: {exc}{hint}") from exc
    tokenizer.model_max_length = max_length
    return tokenizer
