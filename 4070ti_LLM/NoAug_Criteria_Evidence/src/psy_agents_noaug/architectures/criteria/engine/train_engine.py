from __future__ import annotations

import math
import random
import time
from contextlib import nullcontext
from copy import deepcopy
from typing import Any

import numpy as np
import optuna
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_scheduler

from psy_agents_noaug.architectures.criteria.data.dataset import CriteriaDataset
from psy_agents_noaug.architectures.criteria.models.model import Model
from psy_agents_noaug.architectures.criteria.utils import (
    best_model_saver,
    configure_mlflow,
    enable_autologging,
    get_artifact_dir,
    get_logger,
    load_training_state,
    mlflow_run,
    set_seed,
    training_state_exists,
)


def _flatten_dict(prefix: str, value: Any, accumulator: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, val in value.items():
            _flatten_dict(f"{prefix}.{key}" if prefix else key, val, accumulator)
    elif isinstance(value, (list, tuple)):
        accumulator[prefix] = ",".join(map(str, value))
    else:
        accumulator[prefix] = value


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _prepare_datasets(
    config: dict[str, Any],
    tokenizer: AutoTokenizer,
    seed: int,
) -> tuple[Dataset, Dataset, Dataset | None]:
    dataset_cfg = config.get("dataset", {})
    splits = dataset_cfg.get("splits", [0.8, 0.1, 0.1])
    if isinstance(splits, dict):
        splits = [
            splits.get("train", 0.8),
            splits.get("val", 0.1),
            splits.get("test", 0.1),
        ]
    if len(splits) == 2:
        splits = [splits[0], splits[1], 0.0]
    if not math.isclose(sum(splits), 1.0, rel_tol=1e-3):
        raise ValueError("Dataset splits must sum to 1.0")

    dataset = CriteriaDataset(
        csv_path=dataset_cfg.get("path"),
        tokenizer=tokenizer,
        tokenizer_name=dataset_cfg.get("tokenizer_name", tokenizer.name_or_path),
        text_column=dataset_cfg.get("text_column", "sentence_text"),
        label_column=dataset_cfg.get("label_column", "status"),
        max_length=dataset_cfg.get("max_length", 256),
        padding=dataset_cfg.get("padding", "max_length"),
        truncation=dataset_cfg.get("truncation", True),
    )

    total_size = len(dataset)
    train_size = int(total_size * splits[0])
    val_size = int(total_size * splits[1])
    test_size = total_size - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=generator,
    )
    if len(test_dataset) == 0:
        test_dataset = None
    return train_dataset, val_dataset, test_dataset


def _create_dataloaders(
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset | None,
    *,
    config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})
    num_workers = dataset_cfg.get("num_workers", 0)
    pin_memory = device.type == "cuda"
    persistent_workers = num_workers > 0

    train_generator = torch.Generator().manual_seed(seed)
    eval_generator = torch.Generator().manual_seed(seed + 1)
    test_generator = torch.Generator().manual_seed(seed + 2)

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.get("train_batch_size", 16),
        shuffle=True,
        worker_init_fn=_seed_worker,
        generator=train_generator,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg.get("eval_batch_size", 32),
        shuffle=False,
        worker_init_fn=_seed_worker if num_workers else None,
        generator=eval_generator,
        pin_memory=pin_memory,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=training_cfg.get("eval_batch_size", 32),
            shuffle=False,
            worker_init_fn=_seed_worker if num_workers else None,
            generator=test_generator,
            pin_memory=pin_memory,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
        )

    return train_loader, val_loader, test_loader


def _build_model(config: dict[str, Any]) -> Model:
    model_cfg = config.get("model", {})
    model_name = model_cfg.get("pretrained_model", "bert-base-uncased")
    num_labels = model_cfg.get("num_labels", 2)
    return Model(
        model_name=model_name,
        num_labels=num_labels,
        classifier_dropout=model_cfg.get("classifier_dropout", 0.1),
        classifier_layer_num=model_cfg.get("classifier_layer_num", 1),
        classifier_hidden_dims=model_cfg.get("classifier_hidden_dims"),
    )


def _select_device(config: dict[str, Any]) -> torch.device:
    preferred = config.get("training", {}).get("device")
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def _create_optimizer(
    model: nn.Module, config: dict[str, Any]
) -> torch.optim.Optimizer:
    training_cfg = config.get("training", {})
    optimizer_cfg = training_cfg.get("optimizer", {}) or {}
    optimizer_name = optimizer_cfg.get("name", "adamw").lower()
    learning_rate = training_cfg.get("learning_rate", 2e-5)
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)
    no_decay = ["bias", "LayerNorm.weight"]

    grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    if optimizer_name == "adamw":
        return torch.optim.AdamW(grouped_parameters, lr=learning_rate)
    if optimizer_name == "adam":
        return torch.optim.Adam(
            grouped_parameters, lr=learning_rate, weight_decay=weight_decay
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            grouped_parameters,
            lr=learning_rate,
            momentum=optimizer_cfg.get("momentum", 0.9),
            weight_decay=weight_decay,
        )
    if optimizer_name == "adafactor":
        from transformers.optimization import Adafactor

        return Adafactor(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            scale_parameter=False,
            relative_step=False,
        )
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'")


def _create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    num_training_steps: int,
) -> Any | None:
    scheduler_cfg = config.get("training", {}).get("scheduler", {}) or {}
    scheduler_name = scheduler_cfg.get("name", "linear")
    if scheduler_name in (None, "", "none"):
        return None

    warmup_steps = scheduler_cfg.get("warmup_steps", 0)
    if scheduler_name == "cosine_with_restarts":
        num_cycles = scheduler_cfg.get("num_cycles", 1)
        return get_scheduler(
            scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=num_cycles,
        )
    return get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )


def _classification_metrics(
    predictions: torch.Tensor, targets: torch.Tensor
) -> dict[str, float]:
    preds = predictions.argmax(dim=-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()

    tp = ((preds == 1) & (targets == 1)).sum().item()
    fp = ((preds == 1) & (targets == 0)).sum().item()
    fn = ((preds == 0) & (targets == 1)).sum().item()

    accuracy = correct / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
    amp_enabled: bool,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    all_preds: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    autocast_enabled = amp_enabled and device.type == "cuda"

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with autocast(enabled=autocast_enabled):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = loss_fn(outputs, labels)

            losses.append(loss.item())
            all_preds.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())

    predictions = torch.cat(all_preds)
    targets = torch.cat(all_labels)
    metrics = _classification_metrics(predictions, targets)
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def _move_batch_to_device(
    batch: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def train(
    config: dict[str, Any],
    *,
    trial: optuna.Trial | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    logger = get_logger(__name__)
    training_cfg = config.get("training", {})
    deterministic = training_cfg.get("deterministic", True)
    seed_value = training_cfg.get("seed", 42)

    seed = set_seed(seed_value, deterministic=deterministic)
    logger.info("Using random seed %s (deterministic=%s)", seed, deterministic)

    tokenizer = AutoTokenizer.from_pretrained(
        config.get("model", {}).get("pretrained_model", "bert-base-uncased")
    )
    train_dataset, val_dataset, test_dataset = _prepare_datasets(
        config, tokenizer, seed
    )

    device = _select_device(config)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = not deterministic

    train_loader, val_loader, test_loader = _create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        config=config,
        seed=seed,
        device=device,
    )

    model = _build_model(config)
    model.to(device)

    optimizer = _create_optimizer(model, config)
    grad_accum = training_cfg.get("gradient_accumulation", 1)
    epochs = training_cfg.get("epochs", 3)
    total_update_steps = math.ceil(len(train_loader) / grad_accum) * epochs
    scheduler = _create_scheduler(optimizer, config, total_update_steps)

    amp_enabled = (
        training_cfg.get("mixed_precision", device.type == "cuda")
        and device.type == "cuda"
    )
    scaler = GradScaler(enabled=amp_enabled)
    loss_fn = nn.CrossEntropyLoss()

    get_artifact_dir()
    saver = best_model_saver(
        metric_name=training_cfg.get("monitor_metric", "validation_f1"),
        mode=training_cfg.get("monitor_mode", "max"),
    )
    saver.load_existing()

    start_epoch = 0
    global_step = 0

    if resume and training_state_exists("criteria"):
        logger.info("Resuming training from last checkpoint.")
        checkpoint = load_training_state(
            "criteria", model=model, optimizer=optimizer, scheduler=scheduler
        )
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)
        best_metric = checkpoint.get("best_metric")
        if best_metric is not None:
            saver.best_metric = best_metric
        rng_state = checkpoint.get("rng_state")
        if rng_state and "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if (
            rng_state
            and "cuda" in rng_state
            and rng_state["cuda"] is not None
            and torch.cuda.is_available()
        ):
            torch.cuda.set_rng_state_all(rng_state["cuda"])

    mlflow_cfg = deepcopy(config.get("mlflow", {}))
    if mlflow_cfg:
        configure_mlflow(
            tracking_uri=mlflow_cfg.get("tracking_uri"),
            experiment=mlflow_cfg.get("experiment"),
            tags=mlflow_cfg.get("tags"),
            artifact_location=mlflow_cfg.get("artifact_location"),
        )
        enable_autologging()

    flatten_config: dict[str, Any] = {}
    _flatten_dict(
        "", {k: v for k, v in config.items() if k != "mlflow"}, flatten_config
    )

    runtime_metrics: dict[str, Any] = {}
    start_time = time.time()

    with (
        mlflow_run(name=config.get("project", "criteria"))
        if mlflow_cfg
        else nullcontext()
    ):
        if mlflow_cfg:
            import mlflow

            mlflow.log_params(flatten_config)

        for epoch in range(start_epoch, epochs):
            logger.info("Epoch %d/%d", epoch + 1, epochs)
            model.train()
            optimizer.zero_grad(set_to_none=True)

            epoch_loss = 0.0
            epoch_preds: list[torch.Tensor] = []
            epoch_labels: list[torch.Tensor] = []

            for step, batch in enumerate(train_loader):
                batch = _move_batch_to_device(batch, device)

                with autocast(enabled=amp_enabled):
                    outputs = model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                    )
                    loss = loss_fn(outputs, batch["labels"])
                    loss = loss / grad_accum

                loss_scalar = loss.item()
                if amp_enabled:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                epoch_loss += loss_scalar * grad_accum
                epoch_preds.append(outputs.detach().cpu())
                epoch_labels.append(batch["labels"].detach().cpu())

                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    if amp_enabled:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), training_cfg.get("max_grad_norm", 1.0)
                    )
                    if amp_enabled:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                if (
                    mlflow_cfg
                    and global_step % training_cfg.get("logging_steps", 50) == 0
                ):
                    import mlflow

                    mlflow.log_metric(
                        "train_loss_step", loss_scalar * grad_accum, step=global_step
                    )

            train_metrics = _classification_metrics(
                torch.cat(epoch_preds), torch.cat(epoch_labels)
            )
            train_metrics["loss"] = epoch_loss / max(1, len(train_loader))

            val_metrics = _evaluate(model, val_loader, device, loss_fn, amp_enabled)
            prefixed_train = {f"train_{k}": v for k, v in train_metrics.items()}
            prefixed_val = {f"validation_{k}": v for k, v in val_metrics.items()}
