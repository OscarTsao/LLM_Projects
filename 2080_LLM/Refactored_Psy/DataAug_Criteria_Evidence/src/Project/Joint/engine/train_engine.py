from __future__ import annotations

import math
import time
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_scheduler

from Project.Joint.data.dataset import JointDataset
from Project.Joint.models.model import Model
from Project.Joint.utils import (
    best_model_saver,
    configure_mlflow,
    enable_autologging,
    get_artifact_dir,
    get_logger,
    load_best_model,
    load_training_state,
    mlflow_run,
    save_training_state,
    set_seed,
    training_state_exists,
)


def _flatten_dict(prefix: str, value: Any, accumulator: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, val in value.items():
            _flatten_dict(f"{prefix}.{key}" if prefix else key, val, accumulator)
    elif isinstance(value, (list, tuple)):
        accumulator[prefix] = ",".join(map(str, value))
    else:
        accumulator[prefix] = value


def _prepare_datasets(config: Dict[str, Any], seed: int) -> Tuple[Dataset, Dataset, Optional[Dataset]]:
    dataset_cfg = config.get("dataset", {})
    splits = dataset_cfg.get("splits", [0.8, 0.1, 0.1])
    if isinstance(splits, dict):
        splits = [splits.get("train", 0.8), splits.get("val", 0.1), splits.get("test", 0.1)]
    if len(splits) == 2:
        splits = [splits[0], splits[1], 0.0]
    if not math.isclose(sum(splits), 1.0, rel_tol=1e-3):
        raise ValueError("Dataset splits must sum to 1.0")

    criteria_tokenizer = AutoTokenizer.from_pretrained(dataset_cfg.get("criteria_tokenizer_name", config.get("model", {}).get("criteria_model", "bert-base-uncased")))
    evidence_tokenizer = AutoTokenizer.from_pretrained(dataset_cfg.get("evidence_tokenizer_name", config.get("model", {}).get("evidence_model", "bert-base-uncased")))

    dataset = JointDataset(
        csv_path=dataset_cfg.get("path"),
        criteria_tokenizer=criteria_tokenizer,
        evidence_tokenizer=evidence_tokenizer,
        sentence_column=dataset_cfg.get("sentence_column", "sentence_text"),
        context_column=dataset_cfg.get("context_column", "post_text"),
        label_column=dataset_cfg.get("label_column", "status"),
        criteria_max_length=dataset_cfg.get("criteria_max_length", 256),
        evidence_max_length=dataset_cfg.get("evidence_max_length", 512),
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
    test_dataset: Optional[Dataset],
    config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
    dataset_cfg = config.get("dataset", {})
    training_cfg = config.get("training", {})

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_cfg.get("train_batch_size", 8),
        shuffle=True,
        pin_memory=True,
        num_workers=dataset_cfg.get("num_workers", 0),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=training_cfg.get("eval_batch_size", 16),
        shuffle=False,
        pin_memory=True,
        num_workers=dataset_cfg.get("num_workers", 0),
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=training_cfg.get("eval_batch_size", 16),
            shuffle=False,
            pin_memory=True,
            num_workers=dataset_cfg.get("num_workers", 0),
        )

    return train_loader, val_loader, test_loader


def _build_model(config: Dict[str, Any]) -> Model:
    model_cfg = config.get("model", {})
    return Model(
        criteria_model_name=model_cfg.get("criteria_model", "bert-base-uncased"),
        evidence_model_name=model_cfg.get("evidence_model", "bert-base-uncased"),
        criteria_num_labels=model_cfg.get("criteria", {}).get("num_labels", 2),
        criteria_dropout=model_cfg.get("criteria", {}).get("dropout", 0.1),
        criteria_layer_num=model_cfg.get("criteria", {}).get("layer_num", 1),
        criteria_hidden_dims=model_cfg.get("criteria", {}).get("hidden_dims"),
        evidence_dropout=model_cfg.get("evidence", {}).get("dropout", 0.1),
        fusion_dropout=model_cfg.get("fusion", {}).get("dropout", 0.1),
    )


def _select_device(config: Dict[str, Any]) -> torch.device:
    preferred = config.get("training", {}).get("device")
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def _create_optimizer(model: nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    training_cfg = config.get("training", {})
    optimizer_cfg = training_cfg.get("optimizer", {}) or {}
    optimizer_name = optimizer_cfg.get("name", "adamw").lower()
    learning_rate = training_cfg.get("learning_rate", 2e-5)
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)
    no_decay = ["bias", "LayerNorm.weight"]

    grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if optimizer_name == "adamw":
        return torch.optim.AdamW(grouped_parameters, lr=learning_rate)
    if optimizer_name == "adam":
        return torch.optim.Adam(grouped_parameters, lr=learning_rate, weight_decay=weight_decay)
    if optimizer_name == "sgd":
        return torch.optim.SGD(grouped_parameters, lr=learning_rate, momentum=optimizer_cfg.get("momentum", 0.9), weight_decay=weight_decay)
    if optimizer_name == "adafactor":
        from transformers.optimization import Adafactor

        return Adafactor(model.parameters(), lr=learning_rate, weight_decay=weight_decay, scale_parameter=False, relative_step=False)
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'")


def _create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    num_training_steps: int,
) -> Optional[Any]:
    scheduler_cfg = config.get("training", {}).get("scheduler", {}) or {}
    scheduler_name = scheduler_cfg.get("name", "linear")
    if scheduler_name in (None, "", "none"):
        return None
    warmup_steps = scheduler_cfg.get("warmup_steps", 0)
    if scheduler_name == "cosine_with_restarts":
        return get_scheduler(
            scheduler_name,
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=scheduler_cfg.get("num_cycles", 1),
        )
    return get_scheduler(
        scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )


def _classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    preds = logits.argmax(dim=-1)
    correct = (preds == labels).sum().item()
    total = labels.numel()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

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


def _span_metrics(start_logits: torch.Tensor, end_logits: torch.Tensor, start_positions: torch.Tensor, end_positions: torch.Tensor) -> Dict[str, float]:
    start_pred = start_logits.argmax(dim=-1)
    end_pred = end_logits.argmax(dim=-1)
    start_acc = (start_pred == start_positions).float().mean().item()
    end_acc = (end_pred == end_positions).float().mean().item()
    exact_match = ((start_pred == start_positions) & (end_pred == end_positions)).float().mean().item()
    return {
        "start_accuracy": start_acc,
        "end_accuracy": end_acc,
        "span_em": exact_match,
    }


def _evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    cls_loss_fn: nn.Module,
    span_loss_fn: nn.Module,
    loss_weights: Dict[str, float],
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    cls_logits_all: List[torch.Tensor] = []
    cls_labels_all: List[torch.Tensor] = []
    span_start_logits_all: List[torch.Tensor] = []
    span_end_logits_all: List[torch.Tensor] = []
    span_start_labels_all: List[torch.Tensor] = []
    span_end_labels_all: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            criteria_inputs = {k.replace("criteria_", ""): v.to(device) for k, v in batch.items() if k.startswith("criteria_")}
            evidence_inputs = {k.replace("evidence_", ""): v.to(device) for k, v in batch.items() if k.startswith("evidence_")}
            labels = batch["labels"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            criteria_logits, start_logits, end_logits = model(
                **{f"criteria_{k}": v for k, v in criteria_inputs.items()},
                **{f"evidence_{k}": v for k, v in evidence_inputs.items()},
            )

            cls_loss = cls_loss_fn(criteria_logits, labels)
            span_loss = (span_loss_fn(start_logits, start_positions) + span_loss_fn(end_logits, end_positions)) / 2
            loss = loss_weights.get("criteria", 1.0) * cls_loss + loss_weights.get("evidence", 1.0) * span_loss
            losses.append(loss.item())

            cls_logits_all.append(criteria_logits.detach().cpu())
            cls_labels_all.append(labels.detach().cpu())
            span_start_logits_all.append(start_logits.detach().cpu())
            span_end_logits_all.append(end_logits.detach().cpu())
            span_start_labels_all.append(start_positions.detach().cpu())
            span_end_labels_all.append(end_positions.detach().cpu())

    cls_metrics = _classification_metrics(torch.cat(cls_logits_all), torch.cat(cls_labels_all))
    span_metrics = _span_metrics(torch.cat(span_start_logits_all), torch.cat(span_end_logits_all), torch.cat(span_start_labels_all), torch.cat(span_end_labels_all))
    joint_score = (cls_metrics["f1"] + span_metrics["span_em"]) / 2

    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "criteria_accuracy": cls_metrics["accuracy"],
        "criteria_precision": cls_metrics["precision"],
        "criteria_recall": cls_metrics["recall"],
        "criteria_f1": cls_metrics["f1"],
        "evidence_start_accuracy": span_metrics["start_accuracy"],
        "evidence_end_accuracy": span_metrics["end_accuracy"],
        "evidence_span_em": span_metrics["span_em"],
        "joint_score": joint_score,
    }
    return metrics


def train(
    config: Dict[str, Any],
    *,
    trial: Optional[optuna.Trial] = None,
    resume: bool = True,
) -> Dict[str, Any]:
    logger = get_logger(__name__)
    seed = set_seed(config.get("training", {}).get("seed", 42))
    logger.info("Using random seed %s", seed)

    train_dataset, val_dataset, test_dataset = _prepare_datasets(config, seed)
    train_loader, val_loader, test_loader = _create_dataloaders(train_dataset, val_dataset, test_dataset, config)

    model = _build_model(config)
    device = _select_device(config)
    model.to(device)

    optimizer = _create_optimizer(model, config)
    grad_accum = config.get("training", {}).get("gradient_accumulation", 1)
    epochs = config.get("training", {}).get("epochs", 3)
    total_update_steps = math.ceil(len(train_loader) / grad_accum) * epochs
    scheduler = _create_scheduler(optimizer, config, total_update_steps)

    cls_loss_fn = nn.CrossEntropyLoss()
    span_loss_fn = nn.CrossEntropyLoss()
    loss_weights = config.get("training", {}).get("loss_weights", {"criteria": 1.0, "evidence": 1.0})

    get_artifact_dir()
    saver = best_model_saver(metric_name=config.get("training", {}).get("monitor_metric", "validation_joint_score"), mode=config.get("training", {}).get("monitor_mode", "max"))
    saver.load_existing()

    start_epoch = 0
    global_step = 0

    if resume and training_state_exists("joint"):
        logger.info("Resuming training from last checkpoint.")
        checkpoint = load_training_state("joint", model=model, optimizer=optimizer, scheduler=scheduler)
        start_epoch = checkpoint.get("epoch", 0) + 1
        global_step = checkpoint.get("global_step", 0)
        best_metric = checkpoint.get("best_metric")
        if best_metric is not None:
            saver.best_metric = best_metric
        rng_state = checkpoint.get("rng_state")
        if rng_state and "torch" in rng_state:
            torch.set_rng_state(rng_state["torch"])
        if rng_state and "cuda" in rng_state and rng_state["cuda"] is not None and torch.cuda.is_available():
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

    flattened: Dict[str, Any] = {}
    _flatten_dict("", {k: v for k, v in config.items() if k != "mlflow"}, flattened)

    start_time = time.time()
    runtime_metrics: Dict[str, Any] = {}

    with mlflow_run(name=config.get("project", "joint")) if mlflow_cfg else nullcontext():
        if mlflow_cfg:
            import mlflow

            mlflow.log_params(flattened)

        for epoch in range(start_epoch, epochs):
            logger.info("Epoch %d/%d", epoch + 1, epochs)
            model.train()
            optimizer.zero_grad(set_to_none=True)

            epoch_cls_logits: List[torch.Tensor] = []
            epoch_cls_labels: List[torch.Tensor] = []
            epoch_start_logits: List[torch.Tensor] = []
            epoch_end_logits: List[torch.Tensor] = []
            epoch_start_labels: List[torch.Tensor] = []
            epoch_end_labels: List[torch.Tensor] = []
            epoch_losses: List[float] = []

            for step, batch in enumerate(train_loader):
                criteria_inputs = {k.replace("criteria_", ""): v.to(device) for k, v in batch.items() if k.startswith("criteria_")}
                evidence_inputs = {k.replace("evidence_", ""): v.to(device) for k, v in batch.items() if k.startswith("evidence_")}
                labels = batch["labels"].to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)

                criteria_logits, start_logits, end_logits = model(
                    **{f"criteria_{k}": v for k, v in criteria_inputs.items()},
                    **{f"evidence_{k}": v for k, v in evidence_inputs.items()},
                )

                cls_loss = cls_loss_fn(criteria_logits, labels)
                span_loss = (span_loss_fn(start_logits, start_positions) + span_loss_fn(end_logits, end_positions)) / 2
                loss = loss_weights.get("criteria", 1.0) * cls_loss + loss_weights.get("evidence", 1.0) * span_loss
                loss = loss / grad_accum
                loss.backward()

                epoch_losses.append(loss.item() * grad_accum)
                epoch_cls_logits.append(criteria_logits.detach().cpu())
                epoch_cls_labels.append(labels.detach().cpu())
                epoch_start_logits.append(start_logits.detach().cpu())
                epoch_end_logits.append(end_logits.detach().cpu())
                epoch_start_labels.append(start_positions.detach().cpu())
                epoch_end_labels.append(end_positions.detach().cpu())

                if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("training", {}).get("max_grad_norm", 1.0))
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                if mlflow_cfg and global_step % config.get("training", {}).get("logging_steps", 50) == 0:
                    import mlflow

                    mlflow.log_metric("train_loss_step", loss.item() * grad_accum, step=global_step)

            train_cls_metrics = _classification_metrics(torch.cat(epoch_cls_logits), torch.cat(epoch_cls_labels))
            train_span_metrics = _span_metrics(torch.cat(epoch_start_logits), torch.cat(epoch_end_logits), torch.cat(epoch_start_labels), torch.cat(epoch_end_labels))
            train_joint_score = (train_cls_metrics["f1"] + train_span_metrics["span_em"]) / 2
            train_metrics = {
                "loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                "criteria_accuracy": train_cls_metrics["accuracy"],
                "criteria_precision": train_cls_metrics["precision"],
                "criteria_recall": train_cls_metrics["recall"],
                "criteria_f1": train_cls_metrics["f1"],
                "evidence_start_accuracy": train_span_metrics["start_accuracy"],
                "evidence_end_accuracy": train_span_metrics["end_accuracy"],
                "evidence_span_em": train_span_metrics["span_em"],
                "joint_score": train_joint_score,
            }

            val_metrics = _evaluate(model, val_loader, device, cls_loss_fn, span_loss_fn, loss_weights)

            prefixed_train = {f"train_{k}": v for k, v in train_metrics.items()}
            prefixed_val = {f"validation_{k}": v for k, v in val_metrics.items()}
            logger.info("Train metrics: %s", prefixed_train)
            logger.info("Validation metrics: %s", prefixed_val)

            if mlflow_cfg:
                import mlflow

                mlflow.log_metrics(prefixed_train, step=epoch + 1)
                mlflow.log_metrics(prefixed_val, step=epoch + 1)

            monitor_metric = config.get("training", {}).get("monitor_metric", "validation_joint_score")
            metric_value = prefixed_val.get(monitor_metric, val_metrics.get("joint_score"))

            saver.update(
                metric_value=metric_value if metric_value is not None else 0.0,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )

            rng_state = {"torch": torch.get_rng_state()}
            if torch.cuda.is_available():
                rng_state["cuda"] = torch.cuda.get_rng_state_all()
            save_training_state(
                "joint",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                best_metric=saver.best_metric,
                best_metric_name=saver.metric_name,
                rng_state=rng_state,
            )

            if trial is not None:
                trial.report(metric_value, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        runtime_metrics = {"training_time": time.time() - start_time}

    metrics = {"best_metric": saver.best_metric, "best_metric_name": saver.metric_name, **runtime_metrics}

    if test_loader is not None:
        logger.info("Evaluating best checkpoint on test split.")
        best_model = _build_model(config)
        try:
            load_best_model(best_model)
            best_model.to(device)
            test_metrics = _evaluate(best_model, test_loader, device, cls_loss_fn, span_loss_fn, loss_weights)
            metrics.update({f"test_{k}": v for k, v in test_metrics.items()})
        except FileNotFoundError:
            logger.warning("No best checkpoint available; skipping test evaluation.")

    return metrics
