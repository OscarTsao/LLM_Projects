"""Training and evaluation helpers for the criteria baseline rebuild."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
try:
    from torch.amp import autocast as amp_autocast, GradScaler as AmpGradScaler
    _AMP_SUPPORTS_DEVICE_TYPE = True
except ImportError:  # pragma: no cover - fallback for older torch
    from torch.cuda.amp import autocast as amp_autocast, GradScaler as AmpGradScaler
    _AMP_SUPPORTS_DEVICE_TYPE = False
from contextlib import nullcontext
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_with_hard_restarts_schedule_with_warmup, get_linear_schedule_with_warmup
from omegaconf import DictConfig, OmegaConf

from .data import CriteriaDataset, DatasetSplit, assemble_dataset, create_cross_validation_splits
from .model import CriteriaClassifier, CriteriaModelConfig

# Disable tokenizer parallelism to avoid fork warnings on multi-worker dataloaders.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BEST_MODEL_FILENAME = "best_model.pt"
OPTIMIZER_FILENAME = "optimizer.pt"
SCHEDULER_FILENAME = "scheduler.pt"
SCALER_FILENAME = "scaler.pt"
STATE_LATEST_FILENAME = "state_latest.pt"
HISTORY_FILENAME = "train_history.json"
METRICS_FILENAME = "test_metrics.json"
RESOLVED_CONFIG_FILENAME = "resolved_config.yaml"


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module when wrapped by DataParallel."""
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def load_state_dict_flexible(model: torch.nn.Module, state_dict: Mapping[str, torch.Tensor]) -> None:
    """Load a state dict, stripping DataParallel prefixes if necessary."""
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        if any(key.startswith("module.") for key in state_dict):
            sanitized = {key.partition("module.")[2]: value for key, value in state_dict.items()}
            model.load_state_dict(sanitized)
        else:
            raise


def _log_artifact_safe(path: Path, artifact_path: Optional[str] = None) -> None:
    """Log an artifact to MLflow, tolerating filesystem permission quirks."""
    if not path or not path.exists():
        return
    try:
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except PermissionError as err:
        warnings.warn(f"Skipping MLflow artifact upload for {path}: {err}", RuntimeWarning)
    except OSError as err:
        warnings.warn(f"Skipping MLflow artifact upload for {path}: {err}", RuntimeWarning)

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    splits: DatasetSplit,
    tokenizer,
    max_length: int,
    batch_size: int,
    eval_batch_size: int,
    doc_stride: Optional[int] = None,
    num_workers: Optional[int] = None,
    cache_cfg: Optional[Mapping[str, Any]] = None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
    pin_memory_device: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch dataloaders for train/val/test."""
    cache_in_memory = bool(cache_cfg.get("cache_in_memory", False)) if cache_cfg else False
    cache_fraction = float(cache_cfg.get("cache_max_ram_fraction", 0.9)) if cache_cfg else 0.9
    cache_fraction = max(0.0, min(cache_fraction, 1.0))
    stride_value = max(0, int(doc_stride)) if doc_stride is not None else 0
    dataset_kwargs = {
        "cache_in_memory": cache_in_memory,
        "cache_max_ram_fraction": cache_fraction,
        "doc_stride": stride_value,
    }
    train_dataset = CriteriaDataset(splits.train, tokenizer=tokenizer, max_length=max_length, **dataset_kwargs)
    eval_dataset_kwargs = {**dataset_kwargs, "cache_in_memory": False}
    val_dataset = CriteriaDataset(splits.val, tokenizer=tokenizer, max_length=max_length, **eval_dataset_kwargs)
    test_dataset = CriteriaDataset(splits.test, tokenizer=tokenizer, max_length=max_length, **eval_dataset_kwargs)

    env_override = os.environ.get("CRITERIA_DATALOADER_WORKERS")
    if num_workers is not None:
        worker_candidates = max(0, int(num_workers))
    elif env_override is not None:
        try:
            worker_candidates = max(0, int(env_override))
        except ValueError:
            worker_candidates = 0
    else:
        cpu_count = os.cpu_count() or 1
        worker_candidates = max(0, cpu_count - 1)

    pin_memory = torch.cuda.is_available()
    resolved_pin_device: Optional[str]
    if pin_memory_device:
        resolved_pin_device = str(pin_memory_device)
    elif pin_memory:
        resolved_pin_device = "cuda"
    else:
        resolved_pin_device = None

    def _instantiate_loaders(worker_count: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        base_kwargs: Dict[str, Any] = {"num_workers": worker_count, "pin_memory": pin_memory}
        if resolved_pin_device:
            base_kwargs["pin_memory_device"] = resolved_pin_device
        if worker_count > 0:
            resolved_persistent: Optional[bool]
            if persistent_workers is None:
                resolved_persistent = True
            else:
                resolved_persistent = bool(persistent_workers)
            base_kwargs["persistent_workers"] = resolved_persistent
            resolved_prefetch: Optional[int]
            if prefetch_factor is None:
                resolved_prefetch = max(2, batch_size // 2) if batch_size > 1 else 2
            else:
                try:
                    resolved_prefetch = max(2, int(prefetch_factor))
                except (TypeError, ValueError):
                    resolved_prefetch = None
            if resolved_prefetch is not None:
                base_kwargs["prefetch_factor"] = resolved_prefetch

        def _build_loader(dataset, loader_batch_size, shuffle_flag):
            loader_kwargs = dict(base_kwargs)
            try:
                return DataLoader(
                    dataset,
                    batch_size=loader_batch_size,
                    shuffle=shuffle_flag,
                    **loader_kwargs,
                )
            except TypeError as err:
                if "pin_memory_device" in str(err):
                    loader_kwargs.pop("pin_memory_device", None)
                    return DataLoader(
                        dataset,
                        batch_size=loader_batch_size,
                        shuffle=shuffle_flag,
                        **loader_kwargs,
                    )
                raise

        train_loader_local = _build_loader(train_dataset, batch_size, True)
        val_loader_local = _build_loader(val_dataset, eval_batch_size, False)
        test_loader_local = _build_loader(test_dataset, eval_batch_size, False)
        return train_loader_local, val_loader_local, test_loader_local

    try:
        return _instantiate_loaders(worker_candidates)
    except (RuntimeError, OSError, PermissionError) as exc:
        if worker_candidates <= 0:
            raise
        warnings.warn(
            f"Falling back to num_workers=0 for dataloaders due to worker initialization failure: {exc}",
            RuntimeWarning,
        )
        return _instantiate_loaders(0)


def create_model(model_cfg: Mapping[str, Any]) -> CriteriaClassifier:
    """Instantiate the criteria classifier from raw config."""
    config = CriteriaModelConfig(
        model_name=model_cfg["pretrained_model_name"],
        hidden_sizes=list(model_cfg.get("classifier_hidden_sizes", [])),
        dropout=float(model_cfg.get("dropout", 0.1)),
        pooling=str(model_cfg.get("pooling", "cls")),
        activation=str(model_cfg.get("activation", "gelu")),
        loss_type=str(model_cfg.get("loss_type", "adaptive_focal")),
        alpha=float(model_cfg.get("alpha", 0.25)),
        gamma=float(model_cfg.get("gamma", 2.0)),
        delta=float(model_cfg.get("delta", 1.0)),
        use_gradient_checkpointing=bool(model_cfg.get("use_gradient_checkpointing", True)),
        base_model_dropout=(
            float(model_cfg["base_model_dropout"]) if model_cfg.get("base_model_dropout") is not None else None
        ),
        base_model_attention_dropout=(
            float(model_cfg["base_model_attention_dropout"])
            if model_cfg.get("base_model_attention_dropout") is not None
            else None
        ),
    )
    return CriteriaClassifier(config)


def _iter_encoder_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Return the list of transformer layers for the underlying encoder."""
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return []
    if hasattr(encoder, "encoder") and hasattr(encoder.encoder, "layer"):
        return list(encoder.encoder.layer)
    if hasattr(encoder, "layer"):
        return list(encoder.layer)
    return []


def _freeze_encoder_layers(model: torch.nn.Module, num_layers: int) -> int:
    """Freeze the first `num_layers` transformer blocks of the encoder."""
    if num_layers <= 0:
        return 0
    frozen_params = 0
    for idx, layer in enumerate(_iter_encoder_layers(model)):
        if idx >= num_layers:
            break
        for param in layer.parameters():
            if param.requires_grad:
                param.requires_grad = False
                frozen_params += param.numel()
    return frozen_params


_LAYER_PATTERN = re.compile(r"encoder\.(?:encoder\.)?layer\.(\d+)")


def _infer_layer_id(param_name: str, num_hidden_layers: int) -> int:
    """Map a parameter name to a layer index for layer-wise LR decay scheduling."""
    match = _LAYER_PATTERN.search(param_name)
    if match:
        return int(match.group(1)) + 1  # shift so embeddings occupy slot 0
    if "embeddings" in param_name:
        return 0
    if param_name.startswith("classifier") or "classifier." in param_name:
        return num_hidden_layers + 2
    return num_hidden_layers + 1


def create_optimizer(
    model: torch.nn.Module,
    training_cfg: Mapping[str, Any],
    steps_per_epoch: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Set up AdamW optimizer and the configured scheduler."""
    lr = float(training_cfg["learning_rate"])
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    adam_eps = float(training_cfg.get("adam_eps", 1e-8))
    beta1 = float(training_cfg.get("adam_beta1", 0.9))
    beta2 = float(training_cfg.get("adam_beta2", 0.999))
    num_epochs = int(training_cfg["num_epochs"])
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.1))
    scheduler_type = str(training_cfg.get("scheduler_type", "linear")).lower()

    layerwise_decay_raw = training_cfg.get("layerwise_lr_decay")
    try:
        layerwise_decay = float(layerwise_decay_raw) if layerwise_decay_raw is not None else None
    except (TypeError, ValueError):
        layerwise_decay = None

    encoder_config = getattr(getattr(model, "encoder", None), "config", None)
    num_hidden_layers = getattr(encoder_config, "num_hidden_layers", None)

    named_params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not named_params:
        raise ValueError("No trainable parameters found for optimizer setup.")

    use_layerwise = (
        layerwise_decay is not None
        and 0.0 < layerwise_decay < 1.0
        and num_hidden_layers is not None
        and named_params
    )

    if use_layerwise:
        groups: Dict[int, Dict[str, Any]] = {}
        for name, param in named_params:
            layer_id = _infer_layer_id(name, int(num_hidden_layers))
            if layer_id <= num_hidden_layers:
                power = int(num_hidden_layers) - layer_id
                if layer_id == 0:
                    power = int(num_hidden_layers)
            else:
                power = 0
            scale = layerwise_decay ** max(power, 0)
            group = groups.setdefault(layer_id, {"params": [], "lr": lr * scale})
            group["params"].append(param)
        param_groups = [
            {"params": payload["params"], "lr": payload["lr"], "weight_decay": weight_decay}
            for _, payload in sorted(groups.items(), key=lambda item: item[0])
            if payload["params"]
        ]
    else:
        param_groups = [
            {
                "params": [param for _, param in named_params],
                "lr": lr,
                "weight_decay": weight_decay,
            }
        ]

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
        eps=adam_eps,
        betas=(beta1, beta2),
    )
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = max(0, int(round(warmup_ratio * total_steps)))
    if scheduler_type in {"cosine_with_restarts", "cosine_restarts", "cosine"}:
        num_cycles_raw = training_cfg.get("scheduler_num_cycles", 1.0)
        try:
            num_cycles = float(num_cycles_raw)
        except (TypeError, ValueError):
            num_cycles = 1.0
        num_cycles = max(num_cycles, 1.0)
        scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=num_cycles,
        )
    else:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    return optimizer, scheduler


def _flatten_for_mlflow(data: Mapping[str, Any], prefix: str = "", sep: str = ".") -> Dict[str, str]:
    """Flatten a nested mapping into stringified MLflow parameters."""
    flattened: Dict[str, str] = {}
    for key, value in data.items():
        if key == "hydra":
            continue
        name = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_for_mlflow(value, name, sep=sep))
        elif isinstance(value, (list, tuple)):
            flattened[name] = ",".join(str(v) for v in value)
        else:
            flattened[name] = str(value)
    return flattened


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
    """Compute binary classification metrics."""
    probs = torch.sigmoid(logits).cpu().numpy()
    preds = (probs >= 0.5).astype(int)
    labels_np = labels.cpu().numpy()

    accuracy = accuracy_score(labels_np, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels_np, preds, average="binary", zero_division=0)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels_np, preds, average="macro", zero_division=0
    )
    try:
        auc = roc_auc_score(labels_np, probs)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "auc": float(auc),
    }


@torch.no_grad()
def evaluate(
    model: CriteriaClassifier,
    loader: DataLoader,
    device: torch.device,
    non_blocking: Optional[bool] = None,
) -> Dict[str, float]:
    """Evaluate model on a dataloader."""
    model.eval()
    grouped_logits: Dict[str, List[float]] = defaultdict(list)
    grouped_labels: Dict[str, float] = {}
    chunk_counts: Dict[str, int] = defaultdict(int)
    unnamed_counter = 0
    autocast_available = device.type == "cuda"
    transfer_non_blocking = bool(non_blocking) if non_blocking is not None else bool(getattr(loader, "pin_memory", False))
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=transfer_non_blocking)
        attention_mask = batch["attention_mask"].to(device, non_blocking=transfer_non_blocking)
        labels = batch["labels"].to(device, non_blocking=transfer_non_blocking)
        context_manager = (
            amp_autocast("cuda", enabled=True)
            if (_AMP_SUPPORTS_DEVICE_TYPE and autocast_available)
            else amp_autocast(enabled=True)
            if autocast_available
            else nullcontext()
        )
        try:
            with context_manager:
                logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
        except RuntimeError as err:
            if autocast_available and "value cannot be converted to type" in str(err).lower():
                warnings.warn(
                    "Encountered AMP overflow during evaluation; rerunning this batch in FP32.",
                    RuntimeWarning,
                )
                with nullcontext():
                    logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                autocast_available = False
            else:
                raise
        batch_group_ids = batch.get("group_id")
        if batch_group_ids is None:
            resolved_group_ids = [None] * logits.size(0)
        elif isinstance(batch_group_ids, torch.Tensor):
            resolved_group_ids = [str(v) for v in batch_group_ids.view(-1).cpu().tolist()]
        elif isinstance(batch_group_ids, (list, tuple)):
            resolved_group_ids = [str(v) for v in batch_group_ids]
        else:
            resolved_group_ids = [str(batch_group_ids)] * logits.size(0)

        logit_values = logits.detach().cpu().view(-1).tolist()
        label_values = labels.detach().cpu().view(-1).tolist()
        for idx, logit_value in enumerate(logit_values):
            raw_group = resolved_group_ids[idx] if idx < len(resolved_group_ids) else None
            if raw_group is None or raw_group == "None":
                group_id = f"_sample_{unnamed_counter}"
                unnamed_counter += 1
            else:
                group_id = raw_group
            grouped_logits[group_id].append(float(logit_value))
            if group_id not in grouped_labels:
                grouped_labels[group_id] = float(label_values[idx])
            chunk_counts[group_id] += 1

    if not grouped_logits:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": float("nan")}

    aggregated_logits: List[float] = []
    aggregated_labels: List[float] = []
    for group_id, logit_list in grouped_logits.items():
        if logit_list:
            aggregated_logits.append(max(logit_list))
        else:
            aggregated_logits.append(0.0)
        aggregated_labels.append(grouped_labels.get(group_id, 0.0))

    logits_tensor = torch.tensor(aggregated_logits, dtype=torch.float32)
    labels_tensor = torch.tensor(aggregated_labels, dtype=torch.float32)
    metrics = compute_metrics(logits_tensor, labels_tensor)
    chunk_array = np.array(list(chunk_counts.values()), dtype=float)
    if chunk_array.size > 0:
        metrics["chunks_per_group_mean"] = float(np.mean(chunk_array))
        metrics["chunks_per_group_std"] = float(np.std(chunk_array))
        metrics["chunks_per_group_max"] = float(np.max(chunk_array))
    return metrics




def _compute_metric_summary(metric_dicts: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Return mean/std aggregates for a list of metric dictionaries."""
    aggregated: Dict[str, Dict[str, float]] = {"mean": {}, "std": {}}
    if not metric_dicts:
        return aggregated

    metric_names = set()
    for metrics in metric_dicts:
        metric_names.update(metrics.keys())

    if not metric_names:
        return aggregated

    for name in sorted(metric_names):
        values: List[float] = []
        for metrics in metric_dicts:
            raw_value = metrics.get(name)
            try:
                values.append(float(raw_value))
            except (TypeError, ValueError):
                values.append(float("nan"))
        arr = np.array(values, dtype=float)
        aggregated["mean"][name] = float(np.nanmean(arr))
        aggregated["std"][name] = float(np.nanstd(arr))
    return aggregated


def _aggregate_cross_validation(fold_summaries: List[Dict[str, object]]) -> Dict[str, object]:
    """Compute aggregate statistics across fold summaries."""
    val_payloads: List[Dict[str, float]] = []
    test_payloads: List[Dict[str, float]] = []
    best_epochs: List[float] = []

    for summary in fold_summaries:
        val_metrics = summary.get("val_metrics")
        if isinstance(val_metrics, Mapping):
            val_payloads.append({k: float(v) for k, v in val_metrics.items()})
        test_metrics = summary.get("test_metrics")
        if isinstance(test_metrics, Mapping):
            test_payloads.append({k: float(v) for k, v in test_metrics.items()})
        best_epoch_val = summary.get("best_epoch")
        if isinstance(best_epoch_val, (int, float)):
            best_epochs.append(float(best_epoch_val))

    val_summary = _compute_metric_summary(val_payloads)
    test_summary = _compute_metric_summary(test_payloads)

    aggregate: Dict[str, object] = {
        "val_metrics_mean": val_summary["mean"],
        "val_metrics_std": val_summary["std"],
        "test_metrics_mean": test_summary["mean"],
        "test_metrics_std": test_summary["std"],
    }

    if best_epochs:
        arr = np.array(best_epochs, dtype=float)
        aggregate["best_epoch_mean"] = float(np.mean(arr))
        aggregate["best_epoch_std"] = float(np.std(arr))
    else:
        aggregate["best_epoch_mean"] = float("nan")
        aggregate["best_epoch_std"] = float("nan")

    return aggregate


def _train_single(
    cfg: DictConfig,
    output_dir: Path,
    splits: Optional[DatasetSplit] = None,
    tokenizer: Optional[AutoTokenizer] = None,
    fold_index: Optional[int] = None,
) -> Dict[str, object]:
    """Hydra-managed training loop for a single split."""
    output_dir.mkdir(parents=True, exist_ok=True)

    history_path = output_dir / HISTORY_FILENAME
    metrics_path = output_dir / METRICS_FILENAME
    resolved_config_path = output_dir / RESOLVED_CONFIG_FILENAME
    best_model_path = output_dir / BEST_MODEL_FILENAME
    optimizer_path = output_dir / OPTIMIZER_FILENAME
    scheduler_path = output_dir / SCHEDULER_FILENAME
    scaler_artifact_path = output_dir / SCALER_FILENAME
    state_path = output_dir / STATE_LATEST_FILENAME

    history: List[Dict[str, object]] = []
    if history_path.exists():
        try:
            loaded_history = json.loads(history_path.read_text())
            if isinstance(loaded_history, list):
                history = loaded_history
        except json.JSONDecodeError:
            history = []

    base_seed = int(cfg.get("seed", 42))
    effective_seed = base_seed if fold_index is None else base_seed + max(fold_index - 1, 0)
    set_seed(effective_seed)

    dataset_cfg = cfg.dataset
    model_cfg = cfg.model
    training_cfg = cfg.training
    amp_requested = bool(training_cfg.get("use_amp", True))
    logging_cfg = cfg.logging
    mlflow_cfg = cfg.mlflow
    dataset_cache_in_memory = bool(dataset_cfg.get("cache_in_memory", False))
    cache_fraction_raw = dataset_cfg.get("cache_max_ram_fraction", 0.9)
    try:
        dataset_cache_fraction = float(cache_fraction_raw)
    except (TypeError, ValueError):
        dataset_cache_fraction = 0.9
    dataset_cache_fraction = max(0.0, min(dataset_cache_fraction, 1.0))

    use_cuda_available = torch.cuda.is_available()
    enable_tf32_cfg = bool(training_cfg.get("enable_tf32", True)) if use_cuda_available else False
    cudnn_benchmark_cfg = training_cfg.get("cudnn_benchmark")
    matmul_precision_cfg = training_cfg.get("float32_matmul_precision")
    if matmul_precision_cfg is None:
        matmul_precision_cfg = "high" if use_cuda_available else "medium"
    applied_tf32 = False
    applied_cudnn_benchmark = False
    applied_matmul_precision: Optional[str] = None
    if use_cuda_available:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = enable_tf32_cfg
            applied_tf32 = enable_tf32_cfg
        if hasattr(torch.backends, "cudnn"):
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = enable_tf32_cfg
            if cudnn_benchmark_cfg is None:
                torch.backends.cudnn.benchmark = True
                applied_cudnn_benchmark = True
            else:
                torch.backends.cudnn.benchmark = bool(cudnn_benchmark_cfg)
                applied_cudnn_benchmark = bool(cudnn_benchmark_cfg)
    else:
        applied_tf32 = False
        applied_cudnn_benchmark = False
    try:
        torch.set_float32_matmul_precision(str(matmul_precision_cfg))
        if hasattr(torch, "get_float32_matmul_precision"):
            applied_matmul_precision = torch.get_float32_matmul_precision()
        else:
            applied_matmul_precision = str(matmul_precision_cfg)
    except (TypeError, ValueError):
        applied_matmul_precision = None
        warnings.warn(
            f"Invalid float32_matmul_precision={matmul_precision_cfg!r}; using existing precision setting.",
            RuntimeWarning,
        )

    computed_alpha: Optional[float] = None
    alpha_strategy = str(model_cfg.get("alpha_strategy", "")).lower()
    if alpha_strategy == "effective_num" and splits is not None:
        beta_raw = model_cfg.get("alpha_beta", 0.999)
        try:
            beta = float(beta_raw)
        except (TypeError, ValueError):
            beta = 0.999
        beta = min(max(beta, 1e-6), 1 - 1e-9)
        class_counts = splits.train["label"].value_counts().to_dict()
        pos_count = int(class_counts.get(1, 0))
        neg_count = int(class_counts.get(0, 0))
        if pos_count > 0 and neg_count > 0:
            weights = []
            for count in (neg_count, pos_count):
                if count <= 0:
                    weights.append(0.0)
                    continue
                denom = 1.0 - (beta ** count)
                if denom <= 0:
                    weights.append(0.0)
                else:
                    weights.append((1.0 - beta) / denom)
            total_weight = sum(weights)
            if total_weight > 0 and weights[1] > 0:
                computed_alpha = weights[1] / total_weight
                model_cfg.alpha = float(computed_alpha)
        if computed_alpha is None:
            model_cfg.alpha = float(model_cfg.get("alpha", 0.5))

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["pretrained_model_name"])
    if splits is None:
        raise ValueError("Cross-validation splits must be provided for training.")

    auto_resume = bool(training_cfg.get("auto_resume", True))
    resume_cfg = training_cfg.get("resume_checkpoint")
    resume_path: Optional[Path] = None
    if resume_cfg:
        resume_path = Path(str(resume_cfg)).expanduser()
        if not resume_path.is_absolute():
            resume_path = (Path.cwd() / resume_path).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
    elif auto_resume and state_path.exists():
        resume_path = state_path
    else:
        history = []
        for stale_path in (
            history_path,
            state_path,
            scaler_artifact_path,
            best_model_path,
            optimizer_path,
            scheduler_path,
            metrics_path,
        ):
            if stale_path.exists():
                try:
                    stale_path.unlink()
                except OSError:
                    pass

    pin_memory_device_cfg = training_cfg.get("pin_memory_device")
    resolved_pin_memory_device: Optional[str]
    if pin_memory_device_cfg is None:
        resolved_pin_memory_device = None
    else:
        pin_device_str = str(pin_memory_device_cfg).strip()
        resolved_pin_memory_device = pin_device_str or None

    train_loader, val_loader, test_loader = build_dataloaders(
        splits,
        tokenizer,
        max_length=int(model_cfg["max_seq_length"]),
        batch_size=int(training_cfg["batch_size"]),
        eval_batch_size=int(training_cfg.get("eval_batch_size", training_cfg["batch_size"])),
        doc_stride=model_cfg.get("doc_stride"),
        num_workers=training_cfg.get("num_workers"),
        cache_cfg=dataset_cfg,
        prefetch_factor=training_cfg.get("prefetch_factor"),
        persistent_workers=training_cfg.get("persistent_workers"),
        pin_memory_device=resolved_pin_memory_device,
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    non_blocking_cfg_raw = training_cfg.get("non_blocking_transfers", True)
    non_blocking_transfers = bool(non_blocking_cfg_raw)
    train_non_blocking = non_blocking_transfers and getattr(train_loader, "pin_memory", False)
    val_non_blocking = non_blocking_transfers and getattr(val_loader, "pin_memory", False)
    test_non_blocking = non_blocking_transfers and getattr(test_loader, "pin_memory", False)
    model = create_model(model_cfg).to(device)
    multi_gpu = use_cuda and torch.cuda.device_count() > 1
    compile_requested = bool(training_cfg.get("torch_compile", False))
    compiled_model = False
    torch_compile_mode = str(training_cfg.get("torch_compile_mode", "default"))
    torch_compile_backend = training_cfg.get("torch_compile_backend")
    torch_compile_dynamic = training_cfg.get("torch_compile_dynamic")
    torch_compile_fullgraph = training_cfg.get("torch_compile_fullgraph")
    if compile_requested and not hasattr(torch, "compile"):
        warnings.warn("torch.compile requested but not available in this version of PyTorch; skipping.", RuntimeWarning)
        compile_requested = False
    if compile_requested and multi_gpu:
        warnings.warn("torch.compile is skipped because DataParallel is active for multi-GPU training.", RuntimeWarning)
        compile_requested = False
    if compile_requested:
        compile_kwargs: Dict[str, Any] = {"mode": torch_compile_mode}
        if torch_compile_backend:
            compile_kwargs["backend"] = str(torch_compile_backend)
        if torch_compile_dynamic is not None:
            compile_kwargs["dynamic"] = bool(torch_compile_dynamic)
        if torch_compile_fullgraph is not None:
            compile_kwargs["fullgraph"] = bool(torch_compile_fullgraph)
        try:
            model = torch.compile(model, **compile_kwargs)
            compiled_model = True
        except Exception as err:  # pragma: no cover - depend on torch.compile runtime support
            warnings.warn(f"torch.compile failed ({err}); continuing without compilation.", RuntimeWarning)
            compiled_model = False
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    model_to_optimize = unwrap_model(model)

    freeze_layers = int(training_cfg.get("freeze_encoder_layers", 0))
    if freeze_layers > 0:
        _freeze_encoder_layers(model_to_optimize, freeze_layers)

    total_epochs = int(training_cfg["num_epochs"])
    grad_accum = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
    num_train_batches = len(train_loader)
    steps_per_epoch = max(1, math.ceil(num_train_batches / grad_accum))
    optimizer, scheduler = create_optimizer(model_to_optimize, training_cfg, steps_per_epoch)
    amp_enabled = use_cuda and amp_requested
    scaler: Optional[AmpGradScaler]
    if amp_enabled:
        if _AMP_SUPPORTS_DEVICE_TYPE:
            scaler = AmpGradScaler("cuda", enabled=True)
        else:
            scaler = AmpGradScaler(enabled=True)
    else:
        scaler = None

    best_metric = -float("inf")
    best_model_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_metrics: Dict[str, float] = {}
    patience = int(training_cfg.get("early_stopping_patience", 10))
    best_epoch = 0
    max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
    metric_key = str(training_cfg.get("metric_for_best", "f1"))
    save_every = int(logging_cfg.get("save_every_epochs", 0))
    start_epoch = 1
    resumed = False
    resume_prev_epoch = 0
    resume_start_epoch = start_epoch

    if resume_path is not None:
        resume_payload = torch.load(resume_path, map_location=device)
        resumed = True
        load_state_dict_flexible(model_to_optimize, resume_payload["model"])
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        resume_prev_epoch = int(resume_payload.get("epoch", 0))
        start_epoch = max(1, resume_prev_epoch + 1)
        saved_amp = bool(resume_payload.get("amp_enabled", amp_enabled))
        if saved_amp and use_cuda:
            amp_enabled = True
            if scaler is None:
                if _AMP_SUPPORTS_DEVICE_TYPE:
                    scaler = AmpGradScaler("cuda", enabled=True)
                else:
                    scaler = AmpGradScaler(enabled=True)
            scaler_state = resume_payload.get("scaler")
            if scaler is not None and scaler_state is not None:
                scaler.load_state_dict(scaler_state)
        else:
            amp_enabled = False
            scaler = None
        best_metric = float(resume_payload.get("best_metric", best_metric))
        best_epoch = int(resume_payload.get("best_epoch", best_epoch))
        best_val_metrics_payload = resume_payload.get("best_val_metrics", best_val_metrics)
        if isinstance(best_val_metrics_payload, dict):
            best_val_metrics = {k: float(v) for k, v in best_val_metrics_payload.items()}
        resume_start_epoch = start_epoch
        if best_model_path.exists():
            best_model_state = torch.load(best_model_path, map_location=device)
        else:
            best_model_state = copy.deepcopy(model_to_optimize.state_dict())
        if history_path.exists():
            try:
                loaded_history = json.loads(history_path.read_text())
                if isinstance(loaded_history, list):
                    history = loaded_history
            except json.JSONDecodeError:
                pass

    mlflow_db_path = Path(str(mlflow_cfg.get("tracking_db", "mlflow.db"))).expanduser()
    if not mlflow_db_path.is_absolute():
        mlflow_db_path = Path.cwd() / mlflow_db_path
    mlflow_db_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_dir = Path(str(mlflow_cfg.get("artifact_store", "mlruns"))).expanduser()
    if not artifact_dir.is_absolute():
        artifact_dir = Path.cwd() / artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    tracking_uri = f"sqlite:///{mlflow_db_path.resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name=str(mlflow_cfg.get("experiment_name", "default")))
    base_run_name = mlflow_cfg.get("run_name") or f"train_{datetime.now():%Y%m%d_%H%M%S}"
    run_name = base_run_name if fold_index is None else f"{base_run_name}_fold{fold_index}"
    tags = {str(k): str(v) for k, v in mlflow_cfg.get("tags", {}).items()}
    if fold_index is not None:
        tags.setdefault("fold_index", str(fold_index))

    config_container = OmegaConf.to_container(cfg, resolve=True)
    params_to_log = _flatten_for_mlflow(config_container)

    run_id: Optional[str] = None

    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        if params_to_log:
            mlflow.log_params(params_to_log)
        active = mlflow.active_run()
        if active is not None:
            run_id = active.info.run_id
        mlflow.log_param("fold_index", str(fold_index) if fold_index is not None else "")
        mlflow.log_param("seed_effective", str(effective_seed))
        mlflow.log_param("amp_initial_enabled", str(amp_enabled))
        mlflow.log_param("resumed", str(resumed))
        mlflow.log_param("resume_checkpoint_path", str(resume_path) if resume_path else "")
        mlflow.log_param("resume_previous_epoch", str(resume_prev_epoch))
        mlflow.log_param("resume_start_epoch", str(start_epoch))
        mlflow.log_param("multi_gpu", str(multi_gpu))
        mlflow.log_param("cuda_device_count", str(torch.cuda.device_count() if use_cuda else 0))
        mlflow.log_param("dataset_cache_in_memory", str(dataset_cache_in_memory))
        mlflow.log_param("dataset_cache_fraction", f"{dataset_cache_fraction:.3f}")
        mlflow.log_param("dataloader_num_workers", str(getattr(train_loader, "num_workers", 0)))
        mlflow.log_param("enable_tf32_effective", str(applied_tf32))
        mlflow.log_param("cudnn_benchmark_effective", str(applied_cudnn_benchmark))
        if applied_matmul_precision is not None:
            mlflow.log_param("float32_matmul_precision_effective", str(applied_matmul_precision))
        mlflow.log_param("non_blocking_transfers", str(non_blocking_transfers))
        mlflow.log_param("torch_compile_requested", str(bool(training_cfg.get("torch_compile", False))))
        mlflow.log_param("torch_compile_active", str(compiled_model))
        mlflow.log_param("torch_compile_mode", torch_compile_mode)
        if torch_compile_backend:
            mlflow.log_param("torch_compile_backend", str(torch_compile_backend))
        if torch_compile_dynamic is not None:
            mlflow.log_param("torch_compile_dynamic", str(bool(torch_compile_dynamic)))
        if torch_compile_fullgraph is not None:
            mlflow.log_param("torch_compile_fullgraph", str(bool(torch_compile_fullgraph)))
        if resolved_pin_memory_device:
            mlflow.log_param("pin_memory_device", str(resolved_pin_memory_device))

        for epoch in range(start_epoch, total_epochs + 1):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader, start=1):
                input_ids = batch["input_ids"].to(device, non_blocking=train_non_blocking)
                attention_mask = batch["attention_mask"].to(device, non_blocking=train_non_blocking)
                labels = batch["labels"].to(device, non_blocking=train_non_blocking)

                context_manager = (
                    amp_autocast("cuda", enabled=True) if (_AMP_SUPPORTS_DEVICE_TYPE and amp_enabled)
                    else amp_autocast(enabled=True) if (amp_enabled)
                    else nullcontext()
                )
                try:
                    with context_manager:
                        logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = loss / grad_accum
                except RuntimeError as err:
                    if amp_enabled and "value cannot be converted to type" in str(err).lower():
                        amp_enabled = False
                        scaler = None
                        torch.cuda.empty_cache()
                        with nullcontext():
                            logits, loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                            loss = loss / grad_accum
                    else:
                        raise

                if amp_enabled and scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                running_loss += loss.item()

                if step % grad_accum == 0:
                    if amp_enabled and scaler is not None:
                        scaler.unscale_(optimizer)
                        clip_grad_norm_(model_to_optimize.parameters(), max_grad_norm)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        clip_grad_norm_(model_to_optimize.parameters(), max_grad_norm)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

            val_metrics = evaluate(model, val_loader, device, non_blocking=val_non_blocking)
            train_loss = running_loss * grad_accum / max(1, len(train_loader))
            train_metrics = {"loss": train_loss}

            current_lr = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else optimizer.param_groups[0]["lr"]
            mlflow.log_metric("train/loss", float(train_loss), step=epoch)
            mlflow.log_metric("train/lr", float(current_lr), step=epoch)
            mlflow.log_metrics({f"val/{k}": float(v) for k, v in val_metrics.items()}, step=epoch)

            record = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
            history.append(record)

            monitored_metric = float(val_metrics.get(metric_key, float("nan")))
            if math.isfinite(monitored_metric) and monitored_metric > best_metric:
                best_metric = monitored_metric
                best_epoch = epoch
                best_val_metrics = {k: float(v) for k, v in val_metrics.items()}
                best_model_state = copy.deepcopy(model_to_optimize.state_dict())
                torch.save(best_model_state, best_model_path)

            if save_every > 0 and epoch % save_every == 0:
                torch.save(model_to_optimize.state_dict(), output_dir / f"model_epoch_{epoch}.pt")

            state_snapshot = {
                "model": model_to_optimize.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict() if amp_enabled and scaler is not None else None,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "best_val_metrics": best_val_metrics,
                "amp_enabled": amp_enabled,
                "multi_gpu": multi_gpu,
                "epoch": epoch,
            }
            torch.save(state_snapshot, state_path)
            history_path.write_text(json.dumps(history, indent=2))

            if epoch - best_epoch >= patience:
                break

        if best_model_state is None and best_model_path.exists():
            best_model_state = torch.load(best_model_path, map_location=device)
        if best_model_state is not None:
            load_state_dict_flexible(model_to_optimize, best_model_state)

        test_metrics = evaluate(model, test_loader, device, non_blocking=test_non_blocking)
        mlflow.log_metric("best_epoch", float(best_epoch))
        mlflow.log_metrics({f"test/{k}": float(v) for k, v in test_metrics.items()}, step=best_epoch or 1)
        mlflow.log_param("amp_final_enabled", str(amp_enabled and scaler is not None))

        history_path.write_text(json.dumps(history, indent=2))
        metrics_payload = {
            "val": best_val_metrics,
            "test": test_metrics,
        }
        metrics_path.write_text(json.dumps(metrics_payload, indent=2))

        resolved_config_path.write_text(OmegaConf.to_yaml(cfg))

        _log_artifact_safe(history_path)
        _log_artifact_safe(metrics_path)
        if best_model_path.exists():
            _log_artifact_safe(best_model_path)
        if scaler_artifact_path.exists():
            _log_artifact_safe(scaler_artifact_path)
        if resolved_config_path.exists():
            _log_artifact_safe(resolved_config_path)

    return {
        "history": history,
        "best_epoch": best_epoch,
        "val_metrics": best_val_metrics,
        "test_metrics": test_metrics,
        "tokenizer_name": model_cfg["pretrained_model_name"],
        "mlflow_experiment_id": experiment.experiment_id,
        "mlflow_run_id": run_id,
        "fold_index": fold_index,
        "effective_seed": effective_seed,
        "output_dir": str(output_dir.resolve()),
    }


def train(cfg: DictConfig) -> Dict[str, object]:
    """Hydra-managed training entry point that always runs cross-validation."""
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model

    output_base = Path(str(cfg.paths.output_dir)).expanduser()
    output_base.mkdir(parents=True, exist_ok=True)

    assembled = assemble_dataset(dataset_cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["pretrained_model_name"])

    folds = create_cross_validation_splits(assembled, dataset_cfg)
    fold_summaries: List[Dict[str, object]] = []
    for fold_index, split in enumerate(folds, start=1):
        fold_dir = output_base / f"fold_{fold_index}"
        summary = _train_single(cfg, fold_dir, splits=split, tokenizer=tokenizer, fold_index=fold_index)
        fold_summaries.append(summary)

    aggregate = _aggregate_cross_validation(fold_summaries)
    aggregate.update(
        {
            "mode": "cross_validation",
            "fold_summaries": fold_summaries,
            "tokenizer_name": model_cfg["pretrained_model_name"],
            "num_folds": len(folds),
        }
    )
    return aggregate


@torch.no_grad()
def evaluate_checkpoint(cfg: DictConfig, checkpoint: Optional[Path] = None) -> Dict[str, float]:
    """Load a saved fold checkpoint and score it on the corresponding validation split."""
    dataset_cfg = cfg.dataset
    model_cfg = cfg.model
    training_cfg = cfg.training
    evaluation_cfg = cfg.evaluation
    mlflow_cfg = cfg.mlflow

    checkpoint_path = Path(checkpoint or evaluation_cfg["checkpoint"]).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = Path.cwd() / checkpoint_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    assembled = assemble_dataset(dataset_cfg)
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["pretrained_model_name"])

    folds = create_cross_validation_splits(assembled, dataset_cfg)
    fold_index_cfg = evaluation_cfg.get("fold_index")
    if fold_index_cfg is None:
        raise ValueError("evaluation.fold_index must be provided when evaluating cross-validation checkpoints.")
    try:
        fold_index = int(fold_index_cfg)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"evaluation.fold_index must be an integer (received {fold_index_cfg!r}).") from exc
    if not 1 <= fold_index <= len(folds):
        raise ValueError(f"Requested fold_index={fold_index} outside valid range 1..{len(folds)}")
    selected_split = folds[fold_index - 1]

    pin_memory_device_cfg = training_cfg.get("pin_memory_device")
    resolved_pin_memory_device: Optional[str]
    if pin_memory_device_cfg is None:
        resolved_pin_memory_device = None
    else:
        pin_device_str = str(pin_memory_device_cfg).strip()
        resolved_pin_memory_device = pin_device_str or None

    _, _, test_loader = build_dataloaders(
        selected_split,
        tokenizer,
        max_length=int(model_cfg["max_seq_length"]),
        batch_size=int(training_cfg.get("eval_batch_size", training_cfg["batch_size"])),
        eval_batch_size=int(training_cfg.get("eval_batch_size", training_cfg["batch_size"])),
        doc_stride=model_cfg.get("doc_stride"),
        num_workers=training_cfg.get("num_workers"),
        cache_cfg=dataset_cfg,
        prefetch_factor=training_cfg.get("prefetch_factor"),
        persistent_workers=training_cfg.get("persistent_workers"),
        pin_memory_device=resolved_pin_memory_device,
    )
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    non_blocking_cfg_raw = training_cfg.get("non_blocking_transfers", True)
    non_blocking_transfers = bool(non_blocking_cfg_raw)
    test_non_blocking = non_blocking_transfers and getattr(test_loader, "pin_memory", False)

    enable_tf32_cfg = bool(training_cfg.get("enable_tf32", True)) if use_cuda else False
    cudnn_benchmark_cfg = training_cfg.get("cudnn_benchmark")
    matmul_precision_cfg = training_cfg.get("float32_matmul_precision")
    if matmul_precision_cfg is None:
        matmul_precision_cfg = "high" if use_cuda else "medium"
    eval_applied_tf32 = False
    eval_applied_cudnn_benchmark = False
    eval_applied_matmul_precision: Optional[str] = None
    if use_cuda:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = enable_tf32_cfg
            eval_applied_tf32 = enable_tf32_cfg
        if hasattr(torch.backends, "cudnn"):
            if hasattr(torch.backends.cudnn, "allow_tf32"):
                torch.backends.cudnn.allow_tf32 = enable_tf32_cfg
            if cudnn_benchmark_cfg is None:
                torch.backends.cudnn.benchmark = True
                eval_applied_cudnn_benchmark = True
            else:
                torch.backends.cudnn.benchmark = bool(cudnn_benchmark_cfg)
                eval_applied_cudnn_benchmark = bool(cudnn_benchmark_cfg)
    try:
        torch.set_float32_matmul_precision(str(matmul_precision_cfg))
        if hasattr(torch, "get_float32_matmul_precision"):
            eval_applied_matmul_precision = torch.get_float32_matmul_precision()
        else:
            eval_applied_matmul_precision = str(matmul_precision_cfg)
    except (TypeError, ValueError):
        eval_applied_matmul_precision = None
        warnings.warn(
            f"Invalid float32_matmul_precision={matmul_precision_cfg!r} during evaluation; keeping previous precision.",
            RuntimeWarning,
        )

    model = create_model(model_cfg).to(device)
    compile_requested = bool(training_cfg.get("torch_compile", False))
    compiled_model = False
    torch_compile_mode = str(training_cfg.get("torch_compile_mode", "default"))
    torch_compile_backend = training_cfg.get("torch_compile_backend")
    torch_compile_dynamic = training_cfg.get("torch_compile_dynamic")
    torch_compile_fullgraph = training_cfg.get("torch_compile_fullgraph")
    multi_gpu = use_cuda and torch.cuda.device_count() > 1
    if compile_requested and not hasattr(torch, "compile"):
        warnings.warn("torch.compile requested for evaluation but not available in this PyTorch build; skipping.", RuntimeWarning)
        compile_requested = False
    if compile_requested and multi_gpu:
        warnings.warn("torch.compile is skipped during evaluation because DataParallel is in use.", RuntimeWarning)
        compile_requested = False
    if compile_requested:
        compile_kwargs: Dict[str, Any] = {"mode": torch_compile_mode}
        if torch_compile_backend:
            compile_kwargs["backend"] = str(torch_compile_backend)
        if torch_compile_dynamic is not None:
            compile_kwargs["dynamic"] = bool(torch_compile_dynamic)
        if torch_compile_fullgraph is not None:
            compile_kwargs["fullgraph"] = bool(torch_compile_fullgraph)
        try:
            model = torch.compile(model, **compile_kwargs)
            compiled_model = True
        except Exception as err:
            warnings.warn(f"torch.compile failed during evaluation ({err}); continuing without compilation.", RuntimeWarning)
            compiled_model = False
    if multi_gpu:
        model = torch.nn.DataParallel(model)
    model_core = unwrap_model(model)
    state = torch.load(checkpoint_path, map_location=device)
    load_state_dict_flexible(model_core, state)
    metrics = evaluate(model, test_loader, device, non_blocking=test_non_blocking)

    mlflow_db_path = Path(str(mlflow_cfg.get("tracking_db", "mlflow.db"))).expanduser()
    if not mlflow_db_path.is_absolute():
        mlflow_db_path = Path.cwd() / mlflow_db_path
    mlflow_db_path.parent.mkdir(parents=True, exist_ok=True)

    artifact_dir = Path(str(mlflow_cfg.get("artifact_store", "mlruns"))).expanduser()
    if not artifact_dir.is_absolute():
        artifact_dir = Path.cwd() / artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(f"sqlite:///{mlflow_db_path.resolve()}")
    mlflow.set_experiment(experiment_name=str(mlflow_cfg.get("experiment_name", "default")))
    eval_run_name = f"evaluate_{datetime.now():%Y%m%d_%H%M%S}"
    eval_tags = {str(k): str(v) for k, v in mlflow_cfg.get("tags", {}).items()}
    eval_tags["phase"] = "evaluation"

    with mlflow.start_run(run_name=eval_run_name, tags=eval_tags):
        mlflow.log_param("checkpoint", str(checkpoint_path))
        if fold_index is not None:
            mlflow.log_param("fold_index", str(fold_index))
        mlflow.log_param("enable_tf32_effective", str(eval_applied_tf32))
        mlflow.log_param("cudnn_benchmark_effective", str(eval_applied_cudnn_benchmark))
        if eval_applied_matmul_precision is not None:
            mlflow.log_param("float32_matmul_precision_effective", str(eval_applied_matmul_precision))
        mlflow.log_param("non_blocking_transfers", str(non_blocking_transfers))
        mlflow.log_param("torch_compile_requested", str(bool(training_cfg.get("torch_compile", False))))
        mlflow.log_param("torch_compile_active", str(compiled_model))
        mlflow.log_param("torch_compile_mode", torch_compile_mode)
        if torch_compile_backend:
            mlflow.log_param("torch_compile_backend", str(torch_compile_backend))
        if torch_compile_dynamic is not None:
            mlflow.log_param("torch_compile_dynamic", str(bool(torch_compile_dynamic)))
        if torch_compile_fullgraph is not None:
            mlflow.log_param("torch_compile_fullgraph", str(bool(torch_compile_fullgraph)))
        if resolved_pin_memory_device:
            mlflow.log_param("pin_memory_device", str(resolved_pin_memory_device))
        mlflow.log_metrics({f"eval/{k}": float(v) for k, v in metrics.items()})

    return metrics
