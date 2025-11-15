"""Training and evaluation helpers for the criteria baseline rebuild."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

from .data import (
    CriteriaDataset,
    DatasetSplit,
    assemble_dataset,
    compute_label_counts,
    create_cross_validation_splits,
)
from .model import CriteriaClassifier, CriteriaModelConfig

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
    doc_stride: int,
    batch_size: int,
    eval_batch_size: int,
    num_workers: Optional[int] = None,
    cache_cfg: Optional[Mapping[str, Any]] = None,
    prefetch_factor: Optional[int] = None,
    persistent_workers: Optional[bool] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create PyTorch dataloaders for train/val/test."""
    cache_in_memory = bool(cache_cfg.get("cache_in_memory", False)) if cache_cfg else False
    cache_fraction = float(cache_cfg.get("cache_max_ram_fraction", 0.9)) if cache_cfg else 0.9
    cache_fraction = max(0.0, min(cache_fraction, 1.0))
    dataset_kwargs = {
        "cache_in_memory": cache_in_memory,
        "cache_max_ram_fraction": cache_fraction,
    }
    train_dataset = CriteriaDataset(
        splits.train,
        tokenizer=tokenizer,
        max_length=max_length,
        doc_stride=doc_stride,
        **dataset_kwargs,
    )
    eval_dataset_kwargs = {**dataset_kwargs, "cache_in_memory": False}
    val_dataset = CriteriaDataset(
        splits.val,
        tokenizer=tokenizer,
        max_length=max_length,
        doc_stride=doc_stride,
        **eval_dataset_kwargs,
    )
    test_dataset = CriteriaDataset(
        splits.test,
        tokenizer=tokenizer,
        max_length=max_length,
        doc_stride=doc_stride,
        **eval_dataset_kwargs,
    )

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
        if cpu_count <= 2:
            worker_candidates = 0
        else:
            half_cpus = max(1, cpu_count // 2)
            worker_candidates = max(1, min(cpu_count - 1, half_cpus))

    pin_memory = torch.cuda.is_available()

    def _instantiate_loaders(worker_count: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        kwargs = {"num_workers": worker_count, "pin_memory": pin_memory}
        if pin_memory:
            kwargs["pin_memory_device"] = "cuda"
        if worker_count > 0:
            resolved_persistent: Optional[bool]
            if persistent_workers is None:
                resolved_persistent = True
            else:
                resolved_persistent = bool(persistent_workers)
            kwargs["persistent_workers"] = resolved_persistent
            resolved_prefetch: Optional[int]
            if prefetch_factor is None:
                resolved_prefetch = max(2, batch_size // 2) if batch_size > 1 else 2
            else:
                try:
                    resolved_prefetch = max(2, int(prefetch_factor))
                except (TypeError, ValueError):
                    resolved_prefetch = None
            if resolved_prefetch is not None:
                kwargs["prefetch_factor"] = resolved_prefetch
        train_loader_local = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
        val_loader_local = DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, **kwargs)
        test_loader_local = DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, **kwargs)
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
    encoder_dropout_raw = model_cfg.get("encoder_dropout")
    attention_dropout_raw = model_cfg.get("attention_dropout")
    encoder_dropout = float(encoder_dropout_raw) if encoder_dropout_raw is not None else None
    attention_dropout = float(attention_dropout_raw) if attention_dropout_raw is not None else None
    alpha_raw = model_cfg.get("alpha")
    alpha_value: Optional[float]
    if isinstance(alpha_raw, str) and alpha_raw.lower() == "auto":
        alpha_value = None
    elif alpha_raw is None:
        alpha_value = None
    else:
        try:
            alpha_value = float(alpha_raw)
        except (TypeError, ValueError):
            alpha_value = None
    effective_beta_raw = model_cfg.get("effective_beta")
    effective_beta: Optional[float]
    if effective_beta_raw is None:
        effective_beta = None
    else:
        try:
            effective_beta = float(effective_beta_raw)
        except (TypeError, ValueError):
            effective_beta = None
    config = CriteriaModelConfig(
        model_name=model_cfg["pretrained_model_name"],
        hidden_sizes=list(model_cfg.get("classifier_hidden_sizes", [])),
        dropout=float(model_cfg.get("dropout", 0.1)),
        encoder_dropout=encoder_dropout,
        attention_dropout=attention_dropout,
        loss_type=str(model_cfg.get("loss_type", "adaptive_focal")),
        alpha=alpha_value,
        gamma=float(model_cfg.get("gamma", 2.0)),
        delta=float(model_cfg.get("delta", 1.0)),
        effective_beta=effective_beta,
        use_gradient_checkpointing=bool(model_cfg.get("use_gradient_checkpointing", True)),
        freeze_encoder_layers=int(model_cfg.get("freeze_encoder_layers", 0)),
        pooling=str(model_cfg.get("pooling", "cls")),
        classifier_activation=str(model_cfg.get("classifier_activation", "gelu")),
    )
    return CriteriaClassifier(config)


def create_optimizer(
    model: torch.nn.Module,
    training_cfg: Mapping[str, Any],
    steps_per_epoch: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """Set up AdamW optimizer and linear warmup scheduler."""
    lr = float(training_cfg["learning_rate"])
    weight_decay = float(training_cfg.get("weight_decay", 0.0))
    adam_eps = float(training_cfg.get("adam_eps", 1e-8))
    adam_beta1 = float(training_cfg.get("adam_beta1", 0.9))
    adam_beta2 = float(training_cfg.get("adam_beta2", 0.999))
    layerwise_lr_decay = float(training_cfg.get("layerwise_lr_decay", 1.0))
    num_epochs = int(training_cfg["num_epochs"])
    warmup_ratio = float(training_cfg.get("warmup_ratio", 0.1))

    param_groups: List[Dict[str, Any]] = []
    params_assigned: Set[int] = set()

    if layerwise_lr_decay and not math.isclose(layerwise_lr_decay, 1.0):
        encoder_module = getattr(model, "encoder", None)
        transformer_stack = None
        if encoder_module is not None:
            inner_encoder = getattr(encoder_module, "encoder", None)
            transformer_stack = getattr(inner_encoder, "layer", None)
        if transformer_stack is not None:
            layers = list(transformer_stack)
            for depth, layer in enumerate(reversed(layers)):
                scale = layerwise_lr_decay ** depth
                layer_params = [param for param in layer.parameters() if param.requires_grad]
                if layer_params:
                    param_groups.append(
                        {"params": layer_params, "lr": lr * scale, "weight_decay": weight_decay}
                    )
                    params_assigned.update(id(param) for param in layer_params)
            embeddings = getattr(encoder_module, "embeddings", None)
            if embeddings is not None:
                embedding_params = [param for param in embeddings.parameters() if param.requires_grad]
                if embedding_params:
                    scale = layerwise_lr_decay ** (len(layers) + 1)
                    param_groups.append(
                        {"params": embedding_params, "lr": lr * scale, "weight_decay": weight_decay}
                    )
                    params_assigned.update(id(param) for param in embedding_params)

    remaining_params = [
        param for param in model.parameters() if param.requires_grad and id(param) not in params_assigned
    ]
    if remaining_params:
        param_groups.append({"params": remaining_params, "lr": lr, "weight_decay": weight_decay})

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
        weight_decay=weight_decay,
    )
    total_steps = max(1, steps_per_epoch * num_epochs)
    warmup_steps = int(warmup_ratio * total_steps)
    if total_steps > 1:
        warmup_steps = min(warmup_steps, total_steps - 1)
    warmup_steps = max(warmup_steps, 0)
    scheduler_type = str(training_cfg.get("lr_scheduler_type", "linear")).lower()
    if scheduler_type in {"cosine_restarts", "cosine_restart", "cosine_with_restarts"}:
        num_cycles = max(1, int(training_cfg.get("lr_scheduler_num_cycles", 1)))
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
    try:
        auc = roc_auc_score(labels_np, probs)
    except ValueError:
        auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }


@torch.no_grad()
def evaluate(model: CriteriaClassifier, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    """Evaluate model on a dataloader."""
    model.eval()
    all_logits: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    autocast_available = device.type == "cuda"
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
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
        all_logits.append(logits)
        all_labels.append(labels)

    if not all_logits:
        return {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": float("nan")}

    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    return compute_metrics(logits_cat, labels_cat)




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

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    else:
        torch.set_float32_matmul_precision("medium")

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

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["pretrained_model_name"])
    if splits is None:
        raise ValueError("Dataset splits must be provided for cross-validation training.")
    label_counts = compute_label_counts(splits.train)

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

    train_loader, val_loader, test_loader = build_dataloaders(
        splits,
        tokenizer,
        max_length=int(model_cfg["max_seq_length"]),
        doc_stride=int(model_cfg.get("doc_stride", 0)),
        batch_size=int(training_cfg["batch_size"]),
        eval_batch_size=int(training_cfg.get("eval_batch_size", training_cfg["batch_size"])),
        num_workers=training_cfg.get("num_workers"),
        cache_cfg=dataset_cfg,
        prefetch_factor=training_cfg.get("prefetch_factor"),
        persistent_workers=training_cfg.get("persistent_workers"),
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    base_model = create_model(model_cfg).to(device)
    compile_requested = bool(training_cfg.get("use_torch_compile", False))
    compile_mode = str(training_cfg.get("torch_compile_mode", "reduce-overhead"))
    compile_fullgraph = bool(training_cfg.get("torch_compile_fullgraph", False))
    compile_dynamic = bool(training_cfg.get("torch_compile_dynamic", True))
    torch_compile_enabled = False
    if compile_requested and hasattr(torch, "compile"):
        try:
            base_model = torch.compile(
                base_model,
                mode=compile_mode,
                fullgraph=compile_fullgraph,
                dynamic=compile_dynamic,
            )
            torch_compile_enabled = True
        except Exception as compile_err:  # pragma: no cover - fallback for unsupported backends
            warnings.warn(
                f"torch.compile could not be enabled (falling back to eager execution): {compile_err}",
                RuntimeWarning,
            )
    multi_gpu = use_cuda and torch.cuda.device_count() > 1
    if multi_gpu:
        model = torch.nn.DataParallel(base_model)
    else:
        model = base_model
    model_to_optimize = unwrap_model(model)
    model_to_optimize.configure_loss_balancing(label_counts)
    resolved_alpha = getattr(model_to_optimize.loss_fn, "resolved_alpha", None)
    class_weights_tensor = getattr(model_to_optimize.loss_fn, "_class_weights", None)
    class_weights_logged: Optional[List[float]]
    if isinstance(class_weights_tensor, torch.Tensor):
        class_weights_logged = [float(x) for x in class_weights_tensor.detach().cpu().tolist()]
    else:
        class_weights_logged = None

    total_epochs = int(training_cfg["num_epochs"])
    grad_accum = max(1, int(training_cfg.get("gradient_accumulation_steps", 1)))
    total_batches = max(1, math.ceil(len(train_loader.dataset) / int(training_cfg["batch_size"])))
    steps_per_epoch = max(1, math.ceil(total_batches / grad_accum))
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
        if use_cuda:
            try:
                device_props = torch.cuda.get_device_properties(0)
                mlflow.log_param("cuda_device_name", torch.cuda.get_device_name(0))
                mlflow.log_param("cuda_total_memory_mb", str(int(device_props.total_memory // (1024**2))))
            except Exception:  # pragma: no cover - defensive logging
                pass
        mlflow.log_param("dataset_cache_in_memory", str(dataset_cache_in_memory))
        mlflow.log_param("dataset_cache_fraction", f"{dataset_cache_fraction:.3f}")
        mlflow.log_param("dataloader_num_workers", str(getattr(train_loader, "num_workers", 0)))
        mlflow.log_param("class_count_positive", str(label_counts.get(1, 0)))
        mlflow.log_param("class_count_negative", str(label_counts.get(0, 0)))
        if resolved_alpha is not None:
            mlflow.log_param("loss_resolved_alpha", f"{float(resolved_alpha):.6f}")
        if class_weights_logged and len(class_weights_logged) >= 2:
            mlflow.log_param("loss_class_weight_negative", f"{class_weights_logged[0]:.6f}")
            mlflow.log_param("loss_class_weight_positive", f"{class_weights_logged[1]:.6f}")
        mlflow.log_param("torch_compile_requested", str(compile_requested))
        mlflow.log_param("torch_compile_enabled", str(torch_compile_enabled))
        mlflow.log_param("torch_compile_mode", compile_mode)
        mlflow.log_param("torch_compile_fullgraph", str(compile_fullgraph))
        mlflow.log_param("torch_compile_dynamic", str(compile_dynamic))

        for epoch in range(start_epoch, total_epochs + 1):
            model.train()
            running_loss = 0.0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader, start=1):
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

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

            val_metrics = evaluate(model, val_loader, device)
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

        test_metrics = evaluate(model, test_loader, device)
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

        mlflow.log_artifact(history_path)
        mlflow.log_artifact(metrics_path)
        if best_model_path.exists():
            mlflow.log_artifact(best_model_path)
        if scaler_artifact_path.exists():
            mlflow.log_artifact(scaler_artifact_path)
        if resolved_config_path.exists():
            mlflow.log_artifact(resolved_config_path)

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
        "label_counts": {int(k): int(v) for k, v in label_counts.items()},
        "loss_resolved_alpha": float(resolved_alpha) if resolved_alpha is not None else None,
        "loss_class_weights": class_weights_logged,
        "torch_compile_enabled": torch_compile_enabled,
    }


def train(cfg: DictConfig) -> Dict[str, object]:
    """Hydra-managed training entry point for 5-fold cross-validation."""
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
            "num_folds": len(fold_summaries),
        }
    )
    return aggregate


@torch.no_grad()
def evaluate_checkpoint(cfg: DictConfig, checkpoint: Optional[Path] = None) -> Dict[str, float]:
    """Load a saved state dict and score on the held-out fold partition."""
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
    if not folds:
        raise ValueError("No cross-validation folds were generated for evaluation.")
    fold_index = int(evaluation_cfg.get("fold_index", 1))
    if not 1 <= fold_index <= len(folds):
        raise ValueError(f"Requested fold_index={fold_index} outside valid range 1..{len(folds)}")
    selected_split = folds[fold_index - 1]
    label_counts = compute_label_counts(selected_split.train)

    _, _, test_loader = build_dataloaders(
        selected_split,
        tokenizer,
        max_length=int(model_cfg["max_seq_length"]),
        doc_stride=int(model_cfg.get("doc_stride", 0)),
        batch_size=int(training_cfg.get("eval_batch_size", training_cfg["batch_size"])),
        eval_batch_size=int(training_cfg.get("eval_batch_size", training_cfg["batch_size"])),
        num_workers=training_cfg.get("num_workers"),
        cache_cfg=dataset_cfg,
        prefetch_factor=training_cfg.get("prefetch_factor"),
        persistent_workers=training_cfg.get("persistent_workers"),
    )
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    model = create_model(model_cfg).to(device)
    if use_cuda and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model_core = unwrap_model(model)
    model_core.configure_loss_balancing(label_counts)
    eval_resolved_alpha = getattr(model_core.loss_fn, "resolved_alpha", None)
    eval_class_weights_tensor = getattr(model_core.loss_fn, "_class_weights", None)
    if isinstance(eval_class_weights_tensor, torch.Tensor):
        eval_class_weights = [float(x) for x in eval_class_weights_tensor.detach().cpu().tolist()]
    else:
        eval_class_weights = None
    state = torch.load(checkpoint_path, map_location=device)
    load_state_dict_flexible(model_core, state)
    metrics = evaluate(model, test_loader, device)

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
        mlflow.log_param("class_count_positive", str(label_counts.get(1, 0)))
        mlflow.log_param("class_count_negative", str(label_counts.get(0, 0)))
        if use_cuda:
            try:
                device_props = torch.cuda.get_device_properties(0)
                mlflow.log_param("cuda_device_name", torch.cuda.get_device_name(0))
                mlflow.log_param("cuda_total_memory_mb", str(int(device_props.total_memory // (1024**2))))
            except Exception:  # pragma: no cover
                pass
        if eval_resolved_alpha is not None:
            mlflow.log_param("loss_resolved_alpha", f"{float(eval_resolved_alpha):.6f}")
        if eval_class_weights and len(eval_class_weights) >= 2:
            mlflow.log_param("loss_class_weight_negative", f"{eval_class_weights[0]:.6f}")
            mlflow.log_param("loss_class_weight_positive", f"{eval_class_weights[1]:.6f}")
        mlflow.log_metrics({f"eval/{k}": float(v) for k, v in metrics.items()})

    return metrics
