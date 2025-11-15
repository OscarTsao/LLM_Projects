from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, overload

try:
    import optuna  # type: ignore
except ImportError:  # pragma: no cover
    optuna = None  # type: ignore

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from transformers import (
    AutoTokenizer,
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from transformers.optimization import Adafactor

try:
    from transformers.optimization import AdamW as HFAdamW
except ImportError:
    # transformers >= 4.x removed AdamW, use torch.optim.AdamW instead
    HFAdamW = torch.optim.AdamW

from .data import QAExample, build_examples, load_annotations, load_posts, split_examples
from .features import (
    EvalQADataset,
    TrainQADataset,
    eval_collate_fn,
    prepare_eval_features,
    prepare_train_features,
)
from .metrics import aggregate_metrics, save_metrics
from .modeling import SpanBertForQuestionAnswering

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed python random state across common libraries."""

    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_preference: str = "auto") -> torch.device:
    """Resolve the torch.device to run on."""

    if device_preference != "auto":
        return torch.device(device_preference)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _postprocess_predictions(
    examples: List[QAExample],
    features,
    start_logits: List[List[float]],
    end_logits: List[List[float]],
    n_best_size: int,
    max_answer_length: int,
) -> Dict[str, str]:
    """Convert raw start/end scores into string predictions."""

    example_to_features: Dict[int, List[int]] = defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature.example_index].append(idx)

    predictions: Dict[str, str] = {}
    for example_idx, example in enumerate(examples):
        feature_indices = example_to_features.get(example_idx, [])
        best_answer = ""
        best_score = None
        for feature_index in feature_indices:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index].offset_mapping
            for start_index in sorted(
                range(len(start_logit)), key=lambda x: start_logit[x], reverse=True
            )[:n_best_size]:
                for end_index in sorted(
                    range(len(end_logit)), key=lambda x: end_logit[x], reverse=True
                )[:n_best_size]:
                    if end_index < start_index:
                        continue
                    if end_index >= len(offsets) or start_index >= len(offsets):
                        continue
                    start_char, end_char = offsets[start_index]
                    end_char = offsets[end_index][1]
                    if start_char == 0 and end_char == 0:
                        continue
                    length = end_char - start_char
                    if length <= 0 or length > max_answer_length:
                        continue
                    score = start_logit[start_index] + end_logit[end_index]
                    if best_score is None or score > best_score:
                        best_score = score
                        best_answer = example.context[start_char:end_char].strip()
        predictions[example.example_id] = best_answer
    return predictions


@dataclass
class TrainingOutputs:
    best_state_dict: Dict[str, torch.Tensor]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    predictions: Dict[str, str]
    artifact_dir: Path


@dataclass
class TrainingCheckpoint:
    """Training checkpoint for resuming interrupted runs."""

    epoch: int
    global_step: int
    best_metric: float
    epochs_without_improve: int
    cooldown_remaining: int
    model_state_dict: Dict[str, torch.Tensor]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Optional[Dict[str, Any]]
    scaler_state_dict: Optional[Dict[str, Any]]
    best_state_dict: Optional[Dict[str, torch.Tensor]]
    config: Dict[str, Any]


def save_checkpoint(checkpoint: TrainingCheckpoint, checkpoint_path: Path) -> None:
    """Save training checkpoint to disk."""
    torch.save({
        'epoch': checkpoint.epoch,
        'global_step': checkpoint.global_step,
        'best_metric': checkpoint.best_metric,
        'epochs_without_improve': checkpoint.epochs_without_improve,
        'cooldown_remaining': checkpoint.cooldown_remaining,
        'model_state_dict': checkpoint.model_state_dict,
        'optimizer_state_dict': checkpoint.optimizer_state_dict,
        'scheduler_state_dict': checkpoint.scheduler_state_dict,
        'scaler_state_dict': checkpoint.scaler_state_dict,
        'best_state_dict': checkpoint.best_state_dict,
        'config': checkpoint.config,
    }, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def load_checkpoint(checkpoint_path: Path) -> Optional[TrainingCheckpoint]:
    """Load training checkpoint from disk."""
    if not checkpoint_path.exists():
        return None

    try:
        checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
        checkpoint = TrainingCheckpoint(
            epoch=checkpoint_data['epoch'],
            global_step=checkpoint_data['global_step'],
            best_metric=checkpoint_data['best_metric'],
            epochs_without_improve=checkpoint_data['epochs_without_improve'],
            cooldown_remaining=checkpoint_data['cooldown_remaining'],
            model_state_dict=checkpoint_data['model_state_dict'],
            optimizer_state_dict=checkpoint_data['optimizer_state_dict'],
            scheduler_state_dict=checkpoint_data.get('scheduler_state_dict'),
            scaler_state_dict=checkpoint_data.get('scaler_state_dict'),
            best_state_dict=checkpoint_data.get('best_state_dict'),
            config=checkpoint_data.get('config', {}),
        )
        logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {checkpoint.epoch + 1})")
        return checkpoint
    except Exception as e:
        logger.warning(f"Failed to load checkpoint from {checkpoint_path}: {e}")
        return None


def find_latest_checkpoint(artifact_dir: Path) -> Optional[Path]:
    """Find the most recent checkpoint in the artifact directory."""
    if not artifact_dir.exists():
        return None

    # Look for checkpoint.pt in any timestamped subdirectory
    checkpoint_files = list(artifact_dir.glob("*/checkpoint.pt"))
    if not checkpoint_files:
        return None

    # Return the most recently modified checkpoint
    latest_checkpoint = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
    return latest_checkpoint


def _resolve_auto(value, auto_value):
    if isinstance(value, str) and value.lower() == "auto":
        return auto_value
    return value


def _create_optimizer(model: SpanBertForQuestionAnswering, cfg: DictConfig) -> torch.optim.Optimizer:
    """Instantiate optimizer based on configuration."""

    opt_cfg = cfg.training.get("optimizer", {}) or {}
    name = str(opt_cfg.get("name", "adamw_torch")).lower()
    lr = float(cfg.training.get("learning_rate", 5e-5))
    weight_decay = float(cfg.training.get("weight_decay", 0.0))
    eps = float(cfg.training.get("adam_epsilon", 1e-8))
    betas = opt_cfg.get("betas", [0.9, 0.999])
    if isinstance(betas, DictConfig):
        betas = list(betas)
    if isinstance(betas, (list, tuple)):
        if len(betas) == 1:
            betas = (float(betas[0]), 0.999)
        else:
            betas = (float(betas[0]), float(betas[1]))
    else:
        betas = (0.9, 0.999)
    beta1, beta2 = betas

    fused = opt_cfg.get("use_fused", False)
    fused = bool(_resolve_auto(fused, torch.cuda.is_available()))
    amsgrad = bool(opt_cfg.get("amsgrad", False))
    momentum = float(opt_cfg.get("momentum", 0.9))
    nesterov = bool(opt_cfg.get("nesterov", False))
    alpha = float(opt_cfg.get("alpha", 0.99))
    centered = bool(opt_cfg.get("centered", False))
    relative_step = bool(opt_cfg.get("relative_step", False))
    scale_parameter = bool(opt_cfg.get("scale_parameter", False))
    warmup_init = bool(opt_cfg.get("warmup_init", False))

    params = model.parameters()

    try:
        if name in {"adamw_torch", "adamw"}:
            optimizer_kwargs = {
                "lr": lr,
                "eps": eps,
                "weight_decay": weight_decay,
                "betas": (beta1, beta2),
                "amsgrad": amsgrad,
            }
            # Only add fused if explicitly enabled and supported
            if fused and hasattr(torch.optim.AdamW, '__init__'):
                try:
                    import inspect
                    sig = inspect.signature(torch.optim.AdamW.__init__)
                    if 'fused' in sig.parameters:
                        optimizer_kwargs["fused"] = True
                except Exception:
                    pass  # Fall back to unfused if introspection fails
            optimizer = torch.optim.AdamW(params, **optimizer_kwargs)
        elif name == "adamw_hf":
            optimizer = HFAdamW(params, lr=lr, eps=eps, betas=(beta1, beta2), weight_decay=weight_decay)
        elif name == "adam":
            optimizer_kwargs = {
                "lr": lr,
                "eps": eps,
                "betas": (beta1, beta2),
                "amsgrad": amsgrad,
                "weight_decay": weight_decay,
            }
            # Only add fused if explicitly enabled and supported
            if fused and hasattr(torch.optim.Adam, '__init__'):
                try:
                    import inspect
                    sig = inspect.signature(torch.optim.Adam.__init__)
                    if 'fused' in sig.parameters:
                        optimizer_kwargs["fused"] = True
                except Exception:
                    pass  # Fall back to unfused if introspection fails
            optimizer = torch.optim.Adam(params, **optimizer_kwargs)
        elif name == "adamax":
            optimizer = torch.optim.Adamax(params, lr=lr, eps=eps, betas=(beta1, beta2), weight_decay=weight_decay)
        elif name == "rmsprop":
            optimizer = torch.optim.RMSprop(
                params,
                lr=lr,
                alpha=alpha,
                eps=eps,
                weight_decay=weight_decay,
                momentum=momentum,
                centered=centered,
            )
        elif name == "sgd":
            # Validate nesterov requires momentum > 0
            if nesterov and momentum <= 0:
                logger.warning("SGD with nesterov=True requires momentum > 0; setting momentum=0.9")
                momentum = 0.9
            optimizer = torch.optim.SGD(
                params,
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
                nesterov=nesterov,
            )
        elif name == "adafactor":
            if warmup_init and not relative_step:
                logger.warning(
                    "Adafactor warmup_init=True requires relative_step=True; disabling warmup_init to respect fixed-step learning rate"
                )
                warmup_init = False
            adafactor_lr = None if relative_step else lr
            optimizer = Adafactor(
                params,
                lr=adafactor_lr,
                eps=(1e-30, eps),
                weight_decay=weight_decay,
                relative_step=relative_step,
                scale_parameter=scale_parameter,
                warmup_init=warmup_init,
            )
        else:
            logger.warning("Unknown optimizer '%s', falling back to AdamW", name)
            optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                eps=eps,
                weight_decay=weight_decay,
                betas=(beta1, beta2),
                amsgrad=amsgrad,
            )
    except Exception as e:
        logger.error(f"Failed to create optimizer '{name}': {e}")
        logger.info("Falling back to AdamW with safe defaults")
        optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            amsgrad=False,
        )

    return optimizer


def _create_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    total_steps: int,
):
    """Instantiate LR scheduler based on configuration."""

    # Check if optimizer has lr=None (e.g., Adafactor with relative_step=True)
    # In this case, return a no-op scheduler since the optimizer manages its own LR
    if optimizer.param_groups and optimizer.param_groups[0].get('lr') is None:
        logger.info("Optimizer manages its own learning rate schedule; using no-op scheduler")

        # Create a custom no-op scheduler that doesn't interfere with None learning rates
        class NoOpScheduler:
            def __init__(self, optimizer):
                self.optimizer = optimizer

            def step(self, epoch=None):
                pass

            def get_last_lr(self):
                return [None] * len(self.optimizer.param_groups)

        return NoOpScheduler(optimizer)

    scheduler_cfg = cfg.training.get("scheduler", {}) or {}
    name = str(scheduler_cfg.get("name", "linear")).lower()
    warmup_steps = scheduler_cfg.get("warmup_steps")
    warmup_ratio = scheduler_cfg.get("warmup_ratio")
    if warmup_steps is None:
        fallback_ratio = cfg.training.get("warmup_ratio", 0.0)
        warmup_ratio = warmup_ratio if warmup_ratio is not None else fallback_ratio
        warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(int(warmup_steps), 0)

    if total_steps <= 0:
        total_steps = 1

    if name == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if name == "cosine":
        num_cycles = float(scheduler_cfg.get("num_cycles", 0.5))
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps, num_cycles=num_cycles)
    if name == "cosine_with_restarts":
        num_cycles = max(int(round(float(scheduler_cfg.get("num_cycles", 1.0)))), 1)
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, warmup_steps, total_steps, num_cycles=num_cycles
        )
    if name == "polynomial":
        power = float(scheduler_cfg.get("power", 1.0))
        lr_end = scheduler_cfg.get("lr_end")
        if lr_end is None:
            lr_end_ratio = float(scheduler_cfg.get("lr_end_ratio", 0.0))
            lr_end = cfg.training.get("learning_rate", 0.0) * lr_end_ratio
        lr_end = float(lr_end)
        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_steps,
            lr_end=lr_end,
            power=power,
        )
    if name == "constant":
        return get_constant_schedule(optimizer)
    if name == "constant_with_warmup":
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    if name == "step":
        step_size = int(scheduler_cfg.get("step_size", max(total_steps // max(cfg.training.num_train_epochs, 1), 1)))
        gamma = float(scheduler_cfg.get("gamma", 0.1))
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    if name == "cosineannealing":
        t_max = scheduler_cfg.get("t_max")
        if t_max is None:
            t_max_ratio = float(scheduler_cfg.get("t_max_ratio", 1.0))
            t_max = max(int(total_steps * t_max_ratio), 1)
        eta_min = scheduler_cfg.get("eta_min")
        if eta_min is None:
            eta_min_ratio = float(scheduler_cfg.get("eta_min_ratio", 0.0))
            eta_min = cfg.training.get("learning_rate", 0.0) * eta_min_ratio
        return CosineAnnealingLR(optimizer, T_max=max(int(t_max), 1), eta_min=float(eta_min))

    logger.warning("Unknown scheduler '%s', defaulting to linear", name)
    return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


def run_training(
    cfg: DictConfig,
    save_artifacts: bool = True,
    trial: Optional[Any] = None,
    resume_from: Optional[str] = "auto",
) -> TrainingOutputs:
    """End-to-end orchestration: data prep, training loop, and evaluation.

    Args:
        cfg: Configuration object
        save_artifacts: Whether to save model and metrics
        trial: Optional Optuna trial for hyperparameter optimization
        resume_from: Path to checkpoint or "auto" to find latest, None to start fresh
    """

    # Disable tokenizers parallelism to avoid fork warnings with DataLoader workers
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    local_files_only = bool(cfg.model.local_files_only) if "local_files_only" in cfg.model else False
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    set_seed(cfg.training.seed)
    device = select_device(cfg.training.device)

    # Set TensorFloat32 precision for better performance on Ampere+ GPUs
    if device.type == "cuda" and torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = bool(cfg.training.get("cudnn_benchmark", True))
        if hasattr(torch.backends.cudnn, "deterministic"):
            torch.backends.cudnn.deterministic = bool(cfg.training.get("cudnn_deterministic", False))

    posts = load_posts(Path(cfg.data.groundtruth_path))
    annotations = load_annotations(Path(cfg.data.annotations_path), positive_only=cfg.data.positive_only)
    examples = build_examples(posts, annotations)
    train_examples, val_examples, test_examples = split_examples(
        examples, cfg.data.train_ratio, cfg.data.val_ratio, cfg.data.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.pretrained_model_name_or_path,
        use_fast=True,
        local_files_only=local_files_only,
    )

    train_features = prepare_train_features(train_examples, tokenizer, cfg.features.max_length, cfg.features.doc_stride)
    val_features = prepare_eval_features(val_examples, tokenizer, cfg.features.max_length, cfg.features.doc_stride)
    test_features = prepare_eval_features(test_examples, tokenizer, cfg.features.max_length, cfg.features.doc_stride)

    train_dataset = TrainQADataset(train_features)
    val_dataset = EvalQADataset(val_features)
    test_dataset = EvalQADataset(test_features)

    # Optimize DataLoader for better throughput
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    raw_num_workers = cfg.training.get("num_workers", 0)
    if isinstance(raw_num_workers, str) and raw_num_workers.lower() == "auto":
        cpu_count = os.cpu_count() or 1
        num_workers = max(cpu_count - 1, 1)
    else:
        num_workers = int(raw_num_workers)

    pin_memory_opt = cfg.training.get("pin_memory", "auto")
    pin_memory = bool(_resolve_auto(pin_memory_opt, use_cuda))

    persistent_opt = cfg.training.get("persistent_workers", "auto")
    persistent_workers = bool(_resolve_auto(persistent_opt, num_workers > 0)) and num_workers > 0

    prefetch_factor = cfg.training.get("prefetch_factor", None)
    if isinstance(prefetch_factor, str) and prefetch_factor.lower() == "auto":
        prefetch_factor = 2
    if num_workers <= 0:
        prefetch_factor = None
    elif prefetch_factor is not None:
        prefetch_factor = max(int(prefetch_factor), 2)

    eval_prefetch = cfg.training.get("eval_prefetch_factor", prefetch_factor)
    if isinstance(eval_prefetch, str) and eval_prefetch.lower() == "auto":
        eval_prefetch = prefetch_factor
    if num_workers <= 0:
        eval_prefetch = None
    elif eval_prefetch is not None:
        eval_prefetch = max(int(eval_prefetch), 2)

    test_prefetch = cfg.training.get("test_prefetch_factor", eval_prefetch)
    if isinstance(test_prefetch, str) and test_prefetch.lower() == "auto":
        test_prefetch = eval_prefetch
    if num_workers <= 0:
        test_prefetch = None
    elif test_prefetch is not None:
        test_prefetch = max(int(test_prefetch), 2)

    dataloader_timeout = float(cfg.training.get("dataloader_timeout", 0) or 0.0)
    drop_last = bool(cfg.training.get("drop_last", True))
    shuffle_train = bool(cfg.training.get("shuffle_train", True))

    def _make_loaders(current_pin_memory: bool, current_persistent_workers: bool):
        def _build_loader(dataset, batch_size, shuffle=False, drop_last=False, collate_fn=None, prefetch=None):
            loader_kwargs = {
                "batch_size": batch_size,
                "shuffle": shuffle,
                "num_workers": num_workers,
                "pin_memory": current_pin_memory,
                "drop_last": drop_last,
                "timeout": dataloader_timeout,
            }
            if num_workers > 0 and current_persistent_workers:
                loader_kwargs["persistent_workers"] = True
            if prefetch is not None and num_workers > 0:
                loader_kwargs["prefetch_factor"] = prefetch
            if collate_fn is not None:
                loader_kwargs["collate_fn"] = collate_fn
            return DataLoader(dataset, **loader_kwargs)

        train_loader_local = _build_loader(
            train_dataset,
            batch_size=cfg.training.train_batch_size,
            shuffle=shuffle_train,
            drop_last=drop_last,
            prefetch=prefetch_factor,
        )
        val_loader_local = _build_loader(
            val_dataset,
            batch_size=cfg.training.eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=eval_collate_fn,
            prefetch=eval_prefetch,
        )
        test_loader_local = _build_loader(
            test_dataset,
            batch_size=cfg.training.eval_batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=eval_collate_fn,
            prefetch=test_prefetch,
        )
        return train_loader_local, val_loader_local, test_loader_local

    train_loader, val_loader, test_loader = _make_loaders(pin_memory, persistent_workers)

    if pin_memory:
        def _warmup_loader(loader_name, loader_obj):
            iterator = iter(loader_obj)
            success = True
            try:
                next(iterator)
            except StopIteration:
                success = True
            except RuntimeError as err:
                message = str(err).lower()
                if "pin memory" in message:
                    logger.warning(
                        "Pin memory warmup failed on %s loader; disabling pin_memory and persistent_workers for this run.",
                        loader_name,
                    )
                    success = False
                else:
                    raise
            finally:
                shutdown = getattr(iterator, "_shutdown_workers", None)
                if callable(shutdown):
                    shutdown()
            return success

        warmup_ok = True
        for loader_label, loader_obj in (
            ("training", train_loader),
            ("validation", val_loader),
            ("test", test_loader),
        ):
            if not _warmup_loader(loader_label, loader_obj):
                warmup_ok = False
                break

        if not warmup_ok:
            pin_memory = False
            persistent_workers = False

        train_loader, val_loader, test_loader = _make_loaders(pin_memory, persistent_workers)

    # Auto-resume: check for existing checkpoint
    checkpoint = None
    artifact_dir = None
    if resume_from == "auto":
        base_artifact_dir = Path(cfg.training.artifact_dir)
        checkpoint_path = find_latest_checkpoint(base_artifact_dir)
        if checkpoint_path:
            checkpoint = load_checkpoint(checkpoint_path)
            if checkpoint:
                artifact_dir = checkpoint_path.parent
                logger.info(f"Resuming training from {checkpoint_path}")
    elif resume_from:
        checkpoint_path = Path(resume_from)
        checkpoint = load_checkpoint(checkpoint_path)
        if checkpoint:
            artifact_dir = checkpoint_path.parent
            logger.info(f"Resuming training from {checkpoint_path}")

    # Create artifact directory if not resuming
    if artifact_dir is None:
        artifact_dir = Path(cfg.training.artifact_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        if save_artifacts:
            artifact_dir.mkdir(parents=True, exist_ok=True)

    gradient_checkpointing = cfg.model.get("gradient_checkpointing", False)
    model = SpanBertForQuestionAnswering(
        cfg.model.pretrained_model_name_or_path,
        dropout=cfg.model.dropout,
        local_files_only=local_files_only,
        gradient_checkpointing=gradient_checkpointing,
    )
    model.to(device)

    if gradient_checkpointing:
        logger.info("Gradient checkpointing enabled for memory efficiency")

    # Use PyTorch 2.0 compilation for faster training if available
    if hasattr(torch, "compile") and cfg.training.get("compile_model", False):
        compile_kwargs = {
            "mode": cfg.training.get("compile_mode", "default"),
            "dynamic": bool(cfg.training.get("compile_dynamic", False)),
            "fullgraph": bool(cfg.training.get("compile_fullgraph", False)),
        }
        backend = cfg.training.get("compile_backend")
        if backend:
            compile_kwargs["backend"] = backend
        logger.info(
            "Compiling model with torch.compile backend=%s mode=%s",
            compile_kwargs.get("backend", "default"),
            compile_kwargs.get("mode", "default"),
        )
        try:
            model = torch.compile(model, **compile_kwargs)
        except Exception as exc:
            logger.warning("Failed to compile model: %s. Continuing without compilation.", exc)

    optimizer = _create_optimizer(model, cfg)
    grad_accum = max(int(cfg.training.gradient_accumulation_steps), 1)
    steps_per_epoch = math.ceil(len(train_loader) / grad_accum)
    raw_max_steps = cfg.training.get("max_steps")
    max_update_steps = int(raw_max_steps) if raw_max_steps is not None and int(raw_max_steps) > 0 else None
    if max_update_steps is not None:
        total_steps = max(max_update_steps, 1)
    else:
        total_steps = max(steps_per_epoch * int(cfg.training.num_train_epochs), 1)
    scheduler = _create_scheduler(optimizer, cfg, total_steps)

    mixed_precision_mode = str(cfg.training.get("mixed_precision", "auto")).lower()
    amp_dtype_pref = str(cfg.training.get("amp_dtype", "auto")).lower()
    device_type = device.type
    amp_enabled = mixed_precision_mode != "off"

    if mixed_precision_mode == "bf16":
        autocast_dtype = torch.bfloat16
    elif mixed_precision_mode in {"fp16", "float16"}:
        autocast_dtype = torch.float16
    else:
        if amp_dtype_pref in {"bfloat16", "bf16"}:
            autocast_dtype = torch.bfloat16
        elif amp_dtype_pref in {"float16", "fp16"}:
            autocast_dtype = torch.float16
        else:
            if device_type == "cuda" and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                autocast_dtype = torch.bfloat16
            else:
                autocast_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

    if device_type != "cuda" and autocast_dtype == torch.float16:
        amp_enabled = False

    supported_autocast = device_type in {"cuda", "cpu"}
    if not supported_autocast:
        amp_enabled = False

    scaler_enabled = amp_enabled and device_type == "cuda" and autocast_dtype == torch.float16
    scaler = torch.amp.GradScaler(enabled=scaler_enabled)
    autocast_kwargs = {"device_type": device_type, "enabled": amp_enabled and supported_autocast}
    if autocast_kwargs["enabled"] and autocast_dtype is not None:
        autocast_kwargs["dtype"] = autocast_dtype

    opt_cfg = cfg.optimization
    patience = int(opt_cfg.get("patience", 0))
    min_delta = float(opt_cfg.get("min_delta", 0.0))
    warmup_epochs = max(int(opt_cfg.get("warmup_epochs", 0)), 0)
    cooldown_epochs = max(int(opt_cfg.get("cooldown_epochs", 0)), 0)

    # Initialize training state
    start_epoch = 0
    best_metric = -math.inf if opt_cfg.higher_is_better else math.inf
    best_state_dict: Optional[Dict[str, torch.Tensor]] = None
    epochs_without_improve = 0
    global_step = 0
    cooldown_remaining = 0

    # Load checkpoint state if resuming
    if checkpoint:
        model.load_state_dict(checkpoint.model_state_dict)
        model.to(device)
        optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        if checkpoint.scheduler_state_dict:
            scheduler.load_state_dict(checkpoint.scheduler_state_dict)
        if checkpoint.scaler_state_dict and scaler_enabled:
            scaler.load_state_dict(checkpoint.scaler_state_dict)
        start_epoch = checkpoint.epoch + 1
        global_step = checkpoint.global_step
        best_metric = checkpoint.best_metric
        epochs_without_improve = checkpoint.epochs_without_improve
        cooldown_remaining = checkpoint.cooldown_remaining
        best_state_dict = checkpoint.best_state_dict
        logger.info(f"Resumed from epoch {start_epoch}, global_step {global_step}, best_metric {best_metric:.4f}")

    use_tqdm = cfg.logging.get("use_tqdm", True)
    stop_training = False

    for epoch in range(start_epoch, cfg.training.num_train_epochs):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cfg.training.num_train_epochs} [Train]",
            disable=not use_tqdm,
            dynamic_ncols=True,
            unit="batch",
        )

        batches_processed = 0
        for step, batch in enumerate(progress_bar):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            with torch.amp.autocast(**autocast_kwargs):
                outputs = model(**batch)
                loss = outputs["loss"]

            loss = loss / grad_accum
            scaler.scale(loss).backward()
            batches_processed += 1

            if (step + 1) % grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                global_step += 1
                if max_update_steps is not None and global_step >= max_update_steps:
                    stop_training = True
                    break

            epoch_loss += loss.item() * grad_accum

            # Update progress bar with detailed metrics
            if use_tqdm:
                current_lr = optimizer.param_groups[0]["lr"]
                avg_loss = epoch_loss / max(batches_processed, 1)
                progress_bar.set_postfix({
                    "loss": f"{loss.item() * grad_accum:.4f}",
                    "avg_loss": f"{avg_loss:.4f}",
                    "lr": f"{current_lr:.2e}",
                    "step": global_step,
                })

        avg_epoch_loss = epoch_loss / max(batches_processed, 1)
        if stop_training:
            logger.info("Reached maximum update steps (%d); stopping training early.", max_update_steps)
        logger.info(f"Epoch {epoch + 1} training loss: {avg_epoch_loss:.4f}")

        val_metrics = evaluate(
            model,
            val_loader,
            val_examples,
            val_features,
            cfg.features.n_best_size,
            cfg.features.max_answer_length,
            device,
            use_tqdm=use_tqdm,
            split_name="Validation",
        )
        current_metric = val_metrics.get(opt_cfg.metric, 0.0)
        logger.info(
            f"Epoch {epoch + 1} validation - F1: {val_metrics.get('f1', 0.0):.4f}, "
            f"EM: {val_metrics.get('exact_match', 0.0):.4f}"
        )

        if trial is not None and optuna is not None:
            report_step = global_step if global_step > 0 else epoch + 1
            trial.report(current_metric, report_step)
            if trial.should_prune():
                logger.info(
                    "Optuna pruned the trial at epoch %d with %s=%.4f",
                    epoch + 1,
                    opt_cfg.metric,
                    current_metric,
                )
                raise optuna.TrialPruned(f"Pruned at epoch {epoch + 1}")

        if best_state_dict is None:
            improved = True
            improvement_value = float("inf")
        elif opt_cfg.higher_is_better:
            improvement_value = current_metric - best_metric
            improved = improvement_value > min_delta
        else:
            improvement_value = best_metric - current_metric
            improved = improvement_value > min_delta

        if improved:
            best_metric = current_metric
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_without_improve = 0
            cooldown_remaining = cooldown_epochs
            if improvement_value != float("inf"):
                logger.info(
                    "New best %s: %.4f (improved by %.4f)",
                    opt_cfg.metric,
                    best_metric,
                    improvement_value,
                )
            else:
                logger.info("New best %s: %.4f", opt_cfg.metric, best_metric)
        else:
            if epoch + 1 <= warmup_epochs:
                logger.info(
                    "Warmup epoch %d/%d: skipping early stopping counter.",
                    epoch + 1,
                    warmup_epochs,
                )
            elif cooldown_remaining > 0:
                cooldown_remaining -= 1
                logger.info(
                    "Cooldown active (%d epoch(s) remaining); keeping best %s %.4f",
                    cooldown_remaining,
                    opt_cfg.metric,
                    best_metric,
                )
            else:
                epochs_without_improve += 1
                logger.info(
                    "No improvement for %d epoch(s). Best %s: %.4f",
                    epochs_without_improve,
                    opt_cfg.metric,
                    best_metric,
                )
                if patience > 0 and epochs_without_improve >= patience:
                    logger.info(
                        "Early stopping triggered after %d epochs without improvement", epochs_without_improve
                    )
                    break

        # Save checkpoint at end of each epoch (if saving artifacts)
        if save_artifacts and artifact_dir:
            checkpoint_to_save = TrainingCheckpoint(
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                epochs_without_improve=epochs_without_improve,
                cooldown_remaining=cooldown_remaining,
                model_state_dict={k: v.cpu() for k, v in model.state_dict().items()},
                optimizer_state_dict=optimizer.state_dict(),
                scheduler_state_dict=scheduler.state_dict() if scheduler else None,
                scaler_state_dict=scaler.state_dict() if scaler_enabled else None,
                best_state_dict=best_state_dict,
                config=OmegaConf.to_container(cfg, resolve=True),
            )
            checkpoint_path = artifact_dir / "checkpoint.pt"
            save_checkpoint(checkpoint_to_save, checkpoint_path)

        if stop_training:
            break

    if best_state_dict is None:
        best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state_dict)
    model.to(device)

    logger.info("Loading best model for final evaluation")
    val_metrics = evaluate(
        model,
        val_loader,
        val_examples,
        val_features,
        cfg.features.n_best_size,
        cfg.features.max_answer_length,
        device,
        use_tqdm=use_tqdm,
        split_name="Final Validation",
    )
    test_metrics, test_predictions = evaluate(
        model,
        test_loader,
        test_examples,
        test_features,
        cfg.features.n_best_size,
        cfg.features.max_answer_length,
        device,
        return_predictions=True,
        use_tqdm=use_tqdm,
        split_name="Test",
    )
    logger.info(f"Final test metrics - F1: {test_metrics.get('f1', 0.0):.4f}, EM: {test_metrics.get('exact_match', 0.0):.4f}")

    # Save final artifacts (artifact_dir already created during initialization)
    if save_artifacts and artifact_dir:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_path = artifact_dir / "best_model.pt"
        torch.save(best_state_dict, model_path)
        config_path = artifact_dir / "config.yaml"
        OmegaConf.save(cfg, str(config_path))
        metrics_path = artifact_dir / "test_metrics.json"
        save_metrics(test_metrics, metrics_path)

        # Remove checkpoint file since training completed successfully
        checkpoint_path = artifact_dir / "checkpoint.pt"
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info("Removed checkpoint file (training completed)")

        logger.info(f"Saved artifacts to {artifact_dir}")

    return TrainingOutputs(
        best_state_dict=best_state_dict,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        predictions=test_predictions,
        artifact_dir=artifact_dir,
    )


@overload
def evaluate(
    model: SpanBertForQuestionAnswering,
    dataloader: DataLoader,
    examples: List[QAExample],
    features,
    n_best_size: int,
    max_answer_length: int,
    device: torch.device,
    return_predictions: bool = False,
    use_tqdm: bool = True,
    split_name: str = "Eval",
) -> Dict[str, float]: ...


@overload
def evaluate(
    model: SpanBertForQuestionAnswering,
    dataloader: DataLoader,
    examples: List[QAExample],
    features,
    n_best_size: int,
    max_answer_length: int,
    device: torch.device,
    return_predictions: bool = True,
    use_tqdm: bool = True,
    split_name: str = "Eval",
) -> Tuple[Dict[str, float], Dict[str, str]]: ...


def evaluate(
    model: SpanBertForQuestionAnswering,
    dataloader: DataLoader,
    examples: List[QAExample],
    features,
    n_best_size: int,
    max_answer_length: int,
    device: torch.device,
    return_predictions: bool = False,
    use_tqdm: bool = True,
    split_name: str = "Eval",
) -> Union[Dict[str, float], Tuple[Dict[str, float], Dict[str, str]]]:
    """Run model inference and compute QA metrics.

    Args:
        model: The model to evaluate
        dataloader: DataLoader for evaluation data
        examples: List of QAExample instances
        features: List of tokenized features
        n_best_size: Number of top logits to consider
        max_answer_length: Maximum answer span length
        device: Device to run evaluation on
        return_predictions: Whether to return predictions
        use_tqdm: Whether to show progress bar
        split_name: Name of the split being evaluated (for progress bar)

    Returns:
        If return_predictions is False, returns metrics_dict
        If return_predictions is True, returns (metrics_dict, predictions_dict)
    """
    model.eval()
    start_logits: List[List[float]] = []
    end_logits: List[List[float]] = []

    progress_bar = tqdm(
        dataloader,
        desc=f"[{split_name}]",
        disable=not use_tqdm,
        dynamic_ncols=True,
        unit="batch",
    )

    with torch.no_grad():
        for batch in progress_bar:
            inputs = {
                k: v.to(device, non_blocking=True)
                for k, v in batch.items()
                if isinstance(v, torch.Tensor) and k != "example_index"
            }
            outputs = model(**inputs)
            start = outputs["start_logits"].cpu().tolist()
            end = outputs["end_logits"].cpu().tolist()
            start_logits.extend(start)
            end_logits.extend(end)

    if len(start_logits) != len(features):
        logger.warning(
            "Mismatch between features (%d) and logits (%d); check evaluation pipeline.",
            len(features),
            len(start_logits),
        )

    predictions = _postprocess_predictions(
        examples,
        features,
        start_logits,
        end_logits,
        n_best_size,
        max_answer_length,
    )
    references = {ex.example_id: ex.answer_text for ex in examples}
    metrics = aggregate_metrics(predictions, references)

    if return_predictions:
        return metrics, predictions
    return metrics


__all__ = ["run_training", "evaluate", "set_seed", "select_device", "TrainingOutputs"]
