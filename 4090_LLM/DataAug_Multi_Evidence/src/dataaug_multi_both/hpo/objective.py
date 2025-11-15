"""Optuna objective wiring for two-stage HPO."""

from __future__ import annotations

import logging
import os
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import optuna
import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR
from torch.utils.data import DataLoader, Dataset

from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from dataaug_multi_both.augment import AugmentedDataset, create_augmenter
from dataaug_multi_both.hpo.space import auto_eval_batch_sizes, auto_train_batch_sizes
from dataaug_multi_both.mlflow_buffer import MlflowBuffer
from dataaug_multi_both.optim import Lion
from dataaug_multi_both.training.trainer import EvidenceExtractionTrainer

logger = logging.getLogger(__name__)

HEAD_TYPE_ALIASES = {
    "linear": "start_end_linear",
    "mlp": "start_end_mlp",
}

_SPAWN_INITIALISED = False


def _ensure_spawn_start_method() -> None:
    """Ensure DataLoader workers use spawn to avoid CUDA-in-fork issues."""
    global _SPAWN_INITIALISED
    if _SPAWN_INITIALISED:
        return
    try:
        torch.multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError as exc:
        if "context has already been set" not in str(exc):
            raise
    _SPAWN_INITIALISED = True


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ObjectiveConfig:
    """Configuration bundle used to build the Optuna objective."""

    output_root: Path
    default_model: str = "microsoft/deberta-v3-base"
    dataset_config: Path | None = None
    objective_metric: str = "val_f1"
    seed: int = 42
    use_synthetic: bool = True
    synthetic_train_size: int = 128
    synthetic_val_size: int = 64
    synthetic_seq_len: int = 128
    synthetic_vocab_size: int = 32000
    early_stopping_patience: int = 20
    device: str = _default_device()
    mlflow_buffer: MlflowBuffer | None = None


class SyntheticSpanDataset(Dataset):
    """Tiny randomly generated dataset for smoke-training runs."""

    def __init__(
        self,
        size: int,
        seq_len: int,
        vocab_size: int,
        null_ratio: float = 0.25,
    ) -> None:
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.null_ratio = null_ratio

        self.input_ids = torch.randint(0, vocab_size, (size, seq_len), dtype=torch.long)
        self.attention_mask = torch.ones((size, seq_len), dtype=torch.long)
        self.start_positions = torch.empty(size, dtype=torch.long)
        self.end_positions = torch.empty(size, dtype=torch.long)

        for idx in range(size):
            if random.random() < self.null_ratio:
                self.start_positions[idx] = -1
                self.end_positions[idx] = -1
            else:
                start = random.randint(1, max(1, seq_len // 2))
                end = random.randint(start, seq_len - 1)
                self.start_positions[idx] = start
                self.end_positions[idx] = end

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.size

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
            "start_positions": self.start_positions[index],
            "end_positions": self.end_positions[index],
        }


class ToyEvidenceModel(torch.nn.Module):
    """Lightweight model used for synthetic smoke training."""

    def __init__(self, vocab_size: int = 32000, hidden_size: int = 128) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.encoder = torch.nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.start_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, 1),
        )
        self.end_head = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.embedding(input_ids)
        outputs, _ = self.encoder(embeddings)
        hidden = self.layer_norm(outputs)
        start_logits = self.start_head(hidden).squeeze(-1)
        end_logits = self.end_head(hidden).squeeze(-1)
        return start_logits, end_logits


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():  # pragma: no cover - depends on hardware
        torch.cuda.manual_seed_all(seed)


def _build_optimizer(params: Dict[str, Any], model: torch.nn.Module) -> Optimizer:
    lr_enc = float(params.get("opt_lr_enc", 5e-5))
    lr_head = float(params.get("opt_lr_head", lr_enc))
    weight_decay = float(params.get("opt_wd", 0.01))
    optimizer_name = str(params.get("optim_optimizer", "adamw")).lower()

    encoder_params = list(model.encoder.parameters()) if hasattr(model, "encoder") else []
    head_params = list(model.evidence_head.parameters()) if hasattr(model, "evidence_head") else []
    param_groups: list[dict[str, Any]] = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": lr_enc, "weight_decay": weight_decay})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_head, "weight_decay": weight_decay})
    if not param_groups:
        param_groups.append({"params": model.parameters(), "lr": lr_enc, "weight_decay": weight_decay})

    if optimizer_name == "lion":
        return Lion(param_groups, lr=lr_enc, betas=(0.9, 0.99), weight_decay=weight_decay)

    beta1 = float(params.get("adam_beta1", 0.9))
    beta2 = float(params.get("adam_beta2", 0.999))
    eps = float(params.get("adam_eps", 1e-8))
    return AdamW(param_groups, lr=lr_enc, weight_decay=weight_decay, betas=(beta1, beta2), eps=eps)


def _build_scheduler(
    params: Dict[str, Any],
    optimizer: Optimizer,
    steps_per_epoch: int,
    epochs: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    sched_type = str(params.get("sched_type", "linear")).lower()
    warmup_ratio = float(params.get("sched_warmup_ratio", 0.0))
    warmup_ratio = min(max(warmup_ratio, 0.0), 0.9)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(0, int(total_steps * warmup_ratio))

    if total_steps <= 0:
        return None

    if sched_type == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if sched_type == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if sched_type == "one_cycle":
        steps_per_epoch = max(1, steps_per_epoch)
        pct_start = warmup_ratio if warmup_ratio > 0 else 0.1
        max_lr = max(group.get("lr", optimizer.defaults.get("lr", 1e-3)) for group in optimizer.param_groups)
        return OneCycleLR(
            optimizer,
            max_lr=max_lr,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy="cos",
        )
    return None


def build_objective(cfg: ObjectiveConfig):
    """Return an Optuna objective function bound to ``cfg``."""

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    _ensure_spawn_start_method()

    device = torch.device(cfg.device)

    def objective(trial: optuna.Trial, params: Dict[str, Any], settings) -> float:
        stage_name = settings.stage_name
        trial_seed = abs(hash((cfg.seed, stage_name, trial.number))) % (2**31)
        _set_all_seeds(trial_seed)
        trial.set_user_attr("seed", trial_seed)

        max_length = int(params.get("train_max_length", cfg.synthetic_seq_len))
        max_length = max(8, min(512, max_length))
        train_batch_candidates = auto_train_batch_sizes()
        default_train_batch = max(train_batch_candidates) if train_batch_candidates else 8
        batch_size = max(1, int(params.get("train_batch_size", default_train_batch)))
        eval_batch_candidates = auto_eval_batch_sizes(train_batch_candidates)
        default_eval_batch = max(batch_size, max(eval_batch_candidates) if eval_batch_candidates else batch_size)
        eval_batch_size = max(1, int(params.get("eval_batch_size", default_eval_batch)))
        grad_accum = max(1, int(params.get("train_grad_accum", 1)))
        max_grad_norm = float(params.get("train_grad_clip", 1.0))
        trial.set_user_attr("effective_batch_size", batch_size * grad_accum)
        trial.set_user_attr("train_batch_size", batch_size)
        trial.set_user_attr("eval_batch_size", eval_batch_size)

        amp_precision = str(params.get("train_amp_precision", "bf16")).lower()
        if amp_precision not in {"bf16", "fp16"}:
            use_amp = False
            amp_dtype = None
            amp_precision = "none"
        else:
            use_amp = device.type == "cuda"
            amp_dtype = torch.bfloat16 if amp_precision == "bf16" else torch.float16
        trial.set_user_attr("amp_precision", amp_precision if use_amp else "none")

        gradient_checkpointing = bool(params.get("encoder_gradient_checkpointing", False))
        trial.set_user_attr("gradient_checkpointing", gradient_checkpointing)
        compile_model = bool(params.get("train_compile", False))
        trial.set_user_attr("compile", compile_model)

        start_time = time.time()
        output_dir = cfg.output_root / stage_name / f"trial_{trial.number}"
        output_dir.mkdir(parents=True, exist_ok=True)

        model: torch.nn.Module | None = None

        try:
            augmenter = create_augmenter(params, random.Random(trial_seed))

            if cfg.use_synthetic:
                train_dataset = SyntheticSpanDataset(
                    size=cfg.synthetic_train_size,
                    seq_len=max_length,
                    vocab_size=cfg.synthetic_vocab_size,
                )
                val_dataset = SyntheticSpanDataset(
                    size=cfg.synthetic_val_size,
                    seq_len=max_length,
                    vocab_size=cfg.synthetic_vocab_size,
                )
                collate_fn = None
                model = ToyEvidenceModel(vocab_size=cfg.synthetic_vocab_size)
            else:
                from dataaug_multi_both.data import (
                    DatasetConfigurationError,
                    DatasetLoader,
                    build_dataset_config_from_dict,
                    create_collator,
                )
                from dataaug_multi_both.models import EvidenceExtractionModel
                import yaml

                model_name = params.get("encoder_model_name", cfg.default_model)
                head_type = params.get("head_ev_type", "start_end_linear")
                head_type = HEAD_TYPE_ALIASES.get(head_type, head_type)
                model = EvidenceExtractionModel(
                    model_name_or_path=model_name,
                    head_type=head_type,
                    dropout=float(params.get("head_ev_dropout", 0.1)),
                    gradient_checkpointing=gradient_checkpointing,
                )

                if cfg.dataset_config and cfg.dataset_config.exists():
                    with open(cfg.dataset_config, encoding="utf-8") as f:
                        data_yaml = yaml.safe_load(f) or {}
                    ds_section = data_yaml.get("dataset")
                    if not isinstance(ds_section, dict):
                        raise DatasetConfigurationError(
                            f"Dataset config {cfg.dataset_config} does not define a 'dataset' mapping."
                        )
                    ds_cfg = build_dataset_config_from_dict(
                        ds_section,
                        config_dir=None,
                    )
                    loader = DatasetLoader()
                    splits = loader.load(ds_cfg)
                    train_split = splits["train"]
                    val_split = splits["validation"]
                    if hasattr(train_split, "with_format"):
                        train_dataset = train_split.with_format("python")
                    else:
                        train_dataset = train_split
                    if hasattr(val_split, "with_format"):
                        val_dataset = val_split.with_format("python")
                    else:
                        val_dataset = val_split
                else:
                    raise RuntimeError("Real dataset requested but no dataset config available")

                collate_fn = create_collator(model_name_or_path=model_name, max_length=max_length)

            if augmenter and not cfg.use_synthetic:
                try:
                    sample = train_dataset[0]
                except Exception:
                    sample = None
                if isinstance(sample, dict) and "sentence_text" in sample:
                    train_dataset = AugmentedDataset(train_dataset, augmenter, field="sentence_text")

            requested_workers = params.get("resources_num_workers")
            cpu_count = os.cpu_count() or 4
            if requested_workers is None:
                num_workers = max(4, cpu_count - 2)
            else:
                num_workers = max(0, int(requested_workers))
            pin_memory = device.type == "cuda"

            train_loader_kwargs = dict(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            val_loader_kwargs = dict(
                dataset=val_dataset,
                batch_size=eval_batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            if num_workers > 0:
                prefetch_factor = max(2, min(8, num_workers * 2))
                train_loader_kwargs["prefetch_factor"] = prefetch_factor
                train_loader_kwargs["persistent_workers"] = True
                val_loader_kwargs["prefetch_factor"] = prefetch_factor
                val_loader_kwargs["persistent_workers"] = True

            train_loader = DataLoader(**train_loader_kwargs)
            val_loader = DataLoader(**val_loader_kwargs)

            model.to(device)
            if compile_model and hasattr(torch, "compile"):
                try:
                    model = torch.compile(model)  # type: ignore[attr-defined]
                except Exception as exc:  # pragma: no cover - optional optimisation
                    logger.warning("torch.compile failed; continuing without compilation: %s", exc)

            optimizer = _build_optimizer(params, model)
            steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
            scheduler = _build_scheduler(params, optimizer, steps_per_epoch, settings.epochs)
            scheduler_step_per_batch = False
            if scheduler is not None:
                trial.set_user_attr("scheduler", scheduler.__class__.__name__)
                if isinstance(scheduler, (LambdaLR, OneCycleLR)):
                    scheduler_step_per_batch = True

            trainer = EvidenceExtractionTrainer(
                model=model,
                optimizer=optimizer,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                output_dir=output_dir,
                gradient_accumulation_steps=grad_accum,
                max_grad_norm=max_grad_norm,
                early_stopping_patience=cfg.early_stopping_patience,
                metric_for_best_model=cfg.objective_metric,
                mlflow_tracking=False,
                use_amp=use_amp,
                amp_dtype=amp_dtype,
                scheduler=scheduler,
                scheduler_step_per_batch=scheduler_step_per_batch,
            )

            report_metric = cfg.objective_metric.replace("val_", "")
            result = trainer.train(
                num_epochs=settings.epochs,
                trial=trial,
                report_metric=report_metric,
            )

            duration = time.time() - start_time
            best_metric = float(result.get("best_metric", 0.0))
            trial.set_user_attr("duration_seconds", duration)
            if cfg.mlflow_buffer is not None:
                tags = {
                    "trial.stage": stage_name,
                    "trial.number": trial.number,
                    "trial.seed": trial_seed,
                    "trial.gradient_checkpointing": gradient_checkpointing,
                    "trial.compile": compile_model,
                    "trial.amp_precision": amp_precision if use_amp else "none",
                }
                cfg.mlflow_buffer.log_tags(tags)
                cfg.mlflow_buffer.log_metric("trial.duration_seconds", duration)
                cfg.mlflow_buffer.log_metric("trial.best_metric", best_metric)
            return best_metric

        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message:
                logger.warning("Trial %s pruned due to OOM: %s", trial.number, exc)
                if torch.cuda.is_available():  # pragma: no cover - depends on hardware
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned(f"OOM: {exc}") from exc
            trial.set_user_attr("failure", str(exc))
            if cfg.mlflow_buffer is not None:
                cfg.mlflow_buffer.log_tags({"trial.status": "failed", "trial.error": str(exc)})
            raise
        except Exception as exc:
            message = str(exc).lower()
            if "cuda" in message:
                logger.warning("Trial %s pruned due to CUDA failure: %s", trial.number, exc)
                if torch.cuda.is_available():  # pragma: no cover
                    torch.cuda.empty_cache()
                raise optuna.TrialPruned(f"CUDA failure: {exc}") from exc
            trial.set_user_attr("failure", str(exc))
            if cfg.mlflow_buffer is not None:
                cfg.mlflow_buffer.log_tags({"trial.status": "failed", "trial.error": str(exc)})
            raise
        except optuna.TrialPruned as exc:
            if cfg.mlflow_buffer is not None:
                cfg.mlflow_buffer.log_tags({"trial.status": "pruned"})
            raise
        finally:
            if model is not None:
                try:
                    model.to("cpu")
                except Exception:  # pragma: no cover - best effort cleanup
                    pass
            if torch.cuda.is_available():  # pragma: no cover
                torch.cuda.empty_cache()

    return objective


__all__ = ["ObjectiveConfig", "build_objective"]
