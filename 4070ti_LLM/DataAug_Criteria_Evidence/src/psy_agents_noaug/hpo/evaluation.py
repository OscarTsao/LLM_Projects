"""Training + evaluation bridge for Optuna trials."""

from __future__ import annotations

import math
import os
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, log_loss
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)

try:  # Optional dependency in slim environments
    from transformers.optimization import Adafactor
except Exception:  # pragma: no cover - optional import
    Adafactor = None

from ..training.optimizers import create_optimizer
from .null_policy import apply_null_policy, temperature_scale
from .utils import limit_dataframe, set_global_seed
from ..augmentation.pipeline import AugConfig, AugmenterPipeline, AugResources


def _extract_augmentation_config(params: dict[str, Any]) -> dict[str, Any] | None:
    """Extract augmentation configuration from HPO parameters.

    Returns None if augmentation is disabled, otherwise returns a dict with:
    - enabled (bool): Whether augmentation is active
    - p_apply (float): Probability of applying augmentation to a sample
    - ops_per_sample (int): Number of augmentation operations per sample
    - max_replace (float): Maximum fraction of tokens to replace
    - antonym_guard (str): Antonym guard strategy
    - method_strategy (str): Method selection strategy
    - tfidf_model (str | None): Path to TF-IDF cache (if available)
    """
    if not params.get("aug.enabled", False):
        return None

    aug_config = {
        "enabled": True,
        "p_apply": float(params.get("aug.p_apply", 0.15)),
        "ops_per_sample": int(params.get("aug.ops_per_sample", 1)),
        "max_replace": float(params.get("aug.max_replace", 0.3)),
        "antonym_guard": params.get("aug.antonym_guard", "off"),
        "method_strategy": params.get("aug.method_strategy", "all"),
        "tfidf_model": None,  # Will be set from environment or config
    }

    # Check for TF-IDF cache path
    tfidf_cache = Path(
        f"data/augmentation_cache/tfidf/{params.get('agent', 'criteria')}"
    )
    if tfidf_cache.exists():
        aug_config["tfidf_model"] = str(tfidf_cache)

    return aug_config


@dataclass
class EvaluationResult:
    """Holds metrics and raw outputs for a single seed."""

    metrics: dict[str, float]
    probs: np.ndarray
    labels: np.ndarray
    runtime_s: float


class TokenizedDataset(Dataset):
    """Pre-tokenised dataset for classification."""

    def __init__(
        self,
        texts: Iterable[str],
        labels: Iterable[int],
        tokenizer,
        max_length: int,
    ) -> None:
        texts = list(texts)
        labels = list(labels)
        encodings = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return self.labels.numel()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
            "labels": self.labels[idx],
        }


class AugmentedTokenizedDataset(Dataset):
    """Augmentation-aware dataset for classification.

    Applies text augmentation on-the-fly during training using AugmenterPipeline.
    This enables HPO to actually test different augmentation strategies.
    """

    def __init__(
        self,
        texts: Iterable[str],
        labels: Iterable[int],
        tokenizer,
        max_length: int,
        augmenter: AugmenterPipeline | None = None,
    ) -> None:
        self.texts = list(texts)
        self.labels = torch.tensor(list(labels), dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self.texts[idx]

        # Apply augmentation if enabled
        if self.augmenter is not None:
            text = self.augmenter(text)

        # Tokenize on-the-fly (allows augmentation to affect tokenization)
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": self.labels[idx],
        }


def _load_dataframe(agent: str) -> pd.DataFrame:
    dataset_path = Path("data/redsm5/redsm5_annotations.csv")
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset file missing: data/redsm5/redsm5_annotations.csv. "
            "Generate it with `make groundtruth`."
        )

    df = pd.read_csv(dataset_path)
    if "status" in df.columns:
        df["label"] = df["status"].astype(int)
    df = df.dropna(subset=["sentence_text", "label"])

    # Provide slight variation for multi-task agents without requiring
    # dedicated datasets in the smoke-test flows.
    if agent in {"share", "joint"} and "DSM5_symptom" in df.columns:
        df = df.copy()
        df["label"] = df["label"].astype(int) | (
            df["DSM5_symptom"].astype("category").cat.codes % 2
        )

    return df[["sentence_text", "label"]]


def _build_optimizer(
    model: nn.Module,
    name: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Build optimizer using unified optimizer factory (SUPERMAX Phase 4).

    Supports 6 optimizers: adamw, adam, adafactor, lion, lamb, adamw_8bit.
    Factory handles missing dependencies gracefully with fallbacks.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    return create_optimizer(
        name=name,
        model_parameters=params,
        lr=lr,
        weight_decay=weight_decay,
    )


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    sched_name: str,
    total_steps: int,
    warmup_ratio: float,
    epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    sched_name = sched_name.lower()
    warmup_steps = int(total_steps * warmup_ratio)
    if sched_name == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if sched_name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    if sched_name == "cosine_restart":
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
    if sched_name == "polynomial":
        return get_polynomial_decay_schedule_with_warmup(
            optimizer, warmup_steps, total_steps
        )
    if sched_name == "one_cycle" and total_steps > 0:
        # OneCycleLR requires total_steps > warmup_steps to avoid division by zero
        if total_steps <= warmup_steps:
            # Fallback to linear schedule when steps are too small
            return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        # Check if optimizer supports momentum (required for cycle_momentum)
        # Adafactor and some other optimizers don't support momentum
        supports_momentum = any(
            'betas' in group or 'momentum' in group
            for group in optimizer.param_groups
        )

        try:
            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=max(group["lr"] for group in optimizer.param_groups),
                total_steps=total_steps,
                pct_start=max(1, warmup_steps) / max(1, total_steps),
                anneal_strategy="cos",
                cycle_momentum=supports_momentum,  # Disable for optimizers without momentum
            )
        except (ValueError, ZeroDivisionError) as e:
            # Fallback to linear schedule if OneCycleLR fails
            return get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    return None


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    digitized = np.digitize(probs, bins, right=True)
    ece = 0.0
    for i in range(1, len(bins)):
        mask = digitized == i
        if not mask.any():
            continue
        bin_accuracy = labels[mask].mean()
        bin_confidence = probs[mask].mean()
        ece += (mask.sum() / probs.size) * abs(bin_confidence - bin_accuracy)
    return float(ece)


def _train_single_seed(
    agent: str,
    params: dict[str, Any],
    *,
    seed: int,
    epochs: int,
    patience: int,
    max_samples: int | None,
    device: torch.device,
) -> EvaluationResult:
    set_global_seed(seed)
    df = limit_dataframe(_load_dataframe(agent), max_samples, seed)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=seed,
        stratify=df["label"],
    )

    tokenizer = AutoTokenizer.from_pretrained(params["model.name"], use_fast=True)
    max_length = int(params["tok.max_length"])

    # SUPERMAX Phase 3: Extract augmentation config from HPO parameters
    # Create AugmenterPipeline if augmentation is enabled
    aug_config = _extract_augmentation_config(params)
    augmenter = None
    if aug_config and aug_config["enabled"]:
        try:
            # Create augmentation configuration
            aug_cfg = AugConfig(
                enabled=True,
                p_apply=aug_config["p_apply"],
                ops_per_sample=aug_config["ops_per_sample"],
                max_replace=aug_config["max_replace"],
                antonym_guard=aug_config["antonym_guard"],
                methods=["all"] if aug_config["method_strategy"] == "all" else [],
                seed=seed,
            )

            # Create augmentation resources if TF-IDF model exists
            resources = AugResources()
            if aug_config.get("tfidf_model"):
                resources.tfidf_model_path = aug_config["tfidf_model"]

            # Initialize augmenter pipeline
            augmenter = AugmenterPipeline(aug_cfg, resources)

        except Exception as exc:
            # Silently fall back to no augmentation if pipeline fails
            # This prevents HPO trials from crashing due to augmentation issues
            augmenter = None

    # Use augmentation-aware dataset if augmenter exists, otherwise plain dataset
    if augmenter is not None:
        train_dataset = AugmentedTokenizedDataset(
            train_df["sentence_text"], train_df["label"], tokenizer, max_length, augmenter
        )
    else:
        train_dataset = TokenizedDataset(
            train_df["sentence_text"], train_df["label"], tokenizer, max_length
        )

    # Validation dataset never uses augmentation
    val_dataset = TokenizedDataset(
        val_df["sentence_text"], val_df["label"], tokenizer, max_length
    )

    batch_size = int(params["train.batch_size"])
    grad_accum = int(params["train.grad_accum"])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size * 2, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        params["model.name"],
        num_labels=2,
        problem_type="single_label_classification",
    )
    grad_ckpt_enabled = bool(params.get("model.gradient_checkpointing"))
    if grad_ckpt_enabled and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except ValueError as e:
            # Some models (XLNet, GPT-2, etc.) don't support gradient checkpointing
            # Silently skip and continue without it
            pass
    model.to(device)

    optimizer = _build_optimizer(
        model,
        params["optim.name"],
        float(params["optim.lr"]),
        float(params["optim.weight_decay"]),
    )

    total_steps = math.ceil(len(train_loader) / max(1, grad_accum)) * max(1, epochs)
    scheduler = _build_scheduler(
        optimizer,
        params["sched.name"],
        total_steps,
        float(params["sched.warmup_ratio"]),
        epochs,
    )

    criterion = nn.CrossEntropyLoss(
        label_smoothing=float(params["reg.label_smoothing"])
    )
    amp_enabled = bool(params.get("train.amp", True)) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    max_grad_norm = float(params["reg.max_grad_norm"])

    best_state: dict[str, Any] | None = None
    best_f1 = -float("inf")
    epochs_without_improvement = 0

    start_time = time.perf_counter()

    for epoch in range(max(1, epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = criterion(outputs.logits, batch["labels"])
                loss = loss / grad_accum

            retain_graph = grad_ckpt_enabled or ((step + 1) % grad_accum != 0)
            if amp_enabled:
                scaler.scale(loss).backward(retain_graph=retain_graph)
            else:
                loss.backward(retain_graph=retain_graph)

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                if amp_enabled:
                    scaler.unscale_(optimizer)
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                if amp_enabled:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        # Validation pass
        model.eval()
        val_logits: list[torch.Tensor] = []
        val_labels: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits
                val_logits.append(logits.cpu())
                val_labels.append(batch["labels"].cpu())

        logits_tensor = torch.cat(val_logits, dim=0)
        labels_tensor = torch.cat(val_labels, dim=0)
        probs = torch.softmax(logits_tensor, dim=1)[:, 1].numpy()
        labels_np = labels_tensor.numpy()
        preds = apply_null_policy(
            probs,
            strategy=params["null.strategy"],
            threshold=float(params["null.threshold"]),
            ratio=float(params["null.ratio"]),
        )
        f1_val = f1_score(labels_np, preds, average="macro")

        if f1_val > best_f1 + 1e-4:
            best_f1 = f1_val
            epochs_without_improvement = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation with temperature scaling
    model.eval()
    logits_eval: list[torch.Tensor] = []
    labels_eval: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            ).logits
            logits = temperature_scale(logits, float(params["null.temperature"]))
            logits_eval.append(logits.cpu())
            labels_eval.append(batch["labels"].cpu())

    logits_final = torch.cat(logits_eval, dim=0)
    labels_final = torch.cat(labels_eval, dim=0)
    probs = torch.softmax(logits_final, dim=1)[:, 1].numpy()
    probs = np.clip(probs, 1e-6, 1 - 1e-6)
    labels_np = labels_final.numpy()
    preds = apply_null_policy(
        probs,
        strategy=params["null.strategy"],
        threshold=float(params["null.threshold"]),
        ratio=float(params["null.ratio"]),
    )

    f1_macro = f1_score(labels_np, preds, average="macro")
    try:
        logloss_value = log_loss(
            labels_np,
            np.stack([1 - probs, probs], axis=1),
        )
    except ValueError:
        logloss_value = float("nan")

    ece = expected_calibration_error(probs, labels_np)
    runtime = time.perf_counter() - start_time

    metrics = {
        "f1_macro": float(f1_macro),
        "ece": float(ece),
        "logloss": float(logloss_value),
        "runtime_s": float(runtime),
    }
    return EvaluationResult(
        metrics=metrics, probs=probs, labels=labels_np, runtime_s=runtime
    )


def run_experiment(
    agent: str,
    params: dict[str, Any],
    *,
    epochs: int,
    seeds: list[int],
    patience: int,
    max_samples: int | None,
    device: torch.device | None = None,
) -> dict[str, float]:
    """Execute training/evaluation across ``seeds`` and average metrics."""

    if os.getenv("HPO_SMOKE_MODE") == "1":
        # Ultra-fast deterministic path used in unit tests.
        return {
            "f1_macro": 0.5,
            "ece": 0.2,
            "logloss": 0.7,
            "runtime_s": 0.0,
        }

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results: list[EvaluationResult] = []
    for seed in seeds:
        result = _train_single_seed(
            agent,
            params,
            seed=seed,
            epochs=epochs,
            patience=patience,
            max_samples=max_samples,
            device=device,
        )
        results.append(result)

    if not results:
        raise RuntimeError("No evaluation results collected")

    aggregate: dict[str, float] = {}
    for key in results[0].metrics:
        aggregate[key] = float(np.nanmean([res.metrics[key] for res in results]))

    return aggregate
