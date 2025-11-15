#!/usr/bin/env python
"""K-fold training utility for the Criteria architecture.

This script reuses the production trainer to run Stratified K-fold
experiments (default: 5 folds) for a chosen HuggingFace encoder.
Each fold trains a fresh model with early stopping, evaluates on the
held-out fold, and logs metrics to MLflow plus a JSON summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, Subset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
DEFAULT_CFG_PATH = PROJECT_ROOT / "configs" / "criteria" / "train.yaml"

from transformers import AutoTokenizer

# Local imports
from Project.Criteria.data.dataset import CriteriaDataset
from Project.Criteria.models.model import Model
from psy_agents_noaug.training.train_loop import Trainer
from psy_agents_noaug.utils.mlflow_utils import (
    configure_mlflow,
    log_config,
    resolve_artifact_location,
    resolve_tracking_uri,
)
from psy_agents_noaug.utils.reproducibility import (
    get_device,
    get_optimal_dataloader_kwargs,
    print_system_info,
    set_seed,
)
from scripts.train_criteria import create_optimizer, create_scheduler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="K-fold trainer for Criteria task.")
    parser.add_argument(
        "--model-name",
        required=True,
        help="HuggingFace encoder identifier (e.g., mnaylor/psychbert-finetuned-mentalhealth)",
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of StratifiedKFold splits (default: 5)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Maximum epochs per fold (default: 15)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="Early stopping patience (default: 4)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of training subset reserved for validation (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--outdir",
        default=PROJECT_ROOT / "outputs" / "criteria_kfold",
        type=Path,
        help="Directory to store fold artifacts (default: outputs/criteria_kfold)",
    )
    parser.add_argument(
        "--mlflow-uri",
        default=None,
        help="Optional override for MLflow tracking URI",
    )
    parser.add_argument(
        "--artifact-location",
        default=None,
        help="Optional override for MLflow artifact root",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CFG_PATH,
        type=Path,
        help="Path to base Hydra config (default: configs/criteria/train.yaml)",
    )
    return parser.parse_args()


@dataclass
class FoldMetrics:
    fold: int
    best_epoch: int
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    run_id: str


def sanitize_model_name(model_name: str) -> str:
    """Convert HF identifiers into filesystem-safe slugs."""
    return (
        model_name.strip()
        .replace("/", "_")
        .replace(":", "_")
        .replace(" ", "-")
        .lower()
    )


def load_config(path: Path) -> Any:
    cfg = OmegaConf.load(path)
    OmegaConf.set_struct(cfg, False)
    if "hydra" in cfg:
        del cfg["hydra"]
    return cfg


def prepare_dataset(cfg, tokenizer) -> CriteriaDataset:
    return CriteriaDataset(
        csv_path=cfg.dataset.path,
        tokenizer=tokenizer,
        text_column=cfg.dataset.text_column,
        label_column=cfg.dataset.label_column,
        max_length=cfg.dataset.max_length,
    )


def build_dataloader(
    dataset: CriteriaDataset,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    dataloader_kwargs: dict[str, Any],
) -> DataLoader:
    subset = Subset(dataset, indices.tolist())
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        **dataloader_kwargs,
    )


def evaluate_on_loader(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    preds: list[int] = []
    labels: list[int] = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            batch_preds = torch.argmax(logits, dim=-1)

            preds.extend(batch_preds.cpu().tolist())
            labels.extend(batch_labels.cpu().tolist())

    accuracy = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    f1_micro = f1_score(labels, preds, average="micro")
    precision = precision_score(labels, preds, average="macro", zero_division=0)
    recall = recall_score(labels, preds, average="macro", zero_division=0)

    return {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_micro": float(f1_micro),
        "precision": float(precision),
        "recall": float(recall),
    }


def aggregate_metrics(results: list[FoldMetrics], key: str) -> dict[str, dict[str, float]]:
    metrics = {}
    for fold in results:
        for metric, value in getattr(fold, key).items():
            metrics.setdefault(metric, []).append(value)

    summary = {}
    for metric, values in metrics.items():
        summary[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return summary


def run_fold(
    fold_idx: int,
    cfg,
    dataset: CriteriaDataset,
    labels: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    dataloader_kwargs: dict[str, Any],
    device: torch.device,
    outdir: Path,
    args: argparse.Namespace,
    model_slug: str,
) -> FoldMetrics:
    # Split training indices into train/val for early stopping
    val_fraction = args.val_ratio
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_ratio must be in (0, 1).")

    strat_labels = labels[train_indices]
    train_idx, val_idx = train_test_split(
        train_indices,
        test_size=val_fraction,
        random_state=args.seed + fold_idx,
        stratify=strat_labels,
    )

    fold_dir = outdir / model_slug / f"fold_{fold_idx}"
    ckpt_dir = fold_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Fold {fold_idx}/{args.folds} ===")
    print(f"Train size: {len(train_idx)}, Val size: {len(val_idx)}, Test size: {len(test_indices)}")

    train_loader = build_dataloader(
        dataset, train_idx, cfg.training.train_batch_size, True, dataloader_kwargs
    )
    val_loader = build_dataloader(
        dataset, val_idx, cfg.training.eval_batch_size, False, dataloader_kwargs
    )
    test_loader = build_dataloader(
        dataset, test_indices, cfg.training.eval_batch_size, False, dataloader_kwargs
    )

    model = Model(
        model_name=cfg.model.pretrained_model,
        num_labels=2,
        classifier_dropout=cfg.model.classifier_dropout,
        classifier_layer_num=cfg.model.classifier_layer_num,
        classifier_hidden_dims=cfg.model.get("classifier_hidden_dims"),
    ).to(device)

    optimizer = create_optimizer(cfg, model)
    num_training_steps = math.ceil(len(train_loader)) * cfg.training.epochs
    scheduler = create_scheduler(cfg, optimizer, num_training_steps)
    criterion = nn.CrossEntropyLoss()

    tracking_uri = resolve_tracking_uri(
        args.mlflow_uri or cfg.mlflow.tracking_uri, PROJECT_ROOT
    )
    artifact_location = resolve_artifact_location(
        args.artifact_location or cfg.mlflow.get("artifact_location"), PROJECT_ROOT
    )

    experiment_name = f"{cfg.project}-kfold"
    run_name = f"{model_slug}_fold{fold_idx}"
    run_id = configure_mlflow(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        tags={
            "model": cfg.model.pretrained_model,
            "fold": str(fold_idx),
            "kfold": str(args.folds),
        },
        config=cfg,
        artifact_location=artifact_location,
    )
    log_config(cfg)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=cfg.training.epochs,
        patience=cfg.training.patience,
        gradient_clip=cfg.training.max_grad_norm,
        gradient_accumulation_steps=cfg.training.gradient_accumulation,
        scheduler=scheduler,
        save_dir=ckpt_dir,
        use_amp=True,
        amp_dtype="float16",
        early_stopping_metric=cfg.training.monitor_metric,
        early_stopping_mode=cfg.training.monitor_mode,
        min_delta=cfg.training.get("min_delta", 0.0001),
        logging_steps=cfg.training.logging_steps,
    )

    trainer.train()

    # Load best checkpoint for evaluation
    best_ckpt = ckpt_dir / "best_checkpoint.pt"
    if not best_ckpt.exists():
        raise FileNotFoundError(f"Best checkpoint missing for fold {fold_idx}: {best_ckpt}")

    checkpoint = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics = checkpoint.get("metrics", {})
    best_epoch = checkpoint.get("epoch", 0) + 1

    val_metrics = {
        key: float(value)
        for key, value in metrics.items()
        if key.startswith("val_") or key in {"train_loss", "epoch_time"}
    }
    val_metrics["best_metric"] = float(
        metrics.get(cfg.training.monitor_metric, 0.0)
    )
    val_metrics["epoch"] = best_epoch

    test_metrics = evaluate_on_loader(model, test_loader, device)
    mlflow.log_metrics(
        {
            f"fold{fold_idx}_test_accuracy": test_metrics["accuracy"],
            f"fold{fold_idx}_test_f1_macro": test_metrics["f1_macro"],
            f"fold{fold_idx}_test_f1_micro": test_metrics["f1_micro"],
            f"fold{fold_idx}_test_precision": test_metrics["precision"],
            f"fold{fold_idx}_test_recall": test_metrics["recall"],
        }
    )
    mlflow.end_run()

    print(
        f"Fold {fold_idx} complete. "
        f"Val F1 macro: {val_metrics.get('val_f1_macro', 0.0):.4f}, "
        f"Test F1 macro: {test_metrics['f1_macro']:.4f}"
    )

    return FoldMetrics(
        fold=fold_idx,
        best_epoch=best_epoch,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        run_id=run_id,
    )


def save_summary(
    model_slug: str,
    results: list[FoldMetrics],
    outdir: Path,
    cfg,
    args: argparse.Namespace,
) -> Path:
    summary = {
        "model_name": cfg.model.pretrained_model,
        "model_slug": model_slug,
        "folds": args.folds,
        "epochs": cfg.training.epochs,
        "patience": cfg.training.patience,
        "val_ratio": args.val_ratio,
        "results": [
            {
                "fold": fold.fold,
                "best_epoch": fold.best_epoch,
                "val_metrics": fold.val_metrics,
                "test_metrics": fold.test_metrics,
                "run_id": fold.run_id,
            }
            for fold in results
        ],
        "val_summary": aggregate_metrics(results, "val_metrics"),
        "test_summary": aggregate_metrics(results, "test_metrics"),
    }

    summary_path = outdir / model_slug / "kfold_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as fp:
        json.dump(summary, fp, indent=2)
    return summary_path


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    cfg.model.pretrained_model = args.model_name
    cfg.training.epochs = args.epochs
    cfg.training.patience = args.patience

    print_system_info()
    set_seed(
        args.seed,
        deterministic=cfg.training.get("deterministic", True),
        cudnn_benchmark=cfg.training.get("cudnn_benchmark", False),
    )
    device = get_device(prefer_cuda=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model)
    dataset = prepare_dataset(cfg, tokenizer)
    labels = np.array(
        [int(example[cfg.dataset.label_column]) for example in dataset.examples]
    )

    dataloader_kwargs = get_optimal_dataloader_kwargs(
        device=device,
        num_workers=cfg.dataset.get("num_workers"),
        pin_memory=cfg.dataset.get("pin_memory"),
        persistent_workers=cfg.dataset.get("persistent_workers"),
    )

    skf = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )
    model_slug = sanitize_model_name(args.model_name)
    outdir = Path(args.outdir)

    fold_results: list[FoldMetrics] = []
    indices = np.arange(len(labels))
    for fold_idx, (train_indices, test_indices) in enumerate(
        skf.split(indices, labels), start=1
    ):
        result = run_fold(
            fold_idx=fold_idx,
            cfg=cfg,
            dataset=dataset,
            labels=labels,
            train_indices=train_indices,
            test_indices=test_indices,
            dataloader_kwargs=dataloader_kwargs,
            device=device,
            outdir=outdir,
            args=args,
            model_slug=model_slug,
        )
        fold_results.append(result)

    summary_path = save_summary(model_slug, fold_results, outdir, cfg, args)
    print(f"\nK-fold summary saved to: {summary_path}")
    print("Validation summary (mean ± std):")
    for metric, stats in aggregate_metrics(fold_results, "val_metrics").items():
        print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print("Test summary (mean ± std):")
    for metric, stats in aggregate_metrics(fold_results, "test_metrics").items():
        print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")


if __name__ == "__main__":
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    main()
