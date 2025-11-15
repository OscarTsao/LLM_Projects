"""Standalone evaluation script for Criteria architecture.

This script evaluates a trained Criteria model on a dataset.
It supports:
- Loading from checkpoint
- Comprehensive metrics (accuracy, F1, precision, recall, AUROC)
- Confusion matrix and classification report
- Hardware optimization for faster inference

Usage:
    # Evaluate best checkpoint on test set
    python scripts/eval_criteria.py checkpoint=outputs/checkpoints/best_checkpoint.pt

    # Evaluate on custom dataset
    python scripts/eval_criteria.py checkpoint=path/to/checkpoint.pt dataset.path=data/custom.csv

    # Evaluate with custom batch size
    python scripts/eval_criteria.py checkpoint=path/to/checkpoint.pt eval_batch_size=64
"""

import json
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Project.Criteria.data.dataset import CriteriaDataset
from Project.Criteria.models.model import Model
from psy_agents_noaug.utils.reproducibility import (
    get_device,
    get_optimal_dataloader_kwargs,
    set_seed,
)


def load_model_from_checkpoint(
    checkpoint_path: Path,
    cfg: DictConfig,
    device: torch.device,
) -> Model:
    """Load model from checkpoint."""
    print("\n" + "=" * 70)
    print("Loading Model from Checkpoint".center(70))
    print("=" * 70)

    # Create model
    model = Model(
        model_name=cfg.model.pretrained_model,
        num_labels=2,
        classifier_dropout=cfg.model.classifier_dropout,
        classifier_layer_num=cfg.model.classifier_layer_num,
        classifier_hidden_dims=cfg.model.get("classifier_hidden_dims"),
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint from: {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch'] + 1}")
    if "metrics" in checkpoint:
        print(f"Checkpoint metrics: {checkpoint['metrics']}")

    return model


def create_dataloader(
    cfg: DictConfig,
    tokenizer,
    device: torch.device,
) -> DataLoader:
    """Create evaluation data loader."""
    print("\n" + "=" * 70)
    print("Loading Dataset".center(70))
    print("=" * 70)

    dataset = CriteriaDataset(
        csv_path=cfg.dataset.path,
        tokenizer=tokenizer,
        text_column=cfg.dataset.text_column,
        label_column=cfg.dataset.label_column,
        max_length=cfg.dataset.max_length,
    )

    print(f"Total samples: {len(dataset)}")

    # Get optimal DataLoader kwargs
    dataloader_kwargs = get_optimal_dataloader_kwargs(
        device=device,
        num_workers=cfg.dataset.get("num_workers"),
        pin_memory=cfg.dataset.get("pin_memory"),
        persistent_workers=cfg.dataset.get("persistent_workers"),
    )

    return DataLoader(
        dataset,
        batch_size=cfg.get("eval_batch_size", cfg.training.eval_batch_size),
        shuffle=False,
        **dataloader_kwargs,
    )


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> dict:
    """Evaluate model and return comprehensive metrics."""
    print("\n" + "=" * 70)
    print("Evaluating Model".center(70))
    print("=" * 70)

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass with AMP
            with autocast(enabled=use_amp and device.type == "cuda"):
                logits = model(input_ids, attention_mask)

            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average="micro", zero_division=0)
    f1_binary = f1_score(all_labels, all_preds, average="binary", zero_division=0)
    precision_macro = precision_score(
        all_labels, all_preds, average="macro", zero_division=0
    )
    recall_macro = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    # Calculate AUROC
    try:
        if all_probs.shape[1] == 2:
            auroc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            auroc = roc_auc_score(
                all_labels, all_probs, average="macro", multi_class="ovr"
            )
    except Exception:
        auroc = None

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Classification report
    class_report = classification_report(
        all_labels,
        all_preds,
        target_names=["Negative", "Positive"],
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_binary": f1_binary,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "auroc": auroc,
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report,
        "total_samples": len(all_labels),
    }


@hydra.main(version_base=None, config_path="../configs/criteria", config_name="train")
def main(cfg: DictConfig):
    """
    Evaluate Criteria model.

    This script provides:
    - Comprehensive evaluation metrics
    - Confusion matrix and classification report
    - Hardware-optimized inference
    - Results export to JSON
    """
    print("\n" + "=" * 70)
    print("CRITERIA ARCHITECTURE - EVALUATION".center(70))
    print("=" * 70)

    # Check checkpoint path
    if not hasattr(cfg, "checkpoint") or not cfg.checkpoint:
        raise ValueError(
            "Please provide checkpoint path: checkpoint=path/to/checkpoint.pt"
        )

    checkpoint_path = Path(cfg.checkpoint)
    if not checkpoint_path.exists():
        # Try relative to project root
        project_root = Path(get_original_cwd())
        checkpoint_path = project_root / cfg.checkpoint
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint}")

    # Set seed for reproducibility
    set_seed(cfg.training.get("seed", 42), deterministic=True, cudnn_benchmark=False)

    # Get device
    device = get_device(prefer_cuda=True)

    # Create tokenizer
    print("\n" + "=" * 70)
    print("Loading Tokenizer".center(70))
    print("=" * 70)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.pretrained_model)
    print(f"Tokenizer: {cfg.model.pretrained_model}")

    # Load model
    model = load_model_from_checkpoint(checkpoint_path, cfg, device)

    # Create dataloader
    dataloader = create_dataloader(cfg, tokenizer, device)

    # Evaluate
    metrics = evaluate_model(model, dataloader, device, use_amp=True)

    # Print results
    print("\n" + "=" * 70)
    print("Evaluation Results".center(70))
    print("=" * 70)
    print(f"Total samples: {metrics['total_samples']}")
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"F1 Macro:  {metrics['f1_macro']:.4f}")
    print(f"F1 Micro:  {metrics['f1_micro']:.4f}")
    print(f"F1 Binary: {metrics['f1_binary']:.4f}")
    print(f"Precision: {metrics['precision_macro']:.4f}")
    print(f"Recall:    {metrics['recall_macro']:.4f}")
    if metrics["auroc"] is not None:
        print(f"AUROC:     {metrics['auroc']:.4f}")

    print("\nConfusion Matrix:")
    cm = np.array(metrics["confusion_matrix"])
    print("              Predicted")
    print("              Neg   Pos")
    print(f"Actual  Neg   {cm[0, 0]:4d}  {cm[0, 1]:4d}")
    print(f"        Pos   {cm[1, 0]:4d}  {cm[1, 1]:4d}")

    print("\nClassification Report:")
    print(metrics["classification_report"])

    # Save results
    output_dir = Path(cfg.hydra.run.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "evaluation_results.json"

    # Convert numpy types to Python types for JSON serialization
    json_metrics = {
        k: (
            v.tolist()
            if isinstance(v, np.ndarray)
            else float(v) if isinstance(v, np.floating | np.integer) else v
        )
        for k, v in metrics.items()
    }

    with open(results_path, "w") as f:
        json.dump(json_metrics, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
