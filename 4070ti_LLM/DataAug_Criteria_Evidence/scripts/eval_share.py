"""Standalone evaluation script for Share architecture.

⚠️ **WARNING: CURRENTLY BLOCKED**
This script requires dual-task ShareDataset which has span alignment issues.
See SHARE_TEST_EVALUATION_BLOCKED.md for details.

This script evaluates a trained Share model (shared encoder + dual heads) on a dataset.
It supports:
- Dual-task evaluation (criteria classification + evidence span extraction)
- Loading from checkpoint
- Comprehensive metrics for both tasks
- Hardware optimization for faster inference

**Status:** Cannot run until dataset span alignment issue is resolved.

Usage (when unblocked):
    # Evaluate best checkpoint on test set
    python scripts/eval_share.py checkpoint=outputs/checkpoints/best_checkpoint.pt

    # Evaluate on custom dataset
    python scripts/eval_share.py checkpoint=path/to/checkpoint.pt dataset.path=data/custom.csv

    # Evaluate with custom batch size
    python scripts/eval_share.py checkpoint=path/to/checkpoint.pt eval_batch_size=64
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

from Project.Share.data.dataset import ShareDataset
from Project.Share.models.model import Model
from psy_agents_noaug.utils.reproducibility import (
    get_device,
    get_optimal_dataloader_kwargs,
    set_seed,
)


def compute_span_f1(pred_starts, pred_ends, true_starts, true_ends):
    """Compute token-level F1 score for span extraction."""
    total_f1 = 0.0
    exact_matches = 0

    for ps, pe, ts, te in zip(pred_starts, pred_ends, true_starts, true_ends):
        # Exact match
        if ps == ts and pe == te:
            exact_matches += 1

        # Token-level F1
        pred_tokens = set(range(ps, pe + 1))
        true_tokens = set(range(ts, te + 1))

        if len(pred_tokens) == 0 or len(true_tokens) == 0:
            f1 = 0.0
        else:
            overlap = len(pred_tokens & true_tokens)
            precision = overlap / len(pred_tokens) if len(pred_tokens) > 0 else 0
            recall = overlap / len(true_tokens) if len(true_tokens) > 0 else 0
            f1 = (
                (2 * precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

        total_f1 += f1

    avg_f1 = total_f1 / len(pred_starts) if len(pred_starts) > 0 else 0.0
    exact_match_ratio = (
        exact_matches / len(pred_starts) if len(pred_starts) > 0 else 0.0
    )

    return {"span_f1": avg_f1, "exact_match": exact_match_ratio}


def load_model_from_checkpoint(
    checkpoint_path: Path,
    cfg: DictConfig,
    device: torch.device,
) -> Model:
    """Load model from checkpoint."""
    print("\n" + "=" * 70)
    print("Loading Model from Checkpoint".center(70))
    print("=" * 70)

    # Create model with dual heads
    model = Model(
        model_name=cfg.model.pretrained_model,
        criteria_num_labels=2,
        criteria_dropout=cfg.model.criteria.get("dropout", 0.1),
        criteria_layer_num=cfg.model.criteria.get("layer_num", 1),
        criteria_hidden_dims=cfg.model.criteria.get("hidden_dims"),
        evidence_dropout=cfg.model.evidence.get("dropout", 0.1),
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

    dataset = ShareDataset(
        csv_path=cfg.dataset.path,
        tokenizer=tokenizer,
        context_column=cfg.dataset.get("context_column", "post_text"),
        answer_column=cfg.dataset.get("answer_column", "sentence_text"),
        label_column=cfg.dataset.get("label_column", "status"),
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
    """Evaluate model and return comprehensive metrics for both tasks."""
    print("\n" + "=" * 70)
    print("Evaluating Model".center(70))
    print("=" * 70)

    model.eval()

    # Criteria task (classification)
    all_criteria_preds = []
    all_criteria_labels = []
    all_criteria_probs = []

    # Evidence task (span extraction)
    all_pred_starts = []
    all_pred_ends = []
    all_true_starts = []
    all_true_ends = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            criteria_labels = batch["labels"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            # Forward pass with AMP
            with autocast(enabled=use_amp and device.type == "cuda"):
                outputs = model(
                    input_ids, attention_mask=attention_mask, return_dict=True
                )

            # Criteria head outputs
            criteria_logits = outputs["logits"]
            criteria_probs = torch.softmax(criteria_logits, dim=-1)
            criteria_preds = torch.argmax(criteria_probs, dim=-1)

            all_criteria_preds.extend(criteria_preds.cpu().numpy())
            all_criteria_labels.extend(criteria_labels.cpu().numpy())
            all_criteria_probs.extend(criteria_probs.cpu().numpy())

            # Evidence head outputs
            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]

            pred_starts = torch.argmax(start_logits, dim=-1)
            pred_ends = torch.argmax(end_logits, dim=-1)

            all_pred_starts.extend(pred_starts.cpu().numpy())
            all_pred_ends.extend(pred_ends.cpu().numpy())
            all_true_starts.extend(start_positions.cpu().numpy())
            all_true_ends.extend(end_positions.cpu().numpy())

    # Convert to numpy arrays
    all_criteria_preds = np.array(all_criteria_preds)
    all_criteria_labels = np.array(all_criteria_labels)
    all_criteria_probs = np.array(all_criteria_probs)

    # Calculate criteria classification metrics
    criteria_accuracy = accuracy_score(all_criteria_labels, all_criteria_preds)
    criteria_f1_macro = f1_score(
        all_criteria_labels, all_criteria_preds, average="macro", zero_division=0
    )
    criteria_f1_micro = f1_score(
        all_criteria_labels, all_criteria_preds, average="micro", zero_division=0
    )
    criteria_f1_binary = f1_score(
        all_criteria_labels, all_criteria_preds, average="binary", zero_division=0
    )
    criteria_precision = precision_score(
        all_criteria_labels, all_criteria_preds, average="macro", zero_division=0
    )
    criteria_recall = recall_score(
        all_criteria_labels, all_criteria_preds, average="macro", zero_division=0
    )

    # Calculate AUROC
    try:
        if all_criteria_probs.shape[1] == 2:
            criteria_auroc = roc_auc_score(all_criteria_labels, all_criteria_probs[:, 1])
        else:
            criteria_auroc = roc_auc_score(
                all_criteria_labels, all_criteria_probs, average="macro", multi_class="ovr"
            )
    except Exception:
        criteria_auroc = None

    # Confusion matrix
    cm = confusion_matrix(all_criteria_labels, all_criteria_preds)

    # Classification report
    class_report = classification_report(
        all_criteria_labels,
        all_criteria_preds,
        target_names=["Negative", "Positive"],
        zero_division=0,
    )

    # Calculate evidence span metrics
    span_metrics = compute_span_f1(
        all_pred_starts, all_pred_ends, all_true_starts, all_true_ends
    )

    # Compute joint score (average of criteria F1 and span F1)
    joint_score = (criteria_f1_macro + span_metrics["span_f1"]) / 2.0

    return {
        # Criteria metrics
        "criteria_accuracy": criteria_accuracy,
        "criteria_f1_macro": criteria_f1_macro,
        "criteria_f1_micro": criteria_f1_micro,
        "criteria_f1_binary": criteria_f1_binary,
        "criteria_precision": criteria_precision,
        "criteria_recall": criteria_recall,
        "criteria_auroc": criteria_auroc,
        "criteria_confusion_matrix": cm.tolist(),
        "criteria_classification_report": class_report,
        # Evidence metrics
        "evidence_span_f1": span_metrics["span_f1"],
        "evidence_exact_match": span_metrics["exact_match"],
        # Joint score
        "joint_score": joint_score,
        "total_samples": len(all_criteria_labels),
    }


@hydra.main(version_base=None, config_path="../configs/share", config_name="train")
def main(cfg: DictConfig):
    """
    Evaluate Share model.

    This script provides:
    - Dual-task evaluation (criteria classification + evidence span)
    - Comprehensive metrics for both tasks
    - Hardware-optimized inference
    - Results export to JSON
    """
    print("\n" + "=" * 70)
    print("SHARE ARCHITECTURE - EVALUATION".center(70))
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

    print("\n" + "─" * 70)
    print("CRITERIA CLASSIFICATION METRICS".center(70))
    print("─" * 70)
    print(f"Accuracy:  {metrics['criteria_accuracy']:.4f}")
    print(f"F1 Macro:  {metrics['criteria_f1_macro']:.4f}")
    print(f"F1 Micro:  {metrics['criteria_f1_micro']:.4f}")
    print(f"F1 Binary: {metrics['criteria_f1_binary']:.4f}")
    print(f"Precision: {metrics['criteria_precision']:.4f}")
    print(f"Recall:    {metrics['criteria_recall']:.4f}")
    if metrics["criteria_auroc"] is not None:
        print(f"AUROC:     {metrics['criteria_auroc']:.4f}")

    print("\nConfusion Matrix:")
    cm = np.array(metrics["criteria_confusion_matrix"])
    print("              Predicted")
    print("              Neg   Pos")
    print(f"Actual  Neg   {cm[0, 0]:4d}  {cm[0, 1]:4d}")
    print(f"        Pos   {cm[1, 0]:4d}  {cm[1, 1]:4d}")

    print("\nClassification Report:")
    print(metrics["criteria_classification_report"])

    print("\n" + "─" * 70)
    print("EVIDENCE SPAN EXTRACTION METRICS".center(70))
    print("─" * 70)
    print(f"Span F1:      {metrics['evidence_span_f1']:.4f}")
    print(f"Exact Match:  {metrics['evidence_exact_match']:.4f}")

    print("\n" + "─" * 70)
    print("JOINT PERFORMANCE".center(70))
    print("─" * 70)
    print(f"Joint Score:  {metrics['joint_score']:.4f}")
    print("(Average of Criteria F1 Macro and Evidence Span F1)")

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
