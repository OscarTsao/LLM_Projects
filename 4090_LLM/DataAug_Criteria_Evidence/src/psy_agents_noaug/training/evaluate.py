"""Comprehensive evaluation utilities with task-specific metrics."""

import json
from pathlib import Path

import numpy as np
import torch
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
from torch.utils.data import DataLoader
from tqdm import tqdm


class Evaluator:
    """
    Comprehensive model evaluator with task-specific metrics.

    Supports:
    - Criteria task: F1 (macro), Accuracy, AUROC (macro), per-criterion F1
    - Evidence task: F1 (micro/macro) at sentence level, exact/partial match
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        task_type: str = "criteria",
        criterion: nn.Module | None = None,
    ):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            device: Device to run evaluation on
            task_type: Task type (criteria or evidence)
            criterion: Optional loss function
        """
        self.model = model
        self.device = device
        self.task_type = task_type
        self.criterion = criterion

    def evaluate(
        self,
        data_loader: DataLoader,
        class_names: list[str] | None = None,
    ) -> dict:
        """
        Evaluate model on a dataset with comprehensive metrics.

        Args:
            data_loader: Data loader for evaluation
            class_names: Optional list of class names

        Returns:
            Dictionary with evaluation metrics
        """
        # Switch to eval mode to disable dropout/batchnorm noise
        self.model.eval()

        all_predictions = []
        all_labels = []
        all_probabilities = []
        all_criterion_ids: list[str] = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", leave=False):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                logits = self.model(input_ids, attention_mask)

                # Compute loss if criterion provided
                if self.criterion:
                    loss = self.criterion(logits, labels)
                    total_loss += loss.item()

                # Get predictions and probabilities
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                # Store predictions, labels, and probabilities
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                criterion_ids_batch = batch.get("criterion_id")
                if criterion_ids_batch is not None:
                    if isinstance(criterion_ids_batch, torch.Tensor):
                        all_criterion_ids.extend(
                            [str(x) for x in criterion_ids_batch.cpu().tolist()]
                        )
                    elif isinstance(criterion_ids_batch, list):
                        all_criterion_ids.extend([str(x) for x in criterion_ids_batch])

        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        probabilities = (
            np.array(all_probabilities) if all_probabilities else np.empty((0, 0))
        )

        # Compute metrics based on task type. The criteria/evidence branches
        # include task‑specific extras (e.g., per‑criterion F1, evidence match).
        if self.task_type == "criteria":
            metrics = self._compute_criteria_metrics(
                labels, predictions, probabilities, class_names, all_criterion_ids
            )
        elif self.task_type == "evidence":
            metrics = self._compute_evidence_metrics(
                labels, predictions, probabilities, class_names, all_criterion_ids
            )
        else:
            metrics = self._compute_basic_metrics(
                labels, predictions, probabilities, class_names, all_criterion_ids
            )

        # Add average loss if a criterion was provided
        if self.criterion:
            metrics["loss"] = total_loss / max(len(data_loader), 1)

        return metrics

    def _compute_criteria_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        class_names: list[str] | None = None,
        criterion_ids: list[str] | None = None,
    ) -> dict:
        """Compute criteria-specific metrics."""
        metrics = {}

        metrics["accuracy"] = accuracy_score(labels, predictions)
        metrics["precision_macro"] = precision_score(
            labels, predictions, average="macro", zero_division=0
        )
        metrics["recall_macro"] = recall_score(
            labels, predictions, average="macro", zero_division=0
        )
        metrics["f1_macro"] = f1_score(
            labels, predictions, average="macro", zero_division=0
        )
        metrics["f1_micro"] = f1_score(
            labels, predictions, average="micro", zero_division=0
        )
        metrics["f1_weighted"] = f1_score(
            labels, predictions, average="weighted", zero_division=0
        )

        # AUROC (macro) for multi-class
        try:
            if probabilities.size == 0:
                metrics["auroc_macro"] = None
            elif probabilities.shape[1] == 2:
                metrics["auroc_macro"] = roc_auc_score(labels, probabilities[:, 1])
            else:
                metrics["auroc_macro"] = roc_auc_score(
                    labels, probabilities, average="macro", multi_class="ovr"
                )
        except Exception:
            metrics["auroc_macro"] = None

        metrics["per_criterion_f1"] = _compute_grouped_f1(
            labels, predictions, criterion_ids
        )

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        metrics["confusion_matrix"] = cm.tolist()

        # Classification report
        if class_names:
            report = classification_report(
                labels,
                predictions,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
            metrics["classification_report"] = report

        return metrics

    def _compute_evidence_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        class_names: list[str] | None = None,
        criterion_ids: list[str] | None = None,
    ) -> dict:
        """Compute evidence-specific metrics."""
        metrics = {}

        metrics["accuracy"] = accuracy_score(labels, predictions)
        metrics["precision_macro"] = precision_score(
            labels, predictions, average="macro", zero_division=0
        )
        metrics["precision_micro"] = precision_score(
            labels, predictions, average="micro", zero_division=0
        )
        metrics["recall_macro"] = recall_score(
            labels, predictions, average="macro", zero_division=0
        )
        metrics["recall_micro"] = recall_score(
            labels, predictions, average="micro", zero_division=0
        )
        metrics["f1_macro"] = f1_score(
            labels, predictions, average="macro", zero_division=0
        )
        metrics["f1_micro"] = f1_score(
            labels, predictions, average="micro", zero_division=0
        )
        metrics["f1_weighted"] = f1_score(
            labels, predictions, average="weighted", zero_division=0
        )

        metrics["per_criterion_f1"] = _compute_grouped_f1(
            labels, predictions, criterion_ids
        )

        # Binary metrics (if evidence is binary)
        if len(np.unique(labels)) == 2:
            # Exact match (both classes correct)
            metrics["exact_match"] = accuracy_score(labels, predictions)

            # Partial match (at least one class correct)
            # For binary, this is just 1.0 - (both classes wrong)
            both_wrong = ((labels == 1) & (predictions == 0)) | (
                (labels == 0) & (predictions == 1)
            )
            metrics["partial_match"] = 1.0 - (both_wrong.sum() / len(labels))

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        metrics["confusion_matrix"] = cm.tolist()

        # Classification report
        if class_names:
            report = classification_report(
                labels,
                predictions,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
            metrics["classification_report"] = report

        return metrics

    def _compute_basic_metrics(
        self,
        labels: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray,
        class_names: list[str] | None = None,
    ) -> dict:
        """Compute basic classification metrics."""
        metrics = {}

        metrics["accuracy"] = accuracy_score(labels, predictions)
        metrics["precision_macro"] = precision_score(
            labels, predictions, average="macro", zero_division=0
        )
        metrics["recall_macro"] = recall_score(
            labels, predictions, average="macro", zero_division=0
        )
        metrics["f1_macro"] = f1_score(
            labels, predictions, average="macro", zero_division=0
        )
        metrics["f1_weighted"] = f1_score(
            labels, predictions, average="weighted", zero_division=0
        )

        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        metrics["confusion_matrix"] = cm.tolist()

        # Classification report
        if class_names:
            report = classification_report(
                labels,
                predictions,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
            metrics["classification_report"] = report

        return metrics

    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        Get predictions for a dataset.

        Args:
            data_loader: Data loader

        Returns:
            Array of predictions
        """
        self.model.eval()

        all_predictions = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting", leave=False):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Forward pass
                logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1)

                all_predictions.extend(preds.cpu().numpy())

        return np.array(all_predictions)

    def predict_proba(self, data_loader: DataLoader) -> np.ndarray:
        """
        Get prediction probabilities for a dataset.

        Args:
            data_loader: Data loader

        Returns:
            Array of prediction probabilities [N, num_classes]
        """
        self.model.eval()

        all_probabilities = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting", leave=False):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                # Forward pass
                logits = self.model(input_ids, attention_mask)
                probs = torch.softmax(logits, dim=-1)

                all_probabilities.append(probs.cpu().numpy())

        return np.vstack(all_probabilities)


def generate_evaluation_report(
    metrics: dict,
    output_path: Path,
    title: str = "Evaluation Report",
) -> Path:
    """
    Generate comprehensive evaluation report.

    Args:
        metrics: Evaluation metrics dictionary
        output_path: Path to save report
        title: Report title

    Returns:
        Path to saved report
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as JSON
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Evaluation report saved to {output_path}")
    return output_path


def print_evaluation_results(metrics: dict, title: str = "Evaluation Results"):
    """
    Pretty print evaluation results.

    Args:
        metrics: Dictionary of evaluation metrics
        title: Title for the results
    """
    print(f"\n{'=' * 70}")
    print(f"{title:^70}")
    print(f"{'=' * 70}")

    # Main metrics
    print("\nOverall Metrics:")
    print(f"  Accuracy:       {metrics.get('accuracy', 0):.4f}")
    print(f"  Precision (M):  {metrics.get('precision_macro', 0):.4f}")
    print(f"  Recall (M):     {metrics.get('recall_macro', 0):.4f}")
    print(f"  F1 Macro:       {metrics.get('f1_macro', 0):.4f}")
    print(f"  F1 Weighted:    {metrics.get('f1_weighted', 0):.4f}")

    if "auroc_macro" in metrics and metrics["auroc_macro"] is not None:
        print(f"  AUROC (M):      {metrics['auroc_macro']:.4f}")

    if "loss" in metrics:
        print(f"  Loss:           {metrics['loss']:.4f}")

    # Per-criterion metrics if available
    if "per_criterion_f1" in metrics:
        print(f"\n{'-' * 70}")
        print("Per-Criterion F1 Scores:")
        print(f"{'-' * 70}")
        for criterion, f1 in metrics["per_criterion_f1"].items():
            print(f"  {criterion:20s}: {f1:.4f}")

    # Evidence-specific metrics
    if "exact_match" in metrics:
        print("\nEvidence Matching:")
        print(f"  Exact Match:    {metrics['exact_match']:.4f}")
        print(f"  Partial Match:  {metrics['partial_match']:.4f}")

    print(f"\n{'=' * 70}\n")


def _compute_grouped_f1(
    labels: np.ndarray,
    predictions: np.ndarray,
    criterion_ids: list[str] | None,
) -> dict[str, float]:
    """Compute per-criterion F1 scores when identifiers are available."""
    if not criterion_ids or len(criterion_ids) != len(labels):
        return {}

    metrics: dict[str, float] = {}
    unique_ids = sorted(set(criterion_ids))
    for cid in unique_ids:
        mask = np.array([cid == current for current in criterion_ids])
        if mask.sum() == 0:
            continue
        metrics[cid] = float(
            f1_score(labels[mask], predictions[mask], average="macro", zero_division=0)
        )
    return metrics
