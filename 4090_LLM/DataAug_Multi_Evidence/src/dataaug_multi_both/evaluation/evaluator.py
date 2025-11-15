"""
Model evaluation and metrics computation.

Provides:
- Evaluation on test/validation sets
- Comprehensive metrics (exact match, F1, precision, recall)
- Error analysis
- Results export
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Results from model evaluation."""

    exact_match: float
    f1: float
    precision: float
    recall: float
    total_samples: int
    predictions: list[dict[str, Any]] | None = None


class EvidenceExtractionEvaluator:
    """
    Evaluator for evidence extraction models.

    Handles:
    - Model evaluation on datasets
    - Metrics computation
    - Error analysis
    - Results export
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize evaluator.

        Args:
            model: Model to evaluate
            device: Device to run evaluation on
        """
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        logger.info(f"Evaluator initialized on device: {device}")

    def evaluate(
        self,
        dataloader: DataLoader,
        save_predictions: bool = False,
    ) -> EvaluationResult:
        """
        Evaluate model on a dataset.

        Args:
            dataloader: DataLoader for evaluation data
            save_predictions: Whether to save individual predictions

        Returns:
            EvaluationResult with metrics and optional predictions
        """
        logger.info(f"Starting evaluation on {len(dataloader)} batches")

        all_predictions = []
        all_labels = []

        predictions_list: list[dict[str, Any]] | None = [] if save_predictions else None

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                start_positions = batch["start_positions"].to(self.device)
                end_positions = batch["end_positions"].to(self.device)

                # Forward pass
                start_logits, end_logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                # Get predictions
                start_preds = torch.argmax(start_logits, dim=1)
                end_preds = torch.argmax(end_logits, dim=1)

                # Collect for metrics
                for i in range(len(input_ids)):
                    pred = {
                        "start_pred": start_preds[i].item(),
                        "end_pred": end_preds[i].item(),
                        "start_label": start_positions[i].item(),
                        "end_label": end_positions[i].item(),
                    }

                    all_predictions.append(pred)
                    all_labels.append(
                        {
                            "start": start_positions[i].item(),
                            "end": end_positions[i].item(),
                        }
                    )

                    # Save detailed predictions if requested
                    if save_predictions and predictions_list is not None:
                        pred_detail = {
                            **pred,
                            "input_ids": input_ids[i].cpu().tolist(),
                            "start_logits": start_logits[i].cpu().tolist(),
                            "end_logits": end_logits[i].cpu().tolist(),
                        }
                        predictions_list.append(pred_detail)

        # Compute metrics
        metrics = self._compute_metrics(all_predictions)

        logger.info(
            f"Evaluation complete: EM={metrics['exact_match']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"P={metrics['precision']:.4f}, "
            f"R={metrics['recall']:.4f}"
        )

        return EvaluationResult(
            exact_match=metrics["exact_match"],
            f1=metrics["f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            total_samples=len(all_predictions),
            predictions=predictions_list,
        )

    def _compute_metrics(self, predictions: list[dict[str, Any]]) -> dict[str, float]:
        """
        Compute evaluation metrics from predictions.

        Args:
            predictions: List of prediction dictionaries

        Returns:
            Dictionary of metrics
        """
        exact_matches = 0
        f1_scores = []
        precisions = []
        recalls = []

        for pred in predictions:
            start_pred = pred["start_pred"]
            end_pred = pred["end_pred"]
            start_label = pred["start_label"]
            end_label = pred["end_label"]

            # Exact match
            if start_pred == start_label and end_pred == end_label:
                exact_matches += 1

            # Span overlap metrics
            pred_span = set(range(start_pred, end_pred + 1))
            label_span = set(range(start_label, end_label + 1))

            if len(pred_span) == 0 and len(label_span) == 0:
                # Both empty - perfect match
                f1_scores.append(1.0)
                precisions.append(1.0)
                recalls.append(1.0)
            elif len(pred_span) == 0 or len(label_span) == 0:
                # One empty - no match
                f1_scores.append(0.0)
                precisions.append(0.0)
                recalls.append(0.0)
            else:
                # Compute overlap
                intersection = len(pred_span & label_span)
                precision = intersection / len(pred_span)
                recall = intersection / len(label_span)

                precisions.append(precision)
                recalls.append(recall)

                if precision + recall > 0:
                    f1 = 2 * (precision * recall) / (precision + recall)
                else:
                    f1 = 0.0

                f1_scores.append(f1)

        total = len(predictions)

        return {
            "exact_match": exact_matches / total if total > 0 else 0.0,
            "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
            "precision": sum(precisions) / len(precisions) if precisions else 0.0,
            "recall": sum(recalls) / len(recalls) if recalls else 0.0,
        }

    def error_analysis(
        self,
        dataloader: DataLoader,
        output_path: str | Path,
        num_errors: int = 100,
    ) -> None:
        """
        Perform error analysis and save results.

        Args:
            dataloader: DataLoader for evaluation data
            output_path: Path to save error analysis
            num_errors: Maximum number of errors to analyze
        """
        logger.info(f"Performing error analysis, saving to {output_path}")

        result = self.evaluate(dataloader, save_predictions=True)

        # Find errors
        errors = []
        if result.predictions is not None:
            for pred in result.predictions:
                if (
                    pred["start_pred"] != pred["start_label"]
                    or pred["end_pred"] != pred["end_label"]
                ):
                    errors.append(pred)

                    if len(errors) >= num_errors:
                        break

        # Save error analysis
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(
                {
                    "total_samples": result.total_samples,
                    "total_errors": len(errors),
                    "error_rate": len(errors) / result.total_samples,
                    "metrics": {
                        "exact_match": result.exact_match,
                        "f1": result.f1,
                        "precision": result.precision,
                        "recall": result.recall,
                    },
                    "errors": errors,
                },
                f,
                indent=2,
            )

        logger.info(f"Error analysis complete: {len(errors)} errors saved to {output_path}")

    def export_results(self, result: EvaluationResult, output_path: str | Path) -> None:
        """
        Export evaluation results to file.

        Args:
            result: EvaluationResult to export
            output_path: Path to save results
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        results_dict: dict[str, Any] = {
            "exact_match": result.exact_match,
            "f1": result.f1,
            "precision": result.precision,
            "recall": result.recall,
            "total_samples": result.total_samples,
        }

        if result.predictions:
            results_dict["predictions"] = result.predictions

        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2)

        logger.info(f"Results exported to {output_path}")

    def compare_models(
        self,
        other_evaluator: EvidenceExtractionEvaluator,
        dataloader: DataLoader,
    ) -> dict[str, Any]:
        """
        Compare this model with another model.

        Args:
            other_evaluator: Another evaluator to compare against
            dataloader: DataLoader for evaluation data

        Returns:
            Comparison results
        """
        logger.info("Comparing models...")

        # Evaluate both models
        result_a = self.evaluate(dataloader)
        result_b = other_evaluator.evaluate(dataloader)

        comparison = {
            "model_a": {
                "exact_match": result_a.exact_match,
                "f1": result_a.f1,
                "precision": result_a.precision,
                "recall": result_a.recall,
            },
            "model_b": {
                "exact_match": result_b.exact_match,
                "f1": result_b.f1,
                "precision": result_b.precision,
                "recall": result_b.recall,
            },
            "differences": {
                "exact_match": result_a.exact_match - result_b.exact_match,
                "f1": result_a.f1 - result_b.f1,
                "precision": result_a.precision - result_b.precision,
                "recall": result_a.recall - result_b.recall,
            },
        }

        logger.info(
            f"Model A F1: {result_a.f1:.4f}, Model B F1: {result_b.f1:.4f}, "
            f"Difference: {comparison['differences']['f1']:.4f}"
        )

        return comparison
