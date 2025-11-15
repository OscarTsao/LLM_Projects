"""Evaluation engine for model assessment and visualization.

This module provides comprehensive evaluation metrics and visualization
tools for the evidence sentence classification task.
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)


def compute_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute comprehensive evaluation metrics.

    Args:
        predictions: Predicted probabilities for positive class (shape: [n,]).
        labels: True binary labels (shape: [n,]).
        threshold: Classification threshold for converting probs to labels.

    Returns:
        Dictionary containing:
        - macro_f1: Macro-averaged F1 score
        - positive_f1: F1 score for positive class
        - roc_auc: ROC-AUC score
        - pr_auc: Precision-Recall AUC score
        - threshold: Used threshold value
    """
    # Convert probabilities to binary predictions
    pred_labels = (predictions >= threshold).astype(int)

    # Compute F1 scores
    macro_f1 = f1_score(labels, pred_labels, average='macro')
    positive_f1 = f1_score(labels, pred_labels, pos_label=1)

    # Compute AUC scores
    roc_auc = roc_auc_score(labels, predictions)

    precision, recall, _ = precision_recall_curve(labels, predictions)
    pr_auc = auc(recall, precision)

    metrics = {
        'macro_f1': float(macro_f1),
        'positive_f1': float(positive_f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'threshold': float(threshold)
    }

    logger.info(f"Metrics at threshold {threshold:.2f}:")
    for key, value in metrics.items():
        if key != 'threshold':
            logger.info(f"  {key}: {value:.4f}")

    return metrics


def plot_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    save_path: Optional[str] = None
) -> str:
    """Plot and save confusion matrix.

    Args:
        predictions: Predicted probabilities for positive class.
        labels: True binary labels.
        threshold: Classification threshold.
        save_path: Path to save plot. If None, uses temp directory.

    Returns:
        Path to saved plot file.
    """
    pred_labels = (predictions >= threshold).astype(int)
    cm = confusion_matrix(labels, pred_labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Not Evidence', 'Evidence'],
        yticklabels=['Not Evidence', 'Evidence'],
        ax=ax
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix (threshold={threshold:.2f})')

    if save_path is None:
        save_path = 'confusion_matrix.png'

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved confusion matrix to {save_path}")
    return save_path


def plot_roc_curve(
    predictions: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None
) -> str:
    """Plot and save ROC curve.

    Args:
        predictions: Predicted probabilities for positive class.
        labels: True binary labels.
        save_path: Path to save plot. If None, uses temp directory.

    Returns:
        Path to saved plot file.
    """
    fpr, tpr, _ = roc_curve(labels, predictions)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        fpr,
        tpr,
        color='darkorange',
        lw=2,
        label=f'ROC curve (AUC = {roc_auc:.4f})'
    )
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    if save_path is None:
        save_path = 'roc_curve.png'

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved ROC curve to {save_path}")
    return save_path


def plot_precision_recall_curve(
    predictions: np.ndarray,
    labels: np.ndarray,
    save_path: Optional[str] = None
) -> str:
    """Plot and save Precision-Recall curve.

    Args:
        predictions: Predicted probabilities for positive class.
        labels: True binary labels.
        save_path: Path to save plot. If None, uses temp directory.

    Returns:
        Path to saved plot file.
    """
    precision, recall, _ = precision_recall_curve(labels, predictions)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        recall,
        precision,
        color='darkorange',
        lw=2,
        label=f'PR curve (AUC = {pr_auc:.4f})'
    )
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)

    if save_path is None:
        save_path = 'pr_curve.png'

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved PR curve to {save_path}")
    return save_path


def log_evaluation_artifacts(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
    output_dir: Optional[str] = None
) -> Dict[str, float]:
    """Compute metrics and log all evaluation artifacts to MLflow.

    Args:
        predictions: Predicted probabilities for positive class.
        labels: True binary labels.
        threshold: Classification threshold.
        output_dir: Directory to save plots. If None, uses temp directory.

    Returns:
        Dictionary of computed metrics.
    """
    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cm_path = str(Path(output_dir) / 'confusion_matrix.png')
        roc_path = str(Path(output_dir) / 'roc_curve.png')
        pr_path = str(Path(output_dir) / 'pr_curve.png')
    else:
        cm_path = None
        roc_path = None
        pr_path = None

    # Compute metrics
    metrics = compute_metrics(predictions, labels, threshold)

    # Log metrics to MLflow
    for key, value in metrics.items():
        mlflow.log_metric(key, value)

    # Generate and log plots
    cm_file = plot_confusion_matrix(predictions, labels, threshold, cm_path)
    roc_file = plot_roc_curve(predictions, labels, roc_path)
    pr_file = plot_precision_recall_curve(predictions, labels, pr_path)

    mlflow.log_artifact(cm_file)
    mlflow.log_artifact(roc_file)
    mlflow.log_artifact(pr_file)

    logger.info("Successfully logged all evaluation artifacts to MLflow")

    return metrics
