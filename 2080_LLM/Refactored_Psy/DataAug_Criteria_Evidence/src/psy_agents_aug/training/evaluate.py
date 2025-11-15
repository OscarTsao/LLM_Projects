"""Evaluation utilities with comprehensive metrics."""

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


class Evaluator:
    """
    Model evaluator with comprehensive metrics.
    
    Computes:
    - Accuracy, precision, recall, F1
    - Per-class metrics
    - Confusion matrix
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        criterion: Optional[nn.Module] = None,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: Model to evaluate
            device: Device to run evaluation on
            criterion: Optional loss function
        """
        self.model = model
        self.device = device
        self.criterion = criterion
    
    def evaluate(
        self,
        data_loader: DataLoader,
        class_names: Optional[List[str]] = None,
    ) -> Dict:
        """
        Evaluate model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            class_names: Optional list of class names
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
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
                
                # Get predictions
                preds = torch.argmax(logits, dim=-1)
                
                # Store predictions and labels
                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        
        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
            "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
            "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
            "precision_weighted": precision_score(labels, predictions, average="weighted", zero_division=0),
            "recall_weighted": recall_score(labels, predictions, average="weighted", zero_division=0),
            "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
        }
        
        if self.criterion:
            metrics["loss"] = total_loss / len(data_loader)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        metrics["confusion_matrix"] = cm.tolist()
        
        # Per-class metrics
        if class_names:
            report = classification_report(
                labels,
                predictions,
                target_names=class_names,
                output_dict=True,
                zero_division=0,
            )
            metrics["per_class_metrics"] = report
        
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


def print_evaluation_results(metrics: Dict, title: str = "Evaluation Results"):
    """
    Pretty print evaluation results.
    
    Args:
        metrics: Dictionary of evaluation metrics
        title: Title for the results
    """
    print(f"\n{'=' * 60}")
    print(f"{title:^60}")
    print(f"{'=' * 60}")
    
    # Main metrics
    print(f"\nAccuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision_macro']:.4f} (macro), {metrics['precision_weighted']:.4f} (weighted)")
    print(f"Recall:    {metrics['recall_macro']:.4f} (macro), {metrics['recall_weighted']:.4f} (weighted)")
    print(f"F1 Score:  {metrics['f1_macro']:.4f} (macro), {metrics['f1_weighted']:.4f} (weighted)")
    
    if "loss" in metrics:
        print(f"Loss:      {metrics['loss']:.4f}")
    
    # Per-class metrics if available
    if "per_class_metrics" in metrics:
        print(f"\n{'-' * 60}")
        print("Per-Class Metrics:")
        print(f"{'-' * 60}")
        
        for class_name, class_metrics in metrics["per_class_metrics"].items():
            if class_name in ["accuracy", "macro avg", "weighted avg"]:
                continue
            
            print(f"\n{class_name}:")
            print(f"  Precision: {class_metrics['precision']:.4f}")
            print(f"  Recall:    {class_metrics['recall']:.4f}")
            print(f"  F1 Score:  {class_metrics['f1-score']:.4f}")
            print(f"  Support:   {class_metrics['support']}")
    
    print(f"\n{'=' * 60}\n")
