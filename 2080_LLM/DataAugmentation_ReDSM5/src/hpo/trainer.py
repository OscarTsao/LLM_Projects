"""
Training loop for augmented data with early stopping and validation.

Integrates with HPO framework to evaluate augmentation strategies.
"""

from typing import Dict, Any, Optional, Tuple
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import pandas as pd
import optuna


class AugmentationTrainer:
    """
    Trainer for models with augmented data.
    
    Supports:
    - Loading augmented data from cache
    - Training with early stopping
    - Validation metrics (F1, accuracy, precision, recall)
    - Integration with Optuna for pruning
    
    Attributes:
        model_name: Pretrained model name
        num_labels: Number of classification labels
        device: Training device
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        num_labels: int = 2,
        device: Optional[str] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model_name: HuggingFace model name
            num_labels: Number of labels
            device: Device (auto-detect if None)
        """
        self.model_name = model_name
        self.num_labels = num_labels
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def prepare_dataset(
        self,
        df: pd.DataFrame,
        text_field: str = "evidence_sentence",
        label_field: str = "criteria_label",
        max_length: int = 256,
    ) -> Dataset:
        """
        Prepare dataset from DataFrame.
        
        Args:
            df: Input DataFrame
            text_field: Name of text column
            label_field: Name of label column
            max_length: Maximum sequence length
            
        Returns:
            PyTorch Dataset
        """
        # Tokenize texts
        encodings = self.tokenizer(
            df[text_field].tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        
        # Get labels
        labels = torch.tensor(df[label_field].tolist())
        
        # Create dataset
        class SimpleDataset(Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __len__(self):
                return len(self.labels)
            
            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item["labels"] = self.labels[idx]
                return item
        
        return SimpleDataset(encodings, labels)
    
    def compute_metrics(self, eval_pred) -> Dict[str, float]:
        """
        Compute evaluation metrics.
        
        Args:
            eval_pred: Predictions and labels
            
        Returns:
            Dictionary of metrics
        """
        predictions, labels = eval_pred
        predictions = predictions.argmax(axis=-1)
        
        # Compute metrics
        precision, recall, f1_macro, _ = precision_recall_fscore_support(
            labels, predictions, average="macro"
        )
        
        accuracy = accuracy_score(labels, predictions)
        
        return {
            "accuracy": accuracy,
            "f1_macro": f1_macro,
            "precision": precision,
            "recall": recall,
        }
    
    def train(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        hyperparams: Dict[str, Any],
        output_dir: str = "checkpoints",
        trial: Optional[optuna.Trial] = None,
    ) -> Tuple[Dict[str, float], Any]:
        """
        Train model on augmented data.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            hyperparams: Hyperparameters (learning_rate, batch_size, etc.)
            output_dir: Directory for checkpoints
            trial: Optuna trial for pruning
            
        Returns:
            Tuple of (metrics, trained_model)
        """
        # Extract hyperparameters
        learning_rate = hyperparams.get("learning_rate", 2e-5)
        batch_size = hyperparams.get("batch_size", 16)
        num_epochs = hyperparams.get("max_epochs", 5)
        weight_decay = hyperparams.get("weight_decay", 0.01)
        warmup_ratio = hyperparams.get("warmup_ratio", 0.1)
        
        # Prepare datasets
        train_dataset = self.prepare_dataset(train_df)
        val_dataset = self.prepare_dataset(val_df)
        
        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_ratio=warmup_ratio,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            logging_steps=100,
            fp16=torch.cuda.is_available(),
            report_to="none",  # Disable wandb/tensorboard for now
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
        )
        
        # Train
        trainer.train()
        
        # Evaluate
        metrics = trainer.evaluate()
        
        # Report to Optuna trial if provided
        if trial is not None:
            trial.report(metrics["eval_f1_macro"], step=num_epochs)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        return metrics, model
    
    def evaluate(
        self,
        model: Any,
        test_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Evaluate model on test data.
        
        Args:
            model: Trained model
            test_df: Test DataFrame
            
        Returns:
            Dictionary of metrics
        """
        # Prepare dataset
        test_dataset = self.prepare_dataset(test_df)
        
        # Create trainer for evaluation
        trainer = Trainer(
            model=model,
            compute_metrics=self.compute_metrics,
        )
        
        # Evaluate
        metrics = trainer.evaluate(test_dataset)
        
        return metrics
