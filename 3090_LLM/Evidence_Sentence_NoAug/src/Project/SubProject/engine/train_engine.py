"""Training engine for DeBERTa-v3 evidence sentence classification.

This module implements custom loss functions, trainer, and cross-validation
orchestration with MLflow integration.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def create_loss_function(
    loss_type: str,
    class_weights: Optional[torch.Tensor] = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0
) -> nn.Module:
    """Factory function to create loss function."""
    if loss_type == 'weighted_ce':
        if class_weights is None:
            logger.warning("No class weights provided, using uniform weights")
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"Using Weighted Cross-Entropy Loss")
    elif loss_type == 'focal':
        loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        logger.info(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    return loss_fn


class CustomTrainer(Trainer):
    """Custom Trainer with support for custom loss functions."""
    
    def __init__(self, *args, custom_loss_fn: Optional[nn.Module] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_loss_fn = custom_loss_fn
        
    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs: bool = False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.custom_loss_fn is not None:
            loss = self.custom_loss_fn(logits, labels)
        else:
            loss = nn.functional.cross_entropy(logits, labels)
        
        return (loss, outputs) if return_outputs else loss


def detect_precision() -> str:
    """Detect best available precision for training."""
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            logger.info("BF16 precision available and will be used")
            return 'bf16'
        logger.info("FP16 precision will be used")
        return 'fp16'
    logger.info("No GPU available, using FP32 precision")
    return 'fp32'


def get_optimizer_name() -> str:
    """Get best available optimizer name."""
    if torch.cuda.is_available():
        try:
            torch.optim.AdamW([torch.nn.Parameter(torch.randn(1))], fused=True)
            logger.info("Using fused AdamW optimizer")
            return 'adamw_torch_fused'
        except Exception:
            pass
    logger.info("Using standard AdamW optimizer")
    return 'adamw_torch'


def train_single_fold(
    fold_idx: int,
    train_dataset,
    val_dataset,
    model_name: str,
    output_dir: str,
    training_args_dict: Dict,
    loss_config: Dict,
    class_weights: Optional[torch.Tensor] = None
) -> Tuple[Dict[str, float], str]:
    """Train a single fold and return metrics."""
    logger.info(f"Starting training for Fold {fold_idx}")
    
    fold_output_dir = Path(output_dir) / f"fold_{fold_idx}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    logger.info(f"Loaded model: {model_name}")
    
    if class_weights is not None:
        class_weights = class_weights.to(model.device)
    
    loss_fn = create_loss_function(
        loss_type=loss_config['type'],
        class_weights=class_weights,
        focal_alpha=loss_config.get('focal_alpha', 0.25),
        focal_gamma=loss_config.get('focal_gamma', 2.0)
    )
    
    precision = detect_precision()
    optimizer_name = get_optimizer_name()
    
    training_args = TrainingArguments(
        output_dir=str(fold_output_dir),
        bf16=(precision == 'bf16'),
        fp16=(precision == 'fp16'),
        optim=optimizer_name,
        **training_args_dict
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        custom_loss_fn=loss_fn
    )
    
    with mlflow.start_run(run_name=f"fold_{fold_idx}", nested=True):
        mlflow.log_param("fold_index", fold_idx)
        mlflow.log_param("train_size", len(train_dataset))
        mlflow.log_param("val_size", len(val_dataset))
        
        train_result = trainer.train()
        mlflow.log_metrics({'train_loss': train_result.training_loss})
        
        logger.info(f"Evaluating Fold {fold_idx}")
        predictions = trainer.predict(val_dataset)
        
        logits = predictions.predictions
        probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        labels = predictions.label_ids
        
        from .eval_engine import log_evaluation_artifacts
        metrics = log_evaluation_artifacts(
            predictions=probs,
            labels=labels,
            threshold=0.5,
            output_dir=str(fold_output_dir)
        )
        
        model_path = str(fold_output_dir / "final_model")
        trainer.save_model(model_path)
        logger.info(f"Saved model to {model_path}")
        
        mlflow.pytorch.log_model(model, "model")
        
    return metrics, model_path


def run_cross_validation(
    datasets: List[Tuple],
    model_name: str,
    output_dir: str,
    training_args_dict: Dict,
    loss_config: Dict,
    class_weights: Optional[torch.Tensor] = None,
    experiment_name: str = "deberta_cv"
) -> Dict:
    """Run complete cross-validation pipeline."""
    mlflow.set_experiment(experiment_name)
    
    all_metrics = []
    fold_paths = []
    
    with mlflow.start_run(run_name="cross_validation"):
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("n_folds", len(datasets))
        mlflow.log_param("loss_type", loss_config['type'])
        
        for param_name, param_value in training_args_dict.items():
            mlflow.log_param(f"training_{param_name}", param_value)
        
        for fold_idx, (train_ds, val_ds) in enumerate(datasets, 1):
            metrics, model_path = train_single_fold(
                fold_idx=fold_idx,
                train_dataset=train_ds,
                val_dataset=val_ds,
                model_name=model_name,
                output_dir=output_dir,
                training_args_dict=training_args_dict,
                loss_config=loss_config,
                class_weights=class_weights
            )
            all_metrics.append(metrics)
            fold_paths.append(model_path)
        
        aggregate = compute_aggregate_metrics(all_metrics)
        
        for metric_name, stats in aggregate.items():
            mlflow.log_metric(f"{metric_name}_mean", stats['mean'])
            mlflow.log_metric(f"{metric_name}_std", stats['std'])
        
        best_fold_idx = np.argmax([m['macro_f1'] for m in all_metrics])
        best_fold_metrics = all_metrics[best_fold_idx]
        
        mlflow.log_param("best_fold_index", best_fold_idx + 1)
        mlflow.log_metrics({f"best_{k}": v for k, v in best_fold_metrics.items() if k != 'threshold'})
        
        logger.info(f"Best fold: {best_fold_idx + 1}")
        logger.info(f"Best macro F1: {best_fold_metrics['macro_f1']:.4f}")
    
    return {
        'all_metrics': all_metrics,
        'aggregate': aggregate,
        'best_fold_index': best_fold_idx + 1,
        'best_fold_metrics': best_fold_metrics,
        'best_model_path': fold_paths[best_fold_idx],
        'fold_paths': fold_paths
    }


def compute_aggregate_metrics(fold_metrics: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """Compute mean and std for each metric across folds."""
    metric_names = [k for k in fold_metrics[0].keys() if k != 'threshold']
    
    aggregate = {}
    for metric_name in metric_names:
        values = [fold[metric_name] for fold in fold_metrics]
        aggregate[metric_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values))
        }
        logger.info(f"{metric_name}: {aggregate[metric_name]['mean']:.4f} ± {aggregate[metric_name]['std']:.4f}")
    
    return aggregate
