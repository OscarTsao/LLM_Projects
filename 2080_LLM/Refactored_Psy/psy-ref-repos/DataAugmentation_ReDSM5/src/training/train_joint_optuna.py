"""Optuna hyperparameter optimization for joint training (Mode 3)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict

import hydra
import optuna
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.agents.base import setup_hardware_optimizations
from src.training.train_joint import create_joint_model
from src.data.joint_dataset import build_joint_datasets, JointCollator
from src.utils import mlflow_utils
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def suggest_hyperparameters(trial: optuna.Trial, cfg: DictConfig) -> DictConfig:
    """Suggest hyperparameters for the trial."""
    
    # Create a copy of the config
    trial_cfg = OmegaConf.create(OmegaConf.to_yaml(cfg))
    
    # Model hyperparameters
    trial_cfg.model.learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True)
    trial_cfg.model.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    trial_cfg.model.classifier_dropout = trial.suggest_float("classifier_dropout", 0.0, 0.4)
    trial_cfg.model.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    trial_cfg.model.gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4])
    trial_cfg.model.max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 2.0)
    trial_cfg.model.warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    
    # Joint training specific
    trial_cfg.model.criteria_loss_weight = trial.suggest_float("criteria_loss_weight", 0.3, 0.7)
    trial_cfg.model.evidence_loss_weight = 1.0 - trial_cfg.model.criteria_loss_weight
    trial_cfg.model.shared_encoder = trial.suggest_categorical("shared_encoder", [True, False])
    
    # Loss function parameters
    trial_cfg.model.loss_type = trial.suggest_categorical("loss_type", ["bce", "focal", "adaptive_focal"])
    if trial_cfg.model.loss_type in ["focal", "adaptive_focal"]:
        trial_cfg.model.alpha = trial.suggest_float("alpha", 0.1, 0.5)
        trial_cfg.model.gamma = trial.suggest_float("gamma", 1.0, 4.0)
        if trial_cfg.model.loss_type == "adaptive_focal":
            trial_cfg.model.delta = trial.suggest_float("delta", 0.5, 2.0)
    
    # Optimizer
    trial_cfg.model.optimizer = trial.suggest_categorical("optimizer", ["adamw_torch", "adamw_hf"])
    trial_cfg.model.adam_eps = trial.suggest_float("adam_eps", 1e-9, 1e-7, log=True)
    
    # Scheduler
    trial_cfg.model.scheduler = trial.suggest_categorical("scheduler", ["linear", "cosine", "polynomial"])
    
    # Architecture
    hidden_sizes_option = trial.suggest_categorical("hidden_sizes", ["direct", "small", "medium", "large"])
    if hidden_sizes_option == "direct":
        trial_cfg.model.classifier_hidden_sizes = []
    elif hidden_sizes_option == "small":
        trial_cfg.model.classifier_hidden_sizes = [128]
    elif hidden_sizes_option == "medium":
        trial_cfg.model.classifier_hidden_sizes = [256]
    else:  # large
        trial_cfg.model.classifier_hidden_sizes = [512, 256]
    
    # Training parameters
    trial_cfg.model.num_epochs = trial.suggest_int("num_epochs", 3, 100)
    
    # Hardware optimizations
    trial_cfg.model.use_amp = trial.suggest_categorical("use_amp", [True, False])
    trial_cfg.model.use_gradient_checkpointing = trial.suggest_categorical("use_gradient_checkpointing", [True, False])
    
    return trial_cfg


def objective(trial: optuna.Trial, base_cfg: DictConfig) -> float:
    """Objective function for Optuna optimization."""
    
    try:
        # Suggest hyperparameters
        cfg = suggest_hyperparameters(trial, base_cfg)
        
        # Setup hardware optimizations
        setup_hardware_optimizations(
            use_tf32=cfg.model.get("use_tf32", True),
            use_cudnn_benchmark=cfg.model.get("cudnn_benchmark", True)
        )
        
        # Build datasets
        train_dataset, val_dataset, _ = build_joint_datasets(cfg.dataset)
        
        # Create collator
        collator = JointCollator(
            tokenizer_name=cfg.model.pretrained_model_name,
            max_seq_length=cfg.model.max_seq_length
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.model.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.model.batch_size * 2,
            shuffle=False,
            collate_fn=collator,
            num_workers=cfg.dataloader.num_workers,
            pin_memory=cfg.dataloader.pin_memory,
        )
        
        # Create model
        model = create_joint_model(cfg)
        
        # Training loop (simplified for HPO)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.model.learning_rate,
            weight_decay=cfg.model.weight_decay,
            eps=cfg.model.adam_eps
        )
        
        best_val_f1 = 0.0
        
        for epoch in range(cfg.model.num_epochs):
            model.train()
            progress_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{cfg.model.num_epochs}",
                leave=False,
            )
            running_loss = 0.0
            for step, batch in enumerate(progress_bar, start=1):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs["loss"]
                loss.backward()
                running_loss += loss.item()
                
                if cfg.model.get("max_grad_norm", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.model.max_grad_norm)
                
                optimizer.step()
                
                progress_bar.set_postfix(loss=running_loss / step)
            
            # Validation
            model.eval()
            val_metrics = {"f1": 0.0}  # Placeholder - implement proper validation
            
            # Report intermediate value for pruning
            trial.report(val_metrics["f1"], epoch)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
        
        return best_val_f1
        
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        raise


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main HPO function for joint training."""
    
    # Setup MLflow
    mlflow_section = getattr(cfg, "mlflow", None)
    experiment_name = None
    if mlflow_section is not None:
        experiments_cfg = mlflow_section.get("experiments")
        if experiments_cfg is not None:
            experiment_name = experiments_cfg.get("joint_optuna")
    mlflow_utils.setup_mlflow(cfg, experiment_name=experiment_name)
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        study_name=f"joint_training_{cfg.get('study_name_suffix', '')}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, cfg),
        n_trials=cfg.optuna.n_trials,
        timeout=cfg.optuna.get("timeout"),
        n_jobs=1,
    )
    
    # Log best trial
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value}")
    logger.info(f"Best params: {study.best_params}")
    
    # Log to MLflow
    if mlflow_utils.is_mlflow_enabled():
        mlflow_utils.log_best_trial(study)


if __name__ == "__main__":
    main()
