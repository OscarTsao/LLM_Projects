"""Optuna hyperparameter optimization for evidence binding agent (Mode 2)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Dict

import hydra
import optuna
import torch
from omegaconf import DictConfig, OmegaConf

from src.agents.base import setup_hardware_optimizations
from src.training.train_evidence import create_evidence_agent, train_evidence_agent, create_evidence_datasets
from src.data.evidence_loader import EvidenceCollator
from src.utils import mlflow_utils

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
    
    # Evidence-specific parameters
    trial_cfg.model.label_smoothing = trial.suggest_float("label_smoothing", 0.0, 0.2)
    trial_cfg.model.span_threshold = trial.suggest_float("span_threshold", 0.3, 0.7)
    trial_cfg.model.max_span_length = trial.suggest_int("max_span_length", 20, 100)
    
    # Optimizer
    trial_cfg.model.optimizer = trial.suggest_categorical("optimizer", ["adamw_torch", "adamw_hf"])
    trial_cfg.model.adam_eps = trial.suggest_float("adam_eps", 1e-9, 1e-7, log=True)
    
    # Scheduler
    trial_cfg.model.scheduler = trial.suggest_categorical("scheduler", ["linear", "cosine", "polynomial"])
    
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
        
        # Create temporary output directory for this trial
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg.output_dir = temp_dir
            
            # Create datasets
            train_dataset, val_dataset, test_dataset = create_evidence_datasets(cfg)
            
            # Create data loaders
            from torch.utils.data import DataLoader
            collator = EvidenceCollator(
                tokenizer_name=cfg.model.pretrained_model_name,
                max_length=cfg.model.max_seq_length
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=cfg.model.batch_size,
                shuffle=True,
                collate_fn=collator,
                num_workers=0,  # Use 0 for Optuna to avoid multiprocessing issues
                pin_memory=False
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg.model.batch_size,
                shuffle=False,
                collate_fn=collator,
                num_workers=0,
                pin_memory=False
            )
            
            # Create agent
            agent = create_evidence_agent(cfg)
            
            # Setup device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            agent = agent.to(device)
            
            # Create optimizer
            if cfg.model.optimizer == "adamw_torch":
                optimizer = torch.optim.AdamW(
                    agent.parameters(),
                    lr=cfg.model.learning_rate,
                    weight_decay=cfg.model.weight_decay,
                    eps=cfg.model.adam_eps
                )
            else:  # adamw_hf
                from transformers import AdamW
                optimizer = AdamW(
                    agent.parameters(),
                    lr=cfg.model.learning_rate,
                    weight_decay=cfg.model.weight_decay,
                    eps=cfg.model.adam_eps
                )
            
            # Create scheduler
            num_training_steps = len(train_loader) * cfg.model.num_epochs
            num_warmup_steps = int(cfg.model.warmup_ratio * num_training_steps)
            
            if cfg.model.scheduler == "linear":
                from transformers import get_linear_schedule_with_warmup
                scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )
            elif cfg.model.scheduler == "cosine":
                from transformers import get_cosine_schedule_with_warmup
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )
            elif cfg.model.scheduler == "polynomial":
                from transformers import get_polynomial_decay_schedule_with_warmup
                scheduler = get_polynomial_decay_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )
            else:
                scheduler = None
            
            # Training configuration
            train_config = {
                "num_epochs": cfg.model.num_epochs,
                "gradient_accumulation_steps": cfg.model.gradient_accumulation_steps,
                "max_grad_norm": cfg.model.max_grad_norm,
                "use_amp": cfg.model.use_amp,
                "early_stopping_patience": cfg.get("early_stopping_patience", 10),
                "metric_for_best_model": cfg.get("metric_for_best_model", "span_f1"),
                "output_dir": cfg.output_dir,
                "save_steps": None,
                "eval_steps": None,
                "logging_steps": 100,
            }
            
            # Train model
            best_metric = train_evidence_agent(
                agent=agent,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                config=train_config
            )
            
            # Report intermediate values for pruning
            trial.report(best_metric, step=cfg.model.num_epochs)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
            
            return best_metric
            
    except Exception as e:
        logger.error(f"Trial {trial.number} failed: {e}")
        raise


@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function for Optuna hyperparameter optimization."""
    
    # Setup hardware optimizations
    setup_hardware_optimizations()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting Optuna hyperparameter optimization for evidence binding agent")
    
    # Optuna configuration
    n_trials = cfg.optuna.get("n_trials", 500)
    timeout = cfg.optuna.get("timeout", None)
    study_name = cfg.optuna.get("study_name", "evidence-binding-hpo")
    storage = cfg.optuna.get("storage", None)
    
    # Create study
    if storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction="maximize",
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
    else:
        study = optuna.create_study(
            direction="maximize",
            pruner=optuna.pruners.MedianPruner(
                n_startup_trials=5,
                n_warmup_steps=10,
                interval_steps=1
            )
        )
    
    # Initialize MLflow for best trial logging
    mlflow_run = mlflow_utils.start_run(
        experiment_name="redsm5-optuna-evidence",
        run_name=f"evidence-hpo-{study_name}"
    )
    
    try:
        with mlflow_run:
            # Log study configuration
            mlflow_utils.log_params({
                "n_trials": n_trials,
                "timeout": timeout,
                "study_name": study_name,
                "direction": "maximize",
                "pruner": "MedianPruner"
            })
            mlflow_utils.set_tag("optimization_type", "optuna")
            mlflow_utils.set_tag("training_mode", "evidence_only")
            
            # Run optimization
            study.optimize(
                lambda trial: objective(trial, cfg),
                n_trials=n_trials,
                timeout=timeout,
                callbacks=[
                    lambda study, trial: logger.info(
                        f"Trial {trial.number} finished with value: {trial.value} "
                        f"and parameters: {trial.params}"
                    )
                ]
            )
            
            # Log best trial results
            best_trial = study.best_trial
            logger.info(f"Best trial: {best_trial.number}")
            logger.info(f"Best value: {best_trial.value}")
            logger.info(f"Best params: {best_trial.params}")
            
            # Log to MLflow
            mlflow_utils.log_params(best_trial.params)
            mlflow_utils.log_metrics({
                "best_value": best_trial.value,
                "n_trials_completed": len(study.trials),
                "best_trial_number": best_trial.number
            })
            
            # Save study results
            output_dir = Path(cfg.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save best parameters as YAML
            best_config_path = output_dir / "best_evidence_config.yaml"
            with open(best_config_path, 'w') as f:
                import yaml
                yaml.dump(best_trial.params, f)
            
            # Save all trials as CSV
            trials_df = study.trials_dataframe()
            trials_csv_path = output_dir / "evidence_trials.csv"
            trials_df.to_csv(trials_csv_path, index=False)
            
            # Log artifacts
            mlflow_utils.log_artifact(str(best_config_path))
            mlflow_utils.log_artifact(str(trials_csv_path))
            
            logger.info(f"Optimization completed. Best config saved to {best_config_path}")
            
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        raise
    finally:
        if mlflow_run:
            mlflow_run.__exit__(None, None, None)


if __name__ == "__main__":
    main()
