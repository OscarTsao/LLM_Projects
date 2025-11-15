#!/usr/bin/env python3
"""Training script for DeBERTa-v3 evidence sentence classification."""

import logging
import random
import subprocess
import sys
from pathlib import Path

import hydra
import mlflow
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from Project.SubProject.data.dataset import (
    compute_class_weights,
    create_folds,
    load_dsm5_criteria,
    load_redsm5_data,
    ReDSM5Dataset,
    stratified_negative_sampling,
)
from Project.SubProject.engine.train_engine import run_cross_validation

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Set random seed to {seed}")


def get_git_info() -> dict:
    """Capture git repository information."""
    try:
        git_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
        git_status = subprocess.check_output(['git', 'status', '--short']).decode('utf-8').strip()
        return {'git_sha': git_sha, 'git_status': git_status if git_status else 'clean'}
    except Exception as e:
        logger.warning(f"Could not get git info: {e}")
        return {'git_sha': 'unknown', 'git_status': 'unknown'}


def log_environment_info(config: DictConfig):
    """Log environment and configuration to MLflow."""
    git_info = get_git_info()
    mlflow.log_param("git_sha", git_info['git_sha'])
    mlflow.log_param("git_status", git_info['git_status'])
    
    try:
        pip_freeze = subprocess.check_output(['pip', 'freeze']).decode('utf-8')
        requirements_path = Path('requirements.txt')
        requirements_path.write_text(pip_freeze)
        mlflow.log_artifact(str(requirements_path))
        requirements_path.unlink()
    except Exception as e:
        logger.warning(f"Could not log pip freeze: {e}")
    
    config_str = OmegaConf.to_yaml(config)
    config_path = Path('config.yaml')
    config_path.write_text(config_str)
    mlflow.log_artifact(str(config_path))
    config_path.unlink()
    
    mlflow.log_param("cuda_available", torch.cuda.is_available())
    if torch.cuda.is_available():
        mlflow.log_param("cuda_device", torch.cuda.get_device_name(0))
        mlflow.log_param("cuda_version", torch.version.cuda)


@hydra.main(config_path="../configs", config_name="config", version_base="1.1")
def main(config: DictConfig):
    """Main training function."""
    logger.info("Starting DeBERTa-v3 training pipeline")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(config)}")
    
    set_seed(config.training.seed)
    
    if config.mlflow.tracking_uri:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    
    logger.info("Loading DSM-5 criteria...")
    criteria_dict = load_dsm5_criteria(config.data.dsm5_dir)
    
    logger.info("Loading ReDSM5 dataset...")
    df = load_redsm5_data(config.data.csv_path, criteria_dict)
    
    logger.info("Applying stratified negative sampling...")
    df_balanced = stratified_negative_sampling(
        df, pos_neg_ratio=config.data.pos_neg_ratio, seed=config.training.seed
    )
    
    logger.info("Creating cross-validation folds...")
    folds = create_folds(df_balanced, n_splits=config.training.n_folds, seed=config.training.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.name)
    
    fold_datasets = []
    for train_idx, val_idx in folds:
        train_df = df_balanced.iloc[train_idx]
        val_df = df_balanced.iloc[val_idx]
        
        train_dataset = ReDSM5Dataset(train_df, tokenizer, max_length=config.data.max_length)
        val_dataset = ReDSM5Dataset(val_df, tokenizer, max_length=config.data.max_length)
        
        fold_datasets.append((train_dataset, val_dataset))
    
    class_weights = None
    if config.loss.type == 'weighted_ce':
        class_weights = compute_class_weights(df_balanced['label'].values)
    
    training_args_dict = OmegaConf.to_container(config.training.args, resolve=True)
    loss_config = OmegaConf.to_container(config.loss, resolve=True)
    
    log_environment_info(config)
    
    logger.info("Starting cross-validation training...")
    results = run_cross_validation(
        datasets=fold_datasets,
        model_name=config.model.name,
        output_dir=config.training.output_dir,
        training_args_dict=training_args_dict,
        loss_config=loss_config,
        class_weights=class_weights,
        experiment_name=config.mlflow.experiment_name
    )
    
    logger.info("=" * 80)
    logger.info("CROSS-VALIDATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Best Fold: {results['best_fold_index']}")
    logger.info(f"Best Model Path: {results['best_model_path']}")
    logger.info("\nBest Fold Metrics:")
    for metric, value in results['best_fold_metrics'].items():
        logger.info(f"  {metric}: {value:.4f}")
    logger.info("\nAggregate Metrics:")
    for metric, stats in results['aggregate'].items():
        logger.info(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    logger.info("=" * 80)
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
