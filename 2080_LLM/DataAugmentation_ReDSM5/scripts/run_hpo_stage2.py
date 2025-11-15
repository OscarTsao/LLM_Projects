#!/usr/bin/env python3
"""
Run Stage 2 HPO: Hyperparameter Tuning

Fine-tunes augmentation parameters and model hyperparameters for the best combo from Stage 1.

Usage:
    python scripts/run_hpo_stage2.py [--config CONFIG] [--combo COMBO] [--trials N]
"""

import argparse
import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataio.loader import REDSM5Loader
from dataio.parquet_io import ParquetIO
from hpo.search import HPOSearch
from hpo.trainer import AugmentationTrainer
from aug.compose import AugmentationPipeline
from utils.logging import setup_logger
import yaml
import optuna


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Stage 2 HPO: Hyperparameter Tuning"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/run.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--combo",
        nargs="+",
        required=True,
        help="Augmentation combo from Stage 1 (space-separated)",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=None,
        help="Number of trials (overrides config)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="stage2_hyperparameter_tuning",
        help="Study name",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///hpo_stage2.db",
        help="Optuna storage URL",
    )
    
    return parser.parse_args()


def objective(trial, combo, config, logger, trainer, train_df_base, val_df):
    """
    Objective function for Stage 2.
    
    Args:
        trial: Optuna trial
        combo: Fixed augmentation combo
        config: Configuration dictionary
        logger: Logger instance
        trainer: AugmentationTrainer instance
        train_df_base: Base training DataFrame (not augmented)
        val_df: Validation DataFrame
        
    Returns:
        Validation F1 score
    """
    # Suggest hyperparameters
    hpo = HPOSearch(stage=2)
    hyperparams = hpo.suggest_hyperparams(trial)
    
    logger.info(f"\nTrial {trial.number}: {hyperparams}")
    
    # Apply augmentation with suggested intensity
    aug_intensity = hyperparams.pop("aug_intensity", 0.2)
    
    # Map intensity to augmenter-specific parameters
    # This is a simplified approach; in practice, you'd tune individual params
    aug_params = {}
    for aug_name in combo:
        from aug.registry import AugmenterRegistry
        registry = AugmenterRegistry()
        default_params = registry.get_default_params(aug_name)
        
        # Scale augmentation probability by intensity
        aug_params[aug_name] = {}
        for param_name, param_value in default_params.items():
            if "aug_p" in param_name or "aug_char_p" in param_name:
                aug_params[aug_name][param_name] = aug_intensity
            else:
                aug_params[aug_name][param_name] = param_value
    
    # Augment training data
    pipeline = AugmentationPipeline(
        combo=combo,
        params=aug_params,
        seed=config["global"]["seed"],
    )
    
    train_df = pipeline.augment_dataframe(
        train_df_base,
        text_field=config["dataset"]["text_field"],
        verbose=False,
    )
    
    # Train model with suggested hyperparameters
    metrics, model = trainer.train(
        train_df,
        val_df,
        hyperparams,
        output_dir=f"checkpoints/stage2/trial_{trial.number}",
        trial=trial,
    )
    
    f1_score = metrics["eval_f1_macro"]
    logger.info(f"Trial {trial.number} F1: {f1_score:.4f}")
    
    return f1_score


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger("hpo_stage2", level="INFO")
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Stage 2 HPO for combo: {' -> '.join(args.combo)}")
    
    # Load data
    logger.info("Loading data...")
    loader = REDSM5Loader(
        base_path=config["dataset"]["base_path"],
        text_field=config["dataset"]["text_field"],
        label_fields=config["dataset"]["label_fields"],
    )
    
    data = loader.load_from_parquet(splits=["train", "val"])
    train_df = data["train"]
    val_df = data["val"]
    
    logger.info(f"Training set: {len(train_df)} examples")
    logger.info(f"Validation set: {len(val_df)} examples")
    
    # Initialize trainer
    trainer = AugmentationTrainer(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
    )
    
    # Initialize HPO
    hpo = HPOSearch(config_path=args.config, stage=2)
    
    # Create objective with fixed arguments
    objective_fn = lambda trial: objective(
        trial, args.combo, config, logger, trainer, train_df, val_df
    )
    
    # Run optimization
    logger.info(f"\nStarting Stage 2 HPO with {args.trials or hpo.stage_config['trials']} trials...")
    study = hpo.run_optimization(
        objective_fn,
        n_trials=args.trials,
        study_name=args.study_name,
    )
    
    # Print results
    logger.info("\n=== Stage 2 Results ===")
    logger.info(f"Best value: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    # Save study
    output_path = f"results/stage2_{args.study_name}.pkl"
    hpo.save_study(study, output_path)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
