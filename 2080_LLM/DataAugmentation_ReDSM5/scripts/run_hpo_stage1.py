#!/usr/bin/env python3
"""
Run Stage 1 HPO: Augmenter Selection

Searches over valid augmentation combinations to find the best one.

Usage:
    python scripts/run_hpo_stage1.py [--config CONFIG] [--trials N] [--study-name NAME]
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
from utils.logging import setup_logger
from utils.hashing import generate_cache_filename
import yaml
import optuna


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Stage 1 HPO: Augmenter Selection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/run.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--combos",
        type=str,
        default="data/redsm5/combos/valid_combos.json",
        help="Path to valid combinations JSON",
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
        default="stage1_augmenter_selection",
        help="Study name",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///hpo_stage1.db",
        help="Optuna storage URL",
    )
    
    return parser.parse_args()


def objective(trial, valid_combos, config, logger, trainer, val_df):
    """
    Objective function for Stage 1.
    
    Args:
        trial: Optuna trial
        valid_combos: Dictionary of valid combinations
        config: Configuration dictionary
        logger: Logger instance
        trainer: AugmentationTrainer instance
        val_df: Validation DataFrame
        
    Returns:
        Validation F1 score
    """
    # Suggest combo
    hpo = HPOSearch(stage=1)
    combo = hpo.suggest_combo(trial, valid_combos)
    
    logger.info(f"\nTrial {trial.number}: Testing combo {' -> '.join(combo)}")
    
    # Load augmented training data from cache
    from aug.compose import AugmentationPipeline
    pipeline = AugmentationPipeline(combo=combo, seed=config["global"]["seed"])
    combo_hash = pipeline.get_combo_hash()
    
    cache_dir = Path(config["io"]["cache"]["combos_dir"])
    train_cache = cache_dir / generate_cache_filename(combo_hash, "train")
    
    if not train_cache.exists():
        logger.warning(f"Cache not found: {train_cache}")
        return 0.0
    
    # Load cached data
    parquet_io = ParquetIO()
    train_df = parquet_io.read_dataframe(train_cache)
    
    # Train model
    hyperparams = {
        "learning_rate": config["training"]["learning_rate"],
        "batch_size": config["training"]["batch_size"],
        "max_epochs": config["training"]["num_epochs"],
        "weight_decay": config["training"]["weight_decay"],
        "warmup_ratio": config["training"]["warmup_ratio"],
    }
    
    metrics, model = trainer.train(
        train_df,
        val_df,
        hyperparams,
        output_dir=f"checkpoints/stage1/trial_{trial.number}",
        trial=trial,
    )
    
    f1_score = metrics["eval_f1_macro"]
    logger.info(f"Trial {trial.number} F1: {f1_score:.4f}")
    
    return f1_score


def main():
    """Main function."""
    args = parse_args()
    
    # Setup logger
    logger = setup_logger("hpo_stage1", level="INFO")
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load valid combinations
    logger.info(f"Loading combinations from {args.combos}...")
    with open(args.combos) as f:
        valid_combos = {int(k): v for k, v in json.load(f).items()}
    
    total_combos = sum(len(v) for v in valid_combos.values())
    logger.info(f"Found {total_combos} valid combinations")
    
    # Load validation data (not augmented)
    logger.info("Loading validation data...")
    loader = REDSM5Loader(
        base_path=config["dataset"]["base_path"],
        text_field=config["dataset"]["text_field"],
        label_fields=config["dataset"]["label_fields"],
    )
    val_df = loader.load_from_parquet(splits=["val"])["val"]
    logger.info(f"Validation set: {len(val_df)} examples")
    
    # Initialize trainer
    trainer = AugmentationTrainer(
        model_name=config["model"]["name"],
        num_labels=config["model"]["num_labels"],
    )
    
    # Initialize HPO
    hpo = HPOSearch(config_path=args.config, stage=1)
    
    # Create objective with fixed arguments
    objective_fn = lambda trial: objective(
        trial, valid_combos, config, logger, trainer, val_df
    )
    
    # Run optimization
    logger.info(f"\nStarting Stage 1 HPO with {args.trials or hpo.stage_config['trials']} trials...")
    study = hpo.run_optimization(
        objective_fn,
        n_trials=args.trials,
        study_name=args.study_name,
    )
    
    # Print results
    logger.info("\n=== Stage 1 Results ===")
    logger.info(f"Best value: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    # Get best combo
    best_k = study.best_params["k"]
    best_combo_idx = study.best_params["combo_idx"]
    best_combo = valid_combos[best_k][best_combo_idx]
    
    logger.info(f"Best combo: {' -> '.join(best_combo)}")
    
    # Save study
    output_path = f"results/stage1_{args.study_name}.pkl"
    hpo.save_study(study, output_path)
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
