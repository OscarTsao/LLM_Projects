# File: src/cli.py
"""Command line interface for criteria binding model."""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, cast

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from .utils.logging import setup_logging
from .utils.seed import set_seed
from .training.train import create_trainer_from_config
from .training.eval import run_inference
from .utils.io import load_yaml


def setup_cli_logging(log_level: str = "INFO") -> None:
    """Set up logging for CLI."""
    setup_logging(
        level=log_level,
        format_str="%(asctime)s - %(levelname)s - %(message)s"
    )


def train_command(cfg: DictConfig) -> None:
    """Handle the train command."""
    setup_cli_logging(cfg.logging.log_level)
    logger = logging.getLogger(__name__)

    try:
        config_dict = cast(Dict[str, Any], OmegaConf.to_container(cfg, resolve=True))

        # Set random seed
        set_seed(config_dict["logging"]["seed"])

        # Load datasets
        logger.info(f"Loading training data from {config_dict['data']['train_path']}")
        from .data.dataset import CriteriaBindingDataset

        train_dataset = CriteriaBindingDataset(config_dict["data"]["train_path"])

        eval_dataset = None
        if config_dict["data"]["dev_path"]:
            logger.info(f"Loading eval data from {config_dict['data']['dev_path']}")
            eval_dataset = CriteriaBindingDataset(config_dict["data"]["dev_path"])

        # Log dataset statistics
        train_stats = train_dataset.get_statistics()
        logger.info(f"Training dataset: {train_stats['num_examples']} examples")
        if eval_dataset:
            eval_stats = eval_dataset.get_statistics()
            logger.info(f"Eval dataset: {eval_stats['num_examples']} examples")

        # Create trainer
        trainer = create_trainer_from_config(config_dict, train_dataset, eval_dataset)

        # Start training
        results = trainer.train()

        logger.info("Training completed successfully!")
        logger.info(f"Best checkpoint: {results['best_checkpoint']}")
        logger.info(f"Total training time: {results['total_time']:.2f} seconds")

        # Print final metrics
        final_metrics = results["final_metrics"]
        logger.info("Final metrics:")
        for key, value in final_metrics.items():
            if isinstance(value, (int, float)) and key not in ["epoch", "global_step"]:
                logger.info(f"  {key}: {value:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        sys.exit(1)


def eval_command(cfg: DictConfig) -> None:
    """Handle the eval command."""
    setup_cli_logging(cfg.logging.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Load config from checkpoint
        checkpoint_path = Path(cfg.eval.checkpoint)
        config_path = checkpoint_path / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")

        config = load_yaml(config_path)

        # Determine data path based on split
        if cfg.eval.split == "dev":
            data_path = config["data"]["dev_path"]
        elif cfg.eval.split == "test":
            data_path = config["data"]["test_path"]
        else:
            raise ValueError(f"Unknown split: {cfg.eval.split}")

        if not data_path:
            raise ValueError(f"No data path configured for split: {cfg.eval.split}")

        # Set output path
        output_path = cfg.eval.output_path
        if not output_path:
            output_path = checkpoint_path / f"{cfg.eval.split}_predictions.jsonl"

        logger.info(f"Evaluating on {cfg.eval.split} split")
        logger.info(f"Data: {data_path}")
        logger.info(f"Output: {output_path}")

        # Run inference
        results = run_inference(
            model_path=str(checkpoint_path),
            data_path=data_path,
            output_path=str(output_path),
            config=config,
        )

        logger.info(f"Evaluation completed on {results['num_examples']} examples")

        # Print metrics
        if results["metrics"]:
            logger.info("Evaluation metrics:")
            for key, value in results["metrics"].items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


def predict_command(cfg: DictConfig) -> None:
    """Handle the predict command."""
    setup_cli_logging(cfg.logging.log_level)
    logger = logging.getLogger(__name__)

    try:
        # Load config from checkpoint
        checkpoint_path = Path(cfg.predict.checkpoint)
        config_path = checkpoint_path / "config.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Config not found at {config_path}")

        config = load_yaml(config_path)

        logger.info(f"Running prediction on {cfg.predict.input_path}")
        logger.info(f"Output: {cfg.predict.output_path}")

        # Run inference
        results = run_inference(
            model_path=str(checkpoint_path),
            data_path=cfg.predict.input_path,
            output_path=cfg.predict.output_path,
            config=config,
        )

        logger.info(f"Prediction completed on {results['num_examples']} examples")

        # Print metrics if available
        if results["metrics"]:
            logger.info("Metrics (if gold data available):")
            for key, value in results["metrics"].items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


@hydra.main(config_path="config", config_name="train", version_base="1.3")
def train_entrypoint(cfg: DictConfig) -> None:
    train_command(cfg)


@hydra.main(config_path="config", config_name="eval", version_base="1.3")
def eval_entrypoint(cfg: DictConfig) -> None:
    eval_command(cfg)


@hydra.main(config_path="config", config_name="predict", version_base="1.3")
def predict_entrypoint(cfg: DictConfig) -> None:
    predict_command(cfg)


def main() -> None:
    """Main CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.cli [train|eval|predict] ...")
        sys.exit(1)

    command = sys.argv[1]

    # Reset Hydra to allow multiple entrypoints in same process
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()

    sys.argv = sys.argv[1:]

    if command == "train":
        train_entrypoint()
    elif command == "eval":
        eval_entrypoint()
    elif command == "predict":
        predict_entrypoint()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()