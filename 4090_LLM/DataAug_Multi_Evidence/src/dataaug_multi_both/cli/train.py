#!/usr/bin/env python
"""
Training CLI for storage-optimized HPO pipeline.

This module provides the command-line interface for running:
- Single trial training
- HPO study execution
- Resume interrupted studies
- Configuration validation (dry-run)

Usage:
    # Single trial
    python -m dataaug_multi_both.cli.train --config configs/single_trial.yaml --mode single

    # HPO study
    python -m dataaug_multi_both.cli.train --config configs/hpo_study.yaml --mode hpo \\
        --study-db experiments/hpo_study.db --mlflow-uri file://experiments/mlflow_db

    # Resume
    python -m dataaug_multi_both.cli.train --config configs/hpo_study.yaml --mode hpo \\
        --study-db experiments/hpo_study.db --mlflow-uri file://experiments/mlflow_db --resume

    # Dry-run validation
    python -m dataaug_multi_both.cli.train --config configs/hpo_study.yaml --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import mlflow
import torch
import yaml  # type: ignore[import-untyped]
from dataaug_multi_both.cache.dataset_cache import (
    CACHE_ROOT_DEFAULT,
    CacheIndex,
    TokenizedDataset,
    compute_cache_key,
    try_load_tokenized_cache,
)

from dataaug_multi_both.checkpoints.retention import RetentionPolicy
from dataaug_multi_both.data import (
    DatasetConfig,
    DatasetConfigurationError,
    DatasetLoader,
    build_dataset_config_from_dict,
    create_collator,
)
from dataaug_multi_both.hpo import OptunaHPOOptimizer
from dataaug_multi_both.models import EvidenceExtractionModel
from dataaug_multi_both.training import EvidenceExtractionTrainer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from dataaug_multi_both.utils.resources import build_dataloader_kwargs

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Storage-Optimized Training & HPO Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to configuration YAML file (e.g., configs/hpo_study.yaml)",
    )

    parser.add_argument(
        "--mode",
        choices=["single", "hpo"],
        default="single",
        help="Execution mode: 'single' for one trial, 'hpo' for study",
    )

    parser.add_argument(
        "--study-db",
        type=Path,
        help="Path to Optuna study database (required for HPO mode)",
    )

    parser.add_argument(
        "--mlflow-uri",
        type=str,
        help="MLflow tracking URI (required for HPO mode)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for single trial (overrides config)",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume interrupted HPO study from latest checkpoint",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running training",
    )

    return parser.parse_args()


def validate_config(config_path: Path) -> dict[str, Any]:
    """Load and validate configuration file."""
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse configuration file: {e}")
        sys.exit(1)

    if not config:
        logger.error("Configuration file is empty")
        sys.exit(1)

    logger.info(f"Loaded configuration from {config_path}")
    return config  # type: ignore[no-any-return]


def validate_hpo_args(args: argparse.Namespace) -> None:
    """Validate HPO-specific arguments."""
    if args.mode == "hpo":
        if not args.study_db:
            logger.error("--study-db is required for HPO mode")
            sys.exit(1)
        if not args.mlflow_uri:
            logger.error("--mlflow-uri is required for HPO mode")
            sys.exit(1)


def _resolve_dataset_config(config: dict[str, Any]) -> DatasetConfig:
    """
    Build a DatasetConfig object from CLI configuration.

    Supports either an inline ``dataset`` section or a ``dataset.config_path`` pointing to a YAML
    file under ``configs/data``. Inline keys override the referenced YAML when both are present.
    """

    dataset_settings = dict(config.get("dataset") or {})
    config_path_value = dataset_settings.pop("config_path", None)

    base_section: dict[str, Any] = {}
    config_dir: Path | None = Path.cwd()

    if config_path_value:
        config_path = Path(config_path_value)
        if not config_path.is_absolute():
            config_path = Path.cwd() / config_path
        try:
            config_path = config_path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise DatasetConfigurationError(
                f"Dataset config file not found: {config_path_value}"
            ) from exc

        with open(config_path, encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        section = payload.get("dataset")
        if not isinstance(section, dict):
            raise DatasetConfigurationError(
                f"Dataset config {config_path} does not define a 'dataset' mapping."
            )
        base_section = dict(section)
        config_dir = config_path.parent

    merged_section = dict(base_section)
    merged_section.update(dataset_settings)

    if not merged_section:
        raise DatasetConfigurationError(
            "Dataset configuration is empty. Provide dataset.config_path or inline dataset settings."
        )

    return build_dataset_config_from_dict(merged_section, config_dir=config_dir)


def run_single_trial(config: dict, args: argparse.Namespace) -> None:
    """Run a single training trial."""
    logger.info("=" * 80)
    logger.info("SINGLE TRIAL MODE")
    logger.info("=" * 80)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    default_output_dir = Path("outputs/training") / "trial_debug"
    configured_output = args.output_dir or config.get("output_dir")
    output_dir = Path(configured_output) if configured_output else default_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Extract configuration
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    data_config = config.get("data", {})
    retention_config = config.get("checkpoint_retention", {})

    # Log configuration
    logger.info("\nConfiguration summary:")
    logger.info(f"  Model: {model_config.get('model_id', 'google-bert/bert-base-uncased')}")
    logger.info(f"  Epochs: {training_config.get('epochs', 10)}")
    logger.info(f"  Batch size: {training_config.get('batch_size', 16)}")
    logger.info(f"  Learning rate: {training_config.get('learning_rate', 5e-5)}")

    try:
        # Start MLflow run
        mlflow.set_experiment(config.get("experiment_name", "single_trial"))
        with mlflow.start_run(run_name="single_trial"):
            # Log parameters
            mlflow.log_params(model_config)
            mlflow.log_params(training_config)

            # Load datasets
            logger.info("\n📊 Loading datasets...")
            dataset_config = _resolve_dataset_config(config)
            dataset_loader = DatasetLoader()
            dataset_splits = dataset_loader.load(dataset_config)
            train_dataset = dataset_splits["train"].with_format("python")
            val_dataset = dataset_splits["validation"].with_format("python")

            max_length = int(
                training_config.get(
                    "max_length",
                    data_config.get("max_length", 512),
                )
            )

            # Attempt tokenized cache lookup; else create collator for on-the-fly tokenization
            cache_root = Path(str(CACHE_ROOT_DEFAULT))
            index = CacheIndex(cache_root)
            try:
                dataset_section = config.get("dataset", {})
                dataset_files: list[Path] = []
                if isinstance(dataset_section, dict) and dataset_section.get("config_path"):
                    ds_cfg_path = Path(dataset_section["config_path"]).resolve()
                    with open(ds_cfg_path, encoding="utf-8") as f:
                        _ds_yaml = yaml.safe_load(f)
                    _ds = _ds_yaml.get("dataset", {})
                    data_files = _ds.get("data_files")
                    base_dir = ds_cfg_path.parent
                    if isinstance(data_files, dict):
                        for v in data_files.values():
                            if isinstance(v, str):
                                dataset_files.append((base_dir / v).resolve())
                            elif isinstance(v, list):
                                for it in v:
                                    dataset_files.append((base_dir / it).resolve())
                    elif isinstance(data_files, list):
                        for it in data_files:
                            dataset_files.append((base_dir / it).resolve())
                key = compute_cache_key(
                    dataset_files,
                    tokenizer_model=str(model_config.get("model_id", "google-bert/bert-base-uncased")),
                    max_length=max_length,
                    aug_params=None,
                )
                cached_train = try_load_tokenized_cache(cache_root, key, "train", index)
                cached_val = try_load_tokenized_cache(cache_root, key, "validation", index)
                if cached_train is not None and cached_val is not None:
                    train_dataset = cached_train
                    val_dataset = cached_val
                    collator = None
                    logger.info("Cache hit: using tokenized cache for train/validation")
                else:
                    logger.info("Cache miss: proceeding with on-the-fly tokenization")
                    collator = create_collator(
                        model_name_or_path=model_config.get("model_id", "google-bert/bert-base-uncased"),
                        max_length=max_length,
                    )
            except Exception as _cache_exc:
                logger.warning("Cache lookup failed: %s", _cache_exc)
                collator = create_collator(
                    model_name_or_path=model_config.get("model_id", "google-bert/bert-base-uncased"),
                    max_length=max_length,
                )

            collate_kwargs = {"collate_fn": collator} if collator is not None else {}

            # Unified DataLoader settings
            resources_config = config.get("resources", {})
            base_loader_kwargs, train_loader_kwargs = build_dataloader_kwargs(
                resources_cfg=resources_config or {}, training_cfg=training_config or {}
            )

            # Create data loaders
            # Adaptive batch-size fallback on CUDA OOM
            initial_bs = int(training_config.get("batch_size", 16))
            current_bs = max(1, initial_bs)
            attempts = 0
            while True:
                attempts += 1
                try:
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=current_bs,
                        shuffle=True,
                        **collate_kwargs,
                        **train_loader_kwargs,
                    )

                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=current_bs,
                        shuffle=False,
                        **collate_kwargs,
                        **base_loader_kwargs,
                    )

                    logger.info(f"Train samples: {len(train_dataset)}")
                    logger.info(f"Validation samples: {len(val_dataset)}")

                    # Create model
                    logger.info("\n🤖 Initializing model...")
                    model = EvidenceExtractionModel(
                        model_name_or_path=model_config.get("model_id", "google-bert/bert-base-uncased"),
                        head_type=model_config.get("evidence_head_type", "start_end_linear"),
                        dropout=model_config.get("dropout", 0.1),
                    )

                    # Create optimizer
                    adamw_kwargs: dict[str, Any] = {
                        "lr": training_config.get("learning_rate", 5e-5),
                        "weight_decay": training_config.get("weight_decay", 0.01),
                    }
                    if (
                        training_config.get("adam_beta1") is not None
                        and training_config.get("adam_beta2") is not None
                    ):
                        adamw_kwargs["betas"] = (
                            float(training_config["adam_beta1"]),
                            float(training_config["adam_beta2"]),
                        )
                    if training_config.get("adam_eps") is not None:
                        adamw_kwargs["eps"] = float(training_config["adam_eps"])

                    fused_optimizer = bool(training_config.get("fused_optimizer", True)) and torch.cuda.is_available()
                    try:
                        if fused_optimizer:
                            optimizer = AdamW(model.parameters(), fused=True, **adamw_kwargs)
                        else:
                            optimizer = AdamW(model.parameters(), **adamw_kwargs)
                    except TypeError as e:
                        if fused_optimizer:
                            logger.debug(
                                "AdamW fused optimizer unsupported (%s); retrying without fused flag.", e
                            )
                            optimizer = AdamW(model.parameters(), **adamw_kwargs)
                        else:
                            raise

                    # Create scheduler (optional)
                    scheduler = None
                    if training_config.get("use_scheduler", False):
                        scheduler = CosineAnnealingLR(
                            optimizer,
                            T_max=training_config.get("epochs", 10),
                        )

                    # Create retention policy
                    retention_policy = RetentionPolicy(
                        keep_last_n=retention_config.get("keep_last_n", 3),
                        keep_best_k=retention_config.get("keep_best_k", 5),
                        max_checkpoint_size_gb=retention_config.get("max_checkpoint_size_gb", 50.0),
                    )

                    # Create trainer
                    logger.info("\n🚀 Starting training...")
                    trainer = EvidenceExtractionTrainer(
                        model=model,
                        optimizer=optimizer,
                        train_dataloader=train_loader,
                        val_dataloader=val_loader,
                        output_dir=output_dir,
                        scheduler=scheduler,
                        retention_policy=retention_policy,
                        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
                        max_grad_norm=training_config.get("max_grad_norm", 1.0),
                        early_stopping_patience=training_config.get("early_stopping_patience"),
                        metric_for_best_model=training_config.get("metric_for_best_model", "val_f1"),
                        mlflow_tracking=True,
                        use_amp=training_config.get("amp", True),
                        allow_tf32=training_config.get("allow_tf32", True),
                        non_blocking=training_config.get("non_blocking", True),
                    )

                    # Train
                    result = trainer.train(
                        num_epochs=training_config.get("epochs", 10),
                        resume_from_checkpoint=None,
                    )

                    # Log final results
                    logger.info("\n✅ Training complete!")
                    logger.info(f"Best metric: {result['best_metric']:.4f}")
                    logger.info(f"Total epochs: {result['total_epochs']}")
                    logger.info(f"Global step: {result['global_step']}")

                    mlflow.log_metric("final_best_metric", result["best_metric"])
                    if current_bs != initial_bs:
                        mlflow.set_tag("adaptive_bs", f"reduced_from_{initial_bs}_to_{current_bs}")
                    break

                except RuntimeError as e:
                    msg = str(e).lower()
                    if ("out of memory" in msg or "cuda error" in msg) and current_bs > 1:
                        try:
                            import gc
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass
                        new_bs = max(1, current_bs // 2)
                        mlflow.set_tag("oom.retry", f"bs_{current_bs}_to_{new_bs}_attempt_{attempts}")
                        if new_bs == current_bs:
                            raise
                        current_bs = new_bs
                        continue
                    raise

    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        raise


def run_hpo_study(config: dict, args: argparse.Namespace) -> None:
    """Run HPO study with Optuna."""
    logger.info("=" * 80)
    logger.info("HPO STUDY MODE")
    logger.info("=" * 80)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    logger.info(f"Study database: {args.study_db}")
    logger.info(f"MLflow URI: {args.mlflow_uri}")
    logger.info(f"Resume: {args.resume}")

    # Extract configuration
    study_config = config.get("study", {})
    search_space = config.get("search_space", {})
    study_name = study_config.get("name", "mental_health_hpo")

    # Ensure HPO trial checkpoints are written under outputs/hpo/<study_name>/trial_*
    training_cfg = config.setdefault("training", {})
    training_cfg.setdefault("output_dir", str(Path("outputs/hpo")))
    hpo_output_root = Path("outputs/hpo") / study_name
    hpo_output_root.mkdir(parents=True, exist_ok=True)

    # Log configuration
    logger.info("\nStudy configuration:")
    logger.info(f"  Name: {study_name}")
    logger.info(f"  Trials: {study_config.get('n_trials', 100)}")
    logger.info(f"  Direction: {study_config.get('direction', 'maximize')}")
    logger.info(f"\nSearch space parameters: {len(search_space)} parameters")
    for param, spec in list(search_space.items())[:5]:
        logger.info(f"  {param}: {spec.get('type', 'N/A')}")

    try:
        # Create storage URL (special-case in-memory)
        if str(args.study_db) == ":memory:":
            storage_url = "sqlite:///:memory:"
        else:
            storage_url = f"sqlite:///{args.study_db}"
        logger.info(f"\n📁 Storage: {storage_url}")

        # Initialize HPO optimizer
        logger.info("\n🔬 Initializing Optuna HPO optimizer...")
        optimizer = OptunaHPOOptimizer(
            study_name=study_name,
            storage=storage_url,
            direction=study_config.get("direction", "maximize"),
            mlflow_tracking_uri=args.mlflow_uri,
        )

        # Run optimization
        logger.info("\n🚀 Starting HPO study...")
        study = optimizer.run_optimization(
            config=config,
            n_trials=study_config.get("n_trials", 100),
            timeout=study_config.get("timeout"),
            resume=args.resume,
        )

        # Log results
        logger.info("\n✅ HPO study complete!")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value:.4f}")
        logger.info("\nBest hyperparameters:")
        for param, value in study.best_params.items():
            logger.info(f"  {param}: {value}")

        # Export results
        results_path = hpo_output_root / "study_results.csv"
        logger.info(f"\n💾 Exporting results to {results_path}")
        optimizer.export_study_results(results_path, study)

    except Exception as e:
        logger.error(f"\n❌ HPO study failed: {e}")
        raise


def dry_run(config: dict, args: argparse.Namespace) -> None:
    """Validate configuration without running training."""
    logger.info("=" * 80)
    logger.info("DRY-RUN MODE - Configuration Validation")
    logger.info("=" * 80)

    errors = []
    warnings = []

    # Validate required sections
    required_sections = {
        "hpo": ["study", "search_space", "dataset"],
        "single": ["model", "training", "dataset"],
    }

    mode = args.mode
    for section in required_sections.get(mode, []):
        if section not in config:
            errors.append(f"Missing required section: {section}")

    # Validate dataset config
    dataset_section = config.get("dataset")
    require_dataset = mode == "single" or bool(dataset_section)
    if require_dataset:
        try:
            _resolve_dataset_config(config)
        except DatasetConfigurationError as exc:
            errors.append(str(exc))
    else:
        warnings.append("No dataset configuration supplied; assuming synthetic data.")

    # Validate HPO-specific config
    if mode == "hpo":
        study = config.get("study", {})
        if "name" not in study:
            warnings.append("Study name not specified")
        if "n_trials" not in study or study["n_trials"] <= 0:
            errors.append("Invalid or missing study.n_trials")

        search_space = config.get("search_space", {})
        if not search_space:
            errors.append("Search space is empty")

    # Report results
    if errors:
        logger.error("\n❌ Validation FAILED with errors:")
        for error in errors:
            logger.error(f"  • {error}")
        sys.exit(1)

    if warnings:
        logger.warning("\n⚠️  Validation passed with warnings:")
        for warning in warnings:
            logger.warning(f"  • {warning}")

    logger.info("\n✅ Configuration validation PASSED")
    logger.info("Ready to run training!")


def main() -> None:
    """Main entry point for training CLI."""
    args = parse_args()

    logger.info("Storage-Optimized Training & HPO Pipeline")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")

    # Load and validate configuration
    config = validate_config(args.config)

    # Dry-run mode
    if args.dry_run:
        dry_run(config, args)
        return

    # Validate mode-specific arguments
    validate_hpo_args(args)

    # Execute based on mode
    if args.mode == "single":
        run_single_trial(config, args)
    elif args.mode == "hpo":
        run_hpo_study(config, args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

    logger.info("\n" + "=" * 80)
    logger.info("EXECUTION COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
