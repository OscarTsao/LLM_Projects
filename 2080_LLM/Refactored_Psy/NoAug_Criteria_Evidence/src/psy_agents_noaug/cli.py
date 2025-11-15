#!/usr/bin/env python3
"""Unified command-line interface with Hydra integration.

This CLI provides a consistent interface for all operations:
- make_groundtruth: Generate ground truth files from data
- train: Train a model with specified config
- hpo: Run hyperparameter optimization stage
- refit: Refit best model on full train+val
- evaluate_best: Evaluate best model on test set
- export_metrics: Export metrics table from MLflow

Usage:
    # Generate ground truth
    python -m psy_agents_noaug.cli make_groundtruth data=hf_redsm5
    
    # Train model
    python -m psy_agents_noaug.cli train task=criteria model=roberta_base
    
    # Run HPO
    python -m psy_agents_noaug.cli hpo hpo=stage1_coarse task=criteria
    
    # Evaluate
    python -m psy_agents_noaug.cli evaluate_best task=criteria checkpoint=outputs/best_model.pt
    
    # Export metrics
    python -m psy_agents_noaug.cli export_metrics experiment=noaug_baseline
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def _load_training_resources(cfg: DictConfig, project_root: Path):
    """Preload datasets, field maps, and splits for training routines."""
    from psy_agents_noaug.data.datasets import load_splits
    from psy_agents_noaug.data.loaders import ReDSM5DataLoader, load_field_map

    field_map_path = project_root / cfg.data.field_map
    field_map = load_field_map(field_map_path)

    data_dir = cfg.data.get("data_dir")
    data_dir = project_root / data_dir if data_dir else None
    hf_cache_dir = cfg.data.get("hf_cache_dir")
    hf_cache_dir = project_root / hf_cache_dir if hf_cache_dir else None

    loader = ReDSM5DataLoader(
        field_map=field_map,
        data_source=cfg.data.source,
        data_dir=data_dir,
        hf_dataset_name=cfg.data.get("hf_dataset_name"),
        hf_cache_dir=hf_cache_dir,
        posts_file=cfg.data.get("posts_file", "posts.csv"),
        annotations_file=cfg.data.get("annotations_file", "annotations.csv"),
    )

    posts_df = loader.load_posts()
    dsm_entries = loader.load_dsm_criteria(project_root / cfg.data.dsm_criteria)

    processed_dir = project_root / cfg.data.output_dir
    criteria_path = processed_dir / "criteria_groundtruth.csv"
    evidence_path = processed_dir / "evidence_groundtruth.csv"
    splits_path = processed_dir / "splits.json"

    if not criteria_path.exists() or not splits_path.exists():
        raise FileNotFoundError(
            "Ground-truth files are missing. Run `make_groundtruth` before training."
        )

    criteria_df = pd.read_csv(criteria_path)
    if evidence_path.exists():
        evidence_df = pd.read_csv(evidence_path)
    else:
        evidence_df = pd.DataFrame(columns=["post_id", "criterion_id", "evidence_text"])

    splits = load_splits(splits_path)

    return {
        "field_map": field_map,
        "posts_df": posts_df,
        "dsm_entries": dsm_entries,
        "criteria_df": criteria_df,
        "evidence_df": evidence_df,
        "splits": splits,
        "processed_dir": processed_dir,
    }


def _train_once(
    cfg: DictConfig,
    project_root: Path,
    device: torch.device,
    seed: int,
    resources: Optional[dict] = None,
    track_mlflow: bool = True,
    output_suffix: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
) -> Dict[str, Dict]:
    """Run a single training job and return collected metrics."""
    from psy_agents_noaug.data.datasets import build_datasets
    from psy_agents_noaug.models.criteria_head import CriteriaModel
    from psy_agents_noaug.models.encoders import TransformerEncoder
    from psy_agents_noaug.training.evaluate import (
        Evaluator,
        print_evaluation_results,
    )
    from psy_agents_noaug.training.setup import (
        compute_total_steps,
        create_optimizer,
        create_scheduler,
    )
    from psy_agents_noaug.training.train_loop import Trainer
    from psy_agents_noaug.utils.mlflow_utils import (
        configure_mlflow,
        end_run,
        log_artifacts,
        log_config,
        log_evaluation_report,
        log_metrics_dict,
        log_model_checkpoint,
    )
    from psy_agents_noaug.utils.reproducibility import set_seed

    set_seed(seed, cfg.get("deterministic", True), cfg.get("cudnn_benchmark", False))

    resources = resources or _load_training_resources(cfg, project_root)

    field_map = resources["field_map"]
    posts_df = resources["posts_df"]
    dsm_entries = resources["dsm_entries"]
    criteria_df = resources["criteria_df"]
    evidence_df = resources["evidence_df"]
    splits = resources["splits"]

    outputs_dir = project_root / cfg.paths.outputs
    suffix = output_suffix or cfg.task.name
    if track_mlflow:
        checkpoints_dir = outputs_dir / "checkpoints" / suffix
        evaluation_dir = outputs_dir / "evaluation" / suffix
    else:
        checkpoints_dir = outputs_dir / "hpo" / suffix / "checkpoints"
        evaluation_dir = outputs_dir / "hpo" / suffix / "evaluation"

    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)

    run_id = None
    if track_mlflow:
        tracking_uri = cfg.mlflow.tracking_uri
        if not tracking_uri.startswith("http"):
            tracking_uri = str((project_root / tracking_uri).resolve())
        run_id = configure_mlflow(
            tracking_uri=tracking_uri,
            experiment_name=cfg.mlflow.experiment_name,
            run_name=run_name or cfg.mlflow.run_name or f"{cfg.task.name}_{cfg.model.encoder_name}",
            tags=tags or {str(k): str(v) for k, v in cfg.mlflow.get("tags", {}).items()},
            config=cfg,
        )
        log_config(cfg)

    try:
        encoder = TransformerEncoder(
            model_name=cfg.model.encoder_name,
            pooling_strategy="cls",
            max_length=cfg.model.max_length,
            gradient_checkpointing=cfg.model.get("gradient_checkpointing", False),
            lora_config=cfg.model.get("lora"),
        )

        datasets, criterion_to_index, index_to_criterion = build_datasets(
            task_name=cfg.task.name,
            tokenizer=encoder.tokenizer,
            max_length=cfg.model.max_length,
            field_map=field_map,
            posts_df=posts_df,
            dsm_entries=dsm_entries,
            criteria_groundtruth=criteria_df,
            evidence_groundtruth=evidence_df,
            splits=splits,
        )

        train_loader = DataLoader(
            datasets.train,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
        )
        val_loader = DataLoader(
            datasets.val,
            batch_size=cfg.training.eval_batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
        )
        test_loader = DataLoader(
            datasets.test,
            batch_size=cfg.training.eval_batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=cfg.training.pin_memory,
        )

        model = CriteriaModel(
            encoder=encoder,
            num_classes=2,
            dropout=cfg.model.get("dropout", 0.1),
            hidden_dim=cfg.model.get("head_hidden_dim"),
        ).to(device)

        if cfg.task.loss.name != "cross_entropy":
            raise ValueError(f"Unsupported loss: {cfg.task.loss.name}")

        class_weights = cfg.task.loss.get("class_weights")
        weight_tensor = (
            torch.tensor(class_weights, dtype=torch.float32, device=device)
            if class_weights
            else None
        )
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        optimizer = create_optimizer(model, cfg.training)
        total_steps = compute_total_steps(
            num_batches=len(train_loader),
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            num_epochs=cfg.training.num_epochs,
        )
        scheduler = create_scheduler(optimizer, cfg.training.scheduler, total_steps)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_epochs=cfg.training.num_epochs,
            patience=cfg.training.early_stopping.patience,
            gradient_clip=cfg.training.gradient_clip,
            gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
            scheduler=scheduler,
            save_dir=checkpoints_dir,
            use_amp=cfg.training.amp.enabled,
            amp_dtype=cfg.training.amp.dtype,
            early_stopping_metric=cfg.training.early_stopping.metric,
            early_stopping_mode=cfg.training.early_stopping.mode,
            min_delta=cfg.training.early_stopping.min_delta,
            logging_steps=cfg.training.logging_steps,
        )

        trainer.train()

        best_checkpoint = checkpoints_dir / "best_checkpoint.pt"
        if best_checkpoint.exists():
            best_state = torch.load(best_checkpoint, map_location=device)
            model.load_state_dict(best_state["model_state_dict"])

        evaluator = Evaluator(
            model=model,
            device=device,
            task_type=cfg.task.name,
            criterion=criterion,
        )
        val_metrics = evaluator.evaluate(val_loader, class_names=None)
        test_metrics = evaluator.evaluate(test_loader, class_names=None)

        print_evaluation_results(val_metrics, title="Validation Metrics")
        print_evaluation_results(test_metrics, title="Test Metrics")

        evaluation_report = {
            "task": cfg.task.name,
            "validation": val_metrics,
            "test": test_metrics,
            "criterion_index_to_id": index_to_criterion,
        }
        report_path = evaluation_dir / "evaluation_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)

        if track_mlflow:
            log_evaluation_report(evaluation_report, report_path)
            summary_metrics = {
                f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))
            }
            summary_metrics.update(
                {f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, (int, float))}
            )
            log_metrics_dict(summary_metrics)
            log_model_checkpoint(best_checkpoint, artifact_path=f"checkpoints/{cfg.task.name}")
            log_artifacts(evaluation_dir, artifact_path=f"reports/{cfg.task.name}")
        else:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(evaluation_report, f, indent=2)

        return {
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "checkpoints_dir": checkpoints_dir,
            "evaluation_dir": evaluation_dir,
            "run_id": run_id,
        }
    finally:
        if track_mlflow:
            end_run()
@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def make_groundtruth(cfg: DictConfig):
    """
    Generate ground truth files from raw data.
    
    This command:
    1. Loads posts, annotations, and DSM criteria
    2. Validates required columns per field_map.yaml
    3. Generates criteria_groundtruth.csv (uses ONLY status field)
    4. Generates evidence_groundtruth.csv (uses ONLY cases field)
    5. Generates splits.json with train/val/test post_ids
    6. Validates strict field separation
    
    Usage:
        python -m psy_agents_noaug.cli make_groundtruth data=local_csv
        python -m psy_agents_noaug.cli make_groundtruth data=hf_redsm5
        python -m psy_agents_noaug.cli make_groundtruth data.data_dir=./custom/path
    """
    from psy_agents_noaug.data.groundtruth import (
        GroundTruthValidator,
        create_criteria_groundtruth,
        create_evidence_groundtruth,
        load_field_map,
        validate_strict_separation,
    )
    from psy_agents_noaug.data.loaders import (
        ReDSM5DataLoader,
        group_split_by_post_id,
        save_splits_json,
    )
    
    print(f"\n{'=' * 80}")
    print(f"Ground Truth Generation".center(80))
    print(f"{'=' * 80}\n")
    
    # Resolve repository root regardless of Hydra's working directory
    project_root = Path(get_original_cwd())

    # Setup paths
    field_map_path = project_root / cfg.data.field_map
    output_dir = project_root / cfg.data.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load field mapping
    print(f"Loading field mapping from: {field_map_path}")
    field_map = load_field_map(field_map_path)
    print(f"  Status field (criteria): {field_map['annotations']['status']}")
    print(f"  Cases field (evidence): {field_map['annotations']['cases']}")
    
    # Initialize loader
    print(f"\nLoading data from: {cfg.data.source}")
    data_dir = cfg.data.get("data_dir")
    if data_dir:
        data_dir = project_root / data_dir
    hf_cache_dir = cfg.data.get("hf_cache_dir")
    if hf_cache_dir:
        hf_cache_dir = project_root / hf_cache_dir

    loader = ReDSM5DataLoader(
        field_map=field_map,
        data_source=cfg.data.source,
        data_dir=data_dir,
        hf_dataset_name=cfg.data.get("hf_dataset_name"),
        hf_cache_dir=hf_cache_dir,
        posts_file=cfg.data.get("posts_file", "posts.csv"),
        annotations_file=cfg.data.get("annotations_file", "annotations.csv"),
    )
    
    # Load data
    posts = loader.load_posts()
    print(f"  Posts: {len(posts)} rows, {posts['post_id'].nunique()} unique IDs")
    
    annotations = loader.load_annotations()
    print(f"  Annotations: {len(annotations)} rows")
    
    # Load DSM criteria
    dsm_path = project_root / cfg.data.dsm_criteria
    
    dsm_criteria = loader.load_dsm_criteria(dsm_path)
    valid_criterion_ids = {c["id"] for c in dsm_criteria}
    print(f"  DSM Criteria: {len(dsm_criteria)} criteria")
    
    # Create splits
    print(f"\nCreating splits (train={cfg.data.train_ratio}, val={cfg.data.val_ratio}, test={cfg.data.test_ratio})...")
    train_ids, val_ids, test_ids = group_split_by_post_id(
        df=annotations,
        post_id_col=field_map["annotations"]["post_id"],
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        test_ratio=cfg.data.test_ratio,
        random_seed=cfg.seed,
    )
    
    print(f"  Train: {len(train_ids)} posts")
    print(f"  Val: {len(val_ids)} posts")
    print(f"  Test: {len(test_ids)} posts")
    
    # Save splits
    splits_path = output_dir / "splits.json"
    save_splits_json(
        train_post_ids=train_ids,
        val_post_ids=val_ids,
        test_post_ids=test_ids,
        output_path=splits_path,
        metadata={
            "random_seed": cfg.seed,
            "train_ratio": cfg.data.train_ratio,
            "val_ratio": cfg.data.val_ratio,
            "test_ratio": cfg.data.test_ratio,
            "data_source": cfg.data.source,
        },
    )
    
    # Generate criteria groundtruth
    print(f"\nGenerating criteria groundtruth (using ONLY 'status' field)...")
    criteria_gt = create_criteria_groundtruth(
        annotations=annotations,
        posts=posts,
        field_map=field_map,
        valid_criterion_ids=valid_criterion_ids,
    )
    
    criteria_path = output_dir / "criteria_groundtruth.csv"
    criteria_gt.to_csv(criteria_path, index=False)
    print(f"  Saved: {criteria_path} ({len(criteria_gt)} rows)")
    print(f"  Label distribution:")
    for label, count in criteria_gt["label"].value_counts().items():
        print(f"    {label}: {count}")
    
    # Generate evidence groundtruth
    print(f"\nGenerating evidence groundtruth (using ONLY 'cases' field)...")
    evidence_gt = create_evidence_groundtruth(
        annotations=annotations,
        posts=posts,
        field_map=field_map,
        valid_criterion_ids=valid_criterion_ids,
    )
    
    evidence_path = output_dir / "evidence_groundtruth.csv"
    evidence_gt.to_csv(evidence_path, index=False)
    print(f"  Saved: {evidence_path} ({len(evidence_gt)} rows)")
    
    # Validate
    print(f"\nValidating strict field separation...")
    validate_strict_separation(criteria_gt, evidence_gt, field_map)
    
    validator = GroundTruthValidator(field_map, valid_criterion_ids)
    criteria_validation = validator.validate_criteria_groundtruth(criteria_gt)
    evidence_validation = validator.validate_evidence_groundtruth(evidence_gt)
    
    if criteria_validation["errors"] or evidence_validation["errors"]:
        print(f"  ERRORS FOUND - validation failed!")
        for error in criteria_validation["errors"] + evidence_validation["errors"]:
            print(f"    - {error}")
        sys.exit(1)
    
    print(f"  Validation passed!")
    
    print(f"\n{'=' * 80}")
    print(f"Ground truth generation complete!".center(80))
    print(f"{'=' * 80}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def train(cfg: DictConfig):
    """Train a model with the provided Hydra configuration."""

    from psy_agents_noaug.utils.reproducibility import get_device

    project_root = Path(get_original_cwd())

    print(f"\n{'=' * 80}")
    print(f"Training Task: {cfg.task.name}".center(80))
    print(f"{'=' * 80}\n")

    device = get_device(cfg.get("device", "cuda").lower() == "cuda")
    _train_once(
        cfg=cfg,
        project_root=project_root,
        device=device,
        seed=cfg.seed,
        track_mlflow=True,
    )



@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def hpo(cfg: DictConfig):
    """Run Optuna-based hyperparameter optimization."""

    from psy_agents_noaug.hpo.optuna_runner import (
        OptunaRunner,
        create_search_space_from_config,
    )
    from psy_agents_noaug.utils.mlflow_utils import (
        configure_mlflow,
        end_run,
        log_config,
        log_metrics_dict,
    )
    from psy_agents_noaug.utils.reproducibility import get_device, set_seed

    project_root = Path(get_original_cwd())

    print(f"\n{'=' * 80}")
    print(f"HPO Stage {cfg.hpo.stage}: {cfg.hpo.stage_name}".center(80))
    print(f"{'=' * 80}\n")

    set_seed(cfg.seed)

    tracking_uri = cfg.mlflow.tracking_uri
    if not tracking_uri.startswith("http"):
        tracking_uri = str((project_root / tracking_uri).resolve())

    experiment_name = f"{cfg.mlflow.experiment_name}-hpo-stage{cfg.hpo.stage}"
    run_id = configure_mlflow(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        config=cfg,
    )

    print(f"MLflow run ID: {run_id}")
    log_config(cfg)

    try:
        resources = _load_training_resources(cfg, project_root)
        device = get_device(cfg.get("device", "cuda").lower() == "cuda")

        runner = OptunaRunner(
            study_name=f"{cfg.task.name}_stage{cfg.hpo.stage}",
            direction=cfg.hpo.direction,
            metric=cfg.hpo.metric,
            sampler_config=cfg.hpo.get("sampler"),
            pruner_config=cfg.hpo.get("pruner"),
        )

        search_space = create_search_space_from_config(cfg.hpo)

        def objective(trial):
            params = runner.suggest_hyperparameters(trial, search_space)
            trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

            training = trial_cfg.training
            model_cfg = trial_cfg.model

            if "learning_rate" in params:
                training.learning_rate = params.pop("learning_rate")
            if "weight_decay" in params:
                training.weight_decay = params.pop("weight_decay")
            if "batch_size" in params:
                training.batch_size = params.pop("batch_size")
                training.eval_batch_size = max(training.batch_size * 2, training.batch_size)
            if "warmup_ratio" in params:
                training.scheduler.warmup_ratio = params.pop("warmup_ratio")
            if "dropout" in params:
                model_cfg.dropout = params.pop("dropout")
            if "max_length" in params:
                model_cfg.max_length = params.pop("max_length")
            if "scheduler" in params:
                training.scheduler.type = params.pop("scheduler")
            if "encoder" in params:
                model_cfg.encoder_name = params.pop("encoder")
            if "lora_enabled" in params:
                model_cfg.lora["enabled"] = params.pop("lora_enabled")
            if "lora_r" in params:
                model_cfg.lora["r"] = params.pop("lora_r")
            if "lora_alpha" in params:
                model_cfg.lora["alpha"] = params.pop("lora_alpha")

            if cfg.hpo.get("num_epochs"):
                training.num_epochs = cfg.hpo.num_epochs

            # Any remaining parameters are ignored but logged
            for unused_key, unused_value in params.items():
                trial.set_user_attr(f"unused_{unused_key}", unused_value)

            suffix = f"hpo_stage{cfg.hpo.stage}_trial{trial.number}"
            result = _train_once(
                cfg=trial_cfg,
                project_root=project_root,
                device=device,
                seed=cfg.seed + trial.number,
                resources=resources,
                track_mlflow=False,
                output_suffix=suffix,
            )

            trial.set_user_attr("val_metrics", result["val_metrics"])
            trial.set_user_attr("test_metrics", result["test_metrics"])

            metric_name = cfg.hpo.metric
            objective_metric = result["val_metrics"].get(metric_name)
            if objective_metric is None:
                raise ValueError(f"Metric '{metric_name}' not found in validation metrics")
            return objective_metric

        runner.optimize(
            objective_fn=objective,
            n_trials=cfg.hpo.n_trials,
            search_space=search_space,
            timeout=cfg.hpo.timeout,
            mlflow_tracking_uri=None,
        )

        best_trial = runner.get_best_trial()
        print("\nBest Trial Summary:")
        runner.print_best_trial()

        best_metrics = best_trial.user_attrs.get("val_metrics", {})
        if best_metrics:
            log_metrics_dict({f"best_val_{k}": v for k, v in best_metrics.items() if isinstance(v, (int, float))})

        output_root = project_root / cfg.paths.outputs / f"hpo_stage{cfg.hpo.stage}"
        output_root.mkdir(parents=True, exist_ok=True)
        runner.export_best_config(output_root / "best_config.yaml")
        runner.export_trials_history(output_root / "trials.json")

    finally:
        end_run()

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def refit(cfg: DictConfig):
    """
    Refit best model on full train+val dataset (HPO stage 3).
    
    Usage:
        python -m psy_agents_noaug.cli refit task=criteria best_config=outputs/hpo_stage2/best_config.yaml
    """
    from psy_agents_noaug.utils.mlflow_utils import configure_mlflow, log_config
    from psy_agents_noaug.utils.reproducibility import get_device, set_seed
    
    print(f"\n{'=' * 80}")
    print(f"Refitting Best Model: {cfg.task.name}".center(80))
    print(f"{'=' * 80}\n")
    
    # Load best config if provided
    if "best_config" in cfg and cfg.best_config:
        best_config = OmegaConf.load(cfg.best_config)
        cfg = OmegaConf.merge(cfg, best_config)
        print(f"Loaded best config from: {cfg.best_config}")
    
    # Set seed
    set_seed(cfg.seed, cfg.get("deterministic", True))
    
    # Get device
    device = get_device(cfg.device == "cuda")
    print(f"Device: {device}")
    
    # Configure MLflow
    experiment_name = f"{cfg.mlflow.experiment_name}-refit"
    run_id = configure_mlflow(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=experiment_name,
        config=cfg,
    )
    
    print(f"MLflow run ID: {run_id}")
    
    # Log config
    log_config(cfg)
    
    print(f"\nRefit logic not yet fully implemented.")
    print(f"This would train on train+val and evaluate on test.")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def evaluate_best(cfg: DictConfig):
    """
    Evaluate best model on test set.
    
    Usage:
        python -m psy_agents_noaug.cli evaluate_best checkpoint=outputs/best_model.pt
        python -m psy_agents_noaug.cli evaluate_best task=criteria checkpoint=outputs/checkpoints/best_checkpoint.pt
    """
    from psy_agents_noaug.utils.mlflow_utils import configure_mlflow
    from psy_agents_noaug.utils.reproducibility import get_device, set_seed
    
    print(f"\n{'=' * 80}")
    print(f"Evaluating Best Model: {cfg.task.name}".center(80))
    print(f"{'=' * 80}\n")
    
    # Set seed
    set_seed(cfg.seed)
    
    # Get device
    device = get_device(cfg.device == "cuda")
    print(f"Device: {device}")
    
    # Configure MLflow
    experiment_name = f"{cfg.mlflow.experiment_name}-eval"
    run_id = configure_mlflow(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=experiment_name,
        config=cfg,
    )
    
    print(f"MLflow run ID: {run_id}")
    
    if "checkpoint" in cfg:
        print(f"Checkpoint: {cfg.checkpoint}")
    else:
        print(f"WARNING: No checkpoint specified, use checkpoint=path/to/model.pt")
    
    print(f"\nEvaluation logic not yet fully implemented.")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def export_metrics(cfg: DictConfig):
    """
    Export metrics from MLflow to CSV/JSON.
    
    Usage:
        python -m psy_agents_noaug.cli export_metrics experiment=noaug_baseline
        python -m psy_agents_noaug.cli export_metrics mlflow.experiment_name=custom_exp output_dir=./results
    """
    import mlflow
    import pandas as pd
    
    print(f"\n{'=' * 80}")
    print(f"Exporting Metrics from MLflow".center(80))
    print(f"{'=' * 80}\n")
    
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    
    # Get experiment
    experiment = mlflow.get_experiment_by_name(cfg.mlflow.experiment_name)
    
    if not experiment:
        print(f"ERROR: Experiment '{cfg.mlflow.experiment_name}' not found")
        sys.exit(1)
    
    print(f"Experiment: {cfg.mlflow.experiment_name}")
    print(f"Experiment ID: {experiment.experiment_id}")
    
    # Get runs
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    if runs.empty:
        print(f"No runs found in experiment")
        sys.exit(1)
    
    print(f"Found {len(runs)} runs")
    
    # Export to CSV
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / f"{cfg.mlflow.experiment_name}_metrics.csv"
    runs.to_csv(csv_path, index=False)
    print(f"\nExported to: {csv_path}")
    
    # Also export summary
    json_path = output_dir / f"{cfg.mlflow.experiment_name}_summary.json"
    summary = {
        "experiment_name": cfg.mlflow.experiment_name,
        "experiment_id": experiment.experiment_id,
        "num_runs": len(runs),
        "columns": list(runs.columns),
    }
    
    with open(json_path, "w") as f:
        import json
        json.dump(summary, f, indent=2)
    
    print(f"Summary: {json_path}")


def main():
    """Main CLI entry point with subcommands."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PSY Agents NO-AUG: Training infrastructure with Hydra, MLflow, and Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate ground truth
  python -m psy_agents_noaug.cli make_groundtruth data=hf_redsm5
  
  # Train model
  python -m psy_agents_noaug.cli train task=criteria model=roberta_base
  
  # Run HPO
  python -m psy_agents_noaug.cli hpo hpo=stage1_coarse task=criteria
  
  # Refit best model
  python -m psy_agents_noaug.cli refit task=criteria best_config=outputs/best.yaml
  
  # Evaluate
  python -m psy_agents_noaug.cli evaluate_best checkpoint=outputs/best_model.pt
  
  # Export metrics
  python -m psy_agents_noaug.cli export_metrics experiment=noaug_baseline
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Add subcommands
    subparsers.add_parser("make_groundtruth", help="Generate ground truth files")
    subparsers.add_parser("train", help="Train a model")
    subparsers.add_parser("hpo", help="Run hyperparameter optimization")
    subparsers.add_parser("refit", help="Refit best model on train+val")
    subparsers.add_parser("evaluate_best", help="Evaluate best model")
    subparsers.add_parser("export_metrics", help="Export metrics from MLflow")
    
    # Parse only the command
    args, remaining = parser.parse_known_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate function
    # Note: Hydra args should be passed after the command
    if args.command == "make_groundtruth":
        make_groundtruth()
    elif args.command == "train":
        train()
    elif args.command == "hpo":
        hpo()
    elif args.command == "refit":
        refit()
    elif args.command == "evaluate_best":
        evaluate_best()
    elif args.command == "export_metrics":
        export_metrics()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
