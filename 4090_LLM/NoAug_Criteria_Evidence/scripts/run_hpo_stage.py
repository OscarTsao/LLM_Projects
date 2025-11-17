"""Run hyperparameter optimization for a specific stage."""

import json
import sys
from pathlib import Path
from shutil import copy2

import hydra
import mlflow
import optuna
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psy_agents_noaug.hpo.optuna_runner import (
    OptunaRunner,
    create_search_space_from_config,
)
from psy_agents_noaug.utils.mlflow_utils import (
    configure_mlflow,
    log_config,
    resolve_artifact_location,
    resolve_tracking_uri,
)
from psy_agents_noaug.utils.reproducibility import set_seed
from psy_agents_noaug.utils.storage import ensure_directory


def _eager_load_transformers_configs():
    """
    Eagerly load transformers model configurations to prevent lazy loading race conditions.

    In transformers 4.57.1, the lazy module loading system can fail in multi-threaded contexts
    (like Optuna trials) with: "ImportError: cannot import name 'OnnxConfig' from 'transformers.onnx'"

    This function forces eager loading of commonly used model configs before trials start.
    """
    try:
        from transformers import (
            AutoConfig,
            BertConfig,
            RobertaConfig,
            DebertaV2Config,
        )
        # Just importing triggers the lazy loading, no need to instantiate
    except ImportError:
        # If transformers is not installed or configs are unavailable, fail silently
        # The actual error will occur when trying to use the models
        pass


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Run HPO stage.

    Usage:
        python scripts/run_hpo_stage.py hpo=stage0_sanity task=criteria
        python scripts/run_hpo_stage.py hpo=stage1_coarse task=criteria model=roberta_base
    """
    print(f"\n{'=' * 70}")
    print(f"HPO Stage {cfg.hpo.stage}: {cfg.hpo.stage_name}".center(70))
    print(f"{'=' * 70}\n")

    # Eagerly load transformers configs to prevent lazy loading race conditions
    _eager_load_transformers_configs()

    project_root = Path(get_original_cwd())

    # Set seed for reproducibility
    set_seed(
        cfg.seed, cfg.get("deterministic", True), cfg.get("cudnn_benchmark", False)
    )

    # Configure MLflow
    experiment_name = f"{cfg.mlflow.experiment_name}-hpo-stage{cfg.hpo.stage}"
    run_name = f"stage{cfg.hpo.stage}_{cfg.hpo.stage_name}"

    tracking_uri = resolve_tracking_uri(cfg.mlflow.tracking_uri, project_root)
    artifact_location = resolve_artifact_location(
        cfg.mlflow.get("artifact_location"), project_root
    )

    configure_mlflow(
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        artifact_location=artifact_location,
        config=cfg,
    )

    # Log configuration
    log_config(cfg)

    # Create search space
    search_space = create_search_space_from_config(cfg.hpo)

    print("Search space:")
    for param, config in search_space.items():
        print(f"  {param}: {config}")

    # Create Optuna runner
    study_name = f"{cfg.task.name}_stage{cfg.hpo.stage}"

    runner = OptunaRunner(
        study_name=study_name,
        direction=cfg.hpo.direction,
        metric=cfg.hpo.metric,
        storage=None,  # In-memory storage, can use SQLite
        sampler_config=cfg.hpo.get("sampler"),
        pruner_config=cfg.hpo.get("pruner"),
    )

    # Define objective function
    def objective(trial, params):
        """
        Objective function for a single trial.

        This should:
        1. Update config with suggested params
        2. Train model
        3. Return validation metric
        """
        print(f"\nTrial {trial.number}: {params}")

        # Import training modules
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, random_split
        from transformers import AutoTokenizer

        from Project.Criteria.data.dataset import CriteriaDataset
        from Project.Criteria.models.model import Model
        from psy_agents_noaug.utils.reproducibility import (
            get_device,
            get_optimal_dataloader_kwargs,
            set_seed,
        )

        # Set seed for reproducibility
        trial_seed = cfg.seed + trial.number
        set_seed(
            trial_seed,
            cfg.get("deterministic", True),
            cfg.get("cudnn_benchmark", False),
        )

        # Get device
        device = get_device()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

        # Load dataset based on task
        if cfg.task.name == "criteria":
            dataset_path = project_root / "data" / "redsm5" / "redsm5_annotations.csv"
            dataset = CriteriaDataset(
                csv_path=dataset_path,
                tokenizer=tokenizer,
                max_length=params.get("max_length", 256),
            )
        elif cfg.task.name == "evidence":
            from Project.Evidence.data.dataset import EvidenceDataset

            dataset_path = project_root / "data" / "processed" / "redsm5_matched_evidence.csv"
            dataset = EvidenceDataset(
                csv_path=dataset_path,
                tokenizer=tokenizer,
                max_length=params.get("max_length", 512),
            )
        elif cfg.task.name == "joint":
            from Project.Joint.data.dataset import JointDataset

            criteria_path = project_root / "data" / "redsm5" / "redsm5_annotations.csv"
            evidence_path = project_root / "data" / "processed" / "redsm5_matched_evidence.csv"
            dataset = JointDataset(
                criteria_csv_path=criteria_path,
                evidence_csv_path=evidence_path,
                tokenizer=tokenizer,
                max_length=params.get("max_length", 512),
            )
        elif cfg.task.name == "share":
            from Project.Share.data.dataset import ShareDataset

            criteria_path = project_root / "data" / "redsm5" / "redsm5_annotations.csv"
            evidence_path = project_root / "data" / "processed" / "redsm5_matched_evidence.csv"
            dataset = ShareDataset(
                criteria_csv_path=criteria_path,
                evidence_csv_path=evidence_path,
                tokenizer=tokenizer,
                max_length=params.get("max_length", 512),
            )
        else:
            raise NotImplementedError(f"Task {cfg.task.name} not implemented")

        # Split dataset (80/10/10)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size

        generator = torch.Generator().manual_seed(trial_seed)
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size], generator=generator
        )

        # Create data loaders
        dataloader_kwargs = get_optimal_dataloader_kwargs(
            device, cfg.training.num_workers
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=params.get("batch_size", cfg.training.batch_size),
            shuffle=True,
            **dataloader_kwargs,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=params.get("batch_size", cfg.training.batch_size),
            shuffle=False,
            **dataloader_kwargs,
        )

        # Create model based on task
        if cfg.task.name == "criteria":
            model = Model(
                model_name=cfg.model.name,
                num_labels=2,
                dropout=params.get("dropout", 0.1),
            ).to(device)
        elif cfg.task.name == "evidence":
            from Project.Evidence.models.model import Model as EvidenceModel

            model = EvidenceModel(
                model_name=cfg.model.name,
                dropout_prob=params.get("dropout", 0.1),
            ).to(device)
        elif cfg.task.name == "joint":
            from Project.Joint.models.model import Model as JointModel

            model = JointModel(
                criteria_model_name=cfg.model.name,
                evidence_model_name=cfg.model.name,
                criteria_num_labels=2,
                criteria_dropout=params.get("dropout", 0.1),
                evidence_dropout=params.get("dropout", 0.1),
            ).to(device)
        elif cfg.task.name == "share":
            from Project.Share.models.model import Model as ShareModel

            model = ShareModel(
                model_name=cfg.model.name,
                criteria_num_labels=2,
                criteria_dropout=params.get("dropout", 0.1),
                evidence_dropout=params.get("dropout", 0.1),
            ).to(device)

        # Create optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params.get("learning_rate", 2e-5),
            weight_decay=params.get("weight_decay", 0.01),
        )

        # Training loop
        num_epochs = cfg.hpo.num_epochs
        best_val_f1 = 0.0

        for epoch in range(num_epochs):
            # Training
            model.train()
            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                optimizer.zero_grad()

                if cfg.task.name == "criteria":
                    labels = batch["labels"].to(device)
                    logits = model(input_ids, attention_mask)
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(logits, labels)
                elif cfg.task.name == "evidence":
                    start_positions = batch["start_positions"].to(device)
                    end_positions = batch["end_positions"].to(device)
                    start_logits, end_logits = model(input_ids, attention_mask)
                    criterion = nn.CrossEntropyLoss()
                    start_loss = criterion(start_logits, start_positions)
                    end_loss = criterion(end_logits, end_positions)
                    loss = (start_loss + end_loss) / 2
                elif cfg.task.name in ["joint", "share"]:
                    criteria_labels = batch["criteria_labels"].to(device)
                    start_positions = batch["start_positions"].to(device)
                    end_positions = batch["end_positions"].to(device)
                    outputs = model(input_ids, attention_mask)

                    # Outputs are: (criteria_logits, start_logits, end_logits)
                    criteria_logits = outputs[0]
                    start_logits = outputs[1]
                    end_logits = outputs[2]

                    criterion = nn.CrossEntropyLoss()
                    criteria_loss = criterion(criteria_logits, criteria_labels)
                    start_loss = criterion(start_logits, start_positions)
                    end_loss = criterion(end_logits, end_positions)
                    loss = (criteria_loss + start_loss + end_loss) / 3

                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)

                    if cfg.task.name == "criteria":
                        labels = batch["labels"].to(device)
                        logits = model(input_ids, attention_mask)
                        preds = torch.argmax(logits, dim=-1)
                        all_preds.extend(preds.cpu().tolist())
                        all_labels.extend(labels.cpu().tolist())
                    elif cfg.task.name == "evidence":
                        start_positions = batch["start_positions"].to(device)
                        end_positions = batch["end_positions"].to(device)
                        start_logits, end_logits = model(input_ids, attention_mask)
                        start_preds = torch.argmax(start_logits, dim=-1)
                        end_preds = torch.argmax(end_logits, dim=-1)

                        # For span prediction, we need to compute F1 differently
                        # Here we use exact match as a proxy
                        for sp, ep, st, et in zip(
                            start_preds.cpu().tolist(),
                            end_preds.cpu().tolist(),
                            start_positions.cpu().tolist(),
                            end_positions.cpu().tolist(),
                        ):
                            all_preds.append(1 if (sp == st and ep == et) else 0)
                            all_labels.append(1)
                    elif cfg.task.name in ["joint", "share"]:
                        criteria_labels = batch["criteria_labels"].to(device)
                        outputs = model(input_ids, attention_mask)
                        criteria_logits = outputs[0]
                        preds = torch.argmax(criteria_logits, dim=-1)
                        all_preds.extend(preds.cpu().tolist())
                        all_labels.extend(criteria_labels.cpu().tolist())

            # Calculate F1 score
            from sklearn.metrics import f1_score

            val_f1 = f1_score(all_labels, all_preds, average="macro")

            best_val_f1 = max(best_val_f1, val_f1)

            # Report intermediate value for pruning
            trial.report(val_f1, epoch)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

        return best_val_f1

    # Run optimization
    print(f"\nRunning optimization with {cfg.hpo.n_trials} trials...")

    runner.optimize(
        objective_fn=objective,
        n_trials=cfg.hpo.n_trials,
        search_space=search_space,
        mlflow_tracking_uri=tracking_uri,
        timeout=cfg.hpo.get("timeout"),
    )

    # Print results
    runner.print_best_trial()

    # Save results
    output_dir = Path(cfg.output_dir) / f"hpo_stage{cfg.hpo.stage}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export best config
    best_config_path = output_dir / "best_config.yaml"
    runner.export_best_config(best_config_path)

    # Export trials history
    trials_history_path = output_dir / "trials_history.json"
    runner.export_trials_history(trials_history_path)

    # Save study
    study_path = output_dir / "study.pkl"
    runner.save_study(study_path)

    # Log to MLflow
    mlflow.log_artifacts(str(output_dir), artifact_path="hpo_results")

    # End MLflow run
    mlflow.end_run()

    artifact_root = ensure_directory(
        Path("artifacts")
        / "hpo"
        / cfg.task.name
        / f"stage{cfg.hpo.stage}_{cfg.hpo.stage_name}"
    )

    for filename in ("best_config.yaml", "trials_history.json", "study.pkl"):
        src = output_dir / filename
        if src.exists():
            copy2(src, artifact_root / filename)

    summary_payload = {
        "stage": cfg.hpo.stage,
        "stage_name": cfg.hpo.stage_name,
        "n_trials": cfg.hpo.n_trials,
        "best_value": runner.get_best_value(),
        "best_params": runner.get_best_params(),
    }
    with (artifact_root / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary_payload, fp, indent=2)

    print(f"\nHPO completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
