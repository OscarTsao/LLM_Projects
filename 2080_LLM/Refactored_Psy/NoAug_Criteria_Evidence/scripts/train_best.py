"""Train model with best hyperparameters from HPO."""

import json
import sys
from pathlib import Path

import hydra
import mlflow
import torch
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psy_agents_noaug.training.evaluate import (
    Evaluator,
    generate_evaluation_report,
    print_evaluation_results,
)
from psy_agents_noaug.training.train_loop import Trainer
from psy_agents_noaug.utils.mlflow_utils import (
    configure_mlflow,
    log_artifacts,
    log_config,
    save_model_to_mlflow,
)
from psy_agents_noaug.utils.reproducibility import get_device, set_seed


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Train model with best configuration.
    
    Usage:
        python scripts/train_best.py task=criteria model=roberta_base
        python scripts/train_best.py task=evidence best_config=outputs/hpo_stage2/best_config.yaml
    """
    print(f"\n{'=' * 70}")
    print(f"Training Best Model: {cfg.task.name}".center(70))
    print(f"{'=' * 70}\n")
    
    # Load best config if provided
    if "best_config" in cfg and cfg.best_config:
        best_config = OmegaConf.load(cfg.best_config)
        # Merge best config into current config
        cfg = OmegaConf.merge(cfg, best_config)
    
    # Set seed for reproducibility
    set_seed(cfg.seed, cfg.get("deterministic", True), cfg.get("cudnn_benchmark", False))
    
    # Get device
    device = get_device(cfg.device == "cuda")
    
    # Configure MLflow
    experiment_name = f"{cfg.mlflow.experiment_name}-train"
    run_name = f"best_{cfg.task.name}_{cfg.model.encoder_name.split('/')[-1]}"
    
    run_id = configure_mlflow(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        config=cfg,
    )
    
    # Log configuration
    log_config(cfg)
    
    # TODO: Load data
    # This is a placeholder - implement actual data loading
    print("Loading data...")
    # train_loader, val_loader, test_loader = load_data(cfg)
    
    # TODO: Create model
    # This is a placeholder - implement actual model creation
    print("Creating model...")
    # model = create_model(cfg)
    # model = model.to(device)
    
    # TODO: Create optimizer and scheduler
    # optimizer = create_optimizer(cfg, model)
    # scheduler = create_scheduler(cfg, optimizer)
    
    # TODO: Create loss function
    # criterion = create_criterion(cfg)
    
    # TODO: Create trainer
    # print("Starting training...")
    # trainer = Trainer(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     optimizer=optimizer,
    #     criterion=criterion,
    #     device=device,
    #     num_epochs=cfg.training.num_epochs,
    #     patience=cfg.training.early_stopping.patience,
    #     gradient_clip=cfg.training.gradient_clip,
    #     scheduler=scheduler,
    #     save_dir=Path(cfg.output_dir) / "checkpoints",
    #     use_amp=cfg.training.amp.enabled,
    # )
    
    # Train model
    # final_metrics = trainer.train()
    
    # TODO: Evaluate on test set
    # print("\nEvaluating on test set...")
    # evaluator = Evaluator(
    #     model=model,
    #     device=device,
    #     task_type=cfg.task.task_type,
    #     criterion=criterion,
    # )
    
    # test_metrics = evaluator.evaluate(
    #     test_loader,
    #     class_names=cfg.task.class_names,
    # )
    
    # Print results
    # print_evaluation_results(test_metrics, title="Test Set Results")
    
    # Save evaluation report
    # report_path = Path(cfg.output_dir) / "evaluation_report.json"
    # generate_evaluation_report(test_metrics, report_path)
    
    # Log artifacts
    # log_artifacts(Path(cfg.output_dir), artifact_path="outputs")
    
    # Save model
    # save_model_to_mlflow(model, artifact_path="model")
    
    # End MLflow run
    mlflow.end_run()
    
    print(f"\nTraining completed!")
    print(f"Results saved to {cfg.output_dir}")
    print(f"MLflow run ID: {run_id}")


if __name__ == "__main__":
    main()
