"""Run hyperparameter optimization for a specific stage."""

import sys
from pathlib import Path

import hydra
import mlflow
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from psy_agents_noaug.hpo.optuna_runner import (
    OptunaRunner,
    create_search_space_from_config,
)
from psy_agents_noaug.utils.mlflow_utils import configure_mlflow, log_config
from psy_agents_noaug.utils.reproducibility import set_seed


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
    
    # Set seed for reproducibility
    set_seed(cfg.seed, cfg.get("deterministic", True), cfg.get("cudnn_benchmark", False))
    
    # Configure MLflow
    experiment_name = f"{cfg.mlflow.experiment_name}-hpo-stage{cfg.hpo.stage}"
    run_name = f"stage{cfg.hpo.stage}_{cfg.hpo.stage_name}"
    
    run_id = configure_mlflow(
        tracking_uri=cfg.mlflow.tracking_uri,
        experiment_name=experiment_name,
        run_name=run_name,
        config=cfg,
    )
    
    # Log configuration
    log_config(cfg)
    
    # Create search space
    search_space = create_search_space_from_config(cfg.hpo)
    
    print(f"Search space:")
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
        # This is a placeholder - implement actual training logic
        # In practice, this would call the training loop with the suggested params
        
        # For now, return a dummy value
        # In production, replace with actual training and evaluation
        print(f"\nTrial {trial.number}: {params}")
        
        # TODO: Implement actual training with params
        # score = train_and_evaluate(cfg, params)
        
        # Placeholder score
        import random
        score = random.random()
        
        return score
    
    # Run optimization
    print(f"\nRunning optimization with {cfg.hpo.n_trials} trials...")
    
    runner.optimize(
        objective_fn=objective,
        n_trials=cfg.hpo.n_trials,
        search_space=search_space,
        mlflow_tracking_uri=cfg.mlflow.tracking_uri,
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
    
    print(f"\nHPO completed! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
