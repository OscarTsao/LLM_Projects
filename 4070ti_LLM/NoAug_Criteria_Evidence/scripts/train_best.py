"""Train model with best hyperparameters from HPO.

This script loads the best hyperparameter configuration from HPO and retrains
the model with those settings. It supports all architectures (Criteria, Evidence, Joint).

Usage:
    # Train Criteria with best HPO config
    python scripts/train_best.py task=criteria best_config=outputs/hpo_stage2/best_config.yaml

    # Train Evidence with best HPO config
    python scripts/train_best.py task=evidence best_config=outputs/hpo_stage2/best_config.yaml

    # Train Joint with best HPO config
    python scripts/train_best.py task=joint best_config=outputs/hpo_stage2/best_config.yaml

    # Train with default config (no HPO)
    python scripts/train_best.py task=criteria
"""

import subprocess
import sys
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Train model with best configuration by routing to architecture-specific scripts.

    This is a convenience wrapper that:
    1. Loads best HPO config if provided
    2. Routes to the appropriate architecture-specific training script
    3. Ensures reproducibility and optimization settings are applied
    """
    print(f"\n{'=' * 70}")
    print(f"Training Best Model: {cfg.task.name}".center(70))
    print(f"{'=' * 70}\n")

    # Get task name
    task_name = cfg.task.name.lower()

    # Validate task
    valid_tasks = ["criteria", "evidence", "joint"]
    if task_name not in valid_tasks:
        raise ValueError(f"Invalid task: {task_name}. Must be one of {valid_tasks}")

    # Get project root
    project_root = Path(get_original_cwd())
    scripts_dir = project_root / "scripts"

    # Determine which training script to use
    train_script = scripts_dir / f"train_{task_name}.py"

    if not train_script.exists():
        print(f"\nWarning: Architecture-specific script not found: {train_script}")
        print(f"Please implement {train_script.name} for {task_name} architecture.")
        print("\nFor now, you can:")
        print(f"1. Use train_criteria.py as a template")
        print(f"2. Adapt it for {task_name} architecture")
        print(f"3. Save it as {train_script.name}")
        sys.exit(1)

    # Build command to run architecture-specific script
    cmd = [sys.executable, str(train_script)]

    # Add task override
    cmd.extend([f"task={task_name}"])

    # Add best_config if provided
    if hasattr(cfg, "best_config") and cfg.best_config:
        best_config_path = Path(cfg.best_config)
        if not best_config_path.is_absolute():
            best_config_path = project_root / best_config_path
        cmd.extend([f"best_config={best_config_path}"])
        print(f"Loading best HPO config from: {best_config_path}")

    # Add model override if specified
    if hasattr(cfg, "model") and hasattr(cfg.model, "encoder_name"):
        cmd.extend([f"model.pretrained_model={cfg.model.encoder_name}"])

    # Add training overrides if specified
    if hasattr(cfg, "training"):
        training_cfg = cfg.training
        if hasattr(training_cfg, "num_epochs"):
            cmd.extend([f"training.epochs={training_cfg.num_epochs}"])
        if hasattr(training_cfg, "batch_size"):
            cmd.extend([f"training.train_batch_size={training_cfg.batch_size}"])

    print(f"\nLaunching {task_name} training script...")
    print(f"Command: {' '.join(cmd)}\n")

    # Run the architecture-specific training script
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n{'=' * 70}")
        print(f"Training Completed Successfully".center(70))
        print(f"{'=' * 70}\n")
    except subprocess.CalledProcessError as e:
        print(f"\n{'=' * 70}")
        print(f"Training Failed".center(70))
        print(f"{'=' * 70}")
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
