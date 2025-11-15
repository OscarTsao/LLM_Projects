"""Hyperparameter optimization using Optuna."""

from typing import Any, Dict

import hydra
import mlflow
import optuna
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.train import train_loop


class TqdmCallback:
    """Callback to show progress bar during Optuna optimization."""

    def __init__(self, n_trials: int, desc: str = "Optimization"):
        """Initialize the callback.

        Args:
            n_trials: Total number of trials
            desc: Description for the progress bar
        """
        self.n_trials = n_trials
        self.desc = desc
        self.pbar = None

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Update progress bar after each trial.

        Args:
            study: Optuna study object
            trial: Completed trial
        """
        if self.pbar is None:
            self.pbar = tqdm(total=self.n_trials, desc=self.desc)

        self.pbar.update(1)
        if trial.value is not None:
            self.pbar.set_postfix({
                "best": f"{study.best_value:.4f}",
                "current": f"{trial.value:.4f}"
            })

    def close(self) -> None:
        """Close the progress bar."""
        if self.pbar is not None:
            self.pbar.close()


def suggest_value(trial: optuna.Trial, name: str, spec: Dict[str, Any]) -> Any:
    """Suggest a hyperparameter value based on the distribution specification.

    Args:
        trial: Optuna trial object
        name: Parameter name
        spec: Distribution specification

    Returns:
        Suggested parameter value

    Raises:
        ValueError: If distribution type is not supported
    """
    distribution = spec["distribution"]

    if distribution == "loguniform":
        return trial.suggest_float(name, spec["low"], spec["high"], log=True)
    if distribution == "uniform":
        return trial.suggest_float(name, spec["low"], spec["high"])
    if distribution == "categorical":
        return trial.suggest_categorical(name, spec["choices"])

    raise ValueError(f"Unsupported distribution: {distribution}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for hyperparameter optimization.

    Args:
        cfg: Hydra configuration
    """
    study_cfg = cfg.hpo

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    # Instantiate sampler and pruner
    sampler = (
        hydra.utils.instantiate(study_cfg.sampler) if "sampler" in study_cfg else None
    )
    pruner = (
        hydra.utils.instantiate(study_cfg.pruner) if "pruner" in study_cfg else None
    )

    # Create or load study
    storage = study_cfg.storage
    study = optuna.create_study(
        study_name=study_cfg.study_name,
        storage=storage,
        load_if_exists=True,
        direction=study_cfg.direction,
        sampler=sampler,
        pruner=pruner,
    )

    search_space = OmegaConf.to_container(study_cfg.search_space, resolve=True)

    # Mapping for encoder type inference
    encoder_type_map = {
        "roberta-base": "roberta",
        "microsoft/deberta-base": "deberta",
        "bert-base-uncased": "bert",
    }

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function.

        Args:
            trial: Optuna trial object

        Returns:
            Metric to optimize
        """
        # Create trial configuration
        trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

        # Update configuration with trial suggestions
        for param_path, spec in search_space.items():
            value = suggest_value(trial, param_path, spec)
            OmegaConf.update(trial_cfg, param_path, value, merge=False)

        # Ensure encoder type matches pretrained model
        pretrained = trial_cfg.model.encoder.pretrained_model_name_or_path
        encoder_type = encoder_type_map.get(pretrained, trial_cfg.model.encoder.type)
        OmegaConf.update(trial_cfg, "model.encoder.type", encoder_type, merge=False)

        # Convert num_layers to hidden_dims for classification head
        if "num_layers" in trial_cfg.model.heads.symptom_labels.layers:
            num_layers = trial_cfg.model.heads.symptom_labels.layers.num_layers
            hidden_size = 768  # Standard BERT/RoBERTa/DeBERTa hidden size
            if num_layers == 1:
                hidden_dims = []  # Direct to output
            elif num_layers == 2:
                hidden_dims = [hidden_size]  # One intermediate layer
            else:
                hidden_dims = []
            OmegaConf.update(
                trial_cfg,
                "model.heads.symptom_labels.layers.hidden_dims",
                hidden_dims,
                merge=False,
            )
            # Remove num_layers from config as it's not used by the model
            del trial_cfg.model.heads.symptom_labels.layers["num_layers"]

        # Adjust settings for HPO
        trial_cfg.mlflow.nested = True
        trial_cfg.training.early_stopping.patience = max(
            2, trial_cfg.training.early_stopping.patience // 2
        )

        # Run training with trial configuration
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            result = train_loop(trial_cfg)
            metric = result["best_metric"]
            mlflow.log_metric("objective", metric, step=trial.number)

        return metric

    # Run optimization with progress bar
    progress_callback = TqdmCallback(
        n_trials=study_cfg.n_trials,
        desc=f"HPO ({study_cfg.study_name})"
    )

    try:
        study.optimize(
            objective,
            n_trials=study_cfg.n_trials,
            timeout=study_cfg.timeout,
            n_jobs=study_cfg.n_jobs,
            callbacks=[progress_callback],
        )
    finally:
        progress_callback.close()

    # Print best results
    best_trial = study.best_trial
    print(f"\nBest trial: {best_trial.number}")
    print(f"Best value: {best_trial.value}")
    print(f"Best params: {best_trial.params}")


if __name__ == "__main__":
    main()
