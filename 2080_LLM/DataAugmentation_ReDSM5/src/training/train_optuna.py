"""Optuna-driven hyperparameter search for BERT pair classification."""
from __future__ import annotations

import shutil
from pathlib import Path

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf

from src.utils import mlflow_utils
from .engine import train_model

DATASET_OPTIONS = {
    "original": "conf/dataset/original.yaml",
    "original_nlpaug": "conf/dataset/original_nlpaug.yaml",
    "original_textattack": "conf/dataset/original_textattack.yaml",
    "original_hybrid": "conf/dataset/original_hybrid.yaml",
    "original_nlpaug_textattack": "conf/dataset/original_nlpaug_textattack.yaml",
}

MODEL_OPTIONS = {
    "bert_base": "conf/model/bert_base.yaml",
    "roberta_base": "conf/model/roberta_base.yaml",
    "deberta_base": "conf/model/deberta_base.yaml",
}


def _load_dataset_config(name: str) -> DictConfig:
    path = Path(__file__).resolve().parents[2] / DATASET_OPTIONS[name]
    return OmegaConf.load(path)


def _load_model_config(name: str) -> DictConfig:
    path = Path(__file__).resolve().parents[2] / MODEL_OPTIONS[name]
    return OmegaConf.load(path)


def _prepare_cfg(base_cfg: DictConfig, dataset_name: str, model_name: str) -> DictConfig:
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    cfg.dataset = _load_dataset_config(dataset_name)
    cfg.model = _load_model_config(model_name)
    return cfg


@hydra.main(config_path="../../conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    # Setup MLflow for Optuna study tracking
    mlflow_section = getattr(cfg, "mlflow", None)
    optuna_experiment = None
    if mlflow_section is not None:
        experiments_cfg = mlflow_section.get("experiments")
        if experiments_cfg is not None:
            optuna_experiment = experiments_cfg.get("optuna")
    mlflow_utils.setup_mlflow(cfg, experiment_name=optuna_experiment or "redsm5-optuna")

    study_dir = Path(cfg.output_dir) / "optuna"
    study_dir.mkdir(parents=True, exist_ok=True)

    def objective(trial: optuna.trial.Trial) -> float:
        # Model and dataset selection
        model_choice = trial.suggest_categorical("model", list(MODEL_OPTIONS.keys()))
        dataset_choice = trial.suggest_categorical("dataset", list(DATASET_OPTIONS.keys()))
        trial_cfg = _prepare_cfg(cfg, dataset_choice, model_choice)

        # Optimizer & scheduler
        trial_cfg.model.optimizer = trial.suggest_categorical("optimizer", ["adamw_torch", "adamw_hf", "sgd"])
        trial_cfg.model.scheduler = trial.suggest_categorical("scheduler", ["linear", "cosine", "polynomial"])

        # Learning parameters - aligned with reference config
        trial_cfg.model.learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True)
        trial_cfg.model.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1, log=True)
        trial_cfg.model.warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
        trial_cfg.model.gradient_accumulation_steps = trial.suggest_categorical("grad_accum", [1, 2, 4, 8])

        # Batch sizes for RTX 5090 (32GB VRAM) - conservative to avoid OOM with long sequences
        trial_cfg.model.batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256])
        # Eval batch size can be larger since no gradients needed (up to 2x)
        eval_batch_multiplier = trial.suggest_categorical("eval_batch_multiplier", [1, 1.5, 2])
        trial_cfg.model.eval_batch_size = int(trial_cfg.model.batch_size * eval_batch_multiplier)
        trial_cfg.model.max_seq_length = trial.suggest_categorical("max_seq_length", [128, 256, 384, 512])
        trial_cfg.model.classifier_dropout = trial.suggest_float("classifier_dropout", 0.0, 0.5)
        trial_cfg.model.max_grad_norm = trial.suggest_float("max_grad_norm", 0.5, 5.0)
        trial_cfg.model.adam_eps = trial.suggest_float("adam_eps", 1e-9, 1e-6, log=True)

        # GPU optimizations (compile_model disabled for RTX 5090 sm_120 compatibility)
        trial_cfg.model.compile_model = False
        # Note: DeBERTa has mixed precision disabled in engine.py due to overflow in attention masking
        trial_cfg.model.use_bfloat16 = trial.suggest_categorical("use_bfloat16", [True, False])

        # Classifier architecture search - aligned with reference config
        num_layers = trial.suggest_int("classifier_layers", 0, 3)
        hidden_sizes = []
        for layer_idx in range(num_layers):
            hidden_sizes.append(trial.suggest_int(f"classifier_hidden_{layer_idx}", 128, 768, step=64))
        trial_cfg.model.classifier_hidden_sizes = hidden_sizes

        # Dataloader configuration - optimized for 28 CPU cores + RTX 5090
        trial_cfg.dataloader.num_workers = trial.suggest_int("num_workers", 8, 20)
        trial_cfg.dataloader.prefetch_factor = trial.suggest_int("prefetch_factor", 4, 12)
        trial_cfg.dataloader.pin_memory = True
        trial_cfg.dataloader.persistent_workers = True

        # Training duration per spec
        trial_cfg.model.num_epochs = 100
        trial_cfg.seed = trial.suggest_int("seed", 1, 10_000)

        trial_output_dir = study_dir / f"trial_{trial.number:04d}"
        trial_cfg.output_dir = str(trial_output_dir)
        trial_cfg.resume = False  # isolate trials

        result = train_model(trial_cfg, output_dir=trial_output_dir, trial=trial)
        best_metric = result["best_metric"]
        trial.set_user_attr("output_dir", str(result["output_dir"]))
        if result.get("best_model_path"):
            trial.set_user_attr("best_model_path", str(result["best_model_path"]))
        if result.get("test_metrics"):
            trial.set_user_attr("test_metrics", result["test_metrics"])
        return best_metric if best_metric is not None else float("nan")

    pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(direction="maximize", pruner=pruner, study_name="bert_pair_optuna")
    study.optimize(objective, n_trials=cfg.get("n_trials", 500), timeout=cfg.get("timeout"))

    best_trial = study.best_trial
    print(f"Best trial #{best_trial.number} value: {best_trial.value}")
    print("Params:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")

    # Log best trial to MLflow
    with mlflow_utils.start_run(run_name=f"optuna_best_trial_{best_trial.number}"):
        mlflow_utils.log_params(best_trial.params)
        mlflow_utils.log_metrics({"best_value": best_trial.value})
        mlflow_utils.set_tag("trial_number", str(best_trial.number))
        mlflow_utils.set_tag("optimization_study", "optuna")

        # Log best trial test metrics if available
        test_metrics = best_trial.user_attrs.get("test_metrics", {})
        if test_metrics:
            test_metrics_prefixed = {f"test_{k}": v for k, v in test_metrics.items()}
            mlflow_utils.log_metrics(test_metrics_prefixed)

        # Log best model if available
        best_model_path = best_trial.user_attrs.get("best_model_path")
        if best_model_path and Path(best_model_path).exists():
            mlflow_utils.log_model(best_model_path, artifact_path="best_model")

    # Remove artifacts from non-best trials to conserve storage
    best_output = Path(best_trial.user_attrs.get("output_dir", ""))
    for trial_dir in study_dir.glob("trial_*"):
        if trial_dir.resolve() == best_output.resolve():
            continue
        shutil.rmtree(trial_dir, ignore_errors=True)

    if best_output.exists():
        print(f"Best artifacts kept at {best_output}")


if __name__ == "__main__":
    main()
