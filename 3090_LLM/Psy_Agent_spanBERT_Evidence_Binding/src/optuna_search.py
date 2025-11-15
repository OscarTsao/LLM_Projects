from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict

import hydra
import optuna
from omegaconf import DictConfig, OmegaConf

from .psya_agent.train_utils import run_training

logger = logging.getLogger(__name__)

_BATCH_CANDIDATES = [4, 6, 8, 12, 16, 24, 32]


def _sample_params(cfg: DictConfig, trial: optuna.Trial) -> Dict[str, Any]:
    params: Dict[str, Any] = {}

    # Data settings
    params["data.positive_only"] = trial.suggest_categorical("data.positive_only", [True, False])
    train_ratio = trial.suggest_float("data.train_ratio", 0.65, 0.85)
    params["data.train_ratio"] = train_ratio
    max_val = max(0.05, min(0.25, 1.0 - train_ratio - 0.05))
    params["data.val_ratio"] = trial.suggest_float("data.val_ratio", 0.05, max_val)
    params["data.seed"] = trial.suggest_int("data.seed", 1, 10000)

    # Model settings
    params["model.dropout"] = trial.suggest_float("model.dropout", 0.0, 0.3)
    # gradient_checkpointing is a performance parameter - set in _set_performance_defaults

    # Feature settings
    params["features.max_length"] = trial.suggest_int("features.max_length", 256, 512, step=32)
    max_doc_stride = max(params["features.max_length"] - 32, 64)
    params["features.doc_stride"] = trial.suggest_int(
        "features.doc_stride", 64, min(max_doc_stride, 256), step=16
    )
    params["features.n_best_size"] = trial.suggest_int("features.n_best_size", 10, 40)
    params["features.max_answer_length"] = trial.suggest_int("features.max_answer_length", 30, 90)

    # Training core hyperparameters
    params["training.num_train_epochs"] = trial.suggest_int("training.num_train_epochs", 10, 200)
    params["training.train_batch_size"] = trial.suggest_categorical("training.train_batch_size", _BATCH_CANDIDATES)
    # eval_batch_size, gradient_accumulation_steps, log_every_steps are performance parameters - set in _set_performance_defaults
    params["training.learning_rate"] = trial.suggest_float(
        "training.learning_rate", 1e-6, 5e-4, log=True
    )
    params["training.weight_decay"] = trial.suggest_float("training.weight_decay", 0.0, 0.15)
    params["training.warmup_ratio"] = trial.suggest_float("training.warmup_ratio", 0.0, 0.4)
    params["training.max_grad_norm"] = trial.suggest_float("training.max_grad_norm", 0.5, 5.0)
    params["training.adam_epsilon"] = trial.suggest_float("training.adam_epsilon", 1e-9, 1e-6, log=True)
    params["training.seed"] = trial.suggest_int("training.seed", 1, 10000)

    # Optimizer selection & specific knobs
    optimizer_name = trial.suggest_categorical(
        "training.optimizer.name",
        ["adamw_torch", "adamw_hf", "adam", "adamax", "rmsprop", "sgd", "adafactor"],
    )
    params["training.optimizer.name"] = optimizer_name

    # Sample parameters based on optimizer type to avoid incompatibilities
    if optimizer_name in {"adamw_torch", "adamw_hf", "adam", "adamax"}:
        # Adam-family optimizers
        beta1 = trial.suggest_float("training.optimizer.beta1", 0.8, 0.99)
        beta2 = trial.suggest_float("training.optimizer.beta2", max(beta1 + 0.001, 0.9), 0.9999)
        params["training.optimizer.betas"] = (beta1, beta2)
        params["training.optimizer.amsgrad"] = trial.suggest_categorical("training.optimizer.amsgrad", [True, False])
        # use_fused is a performance parameter - set to optimal value in _set_performance_defaults
    elif optimizer_name == "rmsprop":
        # RMSprop-specific parameters
        params["training.optimizer.alpha"] = trial.suggest_float("training.optimizer.alpha", 0.8, 0.99)
        params["training.optimizer.centered"] = trial.suggest_categorical(
            "training.optimizer.centered", [True, False]
        )
        params["training.optimizer.momentum"] = trial.suggest_float("training.optimizer.momentum", 0.0, 0.98)
    elif optimizer_name == "sgd":
        # SGD-specific parameters
        params["training.optimizer.momentum"] = trial.suggest_float("training.optimizer.momentum", 0.0, 0.98)
        params["training.optimizer.nesterov"] = trial.suggest_categorical(
            "training.optimizer.nesterov", [True, False]
        )
    elif optimizer_name == "adafactor":
        # Adafactor-specific parameters
        params["training.optimizer.relative_step"] = trial.suggest_categorical(
            "training.optimizer.relative_step", [True, False]
        )
        params["training.optimizer.scale_parameter"] = trial.suggest_categorical(
            "training.optimizer.scale_parameter", [True, False]
        )
        params["training.optimizer.warmup_init"] = trial.suggest_categorical(
            "training.optimizer.warmup_init", [True, False]
        )
        # Enforce Adafactor constraint
        if (params["training.optimizer.warmup_init"] and
            not params["training.optimizer.relative_step"]):
            logger.debug(
                "Adafactor requires relative_step=True when warmup_init=True; overriding warmup_init to False"
            )
            params["training.optimizer.warmup_init"] = False

    # Scheduler choices
    scheduler_name = trial.suggest_categorical(
        "training.scheduler.name",
        [
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
            "step",
            "cosineannealing",
        ],
    )
    params["training.scheduler.name"] = scheduler_name
    params["training.scheduler.warmup_ratio"] = trial.suggest_float(
        "training.scheduler.warmup_ratio", 0.0, 0.4
    )
    params["training.scheduler.num_cycles"] = trial.suggest_float(
        "training.scheduler.num_cycles", 0.5, 3.0
    )
    params["training.scheduler.power"] = trial.suggest_float("training.scheduler.power", 1.0, 2.5)
    params["training.scheduler.gamma"] = trial.suggest_float("training.scheduler.gamma", 0.1, 0.95)
    params["training.scheduler.step_size"] = trial.suggest_int(
        "training.scheduler.step_size", 500, 5000, step=250
    )
    params["training.scheduler.lr_end_ratio"] = trial.suggest_float(
        "training.scheduler.lr_end_ratio", 0.0, 0.5
    )
    params["training.scheduler.t_max_ratio"] = trial.suggest_float(
        "training.scheduler.t_max_ratio", 0.5, 1.5
    )
    params["training.scheduler.eta_min_ratio"] = trial.suggest_float(
        "training.scheduler.eta_min_ratio", 0.0, 0.2
    )

    # Evaluation / optimization targets
    params["optimization.metric"] = trial.suggest_categorical(
        "optimization.metric", ["f1", "exact_match"]
    )
    params["optimization.patience"] = trial.suggest_int("optimization.patience", 2, 12)
    params["optimization.min_delta"] = trial.suggest_float("optimization.min_delta", 0.0, 0.02)
    params["optimization.warmup_epochs"] = trial.suggest_int("optimization.warmup_epochs", 0, 6)
    params["optimization.cooldown_epochs"] = trial.suggest_int("optimization.cooldown_epochs", 0, 4)

    return params


def _apply_params(cfg: DictConfig, params: Dict[str, Any]) -> DictConfig:
    trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    # Data
    train_ratio = float(params.get("data.train_ratio", trial_cfg.data.train_ratio))
    val_ratio = float(params.get("data.val_ratio", trial_cfg.data.val_ratio))
    test_ratio = max(1.0 - train_ratio - val_ratio, 0.05)
    total = train_ratio + val_ratio + test_ratio
    trial_cfg.data.train_ratio = train_ratio / total
    trial_cfg.data.val_ratio = val_ratio / total
    trial_cfg.data.test_ratio = test_ratio / total
    trial_cfg.data.positive_only = bool(params.get("data.positive_only", trial_cfg.data.positive_only))
    trial_cfg.data.seed = int(params.get("data.seed", trial_cfg.data.seed))

    # Model
    trial_cfg.model.dropout = float(params.get("model.dropout", trial_cfg.model.dropout))
    # gradient_checkpointing is set in _set_performance_defaults

    # Features
    trial_cfg.features.max_length = int(params.get("features.max_length", trial_cfg.features.max_length))
    doc_stride = int(params.get("features.doc_stride", trial_cfg.features.doc_stride))
    doc_stride = min(doc_stride, max(trial_cfg.features.max_length - 32, 32))
    trial_cfg.features.doc_stride = max(doc_stride, 32)
    trial_cfg.features.n_best_size = int(params.get("features.n_best_size", trial_cfg.features.n_best_size))
    trial_cfg.features.max_answer_length = int(
        params.get("features.max_answer_length", trial_cfg.features.max_answer_length)
    )

    # Training base
    trial_cfg.training.num_train_epochs = int(
        params.get("training.num_train_epochs", trial_cfg.training.num_train_epochs)
    )
    train_batch_size = int(
        params.get("training.train_batch_size", trial_cfg.training.train_batch_size)
    )
    trial_cfg.training.train_batch_size = train_batch_size
    # eval_batch_size and gradient_accumulation_steps are set in _set_performance_defaults
    trial_cfg.training.learning_rate = float(
        params.get("training.learning_rate", trial_cfg.training.learning_rate)
    )
    trial_cfg.training.weight_decay = float(
        params.get("training.weight_decay", trial_cfg.training.weight_decay)
    )
    warmup_ratio = float(params.get("training.warmup_ratio", trial_cfg.training.warmup_ratio))
    trial_cfg.training.warmup_ratio = warmup_ratio
    trial_cfg.training.max_grad_norm = float(
        params.get("training.max_grad_norm", trial_cfg.training.max_grad_norm)
    )
    trial_cfg.training.adam_epsilon = float(
        params.get("training.adam_epsilon", trial_cfg.training.adam_epsilon)
    )
    trial_cfg.training.seed = int(params.get("training.seed", trial_cfg.training.seed))
    # log_every_steps is set in _set_performance_defaults

    # Optimizer - only apply parameters that were sampled for the specific optimizer
    opt_cfg = trial_cfg.training.optimizer
    optimizer_name = params.get("training.optimizer.name", opt_cfg.name)
    opt_cfg.name = optimizer_name

    # Apply optimizer-specific parameters based on the chosen optimizer
    if optimizer_name in {"adamw_torch", "adamw_hf", "adam", "adamax"}:
        if "training.optimizer.betas" in params:
            opt_cfg.betas = list(params["training.optimizer.betas"])
        if "training.optimizer.amsgrad" in params:
            opt_cfg.amsgrad = params["training.optimizer.amsgrad"]
        # use_fused is set in _set_performance_defaults
    elif optimizer_name == "rmsprop":
        if "training.optimizer.alpha" in params:
            opt_cfg.alpha = float(params["training.optimizer.alpha"])
        if "training.optimizer.centered" in params:
            opt_cfg.centered = params["training.optimizer.centered"]
        if "training.optimizer.momentum" in params:
            opt_cfg.momentum = float(params["training.optimizer.momentum"])
    elif optimizer_name == "sgd":
        if "training.optimizer.momentum" in params:
            opt_cfg.momentum = float(params["training.optimizer.momentum"])
        if "training.optimizer.nesterov" in params:
            opt_cfg.nesterov = params["training.optimizer.nesterov"]
    elif optimizer_name == "adafactor":
        if "training.optimizer.relative_step" in params:
            opt_cfg.relative_step = params["training.optimizer.relative_step"]
        if "training.optimizer.scale_parameter" in params:
            opt_cfg.scale_parameter = params["training.optimizer.scale_parameter"]
        if "training.optimizer.warmup_init" in params:
            opt_cfg.warmup_init = params["training.optimizer.warmup_init"]

    # Scheduler
    sched_cfg = trial_cfg.training.scheduler
    sched_cfg.name = params.get("training.scheduler.name", sched_cfg.name)
    sched_cfg.warmup_ratio = params.get("training.scheduler.warmup_ratio", warmup_ratio)
    sched_cfg.num_cycles = params.get("training.scheduler.num_cycles", sched_cfg.num_cycles)
    sched_cfg.power = params.get("training.scheduler.power", sched_cfg.power)
    sched_cfg.gamma = params.get("training.scheduler.gamma", sched_cfg.gamma)
    sched_cfg.step_size = params.get("training.scheduler.step_size", sched_cfg.step_size)
    lr_end_ratio = float(params.get("training.scheduler.lr_end_ratio", sched_cfg.lr_end_ratio))
    sched_cfg.lr_end_ratio = lr_end_ratio
    # Handle case where learning_rate is None (e.g., Adafactor with relative_step=True)
    base_lr = trial_cfg.training.learning_rate
    if base_lr is not None:
        sched_cfg.lr_end = base_lr * lr_end_ratio
    else:
        sched_cfg.lr_end = 0.0  # Default value when lr is None
    sched_cfg.t_max_ratio = float(params.get("training.scheduler.t_max_ratio", sched_cfg.t_max_ratio))
    sched_cfg.eta_min_ratio = float(params.get("training.scheduler.eta_min_ratio", sched_cfg.eta_min_ratio))
    if base_lr is not None:
        sched_cfg.eta_min = base_lr * sched_cfg.eta_min_ratio
    else:
        sched_cfg.eta_min = 0.0  # Default value when lr is None

    # Optimization & logging
    opt_cfg = trial_cfg.optimization
    metric = params.get("optimization.metric", opt_cfg.metric)
    opt_cfg.metric = metric
    opt_cfg.patience = int(params.get("optimization.patience", opt_cfg.patience))
    opt_cfg.min_delta = float(params.get("optimization.min_delta", opt_cfg.min_delta))
    opt_cfg.warmup_epochs = int(params.get("optimization.warmup_epochs", opt_cfg.warmup_epochs))
    opt_cfg.cooldown_epochs = int(params.get("optimization.cooldown_epochs", opt_cfg.cooldown_epochs))
    opt_cfg.higher_is_better = metric not in {"loss", "perplexity"}

    return trial_cfg


def _set_performance_defaults(trial_cfg: DictConfig) -> None:
    """
    Set stable, high-performance defaults for training optimization settings.
    These settings maximize speed without causing stability issues.
    They do NOT affect model quality, only training efficiency.
    """
    # Model performance settings
    trial_cfg.model.gradient_checkpointing = False  # Faster training, use more memory

    # Training performance settings - optimized for speed
    # Set eval_batch_size to be larger than train_batch_size for faster evaluation
    train_batch_size = trial_cfg.training.get("train_batch_size", 8)
    trial_cfg.training.eval_batch_size = max(train_batch_size * 2, 16)  # 2x larger for speed
    trial_cfg.training.gradient_accumulation_steps = 1  # No accumulation for fastest training
    trial_cfg.training.log_every_steps = 50  # Reasonable logging frequency

    # Optimizer performance settings - enable fused operations for speed
    optimizer_name = trial_cfg.training.optimizer.get("name", "adamw_torch")
    if optimizer_name in {"adamw_torch", "adam"}:
        trial_cfg.training.optimizer.use_fused = "auto"  # Enable fused ops when available

    # DataLoader settings for optimal performance
    trial_cfg.training.num_workers = 4  # Good balance for most systems
    trial_cfg.training.pin_memory = "auto"  # Enables on CUDA
    trial_cfg.training.persistent_workers = "auto"  # Keeps workers alive between epochs
    trial_cfg.training.prefetch_factor = 2  # Prefetch 2 batches per worker
    trial_cfg.training.eval_prefetch_factor = 2
    trial_cfg.training.test_prefetch_factor = 2
    trial_cfg.training.drop_last = True  # More stable batch normalization
    trial_cfg.training.shuffle_train = True  # Standard practice
    trial_cfg.training.dataloader_timeout = 0  # No timeout issues

    # Mixed precision for ~2-3x speedup on CUDA
    trial_cfg.training.mixed_precision = "auto"  # Will use fp16 on CUDA
    trial_cfg.training.amp_dtype = "auto"  # Appropriate dtype selection

    # PyTorch compilation for 10-20% speedup
    trial_cfg.training.compile_model = True
    trial_cfg.training.compile_mode = "default"  # Most stable mode
    trial_cfg.training.compile_backend = "inductor"  # Default backend
    trial_cfg.training.compile_dynamic = False  # Faster, most models have consistent shapes
    trial_cfg.training.compile_fullgraph = False  # More stable, avoids CUDA graph issues

    # cuDNN optimization
    trial_cfg.training.cudnn_benchmark = True  # Auto-tunes algorithms
    trial_cfg.training.cudnn_deterministic = False  # Faster, seeds handle reproducibility

    # No max_steps limit
    trial_cfg.training.max_steps = None

    # Disable progress bars for Optuna search
    trial_cfg.logging.use_tqdm = False


def build_trial_config(cfg: DictConfig, trial: optuna.Trial) -> DictConfig:
    params = _sample_params(cfg, trial)
    trial.set_user_attr("applied_params", params)
    trial_cfg = _apply_params(cfg, params)
    _set_performance_defaults(trial_cfg)
    return trial_cfg


def _create_sampler(cfg: DictConfig) -> optuna.samplers.BaseSampler:
    sampler_name = str(cfg.optuna.get("sampler", "tpe")).lower()
    seed = cfg.optuna.get("sampler_seed")
    if sampler_name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if sampler_name == "cmaes":
        return optuna.samplers.CmaEsSampler(seed=seed)
    return optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)


def _create_pruner(cfg: DictConfig) -> optuna.pruners.BasePruner:
    pruner_cfg = cfg.optuna.get("pruner", {})
    name = str(pruner_cfg.get("name", "patient")).lower()
    warmup_steps = int(pruner_cfg.get("warmup_steps", 5))
    patience = int(pruner_cfg.get("patience", 2))
    if name == "nop":
        return optuna.pruners.NopPruner()
    if name == "median":
        return optuna.pruners.MedianPruner(n_warmup_steps=warmup_steps)
    base_pruner = optuna.pruners.MedianPruner(n_warmup_steps=warmup_steps)
    return optuna.pruners.PatientPruner(base_pruner, patience=patience)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s", force=True)
    sampler = _create_sampler(cfg)
    pruner = _create_pruner(cfg)
    direction = "maximize" if cfg.optimization.higher_is_better else "minimize"

    # Use persistent storage for auto-resume capability
    study_name = cfg.optuna.get("study_name", "spanbert_qa_optimization")
    storage_path = Path(cfg.optuna.get("storage_path", "optuna_studies"))
    storage_path.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{storage_path / f'{study_name}.db'}"

    # Load or create study
    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage,
            sampler=sampler,
            pruner=pruner,
        )
        logger.info(f"Resuming existing study '{study_name}' with {len(study.trials)} completed trials")
    except KeyError:
        study = optuna.create_study(
            study_name=study_name,
            direction=direction,
            sampler=sampler,
            pruner=pruner,
            storage=storage,
            load_if_exists=True,
        )
        logger.info(f"Created new study '{study_name}'")

    timeout = cfg.optuna.get("timeout")
    n_trials = cfg.optuna.get("n_trials")

    def objective(trial: optuna.Trial) -> float:
        trial_cfg = build_trial_config(cfg, trial)
        metric_name = trial_cfg.optimization.metric
        try:
            outputs = run_training(trial_cfg, save_artifacts=False, trial=trial)
        except optuna.TrialPruned as exc:
            logger.info("Trial %d pruned: %s", trial.number, exc)
            raise
        metric_value = outputs.val_metrics.get(metric_name, float("nan"))
        if metric_value != metric_value:  # NaN guard
            metric_value = float("-inf") if cfg.optimization.higher_is_better else float("inf")
        logger.info("Trial %s %s=%.4f", trial.number, metric_name, metric_value)
        return metric_value

    study.optimize(objective, n_trials=n_trials, timeout=timeout, gc_after_trial=True)
    if not study.best_trials:
        logger.warning("No successful trials completed.")
        return

    best_trial = study.best_trial
    best_params = best_trial.user_attrs.get("applied_params", best_trial.params)
    best_cfg = _apply_params(cfg, best_params)
    best_cfg.training.artifact_dir = str(Path(cfg.training.artifact_dir) / "optuna_best")

    logger.info("Best trial #%d value %.4f", best_trial.number, best_trial.value)
    logger.info("Best params: %s", best_params)

    outputs = run_training(best_cfg, save_artifacts=True)
    logger.info("Optuna best validation metrics: %s", outputs.val_metrics)
    logger.info("Optuna best test metrics: %s", outputs.test_metrics)
    logger.info("Artifacts stored in: %s", outputs.artifact_dir)


if __name__ == "__main__":
    main()
