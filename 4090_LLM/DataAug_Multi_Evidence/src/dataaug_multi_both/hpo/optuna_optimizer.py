"""
Optuna hyperparameter optimization integration.

Provides:
- Study creation and management
- Objective function for trial execution
- Search space configuration from YAML
- Pruning integration
- MLflow logging
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import mlflow
import optuna
import torch
import yaml  # type: ignore[import-untyped]
from dataaug_multi_both.cache.dataset_cache import (
    CACHE_ROOT_DEFAULT,
    CacheIndex,
    TokenizedDataset,
    compute_cache_key,
    try_load_tokenized_cache,
    MixedTokenizedDataset,
)

from dataaug_multi_both.checkpoints.retention import RetentionPolicy
from dataaug_multi_both.data.dataset_loader import DatasetConfig, DatasetLoader
from dataaug_multi_both.data.preprocessing import create_collator
from dataaug_multi_both.models import EvidenceExtractionModel
from dataaug_multi_both.training import EvidenceExtractionTrainer
from dataaug_multi_both.augment.unified_augmenter import UnifiedAugmenter, AugmentedDataset
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from dataaug_multi_both.utils.resources import build_dataloader_kwargs

logger = logging.getLogger(__name__)


class OptunaHPOOptimizer:
    """
    Manages Optuna hyperparameter optimization studies.

    Handles:
    - Study creation and loading
    - Trial execution with Trainer
    - Search space configuration
    - MLflow integration
    - Resumption of interrupted studies
    """

    def __init__(
        self,
        study_name: str,
        storage: str,
        direction: str = "maximize",
        sampler: optuna.samplers.BaseSampler | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
        mlflow_tracking_uri: str | None = None,
    ):
        """
        Initialize HPO optimizer.

        Args:
            study_name: Name of the Optuna study
            storage: Database URL for study persistence (e.g., sqlite:///study.db)
            direction: Optimization direction ("maximize" or "minimize")
            sampler: Optuna sampler (defaults to TPESampler)
            pruner: Optuna pruner (defaults to MedianPruner)
            mlflow_tracking_uri: MLflow tracking URI
        """
        self.study_name = study_name
        self.storage = storage
        self.direction = direction

        # Configure sampler and pruner
        self.sampler = sampler or TPESampler(seed=42)
        self.pruner = pruner or MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=3,
            interval_steps=1,
        )

        # Track the preferred MLflow tracking URI (CLI has precedence over config)
        self.mlflow_tracking_uri = mlflow_tracking_uri
        if self.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)

        logger.info(
            f"OptunaHPOOptimizer initialized: study={study_name}, "
            f"direction={direction}, storage={storage}"
        )

    def create_or_load_study(self, load_if_exists: bool = True) -> optuna.Study:
        """
        Create new study or load existing one.

        Args:
            load_if_exists: If True, load existing study; if False, raise error if exists

        Returns:
            Optuna study object
        """
        loaded_existing = load_if_exists
        try:
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                direction=self.direction,
                sampler=self.sampler,
                pruner=self.pruner,
                load_if_exists=load_if_exists,
            )
        except optuna.exceptions.DuplicatedStudyError:
            if not load_if_exists:
                logger.warning(
                    "Study '%s' already exists in storage '%s'. Falling back to loading the "
                    "existing study. Use the '--resume' flag or set resume=True when calling "
                    "run_optimization to make this explicit, or provide a new study name to "
                    "start a fresh run.",
                    self.study_name,
                    self.storage,
                )
                study = optuna.create_study(
                    study_name=self.study_name,
                    storage=self.storage,
                    direction=self.direction,
                    sampler=self.sampler,
                    pruner=self.pruner,
                    load_if_exists=True,
                )
                loaded_existing = True
            else:
                raise

        logger.info(
            f"Study '{self.study_name}' {'loaded' if loaded_existing else 'created'} "
            f"with {len(study.trials)} existing trials"
        )

        return study

    @staticmethod
    def _create_adamw_optimizer(
        parameters: Any,
        *,
        fused: bool,
        **kwargs: Any,
    ) -> Optimizer:
        try:
            if fused:
                return AdamW(parameters, fused=True, **kwargs)
            return AdamW(parameters, **kwargs)
        except TypeError as exc:
            if fused:
                logger.debug(
                    "AdamW fused optimizer unsupported (%s); falling back to standard implementation.",
                    exc,
                )
                return AdamW(parameters, **kwargs)
            raise

    def run_optimization(
        self,
        config: dict[str, Any],
        n_trials: int,
        timeout: float | None = None,
        resume: bool = False,
    ) -> optuna.Study:
        """
        Run hyperparameter optimization.

        Args:
            config: Configuration dictionary with search_space and other settings
            n_trials: Number of trials to run
            timeout: Maximum time in seconds (optional)
            resume: Whether to resume existing study

        Returns:
            Completed study object
        """
        # Create or load study
        study = self.create_or_load_study(load_if_exists=resume)

        # Extract configuration
        search_space = config.get("search_space", {})
        data_config = config.get("data", {})
        training_config = config.get("training", {})
        # Propagate fixed optimizer/defaults into training_config for downstream use
        fixed_cfg = config.get("fixed", {})
        if "optimizer" not in training_config and fixed_cfg.get("optimizer"):
            training_config["optimizer"] = fixed_cfg.get("optimizer")
        retention_config = config.get("checkpoint_retention", {}) or {
            "keep_last_n": config.get("fixed", {}).get("keep_last_n", 3),
            "keep_best_k": config.get("fixed", {}).get("keep_best_k", 5),
            "max_checkpoint_size_gb": config.get("fixed", {}).get("max_checkpoint_size_gb", 50.0),
        }

        # Create objective function
        def objective(trial: optuna.Trial) -> float:
            return self._objective(
                trial=trial,
                search_space=search_space,
                data_config=data_config,
                training_config=training_config,
                retention_config=retention_config,
                dataset_section=config.get("dataset", {}),
                resources=config.get("resources", {}),
                optimization_metric=(
                    config.get("fixed", {}).get("optimization_metric")
                    or config.get("optimization", {}).get("metric")
                    or "val_f1"
                ),
            )

        # Configure MLflow experiment (CLI-provided URI takes precedence over config)
        mlflow_cfg = config.get("mlflow", {})
        if getattr(self, "mlflow_tracking_uri", None):
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)  # type: ignore[arg-type]
        elif mlflow_cfg.get("tracking_uri"):
            mlflow.set_tracking_uri(mlflow_cfg["tracking_uri"])  # type: ignore[arg-type]
        if mlflow_cfg.get("experiment_name"):
            mlflow.set_experiment(mlflow_cfg["experiment_name"])  # type: ignore[arg-type]

        # Determine optimization metric name (trial runs log metrics directly)
        # metric_name removed as MLflowCallback is not used

        # MLflow callback removed to avoid nested run conflicts; metrics are logged from trial runs directly

        # Run optimization wrapped in a study-level MLflow parent run
        logger.info(f"Starting optimization: {n_trials} trials, timeout={timeout}s")
        with mlflow.start_run(run_name=f"study::{self.study_name}"):
            # Study metadata
            mlflow.set_tag("study.name", self.study_name)
            mlflow.log_params(
                {
                    "study.n_trials": n_trials,
                    "study.timeout": timeout or -1,
                    "study.direction": self.direction,
                }
            )

            # Dataset/model metadata on parent run for discoverability
            dataset_meta: dict[str, Any] = {}
            model_meta: dict[str, Any] = {}
            # Dataset identifiers
            if isinstance(config.get("dataset"), dict) and config["dataset"].get("config_path"):
                dataset_meta["study.dataset.config_path"] = str(config["dataset"]["config_path"])
            if data_config.get("dataset_name"):
                dataset_meta["study.dataset.id"] = str(data_config.get("dataset_name"))
            if data_config.get("local_path") or data_config.get("path"):
                dataset_meta["study.dataset.local_path"] = str(
                    data_config.get("local_path") or data_config.get("path")
                )
            if data_config.get("cache_dir"):
                dataset_meta["study.dataset.cache_dir"] = str(data_config.get("cache_dir"))
            # Model identifiers
            model_space = config.get("search_space", {}).get("model_id")
            if isinstance(model_space, dict) and "choices" in model_space:
                model_meta["study.model.choices"] = ",".join(map(str, model_space["choices"]))
            fixed_model = training_config.get("model_id") or config.get("fixed", {}).get("model_id")
            if fixed_model:
                model_meta["study.model.fixed"] = str(fixed_model)

            # Apply tags/params
            for k, v in dataset_meta.items():
                mlflow.set_tag(k, v)
            for k, v in model_meta.items():
                mlflow.set_tag(k, v)

            study.optimize(
                objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True,
            )

            # Log best trial to parent run
            mlflow.log_metric("study.best_value", study.best_value)
            mlflow.log_dict(study.best_params, "study_best_params.json")

        # Log best trial
        logger.info(f"Optimization complete. Best trial: {study.best_trial.number}")
        logger.info(f"Best value: {study.best_value:.4f}")
        logger.info(f"Best params: {study.best_params}")

        return study

    def _objective(
        self,
        trial: optuna.Trial,
        search_space: dict[str, Any],
        data_config: dict[str, Any],
        training_config: dict[str, Any],
        retention_config: dict[str, Any],
        dataset_section: dict[str, Any] | None = None,
        resources: dict[str, Any] | None = None,
        optimization_metric: str = "val_f1",
    ) -> float:
        """
        Objective function for a single trial.

        Args:
            trial: Optuna trial object
            search_space: Hyperparameter search space
            data_config: Dataset configuration
            training_config: Training configuration
            retention_config: Checkpoint retention configuration

        Returns:
            Validation metric value
        """
        # Sample hyperparameters
        params = self._sample_hyperparameters(trial, search_space)

        # Map/derive commonly used params to trainer/model names
        derived_params = dict(params)
        derived_params["dropout"] = params.get("evidence_dropout", params.get("dropout", 0.1))
        derived_params["num_epochs"] = params.get("epochs", training_config.get("epochs", 10))
        derived_params["gradient_accumulation_steps"] = params.get("accumulation_steps", 1)

        logger.info(f"Trial {trial.number}: Testing params {derived_params}")

        # Unified DataLoader settings
        dataloader_kwargs, train_loader_kwargs = build_dataloader_kwargs(
            resources_cfg=resources or {}, training_cfg=training_config or {}
        )

        # Start MLflow run for this trial
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            # Log trial parameters (both raw and derived)
            mlflow.log_params(derived_params)

            try:
                # Prefer local synthetic dataset for smoke testing if local path given
                local_path = data_config.get("local_path") or data_config.get("path")
                if local_path:
                    from torch.utils.data import Dataset

                    class SyntheticSpanDataset(Dataset):
                        def __init__(
                            self, n: int, seq_len: int = 64, vocab_size: int = 1000
                        ) -> None:
                            self.input_ids = torch.randint(
                                0, vocab_size, (n, seq_len), dtype=torch.long
                            )
                            self.attention_mask = torch.ones((n, seq_len), dtype=torch.long)
                            starts = torch.randint(0, seq_len // 2, (n,), dtype=torch.long)
                            spans = torch.randint(0, seq_len // 2, (n,), dtype=torch.long)
                            ends = torch.minimum(
                                starts + spans, torch.full_like(starts, seq_len - 1)
                            )
                            self.start_positions = starts
                            self.end_positions = ends

                        def __len__(self) -> int:
                            return self.input_ids.size(0)

                        def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
                            return {
                                "input_ids": self.input_ids[idx],
                                "attention_mask": self.attention_mask[idx],
                                "start_positions": self.start_positions[idx],
                                "end_positions": self.end_positions[idx],
                            }

                    n_train = int(training_config.get("synthetic_train_size", 64))
                    n_val = int(training_config.get("synthetic_val_size", 32))
                    seq_len = int(training_config.get("seq_len", 64))
                    train_dataset = SyntheticSpanDataset(n_train, seq_len=seq_len)
                    val_dataset = SyntheticSpanDataset(n_val, seq_len=seq_len)

                    mlflow.set_tag("study.dataset.local_path", str(local_path))
                    mlflow.set_tag("study.dataset.mode", "synthetic")

                # Resolve dataset info (prefer dataset_section.config_path if provided)
                elif (
                    dataset_section
                    and isinstance(dataset_section, dict)
                    and dataset_section.get("config_path")
                ):
                    ds_cfg_path = Path(dataset_section["config_path"])
                    with open(ds_cfg_path, encoding="utf-8") as f:
                        ds_yaml = yaml.safe_load(f)
                    ds = ds_yaml.get("dataset", {})
                    ds_cfg = DatasetConfig(
                        id=ds.get("id", "csv"),
                        revision=ds.get("revision"),
                        splits=ds.get(
                            "splits", {"train": "train", "validation": "validation", "test": "test"}
                        ),
                        streaming=bool(ds.get("streaming", False)),
                        cache_dir=ds.get("cache_dir"),
                        data_files=ds.get("data_files"),
                        split_percentages=ds.get("split_percentages"),
                    )
                    loader = DatasetLoader()
                    splits = loader.load(ds_cfg)
                    train_dataset = splits["train"]
                    val_dataset = splits["validation"]
                else:
                    # Fallback: raise error if no valid configuration provided
                    raise ValueError(
                        "Either local_path (for synthetic data) or dataset.config_path must be "
                        "provided in the configuration."
                    )

                # Attempt to use tokenized cache (speeds up CPU-bound tokenization)
                cache_root = Path(str(CACHE_ROOT_DEFAULT))
                index = CacheIndex(cache_root)

                # Preserve raw (text) datasets for potential on-the-fly augmentation fallback
                raw_train_dataset = train_dataset
                raw_val_dataset = val_dataset

                def _collect_selected_methods(p: dict[str, Any]) -> list[str]:
                    methods: list[str] = []
                    if isinstance(p.get("augmentation_methods"), list):
                        methods = [str(m) for m in p.get("augmentation_methods", []) if m]
                        return methods
                    # Collect aug_method_0..N
                    i = 0
                    while f"aug_method_{i}" in p:
                        methods.append(str(p[f"aug_method_{i}"]))
                        i += 1
                    n = int(p.get("num_augmentations", len(methods)))
                    return methods[:n]

                selected_methods = _collect_selected_methods(derived_params)
                using_cached_train = False
                using_cached_val = False
                augmented_cache_hit = False
                collator = None

                try:
                    # Build dataset file fingerprint inputs
                    dataset_files: list[Path] = []
                    if (
                        dataset_section
                        and isinstance(dataset_section, dict)
                        and dataset_section.get("config_path")
                    ):
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

                    tokenizer_model = str(derived_params["model_id"])
                    max_len = int(training_config.get("max_length", 512))

                    # Always try to use cached validation (non-augmented)
                    base_key = compute_cache_key(
                        dataset_files,
                        tokenizer_model=tokenizer_model,
                        max_length=max_len,
                        aug_params=None,
                    )
                    cached_val = try_load_tokenized_cache(cache_root, base_key, "validation", index)
                    if cached_val is not None:
                        val_dataset = cached_val
                        using_cached_val = True

                    # Training: prefer augmented caches if methods are selected
                    if selected_methods:
                        children: list[TokenizedDataset] = []
                        for m in selected_methods:
                            aug_key = compute_cache_key(
                                dataset_files,
                                tokenizer_model=tokenizer_model,
                                max_length=max_len,
                                aug_params={"methods": [m], "policy": "single"},
                            )
                            cached_aug = try_load_tokenized_cache(cache_root, aug_key, "train", index)
                            if cached_aug is None:
                                children = []
                                break
                            children.append(cached_aug)
                        if children:
                            train_dataset = MixedTokenizedDataset(children, seed=trial.number)
                            collator = None
                            using_cached_train = True
                            augmented_cache_hit = True
                            logger.info(
                                "Cache hit: using pre-tokenized augmented caches for train (%d methods)",
                                len(children),
                            )
                        else:
                            logger.info(
                                "Augmented caches missing for some methods; falling back to on-the-fly augmentation"
                            )
                            augmenter = UnifiedAugmenter(
                                aug_methods=selected_methods,
                                aug_prob=1.0,
                                compose_mode="random_one",
                                seed=trial.number,
                            )
                            train_dataset = AugmentedDataset(raw_train_dataset, augmenter=augmenter)
                    else:
                        # No augmentation selected: try base cached train
                        cached_train = try_load_tokenized_cache(cache_root, base_key, "train", index)
                        if cached_train is not None:
                            train_dataset = cached_train
                            collator = None
                            using_cached_train = True
                            logger.info("Cache hit: using tokenized cache for train/validation")

                except Exception as _cache_exc:
                    logger.warning("Cache lookup failed: %s", _cache_exc)

                # Create collator if not using tokenized caches for train
                if not using_cached_train:
                    collator = create_collator(
                        model_name_or_path=derived_params["model_id"],
                        max_length=training_config.get("max_length", 512),
                    )

                collate_kwargs = {"collate_fn": collator} if collator is not None else {}

                # Log cache usage tags
                try:
                    mlflow.set_tag("cache.train.tokenized", "hit" if using_cached_train else "miss")
                    mlflow.set_tag("cache.train.augmented", "hit" if augmented_cache_hit else "miss")
                    mlflow.set_tag("cache.val.tokenized", "hit" if using_cached_val else "miss")
                except Exception:
                    pass


                # Adaptive batch-size fallback on CUDA OOM
                initial_bs = int(derived_params["batch_size"])
                current_bs = max(1, initial_bs)
                attempts = 0
                while True:
                    attempts += 1
                    try:
                        # Create data loaders with current batch size
                        train_loader = DataLoader(
                            train_dataset,
                            batch_size=current_bs,
                            shuffle=True,
                            **train_loader_kwargs,
                            **collate_kwargs,
                        )

                        val_loader = DataLoader(
                            val_dataset,
                            batch_size=current_bs,
                            shuffle=False,
                            **dataloader_kwargs,
                            **collate_kwargs,
                        )

                        # Create model
                        model = EvidenceExtractionModel(
                            model_name_or_path=derived_params["model_id"],
                            head_type=derived_params.get("evidence_head_type", "start_end_linear"),
                            dropout=derived_params.get("dropout", 0.1),
                        )

                        # Create optimizer (supports optimizer-specific hyperparameters)
                        optimizer_name = str(
                            derived_params.get("optimizer") or training_config.get("optimizer") or "adamw"
                        ).lower()
                        optimizer: Optimizer
                        if optimizer_name == "sgd":
                            optimizer = SGD(
                                model.parameters(),
                                lr=derived_params["learning_rate"],
                                momentum=float(derived_params.get("momentum", 0.0)),
                                nesterov=bool(derived_params.get("nesterov", False)),
                                weight_decay=float(derived_params.get("weight_decay", 0.0)),
                            )
                        else:  # default adamw
                            betas = None
                            beta1 = derived_params.get("adam_beta1")
                            beta2 = derived_params.get("adam_beta2")
                            if beta1 is not None and beta2 is not None:
                                betas = (float(beta1), float(beta2))
                            adamw_kwargs: dict[str, Any] = {
                                "lr": derived_params["learning_rate"],
                                "weight_decay": float(derived_params.get("weight_decay", 0.01)),
                                "betas": betas if betas else (0.9, 0.999),
                            }
                            if derived_params.get("adam_eps") is not None:
                                adamw_kwargs["eps"] = float(derived_params["adam_eps"])
                            fused_optimizer = (
                                bool(training_config.get("fused_optimizer", True))
                                and torch.cuda.is_available()
                            )
                            optimizer = self._create_adamw_optimizer(
                                model.parameters(),
                                fused=fused_optimizer,
                                **adamw_kwargs,
                            )

                        # Create scheduler (optional)
                        scheduler = None
                        if training_config.get("use_scheduler", False):
                            scheduler = CosineAnnealingLR(optimizer, T_max=derived_params["num_epochs"])

                        # Create retention policy
                        retention_policy = RetentionPolicy(
                            keep_last_n=retention_config.get("keep_last_n", 3),
                            keep_best_k=retention_config.get("keep_best_k", 5),
                            max_checkpoint_size_gb=retention_config.get("max_checkpoint_size_gb", 50.0),
                        )

                        # Create trainer
                        output_root = (
                            Path(training_config.get("output_dir", "experiments")) / self.study_name
                        )
                        output_dir = output_root / f"trial_{trial.number}"
                        trainer = EvidenceExtractionTrainer(
                            model=model,
                            optimizer=optimizer,
                            train_dataloader=train_loader,
                            val_dataloader=val_loader,
                            output_dir=output_dir,
                            scheduler=scheduler,
                            retention_policy=retention_policy,
                            gradient_accumulation_steps=derived_params.get(
                                "gradient_accumulation_steps", 1
                            ),
                            early_stopping_patience=training_config.get("early_stopping_patience"),
                            metric_for_best_model=training_config.get(
                                "metric_for_best_model", optimization_metric
                            ),
                            mlflow_tracking=True,
                            use_amp=training_config.get("amp", True),
                            allow_tf32=training_config.get("allow_tf32", True),
                            non_blocking=training_config.get("non_blocking", True),
                        )

                        # Train with Optuna trial reporting
                        report_metric = optimization_metric.replace("val_", "")
                        result = trainer.train(
                            num_epochs=derived_params["num_epochs"],
                            trial=trial,
                            report_metric=report_metric,
                        )

                        # Return metric for optimization
                        best_metric = float(result["best_metric"])  # ensure concrete float for mypy
                        mlflow.log_metric("final_metric", best_metric)
                        if current_bs != initial_bs:
                            mlflow.set_tag("adaptive_bs", f"reduced_from_{initial_bs}_to_{current_bs}")
                        return best_metric

                    except RuntimeError as e:
                        msg = str(e).lower()
                        if ("out of memory" in msg or "cuda error" in msg) and current_bs > 1:
                            # Reduce batch size and retry
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
                        # Not an OOM or cannot reduce further
                        raise


            except Exception as e:
                logger.error(f"Trial {trial.number} failed: {e}")
                raise optuna.TrialPruned() from e

    def _sample_hyperparameters(
        self, trial: optuna.Trial, search_space: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Sample hyperparameters from search space.

        Args:
            trial: Optuna trial object
            search_space: Search space configuration

        Returns:
            Dictionary of sampled hyperparameters
        """
        params = {}

        def _get_range(cfg: dict[str, Any]) -> tuple[float, float]:
            if "min" in cfg and "max" in cfg:
                return cfg["min"], cfg["max"]
            if "low" in cfg and "high" in cfg:
                return cfg["low"], cfg["high"]
            raise ValueError("Range keys not found: expected ('min','max') or ('low','high')")

        # First pass: sample non-conditional parameters
        # Exclude conditional keys that depend on other choices (loss function, optimizer)
        conditional_keys = {
            "focal_gamma",
            "hybrid_weight_alpha",
            # Optimizer-specific
            "adam_beta1",
            "adam_beta2",
            "adam_eps",
            "momentum",
            "nesterov",
        }
        for param_name, param_config in search_space.items():
            if param_name in conditional_keys:
                continue
            param_type = (param_config or {}).get("type")

            if param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, param_config["choices"])

            elif param_type in {"float", "loguniform"}:
                low, high = _get_range(param_config)
                log = param_type == "loguniform" or bool(param_config.get("log", False))
                params[param_name] = trial.suggest_float(param_name, low, high, log=log)

            elif param_type == "int":
                low_f, high_f = _get_range(param_config)
                low, high = int(low_f), int(high_f)
                step = param_config.get("step")
                log = bool(param_config.get("log", False))
                if step is not None:
                    params[param_name] = trial.suggest_int(
                        param_name, low, high, step=int(step), log=log
                    )
                else:
                    params[param_name] = trial.suggest_int(param_name, low, high, log=log)

            elif param_type == "bool":
                # boolean as categorical unless explicit choices are provided
                choices = param_config.get("choices")
                if choices is not None:
                    params[param_name] = trial.suggest_categorical(param_name, choices)  # type: ignore[arg-type]
                else:
                    params[param_name] = trial.suggest_categorical(param_name, [True, False])

            else:
                logger.warning(f"Unknown parameter type: {param_type} for {param_name}")

        # Second pass: sample conditionals based on selected loss_function
        loss_fn = params.get("loss_function")
        if loss_fn == "focal" and "focal_gamma" in search_space:
            cfg = search_space["focal_gamma"]
            low, high = _get_range(cfg)
            log = bool(cfg.get("log", False)) or (cfg.get("type") == "loguniform")
            params["focal_gamma"] = trial.suggest_float("focal_gamma", low, high, log=log)
        if loss_fn == "hybrid" and "hybrid_weight_alpha" in search_space:
            cfg = search_space["hybrid_weight_alpha"]
            low, high = _get_range(cfg)
            params["hybrid_weight_alpha"] = trial.suggest_float("hybrid_weight_alpha", low, high)

        # Optimizer-specific conditionals
        optimizer_name = params.get("optimizer")
        if optimizer_name == "adamw":
            # Optional: sample AdamW-specific params if present
            if "adam_beta1" in search_space:
                cfg = search_space["adam_beta1"]
                low, high = _get_range(cfg)
                params["adam_beta1"] = trial.suggest_float("adam_beta1", low, high)
            if "adam_beta2" in search_space:
                cfg = search_space["adam_beta2"]
                low, high = _get_range(cfg)
                params["adam_beta2"] = trial.suggest_float("adam_beta2", low, high)
            if "adam_eps" in search_space:
                cfg = search_space["adam_eps"]
                low, high = _get_range(cfg)
                log = bool(cfg.get("log", False)) or (cfg.get("type") == "loguniform")
                params["adam_eps"] = trial.suggest_float("adam_eps", low, high, log=log)
        if optimizer_name == "sgd":
            if "momentum" in search_space:
                cfg = search_space["momentum"]
                low, high = _get_range(cfg)
                params["momentum"] = trial.suggest_float("momentum", low, high)
            if "nesterov" in search_space:
                cfg = search_space["nesterov"]
                if cfg.get("type") == "bool":
                    params["nesterov"] = trial.suggest_categorical("nesterov", [True, False])
                elif cfg.get("type") == "categorical" and "choices" in cfg:
                    params["nesterov"] = trial.suggest_categorical("nesterov", cfg["choices"])  # type: ignore[arg-type]

        return params

    def get_best_trial(self, study: optuna.Study | None = None) -> Any:
        """
        Get the best trial from the study.

        Args:
            study: Study object (if None, loads from storage)

        Returns:
            Best trial
        """
        if study is None:
            study = self.create_or_load_study()

        return study.best_trial

    def export_study_results(
        self, output_path: str | Path, study: optuna.Study | None = None
    ) -> None:
        """
        Export study results to CSV.

        Args:
            output_path: Path to save CSV file
            study: Study object (if None, loads from storage)
        """
        if study is None:
            study = self.create_or_load_study()

        df = study.trials_dataframe()
        df.to_csv(output_path, index=False)

        logger.info(f"Study results exported to {output_path}")
