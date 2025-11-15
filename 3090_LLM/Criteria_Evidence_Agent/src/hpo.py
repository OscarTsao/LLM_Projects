"""Hyperparameter optimization using Optuna."""

import os
import traceback
from typing import Any, Dict, Optional

import hydra
import mlflow
import optuna
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from src.train import start_wandb_run, train_loop
from src.utils import (
    configure_mlflow,
    MemoryMonitor,
    set_memory_efficient_environment,
    suggest_memory_optimizations,
    get_gpu_memory_info,
)


class MemoryAwarePruner:
    """Custom pruner that considers memory usage patterns."""

    def __init__(self, base_pruner: optuna.pruners.BasePruner, memory_threshold_gb: float = 20.0):
        """Initialize memory-aware pruner.

        Args:
            base_pruner: Base pruner to wrap
            memory_threshold_gb: Memory threshold for aggressive pruning
        """
        self.base_pruner = base_pruner
        self.memory_threshold_gb = memory_threshold_gb
        self.oom_trials = set()

    def prune(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> bool:
        """Decide whether to prune a trial based on memory and performance.

        Args:
            study: Optuna study
            trial: Trial to potentially prune

        Returns:
            True if trial should be pruned
        """
        # First check base pruner
        if self.base_pruner.prune(study, trial):
            return True

        # Memory-based pruning logic
        if torch.cuda.is_available():
            memory_info = get_gpu_memory_info()

            # If we're running low on memory, be more aggressive with pruning
            if memory_info["free"] < 4.0:  # Less than 4GB free
                # Prune trials that are performing poorly early
                if len(trial.intermediate_values) >= 2:
                    recent_values = list(trial.intermediate_values.values())[-2:]
                    if all(v < 0.1 for v in recent_values):  # Very poor performance
                        print(
                            f"🔪 Memory-based pruning: Trial {trial.number} (low memory + poor performance)"
                        )
                        return True

        # Don't prune trials that have already had OOM issues and were recovered
        # (they might be valuable for understanding memory limits)
        if hasattr(trial, "user_attrs") and trial.user_attrs.get("batch_size_reduced", False):
            return False

        return False


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
            self.pbar.set_postfix(
                {"best": f"{study.best_value:.4f}", "current": f"{trial.value:.4f}"}
            )

    def close(self) -> None:
        """Close the progress bar."""
        if self.pbar is not None:
            self.pbar.close()


def get_memory_constraints(model_name: str) -> Dict[str, Any]:
    """Get memory constraints based on model type.

    Args:
        model_name: Pretrained model name

    Returns:
        Dictionary with memory constraints
    """
    name = (model_name or "").lower()

    # DeBERTa models have higher memory usage
    if "deberta" in name:
        return {
            "max_batch_size": 64,
            "max_length_choices": [128, 256],
            "force_gradient_checkpointing": True,
            "model_type": "deberta",
        }

    # XLNet models benefit from conservative defaults due to relative attention memory use
    if "xlnet" in name:
        return {
            "max_batch_size": 64,
            "max_length_choices": [128, 256, 384],
            "force_gradient_checkpointing": False,
            "model_type": "xlnet",
        }

    # Electra models follow BERT-like memory characteristics
    if "electra" in name:
        return {
            "max_batch_size": 96,
            "max_length_choices": [128, 256, 384, 512],
            "force_gradient_checkpointing": False,
            "model_type": "electra",
        }

    # RoBERTa family
    if "roberta" in name:
        return {
            "max_batch_size": 128,
            "max_length_choices": [128, 256, 384, 512],
            "force_gradient_checkpointing": False,
            "model_type": "roberta",
        }

    # Default to BERT family characteristics
    return {
        "max_batch_size": 128,
        "max_length_choices": [128, 256, 384, 512],
        "force_gradient_checkpointing": False,
        "model_type": "bert",
    }


# Note: filter_choices_by_memory_constraints function removed
# We now use a fixed parameter space with the most restrictive constraints
# to ensure consistent Optuna categorical distributions across all trials


def validate_optimizer_scheduler_compatibility(
    optimizer_name: str, scheduler_name: str
) -> Dict[str, Any]:
    """Validate optimizer-scheduler compatibility.

    Args:
        optimizer_name: Name of the optimizer
        scheduler_name: Name of the scheduler

    Returns:
        Dictionary with compatibility information
    """
    compatibility = {"compatible": True, "warnings": [], "adjustments": {}}

    # Define optimizer momentum support
    momentum_optimizers = {"adamw", "lamb", "sgd", "adam", "rmsprop"}
    no_momentum_optimizers = {"adafactor", "adagrad", "adadelta"}

    optimizer_lower = optimizer_name.lower()
    scheduler_lower = scheduler_name.lower()

    # Check OneCycleLR compatibility
    if scheduler_lower == "onecycle":
        if optimizer_lower in no_momentum_optimizers:
            compatibility["warnings"].append(
                f"OneCycleLR with {optimizer_name}: cycle_momentum will be disabled "
                f"(optimizer doesn't support momentum parameters)"
            )
            compatibility["adjustments"]["onecycle_cycle_momentum"] = False
        elif optimizer_lower in momentum_optimizers:
            compatibility["adjustments"]["onecycle_cycle_momentum"] = True
        else:
            # Unknown optimizer - be conservative
            compatibility["warnings"].append(
                f"Unknown optimizer {optimizer_name} with OneCycleLR: "
                f"cycle_momentum will be disabled for safety"
            )
            compatibility["adjustments"]["onecycle_cycle_momentum"] = False

    # Check for other potential incompatibilities
    if scheduler_lower == "plateau" and optimizer_lower == "adafactor":
        compatibility["warnings"].append(
            f"ReduceLROnPlateau with Adafactor may have suboptimal behavior "
            f"due to Adafactor's internal learning rate scaling"
        )

    return compatibility


def validate_parameter_combination(
    params: Dict[str, Any], memory_constraints: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate and potentially adjust parameter combinations for memory efficiency.

    Args:
        params: Trial parameters
        memory_constraints: Memory constraints for the model

    Returns:
        Dictionary with validation results and suggestions
    """
    validation = {
        "valid": True,
        "warnings": [],
        "adjustments": {},
        "risk_level": "low",  # low, medium, high
    }

    batch_size = params.get("training.batch_size", 8)
    max_length = params.get("data.max_length", 128)
    grad_accum = params.get("training.gradient_accumulation_steps", 1)
    model_name = params.get("model.encoder.pretrained_model_name_or_path", "")

    effective_batch_size = batch_size * grad_accum

    # Check against memory constraints
    if batch_size > memory_constraints["max_batch_size"]:
        validation["warnings"].append(
            f"Batch size {batch_size} exceeds recommended {memory_constraints['max_batch_size']} for {model_name}"
        )
        validation["risk_level"] = "high"

    if max_length not in memory_constraints["max_length_choices"]:
        validation["warnings"].append(
            f"Max length {max_length} not in recommended choices {memory_constraints['max_length_choices']} for {model_name}"
        )
        validation["risk_level"] = "medium" if validation["risk_level"] == "low" else "high"

    # Check for risky combinations
    if effective_batch_size > 512:
        validation["warnings"].append(f"Very large effective batch size: {effective_batch_size}")
        validation["risk_level"] = "high"

    if "deberta" in model_name.lower() and batch_size > 32 and max_length > 256:
        validation["warnings"].append(
            "High-risk combination: DeBERTa with large batch_size and max_length"
        )
        validation["risk_level"] = "high"

    # Suggest adjustments for high-risk combinations
    if validation["risk_level"] == "high":
        validation["adjustments"]["enable_gradient_checkpointing"] = True
        if batch_size > 16:
            validation["adjustments"]["suggested_batch_size"] = min(16, batch_size // 2)
        if max_length > 384:
            validation["adjustments"]["suggested_max_length"] = min(384, max_length // 2)

    return validation


def is_memory_intensive_combination(params: Dict[str, Any]) -> bool:
    """Check if parameter combination is memory intensive.

    Args:
        params: Trial parameters

    Returns:
        True if combination is memory intensive
    """
    batch_size = params.get("training.batch_size", 8)
    max_length = params.get("data.max_length", 128)
    grad_accum = params.get("training.gradient_accumulation_steps", 1)
    model_name = params.get("model.encoder.pretrained_model_name_or_path", "")

    # Calculate effective batch size
    effective_batch_size = batch_size * grad_accum

    # Memory intensive if:
    # 1. DeBERTa with batch_size > 32 or max_length > 256
    # 2. Any model with effective_batch_size > 256 and max_length > 256
    # 3. Very large sequences (max_length > 384) with large batches (batch_size > 32)

    if "deberta" in model_name.lower():
        return batch_size > 32 or max_length > 256

    return (effective_batch_size > 256 and max_length > 256) or (
        max_length > 384 and batch_size > 32
    )


def validate_model_parameter_compatibility(params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate model-specific parameter compatibility.

    Args:
        params: Trial parameters

    Returns:
        Dictionary with validation results and adjustments
    """
    validation = {
        "compatible": True,
        "warnings": [],
        "adjustments": {},
    }

    model_name = params.get("model.encoder.pretrained_model_name_or_path", "").lower()
    gradient_checkpointing = params.get("model.encoder.gradient_checkpointing", False)

    # XLNet models don't support gradient checkpointing
    if "xlnet" in model_name and gradient_checkpointing:
        validation["compatible"] = False
        validation["warnings"].append(
            "XLNet models do not support gradient checkpointing due to their relative attention mechanism"
        )
        validation["adjustments"]["model.encoder.gradient_checkpointing"] = False

    return validation


def try_reduced_batch_size(
    trial: optuna.Trial,
    trial_cfg: DictConfig,
    trial_params: Dict[str, Any],
    memory_constraints: Dict[str, Any],
) -> Optional[float]:
    """Try training with reduced batch size after OOM.

    Args:
        trial: Optuna trial object
        trial_cfg: Trial configuration
        trial_params: Trial parameters
        memory_constraints: Memory constraints for the model

    Returns:
        Training metric if successful, None if still OOM
    """
    original_batch_size = trial_params.get("training.batch_size", 8)

    # Try progressively smaller batch sizes
    for reduction_factor in [0.5, 0.25]:
        new_batch_size = max(1, int(original_batch_size * reduction_factor))

        if new_batch_size >= original_batch_size:
            continue  # Skip if no reduction

        print(
            f"🔄 Retrying trial {trial.number} with reduced batch_size: {original_batch_size} -> {new_batch_size}"
        )

        # Update configuration
        OmegaConf.update(trial_cfg, "training.batch_size", new_batch_size, merge=False)

        # Also increase gradient accumulation to maintain effective batch size
        original_grad_accum = trial_params.get("training.gradient_accumulation_steps", 1)
        new_grad_accum = min(8, original_grad_accum * 2)
        OmegaConf.update(
            trial_cfg, "training.gradient_accumulation_steps", new_grad_accum, merge=False
        )

        # Force gradient checkpointing
        OmegaConf.update(trial_cfg, "model.encoder.gradient_checkpointing", True, merge=False)

        try:
            with MemoryMonitor(trial_number=trial.number, clear_cache=True):
                result = train_loop(trial_cfg)
                metric = result["best_metric"]

                print(
                    f"✅ Trial {trial.number} succeeded with reduced batch_size={new_batch_size}, grad_accum={new_grad_accum}"
                )

                # Log the successful reduction
                if mlflow.active_run():
                    mlflow.log_param("batch_size_reduced", True)
                    mlflow.log_param("final_batch_size", new_batch_size)
                    mlflow.log_param("final_grad_accum", new_grad_accum)

                return metric

        except (torch.cuda.OutOfMemoryError, RuntimeError) as retry_error:
            if "out of memory" in str(retry_error).lower():
                print(f"🚨 Still OOM with batch_size={new_batch_size}, trying smaller...")
                continue
            else:
                raise  # Re-raise non-OOM errors

    print(f"❌ Trial {trial.number} failed even with minimum batch size")
    return None


def handle_oom_error(trial: optuna.Trial, error: Exception, params: Dict[str, Any]) -> float:
    """Handle CUDA out-of-memory errors gracefully.

    Args:
        trial: Optuna trial object
        error: The OOM error that occurred
        params: Trial parameters that caused OOM

    Returns:
        Penalty value for the trial
    """
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # Log OOM information
    oom_info = {
        "trial_number": trial.number,
        "error_type": type(error).__name__,
        "batch_size": params.get("training.batch_size", "unknown"),
        "max_length": params.get("data.max_length", "unknown"),
        "gradient_accumulation_steps": params.get(
            "training.gradient_accumulation_steps", "unknown"
        ),
        "model": params.get("model.encoder.pretrained_model_name_or_path", "unknown"),
        "effective_batch_size": params.get("training.batch_size", 1)
        * params.get("training.gradient_accumulation_steps", 1),
    }

    print(f"\n{'='*80}")
    print(f"🚨 CUDA OOM Error in Trial {trial.number}")
    print(f"Model: {oom_info['model']}")
    print(f"Batch Size: {oom_info['batch_size']}")
    print(f"Max Length: {oom_info['max_length']}")
    print(f"Gradient Accumulation: {oom_info['gradient_accumulation_steps']}")
    print(f"Effective Batch Size: {oom_info['effective_batch_size']}")
    print(f"Error: {str(error)}")
    print("💡 Suggestion: Consider setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    print(f"{'='*80}\n")

    # Log to MLflow if available
    try:
        if mlflow.active_run():
            mlflow.log_params({f"oom_{k}": v for k, v in oom_info.items()})
            mlflow.log_metric("oom_occurred", 1.0)
            mlflow.set_tag("trial_status", "oom_error")
    except Exception as mlflow_error:
        print(f"Warning: Could not log OOM info to MLflow: {mlflow_error}")

    # Return penalty value (0.0 for maximization problems)
    return 0.0


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
    # Set memory-efficient environment variables
    set_memory_efficient_environment()

    study_cfg = cfg.hpo

    tracking_uri, artifact_uri = configure_mlflow(cfg)
    print(f"Using MLflow tracking URI: {tracking_uri}")
    if artifact_uri:
        print(f"Using MLflow artifact URI: {artifact_uri}")

    # Warn about n_jobs > 1 with CUDA and provide memory optimization tips
    if study_cfg.n_jobs > 1 and torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("WARNING: n_jobs > 1 with CUDA detected!")
        print(f"Current n_jobs: {study_cfg.n_jobs}")
        print("Optuna's multiprocessing (n_jobs > 1) can cause issues with CUDA.")
        print("For GPU training, it's recommended to set hpo.n_jobs=1")
        print("If you want parallel trials, consider using multiple GPUs with CUDA_VISIBLE_DEVICES")
        print("=" * 80 + "\n")

    # Display memory optimization information
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"\n🔧 Memory-Aware HPO Enabled")
        print(f"GPU: {gpu_name} ({gpu_memory_gb:.1f} GB)")
        print("Fixed parameter space (most restrictive constraints for all models):")
        print("  • Batch size: max 64 (safe for all models)")
        print("  • Max length: [128, 256] (safe for all models)")
        print("  • Model-specific optimizations applied within these constraints")
        print("  • Gradient checkpointing auto-enabled for DeBERTa models")
        print("  • OOM errors handled gracefully with penalty values")
        print("💡 For additional memory: export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
        print()

    # Instantiate sampler and pruner
    sampler = hydra.utils.instantiate(study_cfg.sampler) if "sampler" in study_cfg else None
    base_pruner = hydra.utils.instantiate(study_cfg.pruner) if "pruner" in study_cfg else None

    # Wrap with memory-aware pruner if we have a base pruner
    if base_pruner is not None:
        gpu_memory_gb = (
            torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if torch.cuda.is_available()
            else 24.0
        )
        pruner = MemoryAwarePruner(base_pruner, memory_threshold_gb=gpu_memory_gb * 0.8)
        print(f"🧠 Using memory-aware pruner with {gpu_memory_gb:.1f} GB GPU memory")
    else:
        pruner = None

    # Create or load study
    storage = study_cfg.storage
    # Use in-memory storage if storage is None or empty string
    if not storage:
        storage = None
        print("Using in-memory storage (no persistence between runs)")
    else:
        print(f"Using persistent storage: {storage}")

    # Try to create study, handle database compatibility issues
    try:
        study = optuna.create_study(
            study_name=study_cfg.study_name,
            storage=storage,
            load_if_exists=True,
            direction=study_cfg.direction,
            sampler=sampler,
            pruner=pruner,
        )
    except RuntimeError as e:
        if "no longer compatible with the table schema" in str(e):
            if storage and storage.startswith("sqlite:///"):
                # Extract database path and delete it
                db_path = storage.replace("sqlite:///", "")
                if os.path.exists(db_path):
                    print(f"\nWarning: Incompatible Optuna database detected at {db_path}")
                    print("Deleting incompatible database and creating a new one...")
                    os.remove(db_path)
                    # Retry creating the study
                    study = optuna.create_study(
                        study_name=study_cfg.study_name,
                        storage=storage,
                        load_if_exists=False,
                        direction=study_cfg.direction,
                        sampler=sampler,
                        pruner=pruner,
                    )
                else:
                    raise
            else:
                print("\nError: Incompatible Optuna database schema.")
                print(
                    "Please delete the database manually or use in-memory storage (set storage to null)."
                )
                raise
        else:
            raise

    search_space = OmegaConf.to_container(study_cfg.search_space, resolve=True)

    def infer_encoder_type(model_name: str, default_type: str) -> str:
        """Infer encoder type from pretrained name for factory compatibility."""
        name = (model_name or "").lower()
        if "deberta" in name:
            return "deberta"
        if "roberta" in name:
            return "roberta"
        if "electra" in name:
            return "electra"
        if "xlnet" in name:
            return "xlnet"
        if "albert" in name:
            return "bert"
        if "spanbert" in name:
            return "bert"
        if "bert" in name:
            return "bert"
        return default_type

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function with fixed parameter space and model-specific optimizations.

        Args:
            trial: Optuna trial object

        Returns:
            Metric to optimize (0.0 if OOM error occurs)
        """
        try:
            # Create trial configuration
            trial_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

            # Track trial parameters for OOM handling and model-specific optimizations
            trial_params = {}

            # Suggest all parameters using the fixed search space (no dynamic filtering)
            for param_path, spec in search_space.items():
                value = suggest_value(trial, param_path, spec)
                trial_params[param_path] = value
                OmegaConf.update(trial_cfg, param_path, value, merge=False)

            # Get model name and constraints for optimizations
            model_name = trial_params.get(
                "model.encoder.pretrained_model_name_or_path",
                trial_cfg.model.encoder.pretrained_model_name_or_path,
            )
            memory_constraints = get_memory_constraints(model_name)

            # Validate model-parameter compatibility
            compatibility = validate_model_parameter_compatibility(trial_params)
            if not compatibility["compatible"]:
                print(f"  🔧 Model compatibility adjustments needed:")
                for warning in compatibility["warnings"]:
                    print(f"    - {warning}")
                # Apply adjustments
                for param_path, value in compatibility["adjustments"].items():
                    print(f"    - Setting {param_path} = {value}")
                    OmegaConf.update(trial_cfg, param_path, value, merge=False)

            print(f"Trial {trial.number}: {model_name} (type: {memory_constraints['model_type']})")
            print(
                f"  Parameters: batch_size={trial_params.get('training.batch_size', 'default')}, "
                f"max_length={trial_params.get('data.max_length', 'default')}, "
                f"grad_accum={trial_params.get('training.gradient_accumulation_steps', 'default')}"
            )

            # Ensure encoder type matches pretrained model
            pretrained = trial_cfg.model.encoder.pretrained_model_name_or_path
            encoder_type = infer_encoder_type(pretrained, trial_cfg.model.encoder.type)
            OmegaConf.update(trial_cfg, "model.encoder.type", encoder_type, merge=False)

            # Apply model-specific optimizations within the fixed parameter space
            optimizations_applied = []

            # Always enable gradient checkpointing for DeBERTa models (higher memory usage)
            if memory_constraints["force_gradient_checkpointing"]:
                print(
                    f"  🔧 Enabling gradient checkpointing for {memory_constraints['model_type']} model"
                )
                OmegaConf.update(
                    trial_cfg, "model.encoder.gradient_checkpointing", True, merge=False
                )
                optimizations_applied.append("gradient_checkpointing")

            # Check for model-specific incompatibilities before applying memory optimizations
            model_name = trial_params.get("model.encoder.pretrained_model_name_or_path", "").lower()
            gradient_checkpointing_requested = trial_params.get("model.encoder.gradient_checkpointing", False)

            # XLNet models don't support gradient checkpointing
            if "xlnet" in model_name and gradient_checkpointing_requested:
                print(f"  🚫 Disabling gradient checkpointing for XLNet model (not supported)")
                OmegaConf.update(trial_cfg, "model.encoder.gradient_checkpointing", False, merge=False)
                optimizations_applied.append("disabled_gradient_checkpointing_xlnet")

            # Check if combination is memory intensive even within our restricted space
            if is_memory_intensive_combination(trial_params):
                print(f"  ⚠️  Memory-intensive combination detected")

                # Enable gradient checkpointing if not already enabled and model supports it
                if not memory_constraints["force_gradient_checkpointing"] and "xlnet" not in model_name:
                    print(f"  🔧 Enabling gradient checkpointing for memory-intensive combination")
                    OmegaConf.update(
                        trial_cfg, "model.encoder.gradient_checkpointing", True, merge=False
                    )
                    optimizations_applied.append("gradient_checkpointing")
                elif "xlnet" in model_name:
                    print(f"  ⚠️  Cannot enable gradient checkpointing for XLNet model - using other memory optimizations")

                # Reduce EMA decay for memory savings if it's high
                if trial_cfg.training.ema_decay > 0.999:
                    print(
                        f"  🔧 Reducing EMA decay from {trial_cfg.training.ema_decay} to 0.995 for memory"
                    )
                    OmegaConf.update(trial_cfg, "training.ema_decay", 0.995, merge=False)
                    optimizations_applied.append("reduced_ema_decay")

            # Validate optimizer-scheduler compatibility
            optimizer_name = trial_params.get("training.optimizer.name", "adamw")
            scheduler_name = trial_params.get("training.scheduler.name", "linear")

            compatibility = validate_optimizer_scheduler_compatibility(
                optimizer_name, scheduler_name
            )
            if compatibility["warnings"]:
                print(f"  ⚠️  Optimizer-scheduler compatibility:")
                for warning in compatibility["warnings"]:
                    print(f"    - {warning}")

            # Apply compatibility adjustments
            if compatibility["adjustments"]:
                for adjustment, value in compatibility["adjustments"].items():
                    if adjustment == "onecycle_cycle_momentum":
                        # This will be handled automatically by the scheduler
                        optimizations_applied.append(f"onecycle_cycle_momentum={value}")

            # Validate the final parameter combination (for informational purposes)
            validation = validate_parameter_combination(trial_params, memory_constraints)
            if validation["warnings"]:
                print(f"  ℹ️  Memory analysis (risk level: {validation['risk_level']}):")
                for warning in validation["warnings"]:
                    print(f"    - {warning}")

            if optimizations_applied:
                print(f"  ✅ Applied optimizations: {', '.join(optimizations_applied)}")

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

            OmegaConf.update(trial_cfg, "training.checkpoint.log_to_mlflow", False, merge=False)
            OmegaConf.update(trial_cfg, "training.checkpoint.keep_local", False, merge=False)

            if OmegaConf.select(trial_cfg, "wandb") is not None:
                OmegaConf.update(trial_cfg, "wandb.job_type", "hpo", merge=False)
                existing_group = OmegaConf.select(trial_cfg, "wandb.group")
                if not existing_group:
                    OmegaConf.update(trial_cfg, "wandb.group", study_cfg.study_name, merge=False)

            # Log memory info before training
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"  GPU Memory Available: {gpu_memory_gb:.1f} GB")
                print(
                    f"  Effective Batch Size: {trial_params.get('training.batch_size', 8) * trial_params.get('training.gradient_accumulation_steps', 1)}"
                )

            # Run training with trial configuration and memory monitoring
            wandb_run = start_wandb_run(trial_cfg, run_name=f"trial_{trial.number}", job_type="hpo")
            try:
                with MemoryMonitor(trial_number=trial.number, clear_cache=True):
                    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
                        # Log trial parameters for debugging
                        mlflow.log_params(
                            {f"trial_{k.replace('.', '_')}": v for k, v in trial_params.items()}
                        )
                        mlflow.log_param(
                            "memory_constraints_type", memory_constraints["model_type"]
                        )

                        result = train_loop(trial_cfg)
                        metric = result["best_metric"]
                        mlflow.log_metric("objective", metric, step=trial.number)
                        mlflow.set_tag("trial_status", "completed")

                        print(
                            f"✅ Trial {trial.number} completed successfully with metric: {metric:.4f}"
                        )
                        return metric

            finally:
                if wandb_run is not None:
                    wandb_run.finish()

        except torch.cuda.OutOfMemoryError as oom_error:
            # Try dynamic batch size reduction first
            reduced_metric = try_reduced_batch_size(
                trial, trial_cfg, trial_params, memory_constraints
            )
            if reduced_metric is not None:
                return reduced_metric
            else:
                return handle_oom_error(trial, oom_error, trial_params)

        except RuntimeError as runtime_error:
            # Check if it's a CUDA OOM error (sometimes wrapped in RuntimeError)
            if (
                "out of memory" in str(runtime_error).lower()
                or "cuda" in str(runtime_error).lower()
            ):
                # Try dynamic batch size reduction first
                reduced_metric = try_reduced_batch_size(
                    trial, trial_cfg, trial_params, memory_constraints
                )
                if reduced_metric is not None:
                    return reduced_metric
                else:
                    return handle_oom_error(trial, runtime_error, trial_params)
            else:
                # Re-raise other runtime errors
                print(f"❌ Trial {trial.number} failed with RuntimeError: {runtime_error}")
                raise

        except Exception as general_error:
            # Handle other unexpected errors
            print(f"❌ Trial {trial.number} failed with unexpected error: {general_error}")
            print(f"Error type: {type(general_error).__name__}")
            print(f"Traceback: {traceback.format_exc()}")

            # Clear GPU cache just in case
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Log error info to MLflow if possible
            try:
                if mlflow.active_run():
                    mlflow.log_param("error_type", type(general_error).__name__)
                    mlflow.log_param("error_message", str(general_error))
                    mlflow.set_tag("trial_status", "error")
            except Exception:
                pass

            # Re-raise the error to let Optuna handle it
            raise

    # Run optimization with progress bar
    progress_callback = TqdmCallback(
        n_trials=study_cfg.n_trials, desc=f"HPO ({study_cfg.study_name})"
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

    # Print optimization summary with memory statistics
    print(f"\n{'='*80}")
    print("🎯 HYPERPARAMETER OPTIMIZATION SUMMARY")
    print(f"{'='*80}")

    # Count trial outcomes
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

    # Count OOM trials (those with value = 0.0)
    oom_trials = [t for t in completed_trials if t.value == 0.0]
    successful_trials = [t for t in completed_trials if t.value > 0.0]

    print(f"Total trials: {len(study.trials)}")
    print(f"  ✅ Successful: {len(successful_trials)}")
    print(f"  🚨 OOM errors: {len(oom_trials)}")
    print(f"  ✂️  Pruned: {len(pruned_trials)}")
    print(f"  ❌ Failed: {len(failed_trials)}")

    if successful_trials:
        best_trial = study.best_trial
        print(f"\n🏆 Best trial: {best_trial.number}")
        print(f"🏆 Best value: {best_trial.value:.4f}")
        print(f"🏆 Best params:")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}")
    else:
        print("\n⚠️  No successful trials completed!")

    # Memory usage insights
    if oom_trials:
        print(f"\n💾 Memory Usage Analysis ({len(oom_trials)} OOM trials):")
        oom_models = {}
        reduced_batch_trials = []

        for trial in oom_trials:
            model = trial.params.get("model.encoder.pretrained_model_name_or_path", "unknown")
            if model not in oom_models:
                oom_models[model] = []
            oom_models[model].append(trial)

            # Check if this trial had batch size reduction
            if hasattr(trial, "user_attrs") and trial.user_attrs.get("batch_size_reduced", False):
                reduced_batch_trials.append(trial)

        for model, trials in oom_models.items():
            print(f"  📊 {model}: {len(trials)} OOM trials")
            if trials:
                avg_batch = sum(t.params.get("training.batch_size", 0) for t in trials) / len(
                    trials
                )
                avg_length = sum(t.params.get("data.max_length", 0) for t in trials) / len(trials)
                avg_grad_accum = sum(
                    t.params.get("training.gradient_accumulation_steps", 1) for t in trials
                ) / len(trials)
                avg_effective_batch = avg_batch * avg_grad_accum

                print(f"    Avg batch_size: {avg_batch:.1f}, Avg max_length: {avg_length:.1f}")
                print(
                    f"    Avg grad_accum: {avg_grad_accum:.1f}, Avg effective_batch: {avg_effective_batch:.1f}"
                )

                # Find memory limits
                max_successful_batch = 0
                max_successful_length = 0
                for trial in successful_trials:
                    if trial.params.get("model.encoder.pretrained_model_name_or_path") == model:
                        max_successful_batch = max(
                            max_successful_batch, trial.params.get("training.batch_size", 0)
                        )
                        max_successful_length = max(
                            max_successful_length, trial.params.get("data.max_length", 0)
                        )

                if max_successful_batch > 0:
                    print(
                        f"    💡 Safe limits: batch_size ≤ {max_successful_batch}, max_length ≤ {max_successful_length}"
                    )

        if reduced_batch_trials:
            print(
                f"\n🔄 Dynamic Recovery: {len(reduced_batch_trials)} trials recovered with batch size reduction"
            )

    # Optimizer-scheduler compatibility analysis
    if successful_trials:
        print(f"\n🔧 Optimizer-Scheduler Compatibility Analysis:")

        # Count optimizer-scheduler combinations
        combinations = {}
        for trial in successful_trials:
            optimizer = trial.params.get("training.optimizer.name", "unknown")
            scheduler = trial.params.get("training.scheduler.name", "unknown")
            combo = f"{optimizer}+{scheduler}"

            if combo not in combinations:
                combinations[combo] = {"count": 0, "avg_metric": 0, "trials": []}
            combinations[combo]["count"] += 1
            combinations[combo]["trials"].append(trial)

        # Calculate averages and show results
        for combo, data in combinations.items():
            trials = data["trials"]
            avg_metric = sum(t.value for t in trials) / len(trials)
            optimizer, scheduler = combo.split("+")

            # Check compatibility
            compatibility = validate_optimizer_scheduler_compatibility(optimizer, scheduler)
            status = "⚠️" if compatibility["warnings"] else "✅"

            print(f"  {status} {combo}: {len(trials)} trials, avg_metric: {avg_metric:.4f}")

            if compatibility["warnings"]:
                for warning in compatibility["warnings"]:
                    print(f"    - {warning}")

    # Parameter distribution analysis
    if successful_trials:
        print(f"\n📈 Successful Parameter Patterns:")

        # Analyze by model type
        model_success = {}
        for trial in successful_trials:
            model = trial.params.get("model.encoder.pretrained_model_name_or_path", "unknown")
            if model not in model_success:
                model_success[model] = {"trials": [], "avg_metric": 0}
            model_success[model]["trials"].append(trial)

        for model, data in model_success.items():
            trials = data["trials"]
            avg_metric = sum(t.value for t in trials) / len(trials)
            avg_batch = sum(t.params.get("training.batch_size", 0) for t in trials) / len(trials)
            avg_length = sum(t.params.get("data.max_length", 0) for t in trials) / len(trials)

            print(f"  🎯 {model}: {len(trials)} trials, avg_metric: {avg_metric:.4f}")
            print(f"    Avg batch_size: {avg_batch:.1f}, Avg max_length: {avg_length:.1f}")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
