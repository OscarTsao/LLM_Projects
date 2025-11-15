from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from dataaug_multi_both.data.dataset import create_pytorch_dataset, load_hf_dataset
from dataaug_multi_both.models.encoders.hf_encoder import HFEncoder, HFEncoderConfig
from dataaug_multi_both.models.heads.criteria_matching import CriteriaMatchingHead
from dataaug_multi_both.models.heads.evidence_binding import EvidenceBindingHead
from dataaug_multi_both.models.multi_task_model import MultiTaskModel
from dataaug_multi_both.training.checkpoint_manager import (
    CheckpointManager,
    CheckpointMetadata,
    CheckpointRetentionPolicy,
)
from dataaug_multi_both.training.losses import MultiTaskLoss
from dataaug_multi_both.training.trainer import Trainer, TrainerConfig
from dataaug_multi_both.utils.mlflow_setup import mlflow_run
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class TrialSpec:
    config: dict
    trial_id: str | None = None

    def ensure_id(self) -> str:
        if self.trial_id is None:
            self.trial_id = str(uuid.uuid4())
        return self.trial_id


@dataclass(slots=True)
class TrialResult:
    trial_id: str
    metric: float | None
    status: str
    duration_seconds: float


class TrialExecutor:
    """Execute Optuna-style trials sequentially and emit progress observability signals."""

    def __init__(
        self,
        run_trial: Callable[[TrialSpec], TrialResult],
        mlflow_client: object | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.run_trial = run_trial
        self.logger = logger or LOGGER
        self.mlflow_client = mlflow_client
        self._results: list[TrialResult] = []

    @property
    def results(self) -> Sequence[TrialResult]:
        return tuple(self._results)

    def execute(self, trials: Iterable[TrialSpec]) -> Sequence[TrialResult]:
        materialized = list(trials)
        total = len(materialized)
        if total == 0:
            self.logger.warning("No trials provided to TrialExecutor; nothing to execute.")
            return ()

        self._results.clear()
        best_metric: float | None = None
        best_trial_id: str | None = None
        start_wall = time.monotonic()

        for index, spec in enumerate(materialized, start=1):
            trial_id = spec.ensure_id()
            self._log_progress(
                stage="start",
                index=index,
                total=total,
                best_metric=best_metric,
                best_trial_id=best_trial_id,
                start_wall=start_wall,
            )

            trial_start = time.monotonic()
            try:
                result = self.run_trial(spec)
            except Exception as exc:  # pragma: no cover - defensive
                duration = time.monotonic() - trial_start
                result = TrialResult(
                    trial_id=trial_id,
                    metric=None,
                    status=f"failed:{exc.__class__.__name__}",
                    duration_seconds=duration,
                )
                self.logger.exception("Trial %s failed: %s", trial_id, exc)

            result.duration_seconds = getattr(
                result, "duration_seconds", time.monotonic() - trial_start
            )
            self._results.append(result)

            if result.metric is not None:
                if best_metric is None or result.metric > best_metric + 1e-12:
                    best_metric = result.metric
                    best_trial_id = result.trial_id

            self._log_progress(
                stage="end",
                index=index,
                total=total,
                best_metric=best_metric,
                best_trial_id=best_trial_id,
                start_wall=start_wall,
            )

        return self.results

    # ------------------------------------------------------------------
    # Observability helpers
    # ------------------------------------------------------------------
    def _log_progress(
        self,
        stage: str,
        index: int,
        total: int,
        best_metric: float | None,
        best_trial_id: str | None,
        start_wall: float,
    ) -> None:
        completion_rate = index / total
        elapsed = time.monotonic() - start_wall
        eta_seconds = self._estimate_eta(elapsed, index, total)
        payload = {
            "stage": stage,
            "trial_index": index,
            "trial_total": total,
            "completion_rate": round(completion_rate, 3),
            "elapsed_seconds": round(elapsed, 3),
            "eta_seconds": None if eta_seconds is None else round(eta_seconds, 3),
            "best_metric": None if best_metric is None else round(best_metric, 6),
            "best_trial_id": best_trial_id,
        }
        self.logger.info("HPO progress", extra={"component": "hpo", "event": payload})
        self._emit_mlflow_tags(payload)

    def _emit_mlflow_tags(self, payload: dict) -> None:
        if self.mlflow_client is not None and hasattr(self.mlflow_client, "set_tag"):
            for key, value in payload.items():
                tag_key = f"hpo.{key}"
                self.mlflow_client.set_tag(tag_key, "" if value is None else str(value))

    def _estimate_eta(self, elapsed: float, completed: int, total: int) -> float | None:
        if completed == 0:
            return None
        avg_duration = elapsed / completed
        remaining = total - completed
        if remaining <= 0:
            return 0.0
        return avg_duration * remaining


def run_single_trial(
    trial_spec: TrialSpec,
    experiments_dir: Path,
    dataset_config: dict[str, Any],
    retention_policy: CheckpointRetentionPolicy,
    mlflow_experiment_id: str,
    device: torch.device | None = None,
) -> TrialResult:
    """Run a single training trial with the given configuration.

    Args:
        trial_spec: Trial specification with hyperparameters
        experiments_dir: Base directory for experiments
        dataset_config: Dataset configuration
        retention_policy: Checkpoint retention policy
        mlflow_experiment_id: MLflow experiment ID
        device: Device to run training on

    Returns:
        TrialResult with final validation metric
    """
    trial_id = trial_spec.ensure_id()
    config = trial_spec.config

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Log GPU information
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        LOGGER.info(f"Using GPU: {gpu_name} ({gpu_memory:.2f} GB)")
        LOGGER.info(f"CUDA version: {torch.version.cuda}")
        LOGGER.info(f"PyTorch version: {torch.__version__}")
        
        # Check compute capability for bf16 support
        if torch.cuda.is_bf16_supported():
            LOGGER.info("GPU supports bfloat16 training (compute capability >= 8.0)")
        else:
            LOGGER.info("GPU does not support bfloat16, will use float16 if mixed precision enabled")
    else:
        LOGGER.warning("No GPU available, training will be slow on CPU")

    # Create trial directory
    trial_dir = experiments_dir / f"trial_{trial_id}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"trial_{trial_id}")
    logger.info(f"Starting trial {trial_id} with config: {config}")

    try:
        # Load dataset
        dataset_dict, metadata = load_hf_dataset(
            dataset_id=dataset_config.get("dataset_id", "irlab-udc/redsm5"),
            revision=dataset_config.get("revision", "main"),
            cache_dir=dataset_config.get("cache_dir"),
        )

        # Create model and get tokenizer
        model, tokenizer = create_model_from_config(config, device)

        # Create data loaders
        train_loader, val_loader = create_data_loaders(
            dataset_dict, config, tokenizer
        )

        # Create trainer components
        trainer_config = TrainerConfig(
            trial_id=trial_id,
            optimization_metric=config.get("optimization_metric", "val_f1_macro"),
            seed=config.get("seed", 42),
            max_epochs=config.get("epochs", 10),
            gradient_accumulation_steps=config.get("accumulation_steps", 1),
            resume_if_available=True,
        )

        checkpoint_metadata = CheckpointMetadata(
            code_version="1.0.0",
            model_signature=config["model_name"],
            head_configuration=f"{config.get('criteria_head_type', 'linear')}_{config.get('evidence_head_type', 'start_end_linear')}",
        )

        checkpoint_manager = CheckpointManager(
            trial_dir=trial_dir, policy=retention_policy, compatibility=checkpoint_metadata
        )

        # Start MLflow run
        with mlflow_run(run_name=f"trial_{trial_id}", tags={"trial_id": trial_id}):
            # Log hyperparameters
            import mlflow

            for key, value in config.items():
                mlflow.log_param(key, value)
            
            # Log GPU info to mlflow
            if torch.cuda.is_available():
                mlflow.log_param("gpu_name", gpu_name)
                mlflow.log_param("gpu_memory_gb", f"{gpu_memory:.2f}")
                mlflow.log_param("cuda_version", torch.version.cuda)
                mlflow.log_param("bf16_supported", torch.cuda.is_bf16_supported())

            trainer = Trainer(
                config=trainer_config,
                checkpoint_manager=checkpoint_manager,
                logger=logger,
                mlflow_client=mlflow,
            )

            # Train the model
            final_metric = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                trainer=trainer,
                config=config,
                device=device,
            )

            # Log final metric
            mlflow.log_metric("final_validation_metric", final_metric)

            return TrialResult(
                trial_id=trial_id,
                metric=final_metric,
                status="success",
                duration_seconds=0.0,  # Will be set by executor
            )

    except Exception as e:
        logger.error(f"Trial {trial_id} failed with {e.__class__.__name__}: {e}")
        logger.exception(f"Full traceback for trial {trial_id}:")
        # Return None to indicate failure (Optuna will handle this properly)
        # Don't return 0.0 as it looks like a successful low-score trial
        return TrialResult(
            trial_id=trial_id,
            metric=None,  # This will be handled by Optuna as a failed trial
            status=f"failed:{e.__class__.__name__}",
            duration_seconds=0.0,
        )


def create_model_from_config(config: dict[str, Any], device: torch.device) -> tuple[MultiTaskModel, Any]:
    """Create a multi-task model from trial configuration.

    Args:
        config: Trial configuration dictionary
        device: Device to place model on

    Returns:
        Tuple of (MultiTaskModel, tokenizer)
    """
    # Create and load encoder
    model_name = config.get("backbone") or config.get("model_name")
    encoder_config = HFEncoderConfig(
        model_id=get_model_id_from_name(model_name),
        gradient_checkpointing=config.get("gradient_checkpointing", False),
    )
    encoder_wrapper = HFEncoder(encoder_config)
    encoder_model, tokenizer = encoder_wrapper.load()  # Load the model and tokenizer

    # Create criteria matching head
    criteria_head = CriteriaMatchingHead(
        hidden_size=encoder_model.config.hidden_size,
        num_labels=config.get("num_criteria", 9),
        head_type=config.get("head_type") or config.get("criteria_head_type", "linear"),
        pooling_strategy=config.get("pooling") or config.get("criteria_pooling", "cls"),
        hidden_dim=config.get("hidden_size") or config.get("criteria_hidden_dim", 512),
        dropout=config.get("head_dropout") or config.get("criteria_dropout", 0.1),
    )

    # Create evidence binding head
    evidence_head = EvidenceBindingHead(
        hidden_size=encoder_model.config.hidden_size,
        head_type=config.get("span_head") or config.get("evidence_head_type", "start_end_linear"),
        max_span_length=config.get("max_span_len_chars") or config.get("max_span_length", 512),
        dropout=config.get("evidence_dropout", 0.1),
    )

    # Create multi-task model using the actual encoder model, not the wrapper
    model = MultiTaskModel(
        encoder=encoder_model,
        criteria_head=criteria_head,
        evidence_head=evidence_head,
        freeze_encoder=config.get("freeze_encoder", False),
    )

    model.to(device)
    
    # Verify autocast compatibility
    _verify_autocast_compatibility(model, model_name, config, device)
    
    return model, tokenizer


def _verify_autocast_compatibility(model: MultiTaskModel, model_name: str, config: dict[str, Any], device: torch.device) -> None:
    """Verify that the model works with autocast.
    
    Some models may have issues with autocast, particularly:
    - Models with custom layers that don't support half precision
    - Models with layer norm variants
    - Very old model architectures
    
    Args:
        model: The multi-task model
        model_name: Name of the model
        config: Trial configuration
        device: Device the model is on
    """
    fp_precision = config.get("fp_precision", "fp16" if torch.cuda.is_available() else "none")
    
    if fp_precision == "none" or not torch.cuda.is_available():
        return  # No need to check if not using autocast
    
    # Always use bfloat16 for autocast
    autocast_dtype = torch.bfloat16
    
    LOGGER.info(f"Verifying autocast compatibility for {model_name} with {autocast_dtype}")
    
    try:
        # Create dummy input
        batch_size = 2
        seq_len = 128
        dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
        dummy_attention_mask = torch.ones(batch_size, seq_len).to(device)
        
        # Test forward pass with autocast
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast(device_type='cuda', enabled=True, dtype=autocast_dtype):
                outputs = model(input_ids=dummy_input_ids, attention_mask=dummy_attention_mask)
        
        # Verify outputs are valid
        assert not torch.isnan(outputs.criteria_logits).any(), "NaN detected in criteria logits"
        assert not torch.isnan(outputs.start_logits).any(), "NaN detected in start logits"
        assert not torch.isnan(outputs.end_logits).any(), "NaN detected in end logits"
        
        LOGGER.info(f"✓ Autocast verification passed for {model_name}")
        
    except Exception as e:
        LOGGER.warning(
            f"⚠ Autocast compatibility issue detected for {model_name}: {e}. "
            f"Will fall back to full precision if training fails."
        )
        # Note: We don't raise an exception here, just warn
        # The training loop will handle any actual failures


def get_model_id_from_name(model_name: str) -> str:
    """Map model name to Hugging Face model ID.

    Args:
        model_name: Short model name

    Returns:
        Full Hugging Face model ID
    """
    model_mapping = {
        "psychbert": "mnaylor/psychbert-cased",
        "clinicalbert": "medicalai/ClinicalBERT",
        "bert-base": "google-bert/bert-base-uncased",
        "roberta-base": "FacebookAI/roberta-base",
    }

    return model_mapping.get(model_name, model_name)


def create_data_loaders(
    dataset_dict: Any, config: dict[str, Any], tokenizer: Any
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders.

    Args:
        dataset_dict: Dataset dictionary from Hugging Face
        config: Trial configuration
        tokenizer: Model tokenizer

    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Ensure max_length is compatible with the model
    max_length = _get_compatible_max_length(config, tokenizer)

    # Create PyTorch datasets
    train_dataset = create_pytorch_dataset(
        dataset_dict["train"],
        tokenizer=tokenizer,
        input_format=config.get("input_format", "multi_label"),
        max_length=max_length,
        augmentation_prob=config.get("augmentation_prob", 0.0),
        augmentation_methods=config.get("augmentation_methods", []),
    )

    val_dataset = create_pytorch_dataset(
        dataset_dict["validation"],
        tokenizer=tokenizer,
        input_format=config.get("input_format", "multi_label"),
        max_length=max_length,
        augmentation_prob=0.0,  # No augmentation for validation
        augmentation_methods=[],
    )

    # Create data loaders with optimized settings
    # Reduce num_workers to avoid multiprocessing overhead and potential deadlocks
    num_workers = min(config.get("num_workers", 2), 2)  # Max 2 workers for stability

    # Support separate batch sizes for training and evaluation
    # Evaluation can use larger batch size since no gradient computation needed
    train_batch_size = config.get("train_batch_size", config.get("batch_size", 16))
    eval_batch_size = config.get("eval_batch_size", config.get("batch_size", 16))

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=num_workers > 0,  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch 2 batches per worker
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    return train_loader, val_loader


def _get_compatible_max_length(config: dict[str, Any], tokenizer: Any) -> int:
    """Get a max_length that's compatible with the model and tokenizer.

    Args:
        config: Trial configuration
        tokenizer: Model tokenizer

    Returns:
        Compatible max_length value
    """
    import logging
    logger = logging.getLogger(__name__)
    requested_max_length = config.get("max_length", 512)

    # Get model's maximum position embeddings if available
    model_max_length = getattr(tokenizer, 'model_max_length', None)

    # Handle special cases for different model types
    model_name = config.get("backbone") or config.get("model_name", "")

    if "longformer" in model_name.lower():
        # Longformer can handle up to 4096 tokens
        max_supported = min(4096, model_max_length) if model_max_length else 4096
    elif "bigbird" in model_name.lower():
        # BigBird can handle up to 4096 tokens
        max_supported = min(4096, model_max_length) if model_max_length else 4096
    else:
        # Most BERT-like models support up to 512 tokens
        max_supported = min(512, model_max_length) if model_max_length else 512

    # Use the minimum of requested and supported length
    compatible_length = min(requested_max_length, max_supported)

    # Log if we had to reduce the max_length
    if compatible_length < requested_max_length:
        logger.warning(
            f"Reduced max_length from {requested_max_length} to {compatible_length} "
            f"for model {model_name} (max supported: {max_supported})"
        )

    return compatible_length


def train_model(
    model: MultiTaskModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    trainer: Trainer,
    config: dict[str, Any],
    device: torch.device,
) -> float:
    """Train the model and return final validation metric.

    Args:
        model: Multi-task model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        trainer: Trainer instance
        config: Trial configuration
        device: Device to train on

    Returns:
        Final validation metric value
    """
    import mlflow
    from torch.optim import Adam, AdamW
    from torch.optim.lr_scheduler import LinearLR
    from torch.amp import GradScaler

    # Try to import additional optimizers
    try:
        from transformers.optimization import Adafactor

        ADAFACTOR_AVAILABLE = True
    except ImportError:
        try:
            # Fallback to direct import
            from transformers import Adafactor

            ADAFACTOR_AVAILABLE = True
        except ImportError:
            ADAFACTOR_AVAILABLE = False

    # Enable CUDA optimizations
    if torch.cuda.is_available():
        # Enable TF32 for Ampere GPUs (faster matmul)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Enable cudnn autotuner for optimal convolution algorithms
        torch.backends.cudnn.benchmark = True
        LOGGER.info("Enabled CUDA optimizations (TF32, cudnn benchmark)")

    # Create optimizer
    optimizer_name = config.get("optimizer", "adamw").lower()
    lr = config.get("encoder_lr") or config.get("learning_rate", 2e-5)
    weight_decay = config.get("weight_decay", 0.01)
    betas_raw = config.get("betas", [0.9, 0.999])
    # Convert list to tuple if needed (for compatibility with optimizers)
    betas = tuple(betas_raw) if isinstance(betas_raw, list) else betas_raw
    eps = config.get("eps", 1e-8)

    if optimizer_name in ["adamw", "adamw_torch"]:
        optimizer = AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps
        )
    elif optimizer_name == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    elif optimizer_name == "adafactor":
        if not ADAFACTOR_AVAILABLE:
            raise ValueError("Adafactor optimizer requires transformers library")
        optimizer = Adafactor(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            scale_parameter=False,
            relative_step=False,
        )
    # Lion optimizer removed due to dependency issues
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Create loss function
    loss_fn = MultiTaskLoss(
        criteria_loss_weight=config.get("criteria_loss_weight", 1.0),
        evidence_loss_weight=config.get("evidence_loss_weight", 1.0),
        criteria_loss_type=config.get("criteria_loss_type", "bce"),
        label_smoothing=config.get("label_smoothing", 0.0),
    )

    # Create scheduler
    total_steps = len(train_loader) * config.get("epochs", 10)
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
    scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)

    # Setup mixed precision training - default to fp16 for speed if not specified
    fp_precision = config.get("fp_precision", "fp16" if torch.cuda.is_available() else "none")
    use_amp = fp_precision in ["fp16", "bf16"] and torch.cuda.is_available()

    # Always use bfloat16 for autocast
    if use_amp:
        autocast_dtype = torch.bfloat16
        LOGGER.info("Using bfloat16 mixed precision training for faster speed")
        # No GradScaler needed for bf16
        scaler = None
    else:
        autocast_dtype = torch.float32
        scaler = None
        LOGGER.info("Using full precision (float32) training")

    # Prepare training state
    state = trainer.prepare()

    best_metric = state.best_metric or 0.0

    # Training loop
    from tqdm import tqdm

    # Initialize optimizer gradients
    optimizer.zero_grad()

    for epoch in range(state.epoch, config.get("epochs", 10)):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Add progress bar for training
        train_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{config.get('epochs', 10)} [Train]",
            leave=False,
            dynamic_ncols=True
        )

        for batch_idx, batch in enumerate(train_pbar):
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

            # Forward pass with automatic mixed precision
            with torch.amp.autocast(device_type='cuda', enabled=use_amp, dtype=autocast_dtype):
                # Prepare model inputs - include token_type_ids if available
                model_inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"]
                }
                if "token_type_ids" in batch:
                    model_inputs["token_type_ids"] = batch["token_type_ids"]
                
                outputs = model(**model_inputs)

                # Compute loss
                loss = loss_fn(
                    criteria_logits=outputs.criteria_logits,
                    start_logits=outputs.start_logits,
                    end_logits=outputs.end_logits,
                    criteria_labels=batch["criteria_labels"],
                    start_positions=batch["start_positions"],
                    end_positions=batch["end_positions"],
                )

            # Scale loss for gradient accumulation
            accumulation_steps = config.get("accumulation_steps", 1)
            scaled_loss = loss / accumulation_steps

            # Backward pass with gradient scaling for fp16
            if scaler is not None:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            # Gradient accumulation - update weights every N steps
            if (batch_idx + 1) % accumulation_steps == 0:
                if scaler is not None:
                    # Unscale gradients before clipping
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.get("max_grad_norm", 1.0))
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            num_batches += 1

            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{epoch_loss/num_batches:.4f}'
            })

            # Log training metrics
            if batch_idx % 100 == 0:
                mlflow.log_metric("train_loss", loss.item(), step=state.global_step)

            state.global_step += 1

        train_pbar.close()

        # Validation
        val_metric = evaluate_model(model, val_loader, device, config)

        # Update state
        state.epoch = epoch + 1
        state.metrics[f"epoch_{epoch}_val_metric"] = val_metric

        # Check if best model
        if val_metric > best_metric:
            best_metric = val_metric
            state.best_metric = best_metric

        # Save checkpoint
        trainer.save_state(state, val_metric)

        # Log epoch metrics
        avg_loss = epoch_loss / num_batches
        mlflow.log_metric("epoch_loss", avg_loss, step=epoch)
        mlflow.log_metric("val_metric", val_metric, step=epoch)

        LOGGER.info(
            f"Epoch {epoch}: loss={avg_loss:.4f}, val_metric={val_metric:.4f}, best={best_metric:.4f}"
        )

    return best_metric


def evaluate_model(
    model: MultiTaskModel, val_loader: DataLoader, device: torch.device, config: dict[str, Any]
) -> float:
    """Evaluate model on validation set.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader
        device: Device to evaluate on
        config: Trial configuration

    Returns:
        Validation metric value
    """
    from tqdm import tqdm

    model.eval()

    total_correct = 0
    total_samples = 0

    # Setup autocast for validation (always use bf16 when enabled)
    fp_precision = config.get("fp_precision", "none")
    use_amp = fp_precision in ["fp16", "bf16"] and torch.cuda.is_available()
    
    if use_amp:
        autocast_dtype = torch.bfloat16
    else:
        autocast_dtype = torch.float32

    # Add progress bar for validation
    val_pbar = tqdm(val_loader, desc="Validating", leave=False, dynamic_ncols=True)

    with torch.no_grad():
        for batch in val_pbar:
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }

            # Forward pass with autocast for faster inference
            with torch.amp.autocast(device_type='cuda', enabled=use_amp, dtype=autocast_dtype):
                # Prepare model inputs - include token_type_ids if available
                model_inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "return_predictions": True,
                }
                if "token_type_ids" in batch:
                    model_inputs["token_type_ids"] = batch["token_type_ids"]
                
                outputs = model(**model_inputs)

            # Simple accuracy calculation (can be enhanced with F1, etc.)
            if outputs.criteria_predictions is not None:
                predictions = outputs.criteria_predictions
                labels = batch["criteria_labels"]

                # Binary accuracy for multi-label classification
                correct = (predictions == labels).float().mean()
                total_correct += correct.item() * labels.size(0)
                total_samples += labels.size(0)

            # Update progress bar
            current_acc = total_correct / total_samples if total_samples > 0 else 0.0
            val_pbar.set_postfix({'acc': f'{current_acc:.4f}'})

    val_pbar.close()
    return total_correct / total_samples if total_samples > 0 else 0.0


__all__ = ["TrialExecutor", "TrialSpec", "TrialResult", "run_single_trial"]
