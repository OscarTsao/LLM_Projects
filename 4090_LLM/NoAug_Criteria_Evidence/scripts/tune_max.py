# Maximal Optuna search for Criteria/Evidence/Joint/Share agents (NoAug).
# - Big conditional spaces (model, head, optimizer, scheduler, tokenization, loss, null-span policy, postproc)
# - Multi-fidelity pruning (Hyperband + Percentile with patience)
# - MLflow logging (file-based by default; keep state outside the repo)
#
# USAGE (examples):
#   python scripts/tune_max.py --agent criteria --study noaug-criteria-max --n-trials 800 --outdir ./_runs
#   python scripts/tune_max.py --agent evidence --study noaug-evidence-max --n-trials 1200 --timeout 10800 --parallel 4
#
# INTEGRATION: Implement `run_training_eval(cfg, callbacks)` to call your trainer once/epoch,
# reporting metrics to the provided callbacks and returning the final metric dict.

import argparse
import json
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# FIX: Protobuf/sentencepiece multiprocessing issue with DeBERTa tokenizers
# When using parallel HPO (--parallel > 1), multiple workers may try to register
# the same protobuf descriptors simultaneously, causing:
# - "Descriptors cannot be created directly" (protobuf >= 3.19.0)
# - "duplicate file name sentencepiece_model.proto" (sentencepiece)
# Setting this to 'python' forces pure-Python implementation (slower but thread-safe)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# FIX: Enable synchronous CUDA error reporting for debugging device-side asserts
# CUDA errors are asynchronous by default - they occur during kernel execution but
# only surface when a synchronizing operation happens (like model.cpu()).
# This makes debugging very difficult. Setting CUDA_LAUNCH_BLOCKING=1 makes CUDA
# operations synchronous, so errors are reported at the exact line that causes them.
# NOTE: This slows down training but is CRITICAL for debugging intermittent CUDA errors.
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# FIX: Enable CUDA Device-Side Assertions for pinpointing exact failure line
# When TORCH_USE_CUDA_DSA=1, PyTorch will show the exact CUDA kernel and line
# that triggered a device-side assert, instead of a generic error.
# This is ESSENTIAL for debugging "device-side assert triggered" errors.
os.environ.setdefault("TORCH_USE_CUDA_DSA", "1")

# FIX: Enable expandable memory segments to prevent CUDA memory fragmentation
# With 268 trials completed but CUDA asserts recurring, the error message showed:
# "1.41 GiB is reserved by PyTorch but unallocated" - this is fragmentation.
# Setting expandable_segments:True allows PyTorch to use fragmented memory efficiently.
# This prevents OOM errors when trying to allocate small amounts (16 MB) with
# plenty of memory available but fragmented into unusable chunks.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# FIX: Transformers lazy module loading race condition
# In transformers 4.57.1, the lazy module loading of OnnxConfig can fail with:
# "ImportError: cannot import name 'OnnxConfig' from 'transformers.onnx'"
# This happens when AutoTokenizer.from_pretrained triggers lazy loading of model
# configs in a multi-threaded Optuna context. We eagerly load configs at startup.

import numpy as np
import optuna
from optuna.pruners import HyperbandPruner, PatientPruner
from optuna.samplers import NSGAIISampler, TPESampler

from psy_agents_noaug.utils.storage import (
    archive_trial_summary,
    cleanup_mlflow_runs,
    download_mlflow_artifacts,
    ensure_directory,
)

try:
    import mlflow

    _HAS_MLFLOW = True
except Exception:
    mlflow = None  # type: ignore[assignment]
    _HAS_MLFLOW = False

if _HAS_MLFLOW:
    from mlflow.tracking import MlflowClient
else:  # pragma: no cover - fallback when MLflow unavailable
    MlflowClient = None  # type: ignore[assignment]


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


def parse_env_int(key: str, default: int, min_val: int = 0, max_val: int | None = None) -> int:
    """Parse and validate integer environment variable."""
    try:
        value = int(os.getenv(key, str(default)))
        if value < min_val:
            print(f"Warning: {key}={value} is below minimum {min_val}. Using default {default}")
            return default
        if max_val is not None and value > max_val:
            print(f"Warning: {key}={value} exceeds maximum {max_val}. Using default {default}")
            return default
        return value
    except ValueError:
        print(f"Warning: Invalid {key} environment variable. Using default {default}")
        return default


def set_seeds(seed: int):
    """
    Set random seeds for reproducibility with CUDA error detection.

    CRITICAL: If CUDA context is corrupted (from a previous device-side assert),
    torch.cuda.manual_seed_all() will fail. We detect this and raise a fatal
    exception to trigger process restart (supervisor will handle it).

    Args:
        seed: Random seed value

    Raises:
        RuntimeError: If CUDA context is corrupted and cannot be recovered
    """
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CUDA seeding with corruption detection
    if torch.cuda.is_available():
        try:
            torch.cuda.manual_seed_all(seed)
        except (RuntimeError, torch.cuda.CudaError) as e:
            error_msg = str(e).lower()
            is_cuda_corrupt = (
                "device-side assert" in error_msg
                or "cuda error" in error_msg
                or "accelerator" in error_msg
            )

            if is_cuda_corrupt:
                # CUDA context is corrupted - this is UNRECOVERABLE in the current process
                # We must exit and let the supervisor restart us
                print("\n" + "="*80)
                print("FATAL: CUDA CONTEXT CORRUPTED")
                print("="*80)
                print(f"Error during torch.cuda.manual_seed_all({seed}): {e}")
                print("\nThis indicates the CUDA context was corrupted by a previous")
                print("device-side assert. Once CUDA context is corrupted, ALL subsequent")
                print("CUDA operations will fail, even simple ones like setting the seed.")
                print("\nThe only solution is to restart the process to get a fresh CUDA context.")
                print("The HPO supervisor will automatically restart this process.")
                print("="*80 + "\n")

                # Clean up and raise fatal error
                try:
                    torch.cuda.empty_cache()
                except:
                    pass

                raise RuntimeError(
                    "CUDA context corrupted (device-side assert from previous trial). "
                    "Process must restart to recover. This is expected behavior - "
                    "the supervisor will handle the restart."
                ) from e
            else:
                # Some other CUDA error - re-raise
                raise

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ----------------------------
# EarlyStopping helper (patience-based)
# ----------------------------
class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.0, mode: str = "max"):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def improved(self, value: float) -> bool:
        if self.best is None:
            return True
        if self.mode == "max":
            return value > self.best + self.min_delta
        else:
            return value < self.best - self.min_delta

    def step(self, value: float) -> bool:
        if self.improved(value):
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def _trial_objective_value(trial: optuna.trial.FrozenTrial | optuna.Trial):
    """Return serialized objective value for single or multi-objective trials."""
    values = getattr(trial, "values", None)
    if values is not None:
        return list(values)
    return getattr(trial, "value", None)


def default_mlflow_setup(outdir: str):
    if not _HAS_MLFLOW:
        return
    os.makedirs(outdir, exist_ok=True)
    mlruns_dir = os.path.join(outdir, "mlruns")
    mlflow.set_tracking_uri(f"file:{mlruns_dir}")
    mlflow.set_experiment("NoAug_Criteria_Evidence")


def on_epoch(
    trial: optuna.Trial, step: int, metric: float, secondary: float | None = None
):
    trial.report(metric, step=step)
    if secondary is not None:
        trial.set_user_attr(f"secondary_epoch_{step}", float(secondary))
    if trial.should_prune():
        raise optuna.TrialPruned(f"Pruned at step {step} with metric {metric:.4f}")


# Optional narrowing via env for hybrid flows
_raw_models = os.environ.get("HPO_MODEL_CHOICES")
MODEL_CHOICES = (
    [m.strip() for m in _raw_models.split(",")]
    if _raw_models
    else [
        "bert-base-uncased",
        "bert-large-uncased",
        "roberta-base",
        "roberta-large",
        "microsoft/deberta-v3-base",
        "microsoft/deberta-v3-large",
        # ELECTRA models excluded - incompatible with CriteriaModel (no pooler_output)
        # "google/electra-base-discriminator",
        # "google/electra-large-discriminator",
        "xlm-roberta-base",
    ]
)

SCHEDULERS = ["linear", "cosine", "cosine_restart", "polynomial", "one_cycle"]
OPTIMS = ["adamw"]  # Only AdamW is currently implemented

ACTS = ["gelu", "relu", "silu"]
POOLING = ["cls", "mean", "max", "attn"]
LOSSES_CLS = ["ce", "ce_label_smooth", "focal"]
LOSSES_QA = ["qa_ce", "qa_ce_ls", "qa_focal"]

NULL_POLICIES = ["none", "threshold", "ratio", "calibrated"]
RERANKERS = ["sum", "product", "softmax"]

# Optional head narrowing via env JSON (for hybrid trust-region)
_HEAD_LIMITS = json.loads(os.environ.get("HPO_HEAD_LIMITS_JSON", "{}"))


def suggest_common(trial: optuna.Trial, heavy_model: bool) -> dict[str, Any]:
    # Model-aware sequence length constraints for 24GB GPU
    # Large models (330M-400M params): use shorter sequences to reduce memory
    # Base models (110M params): can use full 512 tokens
    if heavy_model:
        max_len = trial.suggest_int("tok.max_length", 128, 384, step=32)
    else:
        max_len = trial.suggest_int("tok.max_length", 128, 512, step=32)

    stride = trial.suggest_int("tok.doc_stride", 32, min(128, max_len // 2), step=16)
    fast_tok = trial.suggest_categorical("tok.use_fast", [True, False])

    # Batch size and gradient accumulation sampling
    # NOTE: Optuna's CategoricalDistribution does NOT support dynamic value spaces.
    # All trials in a study must use the SAME distribution for each parameter.
    # Therefore, we sample from the UNION of all possible values and validate later.
    #
    # Model-aware OOM prevention (24GB GPU):
    # Large models (bert-large, roberta-large, deberta-v3-large): 330M-400M params
    # - Safe combinations: bsz <= 16, bsz * accum <= 64
    # Base models (bert-base, roberta-base, deberta-v3-base): 110M params
    # - Safe combinations: bsz <= 64, bsz * accum <= 512

    bsz = trial.suggest_categorical("train.batch_size", [8, 12, 16, 24, 32, 48, 64])
    accum = trial.suggest_categorical("train.grad_accum", [1, 2, 3, 4, 6, 8])

    # Validate OOM safety and prune unsafe combinations
    effective_batch = bsz * accum
    if heavy_model:
        # Large models: conservative limits
        if bsz > 16 or effective_batch > 64:
            raise optuna.TrialPruned(
                f"Pruned: Large model with bsz={bsz}, accum={accum} "
                f"(effective_batch={effective_batch}) likely causes OOM (24GB GPU limit)"
            )
    else:
        # Base models: more relaxed
        if bsz > 64 or effective_batch > 512:
            raise optuna.TrialPruned(
                f"Pruned: Base model with bsz={bsz}, accum={accum} "
                f"(effective_batch={effective_batch}) exceeds reasonable limits"
            )

    optim = trial.suggest_categorical("optim.name", OPTIMS)
    lr_hi = 1.5e-4 if heavy_model else 3e-4
    lr = trial.suggest_float("optim.lr", 5e-6, lr_hi, log=True)
    wd = trial.suggest_float("optim.weight_decay", 1e-6, 2e-1, log=True)

    # AdamW parameters (only optimizer currently supported)
    b1 = trial.suggest_float("optim.beta1", 0.80, 0.95)
    b2 = trial.suggest_float("optim.beta2", 0.95, 0.9999)
    eps = trial.suggest_float("optim.eps", 1e-9, 1e-6, log=True)

    sched = trial.suggest_categorical("sched.name", SCHEDULERS)
    warmup = trial.suggest_float("sched.warmup_ratio", 0.0, 0.2)
    cos_cycles = (
        trial.suggest_int("sched.cosine_cycles", 1, 4)
        if sched in ("cosine_restart", "one_cycle")
        else None
    )
    poly_power = (
        trial.suggest_float("sched.poly_power", 0.5, 2.0)
        if sched == "polynomial"
        else None
    )

    clip = trial.suggest_float("train.clip_grad", 0.0, 1.5)
    dropout = trial.suggest_float("model.dropout", 0.0, 0.5)
    attn_drop = trial.suggest_float("model.attn_dropout", 0.0, 0.3)
    grad_ckpt = trial.suggest_categorical("train.grad_checkpointing", [False, True])
    freeze_layers = trial.suggest_int("train.freeze_encoder_layers", 0, 6)

    lld = trial.suggest_float("optim.layerwise_lr_decay", 0.80, 1.00)

    return {
        "tok": {"max_length": max_len, "doc_stride": stride, "use_fast": fast_tok},
        "train": {
            "batch_size": bsz,
            "grad_accum": accum,
            "epochs": None,
            "clip_grad": clip,
            "grad_checkpointing": grad_ckpt,
            "freeze_encoder_layers": freeze_layers,
        },
        "optim": {
            "name": optim,
            "lr": lr,
            "weight_decay": wd,
            "beta1": b1,
            "beta2": b2,
            "eps": eps,
            "layerwise_lr_decay": lld,
        },
        "sched": {
            "name": sched,
            "warmup_ratio": warmup,
            "cosine_cycles": cos_cycles,
            "poly_power": poly_power,
        },
        "regularization": {"dropout": dropout, "attn_dropout": attn_drop},
    }


def _is_heavy_model(model_name: str) -> bool:
    """Return True for models that require conservative batching."""
    heavy_keywords = ("-large", "large", "xlm-roberta", "deberta")
    return any(keyword in model_name.lower() for keyword in heavy_keywords)


def suggest_criteria(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    heavy = _is_heavy_model(model_name)
    com = suggest_common(trial, heavy)
    pooling = trial.suggest_categorical("head.pooling", POOLING)
    head_layers = trial.suggest_int(
        "head.layers",
        int(_HEAD_LIMITS.get("layers_min", 1)),
        int(_HEAD_LIMITS.get("layers_max", 4)),
    )
    head_hidden = trial.suggest_categorical(
        "head.hidden",
        _HEAD_LIMITS.get("hidden_choices", [256, 384, 512, 768, 1024, 1536, 2048]),
    )
    head_act = trial.suggest_categorical("head.activation", ACTS)
    head_do = trial.suggest_float(
        "head.dropout", 0.0, float(_HEAD_LIMITS.get("dropout_max", 0.5))
    )
    loss = trial.suggest_categorical("loss.cls.type", LOSSES_CLS)
    label_smooth = (
        trial.suggest_float("loss.cls.label_smoothing", 0.0, 0.20)
        if loss != "focal"
        else 0.0
    )
    focal_gamma = (
        trial.suggest_float("loss.cls.gamma", 1.0, 5.0) if loss == "focal" else None
    )
    focal_alpha = (
        trial.suggest_float("loss.cls.alpha", 0.1, 0.9) if loss == "focal" else None
    )
    class_balance = trial.suggest_categorical(
        "loss.cls.balance", ["none", "weighted", "effective_num"]
    )
    epochs = parse_env_int("HPO_EPOCHS", 100, min_val=1, max_val=1000)
    return {
        "task": "criteria",
        "model": {"name": model_name},
        "head": {
            "pooling": pooling,
            "layers": head_layers,
            "hidden": head_hidden,
            "activation": head_act,
            "dropout": head_do,
        },
        "loss": {
            "type": loss,
            "label_smoothing": label_smooth,
            "gamma": focal_gamma,
            "alpha": focal_alpha,
            "balance": class_balance,
        },
        **com,
        "train": {**com["train"], "epochs": epochs},
    }


def suggest_evidence(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    heavy = _is_heavy_model(model_name)
    com = suggest_common(trial, heavy)
    head_layers = trial.suggest_int(
        "head.layers",
        int(_HEAD_LIMITS.get("layers_min", 1)),
        int(_HEAD_LIMITS.get("layers_max", 4)),
    )
    head_hidden = trial.suggest_categorical(
        "head.hidden",
        _HEAD_LIMITS.get("hidden_choices", [256, 384, 512, 768, 1024, 1536, 2048]),
    )
    head_act = trial.suggest_categorical("head.activation", ACTS)
    head_do = trial.suggest_float(
        "head.dropout", 0.0, float(_HEAD_LIMITS.get("dropout_max", 0.5))
    )
    loss = trial.suggest_categorical("loss.qa.type", LOSSES_QA)
    label_smooth = (
        trial.suggest_float("loss.qa.label_smoothing", 0.0, 0.15)
        if loss != "qa_focal"
        else 0.0
    )
    focal_gamma = (
        trial.suggest_float("loss.qa.gamma", 1.0, 5.0) if loss == "qa_focal" else None
    )
    focal_alpha = (
        trial.suggest_float("loss.qa.alpha", 0.1, 0.9) if loss == "qa_focal" else None
    )
    null_pol = trial.suggest_categorical("qa.null.policy", NULL_POLICIES)
    null_threshold = (
        trial.suggest_float("qa.null.threshold", -5.0, 5.0)
        if null_pol in ("threshold", "calibrated")
        else None
    )
    null_ratio = (
        trial.suggest_float("qa.null.ratio", 0.05, 0.8) if null_pol == "ratio" else None
    )
    topk = trial.suggest_int("qa.topk", 1, 20)
    max_ans = trial.suggest_int("qa.max_answer_len", 20, 512, step=4)
    nbest = trial.suggest_int("qa.n_best_size", 10, 50)
    rerank = trial.suggest_categorical("qa.reranker", RERANKERS)
    nms_iou = trial.suggest_float("qa.nms_iou", 0.3, 0.8)
    neg_ratio = trial.suggest_float("qa.neg_ratio", 0.1, 1.0)
    epochs = parse_env_int("HPO_EPOCHS", 100, min_val=1, max_val=1000)
    return {
        "task": "evidence",
        "model": {"name": model_name},
        "head": {
            "layers": head_layers,
            "hidden": head_hidden,
            "activation": head_act,
            "dropout": head_do,
        },
        "loss": {
            "type": loss,
            "label_smoothing": label_smooth,
            "gamma": focal_gamma,
            "alpha": focal_alpha,
        },
        "qa": {
            "null": {
                "policy": null_pol,
                "threshold": null_threshold,
                "ratio": null_ratio,
            },
            "topk": topk,
            "max_answer_len": max_ans,
            "n_best_size": nbest,
            "reranker": rerank,
            "nms_iou": nms_iou,
            "neg_ratio": neg_ratio,
        },
        **com,
        "train": {**com["train"], "epochs": epochs},
    }


def suggest_joint(trial: optuna.Trial, model_name: str) -> dict[str, Any]:
    cfg_e = suggest_evidence(trial, model_name)
    cfg_c = suggest_criteria(trial, model_name)
    share_ratio = trial.suggest_float("joint.share_ratio", 0.0, 1.0)
    multi_task_weight = trial.suggest_float("joint.criteria_weight", 0.2, 0.8)
    cfg = {
        "task": "joint",
        "model": {"name": model_name},
        "shared": {"ratio": share_ratio, "criteria_weight": multi_task_weight},
        "criteria": cfg_c,
        "evidence": cfg_e,
        "tok": cfg_e["tok"],
        "optim": cfg_e["optim"],
        "sched": cfg_e["sched"],
        "regularization": cfg_e["regularization"],
        "train": cfg_e["train"],
    }
    return cfg


def build_config(trial: optuna.Trial, agent: str) -> dict[str, Any]:
    model = trial.suggest_categorical("model.name", MODEL_CHOICES)
    if agent == "criteria":
        return suggest_criteria(trial, model)
    if agent == "evidence":
        return suggest_evidence(trial, model)
    if agent == "joint":
        return suggest_joint(trial, model)
    if agent == "share":
        # For "share", treat similarly to joint with different defaults; reuse joint for now.
        return suggest_joint(trial, model)
    raise ValueError(agent)


def run_training_eval(
    cfg: dict[str, Any],
    callbacks: dict[str, Callable[[int, float, float | None], None]],
    trial_number: int = -1,
) -> dict[str, float]:
    """
    Training bridge for HPO integration with REAL redsm5 data and EarlyStopping.

    Loads real redsm5 dataset, trains the model with EarlyStopping, and reports metrics.

    Args:
        cfg: Configuration dict with model, head, train, optim, etc.
        callbacks: Dict with "on_epoch" callback for reporting metrics
        trial_number: Trial number for logging (default: -1)

    Returns:
        Dict with "primary" metric and "runtime_s"
    """
    import gc
    import sys
    from pathlib import Path

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, random_split
    from transformers import AutoTokenizer

    # Log trial configuration for debugging
    print(f"\n{'='*80}")
    print(f"TRIAL {trial_number} - Configuration:")
    print(f"  Model: {cfg.get('model', 'UNKNOWN')}")
    print(f"  Batch size: {cfg.get('train', {}).get('batch_size', 'UNKNOWN')}")
    print(f"  Learning rate: {cfg.get('optim', {}).get('lr', 'UNKNOWN')}")
    print(f"  Dropout: {cfg.get('regularization', {}).get('dropout', 'UNKNOWN')}")
    print(f"{'='*80}\n")

    # Add src to path
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from Project.Criteria.data.dataset import CriteriaDataset
    from Project.Criteria.models.model import Model as CriteriaModel
    from Project.Evidence.data.dataset import EvidenceDataset
    from Project.Evidence.models.model import Model as EvidenceModel

    epochs = cfg["train"]["epochs"]
    batch_size = cfg["train"]["batch_size"]
    task = cfg.get("task", "criteria")
    model_name = cfg["model"]["name"]

    # EarlyStopping config from environment
    patience = parse_env_int("HPO_PATIENCE", 20, min_val=1, max_val=200)
    min_delta = float(os.getenv("HPO_MIN_DELTA", "0.0"))
    es = EarlyStopping(patience=patience, min_delta=min_delta, mode="max")

    # Detect device and verify CUDA health
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # DEFENSIVE: Verify CUDA is healthy before starting training
    # This catches residual corruption from previous trials
    if device.type == "cuda" and not check_cuda_health():
        raise RuntimeError(
            f"Trial {trial_number}: CUDA health check failed at training start. "
            "CUDA context is corrupted. Process must restart."
        )

    # Initialize cleanup variables for exception handling
    model = None
    optimizer = None
    train_loader = None
    val_loader = None

    # Load real dataset based on task
    project_root = Path(__file__).parent.parent
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if task in ("criteria", "share", "joint"):
        dataset_path = project_root / "data" / "redsm5" / "redsm5_annotations.csv"
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {dataset_path}\n"
                f"Task: {task}\n"
                f"Please ensure data is generated with: make groundtruth"
            )
        dataset = CriteriaDataset(
            csv_path=dataset_path,
            tokenizer=tokenizer,
            max_length=cfg["tok"]["max_length"],
        )
        num_labels = 2
    elif task == "evidence":
        dataset_path = (
            project_root / "data" / "processed" / "redsm5_matched_evidence.csv"
        )
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {dataset_path}\n"
                f"Task: {task}\n"
                f"Please ensure data is generated with: make groundtruth"
            )
        dataset = EvidenceDataset(
            csv_path=dataset_path,
            tokenizer=tokenizer,
            max_length=cfg["tok"]["max_length"],
        )
        num_labels = 2
    else:
        raise ValueError(
            f"Unknown task: {task}. Supported tasks: criteria, evidence, share, joint.\n"
            f"Check cfg['task'] value or --agent argument."
        )

    # Split dataset (80/10/10) - train/val/test
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    seed = cfg.get("meta", {}).get("seed", 42)
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )

    # OPTIMIZATION (2025-10-24): Increased DataLoader workers to improve GPU utilization
    # Previous: num_workers=2 (conservative for parallel HPO)
    # Current: num_workers=4 to better feed GPU and reduce idle time
    # With --parallel 4, total workers = 4*4 = 16, but with persistent_workers,
    # actual CPU usage is manageable on 12-core system
    num_workers = 4

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive between epochs
        prefetch_factor=3 if num_workers > 0 else None,  # Prefetch 3 batches per worker
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=True if num_workers > 0 else False,  # Keep workers alive
        prefetch_factor=3 if num_workers > 0 else None,  # Prefetch 3 batches per worker
    )

    # Helper function to safely move model to device (handles meta device)
    def safe_to_device(model: nn.Module, target_device: torch.device) -> nn.Module:
        """
        Safely move model to target device, handling meta device edge case.

        In parallel HPO with certain configurations, models may be initialized on
        the meta device (virtual device with no data). This requires to_empty()
        instead of to() for device transfer.

        CRITICAL FIX: Check ALL parameters, not just first one. In nested models
        like CriteriaModel, the classifier head (CPU) comes first, but the encoder
        (potentially meta device) comes second. We must check all parameters.
        """
        try:
            # Check if ANY parameter is on meta device (not just first one!)
            has_meta_params = False
            for param in model.parameters():
                if param.device.type == 'meta':
                    has_meta_params = True
                    break

            if has_meta_params:
                # Use to_empty() for meta device transfer
                model = model.to_empty(device=target_device)
                # Reinitialize parameters after transfer
                def init_weights(module):
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()
                model.apply(init_weights)
            else:
                # Normal device transfer
                model = model.to(target_device)
        except StopIteration:
            # Model has no parameters, just move it
            model = model.to(target_device)
        return model

    # Create model based on task
    if task in ("criteria", "share", "joint"):
        model = CriteriaModel(
            model_name=model_name,
            num_labels=num_labels,
            classifier_dropout=cfg["regularization"]["dropout"],
        )
        model = safe_to_device(model, device)
    elif task == "evidence":
        model = EvidenceModel(
            model_name=model_name,
            dropout_prob=cfg["regularization"]["dropout"],
        )
        model = safe_to_device(model, device)

    criterion = nn.CrossEntropyLoss()

    # Create optimizer - only AdamW is supported
    lr = cfg["optim"]["lr"]
    wd = cfg["optim"]["weight_decay"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=wd,
        betas=(cfg["optim"].get("beta1", 0.9), cfg["optim"].get("beta2", 0.999)),
        eps=cfg["optim"].get("eps", 1e-8),
    )

    # Training loop with EarlyStopping
    start = time.time()
    best = 0.0

    try:
        for epoch in range(epochs):
            # Training
            model.train()
            total_loss = 0.0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                # DEFENSIVE: Validate labels are in valid range for CrossEntropyLoss
                # CrossEntropyLoss expects labels in [0, num_classes-1]. For binary
                # classification (num_labels=2), this is [0, 1].
                # CUDA device-side asserts occur when labels are out of range.
                #
                # IMPORTANT: This validation happens AFTER tensors are moved to GPU,
                # so invalid labels will have already triggered the device-side assert.
                # The assert is asynchronous, so we might not see it until later.
                # With TORCH_USE_CUDA_DSA=1, we'll see the exact line that caused it.
                label_min = labels.min().item()
                label_max = labels.max().item()

                if label_min < 0 or label_max >= num_labels:
                    # Synchronize to catch any pending CUDA errors
                    if device.type == "cuda":
                        try:
                            torch.cuda.synchronize()
                        except Exception as cuda_err:
                            raise RuntimeError(
                                f"CUDA error during label validation! "
                                f"Invalid labels (min={label_min}, max={label_max}) "
                                f"triggered device-side assert when moved to GPU. "
                                f"Expected range [0, {num_labels-1}]."
                            ) from cuda_err

                    raise ValueError(
                        f"Invalid labels detected in batch! "
                        f"Expected range [0, {num_labels-1}], "
                        f"but got min={label_min}, max={label_max}. "
                        f"Batch labels: {labels.cpu().tolist()[:10]}... "
                        f"Label dtype: {labels.dtype}, shape: {labels.shape}"
                    )

                optimizer.zero_grad()
                logits = model(input_ids, attention_mask)

                # DEFENSIVE: Validate logits shape matches expected output
                expected_shape = (labels.size(0), num_labels)
                if logits.shape != expected_shape:
                    raise ValueError(
                        f"Model output shape mismatch! "
                        f"Expected {expected_shape}, got {logits.shape}"
                    )

                # DEFENSIVE: Check for NaN/Inf in logits (can cause CUDA asserts)
                if torch.isnan(logits).any():
                    raise ValueError(
                        f"NaN detected in model logits! "
                        f"This indicates unstable training (likely too high learning rate). "
                        f"Logits shape: {logits.shape}, "
                        f"NaN count: {torch.isnan(logits).sum().item()}"
                    )
                if torch.isinf(logits).any():
                    raise ValueError(
                        f"Inf detected in model logits! "
                        f"This indicates numerical overflow. "
                        f"Logits shape: {logits.shape}, "
                        f"Inf count: {torch.isinf(logits).sum().item()}"
                    )

                loss = criterion(logits, labels)

                # DEFENSIVE: Check for NaN/Inf in loss (can cause CUDA asserts)
                if torch.isnan(loss):
                    raise ValueError(
                        f"NaN detected in loss! "
                        f"This indicates unstable training. "
                        f"Loss value: {loss.item()}"
                    )
                if torch.isinf(loss):
                    raise ValueError(
                        f"Inf detected in loss! "
                        f"This indicates numerical overflow. "
                        f"Loss value: {loss.item()}"
                    )
                loss.backward()

                if cfg["train"].get("clip_grad", 0.0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg["train"]["clip_grad"]
                    )

                optimizer.step()
                total_loss += loss.item()

            # Validation
            model.eval()
            all_preds = []
            all_labels = []
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    # DEFENSIVE: Validate validation labels too
                    if labels.min() < 0 or labels.max() >= num_labels:
                        raise ValueError(
                            f"Invalid labels in validation batch! "
                            f"Expected range [0, {num_labels-1}], "
                            f"but got min={labels.min().item()}, max={labels.max().item()}"
                        )

                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    val_loss += loss.item()

                    preds = logits.argmax(dim=1)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(labels.cpu().tolist())

            # Calculate metrics
            from sklearn.metrics import f1_score

            avg_val_loss = val_loss / len(val_loader)
            val_f1 = f1_score(all_labels, all_preds, average="macro")

            # Use F1 as primary metric
            metric = val_f1
            best = max(best, metric)

            # Report to Optuna
            callbacks["on_epoch"](epoch, metric, avg_val_loss)

            # Two-level stopping strategy:
            # 1. EarlyStopping (patience from HPO_PATIENCE env): Prevents overfitting within a trial
            # 2. Optuna PatientPruner (patience=2): Cross-trial comparison to stop unpromising trials early
            if es.step(metric):
                stopped_epoch = epoch + 1
                print(f"EarlyStopping triggered at epoch {stopped_epoch} (patience={patience})")
                break

        runtime = time.time() - start
        return {"primary": float(best), "runtime_s": runtime}

    finally:
        # CRITICAL: Explicit cleanup to prevent CUDA state corruption on OOM
        # Delete GPU resources BEFORE exception propagates to ensure clean state
        # for next trial. Without this, residual GPU allocations cause kernel errors.

        # FIX: Wrap cleanup in try-except to prevent crashes during error handling
        # If a CUDA error occurred during training, model.cpu() may trigger the
        # asynchronous error. We catch this to prevent killing the entire HPO process.
        try:
            if model is not None:
                # Synchronize CUDA before moving to CPU to catch any pending errors
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                model.cpu()  # Move model to CPU first
                del model
        except Exception as cleanup_error:
            print(f"Warning: Error during model cleanup: {cleanup_error}")
            # Force delete even if cleanup failed
            if 'model' in locals():
                del model

        try:
            if optimizer is not None:
                del optimizer
            if train_loader is not None:
                del train_loader
            if val_loader is not None:
                del val_loader
        except Exception as cleanup_error:
            print(f"Warning: Error during resource cleanup: {cleanup_error}")

        # CRITICAL FIX: Aggressive CUDA cache clearing to prevent memory fragmentation
        # After 268 trials, error showed: "1.41 GiB reserved but unallocated" (fragmentation)
        # We must clear cache between trials to prevent cumulative fragmentation
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
            # Clear again after GC to ensure all freed memory is returned to pool
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as cleanup_error:
            print(f"Warning: Error during CUDA cleanup: {cleanup_error}")


def make_pruner() -> optuna.pruners.BasePruner:
    # OPTIMIZATION (2025-10-24): Reduced pruning aggressiveness to improve GPU utilization
    # Previous: min_resource=1, patience=2 (86.5% pruning rate, trials pruned at epoch 2-4)
    # Current: min_resource=3, patience=4 (allow trials to train longer before pruning)
    # Rationale: With 86.5% pruning rate, GPU spends more time on trial startup (30-60s)
    # than training (2-4 min). Longer trials = better GPU utilization.
    hb = HyperbandPruner(
        min_resource=3,  # Wait at least 3 epochs before considering pruning
        max_resource=parse_env_int("HPO_EPOCHS", 100, min_val=1, max_val=1000),
        reduction_factor=3,
    )
    return PatientPruner(hb, patience=4)  # More patient (was 2)


def make_sampler(multi_objective: bool, seed: int) -> optuna.samplers.BaseSampler:
    if multi_objective:
        return NSGAIISampler(seed=seed)
    return TPESampler(seed=seed, multivariate=True, group=True, constant_liar=True)


def check_cuda_health() -> bool:
    """
    Check if CUDA context is healthy by attempting a simple operation.

    Returns:
        True if CUDA is healthy, False if corrupted
    """
    import torch

    if not torch.cuda.is_available():
        return True  # CPU mode, no CUDA to check

    try:
        # Try a simple CUDA operation
        _ = torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
        return True
    except Exception:
        return False


def objective_builder(
    agent: str, outdir: str, multi_objective: bool
) -> Callable[[optuna.Trial], float]:
    # Track consecutive CUDA failures for corruption detection
    consecutive_cuda_failures = [0]  # Use list for closure mutability
    successful_trials = [0]  # Track successful trials for periodic GPU reset

    def _obj(trial: optuna.Trial):
        import torch

        # CRITICAL: Periodic GPU reset every 50 successful trials to prevent cumulative fragmentation
        # Even with expandable_segments, long HPO runs can accumulate fragmentation
        # A full reset every 50 trials ensures clean slate and prevents context corruption
        if successful_trials[0] > 0 and successful_trials[0] % 50 == 0:
            print(f"\n{'='*80}")
            print(f"[GPU RESET] Performing periodic GPU reset after {successful_trials[0]} successful trials")
            print(f"This prevents cumulative memory fragmentation during long HPO runs")
            print(f"{'='*80}\n")
            try:
                import gc
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
                print(f"[GPU RESET] Complete. Continuing HPO...")
            except Exception as e:
                print(f"[GPU RESET] Warning: Reset encountered error (non-fatal): {e}")

        # CRITICAL: Check CUDA health BEFORE attempting to set seeds
        # If CUDA is corrupted from a previous trial, we need to detect it early
        if not check_cuda_health():
            print(f"\n[FATAL] Trial {trial.number}: CUDA context check failed before seeding")
            print("This indicates residual corruption from a previous trial.")
            print("Raising fatal error to trigger process restart...")
            raise RuntimeError(
                "CUDA context corrupted before trial start. Process must restart. "
                "This is expected behavior - the supervisor will handle it."
            )

        seed = trial.suggest_int("seed", 1, 65535)

        # Set seeds with corruption detection
        try:
            set_seeds(seed)
        except RuntimeError as e:
            # CUDA context corruption detected during seeding
            # This is a FATAL error - we cannot continue
            if "CUDA context corrupted" in str(e):
                print(f"\n[FATAL] Trial {trial.number}: CUDA corruption detected during seeding")
                raise  # Re-raise to kill the process
            else:
                # Some other error during seeding
                raise

        cfg = build_config(trial, agent)
        cfg["meta"] = {
            "agent": agent,
            "seed": seed,
            "outdir": outdir,
            "repo": "NoAug_Criteria_Evidence",
            "aug": False,
        }
        trial.set_user_attr("seed", seed)

        active_run = None
        if _HAS_MLFLOW:
            active_run = mlflow.start_run(nested=True)
            mlflow.log_params(
                {k: v for k, v in flatten_dict(cfg).items() if is_loggable(v)}
            )
            trial.set_user_attr("mlflow_run_id", active_run.info.run_id)

        def _cb(epoch, primary, secondary=None):
            on_epoch(trial, epoch, primary, secondary)

        try:
            res = run_training_eval(cfg, {"on_epoch": _cb}, trial_number=trial.number)
        except torch.cuda.OutOfMemoryError as oom_error:
            # Handle CUDA OOM: clean up and prune trial
            if _HAS_MLFLOW:
                mlflow.log_params({"oom_error": True})
                mlflow.end_run(status="FAILED")

            # Log configuration that caused OOM
            model_name = cfg["model"]["name"]
            batch_size = cfg["train"]["batch_size"]
            grad_accum = cfg["train"]["grad_accum"]
            max_len = cfg["tok"]["max_length"]
            effective_bs = batch_size * grad_accum

            print(
                f"\n[OOM] Trial {trial.number} exceeded GPU memory:\n"
                f"  Model: {model_name}\n"
                f"  Batch size: {batch_size} (effective: {effective_bs} with grad_accum={grad_accum})\n"
                f"  Max length: {max_len}\n"
                f"  Error: {str(oom_error)[:200]}\n"
                f"  Pruning trial to allow Optuna to learn memory constraints.\n"
            )

            # CRITICAL FIX: Aggressive CUDA cleanup to prevent kernel assertion errors
            # After OOM, CUDA can be in a corrupted state with residual allocations.
            # We must explicitly clear cache, synchronize, and trigger garbage collection
            # BEFORE starting the next trial to prevent "index out of bounds" errors.
            import gc
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all CUDA operations to complete
            gc.collect()  # Force garbage collection to delete model/optimizer/tensors
            torch.cuda.empty_cache()  # Clear cache again after GC

            # Mark as OOM and prune
            trial.set_user_attr("oom", True)
            trial.set_user_attr("oom_config", {
                "model": model_name,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "max_length": max_len,
                "effective_batch_size": effective_bs,
            })

            # Prune the trial - Optuna will learn to avoid similar configurations
            raise optuna.TrialPruned(f"OOM: {model_name} bs={batch_size} len={max_len}")

        except Exception as error:
            # Handle all other exceptions, with special handling for CUDA errors
            # torch.AcceleratorError is a subclass of Exception, not RuntimeError
            error_msg = str(error).lower()
            error_type = type(error).__name__.lower()
            is_cuda_error = (
                "cuda" in error_msg
                or "device-side assert" in error_msg
                or "accelerator" in error_type
                or ("device" in error_msg and "assert" in error_msg)
            )

            if is_cuda_error:
                # Track consecutive CUDA failures
                consecutive_cuda_failures[0] += 1

                if _HAS_MLFLOW:
                    mlflow.log_params({"cuda_error": True})
                    mlflow.end_run(status="FAILED")

                print(
                    f"\n[CUDA ERROR] Trial {trial.number} encountered CUDA error "
                    f"(consecutive failures: {consecutive_cuda_failures[0]}):\n"
                    f"  Error Type: {type(error).__name__}\n"
                    f"  Model: {cfg['model']['name']}\n"
                    f"  Batch size: {cfg['train']['batch_size']}\n"
                    f"  Learning rate: {cfg['optim']['lr']}\n"
                    f"  Dropout: {cfg['regularization']['dropout']}\n"
                    f"  Error: {str(error)[:300]}\n"
                )

                # CRITICAL: If we have multiple consecutive CUDA failures,
                # the CUDA context is likely corrupted and we need to restart
                if consecutive_cuda_failures[0] >= 3:
                    print(
                        f"\n{'='*80}\n"
                        f"FATAL: {consecutive_cuda_failures[0]} consecutive CUDA failures detected!\n"
                        f"This indicates CUDA context corruption that cannot be recovered.\n"
                        f"Raising fatal error to trigger process restart...\n"
                        f"{'='*80}\n"
                    )
                    # Don't clean up - just die immediately
                    raise RuntimeError(
                        f"CUDA context corrupted after {consecutive_cuda_failures[0]} "
                        "consecutive failures. Process must restart."
                    ) from error

                print(f"  This trial will be pruned. HPO will continue.\n")

                # Aggressive cleanup
                import gc
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except:
                    pass
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except:
                    pass

                # Check CUDA health after cleanup
                if not check_cuda_health():
                    print(
                        f"\n{'='*80}\n"
                        f"FATAL: CUDA health check failed after error cleanup!\n"
                        f"CUDA context is corrupted and cannot be recovered.\n"
                        f"Raising fatal error to trigger process restart...\n"
                        f"{'='*80}\n"
                    )
                    raise RuntimeError(
                        "CUDA context corrupted (health check failed after cleanup). "
                        "Process must restart."
                    ) from error

                # Mark trial and prune (don't kill entire process)
                trial.set_user_attr("cuda_error", True)
                trial.set_user_attr("cuda_error_type", type(error).__name__)
                trial.set_user_attr("cuda_error_msg", str(error)[:500])
                raise optuna.TrialPruned(f"CUDA error ({type(error).__name__}): {str(error)[:100]}")
            else:
                # Not a CUDA error, re-raise to let Optuna handle it
                if _HAS_MLFLOW:
                    mlflow.end_run(status="FAILED")
                raise

        # Trial succeeded - reset consecutive CUDA failure counter and increment success counter
        consecutive_cuda_failures[0] = 0
        successful_trials[0] += 1

        trial.set_user_attr("runtime_s", res.get("runtime_s"))
        trial.set_user_attr("primary", res["primary"])

        if _HAS_MLFLOW:
            mlflow.log_metrics(
                {
                    "final_primary": res["primary"],
                    "runtime_s": res.get("runtime_s", float("nan")),
                }
            )
            mlflow.end_run(status="FINISHED")

        return res["primary"]

    return _obj


def flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def is_loggable(v: Any) -> bool:
    return isinstance(v, str | int | float | bool) or v is None


def validate_configuration(args):
    """Validate configuration before starting expensive HPO."""
    errors = []
    project_root = Path(__file__).parent.parent

    # Check dataset exists
    if args.agent in ("criteria", "share", "joint"):
        path = project_root / "data" / "redsm5" / "redsm5_annotations.csv"
        if not path.exists():
            errors.append(f"Criteria dataset not found: {path}")

    if args.agent == "evidence":
        path = project_root / "data" / "processed" / "redsm5_matched_evidence.csv"
        if not path.exists():
            errors.append(f"Evidence dataset not found: {path}")

    # Check output directory writable
    try:
        os.makedirs(args.outdir, exist_ok=True)
        test_file = Path(args.outdir) / ".write_test"
        test_file.touch()
        test_file.unlink()
    except OSError as e:
        errors.append(f"Cannot write to output directory {args.outdir}: {e}")

    # Validate environment variables
    epochs = parse_env_int("HPO_EPOCHS", 100, min_val=1, max_val=1000)
    if epochs < 1 or epochs > 1000:
        errors.append(f"HPO_EPOCHS={epochs} out of range [1, 1000]")

    patience = parse_env_int("HPO_PATIENCE", 20, min_val=1, max_val=200)
    if patience < 1 or patience > 200:
        errors.append(f"HPO_PATIENCE={patience} out of range [1, 200]")

    if errors:
        raise ValueError("Configuration validation failed:\n  - " + "\n  - ".join(errors))

    print(f"✓ Configuration validation passed")
    print(f"  Agent: {args.agent}")
    print(f"  Epochs: {epochs} | Patience: {patience}")
    print(f"  Output: {args.outdir}")


def finalize_best_trials(
    study: optuna.study.Study,
    agent: str,
    study_name: str,
    tracking_uri: str | None,
    top_limit: int = 8,
) -> list[dict[str, Any]]:
    """
    Persist best trial information into artifacts/ and prune non-winning runs.

    Returns:
        List of dictionaries describing the preserved best trials (for JSON export).
    """
    if not study.best_trials:
        return []

    top_trials = study.best_trials[: top_limit or len(study.best_trials)]
    artifact_root = ensure_directory(Path("artifacts") / "hpo" / agent / study_name)

    summary: list[dict[str, Any]] = []
    preserve_run_ids: set[str] = set()
    client = MlflowClient() if (_HAS_MLFLOW and MlflowClient is not None) else None

    for rank, trial in enumerate(top_trials, start=1):
        run_id = trial.user_attrs.get("mlflow_run_id")
        if run_id:
            preserve_run_ids.add(run_id)

        metrics_payload = {
            "rank": rank,
            "objective": _trial_objective_value(trial),
            "runtime_s": trial.user_attrs.get("runtime_s"),
        }

        attrs = {}
        for key, value in trial.user_attrs.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                attrs[key] = value
            else:
                attrs[key] = str(value)

        trial_dir = artifact_root / f"trial_{trial.number:04d}"
        archive_trial_summary(trial_dir, trial.number, trial.params, metrics_payload, attrs)

        if client and run_id:
            download_mlflow_artifacts(client, run_id, trial_dir / "mlflow_artifacts")

        summary.append(
            {
                "rank": rank,
                "trial_number": trial.number,
                "value": metrics_payload["objective"],
                "params": trial.params,
                "mlflow_run_id": run_id,
            }
        )

    with (artifact_root / "topk_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    if _HAS_MLFLOW:
        all_run_ids = [
            rid
            for rid in (
                trial.user_attrs.get("mlflow_run_id") for trial in study.trials
            )
            if rid
        ]
        cleanup_mlflow_runs(
            all_run_ids,
            tracking_uri=tracking_uri,
            preserve=preserve_run_ids,
        )

    return summary


def check_and_handle_incompatible_study(
    study_name: str, storage: str, expected_model_choices: list[str]
) -> bool:
    """
    Check if an existing study has incompatible search space and handle it.

    Validates:
    - model.name choices
    - train.batch_size choices (all possible values across model types)
    - train.grad_accum choices (all possible values across model types)

    Args:
        study_name: Name of the study to check
        storage: Optuna storage URL
        expected_model_choices: Current MODEL_CHOICES list

    Returns:
        True if study was deleted/renamed, False otherwise
    """
    try:
        # Try to load existing study
        existing_study = optuna.load_study(
            study_name=study_name, storage=storage
        )

        # Check if study has any trials
        if len(existing_study.trials) == 0:
            print(f"[HPO] Study '{study_name}' exists but has no trials. Will reuse.")
            return False

        # Get the first trial to check parameter distributions
        first_trial = existing_study.trials[0]
        if "model.name" not in first_trial.params:
            print(f"[HPO] Study '{study_name}' has no model.name parameter. Will reuse.")
            return False

        # Get distributions from database
        db_path = storage.replace("sqlite:///", "")
        import sqlite3
        import json

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Helper to get distribution choices
        def get_distribution_choices(param_name: str) -> list | None:
            cursor.execute(
                """
                SELECT distribution_json FROM trial_params
                WHERE trial_id IN (
                    SELECT trial_id FROM trials
                    WHERE study_id = (
                        SELECT study_id FROM studies WHERE study_name = ?
                    )
                ) AND param_name = ?
                LIMIT 1
                """,
                (study_name, param_name)
            )
            result = cursor.fetchone()
            if result:
                dist_info = json.loads(result[0])
                return dist_info.get("attributes", {}).get("choices", [])
            return None

        # Check model.name
        existing_model_choices = get_distribution_choices("model.name")

        # Check batch_size (union of all possible values)
        # Current code uses [8,12,16] for large models, [8,12,16,24,32,48,64] for base
        expected_batch_sizes = [8, 12, 16, 24, 32, 48, 64]  # Union of both
        existing_batch_sizes = get_distribution_choices("train.batch_size")

        # Check grad_accum (union of all possible values)
        # Current code uses [1,2,3,4] for large models, [1,2,3,4,6,8] for base
        expected_grad_accum = [1, 2, 3, 4, 6, 8]  # Union of both
        existing_grad_accum = get_distribution_choices("train.grad_accum")

        conn.close()

        # Validate each parameter
        incompatibilities = []

        if existing_model_choices and set(existing_model_choices) != set(expected_model_choices):
            incompatibilities.append({
                "param": "model.name",
                "existing": existing_model_choices,
                "expected": expected_model_choices,
                "removed": list(set(existing_model_choices) - set(expected_model_choices)),
                "added": list(set(expected_model_choices) - set(existing_model_choices))
            })

        if existing_batch_sizes and set(existing_batch_sizes) != set(expected_batch_sizes):
            incompatibilities.append({
                "param": "train.batch_size",
                "existing": existing_batch_sizes,
                "expected": expected_batch_sizes,
                "removed": list(set(existing_batch_sizes) - set(expected_batch_sizes)),
                "added": list(set(expected_batch_sizes) - set(existing_batch_sizes))
            })

        if existing_grad_accum and set(existing_grad_accum) != set(expected_grad_accum):
            incompatibilities.append({
                "param": "train.grad_accum",
                "existing": existing_grad_accum,
                "expected": expected_grad_accum,
                "removed": list(set(existing_grad_accum) - set(expected_grad_accum)),
                "added": list(set(expected_grad_accum) - set(existing_grad_accum))
            })

        # If any incompatibilities found, delete the study
        if incompatibilities:
            print(f"\n{'='*70}")
            print(f"[WARNING] Incompatible search space detected in study '{study_name}'")
            print(f"{'='*70}")
            for incompat in incompatibilities:
                print(f"\nParameter: {incompat['param']}")
                print(f"  Existing ({len(incompat['existing'])}): {incompat['existing']}")
                print(f"  Expected ({len(incompat['expected'])}): {incompat['expected']}")
                if incompat['removed']:
                    print(f"  Removed: {incompat['removed']}")
                if incompat['added']:
                    print(f"  Added: {incompat['added']}")

            print(f"\nDeleting incompatible study to avoid CategoricalDistribution error...")
            print(f"{'='*70}\n")

            # Delete the study
            optuna.delete_study(study_name=study_name, storage=storage)
            print(f"[HPO] Successfully deleted incompatible study '{study_name}'")
            print(f"[HPO] A new study will be created with updated search space")
            return True

        print(f"[HPO] Study '{study_name}' is compatible. Resuming optimization.")
        return False

    except KeyError:
        # Study doesn't exist
        print(f"[HPO] Study '{study_name}' does not exist. Will create new study.")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        choices=["criteria", "evidence", "share", "joint"],
        required=True,
    )
    parser.add_argument("--study", required=True)
    parser.add_argument(
        "--storage",
        default=os.getenv(
            "OPTUNA_STORAGE",
            f"sqlite:///{os.path.abspath('./_optuna/noaug.db')}",
        ),
    )
    parser.add_argument(
        "--outdir", default=os.getenv("HPO_OUTDIR", os.path.abspath("./_runs"))
    )
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--multi-objective", action="store_true")
    args = parser.parse_args()

    # Eagerly load transformers configs to prevent lazy loading race conditions
    _eager_load_transformers_configs()

    os.makedirs(
        os.path.dirname(args.storage.replace("sqlite:///", "")),
        exist_ok=True,
    )
    os.makedirs(args.outdir, exist_ok=True)

    # Validate configuration before starting expensive HPO
    validate_configuration(args)

    default_mlflow_setup(args.outdir)

    epochs = parse_env_int("HPO_EPOCHS", 100, min_val=1, max_val=1000)
    print(f"[HPO] agent={args.agent} epochs={epochs} storage={args.storage}")

    # Check for incompatible study and delete if necessary
    check_and_handle_incompatible_study(args.study, args.storage, MODEL_CHOICES)

    sampler = make_sampler(args.multi_objective, seed=2025)
    pruner = make_pruner()

    study = optuna.create_study(
        study_name=args.study,
        directions=(
            ["maximize"] if not args.multi_objective else ["maximize", "minimize"]
        ),
        storage=args.storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    objective = objective_builder(args.agent, args.outdir, args.multi_objective)
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.parallel,
        gc_after_trial=True,
    )

    print("\n[Best Trials]")
    for trial in study.best_trials[:5]:
        print(f"- value={_trial_objective_value(trial)} | params={trial.params}")

    top_limit = min(8, len(study.best_trials))
    tracking_uri = mlflow.get_tracking_uri() if _HAS_MLFLOW else None
    summary = finalize_best_trials(
        study,
        agent=args.agent,
        study_name=args.study,
        tracking_uri=tracking_uri,
        top_limit=top_limit,
    )

    if not summary:
        summary = [
            {
                "rank": idx + 1,
                "trial_number": trial.number,
                "value": _trial_objective_value(trial),
                "params": trial.params,
                "mlflow_run_id": trial.user_attrs.get("mlflow_run_id"),
            }
            for idx, trial in enumerate(study.best_trials[:top_limit])
        ]

    topk_payload = [
        {
            "rank": item["rank"],
            "trial_number": item["trial_number"],
            "value": item["value"],
            "params": item["params"],
        }
        for item in summary
    ]

    with open(
        os.path.join(args.outdir, f"{args.agent}_{args.study}_topk.json"), "w"
    ) as fh:
        json.dump(topk_payload, fh, indent=2)


if __name__ == "__main__":
    main()
