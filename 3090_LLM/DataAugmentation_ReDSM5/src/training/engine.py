"""Reusable training engine for BERT pair classification."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from src.utils import mlflow_utils
from .data_module import DataModule, DataModuleConfig
from .dataset_builder import build_splits
from .modeling import BertPairClassifier, ModelConfig

try:  # optional dependency
    import optuna  # type: ignore
except ImportError:  # pragma: no cover
    optuna = None


METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def _flatten_dict(d: dict[str, Any], parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested dictionary for MLflow logging."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, (list, tuple)) and v and not isinstance(v[0], (dict, list, tuple)):
            items.append((new_key, str(v)))
        elif not isinstance(v, (list, tuple, dict)):
            items.append((new_key, v))
    return dict(items)


def set_global_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(model: torch.nn.Module, cfg: DictConfig) -> Optimizer:
    optimizer_name = cfg.model.optimizer.lower()
    weight_decay = cfg.model.weight_decay
    learning_rate = cfg.model.learning_rate
    adam_eps = cfg.model.get("adam_eps", 1e-8)

    optimizer_grouped_parameters = [
        {
            "params": [p for _, p in model.named_parameters() if p.requires_grad],
            "weight_decay": weight_decay,
        }
    ]

    if optimizer_name == "adamw_torch":
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    if optimizer_name == "adamw_hf":
        # AdamW removed from transformers in v5+, use torch.optim.AdamW instead
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_eps)
    if optimizer_name == "sgd":
        return torch.optim.SGD(optimizer_grouped_parameters, lr=learning_rate, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(optimizer: Optimizer, cfg: DictConfig, num_warmup_steps: int, num_training_steps: int) -> LambdaLR:
    scheduler_name = cfg.model.scheduler.lower()
    if scheduler_name == "linear":
        from transformers.optimization import get_linear_schedule_with_warmup  # type: ignore

        return get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    if scheduler_name == "cosine":
        from transformers.optimization import get_cosine_schedule_with_warmup  # type: ignore

        return get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps)
    if scheduler_name == "polynomial":
        from transformers.optimization import get_polynomial_decay_schedule_with_warmup  # type: ignore

        return get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=1.0,
        )
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)


def evaluate(model: torch.nn.Module, dataloader, device: torch.device) -> dict[str, float]:
    model.eval()
    all_labels: list[int] = []
    all_probs: list[float] = []
    all_preds: list[int] = []
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            outputs = model(**batch)
            logits = outputs["logits"].detach()
            probs = torch.softmax(logits, dim=-1)[:, 1]
            preds = torch.argmax(logits, dim=-1)
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
    if not all_labels:
        return {metric: math.nan for metric in METRIC_KEYS}
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return metrics


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def save_config(path: Path, cfg: DictConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=str(path))


def get_training_steps(num_samples: int, batch_size: int, epochs: int, grad_accum: int) -> int:
    steps_per_epoch = math.ceil(num_samples / max(batch_size, 1) / max(grad_accum, 1))
    return steps_per_epoch * epochs


def train_model(
    cfg: DictConfig,
    output_dir: Path | None = None,
    trial: "optuna.trial.Trial" | None = None,
) -> dict[str, Any]:
    set_global_seed(cfg.seed)

    # Setup MLflow tracking
    if trial is None:  # Don't log individual Optuna trials to avoid clutter
        mlflow_utils.setup_mlflow(experiment_name="redsm5-classification")
        run_name = f"train_{cfg.get('dataset', {}).get('name', 'default')}"
        mlflow_run = mlflow_utils.start_run(run_name=run_name)
    else:
        mlflow_run = mlflow_utils._NoOpContext()

    with mlflow_run:
        # Log configuration parameters
        if trial is None:
            flat_params = OmegaConf.to_container(cfg, resolve=True)
            mlflow_utils.log_params(_flatten_dict(flat_params))
            mlflow_utils.set_tag("model_type", "bert_pair_classifier")
            mlflow_utils.set_tag("framework", "pytorch")

        splits = build_splits(cfg.dataset)
        data_module = DataModule(
            split=splits,
            config=DataModuleConfig(
                tokenizer_name=cfg.model.pretrained_model_name,
                max_seq_length=cfg.model.max_seq_length,
                batch_size=cfg.model.batch_size,
                eval_batch_size=cfg.model.get("eval_batch_size"),
                num_workers=cfg.dataloader.num_workers,
                pin_memory=cfg.dataloader.pin_memory,
                persistent_workers=cfg.dataloader.persistent_workers,
                prefetch_factor=cfg.dataloader.prefetch_factor,
            ),
        )

        model = BertPairClassifier(
            ModelConfig(
                pretrained_model_name=cfg.model.pretrained_model_name,
                classifier_hidden_sizes=cfg.model.classifier_hidden_sizes,
                dropout=cfg.model.classifier_dropout,
                num_labels=2,
            )
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
            torch.backends.cudnn.allow_tf32 = True  # type: ignore[attr-defined]
        model.to(device)

        # Optional: Compile model for faster training (PyTorch 2.0+)
        if cfg.model.get("compile_model", False) and hasattr(torch, "compile"):
            model = torch.compile(model)  # type: ignore[assignment]

        optimizer = build_optimizer(model, cfg)
        grad_accum = cfg.model.gradient_accumulation_steps
        total_steps = get_training_steps(len(splits.train), cfg.model.batch_size, cfg.model.num_epochs, grad_accum)
        warmup_steps = int(total_steps * cfg.model.warmup_ratio)
        scheduler = build_scheduler(optimizer, cfg, warmup_steps, total_steps)
        # Use bfloat16 if requested (better for modern GPUs like RTX 5090)
        use_bfloat16 = cfg.model.get("use_bfloat16", False)
        dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        # GradScaler not needed for bfloat16
        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and not use_bfloat16)

        if output_dir is None:
            output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = output_dir / "checkpoints"
        best_dir = output_dir / "best"
        checkpoints_dir.mkdir(exist_ok=True)
        best_metric = -float("inf")
        best_metrics: dict[str, float] | None = None
        best_model_path = best_dir / "model.pt"

        last_ckpt_path = checkpoints_dir / "last.pt"
        start_epoch = 0
        global_step = 0

        if cfg.resume and last_ckpt_path.exists():
            state = torch.load(last_ckpt_path, map_location=device)
            model.load_state_dict(state["model_state"])
            optimizer.load_state_dict(state["optimizer_state"])
            scheduler.load_state_dict(state["scheduler_state"])
            scaler.load_state_dict(state["scaler_state"])
            start_epoch = state.get("epoch", 0)
            global_step = state.get("global_step", 0)

        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        for epoch in range(start_epoch, cfg.model.num_epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            epoch_loss = 0.0
            num_batches = 0
            progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.model.num_epochs}", leave=False)
            for step, batch in enumerate(progress):
                batch = {k: v.to(device) for k, v in batch.items()}
                labels = batch.pop("labels")
                with torch.cuda.amp.autocast(enabled=torch.cuda.is_available(), dtype=dtype):
                    outputs = model(**batch, labels=labels)
                    loss = outputs["loss"] / grad_accum
                scaler.scale(loss).backward()
                epoch_loss += loss.item() * grad_accum
                num_batches += 1

                if (step + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.model.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()
                    global_step += 1

            # Log training loss to MLflow
            avg_train_loss = epoch_loss / max(num_batches, 1)
            if trial is None:
                mlflow_utils.log_metrics({"train_loss": avg_train_loss}, step=epoch)

            metrics = evaluate(model, val_loader, device)
            monitor_metric = metrics.get(cfg.metric_for_best_model, float("nan"))

            # Log validation metrics to MLflow
            if trial is None:
                val_metrics_prefixed = {f"val_{k}": v for k, v in metrics.items()}
                mlflow_utils.log_metrics(val_metrics_prefixed, step=epoch)

            if trial is not None:
                trial.report(monitor_metric, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()  # type: ignore
            if not math.isnan(monitor_metric) and monitor_metric > best_metric:
                best_metric = monitor_metric
                best_metrics = metrics
                best_dir.mkdir(exist_ok=True)
                torch.save(model.state_dict(), best_model_path)
                save_config(best_dir / "config.yaml", cfg)
                save_json(best_dir / "val_metrics.json", metrics)

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "epoch": epoch + 1,
                    "global_step": global_step,
                },
                last_ckpt_path,
            )

        test_loader = data_module.test_dataloader()
        if best_model_path.exists():
            model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_metrics = evaluate(model, test_loader, device)
        save_json(output_dir / "test_metrics.json", test_metrics)

        # Log test metrics to MLflow
        if trial is None:
            test_metrics_prefixed = {f"test_{k}": v for k, v in test_metrics.items()}
            mlflow_utils.log_metrics(test_metrics_prefixed)
            # Log best model and config as artifacts
            if best_model_path.exists():
                mlflow_utils.log_model(best_model_path, artifact_path="model")
            mlflow_utils.log_artifact(best_dir / "config.yaml")
            mlflow_utils.log_artifact(output_dir / "test_metrics.json")

        # storage hygiene
        for path in checkpoints_dir.glob("*.pt"):
            if path.name != "last.pt":
                path.unlink(missing_ok=True)

        return {
            "best_metric": best_metric,
            "best_metrics": best_metrics,
            "test_metrics": test_metrics,
            "best_model_path": best_model_path if best_model_path.exists() else None,
            "output_dir": output_dir,
        }

