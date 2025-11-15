from __future__ import annotations

import argparse
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from .data import DatasetBundle, MultiLabelDataCollator, prepare_datasets
from .losses import build_loss_fn
from .metrics import MetricsResult, compute_metrics_bundle
from .models import build_model
from .thresholds import (
    ThresholdResult,
    apply_temperature_scaling,
    apply_thresholds,
    grid_search_thresholds,
    make_grid,
    sigmoid,
    temperature_grid_search,
)
from .utils import ensure_dir, load_yaml, merge_dicts, save_json, save_yaml, set_seed, setup_logger


@dataclass
class ThresholdTracker:
    best_metric: float = float("-inf")
    best_thresholds: Optional[np.ndarray] = None
    best_temperatures: Optional[np.ndarray] = None
    best_doc_ids: Optional[List[str]] = None
    best_probs: Optional[np.ndarray] = None
    best_labels: Optional[np.ndarray] = None
    best_metrics: Optional[Dict[str, float]] = None

    def update(
        self,
        metric: float,
        thresholds: np.ndarray,
        temps: Optional[np.ndarray],
        doc_ids: List[str],
        probs: np.ndarray,
        labels: np.ndarray,
        metrics: Dict[str, float],
    ) -> None:
        if metric >= self.best_metric:
            self.best_metric = float(metric)
            self.best_thresholds = thresholds.copy()
            self.best_temperatures = temps.copy() if temps is not None else None
            self.best_doc_ids = list(doc_ids)
            self.best_probs = probs.copy()
            self.best_labels = labels.copy()
            self.best_metrics = dict(metrics)


def aggregate_by_document(
    logits: np.ndarray,
    doc_ids: List[str],
    window_labels: Optional[np.ndarray],
    doc_targets: Dict[str, np.ndarray],
    pooler: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    grouped_logits: "OrderedDict[str, List[np.ndarray]]" = OrderedDict()
    doc_targets_local: Dict[str, np.ndarray] = {}

    for idx, doc_id in enumerate(doc_ids):
        grouped_logits.setdefault(doc_id, []).append(logits[idx])
        target = doc_targets.get(doc_id)
        if target is None and window_labels is not None:
            target = window_labels[idx]
        if target is not None:
            doc_targets_local[doc_id] = np.asarray(target, dtype=np.float32)

    doc_order = list(grouped_logits.keys())
    pooled_logits: List[np.ndarray] = []
    pooled_labels: List[np.ndarray] = []

    doc_ids_array = np.array(doc_ids)
    for doc_id in doc_order:
        stacked = np.stack(grouped_logits[doc_id], axis=0)
        if pooler == "max":
            pooled = stacked.max(axis=0)
        elif pooler == "mean":
            pooled = stacked.mean(axis=0)
        elif pooler == "logit_sum":
            pooled = stacked.sum(axis=0)
        else:
            raise ValueError(f"Unsupported pooler '{pooler}'")
        pooled_logits.append(pooled)
        label = doc_targets_local.get(doc_id)
        if label is None and doc_id in doc_targets:
            label = doc_targets[doc_id]
        elif label is None and window_labels is not None:
            first_idx = int(np.where(doc_ids_array == doc_id)[0][0])
            label = window_labels[first_idx]
        if label is None:
            raise KeyError(f"Missing label information for document '{doc_id}'")
        pooled_labels.append(np.asarray(label, dtype=np.float32))

    return np.stack(pooled_logits), np.stack(pooled_labels), doc_order


class DSM5Trainer(Trainer):
    def __init__(self, *args, loss_fn: torch.nn.Module, **kwargs) -> None:  # type: ignore[override]
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False):  # type: ignore[override]
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if hasattr(self.loss_fn, "to"):
            self.loss_fn = self.loss_fn.to(logits.device)
        loss = self.loss_fn(logits, labels)
        if return_outputs:
            return loss, outputs
        return loss


def create_training_arguments(cfg: Dict[str, Any], output_dir: Path) -> TrainingArguments:
    evaluation_strategy = "steps" if cfg.get("eval_steps") else "epoch"
    eval_steps = cfg.get("eval_steps") or 500
    save_strategy = evaluation_strategy
    learning_rate = float(cfg.get("lr", 5e-5))
    warmup_ratio = float(cfg.get("warmup_ratio", 0.0))
    max_steps = cfg.get("max_steps") or -1
    logging_dir = output_dir / "logs"

    optim_choice = str(cfg.get("optimizer", "adamw"))
    optim_map = {
        "adamw": "adamw_torch",
        "adafactor": "adafactor",
        "lion": "lion_8bit",
    }
    optim = optim_map.get(optim_choice, optim_choice)

    scheduler_choice = str(cfg.get("scheduler", "linear"))
    scheduler_map = {
        "linear": "linear",
        "cosine": "cosine",
        "cosine_wr": "cosine_with_restarts",
        "polynomial": "polynomial",
    }
    scheduler = scheduler_map.get(scheduler_choice, scheduler_choice)

    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("grad_accum", 1)),
        learning_rate=learning_rate,
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        num_train_epochs=float(cfg.get("epochs", 3)),
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=scheduler,
        optim=optim,
        evaluation_strategy=evaluation_strategy,
        save_strategy=save_strategy,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        logging_steps=int(cfg.get("logging_steps", 50)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        load_best_model_at_end=bool(cfg.get("load_best_at_end", True)),
        metric_for_best_model=str(cfg.get("metric_for_best_model", "macro_f1")),
        greater_is_better=bool(cfg.get("greater_is_better", True)),
        fp16=bool(cfg.get("fp16", False)),
        bf16=bool(cfg.get("bf16", False)),
        tf32=bool(cfg.get("tf32", True)),
        gradient_checkpointing=bool(cfg.get("gradient_checkpointing", False)),
        report_to=("wandb" if cfg.get("use_wandb") else "none"),
        logging_dir=str(logging_dir),
        remove_unused_columns=False,
        dataloader_num_workers=int(cfg.get("num_workers", 4)),
        dataloader_pin_memory=bool(cfg.get("pin_memory", True)),
    )


def evaluate_dataset(
    logits: np.ndarray,
    bundle: DatasetBundle,
    label_names: List[str],
    cfg: Dict[str, Any],
    tracker: Optional[ThresholdTracker] = None,
    window_labels: Optional[np.ndarray] = None,
    tune_thresholds: bool = True,
) -> Tuple[MetricsResult, ThresholdResult, np.ndarray, Optional[np.ndarray], List[str]]:
    doc_ids = bundle.doc_ids[: logits.shape[0]]
    agg_logits, agg_labels, doc_order = aggregate_by_document(
        logits,
        doc_ids,
        window_labels,
        bundle.doc_targets,
        pooler=str(cfg.get("pooler", "mean")) if cfg.get("truncation_strategy") == "window_pool" else "mean",
    )

    temperatures: Optional[np.ndarray] = None
    if cfg.get("calibration") == "temperature":
        if tune_thresholds:
            temp_grid = cfg.get("temperature_grid", [0.5, 0.75, 1.0, 1.5, 2.0])
            temperatures = temperature_grid_search(agg_logits, agg_labels, temp_grid)
        else:
            if tracker is None or tracker.best_temperatures is None:
                raise ValueError("Temperature scaling enabled but no calibrated temperatures available")
            temperatures = tracker.best_temperatures
    calibrated_logits = apply_temperature_scaling(agg_logits, temperatures)
    probs = sigmoid(calibrated_logits)

    if tune_thresholds:
        threshold_grid = make_grid(
            float(cfg.get("threshold_grid_start", 0.01)),
            float(cfg.get("threshold_grid_end", 0.99)),
            float(cfg.get("threshold_grid_step", 0.01)),
        )
        threshold_result = grid_search_thresholds(probs, agg_labels, threshold_grid.tolist())
    else:
        assert tracker is not None and tracker.best_thresholds is not None
        threshold_result = ThresholdResult(thresholds=tracker.best_thresholds, per_label_f1=np.zeros(len(label_names)))
        temperatures = tracker.best_temperatures

    preds = apply_thresholds(probs, threshold_result.thresholds)
    metrics = compute_metrics_bundle(agg_labels, preds, probs, label_names)

    if tracker is not None and tune_thresholds:
        tracker.update(
            metrics.metrics.get("macro_f1", 0.0),
            threshold_result.thresholds,
            temperatures,
            doc_order,
            probs,
            agg_labels,
            metrics.metrics,
        )

    return metrics, threshold_result, probs, temperatures, doc_order


def save_artifacts(
    output_dir: Path,
    config: Dict[str, Any],
    label_names: List[str],
    dev_metrics: MetricsResult,
    test_metrics: MetricsResult,
    tracker: ThresholdTracker,
    test_docs: List[str],
    test_probs: np.ndarray,
    test_preds: np.ndarray,
) -> None:
    ensure_dir(output_dir)
    save_yaml(config, output_dir / "config_used.yaml")
    if best_dir.exists():
        save_yaml(config, best_dir / "config_used.yaml")
    save_json(dev_metrics.metrics, output_dir / "metrics_dev.json")
    save_json(test_metrics.metrics, output_dir / "metrics_test.json")
    thresholds_payload = {
        "thresholds": tracker.best_thresholds.tolist() if tracker.best_thresholds is not None else [],
        "temperatures": tracker.best_temperatures.tolist() if tracker.best_temperatures is not None else None,
        "labels": label_names,
    }
    thresholds_path = output_dir / "thresholds.json"
    save_json(thresholds_payload, thresholds_path)
    best_dir = output_dir / "best"
    if best_dir.exists():
        save_json(thresholds_payload, best_dir / "thresholds.json")

    dev_report = pd.DataFrame(dev_metrics.per_label)
    dev_report.to_csv(output_dir / "label_report_dev.csv", index=False)
    test_report = pd.DataFrame(test_metrics.per_label)
    test_report.to_csv(output_dir / "label_report_test.csv", index=False)

    preds_dict: Dict[str, Any] = {"doc_id": test_docs}
    for idx, label in enumerate(label_names):
        preds_dict[f"prob_{label}"] = test_probs[:, idx]
        preds_dict[f"pred_{label}"] = test_preds[:, idx]
    pred_frame = pd.DataFrame(preds_dict)
    pred_frame.to_csv(output_dir / "predictions_test.csv", index=False)

    confusion_path = output_dir / "confusion_test.json"
    save_json(test_metrics.confusion, confusion_path)


def copy_best_checkpoint(best_path: str, target_dir: Path) -> None:
    if not best_path:
        return
    source = Path(best_path)
    if not source.exists():
        return
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(source, target_dir)


def run_training(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    logger = setup_logger()
    set_seed(int(cfg.get("seed", 42)))

    label_path = args.labels
    cfg = merge_dicts(cfg, {"data_dir": args.data_dir, "hf_id": args.hf_id, "hf_config": args.hf_config})
    cfg["use_wandb"] = args.use_wandb
    cfg["out_dir"] = args.out_dir

    output_dir = ensure_dir(args.out_dir)
    ckpt_dir = ensure_dir(output_dir / "checkpoints")

    bundles, class_weights, label_names, tokenizer = prepare_datasets(cfg, label_path)
    train_bundle = bundles[cfg.get("train_split", "train")]
    dev_bundle = bundles[cfg.get("dev_split", "validation")]
    test_bundle = bundles[cfg.get("test_split", "test")]

    collator = MultiLabelDataCollator(tokenizer)
    model = build_model(cfg, tokenizer, num_labels=len(label_names))

    class_weights_tensor = None
    if class_weights is not None:
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

    loss_fn = build_loss_fn(
        cfg.get("loss_type", "bce"),
        class_weights_tensor,
        label_smoothing=float(cfg.get("label_smoothing", 0.0)),
        focal_gamma=float(cfg.get("focal_gamma", 2.0)),
    )

    training_args = create_training_arguments(cfg, ckpt_dir)

    tracker = ThresholdTracker()

    def compute_metrics(eval_pred):
        logits = eval_pred.predictions
        if isinstance(logits, tuple):  # type: ignore
            logits = logits[0]
        labels = eval_pred.label_ids
        metrics, threshold_result, _, _, _ = evaluate_dataset(
            np.asarray(logits),
            dev_bundle,
            label_names,
            cfg,
            tracker=tracker,
            window_labels=np.asarray(labels),
            tune_thresholds=True,
        )
        metrics_dict = dict(metrics.metrics)
        metrics_dict.update({f"label_f1_{row['label']}": row["f1"] for row in metrics.per_label})
        return metrics_dict

    trainer = DSM5Trainer(
        model=model,
        args=training_args,
        train_dataset=train_bundle.dataset,
        eval_dataset=dev_bundle.dataset,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        loss_fn=loss_fn,
    )

    if cfg.get("early_stopping_patience"):
        trainer.add_callback(
            EarlyStoppingCallback(
                early_stopping_patience=int(cfg.get("early_stopping_patience", 3)),
                early_stopping_threshold=0.0,
            )
        )

    if cfg.get("use_wandb"):
        import wandb

        wandb.init(project=cfg.get("wandb_project", "ReDSM5"), name=cfg.get("wandb_run_name"))

    logger.info("Starting training")
    train_result = trainer.train()
    logger.info("Training finished")

    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)

    best_checkpoint = trainer.state.best_model_checkpoint or training_args.output_dir
    copy_best_checkpoint(best_checkpoint, output_dir / "best")
    tokenizer.save_pretrained(output_dir / "best")

    # Final dev evaluation with tuned thresholds
    dev_predictions = trainer.predict(dev_bundle.dataset)
    dev_metrics, _, _, _, _ = evaluate_dataset(
        np.asarray(dev_predictions.predictions),
        dev_bundle,
        label_names,
        cfg,
        tracker=tracker,
        window_labels=np.asarray(dev_predictions.label_ids) if dev_predictions.label_ids is not None else None,
        tune_thresholds=True,
    )

    best_thresholds = tracker.best_thresholds
    best_temps = tracker.best_temperatures
    if best_thresholds is None:
        raise RuntimeError("Threshold tuning failed; no best thresholds stored")

    test_predictions = trainer.predict(test_bundle.dataset)
    test_logits = np.asarray(test_predictions.predictions)
    test_metrics, _, test_probs, _, test_doc_order = evaluate_dataset(
        test_logits,
        test_bundle,
        label_names,
        cfg,
        tracker=tracker,
        window_labels=np.asarray(test_predictions.label_ids) if test_predictions.label_ids is not None else None,
        tune_thresholds=False,
    )
    test_preds = apply_thresholds(test_probs, tracker.best_thresholds)

    save_artifacts(
        output_dir,
        cfg,
        label_names,
        dev_metrics,
        test_metrics,
        tracker,
        test_doc_order,
        test_probs,
        test_preds,
    )

    return {
        "train_metrics": train_result.metrics,
        "dev_metrics": dev_metrics.metrics,
        "test_metrics": test_metrics.metrics,
        "best_thresholds": tracker.best_thresholds.tolist(),
        "best_temperatures": tracker.best_temperatures.tolist() if tracker.best_temperatures is not None else None,
        "best_checkpoint": str(output_dir / "best"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune decoder LLMs on ReDSM5")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--data_dir", default="", help="Directory with local dataset")
    parser.add_argument("--hf_id", default="", help="Hugging Face dataset ID")
    parser.add_argument("--hf_config", default="", help="Hugging Face dataset config")
    parser.add_argument("--labels", required=True, help="Path to labels YAML")
    parser.add_argument("--use_wandb", default=False, type=lambda x: str(x).lower() in {"1", "true", "t", "yes"})
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    overrides = {}
    if args.max_train_samples is not None:
        overrides["max_train_samples"] = args.max_train_samples
    if args.max_eval_samples is not None:
        overrides["max_eval_samples"] = args.max_eval_samples
    if args.max_test_samples is not None:
        overrides["max_test_samples"] = args.max_test_samples
    cfg = merge_dicts(cfg, overrides)
    run_training(cfg, args)


if __name__ == "__main__":
    main()
