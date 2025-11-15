from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from transformers import Trainer, TrainingArguments

from .data import MultiLabelDataCollator, prepare_datasets
from .metrics import compute_metrics_bundle
from .models import build_model
from .thresholds import apply_temperature_scaling, apply_thresholds, sigmoid
from .utils import ensure_dir, load_json, load_yaml, merge_dicts, save_json, save_yaml, setup_logger


def aggregate_logits(
    logits: np.ndarray,
    doc_ids: List[str],
    doc_targets: Dict[str, np.ndarray],
    window_labels: Optional[np.ndarray],
    pooler: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    grouped: "OrderedDict[str, List[np.ndarray]]" = OrderedDict()
    label_map: Dict[str, np.ndarray] = {}
    doc_ids_arr = np.array(doc_ids)

    for idx, doc_id in enumerate(doc_ids):
        grouped.setdefault(doc_id, []).append(logits[idx])
        if doc_id not in label_map:
            label = doc_targets.get(doc_id)
            if label is None and window_labels is not None:
                label = window_labels[idx]
            if label is not None:
                label_map[doc_id] = np.asarray(label, dtype=np.float32)

    pooled_logits: List[np.ndarray] = []
    pooled_labels: List[np.ndarray] = []
    doc_order = list(grouped.keys())

    for doc_id in doc_order:
        stacked = np.stack(grouped[doc_id], axis=0)
        if pooler == "max":
            pooled = stacked.max(axis=0)
        elif pooler == "mean":
            pooled = stacked.mean(axis=0)
        elif pooler == "logit_sum":
            pooled = stacked.sum(axis=0)
        else:
            raise ValueError(f"Unsupported pooler '{pooler}'")
        pooled_logits.append(pooled)
        label = label_map.get(doc_id)
        if label is None and window_labels is not None:
            index = int(np.where(doc_ids_arr == doc_id)[0][0])
            label = window_labels[index]
        if label is None:
            raise KeyError(f"Missing label for document '{doc_id}'")
        pooled_labels.append(np.asarray(label, dtype=np.float32))

    return np.stack(pooled_logits), np.stack(pooled_labels), doc_order


def run_evaluation(args: argparse.Namespace) -> Dict[str, Any]:
    logger = setup_logger()
    ckpt_dir = Path(args.ckpt).resolve()
    run_dir = ckpt_dir if ckpt_dir.is_dir() else ckpt_dir.parent
    config_path = args.config or (
        (run_dir / "config_used.yaml") if (run_dir / "config_used.yaml").exists() else (run_dir.parent / "config_used.yaml")
    )
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration not found near checkpoint at {config_path}")

    cfg = load_yaml(config_path)
    cfg = merge_dicts(
        cfg,
        {
            "data_dir": args.data_dir or cfg.get("data_dir", ""),
            "hf_id": args.hf_id or cfg.get("hf_id", ""),
            "hf_config": args.hf_config or cfg.get("hf_config", ""),
        },
    )
    cfg["method"] = cfg.get("method", "qlora")
    if not cfg.get("hf_id") and not cfg.get("data_dir"):
        raise ValueError("Either --data_dir or --hf_id must be provided for evaluation")

    bundles, _, label_names, tokenizer = prepare_datasets(cfg, args.labels)
    split = args.split or cfg.get("test_split", "test")
    if split not in bundles:
        raise ValueError(f"Requested split '{split}' not available; available={list(bundles.keys())}")
    bundle = bundles[split]

    thresholds_path = args.thresholds or (run_dir / "thresholds.json")
    if not thresholds_path.exists():
        raise FileNotFoundError(f"Thresholds file not found at {thresholds_path}")
    thresholds_payload = load_json(thresholds_path)
    thresholds = np.asarray(thresholds_payload.get("thresholds"), dtype=np.float32)
    temperatures = thresholds_payload.get("temperatures")
    if temperatures is not None:
        temperatures = np.asarray(temperatures, dtype=np.float32)

    model = build_model(cfg, tokenizer, num_labels=len(label_names), checkpoint_path=str(ckpt_dir))
    collator = MultiLabelDataCollator(tokenizer)

    eval_args = TrainingArguments(
        output_dir=str(ensure_dir(args.out_dir or (run_dir / f"eval_{split}"))),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 2)),
        dataloader_num_workers=int(cfg.get("num_workers", 2)),
        report_to="none",
        fp16=bool(cfg.get("fp16", False)),
        bf16=bool(cfg.get("bf16", False)),
    )

    trainer = Trainer(model=model, args=eval_args, tokenizer=tokenizer, data_collator=collator)
    predictions = trainer.predict(bundle.dataset)
    logits = np.asarray(predictions.predictions)
    window_labels = np.asarray(predictions.label_ids) if predictions.label_ids is not None else None

    pooler = cfg.get("pooler", "mean") if cfg.get("truncation_strategy") == "window_pool" else "mean"
    agg_logits, agg_labels, doc_order = aggregate_logits(
        logits,
        bundle.doc_ids,
        bundle.doc_targets,
        window_labels,
        pooler,
    )

    calibrated = apply_temperature_scaling(agg_logits, temperatures)
    probs = sigmoid(calibrated)
    preds = apply_thresholds(probs, thresholds)

    metrics = compute_metrics_bundle(agg_labels, preds, probs, label_names)

    out_dir = ensure_dir(args.out_dir or (run_dir / f"eval_{split}"))
    payload = {
        "split": split,
        "metrics": metrics.metrics,
        "thresholds": thresholds_payload,
    }
    save_json(payload, out_dir / "metrics.json")
    save_json(metrics.confusion, out_dir / "confusion.json")
    report_df = pd.DataFrame(metrics.per_label)
    report_df.to_csv(out_dir / "label_report.csv", index=False)

    results_dict: Dict[str, Any] = {
        "doc_id": doc_order,
    }
    for idx, name in enumerate(label_names):
        results_dict[f"prob_{name}"] = probs[:, idx]
        results_dict[f"pred_{name}"] = preds[:, idx]
    pd.DataFrame(results_dict).to_csv(out_dir / "predictions.csv", index=False)

    logger.info("Evaluation complete. Macro-F1=%.4f", metrics.metrics.get("macro_f1", float("nan")))
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ReDSM5 checkpoint")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint directory (e.g., outputs/run1/best)")
    parser.add_argument("--labels", required=True, help="Path to labels YAML")
    parser.add_argument("--config", default="", help="Optional override config path")
    parser.add_argument("--data_dir", default="", help="Local data directory")
    parser.add_argument("--hf_id", default="", help="Hugging Face dataset identifier")
    parser.add_argument("--hf_config", default="", help="Hugging Face dataset config")
    parser.add_argument("--split", default="test", help="Dataset split to evaluate")
    parser.add_argument("--out_dir", default="", help="Directory to store evaluation artifacts")
    parser.add_argument("--thresholds", default="", help="Optional thresholds JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
