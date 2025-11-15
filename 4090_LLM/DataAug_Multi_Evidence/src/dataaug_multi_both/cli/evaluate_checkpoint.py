#!/usr/bin/env python
"""
Evaluate a saved checkpoint on the test split and export artifacts.

This CLI restores an :class:`EvidenceExtractionModel` from a checkpoint,
runs it on the configured test split, and writes:

* ``test_metrics.json`` containing span metrics
* ``test_predictions.csv`` with per-example predictions

The command is intended for offline/off-cluster runs where the full
automation stack is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml  # type: ignore[import-untyped]
from transformers import PreTrainedTokenizerBase

from dataaug_multi_both.data import (
    DatasetLoader,
    DatasetConfigurationError,
    build_dataset_config_from_dict,
    create_collator,
)
from dataaug_multi_both.evaluation import EvidenceExtractionEvaluator
from dataaug_multi_both.models import EvidenceExtractionModel


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the configured test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the checkpoint .pt file.",
    )
    parser.add_argument(
        "--dataset-config",
        type=Path,
        default=Path("configs/data/dataset.yaml"),
        help="Path to dataset YAML configuration.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="google-bert/bert-base-uncased",
        help="Hugging Face model identifier used during training.",
    )
    parser.add_argument(
        "--head-type",
        type=str,
        choices=["start_end_linear", "start_end_mlp", "biaffine"],
        default=None,
        help="Override evidence head type. If omitted, it is inferred from the checkpoint.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for the evidence head (kept for parity with training).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length used during tokenization.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device (e.g., cuda, cuda:0, cpu). Defaults to CUDA if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for evaluation artifacts. Defaults to <trial_dir>/evaluation.",
    )
    parser.add_argument(
        "--trial-id",
        type=str,
        default=None,
        help="Trial identifier for bookkeeping. Inferred from checkpoint path when omitted.",
    )
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Permit Hugging Face to fetch model/tokenizer files if they are not cached locally.",
    )
    return parser.parse_args(argv)


def _infer_head_type(state_dict: dict[str, Any]) -> str:
    keys = state_dict.keys()
    if any(k.startswith("evidence_head.start_mlp") for k in keys):
        return "start_end_mlp"
    if any(k.startswith("evidence_head.start_linear") for k in keys):
        return "start_end_linear"
    if any("biaffine" in k for k in keys):
        return "biaffine"
    return "start_end_linear"


def _load_dataset(dataset_config_path: Path) -> tuple[Any, list[str], list[int]]:
    with open(dataset_config_path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    dataset_section = payload.get("dataset")
    if not isinstance(dataset_section, dict):
        raise DatasetConfigurationError(
            f"Dataset config {dataset_config_path} does not define a 'dataset' mapping."
        )

    dataset_section = dict(dataset_section)

    def _normalize_data_files(value: Any) -> Any:
        if isinstance(value, str):
            raw_path = Path(value)
            if raw_path.is_absolute():
                return str(raw_path)

            config_parent = dataset_config_path.parent
            project_root = dataset_config_path.parent.parent.parent

            for base in (config_parent, project_root):
                candidate = (base / raw_path).resolve()
                if candidate.exists():
                    return str(candidate)
            # Fall back to config-relative path even if missing to keep DatasetLoader error message helpful.
            return str((config_parent / raw_path).resolve())

        if isinstance(value, list):
            return [_normalize_data_files(item) for item in value]

        if isinstance(value, dict):
            return {k: _normalize_data_files(v) for k, v in value.items()}

        return value

    dataset_section["data_files"] = _normalize_data_files(dataset_section.get("data_files"))

    dataset_cfg = build_dataset_config_from_dict(
        dataset_section,
        config_dir=dataset_config_path.parent,
    )
    loader = DatasetLoader()
    splits = loader.load(dataset_cfg)
    test_dataset = splits["test"]
    if hasattr(test_dataset, "with_format"):
        test_dataset = test_dataset.with_format("python")

    sentences = [row.get("sentence_text", "") for row in test_dataset]
    statuses = [int(row.get("status", 1)) for row in test_dataset]
    return test_dataset, sentences, statuses


def _decode_span(
    tokenizer: PreTrainedTokenizerBase,
    input_ids: list[int],
    start: int,
    end: int,
) -> str:
    if start < 0 or end < start:
        return ""
    if start >= len(input_ids) or end >= len(input_ids):
        return ""

    token_ids = input_ids[start : end + 1]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    text = tokenizer.convert_tokens_to_string(tokens)
    return text.strip()


def _char_f1(pred_text: str, label_text: str) -> float:
    pred_text = pred_text.strip()
    label_text = label_text.strip()

    if not label_text and not pred_text:
        return 1.0
    if not pred_text or not label_text:
        return 0.0

    pred_counter = Counter(pred_text)
    label_counter = Counter(label_text)
    intersection = sum((pred_counter & label_counter).values())
    if intersection == 0:
        return 0.0

    precision = intersection / sum(pred_counter.values())
    recall = intersection / sum(label_counter.values())
    if precision + recall == 0.0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def _compute_character_f1(
    tokenizer: PreTrainedTokenizerBase,
    predictions: list[dict[str, Any]],
    sentences: list[str],
) -> float:
    if not predictions:
        return 0.0

    scores: list[float] = []
    for idx, pred in enumerate(predictions):
        label_text = _decode_span(
            tokenizer,
            pred.get("input_ids", []),
            int(pred.get("start_label", -1)),
            int(pred.get("end_label", -1)),
        )
        # On positive cases the label span is the entire sentence; prefer raw text when available.
        if not label_text and idx < len(sentences):
            label_text = sentences[idx]

        pred_text = _decode_span(
            tokenizer,
            pred.get("input_ids", []),
            int(pred.get("start_pred", -1)),
            int(pred.get("end_pred", -1)),
        )
        scores.append(_char_f1(pred_text, label_text))

    return sum(scores) / len(scores) if scores else 0.0


def _compute_null_accuracy(
    tokenizer: PreTrainedTokenizerBase,
    predictions: list[dict[str, Any]],
    statuses: list[int],
) -> float:
    null_cases = 0
    null_correct = 0

    for idx, pred in enumerate(predictions):
        label_is_null = int(pred.get("start_label", -1)) < 0 or (
            idx < len(statuses) and statuses[idx] == 0
        )
        if not label_is_null:
            continue

        null_cases += 1
        pred_text = _decode_span(
            tokenizer,
            pred.get("input_ids", []),
            int(pred.get("start_pred", -1)),
            int(pred.get("end_pred", -1)),
        )
        if not pred_text:
            null_correct += 1

    if null_cases == 0:
        return 0.0
    return null_correct / null_cases


def _resolve_output_dir(checkpoint: Path, output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    checkpoint_resolved = checkpoint.resolve()
    trial_dir = checkpoint_resolved.parent.parent
    checkpoint_name = checkpoint_resolved.stem  # e.g., "checkpoint_epoch004_step000230"
    trial_id = trial_dir.name  # e.g., "trial_16"

    # Use outputs/prediction/{trial_id}/{checkpoint_name} structure
    project_root = trial_dir.parent
    return (project_root / "outputs" / "prediction" / trial_id / checkpoint_name).resolve()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device(
        args.device
        if args.device is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )

    test_dataset, sentences, statuses = _load_dataset(args.dataset_config.resolve())
    local_files_only = not args.allow_download

    try:
        collator = create_collator(
            model_name_or_path=args.model_name,
            max_length=args.max_length,
            local_files_only=local_files_only,
        )
    except OSError as exc:  # pragma: no cover - depends on HF cache state
        raise RuntimeError(
            "Tokenizer files not found locally. Run with --allow-download to fetch "
            "from Hugging Face or provide a local model directory via --model-name."
        ) from exc

    tokenizer = collator.tokenizer

    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict: dict[str, Any] = checkpoint["model_state_dict"]

    head_type = args.head_type or _infer_head_type(state_dict)
    try:
        model = EvidenceExtractionModel(
            model_name_or_path=args.model_name,
            head_type=head_type,
            dropout=args.dropout,
            pretrained_kwargs={"local_files_only": local_files_only},
        ).to(device)
    except OSError as exc:  # pragma: no cover - depends on HF cache state
        raise RuntimeError(
            "Model weights not found locally. Run with --allow-download to fetch "
            "from Hugging Face or supply a local checkpoint directory via --model-name."
        ) from exc
    model.load_state_dict(state_dict)

    evaluator = EvidenceExtractionEvaluator(model, device=device)
    result = evaluator.evaluate(dataloader, save_predictions=True)
    predictions = result.predictions or []

    output_dir = _resolve_output_dir(checkpoint_path, args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "test_predictions.csv"
    metrics_path = output_dir / "test_metrics.json"

    input_texts = [
        tokenizer.decode(pred.get("input_ids", []), skip_special_tokens=True).strip()
        for pred in predictions
    ]

    with predictions_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index",
                "sentence_text",
                "status",
                "input_text",
                "predicted_text",
                "label_text",
                "start_pred",
                "end_pred",
                "start_label",
                "end_label",
            ],
        )
        writer.writeheader()
        for idx, pred in enumerate(predictions):
            start_pred = int(pred.get("start_pred", -1))
            end_pred = int(pred.get("end_pred", -1))
            start_label = int(pred.get("start_label", -1))
            end_label = int(pred.get("end_label", -1))
            input_ids = pred.get("input_ids", [])

            writer.writerow(
                {
                    "index": idx,
                    "sentence_text": sentences[idx] if idx < len(sentences) else "",
                    "status": statuses[idx] if idx < len(statuses) else "",
                    "input_text": input_texts[idx] if idx < len(input_texts) else "",
                    "predicted_text": _decode_span(tokenizer, input_ids, start_pred, end_pred),
                    "label_text": _decode_span(tokenizer, input_ids, start_label, end_label),
                    "start_pred": start_pred,
                    "end_pred": end_pred,
                    "start_label": start_label,
                    "end_label": end_label,
                }
            )

    char_f1 = _compute_character_f1(tokenizer, predictions, sentences)
    null_accuracy = _compute_null_accuracy(tokenizer, predictions, statuses)

    trial_id = args.trial_id
    if not trial_id:
        trial_id = checkpoint_path.parent.parent.name

    metrics_payload = {
        "trial_id": trial_id,
        "checkpoint_path": str(checkpoint_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "prediction_rows": len(predictions),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "test_metrics": {
            "evidence_binding": {
                "span_f1": float(result.f1),
                "precision": float(result.precision),
                "recall": float(result.recall),
                "exact_match": float(result.exact_match),
                "char_f1": float(char_f1),
                "null_span_accuracy": float(null_accuracy),
            }
        },
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Wrote predictions to {predictions_path}")
    print(f"Wrote metrics to {metrics_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
