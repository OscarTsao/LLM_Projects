#!/usr/bin/env python
"""
Evaluate checkpoint on validation set with comprehensive metrics.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import torch
import yaml
from transformers import AutoTokenizer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dataaug_multi_both.data import (
    DatasetLoader,
    build_dataset_config_from_dict,
    create_collator,
)
from dataaug_multi_both.evaluation import EvidenceExtractionEvaluator
from dataaug_multi_both.models import EvidenceExtractionModel


def _decode_span(tokenizer, input_ids, start, end):
    if start < 0 or end < start:
        return ""
    if start >= len(input_ids) or end >= len(input_ids):
        return ""

    token_ids = input_ids[start : end + 1]
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    text = tokenizer.convert_tokens_to_string(tokens)
    return text.strip()


def _char_f1(pred_text, label_text):
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


def _compute_character_f1(tokenizer, predictions, sentences):
    if not predictions:
        return 0.0

    scores = []
    for idx, pred in enumerate(predictions):
        label_text = _decode_span(
            tokenizer,
            pred.get("input_ids", []),
            int(pred.get("start_label", -1)),
            int(pred.get("end_label", -1)),
        )
        # On positive cases the label span is the entire sentence
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


def _compute_null_accuracy(tokenizer, predictions, statuses):
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


def main():
    checkpoint_path = Path("./trial_16/checkpoints/checkpoint_epoch019_step000920.pt")
    dataset_config_path = Path("./configs/data/dataset.yaml")
    model_name = "google-bert/bert-base-uncased"
    max_length = 512
    batch_size = 16

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset config
    with open(dataset_config_path, encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    dataset_section = dict(payload.get("dataset", {}))

    # Normalize data_files paths
    def _normalize_path(value):
        if isinstance(value, str):
            raw_path = Path(value)
            if raw_path.is_absolute():
                return str(raw_path)

            for base in (dataset_config_path.parent, Path.cwd()):
                candidate = (base / raw_path).resolve()
                if candidate.exists():
                    return str(candidate)
            return str((dataset_config_path.parent / raw_path).resolve())

        if isinstance(value, dict):
            return {k: _normalize_path(v) for k, v in value.items()}

        return value

    dataset_section["data_files"] = _normalize_path(dataset_section.get("data_files"))

    dataset_cfg = build_dataset_config_from_dict(
        dataset_section,
        config_dir=dataset_config_path.parent,
    )

    # Load validation split
    loader = DatasetLoader()
    splits = loader.load(dataset_cfg)
    val_dataset = splits["validation"]
    if hasattr(val_dataset, "with_format"):
        val_dataset = val_dataset.with_format("python")

    sentences = [row.get("sentence_text", "") for row in val_dataset]
    statuses = [int(row.get("status", 1)) for row in val_dataset]

    print(f"Loaded validation set with {len(val_dataset)} examples")

    # Load tokenizer and create collator
    collator = create_collator(
        model_name_or_path=model_name,
        max_length=max_length,
        local_files_only=False,
    )
    tokenizer = collator.tokenizer

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # Infer head type
    keys = state_dict.keys()
    if any(k.startswith("evidence_head.start_mlp") for k in keys):
        head_type = "start_end_mlp"
    elif any(k.startswith("evidence_head.start_linear") for k in keys):
        head_type = "start_end_linear"
    elif any("biaffine" in k for k in keys):
        head_type = "biaffine"
    else:
        head_type = "start_end_linear"

    print(f"Head type: {head_type}")

    # Create and load model
    model = EvidenceExtractionModel(
        model_name_or_path=model_name,
        head_type=head_type,
        dropout=0.1,
        pretrained_kwargs={"local_files_only": False},
    ).to(device)
    model.load_state_dict(state_dict)

    # Evaluate
    print("Running evaluation...")
    evaluator = EvidenceExtractionEvaluator(model, device=device)
    result = evaluator.evaluate(dataloader, save_predictions=True)
    predictions = result.predictions or []

    # Compute additional metrics
    char_f1 = _compute_character_f1(tokenizer, predictions, sentences)
    null_accuracy = _compute_null_accuracy(tokenizer, predictions, statuses)

    # Format results
    metrics = {
        "trial_id": "trial_16",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "device": str(device),
        "prediction_rows": len(predictions),
        "validation_metrics": {
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

    # Print results
    print("\n" + "="*60)
    print("VALIDATION SET RESULTS - Trial 16, Epoch 19")
    print("="*60)
    print(json.dumps(metrics, indent=2))
    print("="*60)

    return metrics


if __name__ == "__main__":
    main()
