from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from Project.Joint.engine.train_engine import (
    _build_model,
    _evaluate,
    _prepare_datasets,
)
from Project.Joint.utils import get_logger, load_best_model, set_seed

if TYPE_CHECKING:
    from collections.abc import Iterable


def evaluate(
    config: dict[str, Any],
    *,
    split: str = "validation",
    checkpoint_name: str | None = None,
) -> dict[str, float]:
    """Evaluate the joint model on the specified split."""

    logger = get_logger(__name__)
    seed = set_seed(config.get("training", {}).get("seed", 42))

    train_dataset, val_dataset, test_dataset = _prepare_datasets(config, seed)

    if split == "train":
        dataset: Dataset = train_dataset
    elif split == "validation":
        dataset = val_dataset
    elif split == "test":
        if test_dataset is None:
            raise ValueError(
                "Test split not available. Adjust dataset.splits to reserve test data."
            )
        dataset = test_dataset
    else:
        raise ValueError("split must be one of {'train', 'validation', 'test'}")

    eval_batch_size = config.get("training", {}).get("eval_batch_size", 16)
    num_workers = config.get("dataset", {}).get("num_workers", 0)
    dataloader = DataLoader(
        dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    model = _build_model(config)
    load_best_model(model, filename=checkpoint_name or "best_model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    cls_loss_fn = nn.CrossEntropyLoss()
    span_loss_fn = nn.CrossEntropyLoss()
    loss_weights = config.get("training", {}).get(
        "loss_weights", {"criteria": 1.0, "evidence": 1.0}
    )

    metrics = _evaluate(
        model, dataloader, device, cls_loss_fn, span_loss_fn, loss_weights
    )
    logger.info("Evaluation metrics on %s split: %s", split, metrics)
    return metrics


def predict(
    inputs: Iterable[dict[str, str]],
    config: dict[str, Any],
    *,
    checkpoint_name: str | None = None,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Predict criteria label and evidence span for sentence/context pairs.

    Args:
        inputs: Iterable of dictionaries containing keys ``sentence`` and ``context``.
    """

    model = _build_model(config)
    load_best_model(model, filename=checkpoint_name or "best_model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    criteria_tokenizer = AutoTokenizer.from_pretrained(
        config.get("model", {}).get("criteria_model", "bert-base-uncased")
    )
    evidence_tokenizer = AutoTokenizer.from_pretrained(
        config.get("model", {}).get("evidence_model", "bert-base-uncased")
    )
    criteria_max_length = config.get("dataset", {}).get("criteria_max_length", 256)
    evidence_max_length = config.get("dataset", {}).get("evidence_max_length", 512)

    results: list[dict[str, Any]] = []
    items = list(inputs)

    for start_idx in range(0, len(items), batch_size):
        batch_items = items[start_idx : start_idx + batch_size]
        sentences = [item["sentence"] for item in batch_items]
        contexts = [item["context"] for item in batch_items]

        criteria_encoded = criteria_tokenizer(
            sentences,
            padding="max_length",
            truncation=True,
            max_length=criteria_max_length,
            return_tensors="pt",
        )
        evidence_encoded = evidence_tokenizer(
            contexts,
            padding="max_length",
            truncation=True,
            max_length=evidence_max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = evidence_encoded.pop("offset_mapping")

        criteria_inputs = {k: v.to(device) for k, v in criteria_encoded.items()}
        evidence_inputs = {k: v.to(device) for k, v in evidence_encoded.items()}

        with torch.no_grad():
            criteria_logits, start_logits, end_logits = model(
                criteria_input_ids=criteria_inputs["input_ids"],
                criteria_attention_mask=criteria_inputs.get("attention_mask"),
                criteria_token_type_ids=criteria_inputs.get("token_type_ids"),
                evidence_input_ids=evidence_inputs["input_ids"],
                evidence_attention_mask=evidence_inputs.get("attention_mask"),
                evidence_token_type_ids=evidence_inputs.get("token_type_ids"),
            )

            criteria_probs = torch.softmax(criteria_logits, dim=-1).cpu()
            criteria_preds = criteria_probs.argmax(dim=-1)
            start_pred = start_logits.argmax(dim=-1).cpu()
            end_pred = end_logits.argmax(dim=-1).cpu()

        for (
            item,
            cls_pred,
            cls_probs,
            start_idx_tensor,
            end_idx_tensor,
            offset_tensor,
        ) in zip(
            batch_items,
            criteria_preds,
            criteria_probs,
            start_pred,
            end_pred,
            offsets,
            strict=False,
        ):
            context = item["context"]
            offset_pairs = offset_tensor.tolist()
            start_token = int(start_idx_tensor.item())
            end_token = int(end_idx_tensor.item())
            if start_token >= len(offset_pairs):
                start_token = len(offset_pairs) - 1
            if end_token >= len(offset_pairs):
                end_token = len(offset_pairs) - 1
            end_token = max(end_token, start_token)

            start_char = offset_pairs[start_token][0]
            end_char = offset_pairs[end_token][1]
            predicted_span = (
                context[start_char:end_char] if end_char > start_char else ""
            )

            results.append(
                {
                    "sentence": item["sentence"],
                    "context": context,
                    "criteria_prediction": int(cls_pred.item()),
                    "criteria_confidence": float(cls_probs[cls_pred].item()),
                    "criteria_probabilities": cls_probs.tolist(),
                    "span_start": start_char,
                    "span_end": end_char,
                    "predicted_span": predicted_span,
                }
            )

    return results
