from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from Project.Evidence.engine.train_engine import _build_model, _evaluate, _prepare_datasets
from Project.Evidence.utils import get_logger, load_best_model, set_seed


def evaluate(
    config: Dict[str, Any],
    *,
    split: str = "validation",
    checkpoint_name: Optional[str] = None,
) -> Dict[str, float]:
    """Evaluate the trained checkpoint on a dataset split."""

    logger = get_logger(__name__)
    seed = set_seed(config.get("training", {}).get("seed", 42))

    tokenizer = AutoTokenizer.from_pretrained(config.get("model", {}).get("pretrained_model", "bert-base-uncased"))
    train_dataset, val_dataset, test_dataset = _prepare_datasets(config, tokenizer, seed)

    if split == "train":
        dataset: Dataset = train_dataset
    elif split == "validation":
        dataset = val_dataset
    elif split == "test":
        if test_dataset is None:
            raise ValueError("Test split not available. Adjust dataset.splits to reserve test data.")
        dataset = test_dataset
    else:
        raise ValueError("split must be one of {'train', 'validation', 'test'}")

    eval_batch_size = config.get("training", {}).get("eval_batch_size", 16)
    num_workers = config.get("dataset", {}).get("num_workers", 0)
    dataloader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    model = _build_model(config)
    load_best_model(model, filename=checkpoint_name or "best_model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    metrics = _evaluate(model, dataloader, device, loss_fn)
    logger.info("Evaluation metrics on %s split: %s", split, metrics)
    return metrics


def predict(
    contexts: Iterable[str],
    config: Dict[str, Any],
    *,
    checkpoint_name: Optional[str] = None,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """Predict evidence spans from raw post texts."""

    tokenizer = AutoTokenizer.from_pretrained(config.get("model", {}).get("pretrained_model", "bert-base-uncased"))
    model = _build_model(config)
    load_best_model(model, filename=checkpoint_name or "best_model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    max_length = config.get("dataset", {}).get("max_length", 512)
    results: List[Dict[str, Any]] = []
    contexts_list = list(contexts)

    for start_idx in range(0, len(contexts_list), batch_size):
        batch_contexts = contexts_list[start_idx : start_idx + batch_size]
        encoded = tokenizer(
            batch_contexts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping")
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            start_logits, end_logits = model(**encoded)
            start_pred = start_logits.argmax(dim=-1).cpu()
            end_pred = end_logits.argmax(dim=-1).cpu()

        for context, start_idx_tensor, end_idx_tensor, offset_tensor in zip(batch_contexts, start_pred, end_pred, offsets):
            start_token = int(start_idx_tensor.item())
            end_token = int(end_idx_tensor.item())
            offset_pairs = offset_tensor.tolist()
            if start_token >= len(offset_pairs):
                start_token = len(offset_pairs) - 1
            if end_token >= len(offset_pairs):
                end_token = len(offset_pairs) - 1
            if end_token < start_token:
                end_token = start_token

            start_char = offset_pairs[start_token][0]
            end_char = offset_pairs[end_token][1]
            predicted_span = context[start_char:end_char] if end_char > start_char else ""

            results.append(
                {
                    "context": context,
                    "start_index": start_char,
                    "end_index": end_char,
                    "predicted_span": predicted_span,
                }
            )

    return results
