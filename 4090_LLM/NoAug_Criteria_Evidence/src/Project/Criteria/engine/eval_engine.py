from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from Project.Criteria.engine.train_engine import (
    _build_model,
    _evaluate,
    _prepare_datasets,
)
from Project.Criteria.utils import get_logger, load_best_model, set_seed


def evaluate(
    config: dict[str, Any],
    *,
    split: str = "validation",
    checkpoint_name: str | None = None,
) -> dict[str, float]:
    """Evaluate the (best) checkpoint on the requested split."""

    logger = get_logger(__name__)
    seed = set_seed(config.get("training", {}).get("seed", 42))

    tokenizer = AutoTokenizer.from_pretrained(
        config.get("model", {}).get("pretrained_model", "bert-base-uncased")
    )
    train_dataset, val_dataset, test_dataset = _prepare_datasets(
        config, tokenizer, seed
    )

    if split == "train":
        dataset: Dataset = train_dataset
    elif split == "validation":
        dataset = val_dataset
    elif split == "test":
        if test_dataset is None:
            raise ValueError(
                "Test split not available. Adjust dataset.splits to allocate test data."
            )
        dataset = test_dataset
    else:
        raise ValueError("split must be one of {'train', 'validation', 'test'}")

    eval_batch_size = config.get("training", {}).get("eval_batch_size", 32)
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

    loss_fn = nn.CrossEntropyLoss()
    metrics = _evaluate(model, dataloader, device, loss_fn)
    logger.info("Evaluation metrics on %s split: %s", split, metrics)
    return metrics


def predict(
    texts: Iterable[str],
    config: dict[str, Any],
    *,
    checkpoint_name: str | None = None,
    batch_size: int = 32,
) -> list[dict[str, Any]]:
    """Run inference on raw text inputs and return predictions with probabilities."""

    tokenizer = AutoTokenizer.from_pretrained(
        config.get("model", {}).get("pretrained_model", "bert-base-uncased")
    )
    model = _build_model(config)
    load_best_model(model, filename=checkpoint_name or "best_model.pt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    results: list[dict[str, Any]] = []
    texts_list = list(texts)
    for start in range(0, len(texts_list), batch_size):
        batch_texts = texts_list[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=config.get("dataset", {}).get("max_length", 256),
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = model(
                input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"]
            )
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)

        for text, pred, prob_vec in zip(batch_texts, preds.cpu(), probs.cpu()):
            results.append(
                {
                    "text": text,
                    "prediction": int(pred.item()),
                    "confidence": float(prob_vec[pred].item()),
                    "probabilities": prob_vec.tolist(),
                }
            )

    return results
