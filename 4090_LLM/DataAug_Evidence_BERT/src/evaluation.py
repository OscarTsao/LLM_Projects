from __future__ import annotations

import collections
from typing import Dict, List, Tuple

import numpy as np
from datasets import Dataset


def postprocess_qa_predictions(
    examples: Dataset,
    features: Dataset,
    raw_predictions: Tuple[np.ndarray, np.ndarray],
    tokenizer,
    n_best_size: int = 20,
    max_answer_length: int = 64,
) -> Dict[str, str]:
    """Convert raw start/end logits into final text predictions."""
    all_start_logits, all_end_logits = raw_predictions

    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for feature_index, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(feature_index)

    predicted_answers = {}

    for example_index, example in enumerate(examples):
        feature_indices = features_per_example[example_index]
        valid_answers: List[Tuple[float, str]] = []

        context = example["context"]
        for feature_index in feature_indices:
            feature = features[feature_index]
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            offset_mapping = feature["offset_mapping"]
            input_ids = feature["input_ids"]
            cls_index = input_ids.index(tokenizer.cls_token_id)

            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    start_char = offset_mapping[start_index][0]
                    end_char = offset_mapping[end_index][1]
                    valid_answers.append(
                        (start_logits[start_index] + end_logits[end_index], context[start_char:end_char])
                    )

        if valid_answers:
            best_answer = sorted(valid_answers, key=lambda x: x[0], reverse=True)[0]
            predicted_answers[example["id"]] = best_answer[1]
        else:
            predicted_answers[example["id"]] = ""

    return predicted_answers


def compute_metrics(predictions: Dict[str, str], references: Dataset) -> Dict[str, float]:
    exact_matches = []
    f1_scores = []

    reference_dict = {example["id"]: example["answers"]["text"][0] for example in references}

    for example_id, pred in predictions.items():
        gold = reference_dict.get(example_id, "")
        exact_matches.append(float(_normalize_text(pred) == _normalize_text(gold)))
        f1_scores.append(_compute_f1(pred, gold))

    return {
        "exact_match": float(np.mean(exact_matches)) if exact_matches else 0.0,
        "f1": float(np.mean(f1_scores)) if f1_scores else 0.0,
    }


def _normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _compute_f1(prediction: str, truth: str) -> float:
    pred_tokens = _normalize_text(prediction).split()
    truth_tokens = _normalize_text(truth).split()

    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(pred_tokens == truth_tokens)
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    return 2 * precision * recall / (precision + recall)
