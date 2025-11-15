"""
QA evaluation metrics for extractive question answering.

Implements Exact Match (EM) and F1 token overlap metrics following SQuAD evaluation.
"""

import re
import string
from collections import Counter
from typing import List, Tuple, Dict
import numpy as np


def normalize_answer(s: str) -> str:
    """
    Normalize answer text for evaluation.

    Performs:
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Normalize whitespace
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """
    Compute Exact Match (EM) score.

    Returns 1.0 if normalized strings match exactly, 0.0 otherwise.
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """
    Compute F1 score over tokens (word-level overlap).

    Returns F1 score between 0.0 and 1.0.
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()

    # Handle empty predictions/ground truths
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return float(len(pred_tokens) == len(truth_tokens))

    # Compute token overlap
    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1


def compute_metrics_batch(
    predictions: List[str],
    ground_truths: List[str],
) -> Dict[str, float]:
    """
    Compute EM and F1 metrics for a batch of predictions.

    Args:
        predictions: List of predicted answer strings
        ground_truths: List of ground truth answer strings

    Returns:
        Dictionary with 'exact_match' and 'f1' scores (averaged over batch)
    """
    assert len(predictions) == len(ground_truths), "Mismatched lengths"

    em_scores = []
    f1_scores = []

    for pred, truth in zip(predictions, ground_truths):
        em_scores.append(compute_exact_match(pred, truth))
        f1_scores.append(compute_f1(pred, truth))

    return {
        'exact_match': np.mean(em_scores),
        'f1': np.mean(f1_scores),
    }


def extract_answer_from_logits(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    input_ids: List[int],
    tokenizer,
    max_answer_length: int = 30,
) -> Tuple[str, int, int, float]:
    """
    Extract answer span from start/end logits.

    Args:
        start_logits: Start position logits [seq_length]
        end_logits: End position logits [seq_length]
        input_ids: Token IDs [seq_length]
        tokenizer: HuggingFace tokenizer
        max_answer_length: Maximum answer length in tokens

    Returns:
        Tuple of (answer_text, start_idx, end_idx, score)
    """
    # Find best valid span
    best_score = -float('inf')
    best_start = 0
    best_end = 0

    for start_idx in range(len(start_logits)):
        for end_idx in range(start_idx, min(start_idx + max_answer_length, len(end_logits))):
            score = start_logits[start_idx] + end_logits[end_idx]
            if score > best_score:
                best_score = score
                best_start = start_idx
                best_end = end_idx

    # Extract answer text
    answer_tokens = input_ids[best_start:best_end + 1]
    answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer_text, best_start, best_end, float(best_score)


def extract_answers_batch(
    start_logits_batch: np.ndarray,
    end_logits_batch: np.ndarray,
    input_ids_batch: List[List[int]],
    tokenizer,
    max_answer_length: int = 30,
) -> List[str]:
    """
    Extract answer texts for a batch.

    Args:
        start_logits_batch: [batch_size, seq_length]
        end_logits_batch: [batch_size, seq_length]
        input_ids_batch: List of token ID lists
        tokenizer: HuggingFace tokenizer
        max_answer_length: Maximum answer length

    Returns:
        List of answer strings
    """
    answers = []
    for start_logits, end_logits, input_ids in zip(
        start_logits_batch, end_logits_batch, input_ids_batch
    ):
        answer_text, _, _, _ = extract_answer_from_logits(
            start_logits, end_logits, input_ids, tokenizer, max_answer_length
        )
        answers.append(answer_text)

    return answers


def evaluate_predictions(
    predictions: List[Dict],
    include_per_symptom: bool = False,
) -> Dict:
    """
    Evaluate QA predictions with detailed metrics.

    Args:
        predictions: List of dicts with keys:
            - 'prediction': predicted answer string
            - 'ground_truth': ground truth answer string
            - 'symptom_idx': (optional) symptom index for per-class metrics
        include_per_symptom: Whether to compute per-symptom metrics

    Returns:
        Dictionary with overall and optionally per-symptom metrics
    """
    from .qa_metrics import compute_exact_match, compute_f1

    # Overall metrics
    all_em = []
    all_f1 = []

    # Per-symptom metrics
    symptom_metrics = {}

    for pred_dict in predictions:
        pred = pred_dict['prediction']
        truth = pred_dict['ground_truth']

        em = compute_exact_match(pred, truth)
        f1 = compute_f1(pred, truth)

        all_em.append(em)
        all_f1.append(f1)

        # Per-symptom tracking
        if include_per_symptom and 'symptom_idx' in pred_dict:
            symptom_idx = pred_dict['symptom_idx']
            if symptom_idx not in symptom_metrics:
                symptom_metrics[symptom_idx] = {'em': [], 'f1': []}

            symptom_metrics[symptom_idx]['em'].append(em)
            symptom_metrics[symptom_idx]['f1'].append(f1)

    # Compute overall metrics
    results = {
        'exact_match': np.mean(all_em),
        'f1': np.mean(all_f1),
        'num_examples': len(predictions),
    }

    # Compute per-symptom metrics
    if include_per_symptom:
        results['per_symptom'] = {}
        for symptom_idx, metrics in symptom_metrics.items():
            results['per_symptom'][symptom_idx] = {
                'exact_match': np.mean(metrics['em']),
                'f1': np.mean(metrics['f1']),
                'count': len(metrics['em']),
            }

    return results
