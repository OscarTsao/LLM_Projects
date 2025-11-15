from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_per_symptom_metrics(predictions: Iterable[Dict]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    grouped = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for row in predictions:
        symptom = row["symptom"]
        if row["assertion"] == "present":
            if row["gold"] == 1:
                grouped[symptom]["tp"] += 1
            else:
                grouped[symptom]["fp"] += 1
        else:  # predicted absent
            if row["gold"] == 1:
                grouped[symptom]["fn"] += 1

    for symptom, counts in grouped.items():
        precision = _safe_div(counts["tp"], counts["tp"] + counts["fp"])
        recall = _safe_div(counts["tp"], counts["tp"] + counts["fn"])
        if precision + recall:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        metrics[symptom] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return metrics


def compute_negation_precision(predictions: Iterable[Dict]) -> float:
    tn = 0
    total_pred_absent = 0
    for row in predictions:
        if row["assertion"] == "absent":
            total_pred_absent += 1
            if row["gold"] == 0:
                tn += 1
    return round(_safe_div(tn, total_pred_absent), 4)


def compute_evidence_metrics(predictions: List[Dict]) -> Dict[str, float]:
    per_symptom = compute_per_symptom_metrics(predictions)
    macro_f1 = _safe_div(sum(v["f1"] for v in per_symptom.values()), max(len(per_symptom), 1))
    neg_precision = compute_negation_precision(predictions)
    metrics = {
        "evidence_macro_f1_present": round(macro_f1, 4),
        "negation_precision": neg_precision,
        "symptom_metrics": per_symptom,
    }
    return metrics


def compute_post_targets(dataset: Iterable[Dict]) -> Dict[str, int]:
    targets: Dict[str, int] = {}
    for item in dataset:
        post_id = item["post_id"]
        has_positive = any(label.get("status", 0) == 1 for label in item.get("labels", []))
        targets[post_id] = 1 if has_positive else 0
    return targets


def compute_confusion(decisions: Dict[str, str], targets: Dict[str, int]) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for post_id, target in targets.items():
        decision = decisions.get(post_id, "uncertain")
        predicted_positive = decision == "likely"
        if predicted_positive and target == 1:
            tp += 1
        elif predicted_positive and target == 0:
            fp += 1
        elif not predicted_positive and target == 0:
            tn += 1
        else:
            fn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_auc(probs: List[Tuple[float, int]]) -> float:
    if not probs:
        return 0.0
    pos_probs = [p for p, y in probs if y == 1]
    neg_probs = [p for p, y in probs if y == 0]
    if not pos_probs or not neg_probs:
        return 1.0
    wins = ties = 0
    for p_pos in pos_probs:
        for p_neg in neg_probs:
            if p_pos > p_neg:
                wins += 1
            elif p_pos == p_neg:
                ties += 1
    total = len(pos_probs) * len(neg_probs)
    auc = (wins + 0.5 * ties) / total
    return round(auc, 4)


def compute_ece(probs: List[Tuple[float, int]], bins: int = 10) -> float:
    if not probs:
        return 0.0
    bin_totals = [0] * bins
    bin_correct = [0] * bins
    for prob, label in probs:
        idx = min(int(prob * bins), bins - 1)
        bin_totals[idx] += 1
        bin_correct[idx] += label
    total = sum(bin_totals)
    ece = 0.0
    for idx in range(bins):
        if bin_totals[idx] == 0:
            continue
        avg_conf = (idx + 0.5) / bins
        acc = bin_correct[idx] / bin_totals[idx]
        ece += (bin_totals[idx] / total) * abs(acc - avg_conf)
    return round(ece, 4)


def compute_criteria_metrics(
    criteria_results: List[Dict],
    dataset: Iterable[Dict],
) -> Dict[str, float]:
    targets = compute_post_targets(dataset)
    probs = []
    decisions = {}
    for result in criteria_results:
        pid = result["post_id"]
        probs.append((float(result["p_dx"]), targets.get(pid, 0)))
        decisions[pid] = result["decision"]

    confusion = compute_confusion(decisions, targets)
    precision = _safe_div(confusion["tp"], confusion["tp"] + confusion["fp"])
    recall = _safe_div(confusion["tp"], confusion["tp"] + confusion["fn"])
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision + recall else 0.0

    metrics = {
        "criteria_precision": round(precision, 4),
        "criteria_recall": round(recall, 4),
        "criteria_f1": round(f1, 4),
        "criteria_auroc": compute_auc(probs),
        "criteria_ece": compute_ece(probs),
        "confusion": confusion,
    }
    return metrics
