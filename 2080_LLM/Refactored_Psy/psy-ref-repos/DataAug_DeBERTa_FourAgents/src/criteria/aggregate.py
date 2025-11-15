from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from typing import Dict, Iterable, List, Tuple

from src.schema.types import CriteriaResult


def group_by_post(predictions: Iterable[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = defaultdict(list)
    for row in predictions:
        grouped[row["post_id"]].append(row)
    return grouped


def summarise_post(
    post_id: str,
    rows: List[Dict],
    all_symptoms: List[str],
    min_present_ratio: float = 0.5,
) -> CriteriaResult:
    supporting: Dict[str, List[str]] = defaultdict(list)
    present_symptoms = set()
    absent_count = 0
    conflicts: List[str] = []
    missing: List[str] = []
    for row in rows:
        if row["assertion"] == "present":
            present_symptoms.add(row["symptom"])
            supporting[row["symptom"]].append(row["eu_id"])
        elif row["assertion"] == "absent" and row.get("gold") == 1:
            conflicts.append(row["symptom"])
        elif row["assertion"] == "absent":
            absent_count += 1

    for symptom in all_symptoms:
        if symptom not in supporting and symptom not in conflicts:
            missing.append(symptom)

    present_count = len(present_symptoms)
    denominator = max(present_count + absent_count, 1)
    ratio = present_count / denominator

    if present_count == 0 and absent_count > 0:
        decision = "unlikely"
    elif present_count == 0:
        decision = "uncertain"
    elif ratio >= min_present_ratio:
        decision = "likely"
    else:
        decision = "unlikely"

    return CriteriaResult(
        post_id=post_id,
        p_dx=round(ratio, 4),
        decision=decision,
        supporting=dict(supporting),
        conflicts=list(dict.fromkeys(conflicts)),
        missing=missing,
    )


def build_criteria_results(
    predictions: Iterable[Dict],
    symptoms: List[str],
    min_present_ratio: float = 0.5,
) -> List[Dict]:
    results: List[Dict] = []
    grouped = group_by_post(predictions)
    for post_id, rows in grouped.items():
        result = summarise_post(post_id, rows, symptoms, min_present_ratio)
        results.append(asdict(result))
    return results
