from __future__ import annotations

from typing import Dict, Iterable, List, Tuple


def suggest_for_post(
    post_result: Dict,
    predictions: Iterable[Dict],
    top_k: int = 3,
    uncertain_band: Tuple[float, float] = (0.4, 0.6),
) -> List[Dict]:
    lower, upper = uncertain_band
    suggestions: List[Dict] = []

    present_symptoms = set()
    absent_symptoms = set()
    for row in predictions:
        if row["assertion"] == "present":
            present_symptoms.add(row["symptom"])
        else:
            absent_symptoms.add(row["symptom"])

    candidate_symptoms = set(post_result.get("missing", [])) | (
        absent_symptoms - present_symptoms
    )

    for symptom in sorted(candidate_symptoms):
        delta = abs(0.5 - post_result.get("p_dx", 0.0))
        reason = "No supporting evidence; exploring symptom impact"
        if post_result.get("decision") == "uncertain":
            reason = "Decision uncertain; need more evidence"
        elif post_result.get("decision") == "unlikely":
            reason = "Decision unlikely; symptom could flip outcome"

        suggestions.append({
            "symptom": symptom,
            "delta_p": round(delta, 4),
            "reason": reason,
        })

    # Filter by uncertain band when decision probability sits inside band
    if lower <= post_result.get("p_dx", 0.0) <= upper:
        filtered = [s for s in suggestions if s["delta_p"] >= 0.1]
        if filtered:
            suggestions = filtered

    return suggestions[:top_k]


def attach_suggestions(
    criteria_results: List[Dict],
    grouped_predictions: Dict[str, List[Dict]],
    top_k: int,
    uncertain_band: Tuple[float, float],
) -> None:
    for result in criteria_results:
        post_id = result["post_id"]
        preds = grouped_predictions.get(post_id, [])
        result["suggestions"] = suggest_for_post(result, preds, top_k, uncertain_band)

