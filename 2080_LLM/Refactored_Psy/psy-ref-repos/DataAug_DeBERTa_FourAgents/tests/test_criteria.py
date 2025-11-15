from src.criteria.aggregate import build_criteria_results
from src.suggestion.voi import suggest_for_post

PREDICTIONS = [
    {
        "post_id": "P1",
        "sentence_id": "S1",
        "sentence": "I feel down",
        "symptom": "DEPRESSED_MOOD",
        "assertion": "present",
        "score": 0.9,
        "gold": 1,
        "eu_id": "eu1",
    },
    {
        "post_id": "P1",
        "sentence_id": "S2",
        "sentence": "I can't sleep",
        "symptom": "SLEEP_ISSUES",
        "assertion": "absent",
        "score": 0.1,
        "gold": 0,
        "eu_id": "eu2",
    },
]

SYMPTOMS = ["DEPRESSED_MOOD", "SLEEP_ISSUES", "APPETITE_CHANGE"]


def test_build_criteria_results_likely_decision():
    results = build_criteria_results(PREDICTIONS, SYMPTOMS, min_present_ratio=0.3)
    assert len(results) == 1
    res = results[0]
    assert res["decision"] == "likely"
    assert res["supporting"]["DEPRESSED_MOOD"] == ["eu1"]
    assert "APPETITE_CHANGE" in res["missing"]


def test_suggestions_respect_band():
    res = build_criteria_results(PREDICTIONS, SYMPTOMS, min_present_ratio=0.8)[0]
    suggs = suggest_for_post(res, PREDICTIONS, top_k=2)
    assert suggs, "Expected suggestion entries"
    assert all("delta_p" in s for s in suggs)
