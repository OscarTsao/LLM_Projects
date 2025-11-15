import json
from pathlib import Path

from src.eval.metrics import compute_criteria_metrics, compute_evidence_metrics
from src.utils.data import load_dataset

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
        "post_id": "P2",
        "sentence_id": "S1",
        "sentence": "Sleep is fine",
        "symptom": "SLEEP_ISSUES",
        "assertion": "absent",
        "score": 0.1,
        "gold": 0,
        "eu_id": "eu2",
    },
]

CRITERIA = [
    {
        "post_id": "P1",
        "p_dx": 0.9,
        "decision": "likely",
        "supporting": {"DEPRESSED_MOOD": ["eu1"]},
        "conflicts": [],
        "missing": ["SLEEP_ISSUES"],
    },
    {
        "post_id": "P2",
        "p_dx": 0.0,
        "decision": "unlikely",
        "supporting": {},
        "conflicts": [],
        "missing": ["DEPRESSED_MOOD"],
    },
]


def test_evidence_metrics_macro_f1():
    metrics = compute_evidence_metrics(PREDICTIONS)
    assert metrics["evidence_macro_f1_present"] >= 0.5
    assert metrics["negation_precision"] == 1.0


def test_criteria_metrics_auc_ece(tmp_path):
    dataset = list(load_dataset(Path("data/redsm5_sample.jsonl")))
    metrics = compute_criteria_metrics(CRITERIA, dataset)
    assert metrics["criteria_auroc"] >= 0.5
    assert metrics["criteria_ece"] <= 0.5
