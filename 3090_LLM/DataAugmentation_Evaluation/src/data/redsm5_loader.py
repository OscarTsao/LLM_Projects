"""Utilities to load ReDSM5 source data and ground truth labels."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .criteria_descriptions import CRITERIA


DEFAULT_POSTS_PATH = Path("Data/ReDSM5/redsm5_posts.csv")
DEFAULT_ANNOTATIONS_PATH = Path("Data/ReDSM5/redsm5_annotations.csv")
DEFAULT_GROUND_TRUTH_PATH = Path("Data/GroundTruth/Final_Ground_Truth.json")


@dataclass(frozen=True)
class EvidenceRecord:
    post_id: str
    criterion: str
    evidence: str
    status: int
    explanation: str | None = None


def load_posts(path: Path | str = DEFAULT_POSTS_PATH) -> pd.DataFrame:
    """Load the posts dataframe keyed by post_id."""
    df = pd.read_csv(Path(path))
    if "post_id" not in df.columns or "text" not in df.columns:
        raise ValueError("posts csv must contain post_id and text columns")
    return df.set_index("post_id")


def load_annotations(path: Path | str = DEFAULT_ANNOTATIONS_PATH) -> pd.DataFrame:
    """Load the annotations csv."""
    df = pd.read_csv(Path(path))
    expected = {"post_id", "sentence_id", "sentence_text", "DSM5_symptom", "status"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"annotations csv missing columns: {missing}")
    return df


def load_ground_truth(path: Path | str = DEFAULT_GROUND_TRUTH_PATH) -> list[dict[str, object]]:
    """Load the canonical ground truth json list."""
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("ground truth json must be a list")
    return data


def get_positive_evidence(annotations: pd.DataFrame) -> pd.DataFrame:
    """Return positive evidence sentences grouped by post and criterion."""
    filtered = annotations[annotations["status"] == 1].copy()
    filtered["DSM5_symptom"] = filtered["DSM5_symptom"].replace({"LEEP_ISSUES": "SLEEP_ISSUES"})
    filtered = filtered.sort_values(["post_id", "DSM5_symptom", "sentence_id"])
    grouped = (
        filtered.groupby(["post_id", "DSM5_symptom"], as_index=False)
        .first()
        .rename(columns={"sentence_text": "evidence"})
    )
    # Select only columns that exist
    columns = ["post_id", "DSM5_symptom", "evidence", "status"]
    if "explanation" in grouped.columns:
        columns.append("explanation")
    return grouped[columns]


def iter_ground_truth_rows(ground_truth: Sequence[dict[str, object]]) -> Iterable[dict[str, object]]:
    """Yield flattened rows of (post_id, criterion, post, label)."""
    for example in ground_truth:
        post_id = str(example["post_id"])
        post_text = str(example["post"])
        criteria = example.get("criteria", {})
        for criterion, payload in criteria.items():
            if criterion not in CRITERIA:
                continue
            label = int(payload.get("groundtruth", 0))
            row = {
                "post_id": post_id,
                "criterion": criterion,
                "post_text": post_text,
                "label": label,
            }
            evidence = payload.get("evidence")
            if isinstance(evidence, str):
                row["evidence"] = evidence
            yield row


def load_ground_truth_frame(path: Path | str = DEFAULT_GROUND_TRUTH_PATH) -> pd.DataFrame:
    """Return the flattened ground truth as a DataFrame."""
    data = list(iter_ground_truth_rows(load_ground_truth(path)))
    return pd.DataFrame(data)

