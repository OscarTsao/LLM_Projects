"""Utilities to load ReDSM5 source data and ground truth labels.

This module provides functions to load the ReDSM-5 dataset, which consists of:
- Social media posts from Reddit
- DSM-5 psychiatric criteria annotations
- Evidence sentences linking posts to criteria
- Ground truth labels for criteria matching

The data is organized in three main files:
1. redsm5_posts.csv: The source posts
2. redsm5_annotations.csv: Sentence-level evidence annotations
3. Final_Ground_Truth.json: Binary labels for post-criterion pairs
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .criteria_descriptions import CRITERIA


# Default data paths
DEFAULT_POSTS_PATH = Path("Data/ReDSM5/redsm5_posts.csv")
DEFAULT_ANNOTATIONS_PATH = Path("Data/ReDSM5/redsm5_annotations.csv")
DEFAULT_GROUND_TRUTH_PATH = Path("Data/GroundTruth/Final_Ground_Truth.json")


@dataclass(frozen=True)
class EvidenceRecord:
    """Record representing an evidence sentence for a DSM-5 criterion.

    Attributes:
        post_id: Unique identifier for the post
        criterion: DSM-5 criterion ID (e.g., "SLEEP_ISSUES")
        evidence: The evidence sentence text
        status: 1 if evidence supports the criterion, 0 otherwise
        explanation: Optional explanation of why this is evidence
    """
    post_id: str
    criterion: str
    evidence: str
    status: int
    explanation: str | None = None


def load_posts(path: Path | str = DEFAULT_POSTS_PATH) -> pd.DataFrame:
    """Load the posts dataframe indexed by post_id.

    Args:
        path: Path to the posts CSV file

    Returns:
        DataFrame with post_id as index and 'text' column containing post content

    Raises:
        ValueError: If required columns (post_id, text) are missing

    Example:
        >>> posts = load_posts()
        >>> post_text = posts.loc["post_123"]["text"]
    """
    df = pd.read_csv(Path(path))
    if "post_id" not in df.columns or "text" not in df.columns:
        raise ValueError("posts csv must contain post_id and text columns")
    return df.set_index("post_id")


def load_annotations(path: Path | str = DEFAULT_ANNOTATIONS_PATH) -> pd.DataFrame:
    """Load the sentence-level annotations dataframe.

    Each row represents a sentence from a post with its evidence status for a criterion.

    Args:
        path: Path to the annotations CSV file

    Returns:
        DataFrame with columns:
            - post_id: Post identifier
            - sentence_id: Sentence number within post
            - sentence_text: The sentence content
            - DSM5_symptom: Criterion ID (e.g., "SLEEP_ISSUES")
            - status: 1 if sentence is evidence, 0 otherwise

    Raises:
        ValueError: If required columns are missing
    """
    df = pd.read_csv(Path(path))
    expected = {"post_id", "sentence_id", "sentence_text", "DSM5_symptom", "status"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"annotations csv missing columns: {missing}")
    return df


def load_ground_truth(path: Path | str = DEFAULT_GROUND_TRUTH_PATH) -> list[dict[str, object]]:
    """Load the canonical ground truth labels.

    The ground truth file contains binary labels for post-criterion pairs,
    indicating whether each DSM-5 criterion matches each post.

    Args:
        path: Path to the ground truth JSON file

    Returns:
        List of dictionaries, each containing:
            - post_id: Post identifier
            - post: Post text
            - criteria: Dict mapping criterion IDs to labels and metadata

    Raises:
        ValueError: If the JSON is not a list

    Example structure:
        [
            {
                "post_id": "123",
                "post": "I can't sleep...",
                "criteria": {
                    "SLEEP_ISSUES": {"groundtruth": 1, ...},
                    "APPETITE_ISSUES": {"groundtruth": 0, ...}
                }
            },
            ...
        ]
    """
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise ValueError("ground truth json must be a list")
    return data


def get_positive_evidence(annotations: pd.DataFrame) -> pd.DataFrame:
    """Extract positive evidence sentences (status=1) grouped by post and criterion.

    This function:
    1. Filters for evidence sentences (status=1)
    2. Fixes typo "LEEP_ISSUES" → "SLEEP_ISSUES"
    3. Takes the first evidence sentence per post-criterion pair
    4. Renames columns for consistency

    Args:
        annotations: Annotations dataframe from load_annotations()

    Returns:
        DataFrame with one row per post-criterion pair, containing:
            - post_id: Post identifier
            - DSM5_symptom: Criterion ID
            - evidence: First evidence sentence text
            - status: Always 1 (positive evidence)
            - explanation: (if present) Explanation of evidence

    Example:
        >>> annotations = load_annotations()
        >>> evidence = get_positive_evidence(annotations)
        >>> # Get evidence for specific post and criterion
        >>> evidence[(evidence.post_id == "123") & (evidence.DSM5_symptom == "SLEEP_ISSUES")]
    """
    # Filter for positive evidence only (status=1)
    filtered = annotations[annotations["status"] == 1].copy()

    # Fix known typo in criterion name
    filtered["DSM5_symptom"] = filtered["DSM5_symptom"].replace({"LEEP_ISSUES": "SLEEP_ISSUES"})

    # Sort by post, criterion, and sentence order
    filtered = filtered.sort_values(["post_id", "DSM5_symptom", "sentence_id"])

    # Group by post-criterion pairs and take the first sentence
    grouped = (
        filtered.groupby(["post_id", "DSM5_symptom"], as_index=False)
        .first()
        .rename(columns={"sentence_text": "evidence"})
    )

    # Select only columns that exist (explanation is optional)
    columns = ["post_id", "DSM5_symptom", "evidence", "status"]
    if "explanation" in grouped.columns:
        columns.append("explanation")
    return grouped[columns]


def iter_ground_truth_rows(ground_truth: Sequence[dict[str, object]]) -> Iterable[dict[str, object]]:
    """Flatten ground truth JSON into individual post-criterion rows.

    Converts the nested ground truth structure into flat rows, one per post-criterion pair.
    Only includes criteria that are defined in CRITERIA dictionary.

    Args:
        ground_truth: Ground truth data from load_ground_truth()

    Yields:
        Dict for each post-criterion pair with keys:
            - post_id: Post identifier
            - post: Post text
            - criterion: Criterion ID (e.g., "SLEEP_ISSUES")
            - label: Binary label (0 or 1)

    Example:
        >>> gt = load_ground_truth()
        >>> rows = list(iter_ground_truth_rows(gt))
        >>> # Each row is a (post, criterion, label) tuple
        >>> rows[0]
        {'post_id': '123', 'post': '...', 'criterion': 'SLEEP_ISSUES', 'label': 1}
    """
    for example in ground_truth:
        post_id = str(example["post_id"])
        post_text = str(example["post"])
        criteria = example.get("criteria", {})

        # Iterate through all criteria in this post
        for criterion, payload in criteria.items():
            # Skip unknown criteria (not in CRITERIA dictionary)
            if criterion not in CRITERIA:
                continue

            # Extract binary label (default to 0 if missing)
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

