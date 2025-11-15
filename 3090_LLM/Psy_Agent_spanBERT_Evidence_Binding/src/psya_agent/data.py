from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QAExample:
    """Single supervision item containing context text and one annotated answer."""

    example_id: str
    post_id: str
    context: str
    answer_text: str
    answer_start: int


def load_posts(path: Path) -> Dict[str, str]:
    """Load posts from a JSON lines file keyed by post_id.

    Args:
        path: Path to JSONL file containing posts

    Returns:
        Dictionary mapping post_id to post text

    Raises:
        FileNotFoundError: If the path does not exist
        json.JSONDecodeError: If the file contains invalid JSON
        KeyError: If required fields are missing
    """
    if not path.exists():
        raise FileNotFoundError(f"Posts file not found: {path}")

    posts: Dict[str, str] = {}
    line_num = 0

    try:
        with path.open("r", encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
                    continue

                if "post_id" not in data:
                    logger.warning(f"Skipping entry at line {line_num}: missing 'post_id'")
                    continue
                if "text" not in data:
                    logger.warning(f"Skipping entry at line {line_num}: missing 'text'")
                    continue

                post_id = str(data["post_id"])
                posts[post_id] = str(data["text"])

    except Exception as e:
        logger.error(f"Error loading posts from {path}: {e}")
        raise

    logger.info(f"Loaded {len(posts)} posts from {path}")
    return posts


def load_annotations(path: Path, positive_only: bool = True) -> pd.DataFrame:
    """Load the RED-SM5 annotations CSV and filter to positive spans if requested.

    Args:
        path: Path to annotations CSV file
        positive_only: If True, filter to only positive examples (status == 1)

    Returns:
        DataFrame with annotations

    Raises:
        FileNotFoundError: If the path does not exist
        pd.errors.EmptyDataError: If the file is empty
    """
    if not path.exists():
        raise FileNotFoundError(f"Annotations file not found: {path}")

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except pd.errors.EmptyDataError:
        logger.error(f"Annotations file is empty: {path}")
        raise
    except Exception as e:
        logger.error(f"Error reading annotations from {path}: {e}")
        raise

    if df.empty:
        logger.warning(f"No annotations found in {path}")
        return df

    logger.info(f"Loaded {len(df)} annotations from {path}")

    if positive_only and "status" in df.columns:
        df = df[df["status"] == 1].copy()
        logger.info(f"Filtered to {len(df)} positive annotations")

    return df


def _find_answer_span(context: str, answer: str) -> Optional[Tuple[int, int]]:
    """Locate an answer string inside the context with basic fallback heuristics.

    Args:
        context: The full text context
        answer: The answer text to find

    Returns:
        Tuple of (start_index, end_index) if found, None otherwise
    """
    if not answer or not context:
        return None

    answer_clean = answer.strip()
    if not answer_clean:
        return None

    # Try exact match first
    start_idx = context.find(answer_clean)
    if start_idx != -1:
        return start_idx, start_idx + len(answer_clean)

    # Try case-insensitive match
    context_lower = context.lower()
    answer_lower = answer_clean.lower()
    start_idx = context_lower.find(answer_lower)
    if start_idx != -1:
        # Use the original length of the answer for the end index
        return start_idx, start_idx + len(answer_clean)

    return None


def build_examples(
    posts: Dict[str, str], annotations: pd.DataFrame
) -> List[QAExample]:
    """Generate QAExample instances by aligning sentence annotations with posts.

    Args:
        posts: Dictionary mapping post_id to post text
        annotations: DataFrame containing annotations with post_id and sentence_text

    Returns:
        List of QAExample instances

    Raises:
        ValueError: If no valid examples could be created
    """
    if not posts:
        raise ValueError("No posts provided")
    if annotations.empty:
        raise ValueError("No annotations provided")

    examples: List[QAExample] = []
    skipped_no_context = 0
    skipped_no_span = 0

    for idx, row in enumerate(annotations.itertuples()):
        post_id = str(getattr(row, "post_id", ""))
        if not post_id:
            skipped_no_context += 1
            continue

        context = posts.get(post_id)
        if not context:
            skipped_no_context += 1
            continue

        answer_text: str = str(getattr(row, "sentence_text", "")).strip()
        if not answer_text:
            skipped_no_span += 1
            continue

        span = _find_answer_span(context, answer_text)
        if not span:
            skipped_no_span += 1
            continue

        start, _ = span
        sentence_id = getattr(row, "sentence_id", idx)
        example_id = f"{post_id}_{sentence_id}"

        examples.append(
            QAExample(
                example_id=example_id,
                post_id=post_id,
                context=context,
                answer_text=answer_text,
                answer_start=start,
            )
        )

    if skipped_no_context > 0:
        logger.warning(f"Skipped {skipped_no_context} annotations with missing or invalid post_id")
    if skipped_no_span > 0:
        logger.warning(f"Skipped {skipped_no_span} annotations that could not be aligned to context")

    if not examples:
        raise ValueError(
            f"No valid examples created. Skipped {skipped_no_context + skipped_no_span} annotations"
        )

    logger.info(f"Created {len(examples)} examples from {len(annotations)} annotations")
    return examples


def split_examples(
    examples: Sequence[QAExample],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[QAExample], List[QAExample], List[QAExample]]:
    """Split examples ensuring posts do not leak across splits.

    Args:
        examples: Sequence of QAExample instances
        train_ratio: Proportion of posts for training (0.0-1.0)
        val_ratio: Proportion of posts for validation (0.0-1.0)
        seed: Random seed for reproducible splitting

    Returns:
        Tuple of (train_examples, val_examples, test_examples)

    Raises:
        ValueError: If ratios are invalid or no examples provided
    """
    if not examples:
        logger.warning("No examples provided for splitting")
        return [], [], []

    if not (0.0 <= train_ratio <= 1.0):
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    if not (0.0 <= val_ratio <= 1.0):
        raise ValueError(f"val_ratio must be between 0 and 1, got {val_ratio}")
    if train_ratio + val_ratio > 1.0:
        raise ValueError(
            f"train_ratio ({train_ratio}) + val_ratio ({val_ratio}) must not exceed 1.0"
        )

    # Group examples by post_id to prevent leakage
    by_post: Dict[str, List[QAExample]] = {}
    for ex in examples:
        by_post.setdefault(ex.post_id, []).append(ex)

    post_ids = list(by_post.keys())
    rng = random.Random(seed)
    rng.shuffle(post_ids)

    n_posts = len(post_ids)
    train_cut = int(n_posts * train_ratio)
    val_cut = train_cut + int(n_posts * val_ratio)

    # Ensure at least one post per split if possible
    if train_cut == 0 and train_ratio > 0 and n_posts > 0:
        train_cut = 1
    if val_cut == train_cut and val_ratio > 0 and n_posts > train_cut:
        val_cut = train_cut + 1

    train_posts = set(post_ids[:train_cut])
    val_posts = set(post_ids[train_cut:val_cut])
    test_posts = set(post_ids[val_cut:])

    train_examples, val_examples, test_examples = [], [], []
    for post_id, post_examples in by_post.items():
        if post_id in train_posts:
            train_examples.extend(post_examples)
        elif post_id in val_posts:
            val_examples.extend(post_examples)
        else:
            test_examples.extend(post_examples)

    logger.info(
        f"Split {len(examples)} examples into {len(train_examples)} train, "
        f"{len(val_examples)} val, {len(test_examples)} test "
        f"({len(train_posts)} / {len(val_posts)} / {len(test_posts)} posts)"
    )

    return train_examples, val_examples, test_examples


__all__ = [
    "QAExample",
    "load_posts",
    "load_annotations",
    "build_examples",
    "split_examples",
]
