"""Utilities to fit and persist TF-IDF resources for augmentation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


@dataclass
class TfidfResource:
    """Container holding a fitted TF-IDF vectorizer and its storage path."""

    vectorizer: TfidfVectorizer | None
    path: Path  # Directory containing TF-IDF assets
    fitted: bool
    build_time_sec: float | None = None
    vectorizer_path: Path | None = None


def _prepare_texts(texts: Iterable[str]) -> list[str]:
    prepared: list[str] = []
    for text in texts:
        if not isinstance(text, str):
            continue
        stripped = text.strip()
        if stripped:
            prepared.append(stripped)
    if not prepared:
        raise ValueError("No non-empty texts provided for TF-IDF fitting")
    return prepared


def load_or_fit_tfidf(
    train_texts: Iterable[str] | Sequence[str],
    model_path: str | Path,
    *,
    max_features: int = 40000,
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfResource:
    """
    Load an existing TF-IDF vectorizer or fit a new one on the provided texts.

    Args:
        train_texts: Iterable of training evidence texts.
        model_path: File path for the cached model (joblib format).
        max_features: Maximum vocabulary size.
        ngram_range: N-gram range for vectoriser.

    Returns:
        TfidfResource containing the fitted vectoriser and path.
    """
    start = time.time()
    target = Path(model_path)
    if target.suffix:
        base_dir = target.parent
        vectorizer_file = target
    else:
        base_dir = target
        vectorizer_file = base_dir / "tfidf.pkl"

    base_dir.mkdir(parents=True, exist_ok=True)

    idf_file = base_dir / "tfidfaug_w2idf.txt"
    tfidf_file = base_dir / "tfidfaug_w2tfidf.txt"

    if idf_file.exists() and tfidf_file.exists():
        vectorizer = None
        if vectorizer_file.exists():
            with vectorizer_file.open("rb") as handle:
                vectorizer = joblib.load(handle)
        return TfidfResource(
            vectorizer=vectorizer,
            path=base_dir,
            fitted=False,
            build_time_sec=None,
            vectorizer_path=vectorizer_file,
        )

    texts = _prepare_texts(train_texts)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    idf_values = vectorizer.idf_
    # Average TF-IDF score per token across documents
    tfidf_scores = matrix.sum(axis=0).A1 / max(1, matrix.shape[0])

    with idf_file.open("w", encoding="utf-8") as handle:
        for idx, token in enumerate(feature_names):
            safe_token = str(token).replace(" ", "_")
            handle.write(f"{safe_token} {idf_values[idx]}\n")

    with tfidf_file.open("w", encoding="utf-8") as handle:
        for idx, token in enumerate(feature_names):
            safe_token = str(token).replace(" ", "_")
            handle.write(f"{safe_token} {tfidf_scores[idx]}\n")

    with vectorizer_file.open("wb") as handle:
        joblib.dump(vectorizer, handle)

    return TfidfResource(
        vectorizer=vectorizer,
        path=base_dir,
        fitted=True,
        build_time_sec=time.time() - start,
        vectorizer_path=vectorizer_file,
    )
