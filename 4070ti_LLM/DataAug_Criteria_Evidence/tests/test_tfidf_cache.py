"""Tests for TF-IDF caching utilities."""

from psy_agents_noaug.augmentation.tfidf_cache import load_or_fit_tfidf


def test_tfidf_fit_and_reload(tmp_path):
    texts = [
        "Patient reports persistent sadness and fatigue.",
        "No evidence of psychosis or mania.",
        "Sleep disturbances for two weeks.",
    ]
    model_path = tmp_path / "tfidf" / "tfidf.pkl"

    first = load_or_fit_tfidf(texts, model_path)
    assert first.fitted is True
    assert first.path.is_dir()
    assert first.vectorizer_path is not None
    assert first.vectorizer_path.exists()

    second = load_or_fit_tfidf(texts, model_path)
    assert second.fitted is False
    assert second.path == first.path


def test_tfidf_custom_directory(tmp_path):
    texts = ["short note"]
    base_dir = tmp_path / "custom_dir"

    resource = load_or_fit_tfidf(texts, base_dir)
    assert resource.fitted is True
    assert resource.path == base_dir
    assert resource.vectorizer_path == base_dir / "tfidf.pkl"
    assert (base_dir / "tfidfaug_w2idf.txt").exists()
    assert (base_dir / "tfidfaug_w2tfidf.txt").exists()
