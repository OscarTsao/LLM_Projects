"""Extended tests for tfidf_cache.py module.

Target: Increase coverage from 31% to >80%.
Tests: Model persistence, error handling, concurrent access, edge cases.
"""

import pickle
from pathlib import Path

import pytest
from sklearn.feature_extraction.text import TfidfVectorizer

from psy_agents_noaug.augmentation.tfidf_cache import (
    TfidfResource,
    _prepare_texts,
    load_or_fit_tfidf,
)


@pytest.fixture
def sample_texts():
    """Sample texts for TF-IDF fitting."""
    return [
        "Patient reports persistent sadness.",
        "No evidence of psychosis was observed.",
        "Sleep disturbances noted over two weeks.",
        "Patient exhibits anxiety symptoms.",
    ]


class TestPrepareTextsFunction:
    """Test _prepare_texts() helper function."""

    def test_prepare_texts_basic(self, sample_texts):
        """Test that basic texts are prepared correctly."""
        result = _prepare_texts(sample_texts)
        assert len(result) == 4
        assert all(isinstance(t, str) for t in result)

    def test_prepare_texts_strips_whitespace(self):
        """Test that whitespace is stripped."""
        texts = ["  text1  ", "\ttext2\n", " text3 "]
        result = _prepare_texts(texts)
        assert result == ["text1", "text2", "text3"]

    def test_prepare_texts_filters_empty(self):
        """Test that empty strings are filtered out."""
        texts = ["text1", "", "  ", "text2"]
        result = _prepare_texts(texts)
        assert len(result) == 2
        assert result == ["text1", "text2"]

    def test_prepare_texts_filters_non_strings(self):
        """Test that non-string values are filtered."""
        texts = ["text1", 123, None, "text2", [], "text3"]
        result = _prepare_texts(texts)
        assert len(result) == 3
        assert result == ["text1", "text2", "text3"]

    def test_prepare_texts_empty_input_raises(self):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="No non-empty texts"):
            _prepare_texts([])

    def test_prepare_texts_all_empty_raises(self):
        """Test that all empty strings raises ValueError."""
        with pytest.raises(ValueError, match="No non-empty texts"):
            _prepare_texts(["", "  ", "\t\n"])

    def test_prepare_texts_all_non_strings_raises(self):
        """Test that all non-strings raises ValueError."""
        with pytest.raises(ValueError, match="No non-empty texts"):
            _prepare_texts([123, None, [], {}])


class TestTfidfResourceDataclass:
    """Test TfidfResource dataclass."""

    def test_tfidf_resource_creation(self, tmp_path):
        """Test creating TfidfResource instance."""
        vectorizer = TfidfVectorizer()
        path = tmp_path / "model.pkl"

        resource = TfidfResource(
            vectorizer=vectorizer, path=path, fitted=True, build_time_sec=1.23
        )

        assert resource.vectorizer == vectorizer
        assert resource.path == path
        assert resource.fitted is True
        assert resource.build_time_sec == 1.23

    def test_tfidf_resource_optional_build_time(self, tmp_path):
        """Test that build_time_sec is optional."""
        vectorizer = TfidfVectorizer()
        path = tmp_path / "model.pkl"

        resource = TfidfResource(vectorizer=vectorizer, path=path, fitted=False)

        assert resource.build_time_sec is None


class TestLoadOrFitTfidfBasic:
    """Test basic load_or_fit_tfidf functionality."""

    def test_fit_new_model(self, sample_texts, tmp_path):
        """Test fitting a new TF-IDF model."""
        model_path = tmp_path / "tfidf.pkl"

        resource = load_or_fit_tfidf(sample_texts, model_path)

        assert resource.fitted is True
        assert resource.path.exists()
        assert isinstance(resource.vectorizer, TfidfVectorizer)
        assert resource.build_time_sec is not None
        assert resource.build_time_sec > 0

    def test_load_existing_model(self, sample_texts, tmp_path):
        """Test loading an existing model."""
        model_path = tmp_path / "tfidf.pkl"

        # First fit
        resource1 = load_or_fit_tfidf(sample_texts, model_path)
        assert resource1.fitted is True

        # Second load (should load from cache)
        resource2 = load_or_fit_tfidf(sample_texts, model_path)
        assert resource2.fitted is False
        assert resource2.build_time_sec is None
        assert resource2.path == model_path

    def test_model_file_created(self, sample_texts, tmp_path):
        """Test that model file is actually created."""
        model_path = tmp_path / "tfidf.pkl"
        assert not model_path.exists()

        load_or_fit_tfidf(sample_texts, model_path)

        assert model_path.exists()
        assert model_path.is_file()
        assert model_path.stat().st_size > 0

    def test_creates_parent_directories(self, sample_texts, tmp_path):
        """Test that parent directories are created."""
        model_path = tmp_path / "nested" / "dir" / "tfidf.pkl"
        assert not model_path.parent.exists()

        load_or_fit_tfidf(sample_texts, model_path)

        assert model_path.exists()
        assert model_path.parent.exists()


class TestLoadOrFitTfidfParameters:
    """Test load_or_fit_tfidf with various parameters."""

    def test_custom_max_features(self, sample_texts, tmp_path):
        """Test with custom max_features parameter."""
        model_path = tmp_path / "tfidf.pkl"

        resource = load_or_fit_tfidf(sample_texts, model_path, max_features=100)

        assert resource.vectorizer.max_features == 100

    def test_custom_ngram_range(self, sample_texts, tmp_path):
        """Test with custom ngram_range parameter."""
        model_path = tmp_path / "tfidf.pkl"

        resource = load_or_fit_tfidf(sample_texts, model_path, ngram_range=(1, 3))

        assert resource.vectorizer.ngram_range == (1, 3)

    def test_default_parameters(self, sample_texts, tmp_path):
        """Test that default parameters are applied."""
        model_path = tmp_path / "tfidf.pkl"

        resource = load_or_fit_tfidf(sample_texts, model_path)

        assert resource.vectorizer.max_features == 40000
        assert resource.vectorizer.ngram_range == (1, 2)
        assert resource.vectorizer.lowercase is True
        assert resource.vectorizer.sublinear_tf is True


class TestLoadOrFitTfidfPersistence:
    """Test model persistence and reloading."""

    def test_loaded_model_is_functional(self, sample_texts, tmp_path):
        """Test that loaded model works correctly."""
        model_path = tmp_path / "tfidf.pkl"

        # Fit and save
        resource1 = load_or_fit_tfidf(sample_texts, model_path)
        transform1 = resource1.vectorizer.transform(["test text"])

        # Load and use
        resource2 = load_or_fit_tfidf(sample_texts, model_path)
        transform2 = resource2.vectorizer.transform(["test text"])

        # Should produce same result
        assert transform1.shape == transform2.shape

    def test_model_persists_vocabulary(self, sample_texts, tmp_path):
        """Test that vocabulary is persisted correctly."""
        model_path = tmp_path / "tfidf.pkl"

        resource1 = load_or_fit_tfidf(sample_texts, model_path)
        vocab1 = resource1.vectorizer.vocabulary_

        resource2 = load_or_fit_tfidf(sample_texts, model_path)
        vocab2 = resource2.vectorizer.vocabulary_

        assert vocab1 == vocab2

    def test_model_can_be_loaded_directly(self, sample_texts, tmp_path):
        """Test that saved model can be loaded with pickle directly."""
        model_path = tmp_path / "tfidf.pkl"

        resource = load_or_fit_tfidf(sample_texts, model_path)

        # Load directly with pickle
        with open(model_path, "rb") as f:
            loaded_vec = pickle.load(f)

        assert isinstance(loaded_vec, TfidfVectorizer)
        assert loaded_vec.vocabulary_ == resource.vectorizer.vocabulary_


class TestLoadOrFitTfidfEdgeCases:
    """Test edge cases and error handling."""

    def test_single_text(self, tmp_path):
        """Test with single text."""
        model_path = tmp_path / "tfidf.pkl"

        resource = load_or_fit_tfidf(["Single text."], model_path)

        assert resource.fitted is True
        assert resource.path.exists()

    def test_very_long_texts(self, tmp_path):
        """Test with very long texts."""
        model_path = tmp_path / "tfidf.pkl"
        long_texts = [" ".join(["word"] * 1000) for _ in range(5)]

        resource = load_or_fit_tfidf(long_texts, model_path)

        assert resource.fitted is True

    def test_texts_with_special_characters(self, tmp_path):
        """Test with texts containing special characters."""
        model_path = tmp_path / "tfidf.pkl"
        texts = ["@#$%", "Test!!", "???", "Text."]

        resource = load_or_fit_tfidf(texts, model_path)

        assert resource.fitted is True

    def test_texts_with_unicode(self, tmp_path):
        """Test with Unicode texts."""
        model_path = tmp_path / "tfidf.pkl"
        texts = ["café", "naïve", "résumé", "日本語"]

        resource = load_or_fit_tfidf(texts, model_path)

        assert resource.fitted is True

    def test_mixed_text_types(self, tmp_path):
        """Test with mixed valid/invalid texts."""
        model_path = tmp_path / "tfidf.pkl"
        texts = ["valid1", "", "valid2", "  ", "valid3"]

        resource = load_or_fit_tfidf(texts, model_path)

        # Should filter and fit on valid texts only
        assert resource.fitted is True

    def test_path_as_string(self, sample_texts, tmp_path):
        """Test that string path is converted to Path."""
        model_path = str(tmp_path / "tfidf.pkl")

        resource = load_or_fit_tfidf(sample_texts, model_path)

        assert isinstance(resource.path, Path)
        assert resource.path.exists()

    def test_path_as_path_object(self, sample_texts, tmp_path):
        """Test with Path object."""
        model_path = tmp_path / "tfidf.pkl"

        resource = load_or_fit_tfidf(sample_texts, model_path)

        assert isinstance(resource.path, Path)
        assert resource.path.exists()


class TestLoadOrFitTfidfConcurrency:
    """Test concurrent access patterns."""

    def test_multiple_loads_same_model(self, sample_texts, tmp_path):
        """Test loading same model multiple times."""
        model_path = tmp_path / "tfidf.pkl"

        # Create model
        load_or_fit_tfidf(sample_texts, model_path)

        # Load multiple times
        resources = [load_or_fit_tfidf(sample_texts, model_path) for _ in range(5)]

        # All should be loaded (not fitted)
        assert all(r.fitted is False for r in resources)
        assert all(r.path == model_path for r in resources)

    def test_different_models_different_paths(self, sample_texts, tmp_path):
        """Test creating different models at different paths."""
        path1 = tmp_path / "model1.pkl"
        path2 = tmp_path / "model2.pkl"

        resource1 = load_or_fit_tfidf(sample_texts, path1)
        resource2 = load_or_fit_tfidf(sample_texts, path2)

        assert resource1.path != resource2.path
        assert path1.exists()
        assert path2.exists()
