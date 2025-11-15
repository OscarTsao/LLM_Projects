"""
Comprehensive tests for augmentation_utils.py module.

Target: Increase coverage from 0% to >90%.
Tests: build_evidence_augmenter(), resolve_methods(), error handling, edge cases.
"""

import tempfile
from pathlib import Path

import pytest

from psy_agents_noaug.augmentation import AugConfig, AugResources
from psy_agents_noaug.data.augmentation_utils import (
    AugmentationArtifacts,
    build_evidence_augmenter,
    resolve_methods,
)


@pytest.fixture
def sample_train_texts():
    """Sample training texts for augmentation."""
    return [
        "Patient reports persistent sadness and loss of interest.",
        "No evidence of psychotic symptoms was observed during the interview.",
        "Sleep disturbances noted over the past two weeks.",
        "Patient exhibits anxiety and avoidance behaviors.",
        "Mood appears depressed with flat affect.",
    ]


@pytest.fixture
def temp_dir():
    """Temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestResolveMethodsFunction:
    """Test resolve_methods() function with various configurations."""

    def test_resolve_methods_none_lib(self):
        """Test that 'none' lib returns empty list."""
        result = resolve_methods("none", ["all"])
        assert result == []

    def test_resolve_methods_nlpaug_all(self):
        """Test resolving 'all' for nlpaug library."""
        result = resolve_methods("nlpaug", ["all"])
        assert len(result) >= 10
        assert all("nlpaug/" in method for method in result)

    def test_resolve_methods_textattack_all(self):
        """Test resolving 'all' for textattack library."""
        result = resolve_methods("textattack", ["all"])
        assert len(result) >= 7
        assert all("textattack/" in method for method in result)

    def test_resolve_methods_both_all(self):
        """Test resolving 'all' for both libraries."""
        result = resolve_methods("both", ["all"])
        assert len(result) >= 17

    def test_resolve_methods_single_method(self):
        """Test resolving a single specific method."""
        result = resolve_methods("nlpaug", ["nlpaug/char/KeyboardAug"])
        assert result == ["nlpaug/char/KeyboardAug"]

    def test_resolve_methods_multiple_methods(self):
        """Test resolving multiple specific methods."""
        methods = ["nlpaug/char/KeyboardAug", "nlpaug/word/SynonymAug(wordnet)"]
        result = resolve_methods("nlpaug", methods)
        assert result == methods

    def test_resolve_methods_deduplication(self):
        """Test that duplicate methods are removed."""
        methods = [
            "nlpaug/char/KeyboardAug",
            "nlpaug/char/KeyboardAug",
            "nlpaug/word/SynonymAug(wordnet)",
        ]
        result = resolve_methods("nlpaug", methods)
        assert len(result) == 2
        assert result == ["nlpaug/char/KeyboardAug", "nlpaug/word/SynonymAug(wordnet)"]

    def test_resolve_methods_unknown_method_raises(self):
        """Test that unknown method raises KeyError."""
        with pytest.raises(KeyError, match="Unknown augmentation method"):
            resolve_methods("nlpaug", ["unknown_method"])

    def test_resolve_methods_wrong_lib_filters(self):
        """Test that methods from wrong library are filtered out."""
        result = resolve_methods("nlpaug", ["textattack/CharSwapAugmenter"])
        assert result == []

    def test_resolve_methods_both_lib_accepts_all(self):
        """Test that 'both' lib accepts methods from both libraries."""
        methods = ["nlpaug/char/KeyboardAug", "textattack/CharSwapAugmenter"]
        result = resolve_methods("both", methods)
        assert len(result) == 2

    def test_resolve_methods_empty_list(self):
        """Test resolving empty method list."""
        result = resolve_methods("nlpaug", [])
        assert result == []

    def test_resolve_methods_none_value(self):
        """Test resolving None method value."""
        result = resolve_methods("nlpaug", None)
        assert result == []

    def test_resolve_methods_string_single_method(self):
        """Test resolving single method passed as string."""
        result = resolve_methods("nlpaug", "nlpaug/char/KeyboardAug")
        assert result == ["nlpaug/char/KeyboardAug"]

    def test_resolve_methods_all_with_specific(self):
        """Test resolving 'all' with additional specific methods (dedup)."""
        result = resolve_methods("nlpaug", ["all", "nlpaug/char/KeyboardAug"])
        # Should not duplicate methods already in 'all'
        assert "nlpaug/char/KeyboardAug" in result
        # Count occurrences
        keyboard_count = result.count("nlpaug/char/KeyboardAug")
        assert keyboard_count == 1


class TestBuildEvidenceAugmenterBasic:
    """Test build_evidence_augmenter() basic functionality."""

    def test_build_augmenter_disabled(self, sample_train_texts):
        """Test that disabled augmentation returns None."""
        cfg = AugConfig(lib="none")
        result = build_evidence_augmenter(cfg, sample_train_texts)
        assert result is None

    def test_build_augmenter_enabled_basic(self, sample_train_texts, temp_dir):
        """Test building basic augmenter with character-level augmentation."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=0.5,
            seed=42,
        )
        result = build_evidence_augmenter(cfg, sample_train_texts, tfidf_dir=temp_dir)

        assert result is not None
        assert isinstance(result, AugmentationArtifacts)
        assert result.pipeline is not None
        assert result.config == cfg
        assert "nlpaug/char/KeyboardAug" in result.methods

    def test_build_augmenter_multiple_methods(self, sample_train_texts, temp_dir):
        """Test building augmenter with multiple methods."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug", "nlpaug/word/SynonymAug(wordnet)"],
            p_apply=0.5,
            seed=42,
        )
        result = build_evidence_augmenter(cfg, sample_train_texts, tfidf_dir=temp_dir)

        assert result is not None
        assert len(result.methods) == 2
        assert "nlpaug/char/KeyboardAug" in result.methods
        assert "nlpaug/word/SynonymAug(wordnet)" in result.methods

    def test_build_augmenter_textattack(self, sample_train_texts, temp_dir):
        """Test building augmenter with TextAttack methods."""
        cfg = AugConfig(
            lib="textattack",
            methods=["textattack/CharSwapAugmenter"],
            p_apply=0.5,
            seed=42,
        )
        result = build_evidence_augmenter(cfg, sample_train_texts, tfidf_dir=temp_dir)

        assert result is not None
        assert "textattack/CharSwapAugmenter" in result.methods

    def test_build_augmenter_both_libs(self, sample_train_texts, temp_dir):
        """Test building augmenter with methods from both libraries."""
        cfg = AugConfig(
            lib="both",
            methods=["nlpaug/char/KeyboardAug", "textattack/CharSwapAugmenter"],
            p_apply=0.5,
            seed=42,
        )
        result = build_evidence_augmenter(cfg, sample_train_texts, tfidf_dir=temp_dir)

        assert result is not None
        assert len(result.methods) == 2


class TestBuildEvidenceAugmenterTfIdf:
    """Test build_evidence_augmenter() with TF-IDF augmentation."""

    def test_build_augmenter_tfidf_auto_fit(self, sample_train_texts, temp_dir):
        """Test that TF-IDF model is automatically fitted."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/word/TfIdfAug"],
            p_apply=0.5,
            seed=42,
        )
        result = build_evidence_augmenter(cfg, sample_train_texts, tfidf_dir=temp_dir)

        assert result is not None
        assert result.tfidf is not None
        assert result.tfidf.path.exists()
        assert result.tfidf.fitted is True
        assert result.resources.tfidf_model_path == str(result.tfidf.path)

    def test_build_augmenter_tfidf_with_existing_model(
        self, sample_train_texts, temp_dir
    ):
        """Test that existing TF-IDF model is reused."""
        # First build to create model
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/word/TfIdfAug"],
            p_apply=0.5,
            seed=42,
        )
        result1 = build_evidence_augmenter(cfg, sample_train_texts, tfidf_dir=temp_dir)
        assert result1.tfidf.fitted is True

        # Second build should load existing model
        result2 = build_evidence_augmenter(cfg, sample_train_texts, tfidf_dir=temp_dir)
        assert result2.tfidf.fitted is False  # Loaded, not fitted
        assert result2.tfidf.path == result1.tfidf.path

    def test_build_augmenter_tfidf_custom_path(self, sample_train_texts, temp_dir):
        """Test TF-IDF with custom model path."""
        model_path = temp_dir / "custom_tfidf.pkl"
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/word/TfIdfAug"],
            p_apply=0.5,
            seed=42,
            tfidf_model_path=str(model_path),
        )
        result = build_evidence_augmenter(cfg, sample_train_texts, tfidf_dir=temp_dir)

        assert result is not None
        expected_path = model_path.with_suffix("")
        assert result.tfidf.path == expected_path
        assert expected_path.exists()

    def test_build_augmenter_non_tfidf_no_tfidf_resource(
        self, sample_train_texts, temp_dir
    ):
        """Test that non-TF-IDF methods don't create TF-IDF resource."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=0.5,
            seed=42,
        )
        result = build_evidence_augmenter(cfg, sample_train_texts, tfidf_dir=temp_dir)

        assert result is not None
        assert result.tfidf is None

    def test_build_augmenter_mixed_with_tfidf(self, sample_train_texts, temp_dir):
        """Test mixed methods with TF-IDF."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug", "nlpaug/word/TfIdfAug"],
            p_apply=0.5,
            seed=42,
        )
        result = build_evidence_augmenter(cfg, sample_train_texts, tfidf_dir=temp_dir)

        assert result is not None
        assert result.tfidf is not None
        assert len(result.methods) == 2


class TestBuildEvidenceAugmenterResources:
    """Test build_evidence_augmenter() with resource configurations."""

    def test_build_augmenter_with_resources(self, sample_train_texts, temp_dir):
        """Test building augmenter with pre-configured resources."""
        tfidf_path = temp_dir / "tfidf.pkl"
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=0.5,
            seed=42,
            tfidf_model_path=str(tfidf_path),
        )
        result = build_evidence_augmenter(cfg, sample_train_texts, tfidf_dir=temp_dir)

        assert result is not None
        assert result.resources is not None
        assert isinstance(result.resources, AugResources)

    def test_build_augmenter_preserves_config(self, sample_train_texts, temp_dir):
        """Test that config is preserved in result."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=0.75,
            ops_per_sample=2,
            max_replace_ratio=0.5,
            seed=123,
        )
        result = build_evidence_augmenter(cfg, sample_train_texts, tfidf_dir=temp_dir)

        assert result is not None
        assert result.config.p_apply == 0.75
        assert result.config.ops_per_sample == 2
        assert result.config.max_replace_ratio == 0.5
        assert result.config.seed == 123


class TestBuildEvidenceAugmenterEdgeCases:
    """Test edge cases and error handling."""

    def test_build_augmenter_empty_texts(self, temp_dir):
        """Test with empty text list."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=0.5,
            seed=42,
        )
        # Non-TF-IDF methods should work with empty texts
        result = build_evidence_augmenter(cfg, [], tfidf_dir=temp_dir)
        assert result is not None

    def test_build_augmenter_single_text(self, temp_dir):
        """Test with single text."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=0.5,
            seed=42,
        )
        result = build_evidence_augmenter(
            cfg, ["Single text sample."], tfidf_dir=temp_dir
        )
        assert result is not None

    def test_build_augmenter_non_string_texts(self, temp_dir):
        """Test that non-string texts are converted to strings."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=0.5,
            seed=42,
        )
        # Mix of types
        texts = ["text1", 123, None, "text2"]
        result = build_evidence_augmenter(cfg, texts, tfidf_dir=temp_dir)
        assert result is not None

    def test_build_augmenter_all_methods(self, sample_train_texts, temp_dir):
        """Test building with 'all' methods selector."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["all"],
            p_apply=0.5,
            seed=42,
        )
        result = build_evidence_augmenter(cfg, sample_train_texts, tfidf_dir=temp_dir)

        assert result is not None
        assert len(result.methods) >= 10

    def test_build_augmenter_creates_tfidf_dir(self, sample_train_texts):
        """Test that TF-IDF directory is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tfidf_dir = Path(tmpdir) / "nested" / "tfidf"
            assert not tfidf_dir.exists()

            cfg = AugConfig(
                lib="nlpaug",
                methods=["nlpaug/word/TfIdfAug"],
                p_apply=0.5,
                seed=42,
            )
            result = build_evidence_augmenter(
                cfg, sample_train_texts, tfidf_dir=tfidf_dir
            )

            assert result is not None
            assert tfidf_dir.exists()

    def test_build_augmenter_seed_set_on_pipeline(self, sample_train_texts, temp_dir):
        """Test that seed is properly set on pipeline."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=0.5,
            seed=999,
        )
        result = build_evidence_augmenter(cfg, sample_train_texts, tfidf_dir=temp_dir)

        assert result is not None
        # Seed is set during initialization
        assert result.pipeline.cfg.seed == 999


class TestBuildEvidenceAugmenterPipelineUsage:
    """Test that built pipeline works correctly."""

    def test_built_pipeline_can_augment(self, sample_train_texts, temp_dir):
        """Test that built pipeline can perform augmentation."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=1.0,  # Always augment for testing
            seed=42,
        )
        result = build_evidence_augmenter(cfg, sample_train_texts, tfidf_dir=temp_dir)

        assert result is not None
        text = "Patient shows symptoms."
        augmented = result.pipeline(text)
        assert isinstance(augmented, str)
        assert len(augmented) > 0

    def test_built_pipeline_statistics(self, sample_train_texts, temp_dir):
        """Test that built pipeline tracks statistics."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=1.0,
            seed=42,
        )
        result = build_evidence_augmenter(cfg, sample_train_texts, tfidf_dir=temp_dir)

        # Use pipeline
        for text in sample_train_texts[:3]:
            result.pipeline(text)

        stats = result.pipeline.stats()
        assert stats["total"] == 3
        assert stats["applied"] >= 0

    def test_artifacts_dataclass_immutable(self, sample_train_texts, temp_dir):
        """Test that AugmentationArtifacts is frozen."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=0.5,
            seed=42,
        )
        result = build_evidence_augmenter(cfg, sample_train_texts, tfidf_dir=temp_dir)

        # Should not be able to modify frozen dataclass
        with pytest.raises((AttributeError, TypeError)):
            result.methods = ("new_method",)


class TestBuildEvidenceAugmenterTfIdfWorkaround:
    """Test TF-IDF with method_kwargs workaround for device parameter."""

    def test_build_augmenter_tfidf_with_method_kwargs(
        self, sample_train_texts, temp_dir
    ):
        """Test TF-IDF by excluding device from method_kwargs."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/word/TfIdfAug"],
            p_apply=0.5,
            seed=42,
            method_kwargs={
                "nlpaug/word/TfIdfAug": {"action": "substitute"}
            },  # No device
        )
        result = build_evidence_augmenter(cfg, sample_train_texts, tfidf_dir=temp_dir)

        assert result is not None
        assert result.tfidf is not None


class TestResolveMethodsPrivateHelper:
    """Test _ensure_sequence private helper function."""

    def test_ensure_sequence_with_none(self):
        """Test _ensure_sequence with None returns empty list."""
        from psy_agents_noaug.data.augmentation_utils import _ensure_sequence

        result = _ensure_sequence(None)
        assert result == []

    def test_ensure_sequence_with_string(self):
        """Test _ensure_sequence with string returns single-element list."""
        from psy_agents_noaug.data.augmentation_utils import _ensure_sequence

        result = _ensure_sequence("method")
        assert result == ["method"]

    def test_ensure_sequence_with_list(self):
        """Test _ensure_sequence with list returns list."""
        from psy_agents_noaug.data.augmentation_utils import _ensure_sequence

        result = _ensure_sequence(["m1", "m2"])
        assert result == ["m1", "m2"]

    def test_ensure_sequence_with_tuple(self):
        """Test _ensure_sequence with tuple returns list."""
        from psy_agents_noaug.data.augmentation_utils import _ensure_sequence

        result = _ensure_sequence(("m1", "m2"))
        assert result == ["m1", "m2"]
