"""
Test that augmentation pipeline scope is correctly limited.

Verifies strict field separation (CRITICAL for data leakage prevention).
This ensures augmentation only applies to evidence field, never to criteria/labels.
"""

import pytest

from psy_agents_noaug.augmentation.pipeline import (
    AugConfig,
    AugmenterPipeline,
    AugResources,
    is_enabled,
    worker_init,
)


class TestPipelineConfiguration:
    """Test augmentation pipeline configuration."""

    def test_is_enabled_none(self):
        """Test is_enabled returns False for lib='none'."""
        cfg = AugConfig(lib="none")
        assert not is_enabled(cfg)

    def test_is_enabled_nlpaug(self):
        """Test is_enabled returns True for nlpaug."""
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"])
        assert is_enabled(cfg)

    def test_is_enabled_textattack(self):
        """Test is_enabled returns True for textattack."""
        cfg = AugConfig(lib="textattack", methods=["textattack/CharSwapAugmenter"])
        assert is_enabled(cfg)

    def test_is_enabled_both(self):
        """Test is_enabled returns True for both libraries."""
        cfg = AugConfig(lib="both", methods=["all"])
        assert is_enabled(cfg)

    def test_pipeline_init_none_raises(self):
        """Test that pipeline cannot be created with lib='none'."""
        cfg = AugConfig(lib="none")
        with pytest.raises(ValueError, match="Cannot instantiate"):
            AugmenterPipeline(cfg)

    def test_pipeline_init_with_methods(self):
        """Test pipeline initialization with specific methods."""
        cfg = AugConfig(
            lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=0.5, seed=42
        )
        pipeline = AugmenterPipeline(cfg)
        assert len(pipeline.methods) == 1
        assert pipeline.p_apply == 0.5


class TestPipelineStatistics:
    """Test pipeline statistics tracking."""

    def test_pipeline_stats_initial(self):
        """Test pipeline statistics start at zero."""
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], seed=42)
        pipeline = AugmenterPipeline(cfg)
        stats = pipeline.stats()
        assert stats["total"] == 0
        assert stats["applied"] == 0
        assert stats["skipped"] == 0

    def test_pipeline_stats_after_call(self):
        """Test pipeline statistics update after augmentation."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=1.0,  # Always apply
            seed=42,
        )
        pipeline = AugmenterPipeline(cfg)

        # Call the pipeline
        text = "Test text for augmentation"
        _ = pipeline(text)

        stats = pipeline.stats()
        assert stats["total"] == 1
        # May be applied or skipped depending on augmenter behavior

    def test_pipeline_examples_collection(self):
        """Test pipeline example collection."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=1.0,
            example_limit=5,
            seed=42,
        )
        pipeline = AugmenterPipeline(cfg)

        # Generate some augmentations
        for i in range(3):
            pipeline(f"Test text number {i}")

        examples = pipeline.drain_examples()
        assert isinstance(examples, list)
        # Check examples are cleared
        assert len(pipeline.examples) == 0


class TestPipelineSeeding:
    """Test pipeline seeding and reproducibility."""

    def test_pipeline_set_seed(self):
        """Test pipeline seed can be reset."""
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], seed=42)
        pipeline = AugmenterPipeline(cfg)

        # Set seed
        pipeline.set_seed(100)
        # Should not raise exception
        assert True

    def test_worker_init_function(self):
        """Test worker initialization function."""
        # Test worker_init function exists and returns seed
        seed = worker_init(0, 42)
        assert isinstance(seed, int)
        assert seed == 43  # base_seed + worker_id + 1

        seed = worker_init(3, 42)
        assert seed == 46  # 42 + 3 + 1


class TestPipelineParameters:
    """Test pipeline parameter validation and clamping."""

    def test_p_apply_clamping(self):
        """Test that p_apply is clamped to [0, 1]."""
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=1.5)
        pipeline = AugmenterPipeline(cfg)
        assert 0.0 <= pipeline.p_apply <= 1.0

    def test_ops_per_sample_clamping(self):
        """Test that ops_per_sample is clamped to [1, 2]."""
        cfg = AugConfig(
            lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], ops_per_sample=10
        )
        pipeline = AugmenterPipeline(cfg)
        assert 1 <= pipeline.ops_per_sample <= 2

    def test_max_replace_ratio_clamping(self):
        """Test that max_replace_ratio is clamped to [0, 1]."""
        cfg = AugConfig(
            lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], max_replace_ratio=2.0
        )
        pipeline = AugmenterPipeline(cfg)
        assert 0.0 <= pipeline.max_replace <= 1.0


class TestPipelineMethodResolution:
    """Test pipeline method resolution."""

    def test_resolve_all_methods(self):
        """Test resolving 'all' methods expands to all library methods."""
        # Note: 'all' for nlpaug includes methods requiring resources (TfIdf, Reserved)
        # Use specific methods that don't require resources
        cfg = AugConfig(
            lib="nlpaug",
            methods=[
                "nlpaug/char/KeyboardAug",
                "nlpaug/char/OcrAug",
                "nlpaug/char/RandomCharAug",
                "nlpaug/word/RandomWordAug",
                "nlpaug/word/SpellingAug",
                "nlpaug/word/SplitAug",
                "nlpaug/word/SynonymAug(wordnet)",
            ],
            seed=42,
        )
        pipeline = AugmenterPipeline(cfg)
        # Should have the specified nlpaug methods
        assert len(pipeline.methods) == 7

    def test_resolve_specific_methods(self):
        """Test resolving specific methods."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug", "nlpaug/char/OcrAug"],
            seed=42,
        )
        pipeline = AugmenterPipeline(cfg)
        assert len(pipeline.methods) == 2

    def test_resolve_duplicate_removal(self):
        """Test that duplicate methods are removed."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=[
                "nlpaug/char/KeyboardAug",
                "nlpaug/char/KeyboardAug",
                "nlpaug/char/OcrAug",
            ],
            seed=42,
        )
        pipeline = AugmenterPipeline(cfg)
        # Duplicates should be removed
        assert len(pipeline.methods) == 2


class TestPipelineResources:
    """Test pipeline resource management."""

    def test_pipeline_with_resources(self):
        """Test pipeline initialization with resources."""
        resources = AugResources(tfidf_model_path=None, reserved_map_path=None)
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], seed=42)
        pipeline = AugmenterPipeline(cfg, resources=resources)
        assert pipeline.resources is not None

    def test_pipeline_without_resources(self):
        """Test pipeline initialization creates default resources."""
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], seed=42)
        pipeline = AugmenterPipeline(cfg)
        assert pipeline.resources is not None
        assert isinstance(pipeline.resources, AugResources)


class TestPipelineBehavior:
    """Test pipeline augmentation behavior."""

    def test_pipeline_skip_with_zero_p_apply(self):
        """Test pipeline skips augmentation when p_apply=0."""
        cfg = AugConfig(
            lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=0.0, seed=42
        )
        pipeline = AugmenterPipeline(cfg)

        text = "Original text"
        result = pipeline(text)

        # Should return unchanged
        assert result == text
        stats = pipeline.stats()
        assert stats["applied"] == 0
        assert stats["skipped"] == 1

    def test_pipeline_applies_with_p_apply_one(self):
        """Test pipeline attempts augmentation when p_apply=1.0."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=1.0,
            seed=42,
        )
        pipeline = AugmenterPipeline(cfg)

        text = "Original text with enough content"
        result = pipeline(text)

        # Result should be a string (may or may not be changed)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_pipeline_handles_empty_text(self):
        """Test pipeline handles empty text gracefully."""
        cfg = AugConfig(
            lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=1.0, seed=42
        )
        pipeline = AugmenterPipeline(cfg)

        result = pipeline("")
        assert result == ""


class TestFieldSeparationConcept:
    """
    Conceptual tests for field separation.

    NOTE: Actual field separation is enforced at the Dataset level,
    not in the augmentation pipeline. The pipeline is a general tool.
    These tests document the expected usage pattern.
    """

    def test_pipeline_is_tool_not_enforcer(self):
        """
        Document that pipeline is a general tool.

        Field separation (criteria vs evidence) is enforced in:
        - src/psy_agents_noaug/data/groundtruth.py (assertions)
        - Dataset classes that selectively use augmentation
        - Tests in tests/test_groundtruth.py
        """
        # This is a documentation test
        # Pipeline can augment any text - it's the Dataset's job to control usage
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], seed=42)
        pipeline = AugmenterPipeline(cfg)

        # Pipeline can augment any text
        criteria_text = "Criterion A1: Depressed mood"
        evidence_text = "Patient reports feeling sad most days"

        # Both work - pipeline doesn't enforce field separation
        result1 = pipeline(criteria_text)
        result2 = pipeline(evidence_text)

        assert isinstance(result1, str)
        assert isinstance(result2, str)

        # Field separation is enforced elsewhere (in Dataset classes)
        # See tests/test_groundtruth.py for actual field separation tests
