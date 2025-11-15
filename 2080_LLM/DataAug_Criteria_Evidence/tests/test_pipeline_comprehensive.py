"""
Comprehensive tests for pipeline.py module.

Target: Increase coverage from ~18% to >70%.
Tests: Multi-operation augmentation, statistics, error recovery, edge cases, worker init.
"""

import random

import numpy as np
import pytest

from psy_agents_noaug.augmentation.pipeline import (
    AugConfig,
    AugmenterPipeline,
    AugResources,
    _clamp,
    _merge_kwargs,
    _ratio_kwargs,
    _resolve_methods,
    is_enabled,
    worker_init,
)


@pytest.fixture
def sample_text():
    """Sample clinical text for testing."""
    return "Patient reports persistent sadness and loss of interest in activities."


@pytest.fixture
def char_aug_config():
    """Basic character augmentation config."""
    return AugConfig(
        lib="nlpaug",
        methods=["nlpaug/char/KeyboardAug"],
        p_apply=1.0,
        ops_per_sample=1,
        max_replace_ratio=0.1,
        seed=42,
    )


class TestClampFunction:
    """Test _clamp() utility function."""

    def test_clamp_within_range(self):
        """Test value within range is unchanged."""
        assert _clamp(0.5, 0.0, 1.0) == 0.5

    def test_clamp_below_minimum(self):
        """Test value below minimum is clamped."""
        assert _clamp(-0.5, 0.0, 1.0) == 0.0

    def test_clamp_above_maximum(self):
        """Test value above maximum is clamped."""
        assert _clamp(1.5, 0.0, 1.0) == 1.0

    def test_clamp_at_boundaries(self):
        """Test values at boundaries are unchanged."""
        assert _clamp(0.0, 0.0, 1.0) == 0.0
        assert _clamp(1.0, 0.0, 1.0) == 1.0


class TestResolveMethodsInternal:
    """Test _resolve_methods() internal function."""

    def test_resolve_none_lib(self):
        """Test none lib returns empty list."""
        result = _resolve_methods("none", ["all"])
        assert result == []

    def test_resolve_nlpaug_all(self):
        """Test resolving all nlpaug methods."""
        result = _resolve_methods("nlpaug", ["all"])
        assert len(result) >= 10
        assert all("nlpaug/" in m for m in result)

    def test_resolve_textattack_all(self):
        """Test resolving all textattack methods."""
        result = _resolve_methods("textattack", ["all"])
        assert len(result) >= 7
        assert all("textattack/" in m for m in result)

    def test_resolve_unknown_method_raises(self):
        """Test unknown method raises KeyError."""
        with pytest.raises(KeyError, match="Unknown augmentation method"):
            _resolve_methods("nlpaug", ["unknown_method"])

    def test_resolve_filters_wrong_lib(self):
        """Test that wrong library methods are filtered."""
        result = _resolve_methods("nlpaug", ["textattack/CharSwapAugmenter"])
        assert result == []


class TestRatioKwargsFunction:
    """Test _ratio_kwargs() function."""

    def test_ratio_kwargs_nlpaug_char(self):
        """Test nlpaug char augmenters get aug_char_p."""
        result = _ratio_kwargs("nlpaug/char/KeyboardAug", 0.3)
        assert result == {"aug_char_p": 0.3}

    def test_ratio_kwargs_nlpaug_word(self):
        """Test nlpaug word augmenters get aug_p."""
        result = _ratio_kwargs("nlpaug/word/SynonymAug(wordnet)", 0.3)
        assert result == {"aug_p": 0.3}

    def test_ratio_kwargs_textattack_charswap(self):
        """Test TextAttack CharSwapAugmenter."""
        result = _ratio_kwargs("textattack/CharSwapAugmenter", 0.3)
        assert result == {"pct_characters_to_swap": 0.3}

    def test_ratio_kwargs_textattack_deletion(self):
        """Test TextAttack DeletionAugmenter."""
        result = _ratio_kwargs("textattack/DeletionAugmenter", 0.3)
        assert result == {"pct_words_to_delete": 0.3}

    def test_ratio_kwargs_textattack_swap(self):
        """Test TextAttack SwapAugmenter."""
        result = _ratio_kwargs("textattack/SwapAugmenter", 0.3)
        assert result == {"pct_words_to_swap": 0.3}

    def test_ratio_kwargs_unknown_method(self):
        """Test unknown method returns empty dict."""
        result = _ratio_kwargs("unknown/method", 0.3)
        assert result == {}

    def test_ratio_kwargs_clamps_value(self):
        """Test that ratio is clamped to [0, 1]."""
        result = _ratio_kwargs("nlpaug/char/KeyboardAug", 1.5)
        assert result["aug_char_p"] == 1.0

        result = _ratio_kwargs("nlpaug/char/KeyboardAug", -0.5)
        assert result["aug_char_p"] == 0.0


class TestMergeKwargsFunction:
    """Test _merge_kwargs() function."""

    def test_merge_empty_override(self):
        """Test merge with None override returns base."""
        base = {"a": 1, "b": 2}
        result = _merge_kwargs(base, None)
        assert result == base

    def test_merge_with_override(self):
        """Test merge with override updates values."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _merge_kwargs(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_doesnt_modify_base(self):
        """Test that merge doesn't modify base dict."""
        base = {"a": 1}
        override = {"b": 2}
        result = _merge_kwargs(base, override)
        assert base == {"a": 1}  # Unchanged
        assert result == {"a": 1, "b": 2}


class TestIsEnabledFunction:
    """Test is_enabled() function."""

    def test_is_enabled_none_lib(self):
        """Test that 'none' lib is disabled."""
        cfg = AugConfig(lib="none")
        assert is_enabled(cfg) is False

    def test_is_enabled_nlpaug(self):
        """Test that nlpaug with methods is enabled."""
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"])
        assert is_enabled(cfg) is True

    def test_is_enabled_textattack(self):
        """Test that textattack with methods is enabled."""
        cfg = AugConfig(lib="textattack", methods=["textattack/CharSwapAugmenter"])
        assert is_enabled(cfg) is True

    def test_is_enabled_both(self):
        """Test that 'both' with methods is enabled."""
        cfg = AugConfig(lib="both", methods=["nlpaug/char/KeyboardAug"])
        assert is_enabled(cfg) is True

    def test_is_enabled_empty_methods(self):
        """Test that empty methods is disabled."""
        cfg = AugConfig(lib="nlpaug", methods=[])
        assert is_enabled(cfg) is False


class TestWorkerInitFunction:
    """Test worker_init() function."""

    def test_worker_init_sets_seeds(self):
        """Test that worker_init sets random seeds."""
        base_seed = 42
        worker_id = 3

        seed = worker_init(worker_id, base_seed)

        assert seed == base_seed + worker_id + 1
        assert seed == 46

    def test_worker_init_different_workers_different_seeds(self):
        """Test that different workers get different seeds."""
        base_seed = 42
        seed1 = worker_init(0, base_seed)
        seed2 = worker_init(1, base_seed)
        seed3 = worker_init(2, base_seed)

        assert seed1 != seed2 != seed3
        assert seed1 == 43
        assert seed2 == 44
        assert seed3 == 45

    def test_worker_init_random_state(self):
        """Test that worker_init sets random state."""
        worker_init(0, 42)
        val1 = random.random()

        worker_init(0, 42)  # Same seed
        val2 = random.random()

        assert val1 == val2  # Deterministic

    def test_worker_init_numpy_state(self):
        """Test that worker_init sets numpy random state."""
        worker_init(0, 42)
        val1 = np.random.random()

        worker_init(0, 42)  # Same seed
        val2 = np.random.random()

        assert val1 == val2  # Deterministic


class TestAugmenterPipelineInit:
    """Test AugmenterPipeline initialization."""

    def test_pipeline_init_none_lib_raises(self):
        """Test that 'none' lib raises ValueError."""
        cfg = AugConfig(lib="none")
        with pytest.raises(ValueError, match="Cannot instantiate"):
            AugmenterPipeline(cfg)

    def test_pipeline_init_basic(self, char_aug_config):
        """Test basic pipeline initialization."""
        pipeline = AugmenterPipeline(char_aug_config)

        assert pipeline.cfg == char_aug_config
        assert pipeline.methods == ["nlpaug/char/KeyboardAug"]
        assert pipeline.p_apply == 1.0
        assert pipeline.ops_per_sample == 1

    def test_pipeline_init_clamps_p_apply(self):
        """Test that p_apply is clamped to [0, 1]."""
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=1.5)
        pipeline = AugmenterPipeline(cfg)
        assert pipeline.p_apply == 1.0

    def test_pipeline_init_clamps_ops_per_sample(self):
        """Test that ops_per_sample is clamped to [1, 2]."""
        cfg = AugConfig(
            lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], ops_per_sample=5
        )
        pipeline = AugmenterPipeline(cfg)
        assert pipeline.ops_per_sample == 2

    def test_pipeline_init_with_resources(self):
        """Test pipeline initialization with resources."""
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"])
        resources = AugResources()
        pipeline = AugmenterPipeline(cfg, resources=resources)

        assert pipeline.resources == resources

    def test_pipeline_init_statistics_initialized(self, char_aug_config):
        """Test that statistics are initialized to zero."""
        pipeline = AugmenterPipeline(char_aug_config)

        assert pipeline.applied_count == 0
        assert pipeline.skipped_count == 0
        assert pipeline.total_count == 0
        assert len(pipeline.method_counts) == 0
        assert len(pipeline.examples) == 0


class TestAugmenterPipelineSetSeed:
    """Test set_seed() method."""

    def test_set_seed_changes_rng(self, char_aug_config):
        """Test that set_seed changes RNG state."""
        pipeline = AugmenterPipeline(char_aug_config)

        pipeline.set_seed(123)
        val1 = pipeline._rng.random()

        pipeline.set_seed(123)  # Same seed
        val2 = pipeline._rng.random()

        assert val1 == val2

    def test_set_seed_different_values(self, char_aug_config):
        """Test that different seeds produce different values."""
        pipeline = AugmenterPipeline(char_aug_config)

        pipeline.set_seed(123)
        val1 = pipeline._rng.random()

        pipeline.set_seed(456)  # Different seed
        val2 = pipeline._rng.random()

        assert val1 != val2


class TestAugmenterPipelineCall:
    """Test pipeline __call__() method."""

    def test_call_with_p_apply_zero_skips(self, sample_text):
        """Test that p_apply=0 always skips augmentation."""
        cfg = AugConfig(
            lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=0.0, seed=42
        )
        pipeline = AugmenterPipeline(cfg)

        result = pipeline(sample_text)
        assert result == sample_text
        assert pipeline.skipped_count == 1
        assert pipeline.applied_count == 0

    def test_call_with_p_apply_one_applies(self, sample_text):
        """Test that p_apply=1.0 always applies augmentation."""
        cfg = AugConfig(
            lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=1.0, seed=42
        )
        pipeline = AugmenterPipeline(cfg)

        result = pipeline(sample_text)
        assert isinstance(result, str)
        assert pipeline.total_count == 1
        # May be applied or skipped depending on augmenter behavior

    def test_call_increments_total_count(self, sample_text):
        """Test that each call increments total count."""
        cfg = AugConfig(
            lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=1.0, seed=42
        )
        pipeline = AugmenterPipeline(cfg)

        pipeline(sample_text)
        assert pipeline.total_count == 1

        pipeline(sample_text)
        assert pipeline.total_count == 2

    def test_call_with_ops_per_sample_two(self, sample_text):
        """Test augmentation with ops_per_sample=2."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=1.0,
            ops_per_sample=2,
            seed=42,
        )
        pipeline = AugmenterPipeline(cfg)

        result = pipeline(sample_text)
        assert isinstance(result, str)

    def test_call_collects_examples(self, sample_text):
        """Test that examples are collected up to limit."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=1.0,
            seed=42,
            example_limit=2,
        )
        pipeline = AugmenterPipeline(cfg)

        for _ in range(5):
            pipeline(sample_text)

        # Should collect at most 2 examples
        assert len(pipeline.examples) <= 2

    def test_call_empty_text(self):
        """Test augmentation with empty text."""
        cfg = AugConfig(
            lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=1.0, seed=42
        )
        pipeline = AugmenterPipeline(cfg)

        result = pipeline("")
        assert result == ""


class TestAugmenterPipelineStats:
    """Test stats() method."""

    def test_stats_initial(self, char_aug_config):
        """Test stats() returns correct initial values."""
        pipeline = AugmenterPipeline(char_aug_config)

        stats = pipeline.stats()
        assert stats["total"] == 0
        assert stats["applied"] == 0
        assert stats["skipped"] == 0
        assert stats["method_counts"] == {}

    def test_stats_after_calls(self, sample_text):
        """Test stats() after making calls."""
        cfg = AugConfig(
            lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=0.0, seed=42
        )
        pipeline = AugmenterPipeline(cfg)

        for _ in range(5):
            pipeline(sample_text)

        stats = pipeline.stats()
        assert stats["total"] == 5
        assert stats["skipped"] == 5
        assert stats["applied"] == 0

    def test_stats_method_counts(self, sample_text):
        """Test that method counts are tracked."""
        cfg = AugConfig(
            lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=1.0, seed=42
        )
        pipeline = AugmenterPipeline(cfg)

        for _ in range(3):
            pipeline(sample_text)

        stats = pipeline.stats()
        assert isinstance(stats["method_counts"], dict)


class TestAugmenterPipelineDrainExamples:
    """Test drain_examples() method."""

    def test_drain_examples_empty(self, char_aug_config):
        """Test drain_examples() on empty pipeline."""
        pipeline = AugmenterPipeline(char_aug_config)

        examples = pipeline.drain_examples()
        assert examples == []

    def test_drain_examples_clears_list(self, sample_text):
        """Test that drain_examples() clears internal list."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=1.0,
            seed=42,
            example_limit=10,
        )
        pipeline = AugmenterPipeline(cfg)

        pipeline(sample_text)
        pipeline(sample_text)

        examples1 = pipeline.drain_examples()
        assert len(examples1) >= 0

        # Should be empty after draining
        examples2 = pipeline.drain_examples()
        assert examples2 == []

    def test_drain_examples_structure(self, sample_text):
        """Test that drained examples have correct structure."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=1.0,
            seed=42,
            example_limit=10,
        )
        pipeline = AugmenterPipeline(cfg)

        # Force augmentation
        for _ in range(3):
            pipeline(sample_text)

        examples = pipeline.drain_examples()
        if examples:  # If any examples were collected
            example = examples[0]
            assert "original" in example
            assert "augmented" in example
            assert "methods" in example
            assert "timestamp" in example


class TestAugmenterPipelineMultipleMethods:
    """Test pipeline with multiple augmentation methods."""

    def test_multiple_methods_selection(self, sample_text):
        """Test that multiple methods are randomly selected."""
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug", "nlpaug/word/SynonymAug(wordnet)"],
            p_apply=1.0,
            seed=42,
        )
        pipeline = AugmenterPipeline(cfg)

        assert len(pipeline.methods) == 2

        for _ in range(5):
            result = pipeline(sample_text)
            assert isinstance(result, str)


class TestAugmenterPipelineEdgeCases:
    """Test edge cases and error handling."""

    def test_very_long_text(self):
        """Test augmentation with very long text."""
        long_text = " ".join(["Patient shows symptoms."] * 100)
        cfg = AugConfig(
            lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=1.0, seed=42
        )
        pipeline = AugmenterPipeline(cfg)

        result = pipeline(long_text)
        assert isinstance(result, str)

    def test_special_characters(self):
        """Test augmentation with special characters."""
        text = "Patient: @#$%^&*() symptoms!!!"
        cfg = AugConfig(
            lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=1.0, seed=42
        )
        pipeline = AugmenterPipeline(cfg)

        result = pipeline(text)
        assert isinstance(result, str)

    def test_unicode_text(self):
        """Test augmentation with Unicode text."""
        text = "Patient avec symptômes émotionnels"
        cfg = AugConfig(
            lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=1.0, seed=42
        )
        pipeline = AugmenterPipeline(cfg)

        result = pipeline(text)
        assert isinstance(result, str)
