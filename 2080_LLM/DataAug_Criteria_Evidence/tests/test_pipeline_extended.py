"""Extended tests for pipeline.py - focusing on uncovered functionality."""

import random

import numpy as np
import pytest

from psy_agents_noaug.augmentation.pipeline import (
    AugConfig,
    AugmenterPipeline,
    _clamp,
    _merge_kwargs,
    _ratio_kwargs,
    is_enabled,
    worker_init,
)
from psy_agents_noaug.data.augmentation_utils import resolve_methods


class TestHelperFunctions:
    """Test utility functions."""

    def test_clamp(self):
        assert _clamp(0.5, 0, 1) == 0.5
        assert _clamp(-0.5, 0, 1) == 0
        assert _clamp(1.5, 0, 1) == 1

    def test_merge_kwargs_none(self):
        assert _merge_kwargs({"a": 1}, None) == {"a": 1}

    def test_merge_kwargs_override(self):
        result = _merge_kwargs({"a": 1}, {"a": 2, "b": 3})
        assert result == {"a": 2, "b": 3}

    def test_ratio_kwargs_char(self):
        assert _ratio_kwargs("nlpaug/char/KeyboardAug", 0.3) == {"aug_char_p": 0.3}

    def test_ratio_kwargs_word(self):
        assert _ratio_kwargs("nlpaug/word/SynonymAug(wordnet)", 0.3) == {"aug_p": 0.3}

    def test_ratio_kwargs_textattack(self):
        result = _ratio_kwargs("textattack/CharSwapAugmenter", 0.3)
        assert result == {"pct_characters_to_swap": 0.3}


class TestResolveMethodsInternal:
    """Test internal _resolve_methods function."""

    def test_resolve_none(self):
        assert resolve_methods("none", ["all"]) == []

    def test_resolve_nlpaug_all(self):
        result = resolve_methods("nlpaug", ["all"])
        assert len(result) >= 10

    def test_resolve_unknown_raises(self):
        with pytest.raises(KeyError):
            resolve_methods("nlpaug", ["unknown"])


class TestIsEnabled:
    """Test is_enabled function."""

    def test_is_enabled_none(self):
        assert is_enabled(AugConfig(lib="none")) is False

    def test_is_enabled_nlpaug(self):
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"])
        assert is_enabled(cfg)

    def test_is_enabled_empty_methods(self):
        assert is_enabled(AugConfig(lib="nlpaug", methods=[]))


class TestWorkerInit:
    """Test worker_init function."""

    def test_worker_init_returns_seed(self):
        seed = worker_init(0, 42)
        assert seed == 43

    def test_worker_init_different_workers(self):
        s1 = worker_init(0, 42)
        s2 = worker_init(1, 42)
        assert s1 != s2

    def test_worker_init_sets_random(self):
        worker_init(0, 42)
        v1 = random.random()
        worker_init(0, 42)
        v2 = random.random()
        assert v1 == v2

    def test_worker_init_sets_numpy(self):
        worker_init(0, 42)
        v1 = np.random.random()
        worker_init(0, 42)
        v2 = np.random.random()
        assert v1 == v2


class TestAugmenterPipelineBasics:
    """Test AugmenterPipeline class."""

    def test_init_none_raises(self):
        with pytest.raises(ValueError):
            AugmenterPipeline(AugConfig(lib="none"))

    def test_init_basic(self):
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"])
        pipeline = AugmenterPipeline(cfg)
        assert pipeline.p_apply >= 0
        assert pipeline.ops_per_sample >= 1

    def test_init_clamps_p_apply(self):
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=1.5)
        pipeline = AugmenterPipeline(cfg)
        assert pipeline.p_apply == 1.0

    def test_init_clamps_ops_per_sample(self):
        cfg = AugConfig(
            lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], ops_per_sample=10
        )
        pipeline = AugmenterPipeline(cfg)
        assert pipeline.ops_per_sample == 2

    def test_set_seed(self):
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"])
        pipeline = AugmenterPipeline(cfg)
        pipeline.set_seed(123)
        v1 = pipeline._rng.random()
        pipeline.set_seed(123)
        v2 = pipeline._rng.random()
        assert v1 == v2


class TestAugmenterPipelineCall:
    """Test pipeline augmentation calls."""

    def test_call_p_apply_zero(self):
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=0.0)
        pipeline = AugmenterPipeline(cfg)
        text = "test"
        assert pipeline(text) == text
        assert pipeline.skipped_count == 1

    def test_call_increments_total(self):
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=1.0)
        pipeline = AugmenterPipeline(cfg)
        pipeline("test1")
        pipeline("test2")
        assert pipeline.total_count == 2

    def test_call_empty_text(self):
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=1.0)
        pipeline = AugmenterPipeline(cfg)
        assert pipeline("") == ""

    def test_call_with_ops_per_sample_two(self):
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=1.0,
            ops_per_sample=2,
        )
        pipeline = AugmenterPipeline(cfg)
        result = pipeline("test text")
        assert isinstance(result, str)


class TestAugmenterPipelineStats:
    """Test statistics collection."""

    def test_stats_initial(self):
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"])
        pipeline = AugmenterPipeline(cfg)
        stats = pipeline.stats()
        assert stats["total"] == 0
        assert stats["applied"] == 0
        assert stats["skipped"] == 0

    def test_stats_after_calls(self):
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=0.0)
        pipeline = AugmenterPipeline(cfg)
        for _ in range(5):
            pipeline("test")
        stats = pipeline.stats()
        assert stats["total"] == 5
        assert stats["skipped"] == 5


class TestAugmenterPipelineExamples:
    """Test example collection."""

    def test_drain_examples_empty(self):
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"])
        pipeline = AugmenterPipeline(cfg)
        assert pipeline.drain_examples() == []

    def test_drain_examples_clears(self):
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=1.0)
        pipeline = AugmenterPipeline(cfg)
        pipeline("test")
        ex1 = pipeline.drain_examples()
        ex2 = pipeline.drain_examples()
        assert ex2 == []

    def test_examples_limited(self):
        cfg = AugConfig(
            lib="nlpaug",
            methods=["nlpaug/char/KeyboardAug"],
            p_apply=1.0,
            example_limit=2,
        )
        pipeline = AugmenterPipeline(cfg)
        for _ in range(10):
            pipeline("test")
        assert len(pipeline.examples) <= 2


class TestAugmenterPipelineEdgeCases:
    """Test edge cases."""

    def test_long_text(self):
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=1.0)
        pipeline = AugmenterPipeline(cfg)
        long = " ".join(["text"] * 100)
        result = pipeline(long)
        assert isinstance(result, str)

    def test_special_chars(self):
        cfg = AugConfig(lib="nlpaug", methods=["nlpaug/char/KeyboardAug"], p_apply=1.0)
        pipeline = AugmenterPipeline(cfg)
        result = pipeline("@#$%^&*()")
        assert isinstance(result, str)
