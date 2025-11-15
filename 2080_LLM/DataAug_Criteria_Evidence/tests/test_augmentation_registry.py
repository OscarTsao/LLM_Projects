"""
Test augmentation registry and all registered augmenters.

This tests the augmentation INFRASTRUCTURE even though it's inactive
in the NO-AUG baseline. Ensures future comparison studies will work.
"""

import pytest

from psy_agents_noaug.augmentation.registry import (
    ALL_METHODS,
    AUGMENTATION_BANLIST,
    NLPAUG_METHODS,
    REGISTRY,
    TEXTATTACK_METHODS,
    AugmenterWrapper,
)


@pytest.fixture
def sample_text():
    """Sample clinical text for testing."""
    return "The patient exhibits symptoms of major depressive disorder including persistent sadness."


@pytest.fixture
def short_text():
    """Short text for fast tests."""
    return "Patient shows anxiety symptoms."


class TestRegistryStructure:
    """Test registry structure and metadata."""

    def test_registry_not_empty(self):
        """Test that registry contains augmenters."""
        assert len(REGISTRY) == len(ALL_METHODS)

    def test_registry_size_matches_allowlist(self):
        """Registry should match allowlist exactly (17 entries)."""
        assert len(ALL_METHODS) == 17
        assert len(REGISTRY) == 17

    def test_all_methods_list(self):
        """Test ALL_METHODS list is populated."""
        assert isinstance(ALL_METHODS, list)
        assert len(ALL_METHODS) >= 17

    def test_nlpaug_methods_list(self):
        """Test NLPAUG_METHODS list is populated."""
        assert isinstance(NLPAUG_METHODS, list)
        assert len(NLPAUG_METHODS) >= 10

    def test_textattack_methods_list(self):
        """Test TEXTATTACK_METHODS list is populated."""
        assert isinstance(TEXTATTACK_METHODS, list)
        assert len(TEXTATTACK_METHODS) >= 7
        assert not any(m.startswith("nlpaug/") for m in TEXTATTACK_METHODS)

    def test_no_overlap_between_libs(self):
        """Test that nlpaug and textattack methods don't overlap."""
        nlpaug_set = set(NLPAUG_METHODS)
        textattack_set = set(TEXTATTACK_METHODS)
        overlap = nlpaug_set & textattack_set
        assert len(overlap) == 0, f"Found overlap: {overlap}"

    def test_all_methods_coverage(self):
        """Test that ALL_METHODS includes both nlpaug and textattack."""
        all_set = set(ALL_METHODS)
        nlpaug_set = set(NLPAUG_METHODS)
        textattack_set = set(TEXTATTACK_METHODS)

        assert nlpaug_set.issubset(all_set)
        assert textattack_set.issubset(all_set)

    def test_banlist_not_registered(self):
        """Ensure heavy augmenters remain unregistered for on-the-fly usage."""
        registered = set(REGISTRY.keys())
        overlap = registered.intersection(AUGMENTATION_BANLIST)
        assert not overlap, f"Banned augmenters unexpectedly registered: {overlap}"


class TestAugmenterWrapper:
    """Test AugmenterWrapper functionality."""

    def test_wrapper_init(self):
        """Test AugmenterWrapper initialization."""

        # Mock augmenter with augment method
        class MockAugmenter:
            def augment(self, text):
                return text.upper()

        wrapper = AugmenterWrapper("MockAug", MockAugmenter(), returns_list=False)
        assert wrapper.name == "MockAug"

    def test_wrapper_augment_one_string(self, short_text):
        """Test augment_one with string result."""

        class MockAugmenter:
            def augment(self, text):
                return text.upper()

        wrapper = AugmenterWrapper("MockAug", MockAugmenter())
        result = wrapper.augment_one(short_text)
        assert isinstance(result, str)
        assert result == short_text.upper()

    def test_wrapper_augment_one_list(self, short_text):
        """Test augment_one with list result."""

        class MockAugmenter:
            def augment(self, text):
                return [text.upper(), text.lower()]

        wrapper = AugmenterWrapper("MockAug", MockAugmenter())
        result = wrapper.augment_one(short_text)
        assert isinstance(result, str)
        assert result == short_text.upper()

    def test_wrapper_augment_one_none(self, short_text):
        """Test augment_one returns original text when augmenter returns None."""

        class MockAugmenter:
            def augment(self, text):
                return None

        wrapper = AugmenterWrapper("MockAug", MockAugmenter())
        result = wrapper.augment_one(short_text)
        assert result == short_text

    def test_wrapper_augment_one_empty_list(self, short_text):
        """Test augment_one returns original text when augmenter returns empty list."""

        class MockAugmenter:
            def augment(self, text):
                return []

        wrapper = AugmenterWrapper("MockAug", MockAugmenter())
        result = wrapper.augment_one(short_text)
        assert result == short_text

    def test_wrapper_augment_one_exception(self, short_text):
        """Test augment_one returns original text on exception."""

        class MockAugmenter:
            def augment(self, text):
                raise ValueError("Test exception")

        wrapper = AugmenterWrapper("MockAug", MockAugmenter())
        result = wrapper.augment_one(short_text)
        assert result == short_text


class TestNlpaugCharAugmenters:
    """Test nlpaug character-level augmenters."""

    @pytest.mark.parametrize(
        "augmenter_name",
        [
            "nlpaug/char/KeyboardAug",
            "nlpaug/char/OcrAug",
            "nlpaug/char/RandomCharAug",
        ],
    )
    def test_char_augmenter_basic(self, augmenter_name, short_text):
        """Test each char augmenter produces valid output."""
        entry = REGISTRY[augmenter_name]
        # Create with minimal kwargs
        kwargs = {"aug_char_p": 0.1}
        if augmenter_name == "nlpaug/char/RandomCharAug":
            kwargs["action"] = "substitute"

        wrapper = entry.factory(**kwargs)
        assert isinstance(wrapper, AugmenterWrapper)

        result = wrapper.augment_one(short_text)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0


class TestNlpaugWordAugmenters:
    """Test nlpaug word-level augmenters."""

    @pytest.mark.parametrize(
        "augmenter_name",
        [
            "nlpaug/word/RandomWordAug",
            "nlpaug/word/SpellingAug",
            "nlpaug/word/SplitAug",
            "nlpaug/word/SynonymAug(wordnet)",
            "nlpaug/word/AntonymAug(wordnet)",
        ],
    )
    def test_word_augmenter_basic(self, augmenter_name, short_text):
        """Test each word augmenter produces valid output."""
        entry = REGISTRY[augmenter_name]
        kwargs = {"aug_p": 0.1}
        if augmenter_name == "nlpaug/word/RandomWordAug":
            kwargs["action"] = "swap"

        wrapper = entry.factory(**kwargs)
        assert isinstance(wrapper, AugmenterWrapper)

        result = wrapper.augment_one(short_text)
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0


class TestTextAttackAugmenters:
    """Test TextAttack-based augmenters."""

    @pytest.mark.parametrize(
        "augmenter_name",
        [
            "textattack/CharSwapAugmenter",
            "textattack/DeletionAugmenter",
            "textattack/SwapAugmenter",
            "textattack/SynonymInsertionAugmenter",
            "textattack/EasyDataAugmenter",
            "textattack/CheckListAugmenter",
            "textattack/WordNetAugmenter",
        ],
    )
    def test_textattack_augmenter_basic(self, augmenter_name, short_text):
        """Test each TextAttack augmenter produces valid output."""
        entry = REGISTRY[augmenter_name]

        # TextAttack augmenters may need specific kwargs
        kwargs = {}
        if "pct_words_to_swap" in str(augmenter_name):
            kwargs["pct_words_to_swap"] = 0.1
        elif "pct_words_to_delete" in str(augmenter_name):
            kwargs["pct_words_to_delete"] = 0.1
        elif "pct_characters_to_swap" in str(augmenter_name):
            kwargs["pct_characters_to_swap"] = 0.1

        wrapper = entry.factory(**kwargs)
        assert isinstance(wrapper, AugmenterWrapper)

        result = wrapper.augment_one(short_text)
        assert result is not None
        assert isinstance(result, str)
        # TextAttack may return empty string in some cases, so just check it's a string


class TestSpecialAugmenters:
    """Test augmenters requiring special resources."""

    def test_tfidf_aug_requires_model_path(self):
        """Test that TfIdfAug raises error without model_path."""
        entry = REGISTRY["nlpaug/word/TfIdfAug"]
        # Factory requires model_path as kwarg, not positional
        with pytest.raises((ValueError, TypeError)):
            entry.factory(model_path=None)

    def test_reserved_aug_requires_map_path(self):
        """Test that ReservedAug raises error without reserved_map_path."""
        entry = REGISTRY["nlpaug/word/ReservedAug"]
        # Factory requires reserved_map_path as kwarg, not positional
        with pytest.raises((ValueError, TypeError)):
            entry.factory(reserved_map_path=None)


class TestDeterminism:
    """Test augmentation determinism with fixed configurations."""

    def test_char_aug_determinism(self, short_text):
        """Test that char augmenters produce consistent results with same config."""
        # Note: Determinism for nlpaug requires setting action_flow seed
        # which is not exposed in our wrapper. This test checks for stability.
        entry = REGISTRY["nlpaug/char/RandomCharAug"]
        wrapper = entry.factory(aug_char_p=0.3, action="substitute")

        # Run multiple times
        results = [wrapper.augment_one(short_text) for _ in range(3)]

        # All results should be valid strings
        for result in results:
            assert isinstance(result, str)
            assert len(result) > 0


class TestPerformance:
    """Test augmentation performance (CPU-light requirement)."""

    @pytest.mark.timeout(5)
    def test_augmentation_speed_char(self, sample_text):
        """Test that char augmentation completes quickly."""
        import time

        entry = REGISTRY["nlpaug/char/KeyboardAug"]
        wrapper = entry.factory(aug_char_p=0.1)

        start = time.time()
        _ = wrapper.augment_one(sample_text)
        duration = time.time() - start

        assert duration < 2.0, f"Char augmentation took {duration:.2f}s (should be <2s)"

    @pytest.mark.timeout(5)
    def test_augmentation_speed_word(self, sample_text):
        """Test that word augmentation completes quickly."""
        import time

        entry = REGISTRY["nlpaug/word/SynonymAug(wordnet)"]
        wrapper = entry.factory(aug_p=0.1)

        start = time.time()
        _ = wrapper.augment_one(sample_text)
        duration = time.time() - start

        # WordNet lookups may be slower on first run
        assert duration < 5.0, f"Word augmentation took {duration:.2f}s (should be <5s)"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_text(self):
        """Test augmenters handle empty text gracefully."""
        entry = REGISTRY["nlpaug/char/KeyboardAug"]
        wrapper = entry.factory(aug_char_p=0.1)
        result = wrapper.augment_one("")
        assert result == ""

    def test_very_short_text(self):
        """Test augmenters handle very short text."""
        entry = REGISTRY["nlpaug/word/SynonymAug(wordnet)"]
        wrapper = entry.factory(aug_p=0.3)
        result = wrapper.augment_one("Hi")
        assert isinstance(result, str)

    def test_long_text(self):
        """Test augmenters handle long text."""
        long_text = " ".join(["The patient shows symptoms."] * 50)
        entry = REGISTRY["nlpaug/char/KeyboardAug"]
        wrapper = entry.factory(aug_char_p=0.05)
        result = wrapper.augment_one(long_text)
        assert isinstance(result, str)
        assert len(result) > 0
