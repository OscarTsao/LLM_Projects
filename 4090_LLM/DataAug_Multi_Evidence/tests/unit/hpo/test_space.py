from __future__ import annotations

from dataaug_multi_both.hpo.space import (
    decode_mask,
    encode_mask,
    SIMPLE_AUG_METHODS,
    STRUCTURAL_KEYS,
    TEXTATTACK_METHODS,
    narrow_stage_b_space,
    stage_a_search_space,
)


def test_stage_a_space_contains_required_keys() -> None:
    space = stage_a_search_space()
    assert "encoder_model_name" in space
    assert "opt_lr_enc" in space
    assert "aug_simple_mask" in space
    assert "eval_batch_size" in space
    assert "resources_num_workers" in space


def test_structural_keys_subset_of_stage_a() -> None:
    space = stage_a_search_space()
    missing = [key for key in STRUCTURAL_KEYS if key not in space]
    assert not missing


def test_stage_b_narrowing_freezes_structural_keys() -> None:
    space = stage_a_search_space()
    best_params = {key: "dummy" for key in STRUCTURAL_KEYS}
    result = narrow_stage_b_space(best_params, space)
    assert set(result.frozen) == set(STRUCTURAL_KEYS)
    assert all(key not in result.search for key in STRUCTURAL_KEYS)


def test_aug_mask_choices_respect_budget() -> None:
    space = stage_a_search_space()
    simple_choices = space["aug_simple_mask"]["choices"]
    assert "none" in simple_choices
    assert all(len(decode_mask(choice)) <= 2 for choice in simple_choices)

    ta_choices = space["aug_ta_mask"]["choices"]
    assert "none" in ta_choices
    assert all(len(decode_mask(choice)) <= 1 for choice in ta_choices)


def test_constants_have_expected_lengths() -> None:
    assert len(SIMPLE_AUG_METHODS) == 5
    assert len(TEXTATTACK_METHODS) == 9
